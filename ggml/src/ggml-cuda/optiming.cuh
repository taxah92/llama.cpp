#pragma once

// [v100-opt] Тайминг операторов CUDA через события. Только для диагностики.
//
// Зачем. ncu на живом сервере дедлочит (перехват каждого запуска + VMM-пул), и
// даже с фильтром по имени ядра прогон занимает минуты на запуск. Свой
// инструмент даёт инвентарь шага decode прямо на рабочем контексте (100k/240k)
// за один обычный запрос.
//
// Как. События пишутся в поток, ничего не синхронизируя на каждом узле;
// результаты читаются один раз в конце графа (cudaDeviceSynchronize). Времена
// ядер при этом не искажаются, плата — CPU-накладной на 2 cudaEventRecord на
// узел (~1 мкс каждый), что видно по сумме против времени шага.
//
// Включение:
//   GGML_OP_TIMING=1     — с именем узла (подробно)
//   GGML_OP_TIMING_OP=1  — только имя оператора (компактно)
//   GGML_OP_TIMING_EVERY=N — печатать раз в N вызовов графа (по умолчанию 32)
//   GGML_OP_TIMING_ONCE=1 — напечатать один раз в конце и не печатать дальше
//
// Требуется GGML_CUDA_DISABLE_GRAPHS=1: при CUDA-графах узлы не являются
// отдельными запусками, и события вокруг них расставить нельзя.
//
// Ограничение: области не должны вкладываться друг в друга ОДНОИМ именем
// (каждый слот событий уникален, но слот переиспользуется только после flush).

#include "common.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct ggml_cuda_op_timing_slot {
    cudaEvent_t ev_beg = nullptr;
    cudaEvent_t ev_end = nullptr;
    const char * name  = nullptr;
    bool         done   = false;
};

struct ggml_cuda_op_timing_state {
    bool enabled     = false;
    bool by_node     = false;
    bool once        = false;
    int  report_every = 32;
    int  graphs      = 0;
    bool reported    = false;

    std::vector<ggml_cuda_op_timing_slot> slots;   // переиспользуются после flush
    std::vector<int>                       active;  // стек вложенности
    int                                   next_slot = 0;
    std::chrono::steady_clock::time_point  t_last = std::chrono::steady_clock::now();
    std::map<std::string, std::pair<double, long long>> acc; // имя -> (мс, число вызовов)
};

static inline ggml_cuda_op_timing_state & ggml_cuda_op_timing_state_get() {
    static ggml_cuda_op_timing_state st;
    static bool init = false;
    if (!init) {
        init = true;
        const char * e_node = getenv("GGML_OP_TIMING");
        const char * e_op   = getenv("GGML_OP_TIMING_OP");
        st.enabled  = e_node != nullptr || e_op != nullptr;
        st.by_node  = e_node != nullptr;
        const char * once = getenv("GGML_OP_TIMING_ONCE");
        st.once = once != nullptr && atoi(once) == 1;
        const char * every = getenv("GGML_OP_TIMING_EVERY");
        if (every) st.report_every = atoi(every);
    }
    return st;
}

static inline bool ggml_cuda_op_timing_enabled() {
    return ggml_cuda_op_timing_state_get().enabled;
}

// Имя региона: имя узла графа (подробно) или имя оператора (компактно).
static inline bool ggml_cuda_op_timing_by_node() {
    return ggml_cuda_op_timing_state_get().by_node;
}

static inline void ggml_cuda_op_timing_begin(const char * name, cudaStream_t stream) {
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_state_get();
    if (!st.enabled) {
        return;
    }
    // Слоты освобождаются все сразу при flush, поэтому внутри графа достаточно
    // монотонного счётчика — O(1) вместо поиска свободного (важно: вызовов
    // порядка полутора тысяч за граф, поиск давал бы квадратичную работу на CPU).
    const int idx = st.next_slot++;
    if (idx >= (int) st.slots.size()) {
        ggml_cuda_op_timing_slot s;
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ev_beg, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ev_end, cudaEventDefault));
        st.slots.push_back(s);
    }
    st.slots[idx].name = name;
    st.slots[idx].done = false;
    CUDA_CHECK(cudaEventRecord(st.slots[idx].ev_beg, stream));
    st.active.push_back(idx);
}

static inline void ggml_cuda_op_timing_end(cudaStream_t stream) {
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_state_get();
    if (!st.enabled || st.active.empty()) {
        return;
    }
    const int idx = st.active.back();
    st.active.pop_back();
    CUDA_CHECK(cudaEventRecord(st.slots[idx].ev_end, stream));
    st.slots[idx].done = true;
}

// Вызывать один раз в конце выполнения графа: синхронизирует и забирает времена.
static inline void ggml_cuda_op_timing_flush() {
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_state_get();
    if (!st.enabled) {
        return;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t i = 0; i < st.slots.size(); ++i) {
        if (st.slots[i].done && st.slots[i].name) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, st.slots[i].ev_beg, st.slots[i].ev_end) == cudaSuccess) {
                auto & a = st.acc[st.slots[i].name];
                a.first  += ms;
                a.second += 1;
            }
        }
    }
    st.active.clear();
    st.next_slot = 0;
    st.graphs += 1;

    if (st.once && st.reported) {
        return;
    }
    if (st.graphs % (st.report_every > 0 ? st.report_every : 32) != 0) {
        return;
    }
    st.reported = true;

    double total = 0.0;
    for (const auto & kv : st.acc) {
        total += kv.second.first;
    }
    const double graphs = st.graphs;

    // host-время окна: показывает, ограничен ли шаг GPU или хостом (launch-оверхед,
    // аллокации, копирование KV, планирование графа). Без этого нельзя понять, какой
    // потолок у GPU-оптимизаций. Сумма GPU может превышать HOST: потоки идут параллельно.
    const auto t_now = std::chrono::steady_clock::now();
    const double host_ms = std::chrono::duration<double, std::milli>(t_now - st.t_last).count();
    st.t_last = t_now;

    fprintf(stderr, "\n[v100-opt] тайминг операторов: %d графов, GPU = %.3f мс/граф, HOST = %.3f мс/граф\n",
            st.graphs, total/graphs, host_ms/graphs);
    fprintf(stderr, "%-46s %8s %10s %10s %7s\n", "оператор", "н/граф", "мкс/вызов", "мс/граф", "%");
    for (const auto & kv : st.acc) {
        const double ms_graph = kv.second.first / graphs;
        const double per_call = kv.second.first / (double) kv.second.second;
        fprintf(stderr, "%-46s %8.1f %10.1f %10.4f %6.2f%%\n",
                kv.first.c_str(), kv.second.second / graphs, per_call, ms_graph,
                total > 0.0 ? 100.0 * ms_graph / (total/graphs) : 0.0);
    }
    fprintf(stderr, "v100-opt: всего строк: %zu\n", st.acc.size());

    // Дельта-семантика: следующий отчёт покрывает только графы после этого, иначе
    // окно префилла (сотни графов) смешается с окном декода.
    st.acc.clear();
    st.graphs = 0;
    fprintf(stderr, "\n");
}
