#pragma once

// [v100-opt] Тайминг операторов CUDA через события. Только для диагностики.
//
// Зачем. ncu на живом сервере дедлочит, и даже с фильтром по имени ядра даёт
// 0.15 т/с на префилле. Свой инструмент снимает инвентарь шага decode прямо на
// рабочем контексте (100k/240k) за один обычный запрос.
//
// Как. События пишутся в поток, ничего не сериализуя; результаты читаются один раз
// в конце графа (cudaDeviceSynchronize). Времена ядер не искажаются, плата —
// 2 cudaEventRecord на узел. Число узлов в графе порядка полутора тысяч, поэтому
// слоты событий берутся монотонным счётчиком (O(1)), а не поиском свободного.
//
// ВАЖНО: состояние живёт в optiming.cu, а не в static-переменной inline-функции
// здесь. При компиляции CUDA static внутри inline в заголовке НЕ даёт общей копии
// между translation unit, и области, объявленные не в том TU, в отчёт не попадали
// (проверено: таймер вокруг конверсии K/V и вокруг ядра mmvq молча исчезал).
//
// Включение:
//   GGML_OP_TIMING=1     — по именам узлов (подробно)
//   GGML_OP_TIMING_OP=1  — по именам операторов (компактно)
//   GGML_OP_TIMING_EVERY=N — печатать раз в N вызовов графа (по умолчанию 32)
//   GGML_OP_TIMING_ONCE=1 — напечатать один раз и больше не печатать
//
// Требуется GGML_CUDA_DISABLE_GRAPHS=1. Всё выключено по умолчанию через getenv,
// поэтому задеплоенная библиотека ведёт себя как обычно.

#include "common.cuh"

#include <chrono>
#include <cstdio>
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
    bool enabled      = false;
    bool by_node      = false;
    bool once         = false;
    bool reported     = false;
    int  report_every = 32;
    int  graphs       = 0;

    std::vector<ggml_cuda_op_timing_slot> slots;   // переиспользуются после flush
    std::vector<int>                       active;  // стек вложенности
    int                                   next_slot = 0;
    std::chrono::steady_clock::time_point  t_last = std::chrono::steady_clock::now();

    // имя -> (сумма мс, число вызовов)
    std::map<std::string, std::pair<double, long long>> acc;
};

// Единственный экземпляр состояния (определён в optiming.cu).
ggml_cuda_op_timing_state & ggml_cuda_op_timing_acc();

// Забрать времена и напечатать отчёт по дельте (сбрасывает накопитель).
void ggml_cuda_op_timing_report();

static inline bool ggml_cuda_op_timing_enabled() {
    return ggml_cuda_op_timing_acc().enabled;
}

// Имя региона: имя узла графа (подробно) или имя оператора (компактно).
static inline bool ggml_cuda_op_timing_by_node() {
    return ggml_cuda_op_timing_acc().by_node;
}

static inline void ggml_cuda_op_timing_begin(const char * name, cudaStream_t stream) {
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_acc();
    if (!st.enabled) {
        return;
    }
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
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_acc();
    if (!st.enabled || st.active.empty()) {
        return;
    }
    const int idx = st.active.back();
    st.active.pop_back();
    CUDA_CHECK(cudaEventRecord(st.slots[idx].ev_end, stream));
    st.slots[idx].done = true;
}

// Вызывать один раз в конце выполнения графа.
static inline void ggml_cuda_op_timing_flush() {
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_acc();
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
    ggml_cuda_op_timing_report();
}
