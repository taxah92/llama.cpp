// [v100-opt] Единственный экземпляр состояния таймера операторов и печать отчёта.
// См. optiming.cuh — там же объяснение, почему состояние обязано жить здесь, а не в
// static-переменной inline-функции заголовка (при компиляции CUDA такой static не
// общий между translation unit).

#include "optiming.cuh"

#include <cstdlib>

ggml_cuda_op_timing_state & ggml_cuda_op_timing_acc() {
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
        if (every) {
            st.report_every = atoi(every);
        }
    }
    return st;
}

void ggml_cuda_op_timing_report() {
    ggml_cuda_op_timing_state & st = ggml_cuda_op_timing_acc();

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
    // аллокации, копирование KV, планирование графа). Сумма GPU может превышать
    // HOST, если часть операторов идёт на параллельных потоках.
    const auto t_now = std::chrono::steady_clock::now();
    const double host_ms = std::chrono::duration<double, std::milli>(t_now - st.t_last).count();
    st.t_last = t_now;

    fprintf(stderr, "\n[v100-opt] тайминг операторов: %d графов, GPU = %.3f мс/граф, HOST = %.3f мс/граф\n",
            st.graphs, total/graphs, host_ms/graphs);
    fprintf(stderr, "%-46s %8s %10s %10s %7s\n", "оператор", "н/граф", "мс/вызов", "мс/граф", "%");
    for (const auto & kv : st.acc) {
        const double ms_graph = kv.second.first / graphs;
        const double per_call = kv.second.first / (double) kv.second.second;
        fprintf(stderr, "%-46s %8.1f %10.3f %10.4f %6.2f%%\n",
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
