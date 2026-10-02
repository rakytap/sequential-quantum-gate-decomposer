/* ABI-only test double; it performs no FPGA or Groq computation.
 * Build twice as libqgdDFE.so, with and without -DMOCK_GROQ.
 */
#include <stddef.h>

static int calls;

size_t get_accelerator_avail_num(void) { return 1; }
size_t get_accelerator_free_num(void) { return 1; }
int initialize_DFE(int number) { (void)number; return 0; }
void releive_DFE(void) {}
int get_chained_gates_num(void) { return 1; }
int load2LMEM(void *data, size_t rows, size_t cols) {
    (void)data; (void)rows; (void)cols;
    return 0;
}
int calcqgdKernelDFE(size_t rows, size_t cols, void *gates, int gate_count,
                     int set_count, int trace_offset, double *trace) {
    (void)rows; (void)cols; (void)gates; (void)gate_count; (void)trace_offset;
    ++calls;
    for (int i = 0; i < 3 * set_count; ++i) trace[i] = 0;
    return 0;
}
int dfe_test_call_count(void) { return calls; }

#ifdef MOCK_GROQ
/* Only symbol presence is tested; this function is never called. */
void groq_iop_init(void) {}
#endif
