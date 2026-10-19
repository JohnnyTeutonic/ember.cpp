#pragma once
// CUDA kernels for the IR interpreter's device path (graph_ir.cpp), the ops
// cuda_kernels.h has no form for. Tensors are [T, width] row-major device
// buffers; every launch goes on `stream` and none synchronises the host. The
// arithmetic follows the CPU interpreter's kernels (paperkiln ir.hpp gives
// each op's EXACT/ROUNDING class); products are rounded separately where the
// CPU form has no FMA.
#ifdef HAS_CUDA
#include <cuda_runtime.h>

namespace ember_ir {

// out[t] = table[tokens[t]] (rows of d floats).
void ir_embedding_cuda(const float* table, const int* tokens, float* out, int T, int d, cudaStream_t stream);
// out[t] = x[t] + P[start_pos + t]. EXACT.
void ir_pos_embedding_cuda(const float* x, const float* P, float* out, int T, int d, int start_pos,
                           cudaStream_t stream);
// (x - mean) / sqrt(biased var + eps) * w + b, mean and variance accumulated
// in double as the CPU kernel does. out may not alias x.
void ir_layernorm_cuda(const float* x, const float* w, const float* b, float* out, int T, int d, float eps,
                       cudaStream_t stream);
// y[t][o] += b[o].
void ir_bias_add_cuda(float* y, const float* b, int T, int O, cudaStream_t stream);
// Element-wise over n floats: op 0 add, 1 mul.
void ir_binary_cuda(const float* a, const float* b, float* out, size_t n, int op, cudaStream_t stream);
// x * (1 / (1 + exp(-x))) (utils.cpp silu_cpu).
void ir_silu_cuda(const float* x, float* out, size_t n, cudaStream_t stream);
// tanh form or erf form (approx "none").
void ir_gelu_cuda(const float* x, float* out, size_t n, bool tanh_form, cudaStream_t stream);
// Interleaved (adjacent-pair) RoPE on q [T][hq*hd] and k [T][hk*hd] in place,
// positions start_pos + t, table [max_len][hd/2] of (cos, sin) pairs.
void ir_rope_cuda(float* q, float* k, int T, int hq, int hk, int hd, int start_pos, const float* freqs,
                  cudaStream_t stream);

}  // namespace ember_ir
#endif  // HAS_CUDA
