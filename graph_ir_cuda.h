#pragma once
// CUDA kernels for the IR interpreter's device path (graph_ir.cpp), the ops
// cuda_kernels.h has no form for. Tensors are [T, width] row-major device
// buffers; every launch goes on `stream` and none synchronises the host. The
// arithmetic follows the CPU interpreter's kernels (paperkiln ir.hpp gives
// each op's EXACT/ROUNDING class); products are rounded separately where the
// CPU form has no FMA.
//
// Device position (the captured T = 1 decode step): the kernels that depend
// on the position take `dpos`, a device int; when it is given they read the
// first position from it and ignore start_pos, so a CUDA graph captured once
// replays at any position.
#ifdef HAS_CUDA
#include <cuda_runtime.h>

#include <vector>

#include "graph_ir_expr.h"

namespace ember_ir {

// out[t] = table[tokens[t]] (rows of d floats).
void ir_embedding_cuda(const float* table, const int* tokens, float* out, int T, int d, cudaStream_t stream);
// out[t] = x[t] + P[start_pos + t]. EXACT.
void ir_pos_embedding_cuda(const float* x, const float* P, float* out, int T, int d, int start_pos,
                           cudaStream_t stream, const int* dpos = nullptr);
// (x - mean) / sqrt(biased var + eps) * w + b, mean and variance accumulated
// in double as the CPU kernel does. out may not alias x.
void ir_layernorm_cuda(const float* x, const float* w, const float* b, float* out, int T, int d, float eps,
                       cudaStream_t stream);
// y[t][o] += b[o].
void ir_bias_add_cuda(float* y, const float* b, int T, int O, cudaStream_t stream);
// Element-wise over n floats: op 0 add, 1 mul, 2 sub (a - b). EXACT.
void ir_binary_cuda(const float* a, const float* b, float* out, size_t n, int op, cudaStream_t stream);
// x * (1 / (1 + exp(-x))) (utils.cpp silu_cpu).
void ir_silu_cuda(const float* x, float* out, size_t n, cudaStream_t stream);
// tanh form or erf form (approx "none").
void ir_gelu_cuda(const float* x, float* out, size_t n, bool tanh_form, cudaStream_t stream);
// Interleaved (adjacent-pair) RoPE on q [T][hq*hd] and k [T][hk*hd] in place,
// positions start_pos + t, table [max_len][hd/2] of (cos, sin) pairs.
void ir_rope_cuda(float* q, float* k, int T, int hq, int hk, int hd, int start_pos, const float* freqs,
                  cudaStream_t stream, const int* dpos = nullptr);
// RoPE over the first n_rot dims of each head of x [T][heads*hd] in place
// (graph_ir.cpp rope_rows_cpu): pairs (2i, 2i+1) or, neox, (i, i + n_rot/2);
// table [max_len][n_rot/2] of (cos, sin) pairs. Any n_rot (even, <= hd).
void ir_rope_rows_cuda(float* x, int T, int heads, int hd, int n_rot, bool neox, int start_pos, const float* freqs,
                       cudaStream_t stream, const int* dpos = nullptr);
// The rows of k, v [T][row] into the caches K, V [pos][row] at positions
// start_pos + t. EXACT (a copy).
void ir_kv_write_cuda(float* K, float* V, const float* k, const float* v, int T, int row, int start_pos,
                      cudaStream_t stream, const int* dpos = nullptr);
// Sliding-window attention with sinks (paperkiln swa_attention; CPU form
// cpu_attention.cpp attention_window_batch_cpu) for q [T][hq*hd] at positions
// start_pos + t over a cache K, V [pos][hkv][hd] already holding positions <=
// start_pos + T - 1. Query p sees [0, min(sinks, p+1, win_lo)) and [win_lo, p],
// win_lo = max(0, p+1-window). window >= 1 (window 0 is attention_*_cuda's);
// head_dim <= 1024. Any T, decode included.
void ir_window_attention_cuda(const float* q, const float* K, const float* V, float* out, int T, int start_pos,
                              int hq, int hkv, int hd, float scale, int window, int sinks, cudaStream_t stream,
                              const int* dpos = nullptr);

// Per-turn state tokens (IR_SPEC 6.8; CPU forms graph_ir.cpp ts_begin / ts_evict_below and
// cpu_attention.cpp attention_rows_batch_cpu).
// Attention mask turn_state for q [T][hq*hd] over a compact cache K, V [row][hkv][hd]: codes[r] =
// 2 s + c for the position row r holds (s its segment, c = 1 at a state position), the call's token t
// at row row0 + t (from *drow when given); row j <= row0 + t is visible iff s(j) == s(t) or c(j) == 1.
// Rows ascending, an online softmax over chunks of 1024 rows, masked rows never read: ROUNDING against
// the CPU's sequential sums. head_dim <= 1024. Any T, decode included.
void ir_turn_state_attention_cuda(const float* q, const float* K, const float* V, float* out, int T, int row0,
                                  const int* codes, int hq, int hkv, int hd, float scale, cudaStream_t stream,
                                  const int* drow = nullptr);
// Eviction's compaction: rows w0 .. n-1 of K and V [row][row floats] take rows idx[w] (idx ascending,
// idx[w] >= w), in place. EXACT (copies).
void ir_kv_compact_cuda(float* K, float* V, const int* idx, int w0, int n, int row, cudaStream_t stream);
// The extended embedding: out[t] = table[id] for id < vocab, state[id - vocab] otherwise. EXACT.
void ir_embedding_state_cuda(const float* table, const float* state, int vocab, const int* tokens, float* out, int T,
                             int d, cudaStream_t stream);

// The gated and attention-residual ops (CPU forms graph_ir_ops.h, same order
// of operations; every product and sum rounded on its own).
// relu: max(0, x) as std::max(0.0f, x) (-0 and NaN give +0). EXACT.
void ir_relu_cuda(const float* x, float* out, size_t n, cudaStream_t stream);
// 1 / (1 + exp(-x)). ROUNDING (expf).
void ir_sigmoid_cuda(const float* x, float* out, size_t n, cudaStream_t stream);
// (x * scale) + shift. EXACT.
void ir_affine_cuda(const float* x, float* out, size_t n, float scale, float shift, cudaStream_t stream);
// x [T, w] times the column c [T, 1]. EXACT.
void ir_mul_col_cuda(const float* x, const float* c, float* out, int T, int w, cudaStream_t stream);
// Per feature: out[t, j] = (x[t, j] * g[j]) + b[j] (b null: the product alone). EXACT.
void ir_scale_cuda(const float* x, const float* g, const float* b, float* out, int T, int w, cudaStream_t stream);
// [T, w] -> [T, 1]: sqrt(ss / w + eps), ss summed in ascending j (one thread
// per row, the CPU order). Bitwise equal to the CPU form.
void ir_rms_row_cuda(const float* x, float* out, int T, int w, float eps, cudaStream_t stream);
// Attention over depth, S >= 2 sources of [T, d] (S = 1 is a copy, done by
// the caller), pseudo-query w [d]. Each source's logit is summed by one
// thread in ascending j and the softmax by one thread in ascending i, as the
// CPU form; only expf differs from it. S <= kIrDepthSourcesMax. The source
// pointers are passed by value (a kernel parameter), so a graph keeps them.
constexpr int kIrDepthSourcesMax = 128;
void ir_depth_attend_cuda(const float* const* sources, int S, const float* w, float* out, int T, int d, float eps,
                          cudaStream_t stream);

// Recurrent ops over device state (graph_ir.cpp linear_attention_row_cpu and
// ssm_scan_row_cpu, row by row in t: a prefill is the decode update applied
// to each row in order). ROUNDING (expf).
// linear_attention: q, k, v, out [T][width]; num, den [width] updated in
// place, num and den taking row t before out[t] is formed. One thread per
// channel (channels are independent).
void ir_linear_attention_cuda(const float* q, const float* k, const float* v, float* num, float* den, float* out,
                              int T, int width, float den_floor, cudaStream_t stream);
// ssm_scan: u, y [T][n]; A [n, n] row-major; B, C [n]; D a device scalar;
// s [n] updated in place: s_t[i] = (sum_j A[i][j] s_{t-1}[j], ascending j) +
// B[i] u[t, i]; y = C s_t + D u. One block, the state in shared memory; n <=
// kIrSsmMax.
constexpr int kIrSsmMax = 6144;
void ir_ssm_scan_cuda(const float* u, const float* A, const float* B, const float* C, const float* D, float* s,
                      float* y, int T, int n, cudaStream_t stream);

// Mamba-1 (graph_ir_mamba.h cpu::conv1d_causal_row and selective_scan_row,
// row by row in t): one thread per channel, its state updated in place.
// conv1d_causal: x, out [T][C]; w [C][K]; bias [C] or null; state [K-1][C].
void ir_conv1d_causal_cuda(const float* x, const float* w, const float* bias, float* state, float* out, int T, int C,
                           int K, cudaStream_t stream);
// selective_scan: x, dt, y [T][D]; B, C [T][N]; A [D][N]; d [D] or null;
// s [D][N]. delta = softplus(dt) (log1p(exp(dt)), dt itself above 20) when
// softplus. ROUNDING (expf, log1pf).
void ir_selective_scan_cuda(const float* x, const float* dt, const float* B, const float* C, const float* A,
                            const float* d, float* s, float* y, int T, int D, int N, bool softplus,
                            cudaStream_t stream);

// expr (graph_ir_expr.h lower): out [T, w] from ins (each [T, w]); scratch
// holds ir_expr_scratch_floats(stages, T, w) floats (a reduction's argument
// and the columns). Every element runs the stage's stack program; reductions
// are one thread per row in ascending columns, as the CPU's eval.
size_t ir_expr_scratch_floats(const std::vector<expr::Stage>& stages, int T, int w);
void ir_expr_cuda(const std::vector<expr::Stage>& stages, const float* const* ins, int n_ins, float* scratch,
                  float* out, int T, int w, cudaStream_t stream);

}  // namespace ember_ir
#endif  // HAS_CUDA
