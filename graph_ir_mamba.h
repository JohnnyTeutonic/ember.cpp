#pragma once
// CPU kernels for Mamba-1's IR ops (paperkiln include/microtorch/ir.hpp: conv1d_causal and
// selective_scan; numpy reference tools/ir_numpy.py; llama.cpp ggml_ssm_conv / ggml_ssm_scan).
// One token (row) per call, so prefill is the sequential scan and decode its T = 1 case, and the
// interpreter can snapshot the state after any row. Every product is rounded before it is added
// (compiled with -ffp-contract=off) and sums run in the ascending order the spec names. ROUNDING.
namespace ember_ir {
namespace cpu {

// Depthwise causal convolution over time. x [C] is this token; state [K - 1, C] holds the K - 1
// previous inputs, oldest first (0 at the start). With xx = state's rows then x:
//   out[c] = sum over k = 0..K-1, ascending, of w[c, k] * xx[k][c]   (+ bias[c] after the sum)
// then state becomes the last K - 1 rows of xx. w is [C, K] row-major; bias may be null.
void conv1d_causal_row(const float* x, const float* w, const float* bias, float* state, float* out, int C, int K);

// Mamba-1's selective scan, one token. x and dt [D], B and C [N]; A [D, N] as stored (llama.cpp
// writes -exp(A_log)); d [D] or null; s [D, N] the state. For every channel c:
//   delta = softplus ? (dt[c] <= 20 ? log1p(exp(dt[c])) : dt[c]) : dt[c]
//   xd = x[c] * delta;  s[c, n] = s[c, n] * exp(delta * A[c, n]) + B[n] * xd
//   y[c] = sum over n ascending of s[c, n] * C[n]   (+ d[c] * x[c] after the sum)
void selective_scan_row(const float* x, const float* dt, const float* B, const float* C, const float* A,
                        const float* d, float* s, float* y, int D, int N, bool softplus);

}  // namespace cpu
}  // namespace ember_ir
