#pragma once
// CPU kernels for the IR ops of the gated and attention-residual families
// (paperkiln ir.hpp at 176b425: flex highway/plain, attnres, srd's router):
// sub, relu, sigmoid, affine, mul_col, rms_row and depth_attend. Each one is
// the named paperkiln CPU op (src/ops.cpp, nn::AttnResStack) in its order of
// operations, every product and sum rounded to f32 on its own: this file is
// compiled with -ffp-contract=off (CMakeLists.txt), so no a * b + c becomes a
// fused multiply-add. Kept free of ember's other headers so the op-level gate
// can build it alone. Rows are [T, width] row-major.
#include <cstddef>
#include <vector>

namespace ember_ir {
namespace cpu {

// a - b elementwise. EXACT.
void sub(const float* a, const float* b, float* out, size_t n);
// max(0, x) as std::max(0.0f, x) (ops::relu: -0 and NaN give +0). EXACT.
void relu(const float* x, float* out, size_t n);
// 1 / (1 + exp(-x)) (ops::sigmoid). ROUNDING.
void sigmoid(const float* x, float* out, size_t n);
// (x * scale) + shift: the product rounded, then the sum (ops::scale, then
// ops::add_scalar). EXACT.
void affine(const float* x, float* out, size_t n, float scale, float shift);
// x [T, w] times the column c [T, 1]: out[t, j] = x[t, j] * c[t]. EXACT.
void mul_col(const float* x, const float* c, float* out, int T, int w);
// Per feature: out[t, j] = (x[t, j] * g[j]) + b[j], the product rounded, then
// the sum; b null: the product alone (ops `scale`, a gain without a [d, d]
// diagonal linear). EXACT.
void scale(const float* x, const float* g, const float* b, float* out, int T, int w);
// [T, w] -> [T, 1]: sqrt(ss / w + eps), ss = sum of x[t, j]^2 in ascending j
// (ops::rms_row). ROUNDING.
void rms_row(const float* x, float* out, int T, int w, float eps);
// Attention over depth (nn::AttnResStack::depth_attend, block form): sources
// s_0..s_{S-1}, each [T, d]; w the pseudo-query [d]. S = 1: out = s_0 and w
// is not read (may be null). S >= 2, per row: r_i = 1 / sqrt(ss_i / d + eps)
// (unit-weight rmsnorm), l_i = sum_j (s_i[j] * r_i) * w[j] in ascending j,
// a = max-shifted softmax over i (z summed in ascending i, a_i = e_i / z),
// out = s_0 a_0 + s_1 a_1 + ... in ascending i. ROUNDING.
void depth_attend(const std::vector<const float*>& sources, const float* w, float* out, int T, int d, float eps);

}  // namespace cpu
}  // namespace ember_ir
