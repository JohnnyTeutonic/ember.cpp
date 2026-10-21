#include "graph_ir_mamba.h"

#include <cmath>
#include <cstddef>
#include <cstring>

namespace ember_ir {
namespace cpu {

void conv1d_causal_row(const float* x, const float* w, const float* bias, float* state, float* out, int C, int K) {
  for (int c = 0; c < C; ++c) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float v = k < K - 1 ? state[static_cast<size_t>(k) * C + c] : x[c];
      acc = acc + w[static_cast<size_t>(c) * K + k] * v;
    }
    out[c] = bias ? acc + bias[c] : acc;
  }
  if (K > 1) {  // the state becomes the last K - 1 rows of [state; x]
    std::memmove(state, state + C, static_cast<size_t>(K - 2) * C * sizeof(float));
    std::memcpy(state + static_cast<size_t>(K - 2) * C, x, static_cast<size_t>(C) * sizeof(float));
  }
}

void selective_scan_row(const float* x, const float* dt, const float* B, const float* C, const float* A,
                        const float* d, float* s, float* y, int D, int N, bool softplus) {
  for (int c = 0; c < D; ++c) {
    const float delta = softplus ? (dt[c] <= 20.0f ? std::log1p(std::exp(dt[c])) : dt[c]) : dt[c];
    const float xd = x[c] * delta;
    float* sc = s + static_cast<size_t>(c) * N;
    const float* ac = A + static_cast<size_t>(c) * N;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float decay = sc[n] * std::exp(delta * ac[n]);
      sc[n] = decay + B[n] * xd;
      acc = acc + sc[n] * C[n];
    }
    y[c] = d ? acc + d[c] * x[c] : acc;
  }
}

}  // namespace cpu
}  // namespace ember_ir
