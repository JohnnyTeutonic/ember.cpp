// Built with -ffp-contract=off (CMakeLists.txt): every product below is rounded
// before it is added (graph_ir_ops.h). The WebAssembly build has no fused
// multiply-add instruction, so it cannot contract either.
#include "graph_ir_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace ember_ir {
namespace cpu {

void sub(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void relu(const float* x, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = std::max(0.0f, x[i]);
}

void sigmoid(const float* x, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

void affine(const float* x, float* out, size_t n, float scale, float shift) {
  for (size_t i = 0; i < n; ++i) {
    const float p = x[i] * scale;
    out[i] = p + shift;
  }
}

void mor_route(const float* l, int R, int& depth, float& gate) {
  int j_max = 0;
  for (int j = 1; j < R; ++j)
    if (l[j] > l[j_max]) j_max = j;  // strict: a tie keeps the lower index
  float z = 0.0f;
  for (int j = 0; j < R; ++j) z += std::exp(l[j] - l[j_max]);
  depth = j_max + 1;
  gate = std::exp(l[j_max] - l[j_max]) / z;
}

void mor_gate(const float* x_in, float* x, size_t n, float g) {
  for (size_t i = 0; i < n; ++i) {
    const float diff = x[i] - x_in[i];
    const float p = g * diff;
    x[i] = x_in[i] + p;
  }
}

void scale(const float* x, const float* g, const float* b, float* out, int T, int w) {
  for (int t = 0; t < T; ++t) {
    const size_t r = static_cast<size_t>(t) * w;
    if (b)
      for (int j = 0; j < w; ++j) out[r + j] = x[r + j] * g[j] + b[j];
    else
      for (int j = 0; j < w; ++j) out[r + j] = x[r + j] * g[j];
  }
}

void mul_col(const float* x, const float* c, float* out, int T, int w) {
  for (int t = 0; t < T; ++t) {
    const float ct = c[t];
    const size_t r = static_cast<size_t>(t) * w;
    for (int j = 0; j < w; ++j) out[r + j] = x[r + j] * ct;
  }
}

void rms_row(const float* x, float* out, int T, int w, float eps) {
  for (int t = 0; t < T; ++t) {
    const float* r = x + static_cast<size_t>(t) * w;
    float ss = 0.0f;
    for (int j = 0; j < w; ++j) {
      const float sq = r[j] * r[j];
      ss += sq;
    }
    out[t] = std::sqrt(ss / static_cast<float>(w) + eps);
  }
}

void depth_attend(const std::vector<const float*>& sources, const float* w, float* out, int T, int d, float eps) {
  const size_t S = sources.size();
  const size_t n = static_cast<size_t>(T) * d;
  if (S == 1) {  // softmax over one source is the identity mixture: s_0 bit for bit
    std::memcpy(out, sources[0], n * sizeof(float));
    return;
  }
  std::vector<float> a(S);
  for (int t = 0; t < T; ++t) {
    const size_t row = static_cast<size_t>(t) * d;
    // Logits: w . rmsnorm(s_i) (ops::rmsnorm with unit weight, ops::mul_row,
    // then the matmul with a ones column).
    for (size_t i = 0; i < S; ++i) {
      const float* s = sources[i] + row;
      float ss = 0.0f;
      for (int j = 0; j < d; ++j) {
        const float sq = s[j] * s[j];
        ss += sq;
      }
      ss /= static_cast<float>(d);
      const float r = 1.0f / std::sqrt(ss + eps);
      float l = 0.0f;
      for (int j = 0; j < d; ++j) {
        const float normed = s[j] * r;
        const float p = normed * w[j];
        l += p;
      }
      a[i] = l;
    }
    // Softmax over the sources (Matrix::apply_softmax): max-shifted, z in
    // ascending i, a_i = e_i / z.
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < S; ++i) mx = std::max(mx, a[i]);
    float z = 0.0f;
    for (size_t i = 0; i < S; ++i) {
      a[i] = std::exp(a[i] - mx);
      z += a[i];
    }
    for (size_t i = 0; i < S; ++i) a[i] /= z;
    // out = s_0 a_0 + s_1 a_1 + ... (ops::mul_col per source, ops::add in order).
    float* o = out + row;
    const float* s0 = sources[0] + row;
    for (int j = 0; j < d; ++j) o[j] = s0[j] * a[0];
    for (size_t i = 1; i < S; ++i) {
      const float* s = sources[i] + row;
      for (int j = 0; j < d; ++j) {
        const float p = s[j] * a[i];
        o[j] = o[j] + p;
      }
    }
  }
}

void mla_attention(const float* qn, const float* qr, const float* kv, float* out, int T,
                   const std::vector<std::vector<int>>& rows, int H, int r, int dn, int dr, int dv, const float* Kb,
                   const float* Vb, float scale) {
  const size_t w = static_cast<size_t>(r) + dr;
#pragma omp parallel for schedule(dynamic)
  for (int t = 0; t < T; ++t) {
    const std::vector<int>& vis = rows[t];
    std::vector<float> qa(static_cast<size_t>(r)), u(static_cast<size_t>(r)), s(vis.size());
    for (int h = 0; h < H; ++h) {
      const float* qnh = qn + (static_cast<size_t>(t) * H + h) * dn;
      const float* qrh = qr + (static_cast<size_t>(t) * H + h) * dr;
      const float* K = Kb + static_cast<size_t>(h) * r * dn;
      const float* V = Vb + static_cast<size_t>(h) * dv * r;
      for (int i = 0; i < r; ++i) {  // qa = K[h] qn_h
        float acc = 0.0f;
        for (int k = 0; k < dn; ++k) {
          const float p = K[static_cast<size_t>(i) * dn + k] * qnh[k];
          acc += p;
        }
        qa[i] = acc;
      }
      float mx = -std::numeric_limits<float>::infinity();
      for (size_t j = 0; j < vis.size(); ++j) {
        const float* c = kv + static_cast<size_t>(vis[j]) * w;
        float a = 0.0f, b = 0.0f;
        for (int i = 0; i < r; ++i) {
          const float p = qa[i] * c[i];
          a += p;
        }
        for (int e = 0; e < dr; ++e) {
          const float p = qrh[e] * c[r + e];
          b += p;
        }
        const float sum = a + b;
        s[j] = sum * scale;
        mx = std::max(mx, s[j]);
      }
      float z = 0.0f;
      for (size_t j = 0; j < vis.size(); ++j) {
        s[j] = std::exp(s[j] - mx);
        z += s[j];
      }
      std::fill(u.begin(), u.end(), 0.0f);
      for (size_t j = 0; j < vis.size(); ++j) {
        const float a = s[j] / z;
        const float* c = kv + static_cast<size_t>(vis[j]) * w;
        for (int i = 0; i < r; ++i) {
          const float p = a * c[i];
          u[i] += p;
        }
      }
      float* o = out + (static_cast<size_t>(t) * H + h) * dv;
      for (int k = 0; k < dv; ++k) {  // out = V[h] u
        float acc = 0.0f;
        for (int i = 0; i < r; ++i) {
          const float p = V[static_cast<size_t>(k) * r + i] * u[i];
          acc += p;
        }
        o[k] = acc;
      }
    }
  }
}

void split_groups(const float* x, int T, const std::vector<int>& sizes, int g, const std::vector<float*>& outs) {
  int group = 0;
  for (int s : sizes) group += s;
  const size_t width = static_cast<size_t>(group) * g;
  for (int t = 0; t < T; ++t) {
    const float* row = x + static_cast<size_t>(t) * width;
    for (int q = 0; q < g; ++q) {
      int col = 0;
      for (size_t k = 0; k < sizes.size(); ++k) {
        float* dst = outs[k] + (static_cast<size_t>(t) * g + q) * sizes[k];
        std::memcpy(dst, row + static_cast<size_t>(q) * group + col, static_cast<size_t>(sizes[k]) * sizeof(float));
        col += sizes[k];
      }
    }
  }
}

}  // namespace cpu
}  // namespace ember_ir
