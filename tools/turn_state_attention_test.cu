// The device kernels of per-turn state tokens (graph_ir_cuda.h; IR_SPEC 6.8) against host references:
//   ir_turn_state_attention_cuda vs a double-precision reference of the turn_state mask (row j visible
//     from the query at row r0 + t iff s(j) == s(t) or c(j) == 1), for head dims 32, 64, 80 and 128,
//     4 query heads over 2 kv heads, 1, 5 and 37 query rows after 0, 9 and 1100 held rows (1100: a
//     second 1024-row chunk, the first one entirely masked for the later segments' queries), with the
//     row base from the host and, bit for bit the same, from device memory (the captured decode step);
//   ir_kv_compact_cuda vs a host gather of the kept rows, bit for bit, on subsets of 40 and 700 rows;
//   ir_embedding_state_cuda vs host rows of the two tables, bit for bit.
// Pass: attention max |out - ref| / max(1, max |ref|) < 1e-5, every bit comparison equal; exit 1 otherwise.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/turn_state_attention_test.cu graph_ir_cuda.cu graph_ir_expr.cpp -o turn_state_attention_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "graph_ir_cuda.h"

namespace {

template <typename T>
T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

template <typename T>
std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

}  // namespace

int main() {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  int fails = 0;
  double worst = 0.0;
  const int hq = 4, hkv = 2;
  for (int hd : {32, 64, 80, 128}) {
    for (int R0 : {0, 9, 1100}) {
      for (int T : {1, 5, 37}) {
        const int n = R0 + T;
        // Codes: segments ascend; the first 1030 rows of the long case are segment 0's text, then
        // blocks of 4 state rows and text alternate; the call's rows open a new segment half way.
        std::vector<int> code(n);
        int seg = 0;
        for (int r = 0; r < n; ++r) {
          if (R0 == 1100 && r < 1030) {
            code[r] = 0;
            continue;
          }
          if (r == R0 + T / 2 || (r % 23 == 0 && r > 0)) ++seg;
          const bool state = ud(rng) < 0.25f;
          code[r] = 2 * seg + (state ? 1 : 0);
        }
        const size_t rowf = (size_t)hkv * hd;
        std::vector<float> q((size_t)T * hq * hd), K(n * rowf), V(n * rowf);
        for (auto& v : q) v = nd(rng);
        for (auto& v : K) v = nd(rng);
        for (auto& v : V) v = nd(rng);
        const float scale = 1.0f / std::sqrt((float)hd);
        float *dq = up(q), *dK = up(K), *dV = up(V), *dout = nullptr, *dout2 = nullptr;
        int* dcode = up(code);
        int* drow = up(std::vector<int>{R0});
        cudaMalloc(&dout, q.size() * 4);
        cudaMalloc(&dout2, q.size() * 4);
        ember_ir::ir_turn_state_attention_cuda(dq, dK, dV, dout, T, R0, dcode, hq, hkv, hd, scale, 0, nullptr);
        ember_ir::ir_turn_state_attention_cuda(dq, dK, dV, dout2, T, -7, dcode, hq, hkv, hd, scale, 0, drow);
        const std::vector<float> out = down(dout, q.size()), out2 = down(dout2, q.size());
        for (void* p : {(void*)dq, (void*)dK, (void*)dV, (void*)dout, (void*)dout2, (void*)dcode, (void*)drow}) cudaFree(p);
        const bool same = std::memcmp(out.data(), out2.data(), out.size() * 4) == 0;
        double err = 0.0, mref = 0.0;
        for (int t = 0; t < T; ++t) {
          const int qs = code[R0 + t] >> 1;
          for (int h = 0; h < hq; ++h) {
            const float* qr = q.data() + ((size_t)t * hq + h) * hd;
            const int kvh = h / (hq / hkv);
            std::vector<int> vis;
            for (int j = 0; j <= R0 + t; ++j)
              if ((code[j] >> 1) == qs || (code[j] & 1)) vis.push_back(j);
            std::vector<double> s(vis.size());
            double mx = -INFINITY;
            for (size_t i = 0; i < vis.size(); ++i) {
              const float* kr = K.data() + ((size_t)vis[i] * hkv + kvh) * hd;
              double dot = 0.0;
              for (int d = 0; d < hd; ++d) dot += (double)qr[d] * kr[d];
              s[i] = dot * scale;
              mx = std::max(mx, s[i]);
            }
            double sum = 0.0;
            for (double& v : s) sum += (v = std::exp(v - mx));
            for (int d = 0; d < hd; ++d) {
              double ref = 0.0;
              for (size_t i = 0; i < vis.size(); ++i) ref += s[i] / sum * V[((size_t)vis[i] * hkv + kvh) * hd + d];
              mref = std::max(mref, std::fabs(ref));
              err = std::max(err, std::fabs((double)out[((size_t)t * hq + h) * hd + d] - ref));
            }
          }
        }
        const double rel = err / std::max(1.0, mref);
        const bool ok = rel < 1e-5 && same;
        fails += !ok;
        worst = std::max(worst, rel);
        std::printf("attention head_dim %3d held %4d T %2d  worst rel %.3e  device row base %s  %s\n", hd, R0, T, rel,
                    same ? "bit-equal" : "DIFFERS", ok ? "ok" : "FAIL");
      }
    }
  }
  // Compaction.
  for (int n : {40, 700}) {
    const int row = 2 * 64;
    std::vector<float> K((size_t)n * row), V((size_t)n * row);
    for (auto& v : K) v = nd(rng);
    for (auto& v : V) v = nd(rng);
    std::vector<int> idx;
    for (int r = 0; r < n; ++r)
      if (r < 3 || ud(rng) < 0.4f) idx.push_back(r);
    int w0 = 0;
    while (w0 < (int)idx.size() && idx[w0] == w0) ++w0;
    float *dK = up(K), *dV = up(V);
    int* didx = up(idx);
    ember_ir::ir_kv_compact_cuda(dK, dV, didx, w0, (int)idx.size(), row, 0);
    const std::vector<float> k2 = down(dK, K.size()), v2 = down(dV, V.size());
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(didx);
    bool ok = true;
    for (size_t w = 0; w < idx.size(); ++w)
      ok = ok && std::memcmp(k2.data() + w * row, K.data() + (size_t)idx[w] * row, row * 4) == 0 &&
           std::memcmp(v2.data() + w * row, V.data() + (size_t)idx[w] * row, row * 4) == 0;
    fails += !ok;
    std::printf("kv compact: %d rows -> %zu kept (first moved row %d)  %s\n", n, idx.size(), w0, ok ? "ok" : "FAIL");
  }
  // The extended embedding.
  {
    const int vocab = 50, k = 4, d = 96;
    std::vector<float> table((size_t)vocab * d), state((size_t)k * d);
    for (auto& v : table) v = nd(rng);
    for (auto& v : state) v = nd(rng);
    const std::vector<int> ids = {0, 7, 49, 50, 51, 52, 53, 3, 50};
    float *dt = up(table), *ds = up(state), *dout = nullptr;
    int* dids = up(ids);
    cudaMalloc(&dout, ids.size() * d * 4);
    ember_ir::ir_embedding_state_cuda(dt, ds, vocab, dids, dout, (int)ids.size(), d, 0);
    const std::vector<float> out = down(dout, ids.size() * d);
    cudaFree(dt);
    cudaFree(ds);
    cudaFree(dout);
    cudaFree(dids);
    bool ok = true;
    for (size_t t = 0; t < ids.size(); ++t) {
      const float* src = ids[t] >= vocab ? state.data() + (size_t)(ids[t] - vocab) * d : table.data() + (size_t)ids[t] * d;
      ok = ok && std::memcmp(out.data() + t * d, src, d * 4) == 0;
    }
    fails += !ok;
    std::printf("embedding with state rows: %s\n", ok ? "ok" : "FAIL");
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf("turn_state_attention_test: %s (%d checks failed; attention worst %.3e)\n", fails ? "FAIL" : "PASS",
              fails, worst);
  return fails ? 1 : 0;
}
