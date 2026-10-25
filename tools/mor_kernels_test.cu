// The device kernels of Mixture-of-Recursions (graph_ir_cuda.h; IR_SPEC 7.4, 7.5) against host references,
// every comparison bit for bit:
//   ir_rows_gather_cuda / ir_rows_scatter_cuda vs host row copies (active subsets of 1, 7, 23 and 40 of 40
//     rows, widths 1, 32, 640 and 1000; the scatter leaves the other rows untouched);
//   ir_mor_gate_cuda vs cpu::mor_gate (graph_ir_ops.cpp, compiled without contraction), gates in (0, 1]
//     and the edge values 0, 1 and 0.5;
//   ir_rope_at_cuda and ir_rope_rows_at_cuda (interleaved full, NEOX full and partial) at positions
//     start_pos + t vs ir_rope_cuda / ir_rope_rows_cuda at start_pos, and at scattered positions vs one
//     T = 1 launch of the old kernels per row at that row's position.
// Pass: every comparison equal; exit 1 otherwise.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. -Xcompiler -ffp-contract=off tools/mor_kernels_test.cu graph_ir_cuda.cu graph_ir_expr.cpp graph_ir_ops.cpp -o mor_kernels_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "graph_ir_cuda.h"
#include "graph_ir_ops.h"

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

bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  int fails = 0, checks = 0;
  auto report = [&](const char* what, bool ok) {
    ++checks;
    fails += !ok;
    if (!ok) std::printf("FAIL: %s\n", what);
  };
  const int T = 40;
  // Gather and scatter.
  for (int A : {1, 7, 23, 40})
    for (int w : {1, 32, 640, 1000}) {
      std::vector<int> idx;
      for (int t = 0; t < T; ++t) idx.push_back(t);
      std::shuffle(idx.begin(), idx.end(), rng);
      idx.resize(A);
      std::sort(idx.begin(), idx.end());
      std::vector<float> src((size_t)T * w), comp((size_t)A * w), base((size_t)T * w);
      for (auto& v : src) v = nd(rng);
      for (auto& v : comp) v = nd(rng);
      for (auto& v : base) v = nd(rng);
      float *dsrc = up(src), *dcomp = up(comp), *dbase = up(base), *dg = nullptr;
      int* didx = up(idx);
      cudaMalloc(&dg, (size_t)A * w * sizeof(float));
      ember_ir::ir_rows_gather_cuda(dsrc, didx, dg, A, w, 0);
      ember_ir::ir_rows_scatter_cuda(dcomp, didx, dbase, A, w, 0);
      const std::vector<float> g = down(dg, (size_t)A * w), sc = down(dbase, (size_t)T * w);
      std::vector<float> g_ref((size_t)A * w), sc_ref = base;
      for (int a = 0; a < A; ++a) {
        std::memcpy(g_ref.data() + (size_t)a * w, src.data() + (size_t)idx[a] * w, w * sizeof(float));
        std::memcpy(sc_ref.data() + (size_t)idx[a] * w, comp.data() + (size_t)a * w, w * sizeof(float));
      }
      char what[96];
      std::snprintf(what, sizeof what, "gather A %d w %d", A, w);
      report(what, same(g, g_ref));
      std::snprintf(what, sizeof what, "scatter A %d w %d", A, w);
      report(what, same(sc, sc_ref));
      for (void* p : {(void*)dsrc, (void*)dcomp, (void*)dbase, (void*)dg, (void*)didx}) cudaFree(p);
    }
  std::printf("rows gather / scatter: %d checks, %d failed\n", checks, fails);
  // The gate.
  {
    const int A = 23, w = 640;
    std::vector<float> xin((size_t)A * w), x((size_t)A * w), g(A);
    for (auto& v : xin) v = nd(rng);
    for (auto& v : x) v = 3.0f * nd(rng);
    for (int a = 0; a < A; ++a) g[a] = a == 0 ? 0.0f : a == 1 ? 1.0f : a == 2 ? 0.5f : ud(rng);
    std::vector<float> ref = x;
    for (int a = 0; a < A; ++a)
      ember_ir::cpu::mor_gate(xin.data() + (size_t)a * w, ref.data() + (size_t)a * w, w, g[a]);
    float *dxin = up(xin), *dx = up(x), *dgate = up(g);
    ember_ir::ir_mor_gate_cuda(dxin, dx, dgate, A, w, 0);
    report("mor gate vs cpu::mor_gate", same(down(dx, x.size()), ref));
    for (void* p : {(void*)dxin, (void*)dx, (void*)dgate}) cudaFree(p);
    std::printf("mor gate: %s\n", fails ? "see above" : "bit-equal to cpu::mor_gate");
  }
  // RoPE at given positions.
  {
    const int max_len = 512, hq = 4, hk = 2;
    for (int hd : {16, 64, 80}) {
      for (int mode = 0; mode < 3; ++mode) {  // 0 interleaved full (ir_rope_cuda), 1 neox full, 2 neox partial
        const int n_rot = mode == 2 ? hd / 2 : hd;
        const int half = n_rot / 2;
        std::vector<float> tab((size_t)max_len * half * 2);
        for (int p = 0; p < max_len; ++p)
          for (int i = 0; i < half; ++i) {
            const float ang = p * std::pow(10000.0f, -2.0f * i / n_rot);
            tab[((size_t)p * half + i) * 2] = std::cos(ang);
            tab[((size_t)p * half + i) * 2 + 1] = std::sin(ang);
          }
        float* dtab = up(tab);
        const int n = 13;
        std::vector<float> q((size_t)n * hq * hd), k((size_t)n * hk * hd);
        for (auto& v : q) v = nd(rng);
        for (auto& v : k) v = nd(rng);
        auto run_old = [&](std::vector<float> qq, std::vector<float> kk, int rows, int start) {
          float *dq = up(qq), *dk = up(kk);
          if (mode == 0) {
            ember_ir::ir_rope_cuda(dq, dk, rows, hq, hk, hd, start, dtab, 0);
          } else {
            ember_ir::ir_rope_rows_cuda(dq, rows, hq, hd, n_rot, true, start, dtab, 0);
            ember_ir::ir_rope_rows_cuda(dk, rows, hk, hd, n_rot, true, start, dtab, 0);
          }
          qq = down(dq, qq.size());
          kk = down(dk, kk.size());
          cudaFree(dq);
          cudaFree(dk);
          qq.insert(qq.end(), kk.begin(), kk.end());
          return qq;
        };
        auto run_at = [&](const std::vector<int>& pos) {
          float *dq = up(q), *dk = up(k);
          int* dpos = up(pos);
          if (mode == 0) {
            ember_ir::ir_rope_at_cuda(dq, dk, n, hq, hk, hd, dpos, dtab, 0);
          } else {
            ember_ir::ir_rope_rows_at_cuda(dq, n, hq, hd, n_rot, true, dpos, dtab, 0);
            ember_ir::ir_rope_rows_at_cuda(dk, n, hk, hd, n_rot, true, dpos, dtab, 0);
          }
          std::vector<float> qq = down(dq, q.size()), kk = down(dk, k.size());
          cudaFree(dq);
          cudaFree(dk);
          cudaFree(dpos);
          qq.insert(qq.end(), kk.begin(), kk.end());
          return qq;
        };
        // Contiguous positions 100 .. 112: the old kernel's call at start_pos 100.
        std::vector<int> pos(n);
        for (int t = 0; t < n; ++t) pos[t] = 100 + t;
        char what[96];
        std::snprintf(what, sizeof what, "rope at contiguous positions, hd %d mode %d", hd, mode);
        report(what, same(run_at(pos), run_old(q, k, n, 100)));
        // Scattered positions: row by row through the old kernel at T = 1.
        int p = 0;
        for (int t = 0; t < n; ++t) pos[t] = p += 1 + (int)(ud(rng) * 30);
        std::vector<float> ref_q, ref_k;
        for (int t = 0; t < n; ++t) {
          std::vector<float> qr(q.begin() + (size_t)t * hq * hd, q.begin() + (size_t)(t + 1) * hq * hd);
          std::vector<float> kr(k.begin() + (size_t)t * hk * hd, k.begin() + (size_t)(t + 1) * hk * hd);
          const std::vector<float> o = run_old(qr, kr, 1, pos[t]);
          ref_q.insert(ref_q.end(), o.begin(), o.begin() + hq * hd);
          ref_k.insert(ref_k.end(), o.begin() + hq * hd, o.end());
        }
        ref_q.insert(ref_q.end(), ref_k.begin(), ref_k.end());
        std::snprintf(what, sizeof what, "rope at scattered positions, hd %d mode %d", hd, mode);
        report(what, same(run_at(pos), ref_q));
        cudaFree(dtab);
      }
    }
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf("mor_kernels_test: %s (%d of %d checks failed)\n", fails ? "FAIL" : "PASS", fails, checks);
  return fails ? 1 : 0;
}
