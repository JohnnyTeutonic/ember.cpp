// ir_window_attention_cuda (graph_ir_cuda.cu) against the CPU form
// attention_window_batch_cpu (cpu_attention.cpp) on random inputs, grouped
// heads (hq 4, hkv 2), head_dim 64 and 96. Each case fills a KV cache, NaNs
// past the last written position (a kernel that reads a masked future key
// fails), then runs a prefill of T tokens and decode steps (T = 1) or chunks
// past the window: windows smaller than the sinks, window >= T, window 1, and
// a window over 1024 visible keys (the kernel's multi-chunk online softmax).
// Pass: max |d| < 1e-5 everywhere.
// Build (from the CUDA build directory, after make tinyllama_server):
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I EMBER -I JSON_INC EMBER/tools/window_attention_cuda_test.cu
//        libtinyllama_core.a -lcublas -lgomp -lboost_regex -lpthread -o window_attention_cuda_test
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cpu_attention.h"
#include "graph_ir_cuda.h"

namespace {

struct Case {
  int window, sinks, hd, prefill, step, steps;  // prefill T, then `steps` calls of `step` tokens
};

int failures = 0;

}  // namespace

int main() {
  const int hq = 4, hkv = 2;
  const Case cases[] = {
      {8, 2, 64, 20, 1, 30},     // decode well past the window
      {4, 6, 64, 3, 1, 40},      // window smaller than the sinks
      {32, 4, 64, 10, 7, 12},    // window >= T, chunked prefill crossing the window
      {16, 0, 96, 40, 1, 20},    // no sinks, head_dim 96
      {1, 1, 64, 5, 1, 10},      // window 1
      {3, 0, 64, 1, 1, 12},      // decode from position 0
      {2000, 4, 64, 1500, 1, 4}, // > 1024 visible keys: several chunks
      {700, 3, 96, 1200, 5, 3},  // window of 700 + sinks, past it, T = 5
  };
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.0f);
  double worst_all = 0.0;
  for (const Case& c : cases) {
    const int total = c.prefill + c.step * c.steps;
    const size_t row = (size_t)hkv * c.hd;
    const float scale = 1.0f / std::sqrt(float(c.hd));
    std::vector<float> Kall(total * row), Vall(total * row), Qall((size_t)total * hq * c.hd);
    for (auto& x : Kall) x = nd(rng);
    for (auto& x : Vall) x = nd(rng);
    for (auto& x : Qall) x = nd(rng) * 2.0f;  // sharper softmaxes
    KVCacheLayer cache;
    cache.k.assign(total * row, std::numeric_limits<float>::quiet_NaN());
    cache.v = cache.k;
    float *dK, *dV, *dq, *dout;
    cudaMalloc(&dK, total * row * sizeof(float));
    cudaMalloc(&dV, total * row * sizeof(float));
    cudaMalloc(&dq, Qall.size() * sizeof(float));
    cudaMalloc(&dout, Qall.size() * sizeof(float));
    cudaMemcpy(dK, cache.k.data(), cache.k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, cache.v.data(), cache.v.size() * sizeof(float), cudaMemcpyHostToDevice);
    double worst = 0.0;
    int calls = 0;
    for (int start = 0; start < total; ++calls) {
      const int T = start == 0 ? c.prefill : c.step;
      // Write positions [start, start + T) to both caches (later ones stay NaN).
      for (size_t i = start * row; i < (start + T) * row; ++i) {
        cache.k[i] = Kall[i];
        cache.v[i] = Vall[i];
      }
      cudaMemcpy(dK + start * row, Kall.data() + start * row, T * row * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dV + start * row, Vall.data() + start * row, T * row * sizeof(float), cudaMemcpyHostToDevice);
      const size_t qn = (size_t)T * hq * c.hd;
      std::vector<float> q(Qall.begin() + (size_t)start * hq * c.hd, Qall.begin() + (size_t)start * hq * c.hd + qn);
      std::vector<float> ref, got(qn);
      attention_window_batch_cpu(q, cache, ref, T, start, hq, hkv, c.hd, scale, c.window, c.sinks);
      cudaMemcpy(dq, q.data(), qn * sizeof(float), cudaMemcpyHostToDevice);
      ember_ir::ir_window_attention_cuda(dq, dK, dV, dout, T, start, hq, hkv, c.hd, scale, c.window, c.sinks, 0);
      cudaMemcpy(got.data(), dout, qn * sizeof(float), cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < qn; ++i) {
        const double d = std::fabs((double)got[i] - ref[i]);
        if (!(d <= worst)) worst = std::isnan(d) ? 1e30 : d;
      }
      start += T;
    }
    const cudaError_t e = cudaDeviceSynchronize();
    const bool ok = e == cudaSuccess && worst < 1e-5;
    failures += !ok;
    if (worst > worst_all) worst_all = worst;
    std::printf("%s window %4d sinks %d head_dim %3d prefill %4d then %2d x T=%d (%d calls): max |d| %.3g%s\n",
                ok ? "ok  " : "FAIL", c.window, c.sinks, c.hd, c.prefill, c.steps, c.step, calls, worst,
                e == cudaSuccess ? "" : cudaGetErrorString(e));
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dq);
    cudaFree(dout);
  }
  std::printf("%s: max |d| %.3g over all cases (tolerance 1e-5)\n", failures ? "FAIL" : "PASS", worst_all);
  return failures ? 1 : 0;
}
