// moe_ffn_cuda vs a double-precision CPU reference of the IR op moe_ffn
// (top-k of the router logits, ties to the lower index; renorm gates =
// softmax over the selected logits; SwiGLU experts summed in ascending index),
// E = 8, k = 2, d = 96, d_ff = 160, renorm and raw gates, T from 1 (routed:
// T * k <= 32) to 40 (dense). Pass: max |error| < 1e-4 everywhere.
// EMBER_MOE_DENSE=1 tests the dense path at every T.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/moe_ffn_test.cu cuda_kernels.cu logger.cpp -lcublas -o moe_ffn_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

int main() {
  const int E = 8, k = 2, d = 96, d_ff = 160;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 0.3f);
  auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = nd(rng); };
  std::vector<float> R((size_t)E * d), G((size_t)E * d_ff * d), U(G.size()), D((size_t)E * d * d_ff);
  fill(R); fill(G); fill(U); fill(D);
  float *dR, *dG, *dU, *dD, *dx, *dout, *dscratch;
  cudaMalloc(&dR, R.size() * 4); cudaMalloc(&dG, G.size() * 4); cudaMalloc(&dU, U.size() * 4); cudaMalloc(&dD, D.size() * 4);
  cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dG, G.data(), G.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dU, U.data(), U.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dD, D.data(), D.size() * 4, cudaMemcpyHostToDevice);
  cublasHandle_t h;
  cublasCreate(&h);
  int fails = 0;
  for (int renorm = 1; renorm >= 0; --renorm) {
    for (int T : {1, 2, 5, 16, 17, 40}) {
      std::vector<float> x((size_t)T * d);
      for (auto& v : x) v = nd(rng) * 3.0f;
      std::vector<double> ref((size_t)T * d, 0.0);
      for (int t = 0; t < T; ++t) {
        const float* xt = x.data() + (size_t)t * d;
        std::vector<double> r(E);
        for (int e = 0; e < E; ++e) {
          double s = 0;
          for (int i = 0; i < d; ++i) s += (double)R[(size_t)e * d + i] * xt[i];
          r[e] = s;
        }
        std::vector<int> sel;
        std::vector<bool> taken(E, false);
        for (int j = 0; j < k; ++j) {
          int best = -1;
          for (int e = 0; e < E; ++e)
            if (!taken[e] && (best < 0 || r[e] > r[best])) best = e;
          taken[best] = true;
          sel.push_back(best);
        }
        std::sort(sel.begin(), sel.end());
        double mx = -1e300, z = 0;
        for (int e = 0; e < E; ++e) if (!renorm || taken[e]) mx = std::max(mx, r[e]);
        for (int e = 0; e < E; ++e) if (!renorm || taken[e]) z += std::exp(r[e] - mx);
        for (int e : sel) {
          const double gate = std::exp(r[e] - mx) / z;
          std::vector<double> hid(d_ff);
          for (int i = 0; i < d_ff; ++i) {
            double gv = 0, uv = 0;
            for (int c = 0; c < d; ++c) {
              gv += (double)G[((size_t)e * d_ff + i) * d + c] * xt[c];
              uv += (double)U[((size_t)e * d_ff + i) * d + c] * xt[c];
            }
            hid[i] = gv / (1.0 + std::exp(-gv)) * uv;
          }
          for (int o = 0; o < d; ++o) {
            double acc = 0;
            for (int i = 0; i < d_ff; ++i) acc += (double)D[((size_t)e * d + o) * d_ff + i] * hid[i];
            ref[(size_t)t * d + o] += gate * acc;
          }
        }
      }
      cudaMalloc(&dx, x.size() * 4);
      cudaMalloc(&dout, x.size() * 4);
      cudaMalloc(&dscratch, moe_ffn_scratch_floats(T, d, d_ff, E, k) * 4);
      cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
      moe_ffn_cuda(h, dx, T, d, d_ff, E, k, renorm, dR, dG, dU, dD, dscratch, dout, 0);
      std::vector<float> out(x.size());
      cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
      double err = 0;
      for (size_t i = 0; i < out.size(); ++i) err = std::max(err, std::fabs(out[i] - ref[i]));
      const bool ok = err < 1e-4;
      fails += !ok;
      printf("%s T %2d (%s) max |err| %.2e %s\n", renorm ? "renorm" : "raw   ", T, T * k <= 32 ? "routed" : "dense ",
             err, ok ? "ok" : "FAIL");
      cudaFree(dx); cudaFree(dout); cudaFree(dscratch);
    }
  }
  printf(fails ? "FAILED %d\n" : "all ok\n", fails);
  return fails != 0;
}
