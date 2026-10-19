// Decode attention vs a double-precision CPU reference, for the three CUDA
// entry points: attention_cuda_optimized (per-launch decode),
// attention_devpos_cuda (graph decode, length from device memory) and
// attention_cuda (basic). 32 query / 4 KV heads, head_dim 64, a cache of
// 8192 positions, lengths across the split kernel's chunk boundaries.
// Pass: max |error| < 1e-4 everywhere. EMBER_DECODE_ATTN=legacy tests the
// one-block-per-head kernel instead of the split kernel.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/decode_attention_test.cu cuda_kernels.cu logger.cpp -lcublas -o decode_attention_test
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

int main() {
  const int H = 32, KV = 4, D = 64, MAX = 8192;
  const float scale = 1.0f / std::sqrt((float)D);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> K((size_t)MAX * KV * D), V(K.size()), Q(H * D);
  for (auto& x : K) x = nd(rng);
  for (auto& x : V) x = nd(rng);
  for (auto& x : Q) x = nd(rng) * 2.0f;
  float *dK, *dV, *dQ, *dO;
  int* dstep;
  cudaMalloc(&dK, K.size() * 4); cudaMalloc(&dV, V.size() * 4); cudaMalloc(&dQ, Q.size() * 4);
  cudaMalloc(&dO, H * D * 4); cudaMalloc(&dstep, 2 * 4);
  cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dQ, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
  int fails = 0;
  for (int len : {1, 2, 31, 255, 256, 257, 300, 1000, 4096, 4097, 8000, 8192}) {
    std::vector<double> ref(H * D);
    for (int h = 0; h < H; ++h) {
      const int kvh = h / (H / KV);
      std::vector<double> s(len);
      double mx = -1e300;
      for (int k = 0; k < len; ++k) {
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += (double)Q[h * D + d] * K[((size_t)k * KV + kvh) * D + d];
        s[k] = dot * scale;
        mx = std::max(mx, s[k]);
      }
      double z = 0;
      for (auto& x : s) z += (x = std::exp(x - mx));
      for (int d = 0; d < D; ++d) {
        double a = 0;
        for (int k = 0; k < len; ++k) a += s[k] * V[((size_t)k * KV + kvh) * D + d];
        ref[h * D + d] = a / z;
      }
    }
    for (int mode = 0; mode < 3; ++mode) {
      cudaMemset(dO, 0, H * D * 4);
      if (mode == 0) {
        attention_cuda_optimized(dQ, dK, dV, dO, H, len, D, scale, MAX, KV, 0);
      } else if (mode == 2) {
        attention_cuda(dQ, dK, dV, dO, H, len, D, scale, MAX, KV, 0);
      } else {
        int st[2] = {len - 1, len - 1};
        cudaMemcpy(dstep, st, 8, cudaMemcpyHostToDevice);
        attention_devpos_cuda(dQ, dK, dV, dO, H, D, scale, MAX, KV, dstep, 0);
      }
      std::vector<float> o(H * D);
      cudaMemcpy(o.data(), dO, H * D * 4, cudaMemcpyDeviceToHost);
      double err = 0;
      for (int i = 0; i < H * D; ++i) err = std::max(err, std::fabs(o[i] - ref[i]));
      const bool ok = err < 1e-4;
      fails += !ok;
      printf("len %5d %-10s max |err| %.2e %s\n", len, mode == 0 ? "per-launch" : mode == 1 ? "devpos" : "basic", err, ok ? "ok" : "FAIL");
    }
  }
  printf(fails ? "FAILED %d\n" : "all ok\n", fails);
  return fails != 0;
}
