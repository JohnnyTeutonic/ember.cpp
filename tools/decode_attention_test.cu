// Decode attention vs a double-precision CPU reference, for the four CUDA
// entry points: attention_cuda_optimized (per-launch decode),
// attention_devpos_cuda (graph decode, length from device memory),
// attention_decode_at_cuda (host length, the cache's chunking) and
// attention_cuda (basic). Shapes: 32 query / 4 KV heads, head_dim 64, a cache
// of 8192 positions (lengths across the chunk boundaries of both split-K
// kernels), and smaller ones: pick_e1_L16's 5 x 64 in a 512-position cache,
// grouped-query head_dim 128, head_dim 16, head_dim 80 and head_dim 6 (which
// the tiled kernel leaves to the split kernel).
// Pass: max |error| < 1e-4 everywhere; with the default (tiled) kernel also
// per-launch = devpos = decode_at bit for bit (its chunks depend only on the
// length); devpos bit-identical over two runs for every fixed-order kernel (tiled,
// split; legacy only under EMBER_DETERMINISTIC=1, as its default form adds with atomics).
// EMBER_DECODE_ATTN=split tests the 256-key split kernel, =legacy the
// one-block-per-head kernel.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/decode_attention_test.cu cuda_kernels.cu logger.cpp -lcublas -o decode_attention_test
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.h"
#include "determinism.h"

namespace {

struct Shape {
  int H, KV, D, MAX;
  std::vector<int> lens;
};

int run_shape(const Shape& sh, bool tiled) {
  const int H = sh.H, KV = sh.KV, D = sh.D, MAX = sh.MAX;
  const float scale = 1.0f / std::sqrt((float)D);
  std::mt19937 rng(1 + H * 1000 + D);
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
  for (int len : sh.lens) {
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
    std::vector<std::vector<float>> outs;
    for (int mode = 0; mode < 5; ++mode) {
      cudaMemset(dO, 0, H * D * 4);
      if (mode == 0) {
        attention_cuda_optimized(dQ, dK, dV, dO, H, len, D, scale, MAX, KV, 0);
      } else if (mode == 3) {
        attention_cuda(dQ, dK, dV, dO, H, len, D, scale, MAX, KV, 0);
      } else if (mode == 4) {
        attention_decode_at_cuda(dQ, dK, dV, dO, H, len, D, scale, MAX, KV, 0);
      } else {  // 1, 2: devpos twice
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
      static const char* names[] = {"per-launch", "devpos", "devpos(2)", "basic", "decode_at"};
      printf("H %2d KV %2d D %3d MAX %5d len %5d %-10s max |err| %.2e %s\n", H, KV, D, MAX, len, names[mode], err,
             ok ? "ok" : "FAIL");
      outs.push_back(o);
    }
    const bool run2 = std::memcmp(outs[1].data(), outs[2].data(), H * D * 4) == 0;
    const bool same = std::memcmp(outs[0].data(), outs[1].data(), H * D * 4) == 0 &&
                      std::memcmp(outs[4].data(), outs[1].data(), H * D * 4) == 0;
    printf("H %2d KV %2d D %3d MAX %5d len %5d devpos run to run %s; per-launch = devpos = decode_at %s\n", H, KV, D,
           MAX, len, run2 ? "bit-identical" : "DIFFERENT", same ? "bit-identical" : "different");
    if (decode_attention_kind() != 2 || ember::deterministic()) fails += !run2;
    // The tiled kernel's chunks depend only on the length (D % 4 == 0; the others take the split kernel).
    if (tiled && D % 4 == 0 && !same) {
      printf("FAIL: the tiled kernel gave different bits across entry points\n");
      ++fails;
    }
  }
  cudaFree(dK); cudaFree(dV); cudaFree(dQ); cudaFree(dO); cudaFree(dstep);
  return fails;
}

}  // namespace

int main() {
  const bool tiled = decode_attention_kind() == 0;
  int mn = 0, mx = 0;
  decode_attention_tiles(&mn, &mx);
  printf("decode attention kind %d (0 tiled, 1 split, 2 legacy); tiles min_chunk %d max_chunks %d\n",
         decode_attention_kind(), mn, mx);
  const std::vector<Shape> shapes = {
      {32, 4, 64, 8192, {1, 2, 31, 32, 33, 255, 256, 257, 300, 1000, 2048, 2049, 4096, 4097, 8000, 8192}},
      {5, 5, 64, 512, {1, 5, 31, 32, 33, 64, 127, 128, 129, 383, 384, 385, 511, 512}},
      {8, 2, 128, 4096, {1, 100, 1024, 2047, 2048, 2049, 4096}},
      {4, 4, 16, 1024, {1, 17, 100, 1024}},
      {6, 3, 80, 512, {1, 40, 384, 512}},
      {2, 2, 6, 256, {1, 9, 256}},
  };
  int fails = 0;
  for (const Shape& s : shapes) fails += run_shape(s, tiled);
  printf(fails ? "FAILED %d\n" : "all ok\n", fails);
  return fails != 0;
}
