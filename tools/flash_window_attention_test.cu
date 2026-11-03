// The tiled prefill attention kernel (attention_prefill_flash_kernel, cuda_kernels.cu) at the shapes the
// Gemma 3 change adds: head_dim 256 (16 keys per tile) and a sliding window. Two entry points against a
// double-precision CPU reference of grouped-query attention over the KV cache:
//   causal:  attention_batch_prefill_cuda (default kernel selection, flash2 then flash v1 then baseline),
//            which at head_dim 256 now takes the tiled kernel instead of the baseline;
//   window:  attention_window_prefill_cuda, query position p seeing keys [p - window + 1, p], and the IR's
//            window_attention_kernel (ir_window_attention_cuda, sinks 0), the kernel it replaces for
//            multi-token prefill.
// Head dims 32, 64, 96, 128, 256; heads (4 q, 1 kv), (4, 2) and (8, 8); batches of 2, 7, 40, 333 and 600
// tokens at cache offsets 0, 13 and 500; windows 1, 5, 17, 64, 512 and 1024. Pass: every output within
// max |out - ref| / max(1, max |ref|) < 1e-5, and attention_window_prefill_cuda takes every window shape
// (it returns true). Then one timing at Gemma 3 270M's shape (4 q heads, 1 kv head, head_dim 256, 512
// tokens, window 512): median of 20 of each kernel. Exit 1 on any failure.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/flash_window_attention_test.cu cuda_kernels.cu graph_ir_cuda.cu
//        logger.cpp -lcublas -o flash_window_attention_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.h"
#include "graph_ir_cuda.h"

namespace {

struct Heads {
  int hq, hkv;
};

// window 0: causal; else keys [p - window + 1, p]
double reference_err(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V,
                     const std::vector<float>& out, int T, int start, Heads h, int hd, float scale, int window) {
  double err = 0.0, mref = 0.0;
  const int group = h.hq / h.hkv;
  for (int t = 0; t < T; ++t)
    for (int qh = 0; qh < h.hq; ++qh) {
      const float* qr = q.data() + ((size_t)t * h.hq + qh) * hd;
      const int kvh = qh / group, p = start + t;
      const int lo = window > 0 ? std::max(0, p - window + 1) : 0;
      std::vector<double> s(p + 1 - lo);
      double mx = -INFINITY;
      for (int j = lo; j <= p; ++j) {
        const float* kr = K.data() + ((size_t)j * h.hkv + kvh) * hd;
        double dot = 0.0;
        for (int d = 0; d < hd; ++d) dot += (double)qr[d] * kr[d];
        s[j - lo] = dot * scale;
        mx = std::max(mx, s[j - lo]);
      }
      double sum = 0.0;
      for (double& v : s) sum += (v = std::exp(v - mx));
      for (int d = 0; d < hd; ++d) {
        double ref = 0.0;
        for (int j = lo; j <= p; ++j) ref += s[j - lo] / sum * V[((size_t)j * h.hkv + kvh) * hd + d];
        mref = std::max(mref, std::fabs(ref));
        err = std::max(err, std::fabs((double)out[((size_t)t * h.hq + qh) * hd + d] - ref));
      }
    }
  return err / std::max(1.0, mref);
}

struct Dev {
  float *q = nullptr, *K = nullptr, *V = nullptr, *out = nullptr;
  Dev(const std::vector<float>& hq, const std::vector<float>& hK, const std::vector<float>& hV, size_t outn) {
    cudaMalloc(&q, hq.size() * 4);
    cudaMalloc(&K, hK.size() * 4);
    cudaMalloc(&V, hV.size() * 4);
    cudaMalloc(&out, outn * 4);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(K, hK.data(), hK.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(V, hV.data(), hV.size() * 4, cudaMemcpyHostToDevice);
  }
  ~Dev() {
    cudaFree(q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(out);
  }
};

}  // namespace

int main() {
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = 0, cases = 0;
  double worst_causal = 0.0, worst_window = 0.0, worst_window_kernel = 0.0;
  for (int hd : {32, 64, 96, 128, 256}) {
    for (Heads h : {Heads{4, 1}, Heads{4, 2}, Heads{8, 8}}) {
      for (int start : {0, 13, 500}) {
        for (int T : {2, 7, 40, 333, 600}) {
          if (hd == 256 && h.hq == 8 && T == 600 && start == 500) continue;  // the CPU reference's slowest cell
          const int S = start + T + 3;
          const size_t qn = (size_t)T * h.hq * hd, kn = (size_t)S * h.hkv * hd;
          std::vector<float> q(qn), K(kn), V(kn), out(qn);
          for (auto& v : q) v = nd(rng);
          for (auto& v : K) v = nd(rng);
          for (auto& v : V) v = nd(rng);
          const float scale = 1.0f / std::sqrt((float)hd);
          Dev d(q, K, V, qn);
          attention_batch_prefill_cuda(d.q, nullptr, nullptr, d.K, d.V, d.out, T, start, S, h.hq, h.hkv, hd, scale, 0,
                                       nullptr);
          cudaMemcpy(out.data(), d.out, qn * 4, cudaMemcpyDeviceToHost);
          const double ec = reference_err(q, K, V, out, T, start, h, hd, scale, 0);
          worst_causal = std::max(worst_causal, ec);
          ++cases;
          if (!(ec < 1e-5)) {
            ++fails;
            std::printf("FAIL causal hd %d hq %d hkv %d start %d T %d: %.3e\n", hd, h.hq, h.hkv, start, T, ec);
          }
          for (int window : {1, 5, 17, 64, 512, 1024}) {
            if (!attention_window_prefill_cuda(d.q, d.K, d.V, d.out, T, start, h.hq, h.hkv, hd, scale, window, 0)) {
              ++fails;
              std::printf("FAIL window not taken: hd %d hq %d hkv %d window %d T %d\n", hd, h.hq, h.hkv, window, T);
              continue;
            }
            cudaMemcpy(out.data(), d.out, qn * 4, cudaMemcpyDeviceToHost);
            const double ew = reference_err(q, K, V, out, T, start, h, hd, scale, window);
            worst_window = std::max(worst_window, ew);
            ember_ir::ir_window_attention_cuda(d.q, d.K, d.V, d.out, T, start, h.hq, h.hkv, hd, scale, window, 0, 0, nullptr,
                                     0.0f);
            cudaMemcpy(out.data(), d.out, qn * 4, cudaMemcpyDeviceToHost);
            const double ek = reference_err(q, K, V, out, T, start, h, hd, scale, window);
            worst_window_kernel = std::max(worst_window_kernel, ek);
            cases += 2;
            if (!(ew < 1e-5) || !(ek < 1e-5)) {
              ++fails;
              std::printf("FAIL window hd %d hq %d hkv %d start %d T %d window %d: tiled %.3e, window kernel %.3e\n",
                          hd, h.hq, h.hkv, start, T, window, ew, ek);
            }
          }
        }
      }
    }
    std::printf("head_dim %3d done (worst so far: causal %.3e, tiled window %.3e, window kernel %.3e)\n", hd,
                worst_causal, worst_window, worst_window_kernel);
  }
  // timing at Gemma 3 270M's attention shape: 4 query heads over 1 kv head, head_dim 256, 512 tokens
  {
    const int hq = 4, hkv = 1, hd = 256, T = 512, S = 512, window = 512;
    const size_t qn = (size_t)T * hq * hd, kn = (size_t)S * hkv * hd;
    std::vector<float> q(qn), K(kn), V(kn);
    for (auto& v : q) v = nd(rng);
    for (auto& v : K) v = nd(rng);
    for (auto& v : V) v = nd(rng);
    Dev d(q, K, V, qn);
    const float scale = 1.0f / std::sqrt((float)hd);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    auto time = [&](auto launch) {
      std::vector<float> ms;
      for (int r = 0; r < 23; ++r) {
        cudaEventRecord(a);
        launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float x = 0.f;
        cudaEventElapsedTime(&x, a, b);
        if (r >= 3) ms.push_back(x);
      }
      std::sort(ms.begin(), ms.end());
      return ms[ms.size() / 2];
    };
    const float t_window_kernel = time([&] {
      ember_ir::ir_window_attention_cuda(d.q, d.K, d.V, d.out, T, 0, hq, hkv, hd, scale, window, 0, 0, nullptr, 0.0f);
    });
    const float t_tiled_window = time([&] {
      attention_window_prefill_cuda(d.q, d.K, d.V, d.out, T, 0, hq, hkv, hd, scale, window, 0);
    });
    const float t_tiled_causal = time([&] {
      attention_batch_prefill_cuda(d.q, nullptr, nullptr, d.K, d.V, d.out, T, 0, S, hq, hkv, hd, scale, 0, nullptr);
    });
    std::printf("{\"timing\": \"gemma3_270m_attention\", \"hq\": %d, \"hkv\": %d, \"head_dim\": %d, \"tokens\": %d, "
                "\"window\": %d, \"window_kernel_ms\": %.4f, \"tiled_window_ms\": %.4f, \"tiled_causal_ms\": %.4f}\n",
                hq, hkv, hd, T, window, t_window_kernel, t_tiled_window, t_tiled_causal);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf("{\"test\": \"flash_window_attention\", \"pass\": %s, \"cases\": %d, \"failures\": %d, "
              "\"worst_causal\": %.3e, \"worst_tiled_window\": %.3e, \"worst_window_kernel\": %.3e}\n",
              fails ? "false" : "true", cases, fails, worst_causal, worst_window, worst_window_kernel);
  return fails ? 1 : 0;
}
