// The tiled prefill attention kernel (attention_prefill_flash_kernel, cuda_kernels.cu) at the shapes the
// Gemma 3 change adds: head_dim 256 (16 keys per tile) and a sliding window. Two entry points against a
// double-precision CPU reference of grouped-query attention over the KV cache:
//   causal:  attention_batch_prefill_cuda (default kernel selection, flash2 then flash v1 then baseline),
//            which at head_dim 256 now takes the tiled kernel instead of the baseline;
//   window:  attention_window_prefill_cuda, query position p seeing keys [p - window + 1, p], and the IR's
//            window_attention_kernel (ir_window_attention_cuda, sinks 0), the kernel it replaces for
//            multi-token prefill.
// Head dims 32, 64, 96, 128, 256; heads (4 q, 1 kv), (4, 2) and (8, 8); batches of 2, 7, 40, 333 and 600
// tokens at cache offsets 0, 13 and 500; windows 1, 5, 17, 64, 512 and 1024. Each case's reference is
// computed once and every kernel's output is scored against it. Pass: every output finite and within
// max |out - ref| / max(1, max |ref|) < 1e-5 (a non-finite output or reference scores infinity), and
// attention_window_prefill_cuda takes every window shape. A host self-check first confirms that a NaN
// output fails the score. Then Gemma 3 270M's attention shape (4 q heads, 1 kv head, head_dim 256, 512
// tokens, window 512, where the window covers every key): the three kernels' outputs are scored against
// the reference, then timed (median of 20 each). Every CUDA call is checked. Exit 1 on any failure.
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/flash_window_attention_test.cu cuda_kernels.cu graph_ir_cuda.cu
//        logger.cpp -lcublas -o flash_window_attention_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.h"
#include "graph_ir_cuda.h"

#define CK(call)                                                                                   \
  do {                                                                                             \
    const cudaError_t e_ = (call);                                                                 \
    if (e_ != cudaSuccess) {                                                                       \
      std::printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #call); \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

struct Heads {
  int hq, hkv;
};

// The reference output [T, hq, hd] in double. window 0: causal; else keys [p - window + 1, p].
std::vector<double> reference(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V,
                              int T, int start, Heads h, int hd, float scale, int window) {
  std::vector<double> ref((size_t)T * h.hq * hd);
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
      double* r = ref.data() + ((size_t)t * h.hq + qh) * hd;
      for (int d = 0; d < hd; ++d) {
        double a = 0.0;
        for (int j = lo; j <= p; ++j) a += s[j - lo] / sum * V[((size_t)j * h.hkv + kvh) * hd + d];
        r[d] = a;
      }
    }
  return ref;
}

// max |out - ref| / max(1, max |ref|); infinity if any value of either is not finite, or the sizes differ
double score(const std::vector<float>& out, const std::vector<double>& ref) {
  if (out.size() != ref.size()) return std::numeric_limits<double>::infinity();
  double err = 0.0, mref = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(out[i]) || !std::isfinite(ref[i])) return std::numeric_limits<double>::infinity();
    mref = std::max(mref, std::fabs(ref[i]));
    err = std::max(err, std::fabs((double)out[i] - ref[i]));
  }
  return err / std::max(1.0, mref);
}

struct Dev {
  float *q = nullptr, *K = nullptr, *V = nullptr, *out = nullptr;
  size_t outn;
  Dev(const std::vector<float>& hq, const std::vector<float>& hK, const std::vector<float>& hV, size_t n) : outn(n) {
    CK(cudaMalloc(&q, hq.size() * 4));
    CK(cudaMalloc(&K, hK.size() * 4));
    CK(cudaMalloc(&V, hV.size() * 4));
    CK(cudaMalloc(&out, outn * 4));
    CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(K, hK.data(), hK.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(V, hV.data(), hV.size() * 4, cudaMemcpyHostToDevice));
  }
  // the output, after a launch: the launch's own error, then the stream, then the copy
  std::vector<float> fetch() {
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> h(outn);
    CK(cudaMemcpy(h.data(), out, outn * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(out, 0xff, outn * 4));  // NaN fill, so a later launch that writes nothing cannot pass
    return h;
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
  // self-check: the score rejects a NaN output (one element, and all of them) and a matching output passes
  {
    const std::vector<double> ref = {0.5, -0.25, 1.0};
    std::vector<float> ok = {0.5f, -0.25f, 1.0f}, one = ok, all(3, std::nanf(""));
    one[1] = std::nanf("");
    if (!(score(ok, ref) < 1e-5) || score(one, ref) < 1e-5 || score(all, ref) < 1e-5 ||
        score(std::vector<float>(2, 0.f), ref) < 1e-5) {
      std::printf("{\"test\": \"flash_window_attention\", \"pass\": false, \"reason\": \"score self-check\"}\n");
      return 1;
    }
  }
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = 0, cases = 0;
  double worst_causal = 0.0, worst_window = 0.0, worst_window_kernel = 0.0;
  for (int hd : {32, 64, 96, 128, 256}) {
    for (Heads h : {Heads{4, 1}, Heads{4, 2}, Heads{8, 8}}) {
      for (int start : {0, 13, 500}) {
        for (int T : {2, 7, 40, 333, 600}) {
          const int S = start + T + 3;
          const size_t qn = (size_t)T * h.hq * hd, kn = (size_t)S * h.hkv * hd;
          std::vector<float> q(qn), K(kn), V(kn);
          for (auto& v : q) v = nd(rng);
          for (auto& v : K) v = nd(rng);
          for (auto& v : V) v = nd(rng);
          const float scale = 1.0f / std::sqrt((float)hd);
          Dev d(q, K, V, qn);
          CK(cudaMemset(d.out, 0xff, qn * 4));
          attention_batch_prefill_cuda(d.q, nullptr, nullptr, d.K, d.V, d.out, T, start, S, h.hq, h.hkv, hd, scale, 0,
                                       nullptr);
          const double ec = score(d.fetch(), reference(q, K, V, T, start, h, hd, scale, 0));
          worst_causal = std::max(worst_causal, ec);
          ++cases;
          if (!(ec < 1e-5)) {
            ++fails;
            std::printf("FAIL causal hd %d hq %d hkv %d start %d T %d: %.3e\n", hd, h.hq, h.hkv, start, T, ec);
          }
          for (int window : {1, 5, 17, 64, 512, 1024}) {
            const std::vector<double> ref = reference(q, K, V, T, start, h, hd, scale, window);
            if (!attention_window_prefill_cuda(d.q, d.K, d.V, d.out, T, start, h.hq, h.hkv, hd, scale, window, 0)) {
              ++fails;
              std::printf("FAIL window not taken: hd %d hq %d hkv %d window %d T %d\n", hd, h.hq, h.hkv, window, T);
              continue;
            }
            const double ew = score(d.fetch(), ref);
            ember_ir::ir_window_attention_cuda(d.q, d.K, d.V, d.out, T, start, h.hq, h.hkv, hd, scale, window, 0, 0,
                                               nullptr, 0.0f);
            const double ek = score(d.fetch(), ref);
            worst_window = std::max(worst_window, ew);
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
  // Gemma 3 270M's attention shape: scored first, then timed
  {
    const Heads h{4, 1};
    const int hd = 256, T = 512, S = 512, window = 512;
    const size_t qn = (size_t)T * h.hq * hd, kn = (size_t)S * h.hkv * hd;
    std::vector<float> q(qn), K(kn), V(kn);
    for (auto& v : q) v = nd(rng);
    for (auto& v : K) v = nd(rng);
    for (auto& v : V) v = nd(rng);
    Dev d(q, K, V, qn);
    CK(cudaMemset(d.out, 0xff, qn * 4));
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<double> ref = reference(q, K, V, T, 0, h, hd, scale, window);  // = causal: window covers all
    auto window_kernel = [&] {
      ember_ir::ir_window_attention_cuda(d.q, d.K, d.V, d.out, T, 0, h.hq, h.hkv, hd, scale, window, 0, 0, nullptr,
                                         0.0f);
    };
    auto tiled_window = [&] {
      if (!attention_window_prefill_cuda(d.q, d.K, d.V, d.out, T, 0, h.hq, h.hkv, hd, scale, window, 0)) {
        std::printf("FAIL: the Gemma 3 shape is not taken by attention_window_prefill_cuda\n");
        std::exit(1);
      }
    };
    auto tiled_causal = [&] {
      attention_batch_prefill_cuda(d.q, nullptr, nullptr, d.K, d.V, d.out, T, 0, S, h.hq, h.hkv, hd, scale, 0,
                                   nullptr);
    };
    double e512[3];
    window_kernel();
    e512[0] = score(d.fetch(), ref);
    tiled_window();
    e512[1] = score(d.fetch(), ref);
    tiled_causal();
    e512[2] = score(d.fetch(), ref);
    cases += 3;
    for (double e : e512)
      if (!(e < 1e-5)) ++fails;
    worst_window_kernel = std::max(worst_window_kernel, e512[0]);
    worst_window = std::max(worst_window, e512[1]);
    worst_causal = std::max(worst_causal, e512[2]);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    auto time = [&](auto launch) {
      std::vector<float> ms;
      for (int r = 0; r < 23; ++r) {
        CK(cudaEventRecord(a));
        launch();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaGetLastError());
        float x = 0.f;
        CK(cudaEventElapsedTime(&x, a, b));
        if (r >= 3) ms.push_back(x);
      }
      std::sort(ms.begin(), ms.end());
      return ms[ms.size() / 2];
    };
    const float t_window_kernel = time(window_kernel);
    const float t_tiled_window = time(tiled_window);
    const float t_tiled_causal = time(tiled_causal);
    std::printf("{\"timing\": \"gemma3_270m_attention\", \"hq\": %d, \"hkv\": %d, \"head_dim\": %d, \"tokens\": %d, "
                "\"window\": %d, \"err_window_kernel\": %.3e, \"err_tiled_window\": %.3e, \"err_tiled_causal\": %.3e, "
                "\"window_kernel_ms\": %.4f, \"tiled_window_ms\": %.4f, \"tiled_causal_ms\": %.4f}\n",
                h.hq, h.hkv, hd, T, window, e512[0], e512[1], e512[2], t_window_kernel, t_tiled_window,
                t_tiled_causal);
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
  }
  std::printf("{\"test\": \"flash_window_attention\", \"pass\": %s, \"cases\": %d, \"failures\": %d, "
              "\"worst_causal\": %.3e, \"worst_tiled_window\": %.3e, \"worst_window_kernel\": %.3e}\n",
              fails ? "false" : "true", cases, fails, worst_causal, worst_window, worst_window_kernel);
  return fails ? 1 : 0;
}
