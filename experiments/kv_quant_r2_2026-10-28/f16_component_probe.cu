// Diagnostic: identical host inputs isolate storage/read/attention from model
// arithmetic.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "cuda_kernels.h"
#include "ir_digest.h"
#include "kv_quant.h"
using json = nlohmann::json;
void checked(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* p = nullptr;
  explicit Device(size_t n) {
    checked(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
  }
  ~Device() { cudaFree(p); }
  void put(const std::vector<T>& v) {
    checked(
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get(size_t n) const {
    std::vector<T> v(n);
    checked(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
__global__ void conversion(const float* in, uint16_t* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __half_as_ushort(__float2half_rn(in[i]));
}
size_t bit_differences(const std::vector<float>& a,
                       const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("size mismatch");
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i)
    n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}
json metric(const std::vector<float>& a, const std::vector<double>& b,
            int width) {
  if (a.size() != b.size() || a.empty() || a.size() % width)
    throw std::runtime_error("invalid metric shape");
  double worst = 0, absolute = 0;
  for (size_t base = 0; base < a.size(); base += width) {
    double lo = b[base], hi = lo, delta = 0;
    for (int j = 0; j < width; ++j) {
      if (!std::isfinite(a[base + j]) || !std::isfinite(b[base + j]))
        throw std::runtime_error("nonfinite attention");
      lo = std::min(lo, b[base + j]);
      hi = std::max(hi, b[base + j]);
      delta = std::max(delta, std::abs(a[base + j] - b[base + j]));
    }
    absolute = std::max(absolute, delta);
    worst = std::max(worst, hi > lo ? delta / (hi - lo) : delta);
  }
  return {{"worst_absolute", absolute},
          {"worst_row_range_relative", worst},
          {"descriptive_only", true}};
}
std::vector<double> reference(const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v, int T, int L, int hq,
                              int hkv, int hd, float scale) {
  std::vector<double> out(size_t(T) * hq * hd), scores(L);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < hq; ++h) {
      int count = T == 1 ? L : t + 1;
      double maximum = -INFINITY;
      for (int j = 0; j < count; ++j) {
        double dot = 0;
        for (int d = 0; d < hd; ++d)
          dot += double(q[(size_t(t) * hq + h) * hd + d]) *
                 k[(size_t(j) * hkv + h / (hq / hkv)) * hd + d];
        scores[j] = dot * scale;
        maximum = std::max(maximum, scores[j]);
      }
      double sum = 0;
      for (int j = 0; j < count; ++j)
        sum += (scores[j] = std::exp(scores[j] - maximum));
      for (int d = 0; d < hd; ++d)
        for (int j = 0; j < count; ++j)
          out[(size_t(t) * hq + h) * hd + d] +=
              scores[j] / sum * v[(size_t(j) * hkv + h / (hq / hkv)) * hd + d];
    }
  return out;
}
int main() {
  try {
    int devices = 0;
    checked(cudaGetDeviceCount(&devices));
    if (devices < 1) throw std::runtime_error("actual CUDA required");
    cudaDeviceProp prop{};
    checked(cudaGetDeviceProperties(&prop, 0));
    std::mt19937 rng(0xE9F16);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<float> values;
    // Every representable half plus both neighbours of each finite midpoint.
    for (unsigned code = 0; code < 65536; ++code) {
      float x = ember_kvq::f16_to_f32(uint16_t(code));
      values.push_back(x);
      if (code < 0x7bff) {
        float y = ember_kvq::f16_to_f32(uint16_t(code + 1));
        float mid = (x + y) * 0.5f;
        for (float z : {std::nextafter(mid, -INFINITY), mid,
                        std::nextafter(mid, INFINITY)}) {
          values.push_back(z);
          values.push_back(-z);
        }
      }
    }
    for (int i = 0; i < 1000000; ++i) {
      uint32_t bits = rng();
      float x;
      std::memcpy(&x, &bits, 4);
      values.push_back(x);
    }
    Device<float> input(values.size());
    Device<uint16_t> output(values.size());
    input.put(values);
    conversion<<<(values.size() + 255) / 256, 256>>>(input.p, output.p,
                                                     values.size());
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    auto codes = output.get(values.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      auto cpu = ember_kvq::f32_to_f16(values[i]);
      bool nan = std::isnan(values[i]);
      if (nan)
        mismatches += !((codes[i] & 0x7c00) == 0x7c00 && (codes[i] & 0x3ff));
      else
        mismatches += cpu != codes[i];
    }
    json report = {{"diagnostic_only", true},
                   {"gpu", prop.name},
                   {"conversion_values", values.size()},
                   {"conversion_mismatches", mismatches},
                   {"cases", json::array()}};
    for (auto shape : std::vector<std::vector<int>>{{16, 8, 128}, {14, 2, 64}})
      for (int L : {1, 2, 24, 32, 33, 55, 127, 128, 129, 2048, 2049, 4096}) {
        const auto case_rng = rng;
        for (int capacity : {L, 8448}) {
          rng = case_rng;
          int hq = shape[0], hkv = shape[1], hd = shape[2], row = hkv * hd;
          float scale = 1 / std::sqrt(float(hd));
          std::vector<float> k(size_t(L) * row), v(k.size()),
              q(size_t(hq) * hd);
          for (auto* vector : {&k, &v, &q})
            for (float& x : *vector) x = uniform(rng);
          Device<float> dk(k.size()), dv(v.size()), dq(q.size()), out(q.size()),
              deqk(size_t(capacity) * row), deqv(size_t(capacity) * row);
          std::string input_bytes;
          for (const auto* x : {&k, &v, &q})
            input_bytes.append(reinterpret_cast<const char*>(x->data()),
                               x->size() * sizeof(float));
          const auto input_sha = ember_ir::sha256_hex_bytes(input_bytes);
          dk.put(k);
          dv.put(v);
          dq.put(q);
          Device<uint16_t> K(size_t(capacity) * row), V(size_t(capacity) * row),
              Ks(size_t(capacity) * row), Vs(size_t(capacity) * row);
          Device<float> partial(ember_kvq::partial_floats(hq, hd, capacity));
          Device<int> position(1);
          ember_kvq::write_cuda(ember_kvq::kF16, K.p, nullptr, V.p, nullptr,
                                dk.p, dv.p, L, row, 0, nullptr);
          for (int t = 0; t < L; ++t) {
            position.put(std::vector<int>{t});
            ember_kvq::write_cuda(ember_kvq::kF16, Ks.p, nullptr, Vs.p, nullptr,
                                  dk.p + size_t(t) * row,
                                  dv.p + size_t(t) * row, 1, row, 0, nullptr,
                                  position.p);
          }
          checked(cudaDeviceSynchronize());
          auto kc = K.get(k.size()), vc = V.get(v.size());
          auto sk = Ks.get(k.size()), sv = Vs.get(v.size());
          size_t store = 0, split = 0;
          for (size_t i = 0; i < k.size(); ++i) {
            store += kc[i] != ember_kvq::f32_to_f16(k[i]);
            store += vc[i] != ember_kvq::f32_to_f16(v[i]);
            split += kc[i] != sk[i];
            split += vc[i] != sv[i];
            k[i] = ember_kvq::f16_to_f32(kc[i]);
            v[i] = ember_kvq::f16_to_f32(vc[i]);
          }
          ember_kvq::dequant_rows_cuda(ember_kvq::kF16, K.p, nullptr, V.p,
                                       nullptr, deqk.p, deqv.p, L, row,
                                       nullptr);
          ember_kvq::decode_cuda(ember_kvq::kF16, dq.p, K.p, nullptr, V.p,
                                 nullptr, out.p, hq, hkv, hd, scale, capacity,
                                 L, partial.p, nullptr);
          checked(cudaDeviceSynchronize());
          auto decoded = out.get(q.size());
          auto ref = reference(q, k, v, 1, L, hq, hkv, hd, scale);
          json item = {{"hq", hq},
                       {"hkv", hkv},
                       {"hd", hd},
                       {"length", L},
                       {"capacity", capacity},
                       {"input_sha256", input_sha},
                       {"store_mismatches", store},
                       {"bulk_vs_pointer_writes", split},
                       {"readback_mismatches",
                        bit_differences(k, deqk.get(k.size())) +
                            bit_differences(v, deqv.get(v.size()))},
                       {"decode_vs_f64", metric(decoded, ref, hq * hd)}};
          position.put(std::vector<int>{L - 1});
          ember_kvq::decode_cuda(ember_kvq::kF16, dq.p, K.p, nullptr, V.p,
                                 nullptr, out.p, hq, hkv, hd, scale, capacity,
                                 1, partial.p, nullptr, position.p);
          checked(cudaDeviceSynchronize());
          item["position_pointer_decode_bit_differences"] =
              bit_differences(decoded, out.get(q.size()));
          if (L <= 129) {
            std::vector<float> queries(size_t(L) * hq * hd);
            for (float& x : queries) x = uniform(rng);
            Device<float> queries_d(queries.size()), prefill(queries.size());
            queries_d.put(queries);
            attention_batch_prefill_cuda(queries_d.p, nullptr, nullptr, deqk.p,
                                         deqv.p, prefill.p, L, 0, capacity, hq,
                                         hkv, hd, scale, nullptr);
            checked(cudaDeviceSynchronize());
            item["prefill_vs_f64"] = metric(
                prefill.get(queries.size()),
                reference(queries, k, v, L, L, hq, hkv, hd, scale), hq * hd);
          }
          mismatches +=
              store + split + item["readback_mismatches"].get<size_t>() +
              item["position_pointer_decode_bit_differences"].get<size_t>();
          report["cases"].push_back(item);
        }
      }
    report["exact_component_mismatches"] = mismatches;
    report["complete"] = true;
    report["e9_closed"] = false;
    std::cout << report.dump(1) << "\n";
    return mismatches ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}