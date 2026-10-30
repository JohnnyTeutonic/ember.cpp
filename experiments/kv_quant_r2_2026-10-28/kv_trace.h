#pragma once
// Diagnostic-only snapshots. Included exclusively with EMBER_KV_DIAGNOSTICS.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <vector>

#include "kv_quant.h"
namespace ember_kv_diag {
struct Writer {
  std::filesystem::path folder;
  std::ofstream data, index;
  std::set<int> layers;
  explicit Writer(const std::filesystem::path& path) : folder(path) {
    std::filesystem::create_directories(path);
    if (std::filesystem::exists(path / "payload.bin") ||
        std::filesystem::exists(path / "trace.jsonl"))
      throw std::runtime_error("KV trace refuses existing payload/index");
    data.open(path / "payload.bin", std::ios::binary | std::ios::trunc);
    index.open(path / "trace.jsonl", std::ios::trunc);
    if (!data || !index) throw std::runtime_error("KV trace open failed");
  }
};
struct Trace {
  std::shared_ptr<Writer> writer;
  mutable nlohmann::json meta;
  bool active = false;
};
inline void payload(const Trace& trace, const char* name, const void* data,
                    size_t count, size_t element_bytes, const char* dtype) {
  if (!trace.active) return;
  auto& file = trace.writer->data;
  const auto offset = file.tellp();
  if (offset < 0 ||
      !file.write(static_cast<const char*>(data), count * element_bytes))
    throw std::runtime_error("KV trace payload write failed");
  trace.meta["arrays"][name] = {{"offset", static_cast<uint64_t>(offset)},
                                {"bytes", count * element_bytes},
                                {"count", count},
                                {"dtype", dtype}};
}
inline Trace begin(const char* backend, int layer, int pos, int T, int hq,
                   int hkv, int hd, int format, float scale) {
  Trace t;
  const char* directory = std::getenv("EMBER_KV_TRACE_DIR");
  if (!directory || !*directory) return t;
  if (format != ember_kvq::kF32 && format != ember_kvq::kF16)
    throw std::runtime_error("KV diagnostic supports f32/f16 only");
  const std::filesystem::path folder(directory);
  if (!folder.is_absolute())
    throw std::runtime_error("KV trace directory must be absolute");
  static std::shared_ptr<Writer> current;
  if (!current || current->folder != folder)
    current = std::make_shared<Writer>(folder);
  if (!current->layers.insert(layer).second)
    throw std::runtime_error("KV trace refuses duplicate layer");
  t.writer = current;
  t.meta = {{"schema", 1},
            {"backend", backend},
            {"layer", layer},
            {"start_pos", pos},
            {"T", T},
            {"hq", hq},
            {"hkv", hkv},
            {"hd", hd},
            {"format", ember_kvq::format_name(format)},
            {"scale", scale},
            {"complete", false}};
  t.active = true;
  return t;
}
inline void values(const Trace& t, const char* kind, const float* data,
                   size_t count) {
  payload(t, kind, data, count, sizeof(float), "f32");
}
inline void cache_cpu(const Trace& t, const float* k, const float* v) {
  if (!t.active) return;
  const size_t count = size_t(t.meta.at("T").get<int>()) *
                       t.meta.at("hkv").get<int>() * t.meta.at("hd").get<int>();
  values(t, "k_cache", k, count);
  values(t, "v_cache", v, count);
  if (t.meta.at("format") == "f16") {
    std::vector<uint16_t> codes(count);
    for (int which = 0; which < 2; ++which) {
      const float* input = which ? v : k;
      for (size_t i = 0; i < count; ++i)
        codes[i] = ember_kvq::f32_to_f16(input[i]);
      payload(t, which ? "v_codes" : "k_codes", codes.data(), count, 2, "u16");
    }
  }
}
inline void finish(Trace& t, const float* output) {
  if (!t.active) return;
  values(t, "attention", output,
         size_t(t.meta.at("T").get<int>()) * t.meta.at("hq").get<int>() *
             t.meta.at("hd").get<int>());
  t.meta["complete"] = true;
  t.writer->data.flush();
  t.writer->index << t.meta.dump() << "\n";
  t.writer->index.flush();
  if (!t.writer->data || !t.writer->index)
    throw std::runtime_error("KV trace flush failed");
}
#ifdef HAS_CUDA
inline void checked(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("KV diagnostic CUDA: ") +
                             cudaGetErrorString(error));
}
inline std::vector<float> device_values(const float* data, size_t count,
                                        cudaStream_t stream) {
  checked(cudaStreamSynchronize(stream));
  std::vector<float> host(count);
  checked(cudaMemcpy(host.data(), data, count * sizeof(float),
                     cudaMemcpyDeviceToHost));
  return host;
}
inline Trace begin_gpu(int layer, int pos, int T, int hq, int hkv, int hd,
                       int format, float scale, const float* q, const float* k,
                       const float* v, cudaStream_t stream) {
  Trace t = begin("gpu", layer, pos, T, hq, hkv, hd, format, scale);
  if (!t.active) return t;
  const char* graph = std::getenv("EMBER_IR_GRAPH");
  if (!graph || std::string(graph) != "0")
    throw std::runtime_error("KV diagnostic requires EMBER_IR_GRAPH=0");
  auto Q = device_values(q, size_t(T) * hq * hd, stream),
       K = device_values(k, size_t(T) * hkv * hd, stream),
       V = device_values(v, size_t(T) * hkv * hd, stream);
  values(t, "q", Q.data(), Q.size());
  values(t, "k", K.data(), K.size());
  values(t, "v", V.data(), V.size());
  return t;
}
inline void finish_gpu(Trace& t, const void* k, const void* v,
                       const float* output, cudaStream_t stream) {
  if (!t.active) return;
  checked(cudaStreamSynchronize(stream));
  const size_t row =
      size_t(t.meta.at("hkv").get<int>()) * t.meta.at("hd").get<int>();
  const size_t count = size_t(t.meta.at("T").get<int>()) * row,
               offset = size_t(t.meta.at("start_pos").get<int>()) * row;
  if (t.meta.at("format") == "f16") {
    std::vector<uint16_t> codes(count);
    std::vector<float> K(count), V(count);
    for (int which = 0; which < 2; ++which) {
      const auto* input = static_cast<const uint16_t*>(which ? v : k) + offset;
      checked(
          cudaMemcpy(codes.data(), input, count * 2, cudaMemcpyDeviceToHost));
      payload(t, which ? "v_codes" : "k_codes", codes.data(), count, 2, "u16");
      auto& expanded = which ? V : K;
      for (size_t i = 0; i < count; ++i)
        expanded[i] = ember_kvq::f16_to_f32(codes[i]);
    }
    values(t, "k_cache", K.data(), count);
    values(t, "v_cache", V.data(), count);
  } else {
    auto K = device_values(static_cast<const float*>(k) + offset, count,
                           stream),
         V = device_values(static_cast<const float*>(v) + offset, count,
                           stream);
    values(t, "k_cache", K.data(), count);
    values(t, "v_cache", V.data(), count);
  }
  auto O =
      device_values(output,
                    size_t(t.meta.at("T").get<int>()) *
                        t.meta.at("hq").get<int>() * t.meta.at("hd").get<int>(),
                    stream);
  finish(t, O.data());
}
#endif
}  // namespace ember_kv_diag