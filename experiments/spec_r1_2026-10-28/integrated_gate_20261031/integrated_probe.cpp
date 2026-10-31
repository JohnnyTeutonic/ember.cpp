// Diagnostic translation unit. Production source and numerical kernels are
// untouched.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define private public
#define protected public
#include "api.h"
#include "graph_ir.h"
#include "logger.h"
#undef private
#undef protected
#include "sampler_exact.h"
#include "verify_kernels.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
static constexpr int V = 16384;
static_assert(sizeof(ember_verify::RowPick) == 20,
              "RowPick artifact ABI changed");
static void need(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
static void cu(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
static void write_json(const fs::path& path, const json& value) {
  need(!fs::exists(path), "refusing to overwrite existing evidence");
  const auto temporary = path.string() + ".partial";
  std::ofstream stream(temporary);
  stream << value.dump(2) << '\n';
  stream.close();
  need(stream.good(), "JSON output failure");
  fs::rename(temporary, path);
}
static void write_bytes(const fs::path& path, const void* data, size_t bytes) {
  need(!fs::exists(path), "refusing to overwrite existing evidence");
  std::ofstream stream(path, std::ios::binary);
  stream.write(static_cast<const char*>(data), bytes);
  stream.close();
  need(stream.good(), "binary output failure");
}
static std::vector<int> ids_for(int n) {
  std::vector<int> ids(n);
  for (int i = 0; i < n; ++i)
    ids[i] =
        4 + (1103515245u * static_cast<unsigned>(i + 17) + 12345u) % (V - 4);
  return ids;
}
static std::string rng_text(const std::mt19937& rng) {
  std::ostringstream stream;
  stream << rng;
  return stream.str();
}
static json sampler_checks(const std::vector<float>& row,
                           ember_verify::RowPick p, int step) {
  json result = json::array();
  for (double temp : {0.01, 0.05, 0.7, 1.0}) {
    std::mt19937 a(13 + step), b(13 + step);
    const int full = sample_top_k_top_p_temperature(
        row, static_cast<float>(temp), 1, 1.0f, a);
    int compact = greedy_from_pick(p, static_cast<float>(temp), b);
    const bool fallback = compact < 0;
    if (fallback)
      compact = sample_top_k_top_p_temperature(row, static_cast<float>(temp), 1,
                                               1.0f, b);
    result.push_back({{"step", step},
                      {"temperature", temp},
                      {"full_id", full},
                      {"compact_id", compact},
                      {"fallback", fallback},
                      {"full_rng", rng_text(a)},
                      {"compact_rng", rng_text(b)}});
    need(full == compact && rng_text(a) == rng_text(b),
         "sampler ID/RNG mismatch");
  }
  return result;
}
static void prefill(tinyllama::TinyLlamaSession& s, int prefix) {
  s.kv_cache_.clear_data();
  s.kv_cache_.seq_len = 0;
  const auto ids = ids_for(prefix);
  const int hidden = s.config_.hidden_size;
  std::vector<float> embeddings(static_cast<size_t>(prefix) * hidden);
  for (int i = 0; i < prefix; ++i) {
    const auto e = s.model_->lookup_embedding(ids[i]);
    need(e.size() == static_cast<size_t>(hidden), "embedding width mismatch");
    std::copy(e.begin(), e.end(),
              embeddings.begin() + static_cast<size_t>(i) * hidden);
  }
  float* device = nullptr;
  cu(cudaMalloc(&device, embeddings.size() * sizeof(float)));
  cu(cudaMemcpy(device, embeddings.data(), embeddings.size() * sizeof(float),
                cudaMemcpyHostToDevice));
  const auto logits = s.model_->forward_device_batch_prefill(device, prefix, 0,
                                                             &s.kv_cache_, 0);
  cu(cudaFree(device));
  need(logits.size() == V, "prefill did not execute full GPU model");
  s.kv_cache_.seq_len = prefix;
}
static void graph(const fs::path& model, const fs::path& out,
                  const std::string& mode) {
  need(mode == "rows" || mode == "original" || mode == "partition4",
       "unknown graph mode");
  setenv("EMBER_DETERMINISTIC", "1", 1);
  setenv("EMBER_DECODE_GRAPH", "1", 1);
  setenv("EMBER_SPEC", "0", 1);
  setenv("EMBER_GEMM_TF32", "0", 1);
  setenv("EMBER_IR", "0", 1);
  setenv("EMBER_DECODE_PICK_PARTS", mode == "partition4" ? "4" : "0", 1);
  tinyllama::TinyLlamaSession s(model.string(), "tokenizer.json", 1, -1, true);
  need(!s.ir_ && s.config_.num_cpu_offload_layers == 0,
       "CPU/IR fallback refused");
  need(s.config_.vocab_size == V, "unexpected model vocabulary");
  json records = json::array();
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int prefix : {1, 7, 16, 31, 64}) {
      prefill(s, prefix);
      const auto teacher = ids_for(prefix + 8);
      std::vector<float> rows;
      json samples = json::array();
      const void* scratch = nullptr;
      bool stable = true;
      for (int j = 0; j < 8; ++j) {
        ember_verify::RowPick pick{};
        pick.first = -1;
        const bool wants_pick = mode != "rows";
        auto row = s.model_->forward_device_graph(teacher[prefix + j],
                                                  prefix + j, &s.kv_cache_,
                                                  wants_pick ? &pick : nullptr);
        if (wants_pick) {
          need(pick.first >= 0, "compact graph unavailable");
          row = s.model_->decode_row_logits();  // copy current logits only, no
                                                // second forward
        }
        need(row.size() == V, "graph unavailable or incomplete row");
        need(s.model_->decode_graph_exec_ != nullptr,
             "graph execution not captured");
        if (mode == "partition4") {
          const void* current = s.model_->decode_pick_parts_dev_;
          need(current != nullptr, "partition scratch was not allocated");
          if (!scratch) scratch = current;
          stable = stable && current == scratch;
        }
        if (!wants_pick) {
          // Exercise the original production selector on the SAME current
          // logits.
          ember_verify::RowPick* d = nullptr;
          cu(cudaMalloc(&d, sizeof(pick)));
          ember_verify::select_rows(s.model_->logits_dev_, 1, V, d, 0);
          cu(cudaMemcpy(&pick, d, sizeof(pick), cudaMemcpyDeviceToHost));
          cu(cudaFree(d));
        }
        for (const auto& entry : sampler_checks(row, pick, j))
          samples.push_back(entry);
        rows.insert(rows.end(), row.begin(), row.end());
        s.kv_cache_.seq_len = prefix + j + 1;
      }
      const bool captured_has_pick = s.model_->decode_graph_has_pick_;
      const int captured_parts = s.model_->decode_graph_pick_parts_;
      json transitions = json::array();
      if (mode == "partition4") {
        // Same KV identity, same teacher-forced position: no-pick -> pick ->
        // no-pick recaptures, overwriting the same current KV slot only.
        std::vector<float> transition_rows;
        for (bool wants_pick : {false, true, false}) {
          ember_verify::RowPick pick{};
          pick.first = -1;
          auto row = s.model_->forward_device_graph(
              teacher.back(), prefix + 8, &s.kv_cache_,
              wants_pick ? &pick : nullptr);
          if (wants_pick) {
            need(pick.first >= 0, "transition pick graph unavailable");
            row = s.model_->decode_row_logits();
          }
          need(row.size() == V && s.model_->decode_graph_exec_,
               "transition graph failed");
          need(s.model_->decode_pick_parts_dev_ == scratch,
               "scratch changed across recapture");
          transitions.push_back(
              {{"has_pick", s.model_->decode_graph_has_pick_},
               {"parts", s.model_->decode_graph_pick_parts_}});
          transition_rows.insert(transition_rows.end(), row.begin(), row.end());
        }
        const auto transition_name = mode + "_" + std::to_string(repeat) + "_" +
                                     std::to_string(prefix) +
                                     ".transitions.bin";
        write_bytes(out / transition_name, transition_rows.data(),
                    transition_rows.size() * sizeof(float));
        need(std::memcmp(transition_rows.data(), transition_rows.data() + V,
                         V * sizeof(float)) == 0 &&
                 std::memcmp(transition_rows.data(),
                             transition_rows.data() + 2 * V,
                             V * sizeof(float)) == 0,
             "pick-presence recapture changed row bits");
      }
      const auto name = mode + "_" + std::to_string(repeat) + "_" +
                        std::to_string(prefix) + ".rows.bin";
      write_bytes(out / name, rows.data(), rows.size() * sizeof(float));
      records.push_back({{"mode", mode},
                         {"repeat", repeat},
                         {"prefix", prefix},
                         {"teacher_ids", teacher},
                         {"successful_graph_calls", 8},
                         {"graph_has_pick", captured_has_pick},
                         {"graph_pick_parts", captured_parts},
                         {"scratch_nonnull", scratch != nullptr},
                         {"scratch_stable", stable},
                         {"row_file", name},
                         {"sampler", samples},
                         {"transitions", transitions}});
      write_json(out / (mode + "_" + std::to_string(repeat) + "_" +
                        std::to_string(prefix) + ".json"),
                 records.back());
    }
  }
  write_json(out / (mode + ".json"),
             {{"graphs", records}, {"all_layers_gpu", true}, {"ir", false}});
}
static void kernels(const fs::path& prepared, const fs::path& out) {
  std::ifstream manifest_stream(prepared / "prepared.json");
  json manifest;
  manifest_stream >> manifest;
  for (const auto& rec : manifest.at("kernels")) {
    const int n = rec.at("rows"), v = rec.at("vocab");
    std::ifstream input(prepared / rec.at("input").get<std::string>(),
                        std::ios::binary);
    std::vector<float> rows(static_cast<size_t>(n) * v);
    input.read(reinterpret_cast<char*>(rows.data()),
               rows.size() * sizeof(float));
    need(input.gcount() ==
             static_cast<std::streamsize>(rows.size() * sizeof(float)),
         "kernel input shortened");
    float* device_rows = nullptr;
    ember_verify::RowPick *original = nullptr, *partitioned = nullptr,
                          *scratch = nullptr;
    cu(cudaMalloc(&device_rows, rows.size() * sizeof(float)));
    cu(cudaMalloc(&original, n * sizeof(*original)));
    cu(cudaMalloc(&partitioned, n * sizeof(*partitioned)));
    cu(cudaMalloc(&scratch, 4 * n * sizeof(*scratch)));
    cu(cudaMemcpy(device_rows, rows.data(), rows.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    ember_verify::select_rows(device_rows, n, v, original, 0);
    ember_verify::select_rows_partitioned(device_rows, n, v, partitioned,
                                          scratch, 0);
    std::vector<ember_verify::RowPick> a(n), b(n);
    cu(cudaMemcpy(a.data(), original, n * sizeof(*original),
                  cudaMemcpyDeviceToHost));
    cu(cudaMemcpy(b.data(), partitioned, n * sizeof(*partitioned),
                  cudaMemcpyDeviceToHost));
    write_bytes(out / rec.at("original").get<std::string>(), a.data(),
                n * sizeof(*original));
    write_bytes(out / rec.at("partition4").get<std::string>(), b.data(),
                n * sizeof(*partitioned));
    cu(cudaFree(scratch));
    cu(cudaFree(partitioned));
    cu(cudaFree(original));
    cu(cudaFree(device_rows));
    need(std::memcmp(a.data(), b.data(), n * sizeof(a[0])) == 0,
         "original/partitioned field bits differ");
  }
  write_json(out / "kernels.json", manifest.at("kernels"));
}
int main(int argc, char** argv) {
  try {
    need(argc == 5,
         "usage: integrated_pick_probe MODEL PREPARED OUT "
         "rows|original|partition4|kernels");
    const fs::path out(argv[3]);
    need(!fs::exists(out) || fs::is_empty(out),
         "existing output evidence must be preserved");
    fs::create_directories(out);
    int count = 0;
    cu(cudaGetDeviceCount(&count));
    need(count > 0, "actual CUDA device required");
    cudaDeviceProp property{};
    cu(cudaGetDeviceProperties(&property, 0));
    need(std::string(property.name).find("L4") != std::string::npos,
         "actual L4 required");
    write_json(out / "device.json", {{"name", property.name},
                                     {"cuda_device_count", count},
                                     {"major", property.major},
                                     {"minor", property.minor}});
    if (std::string(argv[4]) == "kernels")
      kernels(argv[2], out);
    else
      graph(argv[1], out, argv[4]);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "INTEGRATED_GATE_ERROR: " << e.what() << '\n';
    return 1;
  }
}
