// occ_step: the engine's decode step against the cached context length, in situ (ember.cpp,
// experiments/decode_occupancy_2026-10-27).
//
//   occ_step MODEL.gguf OUT.json [REPS] [PREFIXES]     (PREFIXES: comma-separated, default 32,128,384,500)
//
// The session is built as tinyllama_server builds it (every layer on the GPU, the tuned llama path). For each P the
// cache is filled by one batched prefill of fixed pseudo-random ids, then the decode step (forward_device_graph: the
// captured graph's replay, logits to pinned memory, sync) runs REPS times at position P (the attention reads P + 1
// keys; the KV row P is rewritten each time). Host wall clock per call, p50/p10/p90 after 5 warm-ups; an NVTX range
// "step|P" around the timed calls lets nsys give each kernel's time per range. The decode attention kernel is
// whichever EMBER_DECODE_ATTN selects in this process.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "cuda_kernels.h"
#include "graph_ir.h"
#include "logger.h"
#define private public
#define protected public
#include "api.h"
#undef private
#undef protected

using json = nlohmann::json;
using clk = std::chrono::steady_clock;

namespace {

tinyllama::TinyLlamaSession* S = nullptr;
int V = 0, D = 0;

double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

json summary(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double f) { return v[std::min(v.size() - 1, static_cast<size_t>(f * (v.size() - 1) + 0.5))]; };
  return {{"p50", q(0.5)}, {"p10", q(0.1)}, {"p90", q(0.9)}, {"n", v.size()}};
}

std::vector<int> ids_for(int n, int salt) {
  std::vector<int> ids(n);
  for (int i = 0; i < n; ++i) ids[i] = 4 + static_cast<int>((1103515245u * (unsigned)(i + 17 * salt) + 12345u) % (unsigned)(V - 4));
  return ids;
}

void prefill(int P) {
  S->kv_cache_.clear_data();
  S->kv_cache_.seq_len = 0;
  const std::vector<int> ids = ids_for(P, 1);
  std::vector<float> emb(static_cast<size_t>(P) * D);
  for (int i = 0; i < P; ++i) {
    const std::vector<float> e = S->model_->lookup_embedding(ids[i]);
    std::copy(e.begin(), e.end(), emb.begin() + static_cast<size_t>(i) * D);
  }
  float* d = nullptr;
  cudaMalloc(&d, emb.size() * sizeof(float));
  cudaMemcpy(d, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice);
  const std::vector<float> lg = S->model_->forward_device_batch_prefill(d, P, 0, &S->kv_cache_, 0);
  cudaFree(d);
  if (lg.size() != static_cast<size_t>(V)) throw std::runtime_error("prefill failed");
  S->kv_cache_.seq_len = P;
}

std::vector<float> decode(int id, int pos) {
  std::vector<float> lg = S->model_->forward_device_graph(id, pos, &S->kv_cache_);
  if (lg.size() != static_cast<size_t>(V)) throw std::runtime_error("decode graph unavailable");
  return lg;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: occ_step MODEL.gguf OUT.json [REPS] [PREFIXES]\n";
    return 2;
  }
  const int reps = argc > 3 ? std::stoi(argv[3]) : 300;
  std::vector<int> prefixes{32, 128, 384, 500};
  if (argc > 4) {
    prefixes.clear();
    std::stringstream ss(argv[4]);
    for (std::string x; std::getline(ss, x, ',');) prefixes.push_back(std::stoi(x));
  }
  try {
    tinyllama::TinyLlamaSession s(argv[1], "tokenizer.json", 4, -1, true);
    S = &s;
    const ModelConfig& c = s.config_;
    if (s.ir_ || c.num_cpu_offload_layers != 0) throw std::runtime_error("needs the tuned llama path, every layer on the GPU");
    V = c.vocab_size;
    D = c.hidden_size;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int mn = 0, mx = 0;
    decode_attention_tiles(&mn, &mx);
    json out{{"model", argv[1]}, {"gpu", prop.name}, {"sm", prop.major * 10 + prop.minor},
             {"deterministic", std::getenv("EMBER_DETERMINISTIC") ? std::getenv("EMBER_DETERMINISTIC") : ""},
             {"decode_attention_kind", decode_attention_kind()}, {"tiles", {mn, mx}},
             {"hidden", D}, {"layers", c.num_hidden_layers}, {"heads", c.num_attention_heads},
             {"kv_heads", c.num_key_value_heads}, {"vocab", V}, {"max_seq", s.kv_cache_.allocated_max_seq_len},
             {"reps", reps}};
    json per = json::array();
    for (int P : prefixes) {
      if (P + 1 >= s.kv_cache_.allocated_max_seq_len) continue;
      prefill(P);
      const int tok = ids_for(1, 9)[0];
      std::vector<float> last;
      for (int i = 0; i < 5; ++i) last = decode(tok, P);
      std::vector<double> t;
      t.reserve(reps);
      const std::string name = "step|" + std::to_string(P);
      nvtxRangePushA(name.c_str());
      for (int i = 0; i < reps; ++i) {
        const auto t0 = clk::now();
        last = decode(tok, P);
        t.push_back(ms_since(t0));
      }
      nvtxRangePop();
      int am = 0;
      for (int i = 1; i < V; ++i)
        if (last[i] > last[am]) am = i;
      std::vector<float> head(last.begin(), last.begin() + 8);
      per.push_back({{"prefix", P}, {"keys", P + 1}, {"decode_ms", summary(t)}, {"argmax", am}, {"logits_head", head}});
      std::cout << "P " << P << " decode p50 " << summary(t)["p50"] << " ms, argmax " << am << std::endl;
    }
    out["prefixes"] = per;
    std::ofstream(argv[2]) << out.dump(1) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "occ_step: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
