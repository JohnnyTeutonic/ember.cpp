// tp_session_logits: the path tinyllama_server serves a GGUF on, and that path's logits on raw token ids.
//
//   tp_session_logits MODEL.gguf GPU_LAYERS < {"seqs": [[id, ...], ...], "prompt_len": [P, ...]} > out.json
//
// The session is built as server.cpp builds it (TinyLlamaSession(model, "tokenizer.json", 4, GPU_LAYERS, true)),
// so the tuned-path choice (graph_ir.h tuned_llama_match), the load-time dequantisation of small files and the
// interpreter's device choice are the server's own. Then, per sequence, api.cpp's forwards, each from position 0
// with the cache cleared as generate() clears it:
//   prefill  the whole sequence in one call, every row's logits: the interpreter's forward(ids, 0, all rows); the
//            tuned path's forward_cpu_batch + forward_cpu_logits_batch (every layer on the CPU) or
//            forward_device_batch_prefill(all_logits) (every layer on the GPU);
//   decode   one token at a time as generate()'s loop runs it: the interpreter's forward({id}, pos); the tuned
//            path's forward (CPU) or forward_device_graph, falling back to forward_device (GPU); kv seq_len = pos + 1;
//   free     free-running greedy: the prompt's prefill (its last row), then len - P argmax tokens decoded.
// stdout: {"path": "tuned" | "interpreter", "device": "cpu" | "cuda", ..., "seqs": [{"prefill", "decode", "free"}]}.
// The session's private members are reached by defining private as public around api.h (this test file only; the
// library is unchanged).
// Every system header first, so the access redefinition below reaches ember's headers only.
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
#ifdef HAS_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "graph_ir.h"
#include "logger.h"
#define private public
#define protected public
#include "api.h"
#undef private
#undef protected

using json = nlohmann::json;

namespace {

tinyllama::TinyLlamaSession* S = nullptr;
int V = 0, D = 0;
bool all_cpu = true;

void fresh() {
  S->kv_cache_.clear_data();
  S->kv_cache_.seq_len = 0;
}

// One call from position 0; every row's logits.
std::vector<float> prefill(const std::vector<int>& ids) {
  fresh();
  const int n = static_cast<int>(ids.size());
  std::vector<float> all;
  if (S->ir_) {
    all = S->ir_->forward(ids, 0, true);
  } else {
    std::vector<float> emb(static_cast<size_t>(n) * D);
    for (int i = 0; i < n; ++i) {
      const std::vector<float> e = S->model_->lookup_embedding(ids[i]);
      std::copy(e.begin(), e.end(), emb.begin() + static_cast<size_t>(i) * D);
    }
    if (all_cpu) {
      all = S->model_->forward_cpu_logits_batch(
          S->model_->forward_cpu_batch(emb, n, S->config_.num_cpu_offload_layers, 0, &S->kv_cache_), n);
    } else {
#ifdef HAS_CUDA
      float* d = nullptr;
      cudaMalloc(&d, emb.size() * sizeof(float));
      cudaMemcpy(d, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice);
      all = S->model_->forward_device_batch_prefill(d, n, 0, &S->kv_cache_, 0, true);
      cudaFree(d);
#endif
    }
  }
  S->kv_cache_.seq_len = n;
  if (all.size() != static_cast<size_t>(n) * V)
    throw std::runtime_error("prefill returned " + std::to_string(all.size()) + " values");
  return all;
}

// One token at position pos (the cache holds 0 .. pos - 1).
std::vector<float> step(int id, int pos) {
  std::vector<float> lg;
  if (S->ir_) {
    lg = S->ir_->forward({id}, pos);
  } else {
    std::vector<float> x = S->model_->lookup_embedding(id);
    if (all_cpu) {
      lg = S->model_->forward(x, pos, &S->kv_cache_, nullptr);
    } else {
#ifdef HAS_CUDA
      lg = S->model_->forward_device_graph(id, pos, &S->kv_cache_);
      if (lg.empty()) {
        cudaMemcpy(S->model_->get_x_dev(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
        lg = S->model_->forward_device(S->model_->get_x_dev(), pos, &S->kv_cache_, nullptr);
      }
#endif
    }
  }
  S->kv_cache_.seq_len = pos + 1;
  if (lg.size() != static_cast<size_t>(V)) throw std::runtime_error("decode returned " + std::to_string(lg.size()));
  return lg;
}

int argmax(const float* v) { return static_cast<int>(std::max_element(v, v + V) - v); }

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: tp_session_logits MODEL.gguf GPU_LAYERS < input.json\n";
    return 2;
  }
  try {
    tinyllama::TinyLlamaSession s(argv[1], "tokenizer.json", 4, std::stoi(argv[2]), true);
    S = &s;
    const ModelConfig& c = s.config_;
    V = s.ir_ ? s.ir_->vocab() : c.vocab_size;
    D = c.hidden_size;
    all_cpu = c.num_cpu_offload_layers == c.num_hidden_layers;
    if (!s.ir_ && !all_cpu && c.num_cpu_offload_layers != 0) throw std::runtime_error("mixed CPU/GPU layers");
    json out{{"path", s.ir_ ? "interpreter" : "tuned"},
             {"device", s.ir_ ? (s.ir_->on_gpu() ? "cuda" : "cpu") : (all_cpu ? "cpu" : "cuda")},
             {"num_cpu_offload_layers", c.num_cpu_offload_layers}, {"num_hidden_layers", c.num_hidden_layers},
             {"heads", c.num_attention_heads}, {"kv_heads", c.num_key_value_heads}, {"vocab", V}};
    const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    const json inp = json::parse(in);
    json res = json::array();
    for (size_t k = 0; k < inp["seqs"].size(); ++k) {
      const std::vector<int> ids = inp["seqs"][k].get<std::vector<int>>();
      const int n = static_cast<int>(ids.size()), P = inp["prompt_len"][k].get<int>();
      json rec;
      const std::vector<float> all = prefill(ids);
      json rows = json::array();
      for (int t = 0; t < n; ++t) rows.push_back(std::vector<float>(all.begin() + t * V, all.begin() + (t + 1) * V));
      rec["prefill"] = std::move(rows);
      fresh();
      json drows = json::array();
      for (int t = 0; t < n; ++t) drows.push_back(step(ids[t], t));
      rec["decode"] = std::move(drows);
      std::vector<float> l = prefill(std::vector<int>(ids.begin(), ids.begin() + P));
      std::vector<int> g{argmax(l.data() + static_cast<size_t>(P - 1) * V)};
      for (int pos = P; static_cast<int>(g.size()) < n - P; ++pos) {
        l = step(g.back(), pos);
        g.push_back(argmax(l.data()));
      }
      rec["free"] = g;
      res.push_back(std::move(rec));
    }
    out["seqs"] = std::move(res);
    std::cout << out.dump() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "tp_session_logits: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
