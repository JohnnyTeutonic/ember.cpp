// vf_bench: the cost of a speculative verification pass against one decode step, and its rows against the decode
// step's logits bit for bit (ember.cpp, experiments/verify_2026-10-27).
//
//   vf_bench MODEL.gguf OUT.json [REPS] [PREFIXES]     (PREFIXES: comma-separated cached prefix lengths, default 32,384)
//
// The session is built as tinyllama_server builds it (every layer on the GPU). For each prefix length P the cache is
// filled by one batched prefill of fixed pseudo-random ids, then, all at position P (KV rows past P are scratch):
//   decode     forward_device_graph (the engine's decode step: graph replay, logits to pinned memory, sync), and the
//              host greedy sampler on its row (a replica of api.cpp's top_k = 1 path), timed apart;
//   legacy     forward_device_verify (the per-launch pass before this work) for n = 2..9 rows;
//   ladder     forward_device_verify_graph under each option set (graph on/off, linear kernel, attention batched or
//              per row), n = 2..9, plus the host cost of the old interface (every row's logits to pageable memory);
//   stages     the same pass with EMBER_VERIFY_TIMING's synchronised per-stage events, per linear kernel;
//   bits       for n = 9: the decode step run at P..P+8 on ids t_0..t_8, then the pass on the same ids, each row
//              compared with the decode step's logits (bit-equal count, max |d|, argmax), per linear kernel and for
//              the legacy pass.
// Times are wall clock on the host (std::chrono, the call returning its result), median and p10/p90 over REPS calls
// after warm-up; NVTX ranges name every timed block for nsys.
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

// api.cpp's sample_top_k_top_p_temperature at temperature 1, top_k 1, top_p 1 (its cost, and its answer).
int host_greedy(const std::vector<float>& logits, std::mt19937& rng) {
  const int vocab_size = static_cast<int>(logits.size());
  std::vector<float> scaled_logits(vocab_size);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float logit : logits) max_logit = (std::max)(max_logit, logit);
  for (int i = 0; i < vocab_size; ++i) scaled_logits[i] = (logits[i] - max_logit) * 1.0f;
  std::vector<double> probs_double(vocab_size);
  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    probs_double[i] = std::exp(static_cast<double>(scaled_logits[i]));
    sum_exp += probs_double[i];
  }
  for (int i = 0; i < vocab_size; ++i) probs_double[i] /= sum_exp;
  std::vector<std::pair<float, int>> prob_idx(vocab_size);
  for (int i = 0; i < vocab_size; ++i) prob_idx[i] = {static_cast<float>(probs_double[i]), i};
  std::partial_sort(prob_idx.begin(), prob_idx.begin() + 1, prob_idx.end(), std::greater<std::pair<float, int>>());
  prob_idx.resize(1);
  std::vector<float> one(1, 1.0f);
  std::discrete_distribution<int> dist(one.begin(), one.end());
  (void)dist(rng);
  return prob_idx[0].second;
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

template <class F>
json timed(const char* name, int reps, F&& f) {
  for (int i = 0; i < 5; ++i) f();  // warm-up (graph capture on the first call)
  std::vector<double> t;
  t.reserve(reps);
  nvtxRangePushA(name);
  for (int i = 0; i < reps; ++i) {
    const auto t0 = clk::now();
    f();
    t.push_back(ms_since(t0));
  }
  nvtxRangePop();
  return summary(t);
}

struct Opt {
  const char* name;
  int linear;
  bool graph;
  bool batched;
};

void set_opts(const Opt& o, bool timing = false) {
  auto& v = S->model_->verify_options();
  v.linear = o.linear;
  v.graph = o.graph;
  v.batched_attention = o.batched;
  v.timing = timing;
}

const char* linear_name(int l) {
  static const char* n[] = {"smallm", "gemm", "gemvb", "rows"};
  return n[l];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: vf_bench MODEL.gguf OUT.json [REPS] [PREFIXES]\n";
    return 2;
  }
  const int reps = argc > 3 ? std::stoi(argv[3]) : 200;
  std::vector<int> prefixes{32, 384};
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
    json out{{"model", argv[1]}, {"gpu", prop.name}, {"sm", prop.major * 10 + prop.minor},
             {"deterministic", std::getenv("EMBER_DETERMINISTIC") ? std::getenv("EMBER_DETERMINISTIC") : ""},
             {"hidden", D}, {"layers", c.num_hidden_layers}, {"heads", c.num_attention_heads},
             {"kv_heads", c.num_key_value_heads}, {"ffn", c.intermediate_size}, {"vocab", V},
             {"max_seq", s.kv_cache_.allocated_max_seq_len}, {"reps", reps}};
    const std::vector<Opt> ladder = {
        {"nograph_gemm_perrow", 1, false, false},  // the legacy pass's kernels' shape, device ids, pinned picks
        {"graph_gemm_perrow", 1, true, false},
        {"graph_gemm_batched", 1, true, true},
        {"graph_gemvb_batched", 2, true, true},
        {"graph_rows_batched", 3, true, true},
        {"graph_smallm_batched", 0, true, true},
    };
    json per_prefix = json::array();
    for (int P : prefixes) {
      if (P + 16 > s.kv_cache_.allocated_max_seq_len) continue;
      prefill(P);
      json rec{{"prefix", P}};
      std::mt19937 rng(0);
      const int tok = ids_for(1, 7)[0];
      std::vector<float> last;
      rec["decode_ms"] = timed("decode", reps, [&] { last = decode(tok, P); });
      rec["host_sample_ms"] = timed("host_sample", reps, [&] { (void)host_greedy(last, rng); });
      // Legacy pass and the ladder for every n.
      json legacy = json::object(), lad = json::object(), host_rows = json::object();
      for (int n = 2; n <= 9; ++n) {
        const std::vector<int> ids = ids_for(n, 3);
        legacy[std::to_string(n)] = timed("legacy", reps, [&] {
          const std::vector<float> a = S->model_->forward_device_verify(ids, P, &S->kv_cache_);
          if (a.size() != static_cast<size_t>(n) * V) throw std::runtime_error("legacy verify failed");
        });
        for (const Opt& o : ladder) {
          set_opts(o);
          std::vector<ember_verify::RowPick> picks;
          lad[o.name][std::to_string(n)] = timed(o.name, reps, [&] {
            if (!S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks))
              throw std::runtime_error(std::string("verify graph refused: ") + o.name);
          });
        }
        // The old interface's host side on top of the graph pass: every row's logits copied to pageable memory.
        set_opts(ladder.back());
        std::vector<ember_verify::RowPick> picks;
        host_rows[std::to_string(n)] = timed("graph_plus_rows_to_host", reps, [&] {
          S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks);
          for (int r = 0; r < n; ++r) (void)S->model_->verify_row_logits(r);
        });
      }
      rec["legacy_ms"] = legacy;
      rec["ladder_ms"] = lad;
      rec["graph_smallm_plus_rows_to_host_ms"] = host_rows;
      // Per-stage synchronised timing at n = 5 and 9, per linear kernel (graph off).
      json stages = json::object();
      for (int n : {5, 9}) {
        const std::vector<int> ids = ids_for(n, 3);
        for (int l = 0; l < 4; ++l) {
          set_opts({"t", l, false, true}, true);
          std::vector<ember_verify::RowPick> picks;
          S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks);
          S->model_->verify_stage_ms().clear();
          const int r = std::max(20, reps / 5);
          for (int i = 0; i < r; ++i) S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks);
          json st = json::object();
          double tot = 0;
          for (const auto& kv : S->model_->verify_stage_ms()) {
            st[kv.first] = kv.second / r;
            tot += kv.second / r;
          }
          st["sum"] = tot;
          stages[std::to_string(n)][linear_name(l)] = st;
        }
      }
      rec["stages_ms"] = stages;
      // Bits: decode at P..P+8 vs the pass's rows.
      {
        const int n = 9;
        const std::vector<int> ids = ids_for(n, 5);
        std::vector<std::vector<float>> dec(n);
        for (int i = 0; i < n; ++i) dec[i] = decode(ids[i], P + i);
        auto compare = [&](const std::vector<std::vector<float>>& rows) {
          json r = json::array();
          for (int i = 0; i < n; ++i) {
            int eq = 0;
            double worst = 0;
            for (int j = 0; j < V; ++j) {
              uint32_t a, b;
              std::memcpy(&a, &dec[i][j], 4);
              std::memcpy(&b, &rows[i][j], 4);
              eq += a == b;
              worst = std::max(worst, (double)std::fabs(dec[i][j] - rows[i][j]));
            }
            const int am = (int)(std::max_element(dec[i].begin(), dec[i].end()) - dec[i].begin());
            const int bm = (int)(std::max_element(rows[i].begin(), rows[i].end()) - rows[i].begin());
            r.push_back({{"bit_equal", eq}, {"of", V}, {"max_abs", worst}, {"argmax_equal", am == bm}});
          }
          return r;
        };
        json bits = json::object();
        {
          const std::vector<float> a = S->model_->forward_device_verify(ids, P, &S->kv_cache_);
          std::vector<std::vector<float>> rows(n);
          for (int i = 0; i < n; ++i) rows[i].assign(a.begin() + (size_t)i * V, a.begin() + (size_t)(i + 1) * V);
          bits["legacy"] = compare(rows);
        }
        for (int l = 0; l < 4; ++l) {
          for (bool batched : {true, false}) {
            set_opts({"b", l, true, batched});
            std::vector<ember_verify::RowPick> picks;
            if (!S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks)) throw std::runtime_error("bits");
            std::vector<std::vector<float>> rows(n);
            for (int i = 0; i < n; ++i) rows[i] = S->model_->verify_row_logits(i);
            json cmp = compare(rows);
            // The device pick against the row it was taken from.
            for (int i = 0; i < n; ++i) {
              const auto mx = std::max_element(rows[i].begin(), rows[i].end());
              cmp[i]["pick_first_ok"] = picks[i].first == (int)(mx - rows[i].begin()) && picks[i].max == *mx;
            }
            bits[std::string(linear_name(l)) + (batched ? "_batched" : "_perrow")] = cmp;
          }
        }
        rec["bits_n9"] = bits;
      }
      set_opts(ladder.back());
      per_prefix.push_back(rec);
      std::cerr << "prefix " << P << " done\n";
    }
    out["prefixes"] = per_prefix;
    std::ofstream(argv[2]) << out.dump(1) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "vf_bench: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
