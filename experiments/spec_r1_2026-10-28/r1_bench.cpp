// r1_bench: the decode step and the speculative verification pass at the current decode attention, by linear
// kernel, with the bits of every pass row against the decode step (ember.cpp, experiments/spec_r1_2026-10-28).
//
//   r1_bench MODEL.gguf OUT.json [REPS] [PREFIXES]                 the full bench (PREFIXES default 32,384)
//   r1_bench MODEL.gguf OUT.json REPS P profile decode cublas|fixed  REPS decode steps only (for nsys)
//   r1_bench MODEL.gguf OUT.json REPS P profile pass LINEAR N        REPS passes of N rows only (for nsys)
//
// Built against the tree with vf_bench's trick (private members public). For each cached prefix P the cache is
// filled by one batched prefill of fixed pseudo-random ids; then, all at position P:
//   decode_ms[cublas|fixed]   forward_device_graph (graph replay, logits to pinned memory, sync) with the decode
//                             step's linear layers on cuBLAS or on fixed_linear (the graph recaptured for each);
//   host_sample_ms            the host greedy sampler on one row (api.cpp's top_k = 1 path), timed apart;
//   pass_ms[linear][n]        forward_device_verify_graph for n = 1..9 rows (graph on, attention batched, device
//                             picks) with the linear layers smallm, rows (the decode step's cuBLAS call once per
//                             row) or fixed (fixed_linear over the n rows at once);
//   bits_n9[decode][linear]   the decode step run at P..P+8 on ids t_0..t_8, then the pass of those 9 ids, each row
//                             compared with the decode step's logits bit for bit.
// Times are host wall clock (std::chrono), median and p10/p90 over REPS calls after five warm-up calls.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define private public
#define protected public
#include "graph_ir.h"
#include "logger.h"
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

// api.cpp's sample_top_k_top_p_temperature at temperature 1, top_k 1, top_p 1 (its cost and its answer).
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
  for (int i = 0; i < n; ++i)
    ids[i] = 4 + static_cast<int>((1103515245u * (unsigned)(i + 17 * salt) + 12345u) % (unsigned)(V - 4));
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

// The decode step's linear layers on cuBLAS (0) or fixed_linear (1); the decode graph is recaptured.
void set_decode_linear(int fixed) {
  TinyLlamaModel* m = S->model_.get();
  m->decode_linear_ = fixed;
  if (m->decode_graph_exec_) {
    cudaGraphExecDestroy(m->decode_graph_exec_);
    m->decode_graph_exec_ = nullptr;
  }
}

std::vector<float> decode(int id, int pos) {
  std::vector<float> lg = S->model_->forward_device_graph(id, pos, &S->kv_cache_);
  if (lg.size() != static_cast<size_t>(V)) throw std::runtime_error("decode graph unavailable");
  return lg;
}

template <class F>
json timed(const std::string& name, int reps, F&& f) {
  for (int i = 0; i < 5; ++i) f();  // warm-up (graph capture on the first call)
  std::vector<double> t;
  t.reserve(reps);
  nvtxRangePushA(name.c_str());
  for (int i = 0; i < reps; ++i) {
    const auto t0 = clk::now();
    f();
    t.push_back(ms_since(t0));
  }
  nvtxRangePop();
  return summary(t);
}

const std::map<std::string, int> kLinear = {{"smallm", 0}, {"rows", 3}, {"fixed", 4}};

void set_linear(int l) {
  auto& v = S->model_->verify_options();
  v.linear = l;
  v.graph = true;
  v.batched_attention = true;
  v.timing = false;
}

void pass(const std::vector<int>& ids, int P) {
  std::vector<ember_verify::RowPick> picks;
  if (!S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks))
    throw std::runtime_error("verify graph refused");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: r1_bench MODEL.gguf OUT.json [REPS] [PREFIXES] [profile decode|pass ...]\n";
    return 2;
  }
  const int reps = argc > 3 ? std::stoi(argv[3]) : 200;
  std::vector<int> prefixes{32, 384};
  if (argc > 4) {
    prefixes.clear();
    std::stringstream ss(argv[4]);
    for (std::string x; std::getline(ss, x, ',');) prefixes.push_back(std::stoi(x));
  }
  const bool profile = argc > 5 && std::string(argv[5]) == "profile";
  try {
    tinyllama::TinyLlamaSession s(argv[1], "tokenizer.json", 4, -1, true);
    S = &s;
    const ModelConfig& c = s.config_;
    if (s.ir_ || c.num_cpu_offload_layers != 0) throw std::runtime_error("needs the tuned llama path on the GPU");
    V = c.vocab_size;
    D = c.hidden_size;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const char* det = std::getenv("EMBER_DETERMINISTIC");
    json out{{"model", argv[1]}, {"gpu", prop.name}, {"sm", prop.major * 10 + prop.minor},
             {"deterministic", det ? det : ""}, {"hidden", D}, {"layers", c.num_hidden_layers},
             {"heads", c.num_attention_heads}, {"kv_heads", c.num_key_value_heads}, {"ffn", c.intermediate_size},
             {"vocab", V}, {"max_seq", s.kv_cache_.allocated_max_seq_len}, {"reps", reps}};
    if (profile) {
      // REPS calls of one thing only, after five warm-up calls: nsys's kernel sums divided by REPS + 5.
      const int P = prefixes.at(0);
      const std::string what = argv[6];
      prefill(P);
      if (what == "decode") {
        set_decode_linear(std::string(argv[7]) == "fixed" ? 1 : 0);
        const int tok = ids_for(1, 7)[0];
        out["decode_ms"] = timed("decode", reps, [&] { (void)decode(tok, P); });
        out["profile"] = {{"what", "decode"}, {"decode_linear", argv[7]}, {"prefix", P}, {"calls", reps + 5}};
      } else {
        const int n = std::stoi(argv[8]);
        set_linear(kLinear.at(argv[7]));
        const std::vector<int> ids = ids_for(n, 3);
        out["pass_ms"] = timed("pass", reps, [&] { pass(ids, P); });
        out["profile"] = {{"what", "pass"}, {"linear", argv[7]}, {"n", n}, {"prefix", P}, {"calls", reps + 5}};
      }
      std::ofstream(argv[2]) << out.dump(1) << "\n";
      return 0;
    }
    json per_prefix = json::array();
    for (int P : prefixes) {
      if (P + 16 > s.kv_cache_.allocated_max_seq_len) continue;
      prefill(P);
      json rec{{"prefix", P}};
      std::mt19937 rng(0);
      const int tok = ids_for(1, 7)[0];
      std::vector<float> last;
      for (int dl : {0, 1}) {
        set_decode_linear(dl);
        rec["decode_ms"][dl ? "fixed" : "cublas"] = timed("decode", reps, [&] { last = decode(tok, P); });
      }
      rec["host_sample_ms"] = timed("host_sample", reps, [&] { (void)host_greedy(last, rng); });
      json pm = json::object();
      for (const auto& lk : kLinear) {
        set_linear(lk.second);
        for (int n = 1; n <= 9; ++n) {
          const std::vector<int> ids = ids_for(n, 3);
          pm[lk.first][std::to_string(n)] = timed("pass_" + lk.first, reps, [&] { pass(ids, P); });
        }
      }
      rec["pass_ms"] = pm;
      // Bits: the decode step at P..P+8 (each decode linear) against every pass row (each linear).
      {
        const int n = 9;
        const std::vector<int> ids = ids_for(n, 5);
        json bits = json::object();
        for (int dl : {0, 1}) {
          set_decode_linear(dl);
          std::vector<std::vector<float>> dec(n);
          for (int i = 0; i < n; ++i) dec[i] = decode(ids[i], P + i);
          for (const auto& lk : kLinear) {
            set_linear(lk.second);
            std::vector<ember_verify::RowPick> picks;
            if (!S->model_->forward_device_verify_graph(ids, P, &S->kv_cache_, &picks)) throw std::runtime_error("bits");
            json r = json::array();
            int rows_equal = 0;
            for (int i = 0; i < n; ++i) {
              const std::vector<float> row = S->model_->verify_row_logits(i);
              int eq = 0;
              double worst = 0;
              for (int j = 0; j < V; ++j) {
                uint32_t a, b;
                std::memcpy(&a, &dec[i][j], 4);
                std::memcpy(&b, &row[j], 4);
                eq += a == b;
                worst = std::max(worst, (double)std::fabs(dec[i][j] - row[j]));
              }
              const int am = (int)(std::max_element(dec[i].begin(), dec[i].end()) - dec[i].begin());
              const int bm = (int)(std::max_element(row.begin(), row.end()) - row.begin());
              rows_equal += eq == V;
              r.push_back({{"bit_equal", eq}, {"of", V}, {"max_abs", worst}, {"argmax_equal", am == bm},
                           {"pick_first_ok", picks[i].first == bm}});
            }
            bits[dl ? "fixed" : "cublas"][lk.first] = {{"rows_bit_identical", rows_equal}, {"rows", n}, {"per_row", r}};
          }
        }
        rec["bits_n9"] = bits;
      }
      per_prefix.push_back(rec);
      std::cerr << "prefix " << P << " done\n";
    }
    out["prefixes"] = per_prefix;
    std::ofstream(argv[2]) << out.dump(1) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "r1_bench: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
