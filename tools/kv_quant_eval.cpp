// kv_quant_eval: the quantised KV cache formats (EMBER_KV, kv_quant.h) against the f32 cache, on the CUDA
// device, for one GGUF (R2, experiments/kv_quant_r2_2026-10-28). One process holds an f32 interpreter and
// one per format (EMBER_KV set while each is built), all with the same kv capacity (EMBER_IR_CTX).
//
//   kv_quant_eval MODEL.gguf INPUTS.json OUTDIR [--formats f16,q8,q4] [--contexts 2048,4096,8192]
//                 [--chunk 512] [--gen 200] [--bench 128] [--phases tf_prefill,tf_decode,greedy,bench] [--cpu 1]
//
// --cpu 1: the CPU interpreters instead (any build), which keep the f32 cache and store each written row's
// dequantised value (kv_quant.h roundtrip_rows): the formats' numerics without the device kernels.
//
// INPUTS.json: {"tf": [{"name", "ids"}], "greedy": [{"name", "ctx", "ids"}]}. Phases, each written to
// OUTDIR/<phase>.json as soon as it ends:
//   tf_prefill  teacher forcing through each tf input in prefill chunks of --chunk tokens (every row's
//               logits), the cache filled chunk by chunk;
//   tf_decode   teacher forcing one token at a time (the captured decode graph): the decode kernels;
//               both report, per position p (its row predicts ids[p + 1]): max |dlogit| and mean |dlogit|
//               as fractions of the f32 row's range, top-1 agreement, and the next token's NLL under each
//               cache; summarised over [0, C) and over [C - 512, C) for each context C, and per 512 bins;
//   greedy      each greedy prompt prefilled in one call, then --gen greedy tokens: ids and the first
//               index where a format's ids leave the f32 ids;
//   (the phases run in the order --phases gives)
//   bench       for each context C: the first C ids of the first tf input prefilled (timed), then --bench
//               decode steps teacher-forced on the following ids (wall ms per step, the first 8 dropped);
//               the kv cache's device bytes.
// Each interpreter's tf_decode logits are hashed (FNV-1a over the float bits): the run-to-run check.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;

namespace {

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, c))
    if (!item.empty()) out.push_back(item);
  return out;
}

int argmax(const float* v, int n) { return static_cast<int>(std::max_element(v, v + n) - v); }

double nll(const float* v, int n, int tok) {
  double mx = v[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, static_cast<double>(v[i]));
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += std::exp(static_cast<double>(v[i]) - mx);
  return mx + std::log(s) - v[tok];
}

uint64_t fnv(uint64_t h, const float* v, size_t n) {
  const auto* b = reinterpret_cast<const unsigned char*>(v);
  for (size_t i = 0; i < n * sizeof(float); ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

// Per-position comparison of one format against f32.
struct Pos {
  std::vector<double> rmax, rmean, nll_ref, nll_q;
  std::vector<char> agree;
  void add(const float* q, const float* ref, int V, int next) {
    float lo = ref[0], hi = ref[0];
    double dmax = 0.0, dsum = 0.0;
    for (int i = 0; i < V; ++i) {
      lo = std::min(lo, ref[i]);
      hi = std::max(hi, ref[i]);
      const double d = std::fabs(static_cast<double>(q[i]) - ref[i]);
      dmax = std::max(dmax, d);
      dsum += d;
    }
    const double range = hi > lo ? hi - lo : 1.0;
    rmax.push_back(dmax / range);
    rmean.push_back(dsum / V / range);
    agree.push_back(argmax(q, V) == argmax(ref, V));
    nll_ref.push_back(next >= 0 ? nll(ref, V, next) : NAN);
    nll_q.push_back(next >= 0 ? nll(q, V, next) : NAN);
  }
};

json summarise(const Pos& p, int a, int b) {
  b = std::min<int>(b, static_cast<int>(p.rmax.size()));
  double mx = 0.0, mean = 0.0, nr = 0.0, nq = 0.0;
  int agree = 0, n = 0, nn = 0;
  for (int i = a; i < b; ++i) {
    mx = std::max(mx, p.rmax[i]);
    mean += p.rmean[i];
    agree += p.agree[i];
    ++n;
    if (!std::isnan(p.nll_ref[i])) {
      nr += p.nll_ref[i];
      nq += p.nll_q[i];
      ++nn;
    }
  }
  json j = {{"from", a}, {"to", b}, {"positions", n}, {"max_rel", mx}, {"mean_rel", n ? mean / n : 0.0},
            {"top1", agree}, {"top1_frac", n ? static_cast<double>(agree) / n : 0.0}};
  if (nn) {
    j["ppl_f32"] = std::exp(nr / nn);
    j["ppl_q"] = std::exp(nq / nn);
    j["ppl_rel_diff"] = std::exp(nq / nn) / std::exp(nr / nn) - 1.0;
  }
  return j;
}

json report(const Pos& p, const std::vector<int>& contexts) {
  json ctx = json::object(), bins = json::array();
  for (int c : contexts) {
    if (c > static_cast<int>(p.rmax.size())) continue;
    ctx[std::to_string(c)] = {{"all", summarise(p, 0, c)}, {"tail512", summarise(p, std::max(0, c - 512), c)}};
  }
  for (int a = 0; a < static_cast<int>(p.rmax.size()); a += 512) bins.push_back(summarise(p, a, a + 512));
  return {{"contexts", ctx}, {"bins512", bins}};
}

void write_json(const std::string& path, const json& j) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp);
    f << j.dump(1) << "\n";
  }
  std::rename(tmp.c_str(), path.c_str());
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: kv_quant_eval MODEL.gguf INPUTS.json OUTDIR [--formats f16,q8,q4] [--contexts 2048,4096,8192]"
                 " [--chunk 512] [--gen 200] [--bench 128] [--phases tf_prefill,tf_decode,greedy,bench] [--cpu 1]\n";
    return 2;
  }
  std::vector<std::string> formats = {"f16", "q8", "q4"};
  std::vector<int> contexts = {2048, 4096, 8192};
  std::vector<std::string> phases = {"tf_prefill", "tf_decode", "greedy", "bench"};
  int chunk = 512, gen = 200, bench = 128;
  bool gpu = true;
  for (int i = 4; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--formats") formats = split(v, ',');
    else if (k == "--contexts") {
      contexts.clear();
      for (auto& s : split(v, ',')) contexts.push_back(std::atoi(s.c_str()));
    } else if (k == "--chunk") chunk = std::atoi(v.c_str());
    else if (k == "--gen") gen = std::atoi(v.c_str());
    else if (k == "--bench") bench = std::atoi(v.c_str());
    else if (k == "--phases") phases = split(v, ',');
    else if (k == "--cpu") gpu = v != "1";
    else {
      std::cerr << "unknown option " << k << "\n";
      return 2;
    }
  }
  const std::string outdir = argv[3];
  std::ifstream fi(argv[2]);
  if (!fi) {
    std::cerr << "cannot read " << argv[2] << "\n";
    return 2;
  }
  const json inputs = json::parse(fi);
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  if (ir.empty() || (gpu && !ember_ir::device_supported(ir))) {
    std::cerr << "no " << (gpu ? "device " : "") << "IR for " << argv[1] << "\n";
    return 1;
  }
  // The f32 interpreter first (EMBER_KV unset), then one per format.
  std::vector<std::string> names = {"f32"};
  std::vector<std::unique_ptr<ember_ir::Interpreter>> m;
  unsetenv("EMBER_KV");
  m.emplace_back(new ember_ir::Interpreter(ir, gguf, gpu));
  for (const auto& f : formats) {
    setenv("EMBER_KV", f.c_str(), 1);
    m.emplace_back(new ember_ir::Interpreter(ir, gguf, gpu));
    names.push_back(f);
  }
  unsetenv("EMBER_KV");
  const int V = m[0]->vocab(), Q = static_cast<int>(m.size());
  json meta = {{"gguf", argv[1]}, {"device", m[0]->on_gpu() ? "cuda" : "cpu"}, {"max_len", m[0]->max_len()}, {"vocab", V}, {"layers", m[0]->n_layers()},
               {"family_hint", m[0]->family_hint()}, {"deterministic_env", std::getenv("EMBER_DETERMINISTIC") ? std::getenv("EMBER_DETERMINISTIC") : ""},
               {"kv", json::object()}};
  for (int i = 0; i < Q; ++i) {
    if (std::string(m[i]->kv_format()) != names[i]) {
      std::cerr << "interpreter " << i << " reports kv format " << m[i]->kv_format() << ", expected " << names[i] << "\n";
      return 1;
    }
    meta["kv"][names[i]] = {{"cache_bytes", m[i]->kv_cache_bytes()}, {"scratch_bytes", m[i]->kv_scratch_bytes()},
                            {"bytes_per_position", static_cast<double>(m[i]->kv_cache_bytes()) / m[i]->max_len()}};
  }
  write_json(outdir + "/meta.json", meta);
  std::cerr << meta.dump() << "\n";
  // The phases in the order given (each one's receipt written as it ends).
  for (const std::string& phase_s : phases) {
   if (phase_s == "tf_prefill" || phase_s == "tf_decode") {
    const char* phase = phase_s.c_str();
    const bool decode = std::string(phase) == "tf_decode";
    json out = json::array();
    for (const auto& in : inputs.at("tf")) {
      const std::vector<int> ids = in.at("ids").get<std::vector<int>>();
      const int N = std::min({static_cast<int>(ids.size()), m[0]->max_len(),
                              *std::max_element(contexts.begin(), contexts.end())});
      std::vector<Pos> pos(Q);
      std::vector<uint64_t> hash(Q, 1469598103934665603ull);
      const double t0 = now_ms();
      if (decode) {
        for (int p = 0; p < N; ++p) {
          std::vector<std::vector<float>> rows(Q);
          for (int i = 0; i < Q; ++i) {
            rows[i] = m[i]->forward({ids[p]}, p, false);
            hash[i] = fnv(hash[i], rows[i].data(), rows[i].size());
          }
          const int next = p + 1 < static_cast<int>(ids.size()) ? ids[p + 1] : -1;
          for (int i = 0; i < Q; ++i) pos[i].add(rows[i].data(), rows[0].data(), V, next);
          if (p % 1024 == 1023)
            std::cerr << phase << " " << in.value("name", "") << " " << p + 1 << "/" << N << " "
                      << (now_ms() - t0) / 1000 << " s\n";
        }
      } else {
        for (int c = 0; c < N; c += chunk) {
          const int n = std::min(chunk, N - c);
          const std::vector<int> part(ids.begin() + c, ids.begin() + c + n);
          std::vector<std::vector<float>> rows(Q);
          for (int i = 0; i < Q; ++i) {
            rows[i] = m[i]->forward(part, c, true);
            hash[i] = fnv(hash[i], rows[i].data(), rows[i].size());
          }
          for (int t = 0; t < n; ++t) {
            const int p = c + t;
            const int next = p + 1 < static_cast<int>(ids.size()) ? ids[p + 1] : -1;
            for (int i = 0; i < Q; ++i)
              pos[i].add(rows[i].data() + static_cast<size_t>(t) * V, rows[0].data() + static_cast<size_t>(t) * V, V, next);
          }
        }
      }
      json r = {{"name", in.value("name", "")}, {"positions", N}, {"seconds", (now_ms() - t0) / 1000}, {"formats", json::object()}};
      for (int i = 0; i < Q; ++i) {
        char hx[32];
        std::snprintf(hx, sizeof hx, "%016llx", static_cast<unsigned long long>(hash[i]));
        json f = i == 0 ? json{{"ppl", report(pos[0], contexts)["contexts"]}} : report(pos[i], contexts);
        f["logits_fnv1a"] = hx;
        r["formats"][names[i]] = f;
      }
      out.push_back(r);
      std::cerr << phase << " " << in.value("name", "") << " done in " << (now_ms() - t0) / 1000 << " s\n";
      write_json(outdir + "/" + phase + ".json", out);
    }
   } else if (phase_s == "greedy") {
    json out = json::array();
    for (const auto& in : inputs.at("greedy")) {
      const std::vector<int> ids = in.at("ids").get<std::vector<int>>();
      json r = {{"name", in.value("name", "")}, {"ctx", in.value("ctx", 0)}, {"prompt_tokens", ids.size()}, {"formats", json::object()}};
      std::vector<std::vector<int>> outs(Q);
      for (int i = 0; i < Q; ++i) {
        std::vector<float> l = m[i]->forward(ids, 0, false);
        int p = static_cast<int>(ids.size());
        for (int g = 0; g < gen && p < m[i]->max_len(); ++g) {
          const int tok = argmax(l.data(), V);
          outs[i].push_back(tok);
          if (g + 1 < gen) l = m[i]->forward({tok}, p++, false);
        }
      }
      for (int i = 0; i < Q; ++i) {
        int first_diff = -1;
        for (size_t g = 0; g < outs[i].size() && g < outs[0].size(); ++g)
          if (outs[i][g] != outs[0][g]) {
            first_diff = static_cast<int>(g);
            break;
          }
        r["formats"][names[i]] = {{"ids", outs[i]}, {"first_diff", first_diff}, {"identical", first_diff < 0 && outs[i].size() == outs[0].size()}};
      }
      out.push_back(r);
      std::cerr << "greedy " << in.value("name", "") << ":";
      for (int i = 1; i < Q; ++i) std::cerr << " " << names[i] << " " << r["formats"][names[i]]["first_diff"];
      std::cerr << "\n";
      write_json(outdir + "/greedy.json", out);
    }
   } else if (phase_s == "bench") {
    const std::vector<int> ids = inputs.at("tf").at(0).at("ids").get<std::vector<int>>();
    json out = json::array();
    for (int c : contexts) {
      if (c + bench > static_cast<int>(ids.size()) || c + bench > m[0]->max_len()) {
        std::cerr << "bench: context " << c << " + " << bench << " does not fit the input or the cache\n";
        continue;
      }
      json r = {{"ctx", c}, {"steps", bench}, {"formats", json::object()}};
      for (int i = 0; i < Q; ++i) {
        const std::vector<int> prompt(ids.begin(), ids.begin() + c);
        const double t0 = now_ms();
        m[i]->forward(prompt, 0, false);
        const double prefill = now_ms() - t0;
        std::vector<double> ms;
        for (int s = 0; s < bench; ++s) {
          const double a = now_ms();
          m[i]->forward({ids[c + s]}, c + s, false);
          if (s >= 8) ms.push_back(now_ms() - a);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        r["formats"][names[i]] = {{"prefill_ms", prefill}, {"decode_ms_median", med}, {"decode_ms_min", ms.front()},
                                  {"decode_ms_p90", ms[ms.size() * 9 / 10]}, {"decode_tok_s", 1000.0 / med},
                                  {"kv_bytes_at_ctx", static_cast<double>(m[i]->kv_cache_bytes()) / m[i]->max_len() * c}};
      }
      out.push_back(r);
      std::cerr << "bench " << c << ": " << r["formats"].dump() << "\n";
      write_json(outdir + "/bench.json", out);
    }
   } else {
    std::cerr << "unknown phase " << phase_s << "\n";
    return 2;
   }
  }
  std::cerr << "kv_quant_eval done\n";
  return 0;
}
