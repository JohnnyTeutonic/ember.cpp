// state_wasm_gate: one engine's run of tools/state_device_gate's checks, written to files, so the IR
// interpreter compiled to WebAssembly (the page's sources and flags, in node with NODERAWFS) can be
// compared with ember's native CPU build on per-turn state tokens (IR_SPEC 6.8, 6.9) and any other
// IR. The same source is built for both sides and runs the same sequence of calls on each, so their
// outputs line up call for call; experiments/wasm_state_2026-10-25/compare.py compares them.
//
//   state_wasm_gate MODEL.gguf INPUT.json OUT_DIR
//
// input: state_device_gate's {"seqs", "prompt_len", "user_id", "calls"}.
// Per sequence, "on" = the IR's own eviction setting and "off" = EMBER_EVICT=0 (set before each
// interpreter is built):
//   - logits of every position: one-call prefill on and off, token-by-token decode on, and with
//     user_id the turn-by-turn prefill on (one call per turn, split before each user_id);
//   - with prompt_len: free-running greedy on (prefill the prompt, decode len - P tokens);
//   - kv rows held after a prefill of the sequence less its last id, on and off;
//   - under eviction with user_id: the turn-by-turn state, then rewinds into turn 0 (n = 3) and turn 1
//     (n = start(1) + 3): resume_point(n), whether a direct forward at n is refused, the re-prefill's
//     logits of positions n .. end, kv rows held after it.
// calls: tools/state_tokens_test.cpp's script on an "on" and an "off" interpreter (each call resumes at
// resume_point(from)), plus a fresh prefill per call: resume points, refusals, rows held and the logits
// of positions from .. end.
// OUT_DIR/run.json holds every integer result and, per logits block, its first row and row count in
// OUT_DIR/logits.f32 (float32, rows of vocab values, in the order written).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;

namespace {

int argmax(const float* v, int n) { return static_cast<int>(std::max_element(v, v + n) - v); }

std::unique_ptr<ember_ir::Interpreter> make(const std::string& ir, const GGUFData& g, const char* evict) {
  if (evict)
    setenv("EMBER_EVICT", evict, 1);
  else
    unsetenv("EMBER_EVICT");
  auto m = std::make_unique<ember_ir::Interpreter>(ir, g, false);
  unsetenv("EMBER_EVICT");
  return m;
}

struct Sink {  // logits blocks, appended to one file
  std::ofstream f;
  long rows = 0;
  int V = 0;
  json put(const std::vector<float>& v) {
    if (v.size() % V) throw std::runtime_error("logits not a whole number of rows");
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!f) throw std::runtime_error("cannot write logits");
    const long n = static_cast<long>(v.size() / V);
    json b = {rows, n};
    rows += n;
    return b;
  }
};

std::vector<float> step_all(ember_ir::Interpreter& m, const std::vector<int>& seq, int from, int V) {
  std::vector<float> out;
  for (size_t t = from; t < seq.size(); ++t) {
    const std::vector<float> l = m.forward({seq[t]}, static_cast<int>(t), false);
    out.insert(out.end(), l.end() - V, l.end());
  }
  return out;
}

std::vector<std::vector<int>> turns(const std::vector<int>& seq, int user) {
  std::vector<std::vector<int>> out;
  for (size_t i = 0; i < seq.size(); ++i) {
    if (out.empty() || (seq[i] == user && !out.back().empty())) out.emplace_back();
    out.back().push_back(seq[i]);
  }
  return out;
}

std::vector<int> greedy(ember_ir::Interpreter& m, const std::vector<int>& prompt, int gen, int V) {
  std::vector<int> out;
  std::vector<float> l = m.forward(prompt, 0, false);
  int pos = static_cast<int>(prompt.size());
  for (int g = 0; g < gen; ++g) {
    const int tok = argmax(l.data() + l.size() - V, V);
    out.push_back(tok);
    if (g + 1 < gen) l = m.forward({tok}, pos++, false);
  }
  return out;
}

bool refused(ember_ir::Interpreter& m, const std::vector<int>& seq, int n) {
  try {
    m.forward(std::vector<int>(seq.begin() + n, seq.end()), n, true);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: state_wasm_gate MODEL.gguf INPUT.json OUT_DIR\n";
    return 2;
  }
  try {
    const auto t0 = std::chrono::steady_clock::now();
#ifdef __EMSCRIPTEN__
    const GGUFData gguf = load_gguf_meta(argv[1], false);  // as the page: no mmap
#else
    const GGUFData gguf = load_gguf_meta(argv[1], true);
#endif
    const std::string ir = ember_ir::find_ir(gguf);
    if (ir.empty()) throw std::runtime_error("no IR in " + std::string(argv[1]));
    json input;
    {
      std::ifstream f(argv[2]);
      if (!f) throw std::runtime_error("cannot read " + std::string(argv[2]));
      input = json::parse(f);
    }
    const std::string out = argv[3];
    const std::vector<std::vector<int>> seqs = input.value("seqs", std::vector<std::vector<int>>{});
    const std::vector<int> plen = input.value("prompt_len", std::vector<int>{});
    const int user = input.value("user_id", -1);
    auto on = make(ir, gguf, nullptr), off = make(ir, gguf, "0");
    const int V = on->vocab();
    Sink sink;
    sink.V = V;
    sink.f.open(out + "/logits.f32", std::ios::binary | std::ios::trunc);
    json run = {{"gguf", argv[1]},
#ifdef __EMSCRIPTEN__
                {"engine", "wasm"},
#else
                {"engine", "native"},
#endif
                {"vocab", V},
                {"state_k", on->state_k()},
                {"turn_state", on->turn_state()},
                {"evicts_on", on->evicts()},
                {"evicts_off", off->evicts()}};
    json rows = json::array();
    for (size_t k = 0; k < seqs.size(); ++k) {
      const std::vector<int>& seq = seqs[k];
      const int L = static_cast<int>(seq.size());
      json r = {{"i", k}, {"len", L}};
      r["prefill_on"] = sink.put(on->forward(seq, 0, true));
      r["prefill_off"] = sink.put(off->forward(seq, 0, true));
      r["decode_on"] = sink.put(step_all(*on, seq, 0, V));
      std::vector<std::vector<int>> chunks;
      if (user >= 0) {
        chunks = turns(seq, user);
        std::vector<float> tv;
        int p = 0;
        for (const auto& c : chunks) {
          const std::vector<float> l = on->forward(c, p, true);
          tv.insert(tv.end(), l.begin(), l.end());
          p += static_cast<int>(c.size());
        }
        r["turns"] = chunks.size();
        r["turns_on"] = sink.put(tv);
      }
      if (k < plen.size()) {
        const int P = plen[k];
        r["prompt_len"] = P;
        r["free_greedy"] = greedy(*on, std::vector<int>(seq.begin(), seq.begin() + P), L - P, V);
      }
      {
        const std::vector<int> ctx(seq.begin(), seq.end() - 1);
        on->forward(ctx, 0, false);
        off->forward(ctx, 0, false);
        r["kv_rows_held"] = {{"on", on->kv_rows_held()}, {"off", off->kv_rows_held()}};
      }
      if (on->evicts() && user >= 0 && chunks.size() >= 2) {
        int p = 0;
        for (const auto& c : chunks) {
          on->forward(c, p, true);
          p += static_cast<int>(c.size());
        }
        json rw = json::array();
        const int t1 = static_cast<int>(chunks[0].size());
        for (const int n : {3, t1 + 3}) {
          if (n >= L) continue;
          json x = {{"n", n}};
          const int rp = on->resume_point(n);
          x["resume"] = rp;
          if (rp != n) x["direct_refused"] = refused(*on, seq, n);
          const std::vector<float> l = on->forward(std::vector<int>(seq.begin() + rp, seq.end()), rp, true);
          x["tail"] = sink.put(std::vector<float>(l.begin() + static_cast<size_t>(n - rp) * V, l.end()));
          x["kv_rows_held"] = on->kv_rows_held();
          rw.push_back(x);
        }
        r["rewinds"] = rw;
      }
      rows.push_back(r);
    }
    run["rows"] = rows;
    if (input.contains("calls")) {
      auto s_on = make(ir, gguf, nullptr), s_off = make(ir, gguf, "0");
      auto run_call = [&](ember_ir::Interpreter& m, const std::vector<int>& seq, int from, bool step, int& resumed) {
        const int p = m.resume_point(from);
        resumed = p;
        std::vector<float> v;
        if (step) {
          for (int t = p; t < static_cast<int>(seq.size()); ++t) {
            const std::vector<float> lg = m.forward({seq[t]}, t, false);
            if (t >= from) v.insert(v.end(), lg.end() - V, lg.end());
          }
        } else {
          const std::vector<float> all = m.forward(std::vector<int>(seq.begin() + p, seq.end()), p, true);
          v.assign(all.begin() + static_cast<size_t>(from - p) * V, all.end());
        }
        return v;
      };
      json calls = json::array();
      for (const json& c : input.at("calls")) {
        const std::vector<int> seq = c.at("seq").get<std::vector<int>>();
        const int from = c.at("from").get<int>();
        const bool step = c.value("step", false);
        json r = {{"from", from}, {"length", seq.size()}, {"step", step}};
        if (s_on->resume_point(from) != from) r["direct_refused"] = refused(*s_on, seq, from);
        int ron = 0, roff = 0;
        r["on"] = sink.put(run_call(*s_on, seq, from, step, ron));
        r["off"] = sink.put(run_call(*s_off, seq, from, step, roff));
        auto fresh = make(ir, gguf, nullptr);
        const std::vector<float> f_all = fresh->forward(seq, 0, true);
        r["fresh"] = sink.put(std::vector<float>(f_all.begin() + static_cast<size_t>(from) * V, f_all.end()));
        r["resumed_at"] = {{"on", ron}, {"off", roff}};
        r["kv_rows_held"] = {{"on", s_on->kv_rows_held()}, {"off", s_off->kv_rows_held()}};
        calls.push_back(r);
      }
      run["calls"] = calls;
    }
    sink.f.close();
    run["logit_rows"] = sink.rows;
    run["ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    {
      std::ofstream f(out + "/run.json.tmp");
      f << run.dump(1) << "\n";
      if (!f) throw std::runtime_error("cannot write run.json");
    }
    std::rename((out + "/run.json.tmp").c_str(), (out + "/run.json").c_str());
    std::cout << json{{"engine", run["engine"]}, {"sequences", seqs.size()}, {"logit_rows", sink.rows},
                      {"ms", run["ms"]}}.dump()
              << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "state_wasm_gate: " << e.what() << "\n";
    return 1;
  }
}
