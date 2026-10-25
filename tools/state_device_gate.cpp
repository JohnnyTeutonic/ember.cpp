// state_device_gate: ember's IR interpreter on the CUDA device against the same interpreter on the
// CPU, on given token sequences (state ids included), for per-turn state tokens (IR_SPEC 6.8, 6.9) and
// any other IR the device runs. CPU and device run in one process; eviction is switched with
// EMBER_EVICT before each interpreter is built, as tools/state_tokens_test.cpp does.
//
//   state_device_gate MODEL.gguf [--tol 1e-4] < input.json > report.json
//
// input: {"seqs": [[ids], ...], "prompt_len": [P, ...] (optional), "user_id": 2 (optional),
//         "calls": [{"seq": [ids], "from": p, "step": false}, ...] (optional)}
//
// Per sequence ("on" = the IR's own eviction setting, "off" = EMBER_EVICT=0):
//   - every position's logits: the device's one-call prefill (on and off), its token-by-token decode (the
//     captured CUDA graph after the first step) and, with user_id, its turn-by-turn prefill (one call per
//     turn, split before each user_id), each against the CPU's prefill (on): max |dlogit| over the CPU
//     row's range, worst over positions; argmax agreement at every position; device on vs off; device
//     decode vs device prefill;
//   - with prompt_len: the device's argmax at every scored position (P-1 .. len-2) equal to the next id
//     (teacher-forced greedy), and free-running greedy (prefill the prompt, decode len - P tokens) equal
//     to the continuation, on the device and on the CPU;
//   - kv rows held after a prefill of the sequence less its last id (the context), device and CPU, on
//     and off;
//   - under eviction with user_id: rewinds into turn 0 (n = 3) and turn 1 (n = start(1) + 3) after the
//     turn-by-turn run: resume_point(n) equal on device and CPU, the device's direct forward at n
//     refused when the rows are gone, the re-prefill from resume_point(n) to the end against the CPU's
//     prefill (rows n ..), and kv rows held after it equal to the CPU's after the same rewind.
// calls: tools/state_tokens_test.cpp's script (each call resumes at resume_point(from) and runs seq from
//   there as one prefill or token by token) on the device on, the device off and the CPU on, plus a fresh
//   device prefill per call: resume points and rows held equal to the CPU's, logits of positions from ..
//   end against the CPU's, and device on vs off and vs fresh.
// Pass: every comparison within --tol (ROUNDING; the CPU is the EXACT reference), every argmax and greedy
// count complete, rows held and resume points equal, every direct rewind refused. Exit 1 otherwise.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
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

// max over rows of max |a - ref| / (max ref - min ref), rows of V logits.
double rel_rows(const std::vector<float>& a, const std::vector<float>& ref, int V) {
  if (a.size() != ref.size() || ref.size() % V != 0) return INFINITY;
  double worst = 0.0;
  for (size_t r = 0; r < ref.size() / V; ++r) {
    const float* x = a.data() + r * V;
    const float* y = ref.data() + r * V;
    float lo = y[0], hi = y[0];
    double d = 0.0;
    for (int i = 0; i < V; ++i) {
      lo = std::min(lo, y[i]);
      hi = std::max(hi, y[i]);
      d = std::max(d, std::fabs(static_cast<double>(x[i]) - y[i]));
    }
    worst = std::max(worst, hi > lo ? d / (hi - lo) : d);
  }
  return worst;
}

int argmax_agree(const std::vector<float>& a, const std::vector<float>& ref, int V) {
  int n = 0;
  for (size_t r = 0; r < ref.size() / V; ++r) n += argmax(a.data() + r * V, V) == argmax(ref.data() + r * V, V);
  return n;
}

std::unique_ptr<ember_ir::Interpreter> make(const std::string& ir, const GGUFData& g, bool gpu, const char* evict) {
  if (evict)
    setenv("EMBER_EVICT", evict, 1);
  else
    unsetenv("EMBER_EVICT");
  auto m = std::make_unique<ember_ir::Interpreter>(ir, g, gpu);
  unsetenv("EMBER_EVICT");
  if (gpu && !m->on_gpu()) throw std::runtime_error("the device interpreter is not on the GPU");
  return m;
}

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

// gen greedy ids after prompt (prefill, then decode).
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

struct Worst {
  double v = 0.0;
  void add(double x) { v = std::max(v, x); }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: state_device_gate MODEL.gguf [--tol 1e-4] < input.json\n";
    return 2;
  }
  double tol = 1e-4;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::string(argv[i]) == "--tol")
      tol = std::atof(argv[i + 1]);
    else {
      std::cerr << "unknown option " << argv[i] << "\n";
      return 2;
    }
  }
  try {
    const GGUFData gguf = load_gguf_meta(argv[1], true);
    const std::string ir = ember_ir::find_ir(gguf);
    if (ir.empty() || !ember_ir::device_supported(ir)) {
      std::cout << json{{"device_supported", false}, {"pass", false}}.dump(1) << "\n";
      return 1;
    }
    const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    const json input = json::parse(in);
    const std::vector<std::vector<int>> seqs = input.value("seqs", std::vector<std::vector<int>>{});
    const std::vector<int> plen = input.value("prompt_len", std::vector<int>{});
    const int user = input.value("user_id", -1);
    auto cpu_on = make(ir, gguf, false, nullptr), cpu_off = make(ir, gguf, false, "0");
    auto dev_on = make(ir, gguf, true, nullptr), dev_off = make(ir, gguf, true, "0");
    const int V = cpu_on->vocab();
    const bool evicts = dev_on->evicts();
    json report = {{"gguf", argv[1]},
                   {"on_gpu", dev_on->on_gpu()},
                   {"state_k", dev_on->state_k()},
                   {"turn_state", dev_on->turn_state()},
                   {"evicts_on", evicts},
                   {"evicts_off", dev_off->evicts()},
                   {"cpu_evicts_on", cpu_on->evicts()},
                   {"tol", tol}};
    bool pass = cpu_on->evicts() == evicts && !dev_off->evicts();
    Worst w_on, w_off, w_dec, w_dec_pre, w_turns, w_on_off, w_rewind;
    int top1 = 0, positions = 0, tf_dev = 0, tf_cpu = 0, free_dev = 0, free_cpu = 0, rows_ok = 0, rewinds = 0,
        rewinds_ok = 0, refused = 0, refusals_expected = 0;
    json rows = json::array();
    for (size_t k = 0; k < seqs.size(); ++k) {
      const std::vector<int>& seq = seqs[k];
      const int L = static_cast<int>(seq.size());
      json r = {{"i", k}, {"len", L}};
      const std::vector<float> ref = cpu_on->forward(seq, 0, true);
      const std::vector<float> a_on = dev_on->forward(seq, 0, true), a_off = dev_off->forward(seq, 0, true);
      const std::vector<float> dec = step_all(*dev_on, seq, 0, V);
      const double e_on = rel_rows(a_on, ref, V), e_off = rel_rows(a_off, ref, V), e_dec = rel_rows(dec, ref, V);
      const double e_dec_pre = rel_rows(dec, a_on, V), e_on_off = rel_rows(a_on, a_off, V);
      w_on.add(e_on);
      w_off.add(e_off);
      w_dec.add(e_dec);
      w_dec_pre.add(e_dec_pre);
      w_on_off.add(e_on_off);
      const int agree = argmax_agree(a_on, ref, V) + argmax_agree(a_off, ref, V) + argmax_agree(dec, ref, V);
      top1 += agree;
      positions += 3 * L;
      r["prefill_on_rel"] = e_on;
      r["prefill_off_rel"] = e_off;
      r["decode_rel"] = e_dec;
      r["decode_vs_prefill_rel"] = e_dec_pre;
      r["on_vs_off_rel"] = e_on_off;
      r["top1"] = std::to_string(agree) + "/" + std::to_string(3 * L);
      std::vector<std::vector<int>> chunks;
      if (user >= 0) {
        chunks = turns(seq, user);
        std::vector<float> tv;
        int p = 0;
        for (const auto& c : chunks) {
          const std::vector<float> l = dev_on->forward(c, p, true);
          tv.insert(tv.end(), l.begin(), l.end());
          p += static_cast<int>(c.size());
        }
        const double e_t = rel_rows(tv, ref, V);
        w_turns.add(e_t);
        r["turns"] = chunks.size();
        r["turns_rel"] = e_t;
      }
      if (k < plen.size()) {
        const int P = plen[k];
        bool tfd = true, tfc = true;
        for (int t = P - 1; t + 1 < L; ++t) {
          tfd = tfd && argmax(a_on.data() + static_cast<size_t>(t) * V, V) == seq[t + 1];
          tfc = tfc && argmax(ref.data() + static_cast<size_t>(t) * V, V) == seq[t + 1];
        }
        const std::vector<int> prompt(seq.begin(), seq.begin() + P), cont(seq.begin() + P, seq.end());
        const bool fd = greedy(*dev_on, prompt, L - P, V) == cont, fc = greedy(*cpu_on, prompt, L - P, V) == cont;
        tf_dev += tfd;
        tf_cpu += tfc;
        free_dev += fd;
        free_cpu += fc;
        r["teacher_forced_greedy"] = {{"device", tfd}, {"cpu", tfc}};
        r["free_greedy"] = {{"device", fd}, {"cpu", fc}};
      }
      {  // rows held after the context (the sequence less its last id)
        const std::vector<int> ctx(seq.begin(), seq.end() - 1);
        dev_on->forward(ctx, 0, false);
        dev_off->forward(ctx, 0, false);
        cpu_on->forward(ctx, 0, false);
        cpu_off->forward(ctx, 0, false);
        const json held = {{"device_on", dev_on->kv_rows_held()},
                           {"device_off", dev_off->kv_rows_held()},
                           {"cpu_on", cpu_on->kv_rows_held()},
                           {"cpu_off", cpu_off->kv_rows_held()}};
        r["kv_rows_held"] = held;
        rows_ok += held["device_on"] == held["cpu_on"] && held["device_off"] == held["cpu_off"];
      }
      if (evicts && user >= 0 && chunks.size() >= 2) {
        // The turn-by-turn state on both engines, then rewinds into turn 0 and turn 1.
        int p = 0;
        for (const auto& c : chunks) {
          dev_on->forward(c, p, true);
          cpu_on->forward(c, p, true);
          p += static_cast<int>(c.size());
        }
        json rw = json::array();
        const int t1 = static_cast<int>(chunks[0].size());
        for (const int n : {3, t1 + 3}) {
          if (n >= L) continue;
          ++rewinds;
          json x = {{"n", n}};
          const int rd = dev_on->resume_point(n), rc = cpu_on->resume_point(n);
          x["resume_device"] = rd;
          x["resume_cpu"] = rc;
          bool ok = rd == rc;
          if (rd != n) {
            ++refusals_expected;
            bool threw = false;
            try {
              dev_on->forward(std::vector<int>(seq.begin() + n, seq.end()), n, true);
            } catch (const std::runtime_error&) {
              threw = true;
            }
            refused += threw;
            x["direct_refused"] = threw;
            ok = ok && threw;
          }
          const std::vector<float> ld = dev_on->forward(std::vector<int>(seq.begin() + rd, seq.end()), rd, true);
          const std::vector<float> lc = cpu_on->forward(std::vector<int>(seq.begin() + rc, seq.end()), rc, true);
          const std::vector<float> d_tail(ld.begin() + static_cast<size_t>(n - rd) * V, ld.end());
          const std::vector<float> ref_tail(ref.begin() + static_cast<size_t>(n) * V, ref.end());
          const double e = rel_rows(d_tail, ref_tail, V);
          w_rewind.add(e);
          x["rel"] = e;
          x["top1"] = argmax_agree(d_tail, ref_tail, V) == L - n;
          x["kv_rows_held"] = {{"device", dev_on->kv_rows_held()}, {"cpu", cpu_on->kv_rows_held()}};
          ok = ok && x["top1"].get<bool>() && dev_on->kv_rows_held() == cpu_on->kv_rows_held() && e <= tol;
          (void)lc;
          rewinds_ok += ok;
          x["pass"] = ok;
          rw.push_back(x);
        }
        r["rewinds"] = rw;
      }
      rows.push_back(r);
    }
    const int n = static_cast<int>(seqs.size());
    json summary = {{"sequences", n},
                    {"worst_prefill_on_rel", w_on.v},
                    {"worst_prefill_off_rel", w_off.v},
                    {"worst_decode_rel", w_dec.v},
                    {"worst_decode_vs_prefill_rel", w_dec_pre.v},
                    {"worst_on_vs_off_rel", w_on_off.v},
                    {"worst_turns_rel", w_turns.v},
                    {"top1_agree", std::to_string(top1) + "/" + std::to_string(positions)},
                    {"kv_rows_held_equal", std::to_string(rows_ok) + "/" + std::to_string(n)}};
    pass = pass && top1 == positions && rows_ok == n && w_on.v <= tol && w_off.v <= tol && w_dec.v <= tol &&
           w_dec_pre.v <= tol && w_on_off.v <= tol && w_turns.v <= tol;
    if (!plen.empty()) {
      const int m = static_cast<int>(std::min(plen.size(), seqs.size()));
      summary["teacher_forced_greedy"] = {{"device", std::to_string(tf_dev) + "/" + std::to_string(m)},
                                          {"cpu", std::to_string(tf_cpu) + "/" + std::to_string(m)}};
      summary["free_greedy"] = {{"device", std::to_string(free_dev) + "/" + std::to_string(m)},
                                {"cpu", std::to_string(free_cpu) + "/" + std::to_string(m)}};
      pass = pass && tf_dev == m && free_dev == m;
    }
    if (rewinds > 0) {
      summary["rewinds"] = std::to_string(rewinds_ok) + "/" + std::to_string(rewinds);
      summary["direct_rewinds_refused"] = std::to_string(refused) + "/" + std::to_string(refusals_expected);
      summary["worst_rewind_rel"] = w_rewind.v;
      pass = pass && rewinds_ok == rewinds && refused == refusals_expected;
    }
    if (input.contains("calls")) {  // tools/state_tokens_test.cpp's script, device vs CPU
      auto s_on = make(ir, gguf, true, nullptr), s_off = make(ir, gguf, true, "0");
      auto s_cpu = make(ir, gguf, false, nullptr);
      json calls = json::array();
      Worst wc_cpu, wc_off, wc_fresh;
      int ok_calls = 0, call_refusals = 0, call_refusals_expected = 0;
      auto run_call = [&](ember_ir::Interpreter& m, const std::vector<int>& seq, int from, bool step, int& resumed) {
        const int p = m.resume_point(from);
        resumed = p;
        std::vector<float> v;
        if (step) {
          for (int t = p; t < static_cast<int>(seq.size()); ++t) {
            const std::vector<float> lg = m.forward({seq[t]}, t, false);
            if (t >= from) v.insert(v.end(), lg.begin(), lg.end());
          }
        } else {
          const std::vector<float> all = m.forward(std::vector<int>(seq.begin() + p, seq.end()), p, true);
          v.assign(all.begin() + static_cast<size_t>(from - p) * V, all.end());
        }
        return v;
      };
      for (const json& c : input.at("calls")) {
        const std::vector<int> seq = c.at("seq").get<std::vector<int>>();
        const int from = c.at("from").get<int>();
        const bool step = c.value("step", false);
        json r = {{"from", from}, {"length", seq.size()}, {"step", step}};
        bool ok = true;
        if (s_on->resume_point(from) != from) {
          ++call_refusals_expected;
          bool threw = false;
          try {
            s_on->forward(std::vector<int>(seq.begin() + from, seq.end()), from, true);
          } catch (const std::runtime_error&) {
            threw = true;
          }
          call_refusals += threw;
          r["direct_rewind_refused"] = threw;
          ok = ok && threw;
        }
        int ron = 0, roff = 0, rcpu = 0;
        const std::vector<float> a = run_call(*s_on, seq, from, step, ron);
        const std::vector<float> b = run_call(*s_off, seq, from, step, roff);
        const std::vector<float> cc = run_call(*s_cpu, seq, from, step, rcpu);
        auto fresh = make(ir, gguf, true, nullptr);
        const std::vector<float> f_all = fresh->forward(seq, 0, true);
        const std::vector<float> f(f_all.begin() + static_cast<size_t>(from) * V, f_all.end());
        const double e_cpu = rel_rows(a, cc, V), e_off = rel_rows(a, b, V), e_fresh = rel_rows(a, f, V);
        wc_cpu.add(e_cpu);
        wc_off.add(e_off);
        wc_fresh.add(e_fresh);
        r["resumed_at"] = {{"device_on", ron}, {"device_off", roff}, {"cpu_on", rcpu}};
        r["kv_rows_held"] = {{"device_on", s_on->kv_rows_held()}, {"device_off", s_off->kv_rows_held()},
                             {"cpu_on", s_cpu->kv_rows_held()}};
        r["rel_device_on_vs_cpu"] = e_cpu;
        r["rel_on_vs_off"] = e_off;
        r["rel_on_vs_fresh"] = e_fresh;
        const bool top = argmax_agree(a, cc, V) == static_cast<int>(cc.size() / V);
        r["top1"] = top;
        ok = ok && ron == rcpu && s_on->kv_rows_held() == s_cpu->kv_rows_held() && top && e_cpu <= tol &&
             e_off <= tol && e_fresh <= tol;
        r["pass"] = ok;
        ok_calls += ok;
        calls.push_back(r);
      }
      summary["calls"] = std::to_string(ok_calls) + "/" + std::to_string(calls.size());
      summary["calls_direct_rewinds_refused"] =
          std::to_string(call_refusals) + "/" + std::to_string(call_refusals_expected);
      summary["calls_worst_vs_cpu_rel"] = wc_cpu.v;
      summary["calls_worst_on_vs_off_rel"] = wc_off.v;
      summary["calls_worst_on_vs_fresh_rel"] = wc_fresh.v;
      report["calls"] = calls;
      pass = pass && ok_calls == static_cast<int>(calls.size());
    }
    report["summary"] = summary;
    report["pass"] = pass;
    report["rows"] = rows;
    std::cout << report.dump(1) << "\n";
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "state_device_gate: " << e.what() << "\n";
    return 1;
  }
}
