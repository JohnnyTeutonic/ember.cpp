// mor_device_gate: Mixture-of-Recursions (depth_route, causal token choice, kv_rows "active"; IR_SPEC 7.4,
// 7.5) on ember's CUDA device path against the same interpreter on the CPU, the EXACT reference, on given
// sequences. CPU and device run in one process.
//
//   mor_device_gate MODEL.gguf [--tol 1e-4] [--lp-tol 1e-5] [--bench GEN] < input.json > report.json
//
// input: {"prompt": [[ids], ...], "continuation": [[ids], ...],
//         "depth": [[d, ...], ...] (optional: a reference depth trace per context position, e.g. mtstudio's),
//         "logprobs": [[lp, ...], ...] (optional: reference log-probabilities of the continuation)}
// The context of row k is prompt + continuation without its last id (L ids), as experiments/
// recursion_2026-10-23/mor_gate.py scores it. Per row:
//   - logits of every context position: the device's one-call prefill, its token-by-token decode and a
//     two-call prefill (split at L / 2), each against the CPU's prefill: max |dlogit| over the CPU row's
//     range, worst over positions, and argmax agreement at every position; decode vs the device's prefill;
//   - depth trace (last_depths): device prefill, decode (one depth per step) and two-call prefill equal
//     to the CPU's prefill trace (and the CPU's decode trace), and to the reference trace when given;
//   - kv rows held: every application's positions (held_positions) after each of the three runs equal to
//     the CPU's after its prefill;
//   - log-probabilities of the continuation from the device's prefill vs the CPU's, and vs the reference
//     when given: max |lp - ref| / max(1, max |ref|);
//   - greedy: the argmax at every scored position equal to the continuation (device and CPU), and free
//     running: prefill the prompt, decode len(continuation) ids, device equal to CPU, and each against the
//     continuation;
//   - rewinds, after the device's and the CPU's full prefill, each done identically on both: to n = L / 2
//     as one call and to n = L - 3 token by token, first with an altered suffix (each id replaced by
//     (id + 1) mod vocab), then with the original one; each call's logits and depths against the CPU's
//     same call, and rows held equal after each; the original suffix also against the CPU's fresh
//     prefill rows n ...
// --bench GEN: wall time of a prefill of the first prompt and GEN greedy decode steps, CPU and device
//   (a warm-up run first on each).
// Pass: every logit comparison within --tol (ROUNDING), log-probabilities within --lp-tol, every argmax,
// depth, rows-held and greedy count complete. Exit 1 otherwise.
#include <algorithm>
#include <chrono>
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
  for (size_t r = 0; r < ref.size() / V && r < a.size() / V; ++r)
    n += argmax(a.data() + r * V, V) == argmax(ref.data() + r * V, V);
  return n;
}

// Token-by-token from position `from`: each step's logits, and its depth appended to `depths`.
std::vector<float> step_all(ember_ir::Interpreter& m, const std::vector<int>& seq, int from, int V,
                            std::vector<int>& depths) {
  std::vector<float> out;
  for (size_t t = from; t < seq.size(); ++t) {
    const std::vector<float> l = m.forward({seq[t]}, static_cast<int>(t), false);
    out.insert(out.end(), l.end() - V, l.end());
    depths.push_back(m.last_depths().empty() ? -1 : m.last_depths().at(0));
  }
  return out;
}

std::vector<std::vector<int>> held(const ember_ir::Interpreter& m) {
  std::vector<std::vector<int>> h;
  for (int l = 0; l < m.n_layers(); ++l) h.push_back(m.held_positions(l));
  return h;
}

// Log-probabilities of cont from all-row logits whose row P - 1 + j predicts cont[j].
std::vector<double> logprobs(const std::vector<float>& rows, int P, const std::vector<int>& cont, int V) {
  std::vector<double> lp;
  for (size_t j = 0; j < cont.size(); ++j) {
    const float* r = rows.data() + static_cast<size_t>(P - 1 + j) * V;
    double mx = r[0];
    for (int i = 1; i < V; ++i) mx = std::max(mx, static_cast<double>(r[i]));
    double z = 0.0;
    for (int i = 0; i < V; ++i) z += std::exp(static_cast<double>(r[i]) - mx);
    lp.push_back(static_cast<double>(r[cont[j]]) - (mx + std::log(z)));
  }
  return lp;
}

double lp_rel(const std::vector<double>& a, const std::vector<double>& ref) {
  if (a.size() != ref.size()) return INFINITY;
  double d = 0.0, m = 1.0;
  for (size_t j = 0; j < a.size(); ++j) {
    d = std::max(d, std::fabs(a[j] - ref[j]));
    m = std::max(m, std::fabs(ref[j]));
  }
  return d / m;
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

struct Worst {
  double v = 0.0;
  void add(double x) { v = std::max(v, x); }
};

std::string frac(int a, int b) { return std::to_string(a) + "/" + std::to_string(b); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: mor_device_gate MODEL.gguf [--tol 1e-4] [--lp-tol 1e-5] [--bench GEN] < input.json\n";
    return 2;
  }
  double tol = 1e-4, lp_tol = 1e-5;
  int bench = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--tol")
      tol = std::atof(argv[i + 1]);
    else if (k == "--lp-tol")
      lp_tol = std::atof(argv[i + 1]);
    else if (k == "--bench")
      bench = std::atoi(argv[i + 1]);
    else {
      std::cerr << "unknown option " << k << "\n";
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
    const auto prompts = input.at("prompt").get<std::vector<std::vector<int>>>();
    const auto conts = input.at("continuation").get<std::vector<std::vector<int>>>();
    const auto ref_depth = input.value("depth", std::vector<std::vector<int>>{});
    const auto ref_lp = input.value("logprobs", std::vector<std::vector<double>>{});
    ember_ir::Interpreter cpu(ir, gguf, false), dev(ir, gguf, true);
    if (!dev.on_gpu()) throw std::runtime_error("the device interpreter is not on the GPU");
    if (!dev.has_depth_route()) throw std::runtime_error("the IR has no depth_route");
    const int V = cpu.vocab();
    Worst w_pre, w_dec, w_chunk, w_dec_pre, w_rew, w_rew_fresh, w_lp, w_lp_ref, w_lp_cpu_ref;
    int top1 = 0, positions = 0, depth_pre = 0, depth_dec = 0, depth_chunk = 0, depth_cpu_dec = 0, depth_ref = 0,
        held_ok = 0, tf_dev = 0, tf_cpu = 0, free_dev = 0, free_cpu = 0, free_eq = 0, rew = 0, rew_ok = 0;
    double held_rows = 0.0;
    json rows = json::array();
    const size_t N = prompts.size();
    for (size_t k = 0; k < N; ++k) {
      const std::vector<int>& prompt = prompts[k];
      const std::vector<int>& cont = conts.at(k);
      std::vector<int> seq = prompt;
      seq.insert(seq.end(), cont.begin(), cont.end() - 1);
      const int L = static_cast<int>(seq.size()), P = static_cast<int>(prompt.size());
      json r = {{"i", k}, {"len", L}};
      // The CPU: prefill (the reference), its decode trace.
      const std::vector<float> ref = cpu.forward(seq, 0, true);
      const std::vector<int> d_cpu = cpu.last_depths();
      const auto h_cpu = held(cpu);
      {
        ember_ir::Interpreter c2(ir, gguf, false);
        std::vector<int> dd;
        step_all(c2, seq, 0, V, dd);
        depth_cpu_dec += dd == d_cpu;
      }
      // The device: decode, a two-call prefill, then the one-call prefill (left in place for the rewinds).
      std::vector<int> d_dec;
      const std::vector<float> a_dec = step_all(dev, seq, 0, V, d_dec);
      const bool held_dec = held(dev) == h_cpu;
      const int c = L / 2;
      std::vector<float> a_chunk = dev.forward(std::vector<int>(seq.begin(), seq.begin() + c), 0, true);
      std::vector<int> d_chunk = dev.last_depths();
      {
        const std::vector<float> b = dev.forward(std::vector<int>(seq.begin() + c, seq.end()), c, true);
        a_chunk.insert(a_chunk.end(), b.begin(), b.end());
        d_chunk.insert(d_chunk.end(), dev.last_depths().begin(), dev.last_depths().end());
      }
      const bool held_chunk = held(dev) == h_cpu;
      const std::vector<float> a_pre = dev.forward(seq, 0, true);
      const std::vector<int> d_pre = dev.last_depths();
      const bool held_pre = held(dev) == h_cpu;
      const double e_pre = rel_rows(a_pre, ref, V), e_dec = rel_rows(a_dec, ref, V), e_chunk = rel_rows(a_chunk, ref, V);
      const double e_dec_pre = rel_rows(a_dec, a_pre, V);
      w_pre.add(e_pre);
      w_dec.add(e_dec);
      w_chunk.add(e_chunk);
      w_dec_pre.add(e_dec_pre);
      const int agree = argmax_agree(a_pre, ref, V) + argmax_agree(a_dec, ref, V) + argmax_agree(a_chunk, ref, V);
      top1 += agree;
      positions += 3 * L;
      depth_pre += d_pre == d_cpu;
      depth_dec += d_dec == d_cpu;
      depth_chunk += d_chunk == d_cpu;
      const bool dref = k < ref_depth.size() && ref_depth[k] == d_cpu && ref_depth[k] == d_pre;
      depth_ref += dref;
      held_ok += held_pre && held_dec && held_chunk;
      for (const auto& h : h_cpu) held_rows += static_cast<double>(h.size()) / h_cpu.size();
      // Log-probabilities and greedy.
      const std::vector<double> lp_dev = logprobs(a_pre, P, cont, V), lp_cpu = logprobs(ref, P, cont, V);
      const double e_lp = lp_rel(lp_dev, lp_cpu);
      w_lp.add(e_lp);
      if (k < ref_lp.size()) {
        w_lp_ref.add(lp_rel(lp_dev, ref_lp[k]));
        w_lp_cpu_ref.add(lp_rel(lp_cpu, ref_lp[k]));
      }
      bool tfd = true, tfc = true;
      for (size_t j = 0; j < cont.size(); ++j) {
        tfd = tfd && argmax(a_pre.data() + static_cast<size_t>(P - 1 + j) * V, V) == cont[j];
        tfc = tfc && argmax(ref.data() + static_cast<size_t>(P - 1 + j) * V, V) == cont[j];
      }
      tf_dev += tfd;
      tf_cpu += tfc;
      // Rewinds (the device holds the one-call prefill; the CPU its prefill).
      json rw = json::array();
      for (int mode = 0; mode < 2; ++mode) {
        const int n = mode == 0 ? L / 2 : std::max(1, L - 3);
        if (n <= 0 || n >= L) continue;
        for (int alt = 1; alt >= 0; --alt) {
          std::vector<int> s2 = seq;
          if (alt)
            for (int t = n; t < L; ++t) s2[t] = (s2[t] + 1) % V;
          std::vector<float> a, b;
          std::vector<int> da, db;
          if (mode == 0) {
            a = dev.forward(std::vector<int>(s2.begin() + n, s2.end()), n, true);
            da = dev.last_depths();
            b = cpu.forward(std::vector<int>(s2.begin() + n, s2.end()), n, true);
            db = cpu.last_depths();
          } else {
            a = step_all(dev, s2, n, V, da);
            b = step_all(cpu, s2, n, V, db);
          }
          const double e = rel_rows(a, b, V);
          w_rew.add(e);
          bool ok = e <= tol && argmax_agree(a, b, V) == L - n && da == db && held(dev) == held(cpu);
          json one = {{"n", n}, {"one_call", mode == 0}, {"altered", alt == 1}, {"vs_cpu_rel", e}};
          if (!alt) {
            const std::vector<float> fresh(ref.begin() + static_cast<size_t>(n) * V, ref.end());
            const double ef = rel_rows(a, fresh, V);
            w_rew_fresh.add(ef);
            one["vs_fresh_cpu_rel"] = ef;
            ok = ok && ef <= tol && held(dev) == h_cpu;
          }
          one["pass"] = ok;
          rw.push_back(one);
          ++rew;
          rew_ok += ok;
        }
      }
      // Free-running greedy.
      const int gen = static_cast<int>(cont.size());
      const std::vector<int> g_dev = greedy(dev, prompt, gen, V), g_cpu = greedy(cpu, prompt, gen, V);
      free_dev += g_dev == cont;
      free_cpu += g_cpu == cont;
      free_eq += g_dev == g_cpu;
      r["prefill_rel"] = e_pre;
      r["decode_rel"] = e_dec;
      r["two_call_rel"] = e_chunk;
      r["decode_vs_prefill_rel"] = e_dec_pre;
      r["logprob_rel_vs_cpu"] = e_lp;
      r["top1"] = frac(agree, 3 * L);
      r["depth"] = {{"cpu", d_cpu}, {"device_prefill_equal", d_pre == d_cpu}, {"device_decode_equal", d_dec == d_cpu},
                    {"device_two_call_equal", d_chunk == d_cpu}, {"reference_equal", dref}};
      std::vector<int> nrows;
      for (const auto& h : h_cpu) nrows.push_back(static_cast<int>(h.size()));
      r["rows_held_cpu"] = nrows;
      r["rows_held_equal"] = {{"prefill", held_pre}, {"decode", held_dec}, {"two_call", held_chunk}};
      r["rewinds"] = rw;
      rows.push_back(r);
    }
    const int n = static_cast<int>(N);
    json summary = {{"rows", n},
                    {"top1_vs_cpu", frac(top1, positions)},
                    {"prefill_worst_rel", w_pre.v},
                    {"decode_worst_rel", w_dec.v},
                    {"two_call_worst_rel", w_chunk.v},
                    {"decode_vs_prefill_worst_rel", w_dec_pre.v},
                    {"logprob_worst_rel_vs_cpu", w_lp.v},
                    {"depth_prefill", frac(depth_pre, n)},
                    {"depth_decode", frac(depth_dec, n)},
                    {"depth_two_call", frac(depth_chunk, n)},
                    {"depth_cpu_decode_equal_cpu_prefill", frac(depth_cpu_dec, n)},
                    {"rows_held_equal_cpu", frac(held_ok, n)},
                    {"rows_held_mean_per_application", n ? held_rows / n : 0.0},
                    {"greedy_teacher_forced_device", frac(tf_dev, n)},
                    {"greedy_teacher_forced_cpu", frac(tf_cpu, n)},
                    {"greedy_free_running_device", frac(free_dev, n)},
                    {"greedy_free_running_cpu", frac(free_cpu, n)},
                    {"greedy_free_running_device_equal_cpu", frac(free_eq, n)},
                    {"rewinds", frac(rew_ok, rew)},
                    {"rewind_worst_rel_vs_cpu", w_rew.v},
                    {"rewind_worst_rel_vs_fresh_cpu", w_rew_fresh.v}};
    if (!ref_depth.empty()) summary["depth_reference"] = frac(depth_ref, n);
    if (!ref_lp.empty()) {
      summary["logprob_worst_rel_device_vs_reference"] = w_lp_ref.v;
      summary["logprob_worst_rel_cpu_vs_reference"] = w_lp_cpu_ref.v;
    }
    bool pass = w_pre.v <= tol && w_dec.v <= tol && w_chunk.v <= tol && w_dec_pre.v <= tol && w_rew.v <= tol &&
                w_rew_fresh.v <= tol && w_lp.v <= lp_tol && top1 == positions && depth_pre == n && depth_dec == n &&
                depth_chunk == n && depth_cpu_dec == n && held_ok == n && tf_dev == tf_cpu && free_dev == free_cpu &&
                free_eq == n && rew_ok == rew;
    if (!ref_depth.empty()) pass = pass && depth_ref == n;
    if (!ref_lp.empty()) pass = pass && w_lp_ref.v <= lp_tol;
    json report = {{"gguf", argv[1]}, {"on_gpu", dev.on_gpu()}, {"tol", tol}, {"lp_tol", lp_tol}, {"summary", summary}};
    if (bench > 0 && !prompts.empty()) {
      json b;
      for (int side = 0; side < 2; ++side) {
        ember_ir::Interpreter& m = side ? dev : cpu;
        greedy(m, prompts[0], bench, V);  // warm-up
        const auto t0 = std::chrono::steady_clock::now();
        m.forward(prompts[0], 0, false);
        const auto t1 = std::chrono::steady_clock::now();
        greedy(m, prompts[0], bench, V);
        const auto t2 = std::chrono::steady_clock::now();
        const double pre = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double all = std::chrono::duration<double, std::milli>(t2 - t1).count();
        b[side ? "device" : "cpu"] = {{"prompt_len", prompts[0].size()}, {"prefill_ms", pre},
                                      {"decode_ms_per_token", (all - pre) / std::max(1, bench - 1)}};
      }
      report["bench"] = b;
    }
    report["pass"] = pass;
    report["rows_detail"] = rows;
    std::cout << report.dump(1) << "\n";
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "mor_device_gate: " << e.what() << "\n";
    return 1;
  }
}
