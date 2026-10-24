// state_tokens_test: per-turn state tokens (paperkiln IR_SPEC_DRAFT.md section 1) in ember's
// interpreter, without a trained model.
//
//   state_tokens_test template
//       The template rule (state_tokens.h) on fixed id sequences: k ids after an <|endoftext|>
//       that ends a reply, <|endoftext|> + k ids before a <|user|> after a reply that stopped
//       early, a given complete block kept, malformed or misplaced state ids refused.
//   EMBER_IR=graph.json state_tokens_test run MODEL.gguf < script.json
//       script: {"calls": [{"seq": [ids], "from": p, "step": false}, ...]}. Each call continues
//       the same sequence: the engine resumes at resume_point(from) and runs seq from there,
//       as one prefill (all rows) or one token at a time ("step"), as the generation loop does
//       after prefix reuse. Three interpreters run every call: eviction off (EMBER_EVICT=0),
//       eviction on (EMBER_EVICT=1) and, per call, a fresh one that prefills seq from 0. The
//       report gives, per call, the resume position and kv rows held of each engine, whether the
//       evicting one refused a forward straight into an evicted turn, and whether the logits of
//       positions from .. end are bit-equal between off, on and fresh.
//   Also checked at the start of "run": with EMBER_EVICT unset, the IR's own setting.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"
#include "state_tokens.h"

using json = nlohmann::json;

namespace {

int template_cases() {
  // vocab 10, k 3; <|endoftext|> 1, <|user|> 2, <|assistant|> 3; S_j = 10 + j.
  const int V = 10, K = 3, E = 1, U = 2, A = 3;
  struct Case {
    const char* name;
    std::vector<int> in, want;
    bool refused;
  };
  const std::vector<Case> cases = {
      {"closed_reply", {U, 5, A, 6, 7, E, U, 8, A}, {U, 5, A, 6, 7, E, 10, 11, 12, U, 8, A}, false},
      {"two_closed_replies", {U, 5, A, 6, E, U, 7, A, 8, E, U, 9, A},
       {U, 5, A, 6, E, 10, 11, 12, U, 7, A, 8, E, 10, 11, 12, U, 9, A}, false},
      {"reply_stopped_early", {U, 5, A, 6, 7, U, 8, A}, {U, 5, A, 6, 7, E, 10, 11, 12, U, 8, A}, false},
      {"empty_reply_stopped_early", {U, 5, A, U, 8, A}, {U, 5, A, E, 10, 11, 12, U, 8, A}, false},
      {"open_reply_at_end", {U, 5, A, 6}, {U, 5, A, 6}, false},
      {"closed_reply_at_end", {U, 5, A, 6, E}, {U, 5, A, 6, E, 10, 11, 12}, false},
      {"block_given", {U, 5, A, 6, E, 10, 11, 12, U, 8, A}, {U, 5, A, 6, E, 10, 11, 12, U, 8, A}, false},
      {"eot_outside_exchange", {5, 6, E, 7, U, 8, A}, {5, 6, E, 7, U, 8, A}, false},
      {"eot_after_user_only", {U, 5, E, U, 8, A}, {U, 5, E, U, 8, A}, false},
      {"block_incomplete", {U, 5, A, 6, E, 10, 11, U, 8, A}, {}, true},
      {"block_out_of_order", {U, 5, A, 6, E, 10, 12, 11, U, 8, A}, {}, true},
      {"block_cut_at_end", {U, 5, A, 6, E, 10, 11}, {}, true},
      {"state_id_after_text", {U, 5, A, 6, 10, 11, 12}, {}, true},
      {"state_id_after_unclosing_eot", {5, E, 10, 11, 12}, {}, true},
      {"state_id_past_k", {U, 5, A, 6, E, 13}, {}, true},
  };
  json out = json::object();
  bool ok = true;
  for (const Case& c : cases) {
    json r;
    try {
      const std::vector<int> got = ember::insert_state_blocks(c.in, V, K, E, U, A);
      r["out"] = got;
      r["pass"] = !c.refused && got == c.want;
    } catch (const std::invalid_argument& e) {
      r["refused"] = e.what();
      r["pass"] = c.refused;
    }
    ok = ok && r["pass"].get<bool>();
    out[c.name] = r;
  }
  out["pass"] = ok;
  std::cout << out.dump(1) << "\n";
  return ok ? 0 : 1;
}

struct Rows {
  std::vector<float> v;  // [n][V]
  int resumed_at = -1;
};

// One call on engine m: resume at resume_point(from), run seq[p..] (prefill or step), keep the
// rows of positions from .. end.
Rows run_call(ember_ir::Interpreter& m, const std::vector<int>& seq, int from, bool step, int V) {
  Rows r;
  const int p = m.resume_point(from);
  r.resumed_at = p;
  const int n = static_cast<int>(seq.size());
  if (step) {
    for (int t = p; t < n; ++t) {
      const std::vector<float> lg = m.forward({seq[t]}, t, false);
      if (t >= from) r.v.insert(r.v.end(), lg.begin(), lg.end());
    }
  } else {
    const std::vector<int> part(seq.begin() + p, seq.end());
    const std::vector<float> all = m.forward(part, p, true);
    r.v.assign(all.begin() + static_cast<size_t>(from - p) * V, all.end());
  }
  return r;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  double d = 0;
  for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(static_cast<double>(a[i]) - b[i]));
  return d;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int run(const char* path) {
  const GGUFData gguf = load_gguf_meta(path, true);
  const std::string ir = ember_ir::find_ir(gguf);
  const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const json script = json::parse(in);
  unsetenv("EMBER_EVICT");
  json report;
  {
    ember_ir::Interpreter m(ir, gguf, false);
    report["ir_default"] = {{"state_k", m.state_k()}, {"turn_state", m.turn_state()}, {"evicts", m.evicts()}};
  }
  setenv("EMBER_EVICT", "0", 1);
  ember_ir::Interpreter off(ir, gguf, false);
  setenv("EMBER_EVICT", "1", 1);
  ember_ir::Interpreter on(ir, gguf, false);
  setenv("EMBER_EVICT", "0", 1);
  if (off.evicts() || !on.evicts()) throw std::runtime_error("EMBER_EVICT did not switch eviction");
  const int V = off.vocab();
  json calls = json::array();
  bool all_equal = true;
  for (const json& c : script.at("calls")) {
    const std::vector<int> seq = c.at("seq").get<std::vector<int>>();
    const int from = c.at("from").get<int>();
    const bool step = c.value("step", false);
    json r;
    r["from"] = from;
    r["length"] = seq.size();
    r["step"] = step;
    // The evicting engine must refuse a forward straight into a turn whose rows it dropped.
    if (on.resume_point(from) != from) {
      bool refused = false;
      try {
        const std::vector<int> part(seq.begin() + from, seq.end());
        on.forward(part, from, true);
      } catch (const std::runtime_error& e) {
        refused = true;
        r["direct_rewind_message"] = e.what();
      }
      r["direct_rewind_refused"] = refused;
      all_equal = all_equal && refused;
    }
    const Rows a = run_call(off, seq, from, step, V);
    const Rows b = run_call(on, seq, from, step, V);
    ember_ir::Interpreter fresh(ir, gguf, false);
    const std::vector<float> f_all = fresh.forward(seq, 0, true);
    const std::vector<float> f(f_all.begin() + static_cast<size_t>(from) * V, f_all.end());
    r["resumed_at"] = {{"off", a.resumed_at}, {"on", b.resumed_at}};
    r["kv_rows_held"] = {{"off", off.kv_rows_held()}, {"on", on.kv_rows_held()}};
    r["off_eq_on"] = bit_equal(a.v, b.v);
    r["off_eq_fresh"] = bit_equal(a.v, f);
    r["on_eq_fresh"] = bit_equal(b.v, f);
    r["max_diff_off_on"] = max_diff(a.v, b.v);
    r["max_diff_off_fresh"] = max_diff(a.v, f);
    all_equal = all_equal && r["off_eq_on"].get<bool>() && r["off_eq_fresh"].get<bool>() && r["on_eq_fresh"].get<bool>();
    calls.push_back(r);
  }
  report["calls"] = calls;
  report["pass"] = all_equal;
  std::cout << report.dump(1) << "\n";
  return all_equal ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "template") return template_cases();
    if (argc == 3 && std::string(argv[1]) == "run") return run(argv[2]);
  } catch (const std::exception& e) {
    std::cerr << "state_tokens_test: " << e.what() << "\n";
    return 1;
  }
  std::cerr << "usage: state_tokens_test template | EMBER_IR=graph.json state_tokens_test run MODEL.gguf < script.json\n";
  return 2;
}
