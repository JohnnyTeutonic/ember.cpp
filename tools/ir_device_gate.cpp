// ir_device_gate: ember's IR interpreter on the CUDA device against the same interpreter on the CPU,
// on raw token ids (no tokenizer), for any GGUF ember_ir::find_ir gives an IR for (paperkiln.ir, the
// IR built from a third-party file's metadata, or EMBER_IR).
//
//   ir_device_gate MODEL.gguf [--n 40] [--len 24] [--gen 32] [--seed 0] [--tol 1e-4]
//
// For each of n seeded random prompts of len token ids: the prefill logits of every position on the
// device against the CPU (max |dlogit| over the CPU's range, per position) and their argmax, the
// device's token-by-token decode (the captured CUDA graph after the first step) against its own
// prefill, and gen greedy tokens on each. Prints a JSON summary; exit 1 unless every greedy
// continuation is identical, every argmax agrees and the worst difference is within --tol.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;

namespace {

int argmax(const float* v, int n) { return static_cast<int>(std::max_element(v, v + n) - v); }

double rel_row(const float* a, const float* ref, int n) {
  float lo = ref[0], hi = ref[0];
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    lo = std::min(lo, ref[i]);
    hi = std::max(hi, ref[i]);
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - ref[i]));
  }
  return hi > lo ? d / (hi - lo) : d;
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ir_device_gate MODEL.gguf [--n 40] [--len 24] [--gen 32] [--seed 0] [--tol 1e-4]\n";
    return 2;
  }
  int n = 40, len = 24, gen = 32;
  unsigned seed = 0;
  double tol = 1e-4;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--n") n = std::atoi(argv[i + 1]);
    else if (k == "--len") len = std::atoi(argv[i + 1]);
    else if (k == "--gen") gen = std::atoi(argv[i + 1]);
    else if (k == "--seed") seed = static_cast<unsigned>(std::atoi(argv[i + 1]));
    else if (k == "--tol") tol = std::atof(argv[i + 1]);
    else {
      std::cerr << "unknown option " << k << "\n";
      return 2;
    }
  }
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  if (ir.empty()) {
    std::cerr << "no IR: set EMBER_IR, use a GGUF with paperkiln.ir, or EMBER_IR_BUILD=1 for llama/mistral\n";
    return 1;
  }
  if (!ember_ir::device_supported(ir)) {
    std::cout << json{{"device_supported", false}, {"pass", false}}.dump(1) << "\n";
    return 1;
  }
  ember_ir::Interpreter cpu(ir, gguf, false), dev(ir, gguf, true);
  const int V = cpu.vocab();
  const json doc = json::parse(ir);
  const int hi = std::min<int>(V, static_cast<int>(gguf.tokenizer_tokens.empty() ? V : gguf.tokenizer_tokens.size()));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0, hi - 1);
  double worst = 0.0, worst_decode = 0.0;
  int top1 = 0, positions = 0, greedy_same = 0;
  json rows = json::array();
  for (int s = 0; s < n; ++s) {
    std::vector<int> ids(len);
    for (int& t : ids) t = pick(rng);
    const std::vector<float> a = cpu.forward(ids, 0, true), b = dev.forward(ids, 0, true);
    double w = 0.0, wd = 0.0;
    int agree = 0;
    for (int t = 0; t < len; ++t) {
      const float* ra = a.data() + static_cast<size_t>(t) * V;
      const float* rb = b.data() + static_cast<size_t>(t) * V;
      w = std::max(w, rel_row(rb, ra, V));
      agree += argmax(ra, V) == argmax(rb, V);
    }
    for (int t = 0; t < len; ++t) {  // the device's decode (graph replays) against the CPU's prefill
      const std::vector<float> d = dev.forward({ids[t]}, t, false);
      wd = std::max(wd, rel_row(d.data(), a.data() + static_cast<size_t>(t) * V, V));
    }
    const std::vector<int> ga = greedy(cpu, ids, gen, V), gb = greedy(dev, ids, gen, V);
    int first_diff = -1;
    for (int g = 0; g < gen; ++g)
      if (ga[g] != gb[g]) {
        first_diff = g;
        break;
      }
    greedy_same += first_diff < 0;
    worst = std::max(worst, w);
    worst_decode = std::max(worst_decode, wd);
    top1 += agree;
    positions += len;
    rows.push_back({{"i", s}, {"prefill_rel", w}, {"decode_rel", wd}, {"top1", agree}, {"greedy_first_diff", first_diff}});
  }
  const bool pass = greedy_same == n && top1 == positions && worst <= tol && worst_decode <= tol;
  json out = {{"gguf", argv[1]},
              {"family_hint", doc.value("family_hint", "")},
              {"device_supported", true},
              {"on_gpu", dev.on_gpu()},
              {"prompts", n},
              {"len", len},
              {"gen", gen},
              {"worst_prefill_rel", worst},
              {"worst_decode_rel", worst_decode},
              {"top1_agree", std::to_string(top1) + "/" + std::to_string(positions)},
              {"greedy_identical", std::to_string(greedy_same) + "/" + std::to_string(n)},
              {"tol", tol},
              {"pass", pass},
              {"rows", rows}};
  std::cout << out.dump(1) << "\n";
  return pass ? 0 : 1;
}
