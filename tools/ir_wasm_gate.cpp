// ir_wasm_gate: ember's IR interpreter compiled to WebAssembly against the native CPU build, on raw token
// ids (no tokenizer), for any GGUF ember_ir::find_ir gives an IR for. The prompts, lengths and greedy run
// are tools/ir_device_gate's (40 seeded random prompts of 24 ids, 32 greedy tokens), so a family's WASM
// leg is gated on the same set as its CUDA leg.
//
//   ir_wasm_gate ref   MODEL.gguf DIR [--n 40] [--len 24] [--gen 32] [--seed 0] [--ctx 1024] [--build-ir] [--first K]
//   ir_wasm_gate check MODEL.gguf DIR [--ctx 1024] [--build-ir] [--first K] [--tol 1e-4]
//
// ref (the native build, linked with libtinyllama_core): draws the prompts with ir_device_gate's
// generator (std::mt19937 and libstdc++'s uniform_int_distribution, so the same ids) into DIR/ids.json;
// per prompt (the first K with --first; all n are drawn, so the ids stay the device gate's), the prefill logits of every position (DIR/ref_<i>.bin, float32 [len][vocab]), their
// argmax, token-by-token decode against that prefill, and gen greedy tokens (DIR/ref.jsonl, one row
// per prompt).
// check (the WebAssembly build, in node with NODERAWFS; the same build flags and sources as
// deploy/wasm/build.sh): reads ids.json and the ref rows, and per prompt (the first K with --first)
// compares its prefill logits with ref_<i>.bin (max |dlogit| over the reference's range, per
// position, and the argmax), its token-by-token decode against the reference prefill, and its greedy
// tokens; rows to DIR/check.jsonl, then a JSON summary on stdout. Exit 1 unless every greedy
// continuation is identical, every argmax agrees and the worst difference is within --tol (the device
// gate's receipt bound).
// Both modes resume: a prompt with a row in its .jsonl is not run again (rows are appended after
// the prompt's logits are written by rename). --ctx sets EMBER_IR_CTX (the kv positions held; the
// third-party IRs declare their whole trained context) and --build-ir EMBER_IR_BUILD=1 (llama and
// mistral files without a window), in the process itself, so node needs no environment.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
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

std::vector<json> read_rows(const std::string& path) {
  std::vector<json> rows;
  std::ifstream f(path);
  for (std::string line; std::getline(f, line);)
    if (!line.empty()) rows.push_back(json::parse(line));
  return rows;
}

void append_row(const std::string& path, const json& row) {
  std::ofstream f(path, std::ios::app);
  f << row.dump() << "\n";
  f.flush();
  if (!f) throw std::runtime_error("cannot append to " + path);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: ir_wasm_gate ref|check MODEL.gguf DIR [options]\n";
    return 2;
  }
  const std::string mode = argv[1], model = argv[2], dir = argv[3];
  int n = 40, len = 24, gen = 32, first = -1;
  unsigned seed = 0;
  double tol = 1e-4;
  std::string ctx = "1024";
  bool build_ir = false;
  for (int i = 4; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--build-ir") {
      build_ir = true;
      continue;
    }
    if (i + 1 >= argc) {
      std::cerr << "option " << k << " needs a value\n";
      return 2;
    }
    const std::string v = argv[++i];
    if (k == "--n") n = std::atoi(v.c_str());
    else if (k == "--len") len = std::atoi(v.c_str());
    else if (k == "--gen") gen = std::atoi(v.c_str());
    else if (k == "--seed") seed = static_cast<unsigned>(std::atoi(v.c_str()));
    else if (k == "--ctx") ctx = v;
    else if (k == "--first") first = std::atoi(v.c_str());
    else if (k == "--tol") tol = std::atof(v.c_str());
    else {
      std::cerr << "unknown option " << k << "\n";
      return 2;
    }
  }
  if (mode != "ref" && mode != "check") {
    std::cerr << "mode must be ref or check\n";
    return 2;
  }
  setenv("EMBER_IR_CTX", ctx.c_str(), 1);
  if (build_ir) setenv("EMBER_IR_BUILD", "1", 1);
  const auto t_load = std::chrono::steady_clock::now();
#ifdef __EMSCRIPTEN__
  const GGUFData gguf = load_gguf_meta(model, false);  // as the page: no mmap
#else
  const GGUFData gguf = load_gguf_meta(model, true);
#endif
  const std::string ir = ember_ir::find_ir(gguf);
  if (ir.empty()) {
    std::cerr << "no IR: use a GGUF with paperkiln.ir or a built architecture (--build-ir for llama/mistral)\n";
    return 1;
  }
  ember_ir::Interpreter m(ir, gguf, false);
  const int V = m.vocab();
  const double load_ms = ms_since(t_load);

  if (mode == "ref") {
    // ir_device_gate's prompts: the same generator, bounds and order.
    const int hi = std::min<int>(V, static_cast<int>(gguf.tokenizer_tokens.empty() ? V : gguf.tokenizer_tokens.size()));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, hi - 1);
    json prompts = json::array();
    for (int s = 0; s < n; ++s) {
      std::vector<int> ids(len);
      for (int& t : ids) t = pick(rng);
      prompts.push_back(ids);
    }
    {
      std::ofstream f(dir + "/ids.json.tmp");
      f << json{{"gguf", model}, {"n", n}, {"len", len}, {"gen", gen}, {"seed", seed}, {"vocab", V}, {"ids", prompts}}.dump();
    }
    std::rename((dir + "/ids.json.tmp").c_str(), (dir + "/ids.json").c_str());
    std::set<int> done;
    for (const json& r : read_rows(dir + "/ref.jsonl")) done.insert(r["i"].get<int>());
    for (int s = 0; s < (first > 0 ? std::min(first, n) : n); ++s) {
      if (done.count(s)) continue;
      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<int> ids = prompts[s].get<std::vector<int>>();
      const std::vector<float> a = m.forward(ids, 0, true);
      std::vector<int> top;
      for (int t = 0; t < len; ++t) top.push_back(argmax(a.data() + static_cast<size_t>(t) * V, V));
      double wd = 0.0;
      for (int t = 0; t < len; ++t) {
        const std::vector<float> d = m.forward({ids[t]}, t, false);
        wd = std::max(wd, rel_row(d.data(), a.data() + static_cast<size_t>(t) * V, V));
      }
      const std::vector<int> g = greedy(m, ids, gen, V);
      const std::string bin = dir + "/ref_" + std::to_string(s) + ".bin";
      {
        std::ofstream f(bin + ".tmp", std::ios::binary);
        f.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size() * sizeof(float)));
        if (!f) throw std::runtime_error("cannot write " + bin);
      }
      std::rename((bin + ".tmp").c_str(), bin.c_str());
      append_row(dir + "/ref.jsonl", {{"i", s}, {"top1", top}, {"decode_rel", wd}, {"greedy", g}, {"ms", ms_since(t0)}});
      std::cerr << "ref prompt " << s << ": " << ms_since(t0) << " ms\n";
    }
    std::cout << json{{"mode", "ref"}, {"gguf", model}, {"vocab", V}, {"prompts", n}, {"load_ms", load_ms}}.dump() << "\n";
    return 0;
  }

  // check
  json idoc;
  {
    std::ifstream f(dir + "/ids.json");
    if (!f) throw std::runtime_error("no " + dir + "/ids.json: run ref first");
    idoc = json::parse(f);
  }
  len = idoc["len"].get<int>();
  gen = idoc["gen"].get<int>();
  n = idoc["n"].get<int>();
  if (idoc["vocab"].get<int>() != V) throw std::runtime_error("vocab differs from the reference's");
  const int upto = first > 0 ? std::min(first, n) : n;
  std::vector<json> ref(n);
  for (const json& r : read_rows(dir + "/ref.jsonl")) ref[r["i"].get<int>()] = r;
  std::set<int> done;
  for (const json& r : read_rows(dir + "/check.jsonl")) done.insert(r["i"].get<int>());
  for (int s = 0; s < upto; ++s) {
    if (done.count(s)) continue;
    if (ref[s].is_null()) throw std::runtime_error("reference prompt " + std::to_string(s) + " missing");
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<int> ids = idoc["ids"][s].get<std::vector<int>>();
    std::vector<float> r(static_cast<size_t>(len) * V);
    {
      std::ifstream f(dir + "/ref_" + std::to_string(s) + ".bin", std::ios::binary);
      f.read(reinterpret_cast<char*>(r.data()), static_cast<std::streamsize>(r.size() * sizeof(float)));
      if (!f) throw std::runtime_error("cannot read ref_" + std::to_string(s) + ".bin");
    }
    const std::vector<float> a = m.forward(ids, 0, true);
    double w = 0.0, wd = 0.0;
    int agree = 0;
    json flips = json::array();
    for (int t = 0; t < len; ++t) {
      const float* ra = r.data() + static_cast<size_t>(t) * V;
      const float* rb = a.data() + static_cast<size_t>(t) * V;
      w = std::max(w, rel_row(rb, ra, V));
      const bool same = argmax(ra, V) == argmax(rb, V);
      agree += same;
      if (!same) flips.push_back({{"pos", t}, {"ref_top", argmax(ra, V)}, {"wasm_top", argmax(rb, V)}});
    }
    for (int t = 0; t < len; ++t) {
      const std::vector<float> d = m.forward({ids[t]}, t, false);
      wd = std::max(wd, rel_row(d.data(), r.data() + static_cast<size_t>(t) * V, V));
    }
    const std::vector<int> g = greedy(m, ids, gen, V);
    const std::vector<int> gr = ref[s]["greedy"].get<std::vector<int>>();
    int first_diff = -1;
    for (int k = 0; k < gen; ++k)
      if (g[k] != gr[k]) {
        first_diff = k;
        break;
      }
    json row = {{"i", s}, {"prefill_rel", w}, {"decode_rel", wd}, {"top1", agree}, {"greedy_first_diff", first_diff},
                {"ms", ms_since(t0)}};
    if (!flips.empty()) row["flips"] = flips;
    append_row(dir + "/check.jsonl", row);
    std::cerr << "check prompt " << s << ": rel " << w << ", decode " << wd << ", top1 " << agree << "/" << len
              << ", greedy " << (first_diff < 0 ? "same" : "differs at " + std::to_string(first_diff)) << ", "
              << ms_since(t0) << " ms\n";
  }
  double worst = 0.0, worst_decode = 0.0, ms = 0.0;
  int top1 = 0, positions = 0, greedy_same = 0, rows_n = 0;
  for (const json& row : read_rows(dir + "/check.jsonl")) {
    if (row["i"].get<int>() >= upto) continue;
    ++rows_n;
    worst = std::max(worst, row["prefill_rel"].get<double>());
    worst_decode = std::max(worst_decode, row["decode_rel"].get<double>());
    top1 += row["top1"].get<int>();
    positions += len;
    greedy_same += row["greedy_first_diff"].get<int>() < 0;
    ms += row["ms"].get<double>();
  }
  const bool pass = rows_n == upto && greedy_same == upto && top1 == positions && worst <= tol && worst_decode <= tol;
  std::cout << json{{"gguf", model},
                    {"prompts", upto},
                    {"of", n},
                    {"len", len},
                    {"gen", gen},
                    {"vocab", V},
                    {"worst_prefill_rel", worst},
                    {"worst_decode_rel", worst_decode},
                    {"top1_agree", std::to_string(top1) + "/" + std::to_string(positions)},
                    {"greedy_identical", std::to_string(greedy_same) + "/" + std::to_string(upto)},
                    {"tol", tol},
                    {"pass", pass},
                    {"load_ms", load_ms},
                    {"ms_per_prompt", rows_n ? ms / rows_n : 0.0}}
                   .dump(1)
            << "\n";
  return pass ? 0 : 1;
}
