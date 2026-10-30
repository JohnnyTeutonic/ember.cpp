// e4_device_gate: the E4 synthetic families (gemma2, phi3_longrope, phi3_swa) on ember's CUDA interpreter against
// HF and against ember's CPU interpreter in the same process (ember.cpp experiments/e4_gemma2_phi3_2026-10-28).
//
//   e4_device_gate MODEL.gguf CASES.json OUT.json [DUMP.bin]
//
// CASES.json: thirdparty_synth.py --export's file (40 sequences, HF's greedy continuations, HF's f32 logits at each
// sequence's last position, hex as keramos IR_SPEC 11.2). Per case and per engine (device, then CPU): the ids
// prefilled in one call, then the greedy tokens decoded one call of one id at a time (on the device the captured
// CUDA graph after the first step; under LongRoPE a step whose length passes original_ctx recomputes the slot, on
// either path). Reported: greedy equal to HF and device = CPU, the first token's logit distance from HF's row and
// from the CPU's (max |d| over the reference row's range), and every decode step's row device against CPU. DUMP.bin
// (optional) receives every device logit row in order (prefill's last row, then each step), so two runs compare
// byte for byte (EMBER_DETERMINISTIC=1 run to run). Exit 1 unless every case's greedy matches HF on both engines.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;

namespace {

std::vector<float> unhex(const std::string& h) {
  std::vector<float> v(h.size() / 8);
  for (size_t i = 0; i < v.size(); ++i) {
    const uint32_t b = static_cast<uint32_t>(std::stoul(h.substr(i * 8, 8), nullptr, 16));
    std::memcpy(&v[i], &b, 4);
  }
  return v;
}

int argmax(const float* v, int n) { return static_cast<int>(std::max_element(v, v + n) - v); }

double rel_row(const float* a, const float* ref, int n) {
  float lo = ref[0], hi = ref[0];
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(ref[i])) return std::numeric_limits<double>::infinity();
    lo = std::min(lo, ref[i]);
    hi = std::max(hi, ref[i]);
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - ref[i]));
  }
  return hi > lo ? d / (hi - lo) : d;
}

struct Run {
  std::vector<int> toks;
  std::vector<std::vector<float>> rows;  // the prefill's last row, then each decode step's
};

Run run(ember_ir::Interpreter& m, const std::vector<int>& ids, int gen, int V) {
  Run r;
  std::vector<float> l = m.forward(ids, 0, false);
  int pos = static_cast<int>(ids.size());
  for (int g = 0; g < gen; ++g) {
    r.rows.push_back(std::vector<float>(l.end() - V, l.end()));
    r.toks.push_back(argmax(r.rows.back().data(), V));
    if (g + 1 < gen) l = m.forward({r.toks.back()}, pos++, false);
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: e4_device_gate MODEL.gguf CASES.json OUT.json [DUMP.bin]\n";
    return 2;
  }
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  const bool supported = ember_ir::device_supported(ir);
  std::ifstream cf(argv[2]);
  const json cases = json::parse(cf);
  ember_ir::Interpreter cpu(ir, gguf, false);
  std::unique_ptr<ember_ir::Interpreter> dev;
  if (supported) dev = std::make_unique<ember_ir::Interpreter>(ir, gguf, true);
  const int V = cpu.vocab();
  std::ofstream dump;
  if (argc > 4) dump.open(argv[4], std::ios::binary);
  int dev_hf = 0, cpu_hf = 0, dev_cpu = 0, n = 0, steps = 0, step_top1 = 0;
  double worst_dev_hf = 0, worst_cpu_hf = 0, worst_dev_cpu_first = 0, worst_dev_cpu_step = 0;
  json rows = json::array();
  for (const auto& c : cases.at("cases")) {
    const std::vector<int> ids = c.at("ids").get<std::vector<int>>();
    const std::vector<int> hf = c.at("hf_greedy").get<std::vector<int>>();
    const std::vector<float> hl = unhex(c.at("hf_last_logits").at("f32").get<std::string>());
    const int gen = static_cast<int>(hf.size());
    const Run rc = run(cpu, ids, gen, V);
    json row = {{"case", n}, {"len", ids.size()}, {"cpu_greedy_equal_hf", rc.toks == hf},
                {"cpu_first_rel_hf", rel_row(rc.rows[0].data(), hl.data(), V)}};
    cpu_hf += rc.toks == hf;
    worst_cpu_hf = std::max(worst_cpu_hf, row["cpu_first_rel_hf"].get<double>());
    if (dev) {
      const Run rd = run(*dev, ids, gen, V);
      double ws = 0;
      int agree = 0;
      for (int g = 0; g < gen; ++g) {
        ws = std::max(ws, rel_row(rd.rows[g].data(), rc.rows[g].data(), V));
        agree += argmax(rd.rows[g].data(), V) == argmax(rc.rows[g].data(), V);
        if (dump) dump.write(reinterpret_cast<const char*>(rd.rows[g].data()), static_cast<std::streamsize>(V * 4));
      }
      row["dev_greedy_equal_hf"] = rd.toks == hf;
      row["dev_greedy_equal_cpu"] = rd.toks == rc.toks;
      row["dev_first_rel_hf"] = rel_row(rd.rows[0].data(), hl.data(), V);
      row["dev_first_rel_cpu"] = rel_row(rd.rows[0].data(), rc.rows[0].data(), V);
      row["dev_step_rel_cpu"] = ws;
      row["dev_step_top1_cpu"] = agree;
      dev_hf += rd.toks == hf;
      dev_cpu += rd.toks == rc.toks;
      steps += gen;
      step_top1 += agree;
      worst_dev_hf = std::max(worst_dev_hf, row["dev_first_rel_hf"].get<double>());
      worst_dev_cpu_first = std::max(worst_dev_cpu_first, row["dev_first_rel_cpu"].get<double>());
      worst_dev_cpu_step = std::max(worst_dev_cpu_step, ws);
    }
    rows.push_back(row);
    ++n;
  }
  const auto frac = [&](int k) { return std::to_string(k) + "/" + std::to_string(n); };
  json out = {{"gguf", argv[1]},
              {"device_supported", supported},
              {"on_gpu", dev ? dev->on_gpu() : false},
              {"cases", n},
              {"vocab", V},
              {"rows_dumped", steps},
              {"cpu_greedy_vs_hf", frac(cpu_hf)},
              {"cpu_worst_first_rel_hf", worst_cpu_hf}};
  if (dev) {
    out["dev_greedy_vs_hf"] = frac(dev_hf);
    out["dev_greedy_vs_cpu"] = frac(dev_cpu);
    out["dev_worst_first_rel_hf"] = worst_dev_hf;
    out["dev_worst_first_rel_cpu"] = worst_dev_cpu_first;
    out["dev_worst_step_rel_cpu"] = worst_dev_cpu_step;
    out["dev_step_top1_cpu"] = std::to_string(step_top1) + "/" + std::to_string(steps);
  }
  out["rows"] = rows;
  std::ofstream(argv[3]) << out.dump(1) << "\n";
  json brief = out;
  brief.erase("rows");
  std::cout << brief.dump() << "\n";
  return n > 0 && cpu_hf == n && (!dev || (dev_hf == n && dev_cpu == n && step_top1 == steps)) ? 0 : 1;
}
