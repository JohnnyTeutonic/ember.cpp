// kvq_dump_kv: the CPU interpreter's f32 kv rows after a prefill, written raw for an offline look at the
// keys' and values' channel statistics and at other quantisation layouts (R2,
// experiments/kv_quant_r2_2026-10-28/kv_stats.py). Uses the conformance hook save_state() (CPU only).
//
//   kvq_dump_kv MODEL.gguf IDS.json OUT.bin [--n 512]
//
// IDS.json: tools/kvq_inputs.py's inputs (the first tf input's first n ids are prefilled). OUT.bin:
// int32 layers, n, row, then per layer K [n][row] and V [n][row] float32.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: kvq_dump_kv MODEL.gguf IDS.json OUT.bin [--n 512]\n";
    return 2;
  }
  int n = 512;
  for (int i = 4; i + 1 < argc; i += 2)
    if (std::string(argv[i]) == "--n") n = std::atoi(argv[i + 1]);
  std::ifstream fi(argv[2]);
  const nlohmann::json inputs = nlohmann::json::parse(fi);
  std::vector<int> ids = inputs.at("tf").at(0).at("ids").get<std::vector<int>>();
  ids.resize(n);
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  ember_ir::Interpreter m(ember_ir::find_ir(gguf), gguf, false);
  m.forward(ids, 0, false);
  const ember_ir::Interpreter::State s = m.save_state();
  const int layers = static_cast<int>(s.k.size());
  const int row = layers ? static_cast<int>(s.k[0].size() / n) : 0;
  std::FILE* f = std::fopen(argv[3], "wb");
  const int hdr[3] = {layers, n, row};
  std::fwrite(hdr, sizeof(int), 3, f);
  for (int l = 0; l < layers; ++l) {
    std::fwrite(s.k[l].data(), sizeof(float), static_cast<size_t>(n) * row, f);
    std::fwrite(s.v[l].data(), sizeof(float), static_cast<size_t>(n) * row, f);
  }
  std::fclose(f);
  std::cerr << "layers " << layers << ", positions " << n << ", row " << row << "\n";
  return 0;
}
