// Materialize ir_device_gate's seed=0,n=12,len=24 recipe; no model forward.
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>

#include "gguf_parser.h"
#include "graph_ir.h"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const auto gguf = load_gguf_meta(argv[1], true);
  const auto ir = ember_ir::find_ir(gguf);
  if (ir.empty()) return 2;
  ember_ir::Interpreter cpu(ir, gguf, false);
  const int hi = std::min<int>(
      cpu.vocab(), static_cast<int>(gguf.tokenizer_tokens.empty()
                                        ? cpu.vocab()
                                        : gguf.tokenizer_tokens.size()));
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> pick(0, hi - 1);
  nlohmann::json ids = nlohmann::json::array();
  for (int i = 0; i < 12; ++i) {
    std::vector<int> sequence(24);
    for (int& id : sequence) id = pick(rng);
    ids.push_back(sequence);
  }
  std::cout << ids.dump() << "\n";
}