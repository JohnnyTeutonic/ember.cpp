// The logits leg of the third-party gate (LOCAL_LLM_STACK.md section 7): loads a GGUF, takes its IR
// as the server would (ember_ir::find_ir: paperkiln.ir, or the IR built from a qwen2/qwen3 file's
// metadata), writes that IR to OUTDIR/ember.ir.json, then for each prompt (one line of ids.txt,
// token ids separated by spaces) runs the interpreter on the CPU from position 0 and writes the
// last row's logits to OUTDIR/first_{i}.f32 (raw little-endian float32, [vocab]); a prompt whose
// file exists is skipped, so a rerun resumes.
// Build (from the build directory, after make tinyllama_server; CPU-only build):
//   g++ -O3 -std=c++17 -fopenmp -I EMBER -I JSON_INC EMBER/tools/ir_first_logits.cpp
//       libtinyllama_core.a -lboost_regex -lpthread -o ir_first_logits
//   EMBER_IR_CTX=1024 ./ir_first_logits model.gguf ids.txt outdir
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf_parser.h"
#include "graph_ir.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s model.gguf ids.txt outdir\n", argv[0]);
    return 2;
  }
  try {
    GGUFData gguf = load_gguf_meta(argv[1], true);
    const std::string ir = ember_ir::find_ir(gguf);
    if (ir.empty()) {
      std::fprintf(stderr, "no IR for %s (llama path)\n", argv[1]);
      return 2;
    }
    const std::string out = argv[3];
    std::ofstream(out + "/ember.ir.json") << ir << "\n";
    ember_ir::Interpreter interp(ir, gguf, false);
    std::ifstream in(argv[2]);
    std::string line;
    int i = 0;
    while (std::getline(in, line)) {
      std::istringstream ss(line);
      std::vector<int> ids;
      for (int id; ss >> id;) ids.push_back(id);
      if (ids.empty()) continue;
      const std::string path = out + "/first_" + std::to_string(i) + ".f32";
      if (std::ifstream(path).good()) {
        ++i;
        continue;
      }
      const std::vector<float> logits = interp.forward(ids, 0);
      std::ofstream f(path + ".tmp", std::ios::binary);
      f.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
      f.close();
      std::rename((path + ".tmp").c_str(), path.c_str());
      std::printf("%d: %zu ids -> %zu logits\n", i, ids.size(), logits.size());
      std::fflush(stdout);
      ++i;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ir_first_logits: %s\n", e.what());
    return 1;
  }
  return 0;
}
