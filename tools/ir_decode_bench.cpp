// Decode step time of the IR interpreter in-process (no server, sampling or
// HTTP): loads a GGUF, builds ember_ir::Interpreter (device path when the
// build has CUDA and argv[2] is "gpu"), prefills 16 tokens, then times
// `steps` single-token forward calls, `reps` times over. Prints the median
// and minimum wall ms per step and the calling thread's CPU ms per step (the
// host work, CUDA API calls included; not inflated while the thread waits for
// a core on a loaded host). Interpreter logs go to ./debugging.log.
// Build (from the build directory, after make tinyllama_server; GRAPH_IR is
// the graph_ir.cpp under test, its graph_ir.h beside it):
//   g++ -O3 -std=c++17 -fopenmp -DHAS_CUDA -I EMBER -I JSON_INC -I /usr/local/cuda/include
//       EMBER/tools/ir_decode_bench.cpp GRAPH_IR libtinyllama_core.a -L /usr/local/cuda/lib64
//       -lcudart -lcublas -lboost_regex -lpthread -o ir_decode_bench
//   ./ir_decode_bench model.gguf gpu [steps=128] [reps=5]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "gguf_parser.h"
#include "graph_ir.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.gguf gpu|cpu [steps] [reps]\n", argv[0]);
    return 2;
  }
  const bool gpu = std::string(argv[2]) == "gpu";
  int steps = argc > 3 ? std::atoi(argv[3]) : 128;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
  GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  if (ir.empty()) {
    std::fprintf(stderr, "no IR in %s (set EMBER_IR)\n", argv[1]);
    return 2;
  }
  ember_ir::Interpreter interp(ir, gguf, gpu);
  const int prompt = 16;
  steps = std::min(steps, interp.max_len() - prompt);
  std::vector<int> toks(prompt);
  for (int i = 0; i < prompt; ++i) toks[i] = (i * 37 + 11) % interp.vocab();
  auto thread_ms = [] {  // this thread's CPU time: not inflated while descheduled
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
  };
  std::vector<double> ms, cpu;
  for (int r = 0; r < reps; ++r) {
    std::vector<float> logits = interp.forward(toks, 0);
    for (int s = 0; s < steps; ++s) {
      const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
      const auto t0 = std::chrono::steady_clock::now();
      const double c0 = thread_ms();
      logits = interp.forward({next}, prompt + s);
      cpu.push_back(thread_ms() - c0);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
  }
  std::sort(ms.begin(), ms.end());
  std::sort(cpu.begin(), cpu.end());
  std::printf("%s: %d decode steps (%d x %d): wall median %.3f ms/step, min %.3f, p90 %.3f; thread cpu median %.3f, min %.3f\n",
              interp.on_gpu() ? "cuda" : "cpu", static_cast<int>(ms.size()), reps, steps, ms[ms.size() / 2], ms.front(),
              ms[ms.size() * 9 / 10], cpu[cpu.size() / 2], cpu.front());
  return 0;
}
