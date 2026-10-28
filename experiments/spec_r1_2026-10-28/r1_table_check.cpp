// r1_table_check: ember's draft_table.h drafter on given histories, for r1_table_check.py to compare with
// r1_sim.py's (experiments/spec_r1_2026-10-28). Header-only; no model.
//   r1_table_check TABLE.bin VOCAB < QUERIES > DRAFTS
// Each query line: "k theta n h_0 ... h_{n-1}"; each output line: the draft's ids, space-separated (empty if none).
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "draft_table.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: r1_table_check TABLE.bin VOCAB < QUERIES\n";
    return 2;
  }
  ember_draft::DraftTable t;
  std::string why;
  if (!t.load(argv[1], std::stoi(argv[2]), &why)) {
    std::cerr << "load: " << why << "\n";
    return 1;
  }
  std::cerr << "orders " << t.orders() << ", entries " << t.entries() << "\n";
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line);
    int k = 0, n = 0;
    double theta = 0;
    ss >> k >> theta >> n;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) ss >> h[i];
    const std::vector<int> d = t.draft(h, k, theta);
    for (size_t i = 0; i < d.size(); ++i) std::cout << (i ? " " : "") << d[i];
    std::cout << "\n";
  }
  return 0;
}
