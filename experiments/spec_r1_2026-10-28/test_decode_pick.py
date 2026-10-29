"""CPU differential check of the production compact-pick sampler and full sampler.

Runs g++ in a temporary directory; no model, CUDA device, or network needed.
The function bodies and RowPick layout are read from production sources, not copied.
    python3 experiments/spec_r1_2026-10-28/test_decode_pick.py
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def body(source, start):
    a = source.index(start)
    b = source.index("{", a)
    depth = 1
    end = b + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[a:end]


def main():
    api = (ROOT / "api.cpp").read_text()
    kernels = (ROOT / "verify_kernels.h").read_text()
    functions = body(api, "static int greedy_from_pick(") + "\n" + body(
        api, "static int sample_top_k_top_p_temperature(const std::vector<float>& logits,\n"
    )
    record = body(kernels, "struct RowPick {") + ";"
    source = r"""
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace ember_verify { RECORD }
FUNCTIONS
int main() {
  static_assert(sizeof(ember_verify::RowPick) == 20);
  std::mt19937 data(0x12cafe);
  std::uniform_real_distribution<float> values(-30, 30);
  int accepted = 0, fallback = 0;
  for (int trial = 0; trial < 1200; ++trial) {
    int n = trial % 7 == 0 ? 16384 : 1 + trial % 137;
    std::vector<float> row(n);
    for (float& v : row) v = values(data);
    if (trial % 4 == 0) { row.front() = 40; row.back() = 40; }
    if (trial % 4 == 1 && n > 1) { row.front() = 40; row.back() = std::nextafter(40.f, 0.f); }
    ember_verify::RowPick p{-INFINITY, -INFINITY, -1, -1, 0};
    for (int i = 0; i < n; ++i) p.max = std::max(p.max, row[i]);
    for (int i = 0; i < n; ++i) {
      if (row[i] == p.max) { if (p.first < 0) p.first = i; p.last = i; }
      else p.second = std::max(p.second, row[i]);
    }
    for (float temp : {0.f, .049f, .05f, .1f, 1.f, 10.f, 1e8f}) {
      for (float top_p : {0.f, .9f, 1.f}) {
        std::mt19937 compact(trial), full(trial);
        int pick = greedy_from_pick(p, temp, compact);
        int ref = sample_top_k_top_p_temperature(row, temp, 1, top_p, full);
        if (pick < 0) { ++fallback; continue; }
        if (pick != ref || compact != full) {
          std::cerr << "mismatch trial=" << trial << " temp=" << temp << " top_p=" << top_p << "\n";
          return 1;
        }
        ++accepted;
      }
    }
  }
  for (auto p : {ember_verify::RowPick{INFINITY, 0, 0, 0, 0},
                 ember_verify::RowPick{-INFINITY, -INFINITY, 0, 3, 0},
                 ember_verify::RowPick{1, 0, 0, 0, 1},
                 ember_verify::RowPick{1, 0, -1, -1, 0}}) {
    std::mt19937 rng(17), before = rng;
    if (greedy_from_pick(p, 1, rng) != -1 || rng != before) return 2;
  }
  std::cout << "PASS: " << accepted << " compact decisions, " << fallback
            << " conservative fallbacks; ids and RNG state equal; invalid rows rejected\n";
}
""".replace("RECORD", record).replace("FUNCTIONS", functions)
    with tempfile.TemporaryDirectory(prefix="ember_decode_pick_") as tmp:
        cpp, exe = Path(tmp) / "check.cpp", Path(tmp) / "check"
        cpp.write_text(source)
        subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
