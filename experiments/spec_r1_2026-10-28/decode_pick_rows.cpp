// Handcrafted CUDA record gate: ties, near ties, NaNs and infinities.
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <nlohmann/json.hpp>
#include "verify_kernels.h"

int main() {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<std::vector<float>> cases = {
      {-1, 3, 2, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 1},
      {1, std::nextafter(1.0f, 0.0f), 0, 0, 0, 0, 0, 0},
      {0, nan, 2, 0, 0, 0, 0, 0}, std::vector<float>(8, -inf),
      {inf, 0, 0, 0, 0, 0, 0, 0}};
  std::vector<float> flat;
  for (const auto& r : cases) flat.insert(flat.end(), r.begin(), r.end());
  float* device = nullptr;
  ember_verify::RowPick* selected = nullptr;
  if (cudaMalloc(&device, flat.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&selected, cases.size() * sizeof(ember_verify::RowPick)) != cudaSuccess) return 2;
  if (cudaMemcpy(device, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  ember_verify::select_rows(device, static_cast<int>(cases.size()), 8, selected, 0);
  std::vector<ember_verify::RowPick> got(cases.size());
  if (cudaMemcpy(got.data(), selected, got.size() * sizeof(got[0]), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
  bool pass = true;
  nlohmann::json rows = nlohmann::json::array();
  for (size_t r = 0; r < cases.size(); ++r) {
    float max = -inf, second = -inf;
    int first = -1, last = -1, has_nan = 0;
    for (float v : cases[r]) {
      has_nan |= std::isnan(v);
      if (v > max) max = v;
    }
    for (int i = 0; i < 8; ++i) {
      if (cases[r][i] == max) { if (first < 0) first = i; last = i; }
      else if (cases[r][i] < max && cases[r][i] > second) second = cases[r][i];
    }
    const auto& p = got[r];
    const bool ok = p.max == max && p.second == second && p.first == first && p.last == last && p.nan == has_nan;
    pass = pass && ok;
    rows.push_back({{"case", r}, {"pass", ok}, {"first", p.first}, {"last", p.last}, {"nan", p.nan}});
  }
  cudaFree(selected);
  cudaFree(device);
  std::cout << nlohmann::json({{"pass", pass}, {"cases", rows}}).dump(1) << "\n";
  return pass ? 0 : 1;
}
