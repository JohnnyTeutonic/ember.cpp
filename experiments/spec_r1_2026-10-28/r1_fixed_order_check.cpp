// r1_fixed_order_check: fixed_linear's arithmetic replayed on the host (experiments/spec_r1_2026-10-28).
//   g++ -O2 -std=c++17 -ffp-contract=off -o r1_fixed_order_check r1_fixed_order_check.cpp && ./r1_fixed_order_check
//
// verify_kernels.cu's linear_smallm_kernel<M> (float4 loads) and linear_fixed_scalar_kernel<M> (scalar loads) give
// output o of row m from one warp: lane l keeps one accumulator per row, runs fmaf over the inputs 4 (l + 32 j) ..
// 4 (l + 32 j) + 3 for j = 0, 1, ... (the last group cut at I), and the 32 lane sums are then added by the butterfly
// v += shfl_xor(v, 16), (8), (4), (2), (1). This program replays both kernels' statement order with std::fmaf and
// float adds (lanes as an array, each butterfly step reading the previous step's values as the shuffle does) and
// checks, bit for bit, that a row's result is the same for every row count n = 1..16, for either load path, and
// whatever the other rows hold: the property that makes a verification row equal the decode step's (n = 1) row.
// It replays the order, not the GPU: on the device the same is measured (r1_bench bits_n9, the dump comparisons).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// One output o of rows 0..n-1: the float4 kernel's loop (requires I % 4 == 0) or the scalar twin's.
std::vector<float> warp_rows(const std::vector<float>& X, const float* w, int n, int I, bool float4_path) {
  std::vector<float> out(n);
  for (int m = 0; m < n; ++m) {
    float lane[32];
    for (int l = 0; l < 32; ++l) {
      float t = 0.0f;
      const float* x = X.data() + (size_t)m * I;
      if (float4_path) {
        for (int i4 = l; i4 < I / 4; i4 += 32) {
          const int b = 4 * i4;
          t = std::fmaf(w[b], x[b], t);
          t = std::fmaf(w[b + 1], x[b + 1], t);
          t = std::fmaf(w[b + 2], x[b + 2], t);
          t = std::fmaf(w[b + 3], x[b + 3], t);
        }
      } else {
        for (int i = 4 * l; i < I; i += 128) {
          const int e = i + 4 < I ? i + 4 : I;
          for (int j = i; j < e; ++j) t = std::fmaf(w[j], x[j], t);
        }
      }
      lane[l] = t;
    }
    for (int off = 16; off > 0; off >>= 1) {
      float nxt[32];
      for (int l = 0; l < 32; ++l) nxt[l] = lane[l] + lane[l ^ off];
      std::memcpy(lane, nxt, sizeof lane);
    }
    out[m] = lane[0];
  }
  return out;
}

uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int shapes[] = {320, 768, 864, 1280, 2048, 321, 766, 1283};  // the picks' widths, and widths I % 4 != 0
  long checks = 0, bad = 0, controls = 0, control_differ = 0;
  for (int I : shapes) {
    for (int trial = 0; trial < 8; ++trial) {
      std::vector<float> w(I), X(I);
      for (auto& v : w) v = nd(rng) * 0.05f;
      for (auto& v : X) v = nd(rng);
      // reference: the row alone (the decode step, n = 1), on the path the kernel takes for this I
      const bool f4 = I % 4 == 0;
      const float ref = warp_rows(X, w.data(), 1, I, f4)[0];
      {  // control: another order (one sequential fma chain) must be seen to differ, or the check sees nothing
        float t = 0.0f;
        for (int i = 0; i < I; ++i) t = std::fmaf(w[i], X[i], t);
        ++controls;
        control_differ += bits(t) != bits(ref);
      }
      for (int n = 1; n <= 16; ++n) {
        // the same row as row 0 of an n-row pass, the other n - 1 rows fresh random values
        for (int at : {0, n - 1}) {  // the row first, then last, in the pass
          std::vector<float> P((size_t)n * I);
          for (auto& v : P) v = nd(rng);
          std::copy(X.begin(), X.end(), P.begin() + (size_t)at * I);
          for (bool path : {true, false}) {
            if (path && !f4) continue;
            const float got = warp_rows(P, w.data(), n, I, path)[at];
            ++checks;
            if (bits(got) != bits(ref)) {
              ++bad;
              if (bad <= 5) std::printf("differ: I %d n %d row %d float4 %d: %a vs %a\n", I, n, at, (int)path, got, ref);
            }
          }
        }
      }
    }
  }
  std::printf("{\"checks\": %ld, \"bit_equal\": %ld, \"differ\": %ld, \"control_other_order_differs\": %ld, "
              "\"controls\": %ld}\n", checks, checks - bad, bad, control_differ, controls);
  return bad ? 1 : 0;
}
