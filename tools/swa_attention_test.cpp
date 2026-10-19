// Sliding-window attention with sinks: ember's attention_window_batch_cpu
// (cpu_attention.cpp) against paperkiln's semantics. Three checks:
//   A. paperkiln's own ops::swa_attention outputs (tools/swa_reference_dump.cpp
//      writes them), single head; ember runs each case as one prefill, as a
//      short prefill then T = 1 decode steps, and in chunks of 5, through the
//      KV cache, so decode steps past the window are covered. max |d| < 1e-5.
//   B. an independent rule, written another way: key j is visible from query
//      p iff j <= p and (p - j < window or j < sinks), computed in double,
//      with grouped-query heads, prefill and decode at many start positions.
//      max |d| < 1e-5.
//   C. bitwise: window 0, and any window covering [0, p] (with or without
//      sinks), give exactly attention_batch_cpu's output.
// Build (from the build directory, after make tinyllama_server):
//   g++ -O2 -std=c++17 -fopenmp -I EMBER -I JSON_INC EMBER/tools/swa_attention_test.cpp
//       libtinyllama_core.a -lboost_regex -lpthread -o swa_attention_test
//   ./swa_attention_test [paperkiln_dump.bin]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_attention.h"

namespace {
int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

std::vector<float> randn(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// Runs ember's kernel over all T rows of q/k/v ([T][heads][hd]), fed in
// chunks: first `first` rows, then `step` rows at a time, through the cache.
std::vector<float> run_ember(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                             int T, int hq, int hkv, int hd, float scale, int window, int sinks, int first, int step) {
  KVCacheLayer cache;
  const size_t krow = static_cast<size_t>(hkv) * hd, qrow = static_cast<size_t>(hq) * hd;
  cache.k.assign(static_cast<size_t>(T) * krow, 0.0f);
  cache.v.assign(cache.k.size(), 0.0f);
  std::vector<float> y(static_cast<size_t>(T) * qrow);
  for (int start = 0; start < T;) {
    const int n = std::min(start == 0 ? first : step, T - start);
    std::copy(k.begin() + start * krow, k.begin() + (start + n) * krow, cache.k.begin() + start * krow);
    std::copy(v.begin() + start * krow, v.begin() + (start + n) * krow, cache.v.begin() + start * krow);
    std::vector<float> qc(q.begin() + start * qrow, q.begin() + (start + n) * qrow), out;
    attention_window_batch_cpu(qc, cache, out, n, start, hq, hkv, hd, scale, window, sinks);
    std::copy(out.begin(), out.end(), y.begin() + start * qrow);
    start += n;
  }
  return y;
}

// The same chunked run through today's kernel (no window).
std::vector<float> run_full(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                            int T, int hq, int hkv, int hd, float scale, int first, int step) {
  KVCacheLayer cache;
  const size_t krow = static_cast<size_t>(hkv) * hd, qrow = static_cast<size_t>(hq) * hd;
  cache.k.assign(static_cast<size_t>(T) * krow, 0.0f);
  cache.v.assign(cache.k.size(), 0.0f);
  std::vector<float> y(static_cast<size_t>(T) * qrow);
  for (int start = 0; start < T;) {
    const int n = std::min(start == 0 ? first : step, T - start);
    std::copy(k.begin() + start * krow, k.begin() + (start + n) * krow, cache.k.begin() + start * krow);
    std::copy(v.begin() + start * krow, v.begin() + (start + n) * krow, cache.v.begin() + start * krow);
    std::vector<float> qc(q.begin() + start * qrow, q.begin() + (start + n) * qrow), out;
    attention_batch_cpu(qc, cache, out, n, start, hq, hkv, hd, scale);
    std::copy(out.begin(), out.end(), y.begin() + start * qrow);
    start += n;
  }
  return y;
}

// Independent reference: visibility by membership test, double arithmetic.
std::vector<float> run_reference(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                                 int T, int hq, int hkv, int hd, float scale, int window, int sinks) {
  std::vector<float> y(static_cast<size_t>(T) * hq * hd, 0.0f);
  for (int p = 0; p < T; ++p)
    for (int h = 0; h < hq; ++h) {
      const int g = h / (hq / hkv);
      const float* qp = q.data() + (static_cast<size_t>(p) * hq + h) * hd;
      std::vector<double> s(p + 1, 0.0);
      std::vector<bool> vis(p + 1, false);
      double mx = -1e300;
      for (int j = 0; j <= p; ++j) {
        vis[j] = window == 0 || p - j < window || j < sinks;
        if (!vis[j]) continue;
        const float* kj = k.data() + (static_cast<size_t>(j) * hkv + g) * hd;
        double dot = 0.0;
        for (int d = 0; d < hd; ++d) dot += double(qp[d]) * double(kj[d]);
        s[j] = dot * scale;
        mx = std::max(mx, s[j]);
      }
      double z = 0.0;
      for (int j = 0; j <= p; ++j) z += vis[j] ? std::exp(s[j] - mx) : 0.0;
      for (int d = 0; d < hd; ++d) {
        double acc = 0.0;
        for (int j = 0; j <= p; ++j)
          if (vis[j]) acc += std::exp(s[j] - mx) / z * double(v[(static_cast<size_t>(j) * hkv + g) * hd + d]);
        y[(static_cast<size_t>(p) * hq + h) * hd + d] = static_cast<float>(acc);
      }
    }
  return y;
}

float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return a.size() == b.size() ? m : INFINITY;
}

bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return false;
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  const float tol = 1e-5f;

  // A. paperkiln's own op ---------------------------------------------------
  if (argc > 1) {
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
      std::printf("cannot read %s\n", argv[1]);
      return 1;
    }
    int32_t n = 0;
    check(std::fread(&n, sizeof n, 1, f) == 1, "dump header");
    float worst = 0.0f;
    for (int c = 0; c < n; ++c) {
      int32_t hdr[4];
      float scale;
      check(std::fread(hdr, sizeof(int32_t), 4, f) == 4 && std::fread(&scale, sizeof scale, 1, f) == 1, "case header");
      const int T = hdr[0], W = hdr[1], S = hdr[2], dk = hdr[3];
      std::vector<float> q(T * dk), k(T * dk), v(T * dk), y(T * dk);
      for (auto* m : {&q, &k, &v, &y}) check(std::fread(m->data(), sizeof(float), m->size(), f) == m->size(), "case data");
      const float d_pre = max_diff(run_ember(q, k, v, T, 1, 1, dk, scale, W, S, T, T), y);
      const float d_dec = max_diff(run_ember(q, k, v, T, 1, 1, dk, scale, W, S, 3, 1), y);
      const float d_chk = max_diff(run_ember(q, k, v, T, 1, 1, dk, scale, W, S, 5, 5), y);
      const float d_ref = max_diff(run_reference(q, k, v, T, 1, 1, dk, scale, W, S), y);
      std::printf("A. T=%2d window=%2d sinks=%d: |ember-paperkiln| prefill %.2e, decode %.2e, chunks %.2e; "
                  "|independent-paperkiln| %.2e\n", T, W, S, d_pre, d_dec, d_chk, d_ref);
      worst = std::max({worst, d_pre, d_dec, d_chk, d_ref});
      check(d_pre < tol && d_dec < tol && d_chk < tol && d_ref < tol, "A: ember vs paperkiln swa_attention");
    }
    std::fclose(f);
    std::printf("A. %d paperkiln cases, max |d| %.2e (tolerance %.0e)\n", n, worst, tol);
  } else {
    std::printf("A. skipped: no paperkiln dump given\n");
  }

  // B. independent rule, grouped-query heads -----------------------------------
  {
    const int hq = 4, hkv = 2, hd = 16;
    const float scale = 0.25f;
    // (T, window, sinks, first prefill rows, step)
    const int cases[][5] = {{40, 6, 2, 40, 1}, {40, 6, 2, 1, 1},  {40, 6, 2, 9, 1},   {40, 2, 7, 4, 1},
                            {40, 1, 1, 7, 3},  {40, 1, 0, 40, 40}, {30, 50, 3, 10, 1}, {30, 30, 0, 2, 7},
                            {64, 16, 4, 20, 1}, {64, 16, 4, 64, 64}, {17, 5, 5, 6, 2},  {17, 3, 20, 5, 1}};
    float worst = 0.0f;
    unsigned seed = 7;
    for (const auto& c : cases) {
      const int T = c[0], W = c[1], S = c[2];
      auto q = randn(static_cast<size_t>(T) * hq * hd, seed++);
      auto k = randn(static_cast<size_t>(T) * hkv * hd, seed++);
      auto v = randn(static_cast<size_t>(T) * hkv * hd, seed++);
      const float d = max_diff(run_ember(q, k, v, T, hq, hkv, hd, scale, W, S, c[3], c[4]),
                               run_reference(q, k, v, T, hq, hkv, hd, scale, W, S));
      std::printf("B. T=%2d window=%2d sinks=%2d prefill %2d step %2d: max |d| %.2e\n", T, W, S, c[3], c[4], d);
      worst = std::max(worst, d);
      check(d < tol, "B: ember vs independent reference");
    }
    std::printf("B. %zu cases, max |d| %.2e (tolerance %.0e)\n", sizeof cases / sizeof cases[0], worst, tol);
  }

  // C. bitwise with today's kernel ---------------------------------------------
  {
    const int hq = 4, hkv = 2, hd = 16, T = 33;
    const float scale = 0.25f;
    auto q = randn(static_cast<size_t>(T) * hq * hd, 91);
    auto k = randn(static_cast<size_t>(T) * hkv * hd, 92);
    auto v = randn(static_cast<size_t>(T) * hkv * hd, 93);
    int n = 0;
    for (const int first : {T, 1, 4})
      for (const int step : {1, 3}) {
        const auto full = run_full(q, k, v, T, hq, hkv, hd, scale, first, step);
        for (const auto& ws : {std::pair<int, int>{0, 0}, {0, 4}, {T, 0}, {T, 3}, {1000, 8}}) {
          check(bitwise(run_ember(q, k, v, T, hq, hkv, hd, scale, ws.first, ws.second, first, step), full),
                "C: bitwise equality with attention_batch_cpu");
          ++n;
        }
      }
    std::printf("C. %d runs (window 0, and windows covering every query, sinks 0..8): bitwise equal to "
                "attention_batch_cpu\n", n);
  }

  if (failures) {
    std::printf("%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("all sliding-window attention checks passed\n");
  return 0;
}
