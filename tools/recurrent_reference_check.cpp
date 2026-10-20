// Recurrent IR state, paperkiln side: reads recurrent_state_test's inputs and
// outputs, runs paperkiln's own CPU ops (ops::kimi_attention per head, as
// nn::KimiLinearAttention calls it, and ops::ssm_scan), both as one prefill
// and as token-by-token decode (paperkiln has no incremental state: decode
// step t is the op on rows 0..t, last row), and compares bit for bit.
// Build against a paperkiln checkout PK built in PK/build:
//   g++ -O2 -std=c++17 -fopenmp -I PK/include -I PK/build/_deps/json-src/single_include
//       -I PK/third_party/transformer_cpp/include -I PK/third_party/transformer_cpp
//       recurrent_reference_check.cpp PK/build/libmicrotorch.a libtransformer_core.a -lpthread
// Usage: recurrent_reference_check IN.bin
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "microtorch/autograd.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

static Matrix mat(const std::vector<float>& v, size_t r, size_t c, size_t r0 = 0, size_t col0 = 0, size_t stride = 0) {
  if (stride == 0) stride = c;
  Matrix m(r, c);
  for (size_t i = 0; i < r; ++i)
    for (size_t j = 0; j < c; ++j) m(i, j) = v[(r0 + i) * stride + col0 + j];
  return m;
}

// kimi on rows 0..rows-1, all heads: [rows, w]
static std::vector<float> pk_kimi(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                                  int rows, int H, int hd) {
  const int w = H * hd;
  std::vector<float> out(static_cast<size_t>(rows) * w);
  NoGrad ng;
  for (int h = 0; h < H; ++h) {
    Var qh = make_var(mat(q, rows, hd, 0, h * hd, w)), kh = make_var(mat(k, rows, hd, 0, h * hd, w)),
        vh = make_var(mat(v, rows, hd, 0, h * hd, w));
    Var o = ops::kimi_attention(qh, kh, vh, /*causal=*/true, rows);
    for (int t = 0; t < rows; ++t)
      for (int j = 0; j < hd; ++j) out[static_cast<size_t>(t) * w + h * hd + j] = o->data(t, j);
  }
  return out;
}

static std::vector<float> pk_ssm(const std::vector<float>& u, const std::vector<float>& A, const std::vector<float>& B,
                                 const std::vector<float>& C, const std::vector<float>& D, int rows, int n) {
  NoGrad ng;
  Var y = ops::ssm_scan(make_var(mat(u, rows, n)), make_var(mat(A, n, n)), make_var(mat(B, n, 1)),
                        make_var(mat(C, 1, n)), make_var(mat(D, 1, 1)));
  std::vector<float> out(static_cast<size_t>(rows) * n);
  for (int t = 0; t < rows; ++t)
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(t) * n + i] = y->data(t, i);
  return out;
}

struct Cmp {
  double max_abs = 0, max_rel = 0;
  size_t differ = 0;
};
static Cmp cmp(const std::vector<float>& a, const std::vector<float>& b) {
  Cmp c;
  if (a.size() != b.size()) {
    c.differ = static_cast<size_t>(-1);
    return c;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i] && !(std::isnan(a[i]) && std::isnan(b[i]))) ++c.differ;
    const double d = std::fabs(double(a[i]) - double(b[i]));
    c.max_abs = std::max(c.max_abs, d);
    c.max_rel = std::max(c.max_rel, d / std::max(1e-30, std::fabs(double(b[i]))));
  }
  return c;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) return 1;
  int hdr[5];
  if (std::fread(hdr, sizeof(int), 5, f) != 5) return 1;
  const int T = hdr[0], H = hdr[1], hd = hdr[2], n = hdr[3], w = H * hd;
  auto get = [&](size_t count) {
    std::vector<float> x(count);
    if (std::fread(x.data(), sizeof(float), count, f) != count) throw std::runtime_error("short read");
    return x;
  };
  auto q = get(T * w), k = get(T * w), v = get(T * w), u = get(T * n), A = get(n * n), B = get(n), C = get(n), D = get(1);
  const char* names[] = {"prefill", "decode", "mixed", "spec_rewind", "reset"};
  std::vector<std::vector<float>> la, ss;
  for (int i = 0; i < 5; ++i) la.push_back(get(T * w));
  for (int i = 0; i < 5; ++i) ss.push_back(get(T * n));
  std::fclose(f);

  // paperkiln: prefill, and decode as the last row of each prefix.
  const auto pk_la = pk_kimi(q, k, v, T, H, hd);
  std::vector<float> pk_la_dec, pk_ss_dec;
  for (int t = 1; t <= T; ++t) {
    auto o = pk_kimi(q, k, v, t, H, hd);
    pk_la_dec.insert(pk_la_dec.end(), o.end() - w, o.end());
    auto y = pk_ssm(u, A, B, C, D, t, n);
    pk_ss_dec.insert(pk_ss_dec.end(), y.end() - n, y.end());
  }
  const auto pk_ss = pk_ssm(u, A, B, C, D, T, n);

  bool ok = true;
  auto report = [&](const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    const Cmp c = cmp(a, b);
    const bool pass = c.differ == 0;
    ok = ok && pass && c.max_rel < 1e-6;
    std::printf("%-44s elements %6zu  differing %zu  max|d| %.3g  max rel %.3g  %s\n", what, a.size(), c.differ, c.max_abs,
                c.max_rel, pass ? "BIT-IDENTICAL" : "DIFFERS");
  };
  std::printf("T=%d heads=%d head_dim=%d (width %d), ssm n=%d\n", T, H, hd, w, n);
  report("paperkiln kimi: prefill vs decode (prefixes)", pk_la, pk_la_dec);
  report("paperkiln ssm_scan: prefill vs decode", pk_ss, pk_ss_dec);
  for (int i = 0; i < 5; ++i) {
    report((std::string("ember linear_attention ") + names[i] + " vs pk prefill").c_str(), la[i], pk_la);
    report((std::string("ember ssm_scan ") + names[i] + " vs pk prefill").c_str(), ss[i], pk_ss);
  }
  std::printf("%s\n", ok ? "OP GATE PASS" : "OP GATE FAIL");
  return ok ? 0 : 1;
}
