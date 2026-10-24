// ir_numerics_test: IR_SPEC 4.2's NaN and signed-zero results for ember's EXACT CPU ops, bit for bit.
//
//   relu(x)          x where x > 0, else +0: relu(NaN) = relu(-0) = +0 (5.9; the op and expr's relu)
//   max(a, b)        a where a >= b, else b: max(NaN, b) = b, max(a, NaN) = NaN, max(-0, +0) = -0 (5.19)
//   min(a, b)        a where a <= b, else b
//   where(c, a, b)   a where c > 0, else b: a NaN or -0 condition selects b
//   rowsum(e)        from +0 in ascending columns: a row of -0 sums to +0; rowmean likewise
//   ssm_scan         s_t[i] summed left to right from +0 (6.5): every term -0 gives +0
//   add, sub, mul_col, affine, scale   IEEE's signs (x + -0 = x, (-0) + (-0) = -0, (+0) + (-0) = +0)
//   depth_attend     S = 1 is s_0 bit for bit (NaN payloads and -0 included)
// A NaN result matches any NaN (4.2). Prints one JSON object; exit status 1 on any failure.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph_ir.h"
#include "graph_ir_expr.h"
#include "graph_ir_ops.h"

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

uint32_t bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

std::string show(float f) {
  if (std::isnan(f)) return "NaN";
  if (f == 0.0f) return std::signbit(f) ? "-0" : "+0";
  return nlohmann::json(f).dump();
}

bool same(float got, float want) { return (std::isnan(got) && std::isnan(want)) || bits(got) == bits(want); }

nlohmann::json results = nlohmann::json::array();
int failures = 0;

void expect(const std::string& what, const std::vector<float>& got, const std::vector<float>& want) {
  bool ok = got.size() == want.size();
  std::vector<std::string> g, w;
  for (size_t i = 0; i < got.size(); ++i) {
    g.push_back(show(got[i]));
    if (i < want.size()) {
      w.push_back(show(want[i]));
      ok = ok && same(got[i], want[i]);
    }
  }
  failures += !ok;
  results.push_back({{"case", what}, {"pass", ok}, {"got", g}, {"want", w}});
}

std::vector<float> run_expr(const std::string& text, const std::vector<std::vector<float>>& ins, int T, int w) {
  std::vector<const float*> p;
  for (const auto& v : ins) p.push_back(v.data());
  std::vector<float> out(static_cast<size_t>(T) * w);
  ember_ir::expr::eval(ember_ir::expr::cached(text, ins.size()), p, T, w, out.data());
  return out;
}

}  // namespace

int main() {
  using namespace ember_ir;
  const std::vector<float> xs = {kNaN, -0.0f, 0.0f, -1.0f, 2.0f, -kNaN, kInf, -kInf};
  {
    std::vector<float> out(xs.size());
    cpu::relu(xs.data(), out.data(), xs.size());
    expect("relu op", out, {0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, kInf, 0.0f});
    expect("expr relu", run_expr("relu(x)", {xs}, 1, static_cast<int>(xs.size())), out);
  }
  {
    const std::vector<float> a = {kNaN, 1.0f, -0.0f, 0.0f, kNaN, 3.0f};
    const std::vector<float> b = {1.0f, kNaN, 0.0f, -0.0f, kNaN, 3.0f};
    expect("expr max", run_expr("max(x, a)", {a, b}, 1, 6), {1.0f, kNaN, -0.0f, 0.0f, kNaN, 3.0f});
    expect("expr min", run_expr("min(x, a)", {a, b}, 1, 6), {1.0f, kNaN, -0.0f, 0.0f, kNaN, 3.0f});
    const std::vector<float> c = {kNaN, -0.0f, 0.0f, 1.0f, -1.0f, kInf};
    expect("expr where", run_expr("where(x, a, b)", {c, std::vector<float>(6, 7.0f), std::vector<float>(6, 9.0f)}, 1, 6),
           {9.0f, 9.0f, 9.0f, 7.0f, 9.0f, 7.0f});
    // Constants are folded with the same operations.
    expect("expr max folded", run_expr("x * 0 + max(0 - 0 * 1, 0)", {std::vector<float>{1.0f}}, 1, 1), {0.0f});
    expect("expr where folded NaN", run_expr("where(0 / 0, 1, 2) + 0 * x", {std::vector<float>{1.0f}}, 1, 1), {2.0f});
  }
  {
    const std::vector<float> negz(4, -0.0f);
    expect("expr rowsum of -0", run_expr("rowsum(x)", {negz}, 1, 4), std::vector<float>(4, 0.0f));
    expect("expr rowmean of -0", run_expr("rowmean(x)", {negz}, 1, 4), std::vector<float>(4, 0.0f));
  }
  {
    // ssm_scan with every term -0: A = 1, s = -0, b = 1, u = -0, c = 1, d = 1.
    const int n = 3;
    std::vector<float> A(n * n, 1.0f), B(n, 1.0f), C(n, 1.0f), s(n, -0.0f), u(n, -0.0f), y(n), scratch(n);
    ssm_scan_row_cpu(u.data(), A.data(), B.data(), C.data(), 1.0f, s.data(), scratch.data(), y.data(), n);
    expect("ssm_scan state from +0", s, std::vector<float>(n, 0.0f));
    expect("ssm_scan output", y, std::vector<float>(n, 0.0f));
  }
  {
    const std::vector<float> a = {-0.0f, -0.0f, 0.0f, 5.0f}, b = {-0.0f, 0.0f, -0.0f, -0.0f};
    std::vector<float> out(4);
    cpu::sub(a.data(), b.data(), out.data(), 4);
    expect("sub", out, {0.0f, -0.0f, 0.0f, 5.0f});
    std::vector<float> col = {-0.0f}, mc(4);
    cpu::mul_col(std::vector<float>{1.0f, -1.0f, 0.0f, kNaN}.data(), col.data(), mc.data(), 1, 4);
    expect("mul_col", mc, {-0.0f, 0.0f, -0.0f, kNaN});
    std::vector<float> af(2);
    cpu::affine(std::vector<float>{-0.0f, -0.0f}.data(), af.data(), 1, 1.0f, 0.0f);
    cpu::affine(std::vector<float>{-0.0f}.data(), af.data() + 1, 1, 1.0f, -0.0f);
    expect("affine (x * scale) + shift", af, {0.0f, -0.0f});
    const std::vector<float> g = {1.0f, 1.0f}, bias = {0.0f, -0.0f};
    std::vector<float> sc(2), sn(2);
    cpu::scale(std::vector<float>{-0.0f, -0.0f}.data(), g.data(), bias.data(), sc.data(), 1, 2);
    cpu::scale(std::vector<float>{-0.0f, -0.0f}.data(), g.data(), nullptr, sn.data(), 1, 2);
    expect("scale with bias", sc, {0.0f, -0.0f});
    expect("scale without bias", sn, {-0.0f, -0.0f});
  }
  {
    const std::vector<float> s0 = {-0.0f, kNaN, 1.5f, -kNaN};
    std::vector<float> out(4);
    cpu::depth_attend({s0.data()}, nullptr, out.data(), 1, 4, 1e-5f);
    bool payload = true;
    for (size_t i = 0; i < 4; ++i) payload = payload && bits(out[i]) == bits(s0[i]);
    expect("depth_attend S = 1 is s_0", out, s0);
    results.push_back({{"case", "depth_attend S = 1 keeps NaN payloads"}, {"pass", payload}});
    failures += !payload;
  }
  nlohmann::json report{{"cases", results}, {"failures", failures}, {"pass", failures == 0}};
  std::cout << report.dump(1) << "\n";
  return failures ? 1 : 0;
}
