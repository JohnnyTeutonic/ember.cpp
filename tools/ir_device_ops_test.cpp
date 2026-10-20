// The IR interpreter's device kernels for the gated / attention-residual ops
// and the recurrent ops against the CPU forms, on random inputs.
//  1. Kernels (graph_ir_cuda.h vs graph_ir_ops.h and graph_ir.cpp's row
//     functions), T = 1 and T = 7: sub, relu (with -0 and NaN inputs),
//     sigmoid, affine, mul_col, rms_row, depth_attend with 2 and 5 sources,
//     linear_attention and ssm_scan as one T-row prefill and as T one-row
//     decode steps (which must equal each other bit for bit).
//  2. The whole Interpreter, GPU (use_gpu, so T = 1 calls replay the CUDA
//     graph) vs CPU on in-memory GGUFs: a 4-block gated / attention-residual
//     IR (list, append, depth_attend with 1, 2, 3, 4 and 5 sources, with and
//     without an extra input, and every gating op) as a prefill and as decode
//     steps; and the linear_attention and ssm_scan IRs of
//     recurrent_state_test.cpp through prefill, decode, half prefill then
//     decode, a speculative verify resumed after 3 of 5 rows (snapshot
//     restore), another sequence (reset) and the original again, with
//     resume_point and the throw at a position with no snapshot. On the GPU
//     every pattern must equal its prefill bit for bit.
// Pass: |gpu - cpu| <= 5e-5 * max(1, max |cpu|) per tensor (paperkiln's
// ROUNDING bound), and the EXACT ops (sub, relu, affine, mul_col; list and
// append inside the interpreter) bit for bit.
// Build (from the CUDA build directory, after make tinyllama_server; JSON_INC
// holds nlohmann/json.hpp):
//   g++ -O2 -std=c++17 -fopenmp -DHAS_CUDA -I EMBER -I JSON_INC -I /usr/local/cuda/include
//       EMBER/tools/ir_device_ops_test.cpp libtinyllama_core.a -L /usr/local/cuda/lib64
//       -lcudart -lcublas -lboost_regex -lpthread -o ir_device_ops_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "gguf_structs.h"
#include "graph_ir.h"
#include "graph_ir_cuda.h"
#include "graph_ir_ops.h"

using nlohmann::json;

namespace {

int failures = 0;

// ROUNDING: max |a - b| <= 5e-5 * max(1, max |ref|); exact: bit for bit.
void check(const char* what, const std::vector<float>& ref, const std::vector<float>& got, bool exact) {
  double worst = 0.0, top = 0.0;
  bool bits = ref.size() == got.size() && std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0;
  bool ok = ref.size() == got.size();
  for (size_t i = 0; ok && i < ref.size(); ++i) {
    if (std::isnan(ref[i]) || std::isnan(got[i])) {
      ok = std::isnan(ref[i]) && std::isnan(got[i]);
      continue;
    }
    worst = std::max(worst, std::fabs(static_cast<double>(ref[i]) - got[i]));
    top = std::max(top, std::fabs(static_cast<double>(ref[i])));
  }
  const double tol = 5e-5 * std::max(1.0, top);
  ok = ok && (exact ? bits : worst <= tol);
  if (!ok) ++failures;
  std::printf("  %-48s %s  max |d| %.3g (tol %.3g)%s\n", what, ok ? "ok  " : "FAIL", worst, tol,
              bits ? ", bit for bit" : "");
}

struct Dev {  // a device copy of a host vector
  float* p = nullptr;
  size_t n = 0;
  explicit Dev(const std::vector<float>& v) : n(v.size()) {
    cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float));
    cudaMemcpy(p, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  }
  explicit Dev(size_t count) : n(count) { cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float)); }
  ~Dev() { cudaFree(p); }
  std::vector<float> get() const {
    std::vector<float> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
  }
};

std::vector<float> rnd(std::mt19937& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = u(rng);
  return v;
}

void kernels(int T) {
  std::printf("kernels, T = %d\n", T);
  std::mt19937 rng(7 + T);
  const int w = 96, d = 64;
  const size_t n = static_cast<size_t>(T) * w;
  cudaStream_t st = nullptr;
  auto a = rnd(rng, n, -3.f, 3.f), b = rnd(rng, n, -3.f, 3.f), c = rnd(rng, T, -2.f, 2.f);
  a[0] = -0.0f;
  a[n > 2 ? 2 : 0] = 0.0f;
  std::vector<float> an = a;  // relu gives +0 for NaN (std::max), sigmoid NaN
  an[1 % n] = std::numeric_limits<float>::quiet_NaN();
  Dev da(a), dan(an), db(b), dc(c), dout(n);
  std::vector<float> ref(n);
  ember_ir::cpu::sub(a.data(), b.data(), ref.data(), n);
  ember_ir::ir_binary_cuda(da.p, db.p, dout.p, n, 2, st);
  check("sub", ref, dout.get(), true);
  ember_ir::cpu::relu(an.data(), ref.data(), n);
  ember_ir::ir_relu_cuda(dan.p, dout.p, n, st);
  check("relu (-0, 0 and NaN included)", ref, dout.get(), true);
  ember_ir::cpu::sigmoid(an.data(), ref.data(), n);
  ember_ir::ir_sigmoid_cuda(dan.p, dout.p, n, st);
  check("sigmoid (NaN included)", ref, dout.get(), false);
  ember_ir::cpu::affine(a.data(), ref.data(), n, -1.7f, 0.3f);
  ember_ir::ir_affine_cuda(da.p, dout.p, n, -1.7f, 0.3f, st);
  check("affine", ref, dout.get(), true);
  ember_ir::cpu::mul_col(b.data(), c.data(), ref.data(), T, w);
  ember_ir::ir_mul_col_cuda(db.p, dc.p, dout.p, T, w, st);
  check("mul_col", ref, dout.get(), true);
  {
    std::vector<float> r(T);
    Dev dr(static_cast<size_t>(T));
    ember_ir::cpu::rms_row(b.data(), r.data(), T, w, 1e-6f);
    ember_ir::ir_rms_row_cuda(db.p, dr.p, T, w, 1e-6f, st);
    check("rms_row", r, dr.get(), false);
  }
  for (int S : {2, 5}) {
    std::vector<std::vector<float>> src;
    std::vector<const float*> hsrc;
    std::vector<Dev*> dsrc;
    std::vector<const float*> dptr;
    for (int i = 0; i < S; ++i) src.push_back(rnd(rng, static_cast<size_t>(T) * d, -2.f, 2.f));
    for (int i = 0; i < S; ++i) {
      hsrc.push_back(src[i].data());
      dsrc.push_back(new Dev(src[i]));
      dptr.push_back(dsrc.back()->p);
    }
    const auto q = rnd(rng, d, -3.f, 3.f);  // sharp enough that the softmax is not flat
    Dev dq(q), dd(static_cast<size_t>(T) * d);
    std::vector<float> r(static_cast<size_t>(T) * d);
    ember_ir::cpu::depth_attend(hsrc, q.data(), r.data(), T, d, 1e-5f);
    ember_ir::ir_depth_attend_cuda(dptr.data(), S, dq.p, dd.p, T, d, 1e-5f, st);
    check(("depth_attend, " + std::to_string(S) + " sources").c_str(), r, dd.get(), false);
    for (Dev* p : dsrc) delete p;
  }
  {  // linear_attention: CPU per row; device prefill (one launch) and decode (a launch per row)
    const int lw = 64;
    const size_t ln = static_cast<size_t>(T) * lw;
    auto q = rnd(rng, ln, -4.f, 4.f), k = rnd(rng, ln, -4.f, 4.f), v = rnd(rng, ln, -4.f, 4.f);
    k[0] = 0.0f;
    std::vector<float> num(lw, 0.0f), den(lw, 0.0f), out(ln);
    for (int t = 0; t < T; ++t) {
      const size_t r = static_cast<size_t>(t) * lw;
      ember_ir::linear_attention_row_cpu(q.data() + r, k.data() + r, v.data() + r, num.data(), den.data(),
                                         out.data() + r, lw, 1e-8f);
    }
    Dev dq(q), dk(k), dv(v);
    Dev pn(std::vector<float>(lw, 0.0f)), pd(std::vector<float>(lw, 0.0f)), po(ln);
    ember_ir::ir_linear_attention_cuda(dq.p, dk.p, dv.p, pn.p, pd.p, po.p, T, lw, 1e-8f, st);
    Dev sn(std::vector<float>(lw, 0.0f)), sd(std::vector<float>(lw, 0.0f)), so(ln);
    for (int t = 0; t < T; ++t) {
      const size_t r = static_cast<size_t>(t) * lw;
      ember_ir::ir_linear_attention_cuda(dq.p + r, dk.p + r, dv.p + r, sn.p, sd.p, so.p + r, 1, lw, 1e-8f, st);
    }
    check("linear_attention prefill: out", out, po.get(), false);
    check("linear_attention prefill: num", num, pn.get(), false);
    check("linear_attention prefill: den", den, pd.get(), false);
    check("linear_attention decode steps == prefill: out", po.get(), so.get(), true);
    check("linear_attention decode steps == prefill: num", pn.get(), sn.get(), true);
  }
  {  // ssm_scan, n = 48 (threads over i) and n = 300 (several i per thread)
    for (int sn : {48, 300}) {
      const size_t un = static_cast<size_t>(T) * sn;
      auto u = rnd(rng, un, -4.f, 4.f), A = rnd(rng, static_cast<size_t>(sn) * sn, -0.3f / std::sqrt(sn / 16.f),
                                                0.3f / std::sqrt(sn / 16.f));
      auto B = rnd(rng, sn, -1.f, 1.f), C = rnd(rng, sn, -1.f, 1.f), D = rnd(rng, 1, -1.f, 1.f);
      std::vector<float> s(sn, 0.0f), scratch(sn), y(un);
      for (int t = 0; t < T; ++t)
        ember_ir::ssm_scan_row_cpu(u.data() + static_cast<size_t>(t) * sn, A.data(), B.data(), C.data(), D[0],
                                   s.data(), scratch.data(), y.data() + static_cast<size_t>(t) * sn, sn);
      Dev du(u), dA(A), dB(B), dC(C), dD(D);
      Dev ps(std::vector<float>(sn, 0.0f)), py(un), ss(std::vector<float>(sn, 0.0f)), sy(un);
      ember_ir::ir_ssm_scan_cuda(du.p, dA.p, dB.p, dC.p, dD.p, ps.p, py.p, T, sn, st);
      for (int t = 0; t < T; ++t)
        ember_ir::ir_ssm_scan_cuda(du.p + static_cast<size_t>(t) * sn, dA.p, dB.p, dC.p, dD.p, ss.p,
                                   sy.p + static_cast<size_t>(t) * sn, 1, sn, st);
      const std::string tag = "ssm_scan n " + std::to_string(sn);
      check((tag + " prefill: y").c_str(), y, py.get(), false);
      check((tag + " prefill: s").c_str(), s, ps.get(), false);
      check((tag + " decode steps == prefill: y").c_str(), py.get(), sy.get(), true);
      check((tag + " decode steps == prefill: s").c_str(), ps.get(), ss.get(), true);
    }
  }
  if (cudaGetLastError() != cudaSuccess) {
    std::printf("  CUDA error\n");
    ++failures;
  }
}

void add_tensor(GGUFData& g, const std::string& name, const std::vector<uint64_t>& shape, const std::vector<float>& v) {
  GGUFTensorInfo info;
  info.name = name;
  info.shape = shape;
  info.type = GGMLType::GGML_TYPE_F32;
  info.offset = g.tensor_data.size();
  info.num_elements = v.size();
  info.size_in_bytes = v.size() * sizeof(float);
  g.tensor_data.resize(g.tensor_data.size() + info.size_in_bytes);
  std::memcpy(g.tensor_data.data() + info.offset, v.data(), info.size_in_bytes);
  g.tensor_infos.push_back(info);
  g.tensor_infos_map[name] = info;
}

std::vector<float> eye(int w) {
  std::vector<float> e(static_cast<size_t>(w) * w, 0.0f);
  for (int i = 0; i < w; ++i) e[static_cast<size_t>(i) * w + i] = 1.0f;
  return e;
}

std::vector<int> range(int a, int b) {
  std::vector<int> v;
  for (int i = a; i < b; ++i) v.push_back(i);
  return v;
}

// recurrent_state_test.cpp's call patterns, each returning [rows, w].
struct Runs {
  std::vector<float> prefill, decode, mixed, spec, reset;
};

Runs run_all(ember_ir::Interpreter& ip, int T, int w) {
  Runs r;
  r.prefill = ip.forward(range(0, T), 0, /*all_rows=*/true);
  for (int t = 0; t < T; ++t) {
    auto o = ip.forward({t}, t);
    r.decode.insert(r.decode.end(), o.begin(), o.end());
  }
  {
    const int h = T / 2;
    r.mixed = ip.forward(range(0, h), 0, true);
    for (int t = h; t < T; ++t) {
      auto o = ip.forward({t}, t);
      r.mixed.insert(r.mixed.end(), o.begin(), o.end());
    }
  }
  {
    auto a = ip.forward(range(0, 10), 0, true);
    auto b = ip.forward(range(10, 15), 10, true);
    if (ip.resume_point(13) != 13) throw std::runtime_error("resume_point(13) after a 5-row verify is not 13");
    r.spec = a;
    r.spec.insert(r.spec.end(), b.begin(), b.begin() + 3 * w);
    for (int t = 13; t < T; ++t) {
      auto o = ip.forward({t}, t);
      r.spec.insert(r.spec.end(), o.begin(), o.end());
    }
  }
  {
    std::vector<int> rev;
    for (int t = T - 1; t >= 0; --t) rev.push_back(t);
    ip.forward(rev, 0);
    if (ip.resume_point(5) != 0) throw std::runtime_error("resume_point(5) should fall back to 0");
    if (ip.resume_point(T - 1) != T - 1) throw std::runtime_error("resume_point(T-1) should be the prompt snapshot");
    bool threw = false;
    try {
      ip.forward({3}, 3);
    } catch (const std::exception&) {
      threw = true;
    }
    if (!threw) throw std::runtime_error("forward at a position with no snapshot did not throw");
    r.reset = ip.forward(range(0, T), 0, true);
  }
  return r;
}

void compare_runs(const char* name, const Runs& cpu, const Runs& gpu) {
  std::printf("interpreter, %s: GPU vs CPU\n", name);
  const char* names[] = {"prefill", "decode (graph)", "half prefill, decode", "speculative rewind", "reset"};
  const std::vector<float>* c[] = {&cpu.prefill, &cpu.decode, &cpu.mixed, &cpu.spec, &cpu.reset};
  const std::vector<float>* g[] = {&gpu.prefill, &gpu.decode, &gpu.mixed, &gpu.spec, &gpu.reset};
  for (int i = 0; i < 5; ++i) check(names[i], *c[i], *g[i], false);
  for (int i = 1; i < 5; ++i) check((std::string("GPU ") + names[i] + " == GPU prefill").c_str(), gpu.prefill, *g[i], true);
}

void recurrent_interpreters() {
  const int T = 64, H = 4, hd = 8, w = H * hd, n = 16;
  std::mt19937 rng(20261020);
  std::uniform_real_distribution<float> big(-4.0f, 4.0f), unit(-1.0f, 1.0f), small(-0.3f, 0.3f);
  std::vector<float> q(T * w), k(T * w), v(T * w), u(T * n), A(n * n), B(n), C(n), D(1);
  for (auto* x : {&q, &k, &v})
    for (auto& e : *x) e = big(rng);
  for (int c = 0; c < w; c += 7) k[c] = 0.0f;
  for (auto& e : u) e = big(rng);
  for (auto& e : A) e = small(rng);
  for (auto& e : B) e = unit(rng);
  for (auto& e : C) e = unit(rng);
  D[0] = unit(rng);
  {
    GGUFData g;
    std::vector<float> tok(static_cast<size_t>(T) * 3 * w);
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < w; ++c) {
        tok[static_cast<size_t>(t) * 3 * w + c] = q[t * w + c];
        tok[static_cast<size_t>(t) * 3 * w + w + c] = k[t * w + c];
        tok[static_cast<size_t>(t) * 3 * w + 2 * w + c] = v[t * w + c];
      }
    add_tensor(g, "tok", {static_cast<uint64_t>(3 * w), static_cast<uint64_t>(T)}, tok);
    add_tensor(g, "eye", {static_cast<uint64_t>(w), static_cast<uint64_t>(w)}, eye(w));
    json ir = {
        {"ir_version", 0},
        {"family_hint", "optest"},
        {"model", {{"vocab", w}, {"d", 3 * w}, {"n_layers", 1}, {"ctx", T}}},
        {"embed", {{"nodes", json::array({{{"op", "embedding"}, {"in", {"tokens"}}, {"weight", "tok"}, {"out", {"x"}}}})}}},
        {"block",
         {{"repeat", 1},
          {"state", json::array({{{"name", "lin"},
                                  {"kind", "recurrent"},
                                  {"op", "linear_attention"},
                                  {"dtype", "f32"},
                                  {"tensors", json::array({{{"name", "num"}, {"shape", {H, hd}}, {"init", 0}},
                                                           {{"name", "den"}, {"shape", {H, hd}}, {"init", 0}}})}}})},
          {"nodes", json::array({{{"op", "split"}, {"in", {"x"}}, {"sizes", {w, w, w}}, {"out", {"q", "k", "v"}}},
                                 {{"op", "linear_attention"},
                                  {"in", {"q", "k", "v"}},
                                  {"state", "lin"},
                                  {"heads", H},
                                  {"head_dim", hd},
                                  {"feature_map", "elu_plus_one"},
                                  {"den_floor", 1e-8},
                                  {"causal", true},
                                  {"out", {"o"}}}})}}},
        {"head", {{"nodes", json::array({{{"op", "linear"}, {"in", {"o"}}, {"weight", "eye"}, {"out", {"logits"}}}})}}}};
    ember_ir::Interpreter cpu(ir.dump(), g, false), gpu(ir.dump(), g, true);
    if (!gpu.on_gpu()) throw std::runtime_error("linear_attention IR did not take the device path");
    compare_runs("linear_attention", run_all(cpu, T, w), run_all(gpu, T, w));
  }
  {
    GGUFData g;
    add_tensor(g, "tok", {static_cast<uint64_t>(n), static_cast<uint64_t>(T)}, u);
    add_tensor(g, "blk.0.ssm_a.weight", {static_cast<uint64_t>(n), static_cast<uint64_t>(n)}, A);
    add_tensor(g, "blk.0.ssm_b.weight", {static_cast<uint64_t>(n)}, B);
    add_tensor(g, "blk.0.ssm_c.weight", {static_cast<uint64_t>(n)}, C);
    add_tensor(g, "blk.0.ssm_d.weight", {1}, D);
    add_tensor(g, "eye", {static_cast<uint64_t>(n), static_cast<uint64_t>(n)}, eye(n));
    json ir = {
        {"ir_version", 0},
        {"family_hint", "optest"},
        {"model", {{"vocab", n}, {"d", n}, {"n_layers", 1}, {"ctx", T}}},
        {"embed", {{"nodes", json::array({{{"op", "embedding"}, {"in", {"tokens"}}, {"weight", "tok"}, {"out", {"u"}}}})}}},
        {"block",
         {{"repeat", 1},
          {"state", json::array({{{"name", "ssm"},
                                  {"kind", "recurrent"},
                                  {"op", "ssm_scan"},
                                  {"dtype", "f32"},
                                  {"tensors", json::array({{{"name", "s"}, {"shape", {n}}, {"init", 0}}})}}})},
          {"nodes", json::array({{{"op", "ssm_scan"},
                                  {"in", {"u"}},
                                  {"state", "ssm"},
                                  {"a", "blk.{i}.ssm_a.weight"},
                                  {"b", "blk.{i}.ssm_b.weight"},
                                  {"c", "blk.{i}.ssm_c.weight"},
                                  {"d", "blk.{i}.ssm_d.weight"},
                                  {"n", n},
                                  {"out", {"y"}}}})}}},
        {"head", {{"nodes", json::array({{{"op", "linear"}, {"in", {"y"}}, {"weight", "eye"}, {"out", {"logits"}}}})}}}};
    ember_ir::Interpreter cpu(ir.dump(), g, false), gpu(ir.dump(), g, true);
    if (!gpu.on_gpu()) throw std::runtime_error("ssm_scan IR did not take the device path");
    compare_runs("ssm_scan", run_all(cpu, T, n), run_all(gpu, T, n));
  }
}

// A gated, attention-residual IR: block i attends over a bank of i + 1
// tensors (1 at block 0: a copy), the head over 5; every gating op.
void gated_interpreter() {
  const int V = 40, d = 32, L = 4;
  std::mt19937 rng(5);
  GGUFData g;
  add_tensor(g, "tok", {static_cast<uint64_t>(d), static_cast<uint64_t>(V)}, rnd(rng, static_cast<size_t>(V) * d, -2.f, 2.f));
  for (int l = 0; l < L; ++l) {
    add_tensor(g, "blk." + std::to_string(l) + ".q", {static_cast<uint64_t>(d)}, rnd(rng, d, -2.f, 2.f));
    add_tensor(g, "blk." + std::to_string(l) + ".q2", {static_cast<uint64_t>(d)}, rnd(rng, d, -2.f, 2.f));
  }
  add_tensor(g, "out_q", {static_cast<uint64_t>(d)}, rnd(rng, d, -2.f, 2.f));
  add_tensor(g, "eye", {static_cast<uint64_t>(d), static_cast<uint64_t>(d)}, eye(d));
  auto node = [](const char* op, std::vector<std::string> in, const char* out) {
    return json{{"op", op}, {"in", in}, {"out", out}};
  };
  json blk = json::array();
  json da = node("depth_attend", {"bank"}, "h");
  da["weight"] = "blk.{i}.q";
  da["eps"] = 1e-5;
  blk.push_back(da);
  json rr = node("rms_row", {"h"}, "r");
  rr["eps"] = 1e-6;
  blk.push_back(rr);
  json af = node("affine", {"r"}, "r");
  af["scale"] = 0.75;
  af["shift"] = -0.5;
  blk.push_back(af);
  blk.push_back(node("sigmoid", {"r"}, "gte"));
  blk.push_back(node("relu", {"h"}, "hr"));
  blk.push_back(node("sub", {"hr", "h"}, "dd"));
  blk.push_back(node("mul_col", {"dd", "gte"}, "m"));
  json da2 = node("depth_attend", {"bank", "m"}, "h2");  // the bank plus an extra input
  da2["weight"] = "blk.{i}.q2";
  blk.push_back(da2);
  blk.push_back(node("add", {"h2", "m"}, "o"));
  blk.push_back(node("append", {"bank", "o"}, "bank"));
  json hd = node("depth_attend", {"bank"}, "xo");
  hd["weight"] = "out_q";
  json lin = node("linear", {"xo"}, "logits");
  lin["weight"] = "eye";
  json emb = node("embedding", {"tokens"}, "x");
  emb["weight"] = "tok";
  json ir = {{"ir_version", 0},
             {"family_hint", "optest_gated"},
             {"model", {{"vocab", d}, {"d", d}, {"n_layers", L}, {"ctx", 64}}},
             {"embed", {{"nodes", json::array({emb, node("list", {"x"}, "bank")})}}},
             {"block", {{"repeat", L}, {"nodes", blk}}},
             {"head", {{"nodes", json::array({hd, lin})}}}};
  ember_ir::Interpreter cpu(ir.dump(), g, false), gpu(ir.dump(), g, true);
  if (!gpu.on_gpu()) throw std::runtime_error("gated IR did not take the device path");
  std::printf("interpreter, gated / attention-residual (depth_attend over 1..5 sources): GPU vs CPU\n");
  std::vector<int> toks;
  for (int t = 0; t < 12; ++t) toks.push_back((t * 7 + 3) % V);
  check("prefill, all rows (T = 12)", cpu.forward(toks, 0, true), gpu.forward(toks, 0, true), false);
  check("prefill, last row", cpu.forward(toks, 0), gpu.forward(toks, 0), false);
  std::vector<float> cd, gd;
  for (int t = 0; t < 12; ++t) {
    auto a = cpu.forward({toks[t]}, t), b = gpu.forward({toks[t]}, t);
    cd.insert(cd.end(), a.begin(), a.end());
    gd.insert(gd.end(), b.begin(), b.end());
  }
  check("decode steps (graph)", cd, gd, false);
  check("GPU decode steps == GPU prefill rows", gpu.forward(toks, 0, true), gd, true);
}

}  // namespace

int main() {
  try {
    kernels(1);
    kernels(7);
    recurrent_interpreters();
    gated_interpreter();
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
