// Recurrent IR state, ember side: linear_attention and ssm_scan through the
// real Interpreter (an in-memory GGUF and a one-block IR) as one prefill, as
// T = 1 decode steps, as a half prefill then decode, through a speculative
// verify of 5 rows resumed after 3 (snapshot restore), and after another
// sequence (reset); also resume_point and the throw at a position with no
// snapshot. Every pattern must equal the prefill bit for bit. Writes the
// inputs and outputs to OUT.bin, which tools/recurrent_reference_check.cpp
// (built against paperkiln) checks against ops::kimi_attention and
// ops::ssm_scan.
// Build (from the build directory, after make tinyllama_server):
//   g++ -O2 -std=c++17 -fopenmp -I EMBER -I JSON_INC EMBER/tools/recurrent_state_test.cpp
//       libtinyllama_core.a -lboost_regex -lpthread -o recurrent_state_test
// Usage: recurrent_state_test OUT.bin
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph_ir.h"
#include "gguf_structs.h"

using nlohmann::json;

static void add_tensor(GGUFData& g, const std::string& name, const std::vector<uint64_t>& shape, const std::vector<float>& v) {
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

static std::vector<float> eye(int w) {
  std::vector<float> e(static_cast<size_t>(w) * w, 0.0f);
  for (int i = 0; i < w; ++i) e[static_cast<size_t>(i) * w + i] = 1.0f;
  return e;
}

static std::vector<int> range(int a, int b) {
  std::vector<int> v;
  for (int i = a; i < b; ++i) v.push_back(i);
  return v;
}

// Runs one interpreter through every call pattern; each returns [T, w] rows.
struct Runs {
  std::vector<float> prefill, decode, mixed, spec, reset;
};

static Runs run_all(ember_ir::Interpreter& ip, int T, int w, int T2_tokens_vocab_other) {
  Runs r;
  r.prefill = ip.forward(range(0, T), 0, /*all_rows=*/true);
  for (int t = 0; t < T; ++t) {
    auto o = ip.forward({t}, t);
    r.decode.insert(r.decode.end(), o.begin(), o.end());
  }
  {  // prefill half, then decode
    const int h = T / 2;
    r.mixed = ip.forward(range(0, h), 0, true);
    for (int t = h; t < T; ++t) {
      auto o = ip.forward({t}, t);
      r.mixed.insert(r.mixed.end(), o.begin(), o.end());
    }
  }
  {  // speculative pattern: prefill 10, verify 5 rows, accept 2 (resume at 13), decode on
    auto a = ip.forward(range(0, 10), 0, true);
    auto b = ip.forward(range(10, 15), 10, true);
    if (ip.resume_point(13) != 13) throw std::runtime_error("resume_point(13) after a 5-row verify is not 13");
    r.spec = a;
    r.spec.insert(r.spec.end(), b.begin(), b.begin() + 3 * w);  // rows 10, 11, 12
    for (int t = 13; t < T; ++t) {
      auto o = ip.forward({t}, t);
      r.spec.insert(r.spec.end(), o.begin(), o.end());
    }
  }
  {  // another sequence (tokens reversed), then the original again from 0
    std::vector<int> rev;
    for (int t = T2_tokens_vocab_other - 1; t >= 0; --t) rev.push_back(t);
    ip.forward(rev, 0);
    // resume_point inside the reversed sequence's prompt, with no snapshot there
    if (ip.resume_point(5) != 0) throw std::runtime_error("resume_point(5) should fall back to 0");
    if (ip.resume_point(T - 1) != T - 1) throw std::runtime_error("resume_point(T-1) should be the prompt snapshot");
    // A start with no snapshot must throw.
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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ember_optest OUT.bin\n");
    return 2;
  }
  const int T = 64, H = 4, hd = 8, w = H * hd, n = 16;
  std::mt19937 rng(20261020);
  std::uniform_real_distribution<float> big(-4.0f, 4.0f), unit(-1.0f, 1.0f), small(-0.3f, 0.3f);
  std::vector<float> q(T * w), k(T * w), v(T * w), u(T * n), A(n * n), B(n), C(n), D(1);
  for (auto* x : {&q, &k, &v})
    for (auto& e : *x) e = big(rng);
  for (int c = 0; c < w; c += 7) k[c] = 0.0f;  // the x <= 0 boundary
  for (auto& e : u) e = big(rng);
  for (auto& e : A) e = small(rng);
  for (auto& e : B) e = unit(rng);
  for (auto& e : C) e = unit(rng);
  D[0] = unit(rng);

  Runs la, ss;
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
    ember_ir::Interpreter ip(ir.dump(), g);
    la = run_all(ip, T, w, T);
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
    ember_ir::Interpreter ip(ir.dump(), g);
    ss = run_all(ip, T, n, T);
  }

  // Every call pattern leaves the same outputs bit for bit (pk_optest then
  // checks them against paperkiln's ops).
  bool same = true;
  for (const Runs* r : {&la, &ss})
    for (const auto* x : {&r->decode, &r->mixed, &r->spec, &r->reset}) {
      const bool eq = x->size() == r->prefill.size() &&
                      std::memcmp(x->data(), r->prefill.data(), x->size() * sizeof(float)) == 0;
      same = same && eq;
    }
  std::printf("ember: decode, mixed, speculative rewind and reset %s prefill bit for bit\n", same ? "equal" : "DIFFER FROM");
  FILE* f = std::fopen(argv[1], "wb");
  if (!f) return 1;
  const int hdr[5] = {T, H, hd, n, 0};
  std::fwrite(hdr, sizeof(int), 5, f);
  auto put = [&](const std::vector<float>& x) { std::fwrite(x.data(), sizeof(float), x.size(), f); };
  for (const auto* x : {&q, &k, &v, &u, &A, &B, &C, &D}) put(*x);
  for (const Runs* r : {&la, &ss})
    for (const auto* x : {&r->prefill, &r->decode, &r->mixed, &r->spec, &r->reset}) put(*x);
  std::fclose(f);
  std::printf("wrote %s (T=%d heads=%d head_dim=%d n=%d)\n", argv[1], T, H, hd, n);
  return same ? 0 : 1;
}
