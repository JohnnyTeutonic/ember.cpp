// CPU parity gate: ember's tuned llama path against the IR interpreter (EMBER_INTERPRETER=1, the IR built from
// the file's metadata with EMBER_IR_BUILD=1) on the first-token logits of a tiny llama GGUF that this test
// writes itself (no download): vocabulary 64, width 64, 2 blocks, 4 query heads over 2 key/value heads (GQA),
// feed-forward 128, context 64, f32 weights from a fixed xorshift seed. For each prompt the last row's logits
// (TinyLlamaSession::score_rows, the forward /score serves) must agree within 1e-4 absolute, and the argmax
// must be the same. 1e-4 is the bound gate 1 recorded for llama-in-IR against the llama path on a real
// model (experiments/ir_2026-10-20: max 1.0e-4, 3.6e-6 of the logit range, top-1 40/40); a tiny model has
// fewer and shorter accumulations, so it should sit well inside it. The measured maximum is printed.
// Exit 1 on failure. Runs in well under a second.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "api.h"

namespace fs = std::filesystem;

namespace {

constexpr int kVocab = 64, kEmbd = 64, kLayers = 2, kHeads = 4, kKvHeads = 2, kFf = 128, kCtx = 64;
constexpr int kHeadDim = kEmbd / kHeads, kKvDim = kKvHeads * kHeadDim;
constexpr double kTolerance = 1e-4;

// GGUF v3 writer: only the value types this fixture needs.
struct Writer {
  std::string out;
  void u32(uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); }
  void i32(int32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); }
  void u64(uint64_t v) { out.append(reinterpret_cast<const char*>(&v), 8); }
  void f32(float v) { out.append(reinterpret_cast<const char*>(&v), 4); }
  void str(const std::string& s) { u64(s.size()); out += s; }
  enum : uint32_t { UINT32 = 4, INT32 = 5, FLOAT32 = 6, STRING = 8, ARRAY = 9 };
  void kv_u32(const std::string& k, uint32_t v) { str(k); u32(UINT32); u32(v); }
  void kv_f32(const std::string& k, float v) { str(k); u32(FLOAT32); f32(v); }
  void kv_str(const std::string& k, const std::string& v) { str(k); u32(STRING); str(v); }
};

struct Tensor {
  std::string name;
  std::vector<uint64_t> ne;  // ne[0] contiguous
  std::vector<float> data;
};

uint64_t rng_state = 0x9e3779b97f4a7c15ull;
float uniform(float a) {  // xorshift64*, in [-a, a)
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  const uint64_t r = rng_state * 0x2545f4914f6cdd1dull;
  return static_cast<float>((static_cast<double>(r >> 11) / 9007199254740992.0) * 2.0 - 1.0) * a;
}

Tensor make(const std::string& name, std::vector<uint64_t> ne, float scale, float offset = 0.0f) {
  Tensor t{name, ne, {}};
  uint64_t n = 1;
  for (uint64_t d : ne) n *= d;
  t.data.resize(n);
  for (auto& x : t.data) x = offset + uniform(scale);
  return t;
}

void write_fixture(const std::string& path) {
  std::vector<Tensor> ts;
  ts.push_back(make("token_embd.weight", {kEmbd, kVocab}, 1.0f));
  for (int l = 0; l < kLayers; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    ts.push_back(make(b + "attn_norm.weight", {kEmbd}, 0.1f, 1.0f));
    ts.push_back(make(b + "attn_q.weight", {kEmbd, kEmbd}, 0.2f));
    ts.push_back(make(b + "attn_k.weight", {kEmbd, kKvDim}, 0.2f));
    ts.push_back(make(b + "attn_v.weight", {kEmbd, kKvDim}, 0.2f));
    ts.push_back(make(b + "attn_output.weight", {kEmbd, kEmbd}, 0.2f));
    ts.push_back(make(b + "ffn_norm.weight", {kEmbd}, 0.1f, 1.0f));
    ts.push_back(make(b + "ffn_gate.weight", {kEmbd, kFf}, 0.2f));
    ts.push_back(make(b + "ffn_up.weight", {kEmbd, kFf}, 0.2f));
    ts.push_back(make(b + "ffn_down.weight", {kFf, kEmbd}, 0.15f));
  }
  ts.push_back(make("output_norm.weight", {kEmbd}, 0.1f, 1.0f));
  ts.push_back(make("output.weight", {kEmbd, kVocab}, 0.3f));

  Writer w;
  w.out = "GGUF";
  w.u32(3);
  w.u64(ts.size());
  w.u64(18);  // metadata pairs below
  w.kv_str("general.architecture", "llama");
  w.kv_str("general.name", "ember-ci-tiny-llama");
  w.kv_u32("general.alignment", 32);
  w.kv_u32("llama.vocab_size", kVocab);
  w.kv_u32("llama.context_length", kCtx);
  w.kv_u32("llama.embedding_length", kEmbd);
  w.kv_u32("llama.block_count", kLayers);
  w.kv_u32("llama.feed_forward_length", kFf);
  w.kv_u32("llama.attention.head_count", kHeads);
  w.kv_u32("llama.attention.head_count_kv", kKvHeads);
  w.kv_u32("llama.rope.dimension_count", kHeadDim);
  w.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
  w.kv_f32("llama.rope.freq_base", 10000.0f);
  w.kv_str("tokenizer.ggml.model", "llama");
  w.kv_u32("tokenizer.ggml.bos_token_id", 1);
  w.kv_u32("tokenizer.ggml.eos_token_id", 2);
  w.str("tokenizer.ggml.tokens");
  w.u32(Writer::ARRAY);
  w.u32(Writer::STRING);
  w.u64(kVocab);
  for (int i = 0; i < kVocab; ++i)
    w.str(i == 0 ? "<unk>" : i == 1 ? "<s>" : i == 2 ? "</s>" : "\xE2\x96\x81t" + std::to_string(i));
  w.str("tokenizer.ggml.token_type");
  w.u32(Writer::ARRAY);
  w.u32(Writer::INT32);
  w.u64(kVocab);
  for (int i = 0; i < kVocab; ++i) w.i32(i == 0 ? 2 : i < 3 ? 3 : 1);

  uint64_t offset = 0;
  std::vector<uint64_t> offsets;
  for (const auto& t : ts) {
    w.str(t.name);
    w.u32(static_cast<uint32_t>(t.ne.size()));
    for (uint64_t d : t.ne) w.u64(d);
    w.u32(0);  // GGML_TYPE_F32
    w.u64(offset);
    offsets.push_back(offset);
    offset += (t.data.size() * 4 + 31) / 32 * 32;
  }
  while (w.out.size() % 32) w.out += '\0';
  const size_t base = w.out.size();
  for (size_t i = 0; i < ts.size(); ++i) {
    w.out.resize(base + offsets[i], '\0');
    w.out.append(reinterpret_cast<const char*>(ts[i].data.data()), ts[i].data.size() * 4);
  }
  w.out.resize(base + offset, '\0');
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(w.out.data(), static_cast<std::streamsize>(w.out.size()));
}

void set_env(const char* k, const char* v) {
#ifdef _WIN32
  _putenv_s(k, v ? v : "");
#else
  if (v) setenv(k, v, 1);
  else unsetenv(k);
#endif
}

}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() / ("ember_llama_ir_parity_" + std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  const std::string gguf = (dir / "tiny_llama.gguf").string();
  int failures = 0;
  try {
    write_fixture(gguf);
    for (const char* k : {"EMBER_IR", "EMBER_IR_ONLY", "EMBER_KV", "EMBER_DEQUANT_AT_LOAD", "EMBER_SPEC"})
      set_env(k, nullptr);
    set_env("EMBER_INTERPRETER", "0");
    set_env("EMBER_IR_BUILD", nullptr);
    tinyllama::TinyLlamaSession llama(gguf, "tokenizer.json", 1, 0, true);
    set_env("EMBER_INTERPRETER", "1");
    set_env("EMBER_IR_BUILD", "1");
    tinyllama::TinyLlamaSession interp(gguf, "tokenizer.json", 1, 0, true);
    set_env("EMBER_INTERPRETER", nullptr);
    set_env("EMBER_IR_BUILD", nullptr);
    if (!llama.ir_document().empty() || interp.ir_document().empty()) {
      std::fprintf(stderr, "FAIL: the two sessions do not run the two paths (llama IR %zu bytes, interpreter %zu)\n",
                   llama.ir_document().size(), interp.ir_document().size());
      return 1;
    }
    std::vector<std::vector<int>> prompts = {{1}, {1, 7, 12, 3}, {1, 40, 41, 42, 5, 9, 63, 17, 33}};
    std::vector<int> long_prompt = {1};
    for (int i = 0; i < 47; ++i) long_prompt.push_back(3 + (i * 37) % 61);
    prompts.push_back(long_prompt);
    double worst = 0.0;
    for (const auto& ids : prompts) {
      const int last = static_cast<int>(ids.size()) - 1;
      const std::vector<float> a = llama.score_rows(ids, last, false);
      const std::vector<float> b = interp.score_rows(ids, last, false);
      if (a.size() != static_cast<size_t>(kVocab) || b.size() != a.size()) {
        std::fprintf(stderr, "FAIL: %zu ids: logits sizes %zu and %zu\n", ids.size(), a.size(), b.size());
        ++failures;
        continue;
      }
      double diff = 0.0, lo = a[0], hi = a[0];
      for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) diff = INFINITY;
        diff = std::max(diff, std::fabs(static_cast<double>(a[i]) - b[i]));
        lo = std::min(lo, static_cast<double>(a[i]));
        hi = std::max(hi, static_cast<double>(a[i]));
      }
      const auto am = std::max_element(a.begin(), a.end()) - a.begin();
      const auto bm = std::max_element(b.begin(), b.end()) - b.begin();
      worst = std::max(worst, diff);
      const bool ok = diff <= kTolerance && am == bm && hi - lo > 0.5;
      std::printf("%s %2zu ids: max |llama - interpreter| %.3g (logit range %.3g), argmax %ld / %ld\n",
                  ok ? "PASS" : "FAIL", ids.size(), diff, hi - lo, static_cast<long>(am), static_cast<long>(bm));
      failures += !ok;
    }
    std::printf("worst %.3g, tolerance %.0e\n", worst, kTolerance);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    failures = 1;
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
  return failures ? 1 : 0;
}
