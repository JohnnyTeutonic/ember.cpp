#pragma once
// Corpus draft table for speculative decoding (EMBER_DRAFT=table, experiments/draft_2026-10-27).
//
// An n-gram table of the most frequent next token, built offline from chat text by
// experiments/draft_2026-10-27/dr_table.py: for each order o = 1..orders and each context c (the last o tokens)
// it holds the most frequent next token and its share of c's occurrences (as round(255 p)). A draft walks it:
// from the history plus the draft so far, the longest order whose context is in the table gives the next token;
// the draft stops when the product of the shares falls below theta, or at k tokens. It needs no model: the
// verification pass decides every token, so the replies are plain decode's whatever the table holds.
//
// Keys (dr_table.py has the same): t0 is the most recent token; for o <= 3 the key is t0 | t1 << 16 | t2 << 32;
// for o = 4 it is mix64(t0 | t1 << 16 | t2 << 32 | t3 << 48) >> 16 (splitmix64's finaliser).
// File (little-endian): "EMBERDT1", u32 vocab_size, u32 orders, u64 entries[orders]; then per order ascending
// keys u64[n], next u16[n], conf u8[n], each array zero-padded to 8 bytes.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ember_draft {

class DraftTable {
 public:
  // Reads `path`; false (with `why` set) when the file is unreadable or malformed, or was built for another
  // vocabulary size than `vocab_size`.
  bool load(const std::string& path, int vocab_size, std::string* why) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(why, "cannot open " + path);
    buf_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (buf_.size() < 16 || std::memcmp(buf_.data(), "EMBERDT1", 8) != 0) return fail(why, "not a draft table");
    uint32_t vocab = 0, orders = 0;
    std::memcpy(&vocab, buf_.data() + 8, 4);
    std::memcpy(&orders, buf_.data() + 12, 4);
    if (orders < 1 || orders > 4) return fail(why, "orders must be 1..4");
    if (static_cast<int>(vocab) != vocab_size)
      return fail(why, "built for vocabulary size " + std::to_string(vocab) + ", the model has " +
                           std::to_string(vocab_size));
    size_t p = 16 + 8 * static_cast<size_t>(orders);
    if (buf_.size() < p) return fail(why, "truncated header");
    std::vector<uint64_t> ns(orders);
    std::memcpy(ns.data(), buf_.data() + 16, 8 * orders);
    tabs_.assign(orders, Order{});
    auto pad8 = [](size_t b) { return (b + 7) / 8 * 8; };
    for (uint32_t o = 0; o < orders; ++o) {
      const size_t n = ns[o];
      const size_t need = pad8(8 * n) + pad8(2 * n) + pad8(n);
      if (buf_.size() - p < need) return fail(why, "truncated tables");
      tabs_[o].n = n;
      tabs_[o].keys.resize(n);
      tabs_[o].next.resize(n);
      tabs_[o].conf.resize(n);
      std::memcpy(tabs_[o].keys.data(), buf_.data() + p, 8 * n);
      p += pad8(8 * n);
      std::memcpy(tabs_[o].next.data(), buf_.data() + p, 2 * n);
      p += pad8(2 * n);
      std::memcpy(tabs_[o].conf.data(), buf_.data() + p, n);
      p += pad8(n);
      for (size_t i = 0; i < n; ++i)
        if (tabs_[o].next[i] >= vocab || (i > 0 && tabs_[o].keys[i] <= tabs_[o].keys[i - 1]))
          return fail(why, "entries out of range or not ascending");
    }
    if (p != buf_.size()) return fail(why, "trailing bytes");
    buf_.clear();
    buf_.shrink_to_fit();
    return true;
  }

  int orders() const { return static_cast<int>(tabs_.size()); }

  size_t entries() const {
    size_t s = 0;
    for (const auto& t : tabs_) s += t.n;
    return s;
  }

  // Up to k tokens continuing h (see the header comment). Empty when nothing qualifies.
  std::vector<int> draft(const std::vector<int>& h, int k, double theta) const {
    std::vector<int> d;
    if (tabs_.empty() || k <= 0) return d;
    const int no = orders();
    std::vector<int> ctx(h.end() - std::min<size_t>(h.size(), static_cast<size_t>(no)), h.end());
    double p = 1.0;
    while (static_cast<int>(d.size()) < k) {
      int tok = -1;
      double c = 0.0;
      if (!lookup(ctx, &tok, &c)) break;
      p *= c;
      if (p < theta) break;
      d.push_back(tok);
      ctx.push_back(tok);
      if (static_cast<int>(ctx.size()) > no) ctx.erase(ctx.begin());
    }
    return d;
  }

  static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
  }

 private:
  struct Order {
    size_t n = 0;
    std::vector<uint64_t> keys;
    std::vector<uint16_t> next;
    std::vector<uint8_t> conf;
  };

  // The longest order whose context (the last o tokens of ctx) is in the table.
  bool lookup(const std::vector<int>& ctx, int* tok, double* c) const {
    const int len = static_cast<int>(ctx.size());
    for (int o = std::min(orders(), len); o >= 1; --o) {
      const Order& t = tabs_[o - 1];
      if (t.n == 0) continue;
      uint64_t key = 0;
      bool ok = true;
      for (int j = 0; j < o; ++j) {
        const int v = ctx[len - 1 - j];
        if (v < 0 || v > 0xFFFF) { ok = false; break; }
        key |= static_cast<uint64_t>(v) << (16 * j);
      }
      if (!ok) continue;
      if (o == 4) key = mix64(key) >> 16;
      const auto it = std::lower_bound(t.keys.begin(), t.keys.end(), key);
      if (it != t.keys.end() && *it == key) {
        const size_t i = static_cast<size_t>(it - t.keys.begin());
        *tok = t.next[i];
        *c = static_cast<double>(t.conf[i]) / 255.0;
        return true;
      }
    }
    return false;
  }

  static bool fail(std::string* why, const std::string& w) {
    if (why) *why = w;
    return false;
  }

  std::vector<char> buf_;
  std::vector<Order> tabs_;
};

}  // namespace ember_draft
