// Valid UTF-8 for JSON replies. A reply decoded from BPE tokens is bytes: a model can stop (end-of-text,
// or max_new_tokens) inside a multi-byte character, or emit byte tokens that form no character at all.
// nlohmann::json's dump() throws on such a string (type_error 316), which /chat turned into HTTP 500: on
// 26 Oct 2026 pk-bench2's IF-tiny job (paperkiln experiments/benchmarks_2026-10-26) stopped at prompt 375
// of pick_e1_60m_L32, whose greedy reply ended on the lone lead byte 0xD7. utf8_valid replaces every
// ill-formed subsequence with U+FFFD, one per maximal subpart (Unicode 15, section 3.9, U+FFFD
// substitution of maximal subparts), the substitution Python's bytes.decode("utf-8", "replace") makes, so
// a client decoding the same ids with errors="replace" gets the same text. The ids ("return_ids") are
// untouched.
#pragma once

#include <string>

namespace ember {

inline std::string utf8_valid(const std::string& s) {
  const char rep[] = {static_cast<char>(0xEF), static_cast<char>(0xBF), static_cast<char>(0xBD)};
  std::string out;
  out.reserve(s.size());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      out.push_back(static_cast<char>(c));
      ++i;
      continue;
    }
    // continuation bytes needed after the lead, and the range of the first one (Table 3-7)
    size_t need = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) {
      need = 1;
    } else if (c == 0xE0) {
      need = 2;
      lo = 0xA0;
    } else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
      need = 2;
    } else if (c == 0xED) {
      need = 2;
      hi = 0x9F;
    } else if (c == 0xF0) {
      need = 3;
      lo = 0x90;
    } else if (c >= 0xF1 && c <= 0xF3) {
      need = 3;
    } else if (c == 0xF4) {
      need = 3;
      hi = 0x8F;
    } else {  // a continuation byte without a lead, C0, C1, F5..FF
      out.append(rep, 3);
      ++i;
      continue;
    }
    size_t j = 1;
    for (; j <= need && i + j < n; ++j) {
      const unsigned char d = static_cast<unsigned char>(s[i + j]);
      if (d < (j == 1 ? lo : 0x80) || d > (j == 1 ? hi : 0xBF)) break;
    }
    if (j == need + 1) {
      out.append(s, i, need + 1);
    } else {
      out.append(rep, 3);  // the lead and the continuations that fitted: one maximal subpart
    }
    i += j;
  }
  return out;
}

}  // namespace ember
