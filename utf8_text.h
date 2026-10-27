// Valid UTF-8 for JSON replies. A reply decoded from BPE tokens is bytes: a model can stop (end-of-text,
// or max_new_tokens) inside a multi-byte character, or emit byte tokens that form no character at all.
// nlohmann::json's dump() throws on such a string (type_error 316), which /chat turned into HTTP 500: on
// 26 Oct 2026 pk-bench2's IF-tiny job (paperkiln experiments/benchmarks_2026-10-26) stopped at prompt 375
// of pick_e1_60m_L32, whose greedy reply ended on the lone lead byte 0xD7. utf8_valid replaces every
// ill-formed subsequence with U+FFFD, one per maximal subpart (Unicode 15, section 3.9, U+FFFD
// substitution of maximal subparts), the substitution Python's bytes.decode("utf-8", "replace") makes, so
// a client decoding the same ids with errors="replace" gets the same text. The ids ("return_ids") are
// untouched.
//
// Every place ember returns model text goes through here: /chat (plain, streamed, the legacy path),
// /v1/chat/completions (plain and streamed), WebAssembly ember_chat / ember_finish, and the Python
// bindings' generate / generate_batch. A streamed reply (ReplyStream) holds back a character a token
// boundary cut (at most 3 bytes) until the next token completes it, so the concatenated pieces equal the
// plain reply byte for byte. dump_valid is the last line for every other JSON string (error messages,
// paths): a response is never HTTP 500 for its text.
#pragma once

#include <algorithm>
#include <string>

namespace ember {

// What the maximal subpart at s[i] is: a whole character (ASCII included), ill-formed bytes that no
// following byte can repair, or the start of a character that the end of s cut (more bytes may complete it).
enum class Utf8Part { whole, ill_formed, cut };

// The maximal subpart at s[i] (i < s.size()): its length, and what it is.
inline size_t utf8_part(const std::string& s, size_t i, Utf8Part& part) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  part = Utf8Part::whole;
  if (c < 0x80) return 1;
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
    part = Utf8Part::ill_formed;
    return 1;
  }
  const size_t n = s.size();
  size_t j = 1;
  for (; j <= need; ++j) {
    if (i + j >= n) {  // the lead and the continuations that fitted, then the end of s
      part = Utf8Part::cut;
      return j;
    }
    const unsigned char d = static_cast<unsigned char>(s[i + j]);
    if (d < (j == 1 ? lo : 0x80) || d > (j == 1 ? hi : 0xBF)) {
      part = Utf8Part::ill_formed;  // the lead and the continuations that fitted: one maximal subpart
      return j;
    }
  }
  return j;
}

inline std::string utf8_valid(const std::string& s) {
  const char rep[] = {static_cast<char>(0xEF), static_cast<char>(0xBF), static_cast<char>(0xBD)};
  std::string out;
  out.reserve(s.size());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    Utf8Part part;
    const size_t len = utf8_part(s, i, part);
    if (part == Utf8Part::whole)
      out.append(s, i, len);
    else
      out.append(rep, 3);  // ill-formed, or cut by the end of s: one U+FFFD per maximal subpart
    i += len;
  }
  return out;
}

// The end of what is settled in s from `from` (a subpart boundary): every byte before it decodes the same
// whatever bytes follow (whole characters, and ill-formed bytes no continuation can repair); the bytes
// after it, at most 3, are a character the end of s cut, which the next token may complete.
inline size_t utf8_settled(const std::string& s, size_t from) {
  const size_t n = s.size();
  size_t i = from;
  while (i < n) {
    Utf8Part part;
    const size_t len = utf8_part(s, i, part);
    if (part == Utf8Part::cut) return i;
    i += len;
  }
  return n;
}

// A reply streamed as it is generated. generate() returns the reply's bytes with leading and trailing
// whitespace (the C isspace set) stripped (api.cpp), and the reply object carries utf8_valid of that; the
// pieces next() returns concatenate to exactly that text. Each call takes the decoded reply so far (each a
// prefix of the next, as byte-level BPE decoding is) and returns the newly final text, valid UTF-8 (often
// empty): leading whitespace skipped; trailing whitespace held until something other than whitespace
// follows it; a cut character held until the next token completes it, or until end. U+FFFD replaces only
// bytes that can never become valid, or a character still cut at the end.
class ReplyStream {
 public:
  std::string next(const std::string& full, bool end) {
    static const char* ws = " \t\n\v\f\r";
    const size_t first = full.find_first_not_of(ws);
    if (first == std::string::npos) return std::string();
    if (sent_ == std::string::npos) sent_ = first;
    size_t limit = full.find_last_not_of(ws) + 1;
    if (!end) limit = std::min(limit, utf8_settled(full, sent_));
    if (limit <= sent_) return std::string();
    std::string out = utf8_valid(full.substr(sent_, limit - sent_));
    sent_ = limit;
    return out;
  }

 private:
  size_t sent_ = std::string::npos;  // bytes of the decoded reply already sent; npos before the first non-space
};

// Json (nlohmann::json) with utf8_valid applied to every string and object key.
template <class Json>
Json utf8_valid_json(const Json& j) {
  if (j.is_string()) return Json(utf8_valid(j.template get<std::string>()));
  if (j.is_array()) {
    Json a = Json::array();
    for (const auto& x : j) a.push_back(utf8_valid_json(x));
    return a;
  }
  if (j.is_object()) {
    Json o = Json::object();
    for (auto it = j.begin(); it != j.end(); ++it) o[utf8_valid(it.key())] = utf8_valid_json(it.value());
    return o;
  }
  return j;
}

// j.dump(), or, when a string in j is not valid UTF-8 (dump() throws type_error 316), the dump of j with
// utf8_valid applied to every string: the same bytes as dump() whenever dump() succeeds.
template <class Json>
std::string dump_valid(const Json& j) {
  try {
    return j.dump();
  } catch (const typename Json::type_error&) {
    return utf8_valid_json(j).dump();
  }
}

}  // namespace ember
