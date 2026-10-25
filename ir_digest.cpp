#include "ir_digest.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ember_ir {
namespace {

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4)

const uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_block(uint32_t h[8], const unsigned char* p) {
  uint32_t w[64];
  for (int t = 0; t < 16; ++t)
    w[t] = (uint32_t(p[4 * t]) << 24) | (uint32_t(p[4 * t + 1]) << 16) | (uint32_t(p[4 * t + 2]) << 8) | uint32_t(p[4 * t + 3]);
  for (int t = 16; t < 64; ++t) {
    const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
    const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
    w[t] = w[t - 16] + s0 + w[t - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
  for (int t = 0; t < 64; ++t) {
    const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = k + S1 + ch + kK[t] + w[t];
    const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = S0 + maj;
    k = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

// ---------------------------------------------------------------- I-JSON values

struct Value {
  enum Type { Null, False, True, Number, String, Array, Object } type = Null;
  double number = 0.0;
  bool integer = false;  // an integer token (no fraction, no exponent)
  std::string str;       // String: UTF-8
  std::vector<Value> items;
  std::vector<std::pair<std::string, Value>> members;
};

struct NoDigest {
  std::string why;
};

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// RFC 8259 grammar, strictly; the I-JSON conditions as the text is read.
class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}

  Value document() {
    ws();
    Value v = value(0);
    ws();
    if (i_ != s_.size()) fail("not JSON: text after the value at byte " + std::to_string(i_));
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& why) const { throw NoDigest{why}; }

  void ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
  }

  bool lit(const char* w) {
    const size_t n = std::strlen(w);
    if (s_.compare(i_, n, w) != 0) return false;
    i_ += n;
    return true;
  }

  Value value(int depth) {
    if (depth > 10000) fail("not JSON: nested too deeply");
    if (i_ >= s_.size()) fail("not JSON: a value expected at the end");
    const char c = s_[i_];
    Value v;
    if (c == '{') {
      v.type = Value::Object;
      ++i_;
      ws();
      if (i_ < s_.size() && s_[i_] == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        ws();
        if (i_ >= s_.size() || s_[i_] != '"') fail("not JSON: a member name expected at byte " + std::to_string(i_));
        std::string name = string();
        for (const auto& m : v.members)
          if (m.first == name) fail("a duplicate member name");
        ws();
        if (i_ >= s_.size() || s_[i_] != ':') fail("not JSON: ':' expected at byte " + std::to_string(i_));
        ++i_;
        ws();
        Value item = value(depth + 1);
        v.members.emplace_back(std::move(name), std::move(item));
        ws();
        if (i_ < s_.size() && s_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < s_.size() && s_[i_] == '}') {
          ++i_;
          return v;
        }
        fail("not JSON: ',' or '}' expected at byte " + std::to_string(i_));
      }
    }
    if (c == '[') {
      v.type = Value::Array;
      ++i_;
      ws();
      if (i_ < s_.size() && s_[i_] == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        ws();
        v.items.push_back(value(depth + 1));
        ws();
        if (i_ < s_.size() && s_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < s_.size() && s_[i_] == ']') {
          ++i_;
          return v;
        }
        fail("not JSON: ',' or ']' expected at byte " + std::to_string(i_));
      }
    }
    if (c == '"') {
      v.type = Value::String;
      v.str = string();
      return v;
    }
    if (lit("true")) {
      v.type = Value::True;
      return v;
    }
    if (lit("false")) {
      v.type = Value::False;
      return v;
    }
    if (lit("null")) return v;
    if (c == '-' || (c >= '0' && c <= '9')) return number();
    fail("not JSON: unexpected character at byte " + std::to_string(i_));
  }

  Value number() {
    const size_t b = i_;
    bool neg = false;
    if (s_[i_] == '-') {
      neg = true;
      ++i_;
    }
    auto digits = [&] {
      const size_t d0 = i_;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
      return i_ - d0;
    };
    const size_t int0 = i_;
    if (i_ < s_.size() && s_[i_] == '0') {
      ++i_;
    } else if (digits() == 0) {
      fail("not JSON: a malformed number at byte " + std::to_string(b));
    }
    const std::string int_digits = s_.substr(int0, i_ - int0);
    bool integer = true;
    if (i_ < s_.size() && s_[i_] == '.') {
      ++i_;
      integer = false;
      if (digits() == 0) fail("not JSON: a malformed number at byte " + std::to_string(b));
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      integer = false;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (digits() == 0) fail("not JSON: a malformed number at byte " + std::to_string(b));
    }
    const std::string tok = s_.substr(b, i_ - b);
    Value v;
    v.type = Value::Number;
    v.integer = integer;
    if (integer) {
      // I-JSON: an integer within -2^53 .. 2^53 (compared as digits: a double would round the limit).
      static const std::string kLimit = "9007199254740992";
      if (int_digits.size() > kLimit.size() || (int_digits.size() == kLimit.size() && int_digits > kLimit))
        fail("the integer " + tok + " is outside I-JSON's exact range (2^53)");
      v.number = std::strtod(int_digits.c_str(), nullptr);  // exact: at most 2^53
      if (neg && v.number != 0.0) v.number = -v.number;  // "-0" is the integer zero
      return v;
    }
    v.number = std::strtod(tok.c_str(), nullptr);  // correctly rounded (C locale)
    if (!std::isfinite(v.number)) fail("the number " + tok + " is not finite in binary64");
    return v;
  }

  uint32_t hex4() {
    if (i_ + 4 > s_.size()) fail("not JSON: a short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= c - '0';
      else if (c >= 'a' && c <= 'f')
        v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        v |= c - 'A' + 10;
      else
        fail("not JSON: a malformed \\u escape");
    }
    return v;
  }

  // A JSON string at i_ (the opening quote), as UTF-8; refuses lone surrogates and invalid UTF-8.
  std::string string() {
    ++i_;
    std::string out;
    for (;;) {
      if (i_ >= s_.size()) fail("not JSON: an unterminated string");
      const unsigned char c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        ++i_;
        return out;
      }
      if (c < 0x20) fail("not JSON: a control character in a string");
      if (c == '\\') {
        ++i_;
        if (i_ >= s_.size()) fail("not JSON: an unterminated string");
        const char e = s_[i_++];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            uint32_t cp = hex4();
            if (cp >= 0xDC00 && cp <= 0xDFFF) fail("a string holds a lone surrogate");
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              if (i_ + 6 > s_.size() || s_[i_] != '\\' || s_[i_ + 1] != 'u') fail("a string holds a lone surrogate");
              i_ += 2;
              const uint32_t lo = hex4();
              if (lo < 0xDC00 || lo > 0xDFFF) fail("a string holds a lone surrogate");
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            append_utf8(out, cp);
            break;
          }
          default: fail("not JSON: an unknown escape in a string");
        }
        continue;
      }
      // Raw UTF-8: well-formed sequences only (no overlong forms, no surrogates, at most U+10FFFF).
      int n = 0;
      uint32_t cp = 0;
      if (c < 0x80) {
        n = 1;
        cp = c;
      } else if ((c & 0xE0) == 0xC0) {
        n = 2;
        cp = c & 0x1F;
      } else if ((c & 0xF0) == 0xE0) {
        n = 3;
        cp = c & 0x0F;
      } else if ((c & 0xF8) == 0xF0) {
        n = 4;
        cp = c & 0x07;
      } else {
        fail("a string is not well-formed UTF-8");
      }
      if (i_ + n > s_.size()) fail("a string is not well-formed UTF-8");
      for (int k = 1; k < n; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s_[i_ + k]);
        if ((cc & 0xC0) != 0x80) fail("a string is not well-formed UTF-8");
        cp = (cp << 6) | (cc & 0x3F);
      }
      static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
      if (cp < kMin[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) fail("a string is not well-formed UTF-8");
      out.append(s_, i_, n);
      i_ += n;
    }
  }

  const std::string& s_;
  size_t i_ = 0;
};

bool holds_negative_zero(const Value& v) {
  switch (v.type) {
    case Value::Number: return !v.integer && v.number == 0.0 && std::signbit(v.number);
    case Value::Array:
      for (const auto& x : v.items)
        if (holds_negative_zero(x)) return true;
      return false;
    case Value::Object:
      for (const auto& m : v.members)
        if (holds_negative_zero(m.second)) return true;
      return false;
    default: return false;
  }
}

// UTF-8 to UTF-16 code units (the text is well-formed: the parser checked it).
std::vector<uint16_t> utf16(const std::string& s) {
  std::vector<uint16_t> out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp;
    int n;
    if (c < 0x80) {
      cp = c;
      n = 1;
    } else if ((c & 0xE0) == 0xC0) {
      cp = c & 0x1F;
      n = 2;
    } else if ((c & 0xF0) == 0xE0) {
      cp = c & 0x0F;
      n = 3;
    } else {
      cp = c & 0x07;
      n = 4;
    }
    for (int k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    i += n;
    if (cp >= 0x10000) {
      cp -= 0x10000;
      out.push_back(static_cast<uint16_t>(0xD800 + (cp >> 10)));
      out.push_back(static_cast<uint16_t>(0xDC00 + (cp & 0x3FF)));
    } else {
      out.push_back(static_cast<uint16_t>(cp));
    }
  }
  return out;
}

// JSON.stringify of a string: '"' and '\' escaped, the controls U+0000 .. U+001F as \b \t \n \f \r or
// \u00xx (lowercase hex); everything else as it is (UTF-8).
void write_string(std::string& out, const std::string& s) {
  static const char* hex = "0123456789abcdef";
  out += '"';
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\t': out += "\\t"; break;
      case '\n': out += "\\n"; break;
      case '\f': out += "\\f"; break;
      case '\r': out += "\\r"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out += hex[c >> 4];
          out += hex[c & 15];
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

void write(std::string& out, const Value& v, bool prose) {
  switch (v.type) {
    case Value::Null: out += "null"; return;
    case Value::False: out += "false"; return;
    case Value::True: out += "true"; return;
    case Value::Number: out += es_number_string(v.number); return;
    case Value::String: write_string(out, v.str); return;
    case Value::Array: {
      out += '[';
      for (size_t k = 0; k < v.items.size(); ++k) {
        if (k) out += ',';
        write(out, v.items[k], prose);
      }
      out += ']';
      return;
    }
    case Value::Object: {
      std::vector<std::pair<std::vector<uint16_t>, const std::pair<std::string, Value>*>> keep;
      for (const auto& m : v.members)
        if (prose || m.first.empty() || m.first[0] != '_') keep.emplace_back(utf16(m.first), &m);
      std::sort(keep.begin(), keep.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      out += '{';
      for (size_t k = 0; k < keep.size(); ++k) {
        if (k) out += ',';
        write_string(out, keep[k].second->first);
        out += ':';
        write(out, keep[k].second->second, prose);
      }
      out += '}';
      return;
    }
  }
}

}  // namespace

std::string sha256_hex_bytes(const std::string& bytes) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8;
  std::string msg = bytes;
  msg += static_cast<char>(0x80);
  while (msg.size() % 64 != 56) msg += '\0';
  for (int k = 7; k >= 0; --k) msg += static_cast<char>((bits >> (8 * k)) & 0xFF);
  for (size_t off = 0; off < msg.size(); off += 64) sha256_block(h, reinterpret_cast<const unsigned char*>(msg.data()) + off);
  static const char* hex = "0123456789abcdef";
  std::string out;
  for (uint32_t x : h)
    for (int k = 7; k >= 0; --k) out += hex[(x >> (4 * k)) & 15];
  return out;
}

// ECMAScript Number::toString(x) (ECMA-262 Number::toString, radix 10) for a finite x: the shortest digits
// s (k of them) with s * 10^(n - k) round-tripping to x, closest to x on a tie (std::to_chars's shortest
// scientific form), then laid out by n.
std::string es_number_string(double v) {
  if (v == 0.0) return "0";
  if (v < 0) return "-" + es_number_string(-v);
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
  if (r.ec != std::errc()) throw std::runtime_error("es_number_string: to_chars failed");
  const std::string sci(buf, r.ptr);
  const size_t e = sci.find('e');
  std::string digits;
  for (size_t i = 0; i < e; ++i)
    if (sci[i] != '.') digits += sci[i];
  const int exp10 = std::atoi(sci.c_str() + e + 1);
  const int k = static_cast<int>(digits.size()), n = exp10 + 1;
  if (k <= n && n <= 21) return digits + std::string(static_cast<size_t>(n - k), '0');
  if (0 < n && n <= 21) return digits.substr(0, n) + "." + digits.substr(n);
  if (-6 < n && n <= 0) return "0." + std::string(static_cast<size_t>(-n), '0') + digits;
  const int ex = n - 1;
  std::string out = digits.substr(0, 1);
  if (k > 1) out += "." + digits.substr(1);
  out += ex >= 0 ? "e+" : "e-";
  out += std::to_string(ex >= 0 ? ex : -ex);
  return out;
}

DigestResult ir_digest(const std::string& text, bool prose) {
  DigestResult r;
  try {
    const Value v = Parser(text).document();
    if (holds_negative_zero(v)) throw NoDigest{"a negative zero (RFC 8785 writes it as 0, so it would share +0's digest)"};
    write(r.canonical, v, prose);
  } catch (const NoDigest& e) {
    r.refused = e.why;
    r.canonical.clear();
    return r;
  }
  r.sha256 = sha256_hex_bytes(r.canonical);
  r.ok = true;
  return r;
}

}  // namespace ember_ir
