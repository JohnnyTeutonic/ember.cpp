// utf8_text_test: ember::utf8_valid (utf8_text.h) against Python's bytes.decode("utf-8", "replace"), and
// ember::ReplyStream against the plain reply.
//
//   utf8_text_test          fixed cases (hex in, hex out), exit 1 on any difference
//   utf8_text_test hex      one hex string per line on stdin, the sanitised bytes as hex on stdout, for
//                           tools/utf8_text_check.py's comparison with Python on random byte strings
//   utf8_text_test stream   one "HEX C1,C2,..." per line on stdin: the reply's bytes and the prefix lengths
//                           the tokens end at (ascending, the last the whole length); on stdout, per line, the
//                           text ReplyStream has sent after each prefix and then after the end, as hex joined by
//                           "," (utf8_text_check.py compares it with Python's incremental decoder)
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "utf8_text.h"

static std::string from_hex(const std::string& h) {
  std::string out;
  for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
  return out;
}

static std::string to_hex(const std::string& s) {
  static const char* d = "0123456789abcdef";
  std::string out;
  for (unsigned char c : s) {
    out.push_back(d[c >> 4]);
    out.push_back(d[c & 15]);
  }
  return out;
}

// The text sent after each prefix (cuts), then after the end.
static std::vector<std::string> stream_totals(const std::string& raw, const std::vector<size_t>& cuts) {
  ember::ReplyStream st;
  std::vector<std::string> totals;
  std::string sent;
  for (size_t c : cuts) {
    sent += st.next(raw.substr(0, c), false);
    totals.push_back(sent);
  }
  sent += st.next(raw, true);
  totals.push_back(sent);
  return totals;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "hex") {
    std::string line;
    while (std::getline(std::cin, line)) std::cout << to_hex(ember::utf8_valid(from_hex(line))) << "\n";
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "stream") {
    std::string line;
    while (std::getline(std::cin, line)) {
      std::istringstream in(line);
      std::string hex, cuts_s;
      in >> hex >> cuts_s;
      std::vector<size_t> cuts;
      std::istringstream cs(cuts_s);
      for (std::string c; std::getline(cs, c, ',');)
        if (!c.empty()) cuts.push_back(static_cast<size_t>(std::stoul(c)));
      const std::vector<std::string> totals = stream_totals(from_hex(hex), cuts);
      for (size_t i = 0; i < totals.size(); ++i) std::cout << (i ? "," : "") << to_hex(totals[i]);
      std::cout << "\n";
    }
    return 0;
  }
  int bad = 0, n = 0;
  // {input, Python's bytes.fromhex(input).decode("utf-8", "replace").encode().hex()}
  const char* cases[][2] = {
      {"", ""},
      {"4974", "4974"},                            // ASCII
      {"d7", "efbfbd"},                            // pk-bench2 prompt 375: a lone lead byte at the end
      {"41d7", "41efbfbd"},
      {"d7a9", "d7a9"},                            // the whole Hebrew letter
      {"e282", "efbfbd"},                          // three-byte character cut after two
      {"e28241", "efbfbd41"},
      {"e282ac", "e282ac"},                        // euro sign
      {"f09f98", "efbfbd"},                        // four-byte character cut after three
      {"f09f9880", "f09f9880"},
      {"80", "efbfbd"},                            // continuation without a lead
      {"8080", "efbfbdefbfbd"},
      {"c0af", "efbfbdefbfbd"},                    // overlong lead C0
      {"e080af", "efbfbdefbfbdefbfbd"},            // E0 needs A0..BF next
      {"eda080", "efbfbdefbfbdefbfbd"},            // surrogate D800
      {"f4908080", "efbfbdefbfbdefbfbdefbfbd"},    // above U+10FFFF
      {"ff41", "efbfbd41"},
      {"e2d7a9", "efbfbdd7a9"},                    // a cut character, then a whole one
  };
  for (const auto& c : cases) {
    ++n;
    const std::string got = to_hex(ember::utf8_valid(from_hex(c[0])));
    if (got != c[1]) {
      std::printf("FAIL %s: got %s, want %s\n", c[0], got.c_str(), c[1]);
      ++bad;
    }
  }
  // ReplyStream: {reply bytes, token end offsets, the text sent after each token then after the end, as hex}
  struct StreamCase {
    const char* raw;
    std::vector<size_t> cuts;
    const char* want;
  };
  const StreamCase streams[] = {
      {"41d7a9", {1, 2, 3}, "41,41,41d7a9,41d7a9"},              // a token boundary inside the Hebrew letter
      {"41d7", {1, 2}, "41,41,41efbfbd"},                        // the reply ends inside it (max_new_tokens)
      {"41f09f9880", {2, 3, 5}, "41,41,41f09f9880,41f09f9880"},  // a four-byte character over three tokens
      {"41e08041", {2, 3, 4}, "41,41efbfbdefbfbd,41efbfbdefbfbd41,41efbfbdefbfbd41"},  // ill-formed at once
      {"2041200a", {1, 2, 3, 4}, ",41,41,41,41"},                // leading space skipped, trailing held, stripped
      {"41200a42", {2, 3, 4}, "41,41,41200a42,41200a42"},        // held whitespace sent once text follows
      {"4120d7", {2, 3}, "41,4120,4120efbfbd"},                  // a space before a cut character is final
      {"20", {1}, ","},                                          // whitespace only: nothing
  };
  for (const StreamCase& c : streams) {
    ++n;
    const std::vector<std::string> totals = stream_totals(from_hex(c.raw), c.cuts);
    std::string got;
    for (size_t i = 0; i < totals.size(); ++i) got += (i ? "," : "") + to_hex(totals[i]);
    if (got != c.want) {
      std::printf("FAIL stream %s: got %s, want %s\n", c.raw, got.c_str(), c.want);
      ++bad;
    }
  }
  // dump_valid: dump()'s bytes for valid text; for invalid text, utf8_valid on every string and key.
  {
    ++n;
    using json = nlohmann::json;
    const json ok = {{"reply", "d\xc3\xa9j\xc3\xa0"}, {"n", 3}};
    const json cut = {{"reply", std::string("A\xd7")}, {"k\xff", json::array({std::string("\x80"), 1})}};
    const std::string want_cut = "{\"k\xef\xbf\xbd\":[\"\xef\xbf\xbd\",1],\"reply\":\"A\xef\xbf\xbd\"}";
    if (ember::dump_valid(ok) != ok.dump() || ember::dump_valid(cut) != want_cut) {
      std::printf("FAIL dump_valid: %s\n", to_hex(ember::dump_valid(cut)).c_str());
      ++bad;
    }
  }
  std::printf("%s: %d of %d cases\n", bad ? "FAIL" : "ok", n - bad, n);
  return bad ? 1 : 0;
}
