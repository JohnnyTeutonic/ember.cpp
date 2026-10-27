// utf8_text_test: ember::utf8_valid (utf8_text.h) against Python's bytes.decode("utf-8", "replace").
//
//   utf8_text_test          fixed cases (hex in, hex out), exit 1 on any difference
//   utf8_text_test hex      one hex string per line on stdin, the sanitised bytes as hex on stdout, for
//                           tools/utf8_text_check.py's comparison with Python on random byte strings
#include <cstdio>
#include <iostream>
#include <string>

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

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "hex") {
    std::string line;
    while (std::getline(std::cin, line)) std::cout << to_hex(ember::utf8_valid(from_hex(line))) << "\n";
    return 0;
  }
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
  int bad = 0;
  for (const auto& c : cases) {
    const std::string got = to_hex(ember::utf8_valid(from_hex(c[0])));
    if (got != c[1]) {
      std::printf("FAIL %s: got %s, want %s\n", c[0], got.c_str(), c[1]);
      ++bad;
    }
  }
  std::printf("%s: %d of %zu cases\n", bad ? "FAIL" : "ok", static_cast<int>(sizeof(cases) / sizeof(cases[0])) - bad,
              sizeof(cases) / sizeof(cases[0]));
  return bad ? 1 : 0;
}
