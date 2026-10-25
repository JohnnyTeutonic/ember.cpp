// unicode_nfc.h on NUL-separated UTF-8 texts from stdin: writes each text's NFC, NUL-terminated, for
// experiments/qwen_nfc_2026-10-25/nfc_gate.py to compare with HF tokenizers' and Python's NFC.
//   g++ -O2 -std=c++17 -I. tools/nfc_check.cpp -o nfc_check
//   ./nfc_check < texts.nul > nfc.nul
#include <iostream>
#include <iterator>
#include <string>

#include "unicode_nfc.h"

int main() {
    std::ios::sync_with_stdio(false);
    const std::string all((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::string out;
    out.reserve(all.size() + all.size() / 4);
    size_t at = 0;
    while (at < all.size()) {
        size_t end = all.find('\0', at);
        if (end == std::string::npos) end = all.size();
        out += ember::nfc::nfc_utf8(all.substr(at, end - at));
        out.push_back('\0');
        at = end + 1;
    }
    std::cout.write(out.data(), static_cast<std::streamsize>(out.size()));
    std::cerr << "unicode_nfc.h: Unicode " << ember::nfc::tables::kUnicodeVersion
              << (ember::nfc::tables::kHfFiltered ? ", filtered to HF's NFC" : ", unfiltered") << "\n";
    return 0;
}
