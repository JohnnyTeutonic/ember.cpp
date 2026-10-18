// Vendored verbatim from paperkiln (microtorch) include/microtorch/bpe_tokenizer.hpp
// at paperkiln fb810ca1066b. ember.cpp uses it for GGUFs whose tokenizer.ggml.model
// is "bpe" (paperkiln byte-level BPE: merges are token-id pairs). Do not
// edit here: change it in paperkiln (tests/test_bpe.py checks C++/Python
// parity) and re-vendor.
#pragma once
// Byte-level BPE encoder/decoder: the C++ side of tools/bpe.py (the
// reference, which also trains). Byte-identical to it: tests/test_bpe.py
// encodes a corpus with both and compares ids.
//
// Pre-tokenization works on bytes so no Unicode tables are needed: a letter
// is an ASCII letter, an apostrophe or any byte >= 0x80 (a multibyte
// character stays inside its word); a digit is 0-9 and is always its own
// chunk; whitespace is the C isspace set. Chunks: a letter run with at most
// one leading space, a single digit, a run of whitespace (leaving a space
// that precedes a letter to the next word), a run of other bytes, or a
// special token. Merges never cross a chunk; within a chunk the
// lowest-ranked merge present is applied until none applies.
//
// Lossless: decode(encode(s)) == s for every byte string, and there is no
// <unk> in normal use.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace microtorch {
namespace bpe {

inline const std::vector<std::string>& specials() {
    static const std::vector<std::string> s = {"<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"};
    return s;
}

class Tokenizer {
public:
    // tokens: the GGUF tokenizer.ggml.tokens list (specials, 256 bytes,
    // then one per merge); merges: "a b" token-id pairs in rank order.
    Tokenizer(std::vector<std::string> tokens, const std::vector<std::string>& merges) : tokens_(std::move(tokens)) {
        const int base = static_cast<int>(specials().size()) + 256;
        for (size_t r = 0; r < merges.size(); ++r) {
            const auto sp = merges[r].find(' ');
            const int a = std::stoi(merges[r].substr(0, sp)), b = std::stoi(merges[r].substr(sp + 1));
            rank_[key(a, b)] = {static_cast<int>(r), base + static_cast<int>(r)};
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        for (const auto& [chunk, special] : pretokenize(text)) {
            if (special >= 0) {
                out.push_back(special);
                continue;
            }
            encode_chunk(chunk, out);
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids)
            if (id >= 0 && static_cast<size_t>(id) < tokens_.size()) s += tokens_[id];
        return s;
    }

    size_t size() const { return tokens_.size(); }
    const std::vector<std::string>& tokens() const { return tokens_; }

    static bool letter(unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '\'' || c >= 0x80;
    }
    static bool digit(unsigned char c) { return c >= '0' && c <= '9'; }
    static bool space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

    // (chunk, special id or -1), in order.
    static std::vector<std::pair<std::string, int>> pretokenize(const std::string& b) {
        std::vector<std::pair<std::string, int>> out;
        const size_t n = b.size();
        auto special_at = [&](size_t i) -> int {
            for (size_t k = 1; k < specials().size(); ++k)
                if (b.compare(i, specials()[k].size(), specials()[k]) == 0) return static_cast<int>(k);
            return -1;
        };
        size_t i = 0;
        while (i < n) {
            const int sp = special_at(i);
            if (sp >= 0) {
                out.emplace_back(specials()[sp], sp);
                i += specials()[sp].size();
                continue;
            }
            const unsigned char c = b[i];
            const size_t j = (c == ' ' && i + 1 < n && letter(b[i + 1])) ? i + 1 : i;
            size_t k;
            if (letter(b[j])) {
                k = j;
                while (k < n && letter(b[k])) ++k;
            } else if (digit(c)) {
                k = i + 1;
            } else if (space(c)) {
                k = i;
                while (k < n && space(b[k]) && !(b[k] == ' ' && k + 1 < n && letter(b[k + 1]))) ++k;
                if (k == i) k = i + 1;
            } else {
                k = i;
                while (k < n && !(space(b[k]) || letter(b[k]) || digit(b[k]) || special_at(k) >= 0)) ++k;
                if (k == i) k = i + 1;
            }
            out.emplace_back(b.substr(i, k - i), -1);
            i = k;
        }
        return out;
    }

private:
    struct Merge {
        int rank, id;
    };
    static uint64_t key(int a, int b) { return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b); }

    void encode_chunk(const std::string& chunk, std::vector<int>& out) const {
        const int off = static_cast<int>(specials().size());
        std::vector<int> ids;
        for (unsigned char c : chunk) ids.push_back(off + c);
        while (ids.size() > 1) {
            int best = -1, at = -1, id = -1;
            for (size_t k = 0; k + 1 < ids.size(); ++k) {
                const auto it = rank_.find(key(ids[k], ids[k + 1]));
                if (it != rank_.end() && (best < 0 || it->second.rank < best)) {
                    best = it->second.rank;
                    at = static_cast<int>(k);
                    id = it->second.id;
                }
            }
            if (best < 0) break;
            ids[at] = id;
            ids.erase(ids.begin() + at + 1);
        }
        out.insert(out.end(), ids.begin(), ids.end());
    }

    std::vector<std::string> tokens_;
    std::unordered_map<uint64_t, Merge> rank_;
};

struct GgufTokenizer {
    std::string model;
    std::vector<std::string> tokens, merges;
};

// The tokenizer block of any GGUF (tokenizer.ggml.model, tokens, merges),
// unvalidated: a vocabulary file from tools/bpe.py or a model export.
inline GgufTokenizer read_gguf_tokenizer(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t p = 0;
    auto need = [&](size_t n) {
        if (p + n > buf.size()) throw std::runtime_error(path + ": truncated GGUF");
    };
    auto u32 = [&]() { need(4); uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
    auto u64 = [&]() { need(8); uint64_t v; std::memcpy(&v, buf.data() + p, 8); p += 8; return v; };
    auto str = [&]() { const uint64_t n = u64(); need(n); std::string s = buf.substr(p, n); p += n; return s; };
    if (u32() != 0x46554747u) throw std::runtime_error(path + ": not GGUF");
    u32();
    u64();
    const uint64_t nkv = u64();
    std::vector<std::string> tokens, merges;
    std::string model;
    std::function<void(uint32_t)> skip = [&](uint32_t t) {
        static const size_t sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (t == 8) {
            str();
        } else if (t == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            for (uint64_t i = 0; i < n; ++i) skip(et);
        } else if (t < 13) {
            need(sz[t]);
            p += sz[t];
        } else {
            throw std::runtime_error(path + ": unknown GGUF type");
        }
    };
    for (uint64_t i = 0; i < nkv; ++i) {
        const std::string k = str();
        const uint32_t t = u32();
        if ((k == "tokenizer.ggml.tokens" || k == "tokenizer.ggml.merges") && t == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            if (et != 8) throw std::runtime_error(path + ": " + k + " is not a string array");
            auto& dst = k == "tokenizer.ggml.tokens" ? tokens : merges;
            for (uint64_t j = 0; j < n; ++j) dst.push_back(str());
        } else if (k == "tokenizer.ggml.model" && t == 8) {
            model = str();
        } else {
            skip(t);
        }
    }
    return {std::move(model), std::move(tokens), std::move(merges)};
}

// Reads a tokenizer GGUF written by tools/bpe.py (or a BPE model export):
// tokenizer.ggml.model must be "bpe"; tokens and merges are taken from
// their arrays.
inline Tokenizer load_gguf(const std::string& path) {
    GgufTokenizer g = read_gguf_tokenizer(path);
    if (g.model != "bpe") throw std::runtime_error(path + ": tokenizer.ggml.model is '" + g.model + "', not 'bpe'");
    return Tokenizer(std::move(g.tokens), g.merges);
}

}  // namespace bpe
}  // namespace microtorch
