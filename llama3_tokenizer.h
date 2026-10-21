// Vendored verbatim from paperkiln (microtorch) include/microtorch/llama3_tokenizer.hpp
// at paperkiln e70f75d44f6e (include names .hpp -> .h). ember.cpp uses it for paperkiln chat GGUFs whose tokenizer.ggml.model
// is "gpt2" with pre "llama-bpe" (the llama.cpp-compatible vocabulary, chat_tokenizer.h).
// Do not edit here: change it in paperkiln (its tests check C++/Python
// parity) and re-vendor.
#pragma once
// Llama 3 byte-level BPE, encoding exactly as llama.cpp does for a GGUF with
// tokenizer.ggml.model "gpt2" and tokenizer.ggml.pre "llama-bpe" (the
// vocabulary scheme of paperkiln models from the 21 Oct 2026 decision, so
// llama.cpp, Ollama and LM Studio tokenise them natively).
//
// Ported from llama.cpp b6000 (MIT License, Copyright (c) 2023-2024 The ggml
// authors): src/unicode.cpp (UTF-8 decoding, the GPT-2 byte-to-unicode map,
// unicode_regex_split_custom_llama3, a hand-written splitter for the llama3
// regex) and src/llama-vocab.cpp (special-token partition, the BPE merge
// queue with ignore_merges). The category tables come from
// src/unicode-data.cpp via tools/gen_llama3_tables.py.
//
// Encoding: the text is cut at special tokens (token types control,
// user-defined and unknown, longest first); each remaining fragment is split
// by the llama3 pre-tokeniser regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// each word is byte-encoded (GPT-2 map) and, unless the whole word is a
// token, merged by rank (lowest rank first, leftmost on ties). As in
// llama.cpp, invalid UTF-8 becomes U+FFFD before splitting, so only valid
// UTF-8 round-trips. No BOS or EOS is added.
#include <algorithm>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bpe_tokenizer.h"
#include "llama3_unicode_tables.h"

namespace microtorch {
namespace llama3 {

struct CptFlags {
    bool number = false, letter = false, whitespace = false;
    bool any = false;  // llama.cpp's as_uint() != 0: a code point inside the fragment
};

inline uint16_t category_bits(uint32_t cpt) {
    if (cpt >= 0x110000) return 0x0001;  // undefined
    uint32_t lo = 0, hi = tables::kRangeCount - 1;  // kRangeStart[lo] <= cpt < kRangeStart[hi]
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        (tables::kRangeStart[mid] <= cpt ? lo : hi) = mid;
    }
    return tables::kRangeFlags[lo];
}

inline CptFlags flags_of(uint32_t cpt) {
    const uint16_t c = category_bits(cpt);
    CptFlags f;
    f.number = c & 0x0002;
    f.letter = c & 0x0004;
    for (uint32_t i = 0; i < tables::kWhitespaceCount; ++i)
        if (tables::kWhitespace[i] == cpt) f.whitespace = true;
    f.any = true;
    return f;
}

// llama.cpp's unicode_cpts_from_utf8: an invalid byte is skipped and becomes U+FFFD.
inline std::vector<uint32_t> cpts_from_utf8(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    const auto b = [&](size_t i) { return static_cast<unsigned char>(s[i]); };
    const auto cont = [&](size_t i) { return (b(i) & 0xC0) == 0x80; };
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = b(i);
        if (!(c & 0x80)) {
            out.push_back(c);
            i += 1;
        } else if (!(c & 0x40)) {
            out.push_back(0xFFFD);
            i += 1;
        } else if (!(c & 0x20)) {
            if (i + 1 >= s.size() || !cont(i + 1)) {
                out.push_back(0xFFFD);
                i += 1;
            } else {
                out.push_back(((c & 0x1Fu) << 6) | (b(i + 1) & 0x3Fu));
                i += 2;
            }
        } else if (!(c & 0x10)) {
            if (i + 2 >= s.size() || !cont(i + 1) || !cont(i + 2)) {
                out.push_back(0xFFFD);
                i += 1;
            } else {
                out.push_back(((c & 0x0Fu) << 12) | ((b(i + 1) & 0x3Fu) << 6) | (b(i + 2) & 0x3Fu));
                i += 3;
            }
        } else if (!(c & 0x08)) {
            if (i + 3 >= s.size() || !cont(i + 1) || !cont(i + 2) || !cont(i + 3)) {
                out.push_back(0xFFFD);
                i += 1;
            } else {
                out.push_back(((c & 0x07u) << 18) | ((b(i + 1) & 0x3Fu) << 12) | ((b(i + 2) & 0x3Fu) << 6) |
                              (b(i + 3) & 0x3Fu));
                i += 4;
            }
        } else {
            out.push_back(0xFFFD);
            i += 1;
        }
    }
    return out;
}

inline void append_utf8(uint32_t cpt, std::string& out) {
    if (cpt <= 0x7F) {
        out.push_back(static_cast<char>(cpt));
    } else if (cpt <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cpt >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
    } else if (cpt <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cpt >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
    } else if (cpt <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | ((cpt >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
    } else {
        throw std::invalid_argument("invalid codepoint");
    }
}

// The llama3 pre-tokeniser (unicode_regex_split_custom_llama3) on one
// fragment with no special tokens in it: the words, as UTF-8.
inline std::vector<std::string> split_words(const std::string& text) {
    const std::vector<uint32_t> cp = cpts_from_utf8(text);
    const size_t n = cp.size();
    constexpr uint32_t kOut = 0xFFFFFFFF;
    const auto get = [&](size_t p) { return p < n ? cp[p] : kOut; };
    const auto fl = [&](size_t p) { return p < n ? flags_of(cp[p]) : CptFlags{}; };
    // Only ASCII capitals lowercase to s, t, r, e, v, m, l or d in llama.cpp's
    // table (checked by tools/gen_llama3_tables.py's source), so ASCII suffices.
    const auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    std::vector<std::string> words;
    size_t prev = 0;
    const auto add = [&](size_t end) {
        const size_t len = end - prev;
        if (len > 0) {
            words.emplace_back();
            for (size_t i = prev; i < end; ++i) append_utf8(cp[i], words.back());
        }
        prev = end;
        return len;
    };
    for (size_t pos = 0; pos < n;) {
        const uint32_t c = get(pos);
        const CptFlags f = fl(pos);
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && pos + 1 < n) {
            const uint32_t c1 = lower(get(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += add(pos + 2);
                continue;
            }
            if (pos + 2 < n) {
                const uint32_t c2 = lower(get(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += add(pos + 3);
                    continue;
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(c == '\r' || c == '\n' || f.number)) {
            if (f.letter || fl(pos + 1).letter) {
                ++pos;
                while (fl(pos).letter) ++pos;
                add(pos);
                continue;
            }
        }
        // \p{N}{1,3}
        if (f.number) {
            size_t ini = pos;
            while (fl(pos).number) {
                if (++pos - ini >= 3) {
                    add(pos);
                    ini = pos;
                }
            }
            add(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        CptFlags f2 = c == ' ' ? fl(pos + 1) : f;
        if (!(f2.whitespace || f2.letter || f2.number) && f.any) {
            pos += c == ' ' ? 1 : 0;
            while (!(f2.whitespace || f2.letter || f2.number) && f2.any) f2 = fl(++pos);
            uint32_t c2 = get(pos);
            while (c2 == '\r' || c2 == '\n') c2 = get(++pos);
            add(pos);
            continue;
        }
        size_t nws = 0, last_rn = 0;
        while (fl(pos + nws).whitespace) {
            const uint32_t c2 = get(pos + nws);
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
            ++nws;
        }
        if (last_rn > 0) {  // \s*[\r\n]+
            pos = last_rn;
            add(pos);
            continue;
        }
        if (nws > 1 && get(pos + nws) != kOut) {  // \s+(?!\S)
            pos += nws - 1;
            add(pos);
            continue;
        }
        if (nws > 0) {  // \s+
            pos += nws;
            add(pos);
            continue;
        }
        add(++pos);  // no match
    }
    return words;
}

// GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to
// U+0100 upward in byte order.
inline const std::vector<std::string>& byte_to_unicode() {
    static const std::vector<std::string> m = [] {
        std::vector<std::string> v(256);
        std::vector<bool> direct(256, false);
        for (int c = 0x21; c <= 0x7E; ++c) direct[c] = true;
        for (int c = 0xA1; c <= 0xAC; ++c) direct[c] = true;
        for (int c = 0xAE; c <= 0xFF; ++c) direct[c] = true;
        uint32_t n = 0;
        for (int c = 0; c < 256; ++c) append_utf8(direct[c] ? static_cast<uint32_t>(c) : 256 + n++, v[c]);
        return v;
    }();
    return m;
}

inline std::string byte_encode(const std::string& word) {
    std::string out;
    for (unsigned char c : word) out += byte_to_unicode()[c];
    return out;
}

class Tokenizer {
public:
    // tokens: tokenizer.ggml.tokens (byte-encoded strings); merges: "a b"
    // in rank order; token_type: tokenizer.ggml.token_type (1 normal,
    // 2 unknown, 3 control, 4 user-defined, ...), empty = all normal.
    Tokenizer(std::vector<std::string> tokens, const std::vector<std::string>& merges,
              const std::vector<int32_t>& token_type)
        : tokens_(std::move(tokens)) {
        // A repeated string maps to its last id, as llama.cpp's token_to_id.
        for (size_t i = 0; i < tokens_.size(); ++i) id_[tokens_[i]] = static_cast<int>(i);
        for (size_t r = 0; r < merges.size(); ++r) rank_.emplace(merges[r], static_cast<int>(r));
        special_.assign(tokens_.size(), false);
        for (size_t i = 0; i < token_type.size() && i < tokens_.size(); ++i) {
            const int t = token_type[i];
            if (t == 2 || t == 3 || t == 4) {
                special_[i] = true;
                specials_.push_back(static_cast<int>(i));
            }
        }
        // Longest first, as llama.cpp's special-token cache (ties by id).
        std::stable_sort(specials_.begin(), specials_.end(),
                         [&](int a, int b) { return tokens_[a].size() > tokens_[b].size(); });
        for (int c = 0; c < 256; ++c) unbyte_.emplace(byte_to_unicode()[c], static_cast<unsigned char>(c));
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<std::pair<std::string, int>> frags = {{text, -1}};
        for (int sid : specials_) {
            const std::string& s = tokens_[sid];
            if (s.empty()) continue;
            std::vector<std::pair<std::string, int>> next;
            for (auto& [t, id] : frags) {
                if (id >= 0) {
                    next.emplace_back(std::move(t), id);
                    continue;
                }
                size_t at = 0;
                for (size_t m; (m = t.find(s, at)) != std::string::npos; at = m + s.size()) {
                    if (m > at) next.emplace_back(t.substr(at, m - at), -1);
                    next.emplace_back(std::string(), sid);
                }
                if (at < t.size()) next.emplace_back(t.substr(at), -1);
            }
            frags = std::move(next);
        }
        std::vector<int> out;
        for (const auto& [t, id] : frags) {
            if (id >= 0) {
                out.push_back(id);
                continue;
            }
            for (const std::string& w : split_words(t)) encode_word(byte_encode(w), out);
        }
        return out;
    }

    // Raw bytes: special tokens as their text, others byte-decoded.
    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
            if (special_[id]) {
                s += tokens_[id];
                continue;
            }
            const std::string& t = tokens_[id];
            for (size_t i = 0; i < t.size();) {
                const size_t len = utf8_len(static_cast<unsigned char>(t[i]));
                const auto it = unbyte_.find(t.substr(i, len));
                if (it != unbyte_.end()) s.push_back(static_cast<char>(it->second));
                i += len;
            }
        }
        return s;
    }

    size_t size() const { return tokens_.size(); }
    const std::vector<std::string>& tokens() const { return tokens_; }

private:
    static size_t utf8_len(unsigned char c) {
        static const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
        return lookup[c >> 4];
    }

    struct Symbol {
        int prev, next;
        size_t start, n;
    };
    struct Bigram {
        int left, right, rank;
        std::string text;
    };
    struct Later {  // llama.cpp's comparator: lowest rank first, then leftmost
        bool operator()(const Bigram& l, const Bigram& r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    // llm_tokenizer_bpe_session::tokenize for one byte-encoded word.
    void encode_word(const std::string& word, std::vector<int>& out) const {
        std::vector<Symbol> sym;
        if (id_.count(word)) {  // ignore_merges: a whole-word token wins
            sym.push_back({-1, -1, 0, word.size()});
        } else {
            for (size_t off = 0; off < word.size();) {
                const size_t len = std::min(word.size() - off, utf8_len(static_cast<unsigned char>(word[off])));
                const int idx = static_cast<int>(sym.size());
                sym.push_back({idx - 1, off + len == word.size() ? -1 : idx + 1, off, len});
                off += len;
            }
        }
        std::priority_queue<Bigram, std::vector<Bigram>, Later> q;
        const auto text = [&](int i) { return word.substr(sym[i].start, sym[i].n); };
        const auto push = [&](int l, int r) {
            if (l < 0 || r < 0) return;
            const std::string a = text(l), b = text(r);
            const auto it = rank_.find(a + " " + b);
            if (it != rank_.end()) q.push({l, r, it->second, a + b});
        };
        for (int i = 1; i < static_cast<int>(sym.size()); ++i) push(i - 1, i);
        while (!q.empty()) {
            const Bigram bg = q.top();
            q.pop();
            Symbol& L = sym[bg.left];
            Symbol& R = sym[bg.right];
            if (L.n == 0 || R.n == 0 || text(bg.left) + text(bg.right) != bg.text) continue;  // outdated
            L.n += R.n;
            R.n = 0;
            L.next = R.next;
            if (R.next >= 0) sym[R.next].prev = bg.left;
            push(L.prev, bg.left);
            push(bg.left, L.next);
        }
        for (const Symbol& s : sym) {
            if (s.n == 0) continue;
            const std::string t = word.substr(s.start, s.n);
            const auto it = id_.find(t);
            if (it != id_.end()) {
                out.push_back(it->second);
                continue;
            }
            for (char c : t) {  // llama.cpp's fallback: single bytes that are tokens
                const auto b = id_.find(std::string(1, c));
                if (b != id_.end()) out.push_back(b->second);
            }
        }
    }

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> id_, rank_;
    std::vector<bool> special_;
    std::vector<int> specials_;
    std::unordered_map<std::string, unsigned char> unbyte_;
};

// A llama.cpp-compatible paperkiln vocabulary GGUF: model "gpt2", pre "llama-bpe".
inline Tokenizer load_gguf(const std::string& path) {
    bpe::GgufTokenizer g = bpe::read_gguf_tokenizer(path);
    if (g.model != "gpt2" || g.pre != "llama-bpe")
        throw std::runtime_error(path + ": tokenizer is '" + g.model + "'/'" + g.pre + "', not 'gpt2'/'llama-bpe'");
    return Tokenizer(std::move(g.tokens), g.merges, g.token_type);
}

}  // namespace llama3
}  // namespace microtorch
