#pragma once
// SentencePiece vocabularies (tokenizer.ggml.model "llama") as llama.cpp b6000 encodes them
// (llama-vocab.cpp: tokenizer_st_partition, then llm_tokenizer_spm_session on each text
// fragment), for the chat mode of Gemma 3 GGUFs (chat_tokenizer.h). ember's older SentencePiece
// path (tokenizer.cpp) approximates merges from scores and stays with the TinyLlama models.
//
// - Special tokens: every CONTROL, USER_DEFINED and UNKNOWN token (GGUF token types 3, 4, 2) is
//   cut out of the text before encoding, longest token text first, each occurrence left to right
//   (llama.cpp's parse_special = true: chat prompts carry <bos> and <start_of_turn> as text).
// - Each remaining fragment: " " -> U+2581 (and a leading U+2581 when
//   tokenizer.ggml.add_space_prefix is true and the fragment starts the text or follows a special
//   token), split into UTF-8 characters, then the adjacent pair whose concatenation is a token with
//   the highest score is merged, ties to the leftmost, until no pair is a token. A final symbol that
//   is not a token is written as its bytes (<0xXX>).
// - No BOS is added: a chat template writes its own.
// Decoding: U+2581 -> " ", byte tokens -> their byte, every other token -> its text.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ember {

class SpmTokenizer {
public:
    SpmTokenizer(std::vector<std::string> tokens, std::vector<float> scores, std::vector<int32_t> types,
                 bool add_space_prefix)
        : tokens_(std::move(tokens)), scores_(std::move(scores)), types_(std::move(types)),
          add_space_prefix_(add_space_prefix) {
        if (scores_.size() != tokens_.size()) throw std::runtime_error("SentencePiece GGUF: scores and tokens differ in length");
        if (types_.size() != tokens_.size()) throw std::runtime_error("SentencePiece GGUF: token types and tokens differ in length");
        id_.reserve(tokens_.size());
        for (size_t i = 0; i < tokens_.size(); ++i) id_[tokens_[i]] = static_cast<int>(i);  // the last one wins
        for (size_t i = 0; i < tokens_.size(); ++i)
            if ((types_[i] == kControl || types_[i] == kUserDefined || types_[i] == kUnknown) && !tokens_[i].empty())
                specials_.push_back(static_cast<int>(i));
        // Longest text first (llama.cpp sorts by length with std::sort; among equal lengths the
        // order only matters for overlapping specials, which the gate against HF would show).
        std::stable_sort(specials_.begin(), specials_.end(),
                         [&](int a, int b) { return tokens_[a].size() > tokens_[b].size(); });
        for (int b = 0; b < 256; ++b) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "<0x%02X>", b);
            auto it = id_.find(buf);
            if (it == id_.end()) it = id_.find(std::string(1, static_cast<char>(b)));
            byte_id_[b] = it == id_.end() ? -1 : it->second;
        }
    }

    std::vector<int> encode(const std::string& text) const {
        // Fragments: (special token id, or -1 with [offset, offset + length) of text).
        struct Frag {
            int id;
            size_t off, len;
        };
        std::vector<Frag> frags = {{-1, 0, text.size()}};
        for (int sid : specials_) {
            const std::string& s = tokens_[sid];
            std::vector<Frag> next;
            next.reserve(frags.size());
            for (const Frag& f : frags) {
                if (f.id >= 0) {
                    next.push_back(f);
                    continue;
                }
                size_t at = f.off;
                const size_t end = f.off + f.len;
                while (at < end) {
                    const size_t hit = text.find(s, at);
                    if (hit == std::string::npos || hit + s.size() > end) break;
                    if (hit > at) next.push_back({-1, at, hit - at});
                    next.push_back({sid, hit, s.size()});
                    at = hit + s.size();
                }
                if (at < end) next.push_back({-1, at, end - at});
            }
            frags.swap(next);
        }
        std::vector<int> out;
        bool prev_special = true;
        for (const Frag& f : frags) {
            if (f.id >= 0) {
                out.push_back(f.id);
                prev_special = true;
                continue;
            }
            if (f.len == 0) continue;
            std::string piece = add_space_prefix_ && prev_special ? std::string(" ") : std::string();
            piece += text.substr(f.off, f.len);
            encode_fragment(escape(piece), out);
            prev_special = false;
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
            const std::string& t = tokens_[id];
            if (types_[id] == kByte && t.size() == 6 && t.compare(0, 3, "<0x") == 0) {
                s += static_cast<char>(std::stoi(t.substr(3, 2), nullptr, 16));
                continue;
            }
            for (size_t i = 0; i < t.size();) {
                if (t.compare(i, 3, kSpace) == 0) {
                    s += ' ';
                    i += 3;
                } else {
                    s += t[i++];
                }
            }
        }
        return s;
    }

    size_t size() const { return tokens_.size(); }

private:
    static constexpr int32_t kUnknown = 2, kControl = 3, kUserDefined = 4, kByte = 6;
    static constexpr const char* kSpace = "\xe2\x96\x81";  // U+2581

    static std::string escape(const std::string& s) {
        std::string o;
        o.reserve(s.size() + s.size() / 2);
        for (char c : s) {
            if (c == ' ') o += kSpace;
            else o += c;
        }
        return o;
    }

    struct Symbol {
        int prev, next;
        size_t off, n;
    };
    struct Bigram {
        int left, right;
        float score;
        size_t size;
    };
    struct Lower {  // llama.cpp's comparator: the highest score on top, ties to the leftmost
        bool operator()(const Bigram& l, const Bigram& r) const {
            return l.score < r.score || (l.score == r.score && l.left > r.left);
        }
    };

    int lookup(const std::string& s) const {
        auto it = id_.find(s);
        return it == id_.end() ? -1 : it->second;
    }

    void encode_fragment(const std::string& text, std::vector<int>& out) const {
        static const uint8_t utf8_len[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
        std::vector<Symbol> sym;
        for (size_t off = 0; off < text.size();) {
            const size_t len = std::min<size_t>(utf8_len[static_cast<uint8_t>(text[off]) >> 4], text.size() - off);
            const int index = static_cast<int>(sym.size());
            sym.push_back({index - 1, off + len == text.size() ? -1 : index + 1, off, len});
            off += len;
        }
        std::priority_queue<Bigram, std::vector<Bigram>, Lower> queue;
        std::unordered_map<std::string, std::pair<int, int>> rev_merge;
        auto try_add = [&](int left, int right) {
            if (left == -1 || right == -1) return;
            const std::string s = text.substr(sym[left].off, sym[left].n + sym[right].n);
            const int id = lookup(s);
            if (id < 0) return;
            queue.push({left, right, scores_[id], s.size()});
            rev_merge[s] = {left, right};
        };
        for (int i = 1; i < static_cast<int>(sym.size()); ++i) try_add(i - 1, i);
        while (!queue.empty()) {
            const Bigram b = queue.top();
            queue.pop();
            Symbol& l = sym[b.left];
            Symbol& r = sym[b.right];
            if (l.n == 0 || r.n == 0 || l.n + r.n != b.size) continue;
            l.n += r.n;
            r.n = 0;
            l.next = r.next;
            if (r.next >= 0) sym[r.next].prev = b.left;
            try_add(l.prev, b.left);
            try_add(b.left, l.next);
        }
        std::function<void(const Symbol&)> resegment = [&](const Symbol& s) {
            const std::string t = text.substr(s.off, s.n);
            const int id = lookup(t);
            if (id >= 0) {
                out.push_back(id);
                return;
            }
            const auto p = rev_merge.find(t);
            if (p == rev_merge.end()) {
                for (size_t j = 0; j < s.n; ++j) {
                    const int bid = byte_id_[static_cast<uint8_t>(text[s.off + j])];
                    if (bid < 0) throw std::runtime_error("SentencePiece GGUF has no byte token for a byte of the text");
                    out.push_back(bid);
                }
                return;
            }
            resegment(sym[p->second.first]);
            resegment(sym[p->second.second]);
        };
        for (int i = sym.empty() ? -1 : 0; i != -1; i = sym[i].next) resegment(sym[i]);
    }

    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int32_t> types_;
    bool add_space_prefix_;
    std::unordered_map<std::string, int> id_;
    std::vector<int> specials_;
    int byte_id_[256];
};

}  // namespace ember
