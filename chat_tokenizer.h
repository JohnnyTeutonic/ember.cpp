#pragma once
// The tokenizer behind ember's paperkiln chat mode (TinyLlamaSession::is_bpe()): one interface over
// the two vocabulary forms paperkiln exports, both vendored from paperkiln.
// - tokenizer.ggml.model "bpe": paperkiln's byte-level BPE with id-pair merges (bpe_tokenizer.h).
// - tokenizer.ggml.model "gpt2" with pre "llama-bpe" and ids 0-3 = <unk>, <|endoftext|>, <|user|>,
//   <|assistant|>: the llama.cpp-compatible Llama 3 scheme (llama3_tokenizer.h), encoded exactly as
//   llama.cpp does. A real Llama 3 GGUF has the same model/pre pair but other tokens at ids 0-3,
//   so it keeps ember's Llama 3 path.
// Either way the chat template, the stops at ids 1-3 and the /chat contract are the same.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bpe_tokenizer.h"
#include "llama3_tokenizer.h"

namespace ember {

class ChatTokenizer {
public:
    virtual ~ChatTokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& ids) const = 0;
    virtual size_t size() const = 0;
    virtual const char* kind() const = 0;
};

template <class T>
class ChatTokenizerOf : public ChatTokenizer {
public:
    ChatTokenizerOf(T t, const char* kind) : t_(std::move(t)), kind_(kind) {}
    std::vector<int> encode(const std::string& text) const override { return t_.encode(text); }
    std::string decode(const std::vector<int>& ids) const override { return t_.decode(ids); }
    size_t size() const override { return t_.size(); }
    const char* kind() const override { return kind_; }

private:
    T t_;
    const char* kind_;
};

// True when a GGUF's tokens start with paperkiln's four chat specials.
inline bool has_chat_specials(const std::vector<std::string>& tokens) {
    const auto& s = microtorch::bpe::specials();
    if (tokens.size() < s.size()) return false;
    for (size_t i = 0; i < s.size(); ++i)
        if (tokens[i] != s[i]) return false;
    return true;
}

// The chat tokenizer for a GGUF's tokenizer block, or nullptr when the GGUF is not a paperkiln chat
// model (ember's own tokenizers handle it).
inline std::unique_ptr<ChatTokenizer> make_chat_tokenizer(microtorch::bpe::GgufTokenizer g) {
    if (g.model == "bpe") {
        if (g.merges.empty()) throw std::runtime_error("BPE GGUF has no tokenizer.ggml.merges");
        return std::make_unique<ChatTokenizerOf<microtorch::bpe::Tokenizer>>(
            microtorch::bpe::Tokenizer(std::move(g.tokens), g.merges), "bpe");
    }
    if (g.model == "gpt2" && g.pre == "llama-bpe" && has_chat_specials(g.tokens)) {
        if (g.merges.empty()) throw std::runtime_error("llama-bpe GGUF has no tokenizer.ggml.merges");
        return std::make_unique<ChatTokenizerOf<microtorch::llama3::Tokenizer>>(
            microtorch::llama3::Tokenizer(std::move(g.tokens), g.merges, g.token_type), "llama3_bpe");
    }
    return nullptr;
}

}  // namespace ember
