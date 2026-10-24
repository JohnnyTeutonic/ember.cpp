#pragma once
// A GGUF's tokenizer block and template keys as chat_tokenizer.h's ChatVocab, from the metadata
// ember's GGUF parser has already read (TinyLlamaSession's load, tools/chat_template_check.cpp).
#include <string>
#include <variant>

#include "bpe_tokenizer.h"
#include "chat_tokenizer.h"
#include "gguf_structs.h"

namespace ember {

inline ChatVocab chat_vocab_from_gguf(const GGUFData& gd, const std::string& path) {
    ChatVocab v;
    auto str = [&](const char* key) {
        auto it = gd.metadata.find(key);
        const std::string* s = it == gd.metadata.end() ? nullptr : std::get_if<std::string>(&it->second);
        return s ? *s : std::string();
    };
    v.arch = str("general.architecture");
    v.chat_template = str("tokenizer.chat_template");
    v.declared = str("paperkiln.chat_template");
    auto asp = gd.metadata.find("tokenizer.ggml.add_space_prefix");
    if (asp != gd.metadata.end())
        if (const bool* b = std::get_if<bool>(&asp->second)) v.add_space_prefix = *b;
    // Qwen and SentencePiece vocabularies come from the metadata already parsed:
    // read_gguf_tokenizer reads the whole file into memory (1.3 GB for Qwen2.5-0.5B f16, 1.1 GB for
    // Gemma-3-270m f32). paperkiln files keep that reader.
    const std::string model = str("tokenizer.ggml.model");
    if ((str("tokenizer.ggml.pre") == "qwen2" || model == "llama") && !gd.tokenizer_tokens.empty()) {
        v.g.model = model;
        v.g.pre = str("tokenizer.ggml.pre");
        v.g.tokens = gd.tokenizer_tokens;
        v.g.merges = gd.tokenizer_merges;
        v.g.token_type.assign(gd.tokenizer_token_types.begin(), gd.tokenizer_token_types.end());
        v.scores = gd.tokenizer_scores;
    } else {
        v.g = microtorch::bpe::read_gguf_tokenizer(path);
    }
    return v;
}

}  // namespace ember
