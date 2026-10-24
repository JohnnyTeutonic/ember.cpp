// ember's vendored Llama 3 encoder (llama3_tokenizer.h, through chat_tokenizer.h) on a vocabulary GGUF:
// reads NUL-separated texts on stdin and prints one line of space-separated ids per text, for
// tools/llama3_vocab_check.py to compare with HF tokenizers' tokenizer.json.
//   g++ -O2 -std=c++17 -I. tools/llama3_vocab_check.cpp -o llama3_vocab_check
//   ./llama3_vocab_check VOCAB.gguf < texts.nul
//   ./llama3_vocab_check QWEN.gguf qwen2 < texts.nul    (a Qwen2/Qwen3 vocabulary: ARCH selects ChatML;
//                                                         tools/thirdparty_tokenizer_check.py)
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "chat_tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: llama3_vocab_check VOCAB.gguf [ARCH] < NUL-separated texts\n";
        return 2;
    }
    ember::ChatVocab v;
    v.g = microtorch::bpe::read_gguf_tokenizer(argv[1]);
    v.arch = argc == 3 ? argv[2] : "";  // resolves a Qwen vocabulary by its markers and architecture
    auto tok = ember::make_chat_setup(std::move(v)).tok;
    if (!tok) {
        std::cerr << argv[1] << ": not a chat vocabulary ember tokenises\n";
        return 1;
    }
    std::cerr << "chat tokenizer: " << tok->kind() << ", " << tok->size() << " tokens\n";
    const std::string all((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    size_t at = 0;
    while (at <= all.size()) {
        size_t end = all.find('\0', at);
        if (end == std::string::npos) end = all.size();
        if (end == at && end == all.size()) break;
        const std::string text = all.substr(at, end - at);
        std::ostringstream line;
        for (int id : tok->encode(text)) line << id << ' ';
        std::cout << line.str() << '\n';
        at = end + 1;
    }
    return 0;
}
