// e22_page_gate: the browser page's own C entry points (deploy/wasm/ember_wasm.cpp, included whole)
// driven the same way on both sides: built natively (linked with the CPU build's libtinyllama_core.a,
// a stub emscripten.h) and with the page's em++ sources and flags (plus NODERAWFS, run in node).
//
//   e22_page_gate greedy MODEL.gguf PROMPTS.jsonl N OUT_DIR
//       ember_load + ember_chat per prompt, the parity request (60 new tokens, temperature 1,
//       top_k 1, top_p 1, seed 0). TINYLLAMA_LOGITS_DUMP=OUT_DIR/logits.txt is set before the
//       first call, so the served decode path appends the full logits row of every sampled position.
//       OUT_DIR/greedy.jsonl: {i, prompt, reply, tokens, stop_reason, prompt_tokens, prompt_ids, ids,
//       stop_id} per prompt.
//   e22_page_gate tf MODEL.gguf FIXTURE.jsonl OUT.f32
//       teacher forcing: per fixture row, score_rows(prompt_ids + ids, len(prompt_ids) - 1), i.e. the
//       logits predicting every reply token and the token after the last one; float32 rows appended
//       to OUT.f32, the row counts to OUT.f32.json.
#include "deploy/wasm/ember_wasm.cpp"

#include <fstream>
#include <iostream>

static std::vector<std::string> read_lines(const std::string& p) {
    std::ifstream f(p);
    std::vector<std::string> out;
    std::string l;
    while (std::getline(f, l))
        if (!l.empty()) out.push_back(l);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: e22_page_gate greedy MODEL PROMPTS N OUT_DIR | tf MODEL FIXTURE OUT.f32\n";
        return 2;
    }
    const std::string mode = argv[1], model = argv[2];
    if (mode == "greedy") {
        if (argc != 6) return 2;
        const std::string dir = argv[5];
        const std::string dump = dir + "/logits.txt";
        std::remove(dump.c_str());
        setenv("TINYLLAMA_LOGITS_DUMP", dump.c_str(), 1);
        if (ember_load(model.c_str()) != 1) { std::cerr << "ember_load failed\n"; return 1; }
        std::cerr << "ir_doc empty (tuned llama path): " << (std::string(ember_ir_doc()).empty() ? "yes" : "no")
                  << ", ctx " << ember_ctx() << "\n";
        const auto rows = read_lines(argv[3]);
        const int n = std::min<int>(std::stoi(argv[4]), static_cast<int>(rows.size()));
        std::ofstream out(dir + "/greedy.jsonl");
        for (int i = 0; i < n; ++i) {
            const json r = json::parse(rows[i]);
            const json req = {{"user_input", r["prompt"]}, {"max_new_tokens", 60}, {"temperature", 1.0},
                              {"top_k", 1}, {"top_p", 1.0}, {"seed", 0}};
            const json o = json::parse(ember_chat(req.dump().c_str()));
            if (o.contains("error")) { std::cerr << "prompt " << i << ": " << o["error"] << "\n"; return 1; }
            json rec = {{"i", i}, {"prompt", r["prompt"]}, {"reply", o["reply"]}, {"tokens", o["tokens"]},
                        {"stop_reason", o["stop_reason"]}, {"prompt_tokens", o["prompt_tokens"]},
                        {"prompt_ids", g_session->last_prompt_ids()}, {"ids", g_session->last_generated_ids()},
                        {"stop_id", g_session->last_stop_id()}};
            out << rec.dump() << "\n";
            out.flush();
            std::cerr << "[" << i << "] " << o["tokens"] << " tokens, " << o["stop_reason"] << "\n";
        }
        return 0;
    }
    if (mode == "tf") {
        if (ember_load(model.c_str()) != 1) { std::cerr << "ember_load failed\n"; return 1; }
        const auto rows = read_lines(argv[3]);
        const std::string outp = argv[4];
        std::ofstream bin(outp, std::ios::binary);
        json counts = json::array();
        for (size_t k = 0; k < rows.size(); ++k) {
            const json r = json::parse(rows[k]);
            std::vector<int> ids = r["prompt_ids"].get<std::vector<int>>();
            const int first = static_cast<int>(ids.size()) - 1;
            for (int t : r["ids"].get<std::vector<int>>()) ids.push_back(t);
            const std::vector<float> lg = g_session->score_rows(ids, first, false);
            bin.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * sizeof(float)));
            counts.push_back({{"i", r["i"]}, {"rows", static_cast<int>(ids.size()) - first},
                              {"vocab", g_session->score_vocab()}});
            std::cerr << "[" << k << "] " << (ids.size() - first) << " rows\n";
        }
        std::ofstream(outp + ".json") << counts.dump() << "\n";
        return 0;
    }
    std::cerr << "unknown mode " << mode << "\n";
    return 2;
}
