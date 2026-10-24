// ember's chat templates against paperkiln's conformance file and against reference tokenisers
// (experiments/chat_template_2026-10-24/).
//
//   chat_template_check cases CASES.json > report.json
//     paperkiln's tests/chat_template_cases.json against ember's vendored chat_template.h (every
//     check of paperkiln's tests/test_chat_template.cpp: the edge cases, the registry's records,
//     unknown fields refused, Jinja, Modelfile, sha256, resolution, the legacy builder), then each
//     edge case through ember's request paths (chat_request.h): as a /chat request (the system
//     message as "system" and as a leading history entry) and as an OpenAI messages list, rendered
//     by build_chat_prompt and ChatFormat::prompt, the code the server and the WebAssembly build
//     run. A case a request cannot express (a training conversation, a last message that is not the
//     user's, a loss flag) is reported as such. The report holds every request it built, so a
//     driver can post the same ones to a live server.
//
//   chat_template_check render MODEL.gguf < requests.jsonl > out.jsonl
//     the GGUF's chat setup as the session loads it (chat_gguf.h, make_chat_setup), then per input
//     line {"chat": /chat body} or {"openai": /v1/chat/completions body} (optional "ctx", default
//     2^30, and "max_new_tokens", default 1, for the context rule): {"prompt", "ids",
//     "prompt_tokens", "history_turns_used"} or {"error"}. The first output line is
//     {"template", "why", "kind", "stop_ids"}.
//
//   chat_template_check record ID [SYSTEM_MODE]
//     {"paperkiln.chat_template", "tokenizer.chat_template"}: the strings paperkiln's export writes
//     for registry record ID (its JSON dump and its Jinja), optionally with another system mode.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "chat_gguf.h"
#include "chat_request.h"
#include "gguf_parser.h"

using json = nlohmann::json;
namespace chat = microtorch::chat;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// mtstudio's chat_prompt_bpe as it was (paperkiln tests/test_chat_template.cpp's frozen copy).
std::string legacy_prompt(const std::vector<std::pair<std::string, std::string>>& turns, const std::string& user) {
    std::string p;
    for (const auto& t : turns) p += "<|user|>" + t.first + "<|assistant|>" + t.second + "<|endoftext|>";
    return p + "<|user|>" + user + "<|assistant|>";
}

// One request through ember's path: {"text"} or {"refused": message}.
template <class Parse>
json through(const chat::Template& t, const json& req, Parse parse) {
    ember::ChatFormat f;
    f.tmpl = t;
    f.name = t.id();
    try {
        const ember::Conversation c = parse(req);
        const ember::ChatPrompt p = ember::build_chat_prompt(f, c, 1, 1 << 30, [](const std::string& s) {
            return static_cast<int>(s.size());
        });
        return json{{"text", p.prompt}};
    } catch (const ember::RequestError& e) {
        return json{{"refused", e.what()}};
    } catch (const chat::Error& e) {
        return json{{"refused", e.what()}};
    }
}

int cases_cmd(const std::string& path) {
    const json fx = json::parse(slurp(path));
    json report;
    size_t fails = 0;
    auto check = [&](bool ok, const std::string& what) {
        if (!ok) {
            ++fails;
            report["header_failures"].push_back(what);
        }
    };

    // 1. The vendored header, as paperkiln's tests/test_chat_template.cpp checks it.
    size_t rendered = 0, refused = 0;
    for (const auto& c : fx["cases"]) {
        const chat::Template t = c["template"].is_string() ? chat::get(c["template"].get<std::string>())
                                                           : chat::Template::from_json(c["template"]);
        const int thinking = c.contains("thinking") ? (c["thinking"].get<bool>() ? 1 : 0) : -1;
        std::string got;
        bool threw = false;
        try {
            got = chat::render(chat::messages_from_json(c["messages"]), t, c["generation_prompt"].get<bool>(), thinking);
        } catch (const chat::Error&) {
            threw = true;
        }
        const std::string name = c["name"].get<std::string>();
        if (c["expect"].is_null()) {
            check(threw, "case '" + name + "' rendered, the Python twin refuses it");
            ++refused;
        } else {
            check(!threw && got == c["expect"].get<std::string>(), "case '" + name + "' renders differently");
            ++rendered;
        }
    }
    size_t records = 0;
    for (const auto& t : chat::registry()) {
        check(fx["records"].contains(t.id()), "record " + t.id() + " missing from the cases file");
        if (fx["records"].contains(t.id()))
            check(json::parse(t.to_json().dump()) == fx["records"][t.id()], "record " + t.id() + " differs");
        check(chat::Template::from_json(t.to_json()).to_json() == t.to_json(), "record " + t.id() + " round trip");
        ++records;
    }
    check(records == fx["records"].size(), "record count");
    bool unknown = false;
    try {
        chat::json bad = chat::v1().to_json();
        bad["content"] = "lower";
        chat::Template::from_json(bad);
    } catch (const chat::Error&) {
        unknown = true;
    }
    check(unknown, "an unknown content value is accepted");
    chat::Template none = chat::v1();
    none.system_mode = "none";
    check(chat::jinja(chat::v1()) == fx["jinja"]["paperkiln/1"].get<std::string>(), "jinja paperkiln/1");
    check(chat::jinja(none) == fx["jinja"]["paperkiln/1 none"].get<std::string>(), "jinja paperkiln/1 none");
    check(chat::jinja(chat::legacy_word()) == fx["jinja"]["paperkiln/legacy-word"].get<std::string>(), "jinja legacy-word");
    check(chat::modelfile("model.gguf", chat::v1()) == fx["modelfile"]["paperkiln/1"].get<std::string>(), "modelfile");
    size_t shas = 0;
    for (const auto& [k, v] : fx["sha256"].items()) {
        check(chat::detail::sha256_hex(k) == v.get<std::string>(), "sha256 of '" + k + "'");
        ++shas;
    }
    size_t resolved = 0;
    for (const auto& r : fx["resolve"]) {
        const auto got = chat::resolve(r["declared"].get<std::string>(), r["tokens"].get<std::vector<std::string>>(),
                                       r["model"].get<std::string>(), r["chat_template"].get<std::string>(),
                                       r["arch"].get<std::string>());
        const std::string want = r["expect"].is_null() ? "(none)" : r["expect"].get<std::string>();
        const std::string have = got.chat ? got.tmpl.id() : "(none)";
        check(want == have, "resolution " + std::to_string(resolved) + ": " + have + ", expected " + want);
        report["resolutions"].push_back({{"expect", want}, {"got", have}, {"why", got.why}});
        ++resolved;
    }
    const std::vector<std::string> words = {"", "\n", "hi", "naïve café", " spaced ", "x y z"};
    size_t legacy = 0;
    for (size_t n = 0; n < 5; ++n)
        for (size_t w = 0; w < words.size(); ++w) {
            std::vector<std::pair<std::string, std::string>> turns;
            for (size_t i = 0; i < n; ++i) turns.push_back({words[(w + i) % words.size()], words[(w + 2 * i + 1) % words.size()]});
            const std::string user = words[(w + n) % words.size()];
            check(chat::render(chat::from_pairs(turns, &user), chat::v1(), true) == legacy_prompt(turns, user),
                  "legacy prompt " + std::to_string(legacy));
            // ember's own path for the same pairs: a /chat request in both history forms.
            json pairs = json::array(), roles = json::array();
            for (const auto& t : turns) {
                pairs.push_back({{"user", t.first}, {"assistant", t.second}});
                roles.push_back({{"role", "user"}, {"content", t.first}});
                roles.push_back({{"role", "assistant"}, {"content", t.second}});
            }
            for (const json& h : {pairs, roles}) {
                const json out = through(chat::v1(), json{{"user_input", user}, {"history", h}}, ember::chat_conversation);
                check(out.contains("text") && out["text"] == legacy_prompt(turns, user),
                      "legacy prompt through /chat " + std::to_string(legacy));
            }
            ++legacy;
        }
    report["header"] = {{"cases_rendered", rendered}, {"cases_refused", refused}, {"records", records},
                        {"jinja_modelfile", 4}, {"sha256", shas}, {"resolutions", resolved},
                        {"legacy_prompts", legacy}, {"failures", fails}};

    // 2. Each edge case through ember's request paths.
    size_t same = 0, differ = 0, inexpressible = 0;
    for (const auto& c : fx["cases"]) {
        const chat::Template t = c["template"].is_string() ? chat::get(c["template"].get<std::string>())
                                                           : chat::Template::from_json(c["template"]);
        const json& msgs = c["messages"];
        json row = {{"name", c["name"]}, {"template", t.id()}, {"expect", c["expect"]}};
        std::string why_not;
        bool has_loss = false;
        for (const auto& m : msgs) has_loss = has_loss || m.contains("loss");
        if (!c["generation_prompt"].get<bool>()) why_not = "a training conversation (requests are prompts)";
        else if (has_loss) why_not = "a loss flag (no request field carries one)";
        auto verdict = [&](const json& out) {
            const bool ok = c["expect"].is_null() ? out.contains("refused")
                                                  : out.contains("text") && out["text"] == c["expect"];
            return ok;
        };
        json paths = json::object();
        if (why_not.empty()) {
            // /chat: the last message is user_input; a leading system message goes as "system" or as
            // the first history entry; every other message is a {"role", "content"} history entry.
            if (msgs.back().value("role", "") == "user") {
                json hist = json::array();
                for (size_t i = 0; i + 1 < msgs.size(); ++i) hist.push_back({{"role", msgs[i]["role"]}, {"content", msgs[i]["content"]}});
                json base = {{"user_input", msgs.back()["content"]}};
                if (c.contains("thinking")) base["thinking"] = c["thinking"];
                json in_history = base;
                in_history["history"] = hist;
                paths["chat_history_system"] = {{"request", in_history}};
                if (!hist.empty() && hist[0]["role"] == "system") {
                    json field = base;
                    field["system"] = hist[0]["content"];
                    field["history"] = json(hist.begin() + 1, hist.end());
                    paths["chat_system_field"] = {{"request", field}};
                }
            } else {
                paths["chat_note"] = "not a /chat request: the last message is not the user's";
            }
            json oa = {{"messages", msgs}};
            if (c.contains("thinking")) oa["thinking"] = c["thinking"];
            paths["openai"] = {{"request", oa}};
            for (auto& [k, v] : paths.items()) {
                if (!v.is_object()) continue;
                const json out = k == "openai" ? through(t, v["request"], ember::openai_conversation)
                                               : through(t, v["request"], ember::chat_conversation);
                v["result"] = out;
                v["same"] = verdict(out);
                (verdict(out) ? same : differ) += 1;
            }
        } else {
            ++inexpressible;
            row["not_a_request"] = why_not;
        }
        row["paths"] = paths;
        report["paths"].push_back(row);
    }
    report["paths_summary"] = {{"same", same}, {"differ", differ}, {"cases_not_a_request", inexpressible}};

    // 3. /chat role-form history pairs strictly (paperkiln 7a6c7659, mtstudio's test_history_pairs_strictly):
    // a well-formed role history renders as the same turns in pair form, and each violation is refused
    // naming its rule.
    {
        const json u = {{"role", "user"}, {"content", "hi"}}, a = {{"role", "assistant"}, {"content", "hello there"}};
        auto req = [](const json& hist) {
            json r = json::object();
            r["user_input"] = "hello";
            r["history"] = hist;
            return r;
        };
        json pair_entry = json::object();
        pair_entry["user"] = "hi";
        pair_entry["assistant"] = "hello there";
        json narrator = json::object();
        narrator["role"] = "narrator";
        narrator["content"] = "x";
        const json role = through(chat::v1(), req(json::array({u, a})), ember::chat_conversation);
        const json pair = through(chat::v1(), req(json::array({pair_entry})), ember::chat_conversation);
        json strict = json::object();
        strict["role_equals_pair"] = role.contains("text") && role == pair;
        strict["role_text"] = role;
        const std::vector<std::pair<json, std::string>> bad = {
            {json::array({a, u}), "without a user message before it"},
            {json::array({u, u, a}), "two user messages"},
            {json::array({u, a, u}), "ends with a user message"},
            {json::array({u, narrator}), "role"}};
        size_t refused_ok = 0;
        for (const auto& [hist, rule] : bad) {
            const json out = through(chat::v1(), req(hist), ember::chat_conversation);
            const bool ok = out.contains("refused") && out["refused"].get<std::string>().find(rule) != std::string::npos;
            refused_ok += ok;
            strict["violations"].push_back({{"history", hist}, {"rule", rule}, {"result", out}, {"ok", ok}});
        }
        strict["violations_refused"] = refused_ok;
        check(strict["role_equals_pair"].get<bool>() && refused_ok == bad.size(), "strict history pairing");
        report["history_strict"] = strict;
    }
    std::cout << report.dump(1) << "\n";
    return fails == 0 ? 0 : 1;
}

int render_cmd(const std::string& gguf) {
    GGUFData gd = load_gguf_meta(gguf, true);
    ember::ChatSetup s = ember::make_chat_setup(ember::chat_vocab_from_gguf(gd, gguf));
    json head = {{"why", s.why}, {"resolved", s.resolved}};
    head["template"] = s.resolved ? json(s.tmpl.id()) : json(nullptr);
    if (s.tok) {
        head["kind"] = s.tok->kind();
        head["stop_ids"] = s.tok->format().stop_ids();
    }
    std::cout << head.dump() << "\n";
    if (!s.tok) return 1;
    const ember::ChatFormat& f = s.tok->format();
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const json in = json::parse(line);
        json out;
        try {
            const bool oa = in.contains("openai");
            const ember::Conversation c = oa ? ember::openai_conversation(in["openai"]) : ember::chat_conversation(in["chat"]);
            const ember::ChatPrompt p = ember::build_chat_prompt(
                f, c, in.value("max_new_tokens", 1), in.value("ctx", 1 << 30),
                [&](const std::string& text) { return static_cast<int>(s.tok->encode(text).size()); });
            if (!p.error.empty()) throw ember::RequestError(p.error);
            out = {{"prompt", p.prompt}, {"ids", s.tok->encode(p.prompt)}, {"prompt_tokens", p.prompt_tokens},
                   {"history_turns_used", p.turns_used}};
        } catch (const std::exception& e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "cases") return cases_cmd(argv[2]);
        if (argc == 3 && std::string(argv[1]) == "render") return render_cmd(argv[2]);
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "record") {
            // The keys paperkiln's export writes for a record (mtstudio: t.to_json().dump() and
            // chat::jinja(t)); argv[3] "none" sets the system mode to none.
            chat::Template t = chat::get(argv[2]);
            if (argc == 4) t.system_mode = argv[3];
            json out = {{"paperkiln.chat_template", t.to_json().dump()}};
            if (t.name == "paperkiln") out["tokenizer.chat_template"] = chat::jinja(t);
            std::cout << out.dump() << "\n";
            return 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "chat_template_check: %s\n", e.what());
        return 2;
    }
    std::fprintf(stderr,
                 "usage: chat_template_check cases CASES.json | render MODEL.gguf < requests.jsonl | record ID [MODE]\n");
    return 2;
}
