/**
 * @file server.cpp
 * @brief HTTP server implementation for TinyLlama chat interface.
 *
 * This server provides a REST API for interacting with TinyLlama models.
 * It handles both GGUF and SafeTensors models, applying appropriate prompt
 * formatting for each:
 * - For GGUF models: Applies Q:A format
 * - For SafeTensors models: Uses the tokenizer's chat template
 *
 * The server exposes a /chat endpoint that accepts POST requests with JSON body:
 * {
 *   "user_input": "string",      // Required: The prompt text
 *   "temperature": float,        // Optional: Sampling temperature (default: 0.1)
 *   "max_new_tokens": int,       // Optional: Max tokens to generate (default: 60)
 *   "top_k": int,               // Optional: Top-K sampling parameter (default: 40)
 *   "top_p": float              // Optional: Top-P sampling parameter (default: 0.9)
 * }
 *
 * Usage:
 *   tinyllama_server [model_path] [port] [host] [www_path]
 *     model_path: Path to model directory or .gguf file (default: data)
 *     port: Server port (default: 8080)
 *     host: Host to bind to (default: localhost)
 *     www_path: Path to static web files (default: ./www)
 */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)  
                                 
#pragma warning(disable : 4996)  
#endif

#include "httplib.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cstdio>
#include <random>
#include <ctime>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>  
#include <vector>

#include "api.h"
#include "chat_request.h"
#include "determinism.h"
#include "logger.h"
#include "score.h"
#include "tokenizer.h"
#include "model_macros.h"
#include "utf8_text.h"

using json = nlohmann::json;

// Length of the longest prefix of s that ends on a whole UTF-8 character, so
// a streamed piece never splits a multi-byte character (a BPE token can end
// inside one).
static size_t utf8_complete_prefix(const std::string& s) {
  size_t n = s.size();
  for (size_t back = 1; back <= 3 && back <= n; ++back) {
    const unsigned char c = static_cast<unsigned char>(s[n - back]);
    if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
    const size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
    return need > back ? n - back : n;
  }
  return n;
}

// ---- BPE chat core, shared by /chat and /v1/chat/completions ------------
// Chat-mode models: the prompt is the conversation rendered by the model's
// chat template, the record resolved at load (chat_tokenizer.h; LOCAL_LLM_STACK.md
// section 3, point 3): paperkiln/1 writes <|user|>U<|assistant|>A<|endoftext|>
// per past turn then <|user|>U<|assistant|>; the third-party records write
// ChatML (Qwen2.5, Qwen3) or Gemma 3's turns as HF's apply_chat_template does.
// The request becomes a Conversation (chat_request.h), and the oldest whole
// exchanges are dropped until prompt + max_new_tokens fits the context; the
// system message is pinned.
struct BpeChatRequest {
  ember::Conversation conv;
  int max_new_tokens = 60;
  float temperature = 0.1f;
  int top_k = 40;
  float top_p = 0.9f;
  bool has_seed = false;
  unsigned seed = 0;
  bool want_ids = false;
  float repeat_penalty = 1.0f;  // the contract's repetition penalty (point 4); 1 = off
  int repeat_last_n = 64;
};

// repeat_penalty / repeat_last_n from a request body (LOCAL_LLM_STACK.md
// section 3, point 4): p in [1, 2], n in [0, ctx]. Public mode clamps; other
// requests get an error message (HTTP 400). Empty string: fine.
static std::string bpe_repeat_fields(const json& body, BpeChatRequest& r, int ctx, bool clamp) {
  if (body.contains("repeat_penalty")) {
    if (!body["repeat_penalty"].is_number()) return "repeat_penalty must be a number";
    r.repeat_penalty = body["repeat_penalty"].get<float>();
  }
  if (body.contains("repeat_last_n")) {
    if (!body["repeat_last_n"].is_number_integer()) return "repeat_last_n must be an integer";
    r.repeat_last_n = body["repeat_last_n"].get<int>();
  } else {
    r.repeat_last_n = std::min(r.repeat_last_n, ctx);  // default min(64, ctx), as mtstudio
  }
  if (clamp) {
    r.repeat_penalty = std::min(std::max(r.repeat_penalty, 1.0f), 2.0f);
    r.repeat_last_n = std::min(std::max(r.repeat_last_n, 0), ctx);
    return "";
  }
  if (!(r.repeat_penalty >= 1.0f && r.repeat_penalty <= 2.0f)) return "repeat_penalty must be in [1.0, 2.0]";
  if (r.repeat_last_n < 0 || r.repeat_last_n > ctx)
    return "repeat_last_n must be in [0, " + std::to_string(ctx) + "]";
  return "";
}

struct BpePrompt {
  std::string prompt;
  int prompt_tokens = 0;
  int turns_used = 0;
  std::string error;  // non-empty: the newest message alone does not fit
};

// Throws chat::Error when the template refuses the conversation (HTTP 400).
static BpePrompt bpe_build_prompt(tinyllama::TinyLlamaSession& sess, const BpeChatRequest& r) {
  const ember::ChatPrompt p = ember::build_chat_prompt(
      *sess.chat_format(), r.conv, r.max_new_tokens, sess.get_config().max_position_embeddings,
      [&](const std::string& text) { return sess.bpe_token_count(text); });
  BpePrompt out;
  out.prompt = p.prompt;
  out.prompt_tokens = p.prompt_tokens;
  out.turns_used = p.turns_used;
  out.error = p.error;
  return out;
}

// Generate (the caller holds the session lock) and return the /chat reply
// object. on_piece, if set, receives the reply text as it is produced:
// leading whitespace skipped, never a split UTF-8 character.
static json bpe_run(tinyllama::TinyLlamaSession& sess, const BpeChatRequest& r, const BpePrompt& p,
                    std::chrono::steady_clock::time_point t_request, double waited_ms,
                    const std::function<void(const std::string&)>& on_piece) {
  if (r.has_seed) sess.set_seed(r.seed);
  std::vector<int> ids;
  size_t sent = 0;
  if (on_piece) {
    sess.set_token_callback([&](int id) {
      ids.push_back(id);
      const std::string full = sess.bpe_decode(ids);
      const size_t end = utf8_complete_prefix(full);
      const size_t start = full.find_first_not_of(" \t\n\v\f\r");
      if (start == std::string::npos || start >= end) return;
      if (sent < start) sent = start;
      if (end > sent) {
        on_piece(ember::utf8_valid(full.substr(sent, end - sent)));
        sent = end;
      }
    });
  }
  sess.set_repeat_penalty(r.repeat_penalty, r.repeat_last_n);
  const std::string reply = sess.generate(p.prompt, r.max_new_tokens, r.temperature, r.top_k, r.top_p, "", false);
  sess.set_repeat_penalty(1.0f, 64);
  sess.set_token_callback(nullptr);
  const tinyllama::TtftBreakdown& tb = sess.last_ttft();
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  json res_json;
  // the decoded bytes can end inside a character or hold byte tokens that make none (utf8_text.h)
  res_json["reply"] = ember::utf8_valid(reply);
  res_json["stop_reason"] = sess.last_stop_reason();
  res_json["tokens"] = sess.last_generated_tokens();
  res_json["prompt_tokens"] = p.prompt_tokens;
  res_json["history_turns_used"] = p.turns_used;
  // Opt-in exact ids (preference data needs the sampled ids, not a
  // re-encoding of the stripped reply text).
  if (r.want_ids) {
    res_json["prompt_ids"] = sess.last_prompt_ids();
    res_json["ids"] = sess.last_generated_ids();
    res_json["stop_id"] = sess.last_stop_id();
  }
  res_json["ttft"] = {{"first_token", tb.first_token},
                      {"queue_ms", waited_ms},
                      {"request_to_first_token_ms", tb.first_token ? ms(t_request, tb.t_first_token) : -1.0},
                      {"ttft_ms", tb.ttft_ms},
                      {"num_prompt_tokens", tb.num_prompt_tokens},
                      {"prefix_reused_tokens", tb.prefix_reused_tokens}, {"spec_drafted", tb.spec_drafted}, {"spec_accepted", tb.spec_accepted}, {"spec_passes", tb.spec_passes},
                      {"batch_prefill", tb.batch_prefill},
                      {"prefill_wall_ms", tb.prefill_wall_ms},
                      {"prefill_gpu_span_ms", tb.prefill_gpu_span_ms}};
  if (tb.spec_draft_calls > 0) {  // speculation on (EMBER_SPEC): the drafter's own time
    res_json["ttft"]["spec_draft_ms"] = tb.spec_draft_ms;
    res_json["ttft"]["spec_draft_calls"] = tb.spec_draft_calls;
  }
  return res_json;
}

int main(int argc, char** argv) {
  std::string model_dir = "data";
  std::string host = "localhost";
  int port = 8080;
  std::string www_path = "./www";  
  
  if (argc > 1) {
    model_dir = argv[1];
  }
  if (argc > 2) {
    port = std::stoi(argv[2]);
  }
  if (argc > 3) {
    host = argv[3];
  }
  if (argc > 4) {
    www_path = argv[4];
  }
  int n_gpu_layers = -1;  // -1 all layers on GPU, 0 none, N that many
  if (argc > 5) {
    n_gpu_layers = std::stoi(argv[5]);
  }

  Logger::info("Starting TinyLlama Chat Server...");
  if (ember::deterministic())
    Logger::info("EMBER_DETERMINISTIC set: deterministic CUDA mode (run-to-run bit-identical logits; /health "
                 "reports \"deterministic\": true).");

  std::shared_ptr<tinyllama::TinyLlamaSession> session;
  try {
    Logger::info("Loading model from: " + model_dir);
    session = std::make_shared<tinyllama::TinyLlamaSession>(model_dir, "tokenizer.json", 4, n_gpu_layers, true);
    Logger::info("Model loaded successfully.");
  } catch (const std::exception& e) {
    Logger::error(std::string("Failed to load model: ") + e.what());
    return 1;
  }
  
  httplib::Server svr;
  // EMBER_PUBLIC=1: settings for serving strangers on the open internet.
  // Warnings-only logging (nothing a visitor types reaches the log), a small
  // request size cap, socket timeouts, a bounded wait queue, and per-request
  // caps on input, history and generation (applied in /chat below).
  static const bool public_mode = [] {
    const char* v = std::getenv("EMBER_PUBLIC");
    return v && std::string(v) == "1";
  }();
  if (public_mode) {
    Logger::set_level(Logger::Level::WARNING);
    svr.set_payload_max_length(64 * 1024);
    svr.set_read_timeout(10, 0);
    svr.set_write_timeout(30, 0);
  }
  
  if (std::filesystem::exists(www_path) &&
      std::filesystem::is_directory(www_path)) {
    Logger::info("Serving static files from: " + www_path);
    bool mount_ok = svr.set_mount_point("/", www_path);
    if (!mount_ok) {
      Logger::error("Failed to mount static file directory: " + www_path);
      return 1;
    }
  } else {
    Logger::info("Static file directory not found: " + www_path +
                 ". Web client will not be served.");
  }

  // One session (one KV cache, one set of device buffers, one RNG) serves
  // every request, and httplib runs handlers on a thread pool: requests
  // must take turns. Without this, parallel /chat calls raced on the
  // session and returned HTTP 500s or each other's state (19 Oct 2026,
  // tools/chat_concurrency_check.py).
  static std::mutex session_mu;
  // Requests in a handler or a stream provider (waiting or generating).
  static std::atomic<int> waiting{0};
  struct Waiting {
    Waiting() { ++waiting; }
    ~Waiting() { --waiting; }
  };
  svr.Post("/chat", [&session](const httplib::Request& req,
                               httplib::Response& res) {
        // TTFT clock starts here: the earliest point the server owns the request.
        const auto t_request = std::chrono::steady_clock::now();
        // Public mode: refuse rather than queue without bound.
        Waiting waiting_guard;
        if (public_mode && waiting.load() > 8) {
          res.status = 503;
          res.set_header("Access-Control-Allow-Origin", "*");
          res.set_header("Retry-After", "2");
          res.set_content(json{{"error", "busy: too many people chatting at once, try again in a moment"}}.dump(),
                          "application/json");
          return;
        }
        std::lock_guard<std::mutex> session_lock(session_mu);
        const double queue_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_request).count();
        Logger::info("Received request for /chat");
    res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");

        std::string user_input_from_client;

    float temperature = 0.1f;  // Lower temperature for more focused chat responses
        int max_new_tokens = 60;
    int top_k = 40;           // Default top-k value
    float top_p = 0.9f;       // Default top-p value

        try {
          json req_json = json::parse(req.body);
          if (req_json.contains("user_input")) {
            user_input_from_client = req_json["user_input"].get<std::string>();
          } else {
            throw std::runtime_error("Missing 'user_input' field in request JSON");
          }
          
          if (req_json.contains("max_new_tokens"))
            max_new_tokens = req_json["max_new_tokens"].get<int>();
          if (req_json.contains("temperature"))
            temperature = req_json["temperature"].get<float>();
      if (req_json.contains("top_k"))
        top_k = req_json["top_k"].get<int>();
      if (req_json.contains("top_p"))
        top_p = req_json["top_p"].get<float>();

          if (public_mode) {
            constexpr size_t kMaxText = 600;  // bytes per message
            if (user_input_from_client.size() > kMaxText) {
              res.status = 400;
              res.set_header("Access-Control-Allow-Origin", "*");
              res.set_content(json{{"error", "message too long (600 characters at most)"}}.dump(), "application/json");
              return;
            }
            max_new_tokens = std::min(std::max(max_new_tokens, 1), 64);
            temperature = std::min(std::max(temperature, 0.0f), 1.5f);
            top_k = std::min(std::max(top_k, 0), 200);
            top_p = std::min(std::max(top_p, 0.05f), 1.0f);
            req_json.erase("return_ids");
            if (req_json.contains("system") && req_json["system"].is_string() &&
                req_json["system"].get<std::string>().size() > kMaxText)
              req_json["system"] = req_json["system"].get<std::string>().substr(0, kMaxText);
            if (req_json.contains("history") && req_json["history"].is_array()) {
              json& hist = req_json["history"];
              // The context holds ~5 turns anyway; a leading system entry is pinned.
              const bool lead_system = !hist.empty() && hist[0].is_object() && hist[0].value("role", "") == "system";
              while (hist.size() > 16) hist.erase(hist.begin() + (lead_system ? 1 : 0));
              for (auto& h : hist)
                for (const char* key : {"content", "user", "assistant"})
                  if (h.is_object() && h.contains(key) && h[key].is_string() &&
                      h[key].get<std::string>().size() > kMaxText)
                    h[key] = h[key].get<std::string>().substr(0, kMaxText);
            }
          }

          // Chat-mode models: the same /chat contract as mtstudio chat
          // (LOCAL_LLM_STACK.md section 3, points 3-5). The request is a prompt
          // conversation (optional "system" or leading system history entry,
          // the history, user_input, "thinking"), rendered by the model's chat
          // template, with the oldest whole exchanges dropped until
          // prompt + max_new_tokens fits the context (the system message pinned).
          if (session->is_bpe()) {
            BpeChatRequest r;
            try {
              r.conv = ember::chat_conversation(req_json);
            } catch (const ember::RequestError& e) {
              res.status = 400;
              res.set_content(json{{"error", e.what()}}.dump(), "application/json");
              return;
            }
            r.max_new_tokens = max_new_tokens;
            r.temperature = temperature;
            r.top_k = top_k;
            r.top_p = top_p;
            r.has_seed = req_json.contains("seed");
            r.seed = r.has_seed ? req_json["seed"].get<unsigned>() : 0u;
            r.want_ids = req_json.value("return_ids", false);
            const std::string rp_error = bpe_repeat_fields(
                req_json, r, session->get_config().max_position_embeddings, public_mode);
            if (!rp_error.empty()) {
              res.status = 400;
              res.set_content(json{{"error", rp_error}}.dump(), "application/json");
              return;
            }
            tinyllama::TinyLlamaSession* sess = session.get();
            BpePrompt bp;
            try {
              bp = bpe_build_prompt(*sess, r);
            } catch (const microtorch::chat::Error& e) {
              res.status = 400;  // the template refuses the conversation (section 3, point 3)
              res.set_content(json{{"error", e.what()}}.dump(), "application/json");
              return;
            }
            if (!bp.error.empty()) {
              res.status = 400;
              res.set_content(json{{"error", bp.error}}.dump(), "application/json");
              return;
            }
            if (!req_json.value("stream", false)) {
              res.set_content(bpe_run(*sess, r, bp, t_request, queue_ms, nullptr).dump(), "application/json");
              return;
            }
            // "stream": true -> server-sent events: {"token": text} as the
            // reply is generated, then the full reply object with
            // "done": true. The provider runs after this handler returns, so
            // it takes the session lock itself.
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider(
                "text/event-stream", [r, bp, t_request, sess](size_t, httplib::DataSink& sink) {
                  Waiting stream_waiting;
                  std::lock_guard<std::mutex> stream_lock(session_mu);
                  const double waited = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t_request).count();
                  auto send = [&sink](const json& j) {
                    const std::string ev = "data: " + j.dump() + "\n\n";
                    sink.write(ev.data(), ev.size());
                  };
                  json final_json = bpe_run(*sess, r, bp, t_request, waited,
                                            [&](const std::string& piece) { send(json{{"token", piece}}); });
                  final_json["done"] = true;
                  send(final_json);
                  sink.done();
                  return true;
                });
            return;
          }

          Logger::info("Processing user input: " +
                       user_input_from_client.substr(0, 100) + "...");

          const ModelConfig& config = session->get_config();
          std::string prompt_for_session_generate;
          bool use_q_a_format_for_session_generate = false;

          const Tokenizer* tokenizer = session->get_tokenizer(); 

          if (config.is_gguf_file_loaded) {
        prompt_for_session_generate = user_input_from_client;
            // WORD_LEVEL models (transformer_cpp exports) understand their
            // training format, not free-form chat. Probe the embedded vocab
            // once to tell dialogue models (DailyDialog: has "assistant:")
            // from story models (TinyStories-Instruct), then wrap natural
            // input in the matching template. Power users typing explicit
            // "summary:/words:/story:/user:" prompts pass through verbatim.
            if (config.tokenizer_family == ModelConfig::TokenizerFamily::WORD_LEVEL) {
                static int dialogue_model = -1;  // -1 unknown, probed once
                if (dialogue_model < 0 && tokenizer) {
                    auto ids = tokenizer->encode("assistant:", false, false);
                    int unk = 0;  // word-level UNK id
                    dialogue_model = (!ids.empty() && ids[0] != unk) ? 1 : 0;
                    Logger::info(std::string("WORD_LEVEL model type probed: ") +
                                 (dialogue_model ? "dialogue" : "story"));
                }
                std::string lower = user_input_from_client;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                bool has_field = lower.find("story:") != std::string::npos ||
                                 lower.find("summary:") != std::string::npos ||
                                 lower.find("words:") != std::string::npos ||
                                 lower.rfind("once upon", 0) == 0 ||
                                 lower.rfind("user:", 0) == 0;
                if (!has_field) {
                    if (dialogue_model == 1) {
                        // The word-level record (paperkiln/legacy-word, resolved at load):
                        // "user: X assistant:". History and system are not served on this path.
                        const microtorch::chat::Template* wt = session->chat_template();
                        prompt_for_session_generate =
                            wt ? microtorch::chat::render(
                                     microtorch::chat::from_pairs({}, &user_input_from_client), *wt, true)
                               : "user: " + user_input_from_client + " assistant:";
                        Logger::info("WORD_LEVEL: wrapped input as dialogue turn.");
                    } else {
                        prompt_for_session_generate =
                            "summary: " + user_input_from_client + " story:";
                        Logger::info("WORD_LEVEL: wrapped free-form input as story request.");
                    }
                }
            }
            // Check for Llama 3 tokenizer family to disable Q&A for it
            if (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
                use_q_a_format_for_session_generate = false;
                Logger::info("GGUF Llama 3 model detected (via tokenizer_family). Q&A prompt formatting will be DISABLED for session generate.");
            } else {
        use_q_a_format_for_session_generate = true;
        Logger::info(
                    "GGUF (Non-Llama 3) model detected. Using Q:A: format via session->generate.");
            }
          } else {
        std::string system_prompt_text = "You are a helpful AI.";
            if (tokenizer) {
          prompt_for_session_generate = tokenizer->apply_chat_template(
              user_input_from_client, system_prompt_text, config);
          Logger::info(
              "Safetensors model detected. Applied chat template via "
              "tokenizer. Prompt: " +
              prompt_for_session_generate.substr(0, 200) + "...");
            } else {
          Logger::error(
              "CRITICAL: Tokenizer not available for Safetensors model in "
              "server. Cannot apply chat template.");

              prompt_for_session_generate = user_input_from_client; 
            }
        use_q_a_format_for_session_generate = false;
          }
          
          std::string reply = session->generate(
          prompt_for_session_generate, max_new_tokens, temperature, top_k, top_p, "",
          use_q_a_format_for_session_generate);
          // Dialogue models continue the whole transcript (both speakers):
          // cut the reply at the next turn marker so the UI shows one turn.
          {
              size_t cut = reply.find("user:");
              if (cut != std::string::npos && cut > 0) {
                  reply = reply.substr(0, cut);
              }
          }
          Logger::info("Generated reply: " + reply.substr(0, 50) + "...");

          json res_json;
          res_json["reply"] = ember::utf8_valid(reply);
          res_json["stop_reason"] = session->last_stop_reason();
          res_json["tokens"] = session->last_generated_tokens();

          // TTFT receipt (docs/TTFT.md). request_to_first_token_ms adds the
          // server's own parsing/templating before generate() to ttft_ms.
          const tinyllama::TtftBreakdown& tb = session->last_ttft();
          auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
          };
          res_json["ttft"] = {
              {"first_token", tb.first_token},
              {"queue_ms", queue_ms},
              {"request_to_first_token_ms",
               tb.first_token ? ms(t_request, tb.t_first_token) : -1.0},
              {"server_pre_generate_ms", ms(t_request, tb.t_entry)},
              {"ttft_ms", tb.ttft_ms},
              {"num_prompt_tokens", tb.num_prompt_tokens},
              {"batch_prefill", tb.batch_prefill},
              {"tokenize_ms", tb.tokenize_ms},
              {"kv_reset_ms", tb.kv_reset_ms},
              {"embed_ms", tb.embed_ms},
              {"prefill_wall_ms", tb.prefill_wall_ms},
              {"prefill_gpu_span_ms", tb.prefill_gpu_span_ms},
              {"sample_ms", tb.sample_ms},
              {"prefix_prev_cached_tokens", tb.prefix_prev_cached_tokens},
              {"prefix_reusable_tokens", tb.prefix_reusable_tokens},
              {"prefix_reused_tokens", tb.prefix_reused_tokens}, {"spec_drafted", tb.spec_drafted}, {"spec_accepted", tb.spec_accepted}, {"spec_passes", tb.spec_passes},
              {"prefix_hit_rate", tb.prefix_hit_rate},
              {"op_ms", tb.op_ms}};
          if (tb.spec_draft_calls > 0) {  // speculation on (EMBER_SPEC): the drafter's own time
            res_json["ttft"]["spec_draft_ms"] = tb.spec_draft_ms;
            res_json["ttft"]["spec_draft_calls"] = tb.spec_draft_calls;
          }

          res.set_content(res_json.dump(), "application/json");
          Logger::info("Response sent successfully.");

        } catch (const json::parse_error& e) {
          Logger::error("JSON parsing error: " + std::string(e.what()));
          res.status = 400;  
          json err_json;
          err_json["error"] = "Invalid JSON format: " + std::string(e.what());
          res.set_content(err_json.dump(), "application/json");
        } catch (const std::exception& e) {
          Logger::error("Generation error: " + std::string(e.what()));
          res.status = 500;  
          json err_json;
          err_json["error"] = "Internal server error: " + std::string(e.what());
          res.set_content(err_json.dump(), "application/json");
        }
      });
  
  // Liveness and what is being served (same path as mtstudio chat).
  svr.Get("/health", [&session, &model_dir](const httplib::Request&, httplib::Response& res) {
    const ModelConfig& c = session->get_config();
    json h;
    h["status"] = "ok";
    h["model"] = model_dir;
    h["tokenizer"] = session->is_bpe() ? "bpe" : "ember";
    // The chat template resolved at load (LOCAL_LLM_STACK.md section 3, point 4), null for raw
    // completion only; chat_template_why says how it was resolved, or why there is none.
    if (const microtorch::chat::Template* t = session->chat_template())
      h["chat_template"] = {{"name", t->name}, {"version", json::parse(t->version.dump())}};
    else
      h["chat_template"] = nullptr;
    h["chat_template_why"] = session->chat_template_why();
    h["context"] = c.max_position_embeddings;
    h["vocab_size"] = c.vocab_size;
    h["hidden_size"] = c.hidden_size;
    h["layers"] = c.num_hidden_layers;
    // EMBER_DETERMINISTIC=1 (determinism.h): reported only when on, so the default /health is unchanged.
    if (ember::deterministic()) h["deterministic"] = true;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(h.dump(), "application/json");
  });

  // ---- OpenAI-compatible API (chat-mode models) -------------------------
  // POST /v1/chat/completions and GET /v1/models, so standard clients (the
  // openai SDKs, chat front ends) can talk to ember. `messages` map one to
  // one onto the model's chat template (ember::openai_conversation): system
  // and developer messages before the first user message join with "\n\n"
  // into the system message, a later one or a tool message is a 400, user and
  // assistant messages alternate, and the last message must be from the user.
  // Defaults follow OpenAI's (temperature 1, top_p 1); top_k, thinking and
  // chat_template_kwargs.enable_thinking are accepted as extensions.
  const std::string model_id = std::filesystem::path(model_dir).filename().string();
  auto openai_error = [](httplib::Response& res, int status, const std::string& msg) {
    res.status = status;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(json{{"error", {{"message", msg}, {"type", "invalid_request_error"}}}}.dump(),
                    "application/json");
  };
  svr.Get("/v1/models", [model_id](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(json{{"object", "list"},
                         {"data", json::array({json{{"id", model_id},
                                                    {"object", "model"},
                                                    {"created", 0},
                                                    {"owned_by", "ember.cpp"}}})}}
                        .dump(),
                    "application/json");
  });
  svr.Options("/v1/chat/completions", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.status = 204;
  });
  svr.Post("/v1/chat/completions", [&session, model_id, openai_error](const httplib::Request& req,
                                                                     httplib::Response& res) {
    const auto t_request = std::chrono::steady_clock::now();
    Waiting waiting_guard;
    if (public_mode && waiting.load() > 8) {
      res.set_header("Retry-After", "2");
      openai_error(res, 503, "busy: too many requests at once, try again in a moment");
      return;
    }
    if (!session->is_bpe()) {
      openai_error(res, 400, "this model has no chat template ember can apply; use POST /chat");
      return;
    }
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded()) {
      openai_error(res, 400, "body must be JSON with a non-empty 'messages' array");
      return;
    }
    BpeChatRequest r;
    r.temperature = 1.0f;
    r.top_p = 1.0f;
    r.top_k = 0;
    r.max_new_tokens = 64;
    bool include_usage = false;
    bool stream = false;
    try {
      // messages map one to one (chat_request.h): leading system/developer messages join into the
      // system message, a later one or a tool message is refused, the last message is the user's.
      try {
        r.conv = ember::openai_conversation(body);
      } catch (const ember::RequestError& e) {
        openai_error(res, 400, e.what());
        return;
      }
      if (body.contains("temperature") && body["temperature"].is_number()) r.temperature = body["temperature"].get<float>();
      if (body.contains("top_p") && body["top_p"].is_number()) r.top_p = body["top_p"].get<float>();
      if (body.contains("top_k") && body["top_k"].is_number()) r.top_k = body["top_k"].get<int>();
      for (const char* key : {"max_completion_tokens", "max_tokens"})
        if (body.contains(key) && body[key].is_number()) {
          r.max_new_tokens = body[key].get<int>();
          break;
        }
      if (body.contains("seed") && body["seed"].is_number()) {
        r.has_seed = true;
        r.seed = static_cast<unsigned>(body["seed"].get<long long>());
      }
      stream = body.value("stream", false);
      if (body.contains("stream_options") && body["stream_options"].is_object())
        include_usage = body["stream_options"].value("include_usage", false);
    } catch (const json::exception& e) {
      openai_error(res, 400, std::string("bad request field: ") + e.what());
      return;
    }
    // OpenAI's frequency/presence penalties are not approximated: only 0 is
    // accepted. The contract's repeat_penalty / repeat_last_n are extensions.
    for (const char* key : {"frequency_penalty", "presence_penalty"})
      if (body.contains(key) && !body[key].is_null() && !(body[key].is_number() && body[key].get<double>() == 0.0)) {
        openai_error(res, 400, std::string(key) + " is not supported (only 0); use repeat_penalty and repeat_last_n");
        return;
      }
    {
      const std::string rp_error =
          bpe_repeat_fields(body, r, session->get_config().max_position_embeddings, public_mode);
      if (!rp_error.empty()) {
        openai_error(res, 400, rp_error);
        return;
      }
    }
    r.max_new_tokens = std::max(r.max_new_tokens, 1);
    if (public_mode) {
      constexpr size_t kMaxText = 600;
      if (r.conv.user.size() > kMaxText) {
        openai_error(res, 400, "message too long (600 characters at most)");
        return;
      }
      r.max_new_tokens = std::min(r.max_new_tokens, 64);
      r.temperature = std::min(std::max(r.temperature, 0.0f), 1.5f);
      r.top_k = std::min(std::max(r.top_k, 0), 200);
      r.top_p = std::min(std::max(r.top_p, 0.05f), 1.0f);
      if (r.conv.system.size() > kMaxText) r.conv.system.resize(kMaxText);
      while (r.conv.turns.size() > 8) r.conv.turns.erase(r.conv.turns.begin());
      for (auto& t : r.conv.turns) {
        if (t.first.size() > kMaxText) t.first.resize(kMaxText);
        if (t.second.size() > kMaxText) t.second.resize(kMaxText);
      }
    }
    std::lock_guard<std::mutex> session_lock(session_mu);
    const double queue_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_request).count();
    tinyllama::TinyLlamaSession* sess = session.get();
    BpePrompt bp;
    try {
      bp = bpe_build_prompt(*sess, r);
    } catch (const microtorch::chat::Error& e) {
      openai_error(res, 400, e.what());  // the template refuses the conversation
      return;
    }
    if (!bp.error.empty()) {
      openai_error(res, 400, bp.error);
      return;
    }
    static std::mt19937_64 id_rng{std::random_device{}()};
    char idbuf[32];
    std::snprintf(idbuf, sizeof idbuf, "chatcmpl-%016llx", static_cast<unsigned long long>(id_rng()));
    const std::string id = idbuf;
    const long long created = static_cast<long long>(std::time(nullptr));
    res.set_header("Access-Control-Allow-Origin", "*");
    auto usage_of = [&bp](const json& result) {
      const int completion = result.value("tokens", 0);
      return json{{"prompt_tokens", bp.prompt_tokens},
                  {"completion_tokens", completion},
                  {"total_tokens", bp.prompt_tokens + completion}};
    };
    auto finish_of = [](const json& result) {
      return result.value("stop_reason", std::string("eos")) == "length" ? "length" : "stop";
    };
    if (!stream) {
      const json result = bpe_run(*sess, r, bp, t_request, queue_ms, nullptr);
      json out = {{"id", id},
                  {"object", "chat.completion"},
                  {"created", created},
                  {"model", model_id},
                  {"choices", json::array({json{{"index", 0},
                                                {"message", {{"role", "assistant"}, {"content", result["reply"]}}},
                                                {"logprobs", nullptr},
                                                {"finish_reason", finish_of(result)}}})},
                  {"usage", usage_of(result)}};
      res.set_content(out.dump(), "application/json");
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [r, bp, t_request, sess, id, created, model_id, include_usage, usage_of, finish_of](
            size_t, httplib::DataSink& sink) {
          Waiting stream_waiting;
          std::lock_guard<std::mutex> stream_lock(session_mu);
          const double waited =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_request).count();
          auto chunk = [&](const json& delta, const json& finish) {
            json c = {{"id", id},
                      {"object", "chat.completion.chunk"},
                      {"created", created},
                      {"model", model_id},
                      {"choices", json::array({json{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
            const std::string ev = "data: " + c.dump() + "\n\n";
            sink.write(ev.data(), ev.size());
          };
          chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr);
          const json result = bpe_run(*sess, r, bp, t_request, waited,
                                      [&](const std::string& piece) { chunk(json{{"content", piece}}, nullptr); });
          chunk(json::object(), finish_of(result));
          if (include_usage) {
            json u = {{"id", id},
                      {"object", "chat.completion.chunk"},
                      {"created", created},
                      {"model", model_id},
                      {"choices", json::array()},
                      {"usage", usage_of(result)}};
            const std::string ev = "data: " + u.dump() + "\n\n";
            sink.write(ev.data(), ev.size());
          }
          const std::string done = "data: [DONE]\n\n";
          sink.write(done.data(), done.size());
          sink.done();
          return true;
        });
  });

  // ---- POST /score: log-probabilities of continuations (score.h) ----------
  // mtstudio score's semantics on token ids, for log-prob benchmarks: per request, one fresh forward
  // over prompt + continuation[:-1] from position 0, the continuation's log-probabilities (log-softmax
  // at temperature 1 in double), the argmax at each of its positions (mtstudio's banned specials left
  // out) and their sum; over_ctx past ctx + 1 ids. A batch holds the session from its first request
  // to its last. Off under EMBER_PUBLIC=1 (the open-internet settings serve chat only).
  svr.Post("/score", [&session](const httplib::Request& req, httplib::Response& res) {
    const auto t_request = std::chrono::steady_clock::now();
    Waiting waiting_guard;
    res.set_header("Access-Control-Allow-Origin", "*");
    auto refuse = [&res](int status, const std::string& msg) {
      res.status = status;
      res.set_content(json{{"error", msg}}.dump(), "application/json");
    };
    if (public_mode) {
      refuse(403, "/score is off on a public server (EMBER_PUBLIC=1)");
      return;
    }
    const json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded()) {
      refuse(400, "body must be JSON: {\"requests\": [{\"key\", \"prompt_ids\", \"continuation_ids\"}, ...]}");
      return;
    }
    ember::ScoreModel m;
    m.vocab = static_cast<size_t>(session->score_vocab());
    m.ctx = static_cast<size_t>(session->score_context());
    if (session->state_k() > 0 && session->chat_format()) {
      const ember::ChatFormat& f = *session->chat_format();
      m.state_k = static_cast<size_t>(session->state_k());
      m.eot = f.eos_ids.at(0);
      m.user = f.turn_ids.at(0);
      m.assistant = f.turn_ids.at(1);
    }
    std::vector<ember::ScoreItem> items;
    const std::string err = ember::parse_score_body(body, m, items);
    if (!err.empty()) {
      refuse(400, err);
      return;
    }
    static const std::vector<char> banned = ember::score_banned(session->gguf_tokens(), m.vocab);
    std::lock_guard<std::mutex> session_lock(session_mu);
    const auto t_start = std::chrono::steady_clock::now();
    json results = json::array();
    size_t scored = 0, over = 0, positions = 0, reused_total = 0;
    try {
      bool first = true;
      for (ember::ScoreItem& it : items) {
        if (it.over_ctx) {
          it.rec["over_ctx"] = true;
          results.push_back(std::move(it.rec));
          ++over;
          continue;
        }
        std::vector<int> ctx = it.prompt;  // prompt + continuation[:-1]
        ctx.insert(ctx.end(), it.cont.begin(), it.cont.end() - 1);
        int reused = 0;
        const std::vector<float> rows =
            session->score_rows(ctx, static_cast<int>(it.prompt.size()) - 1, !first, &reused);
        first = false;
        ember::score_fill(it.rec, it.cont, rows.data(), m.vocab, banned);
        results.push_back(std::move(it.rec));
        ++scored;
        positions += ctx.size();
        reused_total += static_cast<size_t>(reused);
      }
    } catch (const std::invalid_argument& e) {
      refuse(400, std::string("score: ") + e.what());
      return;
    } catch (const std::exception& e) {
      refuse(500, std::string("score: ") + e.what());
      return;
    }
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t_end = std::chrono::steady_clock::now();
    json out = {{"results", std::move(results)},
                {"stats", {{"requests", items.size()},
                           {"scored", scored},
                           {"over_ctx", over},
                           {"positions", positions},
                           {"reused_positions", reused_total},
                           {"queue_ms", ms(t_request, t_start)},
                           {"score_ms", ms(t_start, t_end)}}}};
    res.set_content(out.dump(), "application/json");
  });
  svr.Options("/score", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.status = 204;
  });

  svr.Options("/chat", [](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.status = 204;  
  });

  unsigned int num_threads = SAFE_MAX(1u, std::thread::hardware_concurrency() / 2);
  Logger::info("Starting server on " + host + ":" + std::to_string(port) +
               " with " + std::to_string(num_threads) + " threads.");


  svr.listen(host.c_str(), port);
  
  Logger::info("Server stopped.");
  return 0;
}