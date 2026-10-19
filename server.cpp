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
#include "logger.h"
#include "tokenizer.h"
#include "model_macros.h"

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
// paperkiln byte-level BPE chat models: the prompt is exactly the SFT
// template, each past turn <|user|>U<|assistant|>A<|endoftext|> then
// <|user|>U<|assistant|> (no BOS, no separators), with the oldest turns
// dropped until prompt + max_new_tokens fits the context.
struct BpeChatRequest {
  std::string user_input;
  std::vector<std::pair<std::string, std::string>> turns;  // (user, assistant), oldest first
  int max_new_tokens = 60;
  float temperature = 0.1f;
  int top_k = 40;
  float top_p = 0.9f;
  bool has_seed = false;
  unsigned seed = 0;
  bool want_ids = false;
};

struct BpePrompt {
  std::string prompt;
  int prompt_tokens = 0;
  int turns_used = 0;
  std::string error;  // non-empty: the newest message alone does not fit
};

static BpePrompt bpe_build_prompt(tinyllama::TinyLlamaSession& sess, const BpeChatRequest& r) {
  const std::string tail = "<|user|>" + r.user_input + "<|assistant|>";
  auto build = [&](size_t from) {
    std::string p;
    for (size_t i = from; i < r.turns.size(); ++i)
      p += "<|user|>" + r.turns[i].first + "<|assistant|>" + r.turns[i].second + "<|endoftext|>";
    return p + tail;
  };
  const int ctx = sess.get_config().max_position_embeddings;
  size_t from = 0;
  BpePrompt out;
  out.prompt = build(from);
  while (from < r.turns.size() && sess.bpe_token_count(out.prompt) + r.max_new_tokens > ctx)
    out.prompt = build(++from);
  out.prompt_tokens = sess.bpe_token_count(out.prompt);
  out.turns_used = static_cast<int>(r.turns.size() - from);
  if (out.prompt_tokens >= ctx)
    out.error = "message is " + std::to_string(out.prompt_tokens) + " tokens; the model's context is " +
                std::to_string(ctx);
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
        on_piece(full.substr(sent, end - sent));
        sent = end;
      }
    });
  }
  const std::string reply = sess.generate(p.prompt, r.max_new_tokens, r.temperature, r.top_k, r.top_p, "", false);
  sess.set_token_callback(nullptr);
  const tinyllama::TtftBreakdown& tb = sess.last_ttft();
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  json res_json;
  res_json["reply"] = reply;
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
                      {"batch_prefill", tb.batch_prefill},
                      {"prefill_wall_ms", tb.prefill_wall_ms},
                      {"prefill_gpu_span_ms", tb.prefill_gpu_span_ms}};
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
            if (req_json.contains("history") && req_json["history"].is_array()) {
              json& hist = req_json["history"];
              while (hist.size() > 16) hist.erase(hist.begin());  // the context holds ~5 turns anyway
              for (auto& h : hist)
                for (const char* key : {"content", "user", "assistant"})
                  if (h.is_object() && h.contains(key) && h[key].is_string() &&
                      h[key].get<std::string>().size() > kMaxText)
                    h[key] = h[key].get<std::string>().substr(0, kMaxText);
            }
          }

          // paperkiln byte-level BPE chat models: the same /chat contract as
          // mtstudio chat. The prompt is exactly the SFT template: each past
          // turn <|user|>U<|assistant|>A<|endoftext|>, then
          // <|user|>U<|assistant|> (no BOS, no separators), with the oldest
          // turns dropped until prompt + max_new_tokens fits the context.
          if (session->is_bpe()) {
            BpeChatRequest r;
            r.user_input = user_input_from_client;
            r.max_new_tokens = max_new_tokens;
            r.temperature = temperature;
            r.top_k = top_k;
            r.top_p = top_p;
            r.has_seed = req_json.contains("seed");
            r.seed = r.has_seed ? req_json["seed"].get<unsigned>() : 0u;
            r.want_ids = req_json.value("return_ids", false);
            if (req_json.contains("history") && req_json["history"].is_array()) {
              std::string pending_user;
              bool have_user = false;
              for (const auto& h : req_json["history"]) {
                if (h.contains("role")) {   // [{"role": "user"|"assistant", "content": ...}]
                  const std::string role = h.value("role", "");
                  const std::string content = h.value("content", "");
                  if (role == "user") {
                    pending_user = content;
                    have_user = true;
                  } else if (role == "assistant" && have_user) {
                    r.turns.emplace_back(pending_user, content);
                    have_user = false;
                  }
                } else {                    // [{"user": ..., "assistant": ...}]
                  r.turns.emplace_back(h.value("user", ""), h.value("assistant", ""));
                }
              }
            }
            tinyllama::TinyLlamaSession* sess = session.get();
            const BpePrompt bp = bpe_build_prompt(*sess, r);
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
                        prompt_for_session_generate =
                            "user: " + user_input_from_client + " assistant:";
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
          res_json["reply"] = reply;

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
              {"prefix_hit_rate", tb.prefix_hit_rate},
              {"op_ms", tb.op_ms}};

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
    h["context"] = c.max_position_embeddings;
    h["vocab_size"] = c.vocab_size;
    h["hidden_size"] = c.hidden_size;
    h["layers"] = c.num_hidden_layers;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(h.dump(), "application/json");
  });

  // ---- OpenAI-compatible API (paperkiln BPE chat models) ----------------
  // POST /v1/chat/completions and GET /v1/models, so standard clients (the
  // openai SDKs, chat front ends) can talk to ember. `messages` map onto the
  // chat template: the last message must be from the user; earlier
  // user/assistant pairs become history; system messages are ignored (the
  // models are not trained with one). Defaults follow OpenAI's (temperature
  // 1, top_p 1); top_k is accepted as an extension (0 = whole vocabulary).
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
    if (body.is_discarded() || !body.is_object() || !body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
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
      std::string pending_user;
      bool have_user = false;
      const json& msgs = body["messages"];
      for (size_t i = 0; i < msgs.size(); ++i) {
        const json& m = msgs[i];
        const std::string role = m.value("role", "");
        std::string content;
        if (m.contains("content") && m["content"].is_string()) {
          content = m["content"].get<std::string>();
        } else if (m.contains("content") && m["content"].is_array()) {  // [{"type": "text", "text": ...}]
          for (const auto& part : m["content"])
            if (part.is_object() && part.value("type", "") == "text") content += part.value("text", "");
        }
        const bool last = i + 1 == msgs.size();
        if (role == "user") {
          if (last) {
            r.user_input = content;
          } else {
            pending_user = content;
            have_user = true;
          }
        } else if (role == "assistant") {
          if (last) {
            openai_error(res, 400, "the last message must be from the user");
            return;
          }
          if (have_user) r.turns.emplace_back(pending_user, content);
          have_user = false;
        }
        // "system", "developer", "tool": ignored.
      }
      if (msgs.back().value("role", "") != "user") {
        openai_error(res, 400, "the last message must be from the user");
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
    r.max_new_tokens = std::max(r.max_new_tokens, 1);
    if (public_mode) {
      constexpr size_t kMaxText = 600;
      if (r.user_input.size() > kMaxText) {
        openai_error(res, 400, "message too long (600 characters at most)");
        return;
      }
      r.max_new_tokens = std::min(r.max_new_tokens, 64);
      r.temperature = std::min(std::max(r.temperature, 0.0f), 1.5f);
      r.top_k = std::min(std::max(r.top_k, 0), 200);
      r.top_p = std::min(std::max(r.top_p, 0.05f), 1.0f);
      while (r.turns.size() > 8) r.turns.erase(r.turns.begin());
      for (auto& t : r.turns) {
        if (t.first.size() > kMaxText) t.first.resize(kMaxText);
        if (t.second.size() > kMaxText) t.second.resize(kMaxText);
      }
    }
    std::lock_guard<std::mutex> session_lock(session_mu);
    const double queue_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_request).count();
    tinyllama::TinyLlamaSession* sess = session.get();
    const BpePrompt bp = bpe_build_prompt(*sess, r);
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