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
#include <chrono>
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
  svr.Post("/chat", [&session](const httplib::Request& req,
                               httplib::Response& res) {
        // TTFT clock starts here: the earliest point the server owns the request.
        const auto t_request = std::chrono::steady_clock::now();
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

          // paperkiln byte-level BPE chat models: the same /chat contract as
          // mtstudio chat. The prompt is exactly the SFT template: each past
          // turn <|user|>U<|assistant|>A<|endoftext|>, then
          // <|user|>U<|assistant|> (no BOS, no separators), with the oldest
          // turns dropped until prompt + max_new_tokens fits the context.
          if (session->is_bpe()) {
            std::vector<std::pair<std::string, std::string>> turns;
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
                    turns.emplace_back(pending_user, content);
                    have_user = false;
                  }
                } else {                    // [{"user": ..., "assistant": ...}]
                  turns.emplace_back(h.value("user", ""), h.value("assistant", ""));
                }
              }
            }
            const std::string tail = "<|user|>" + user_input_from_client + "<|assistant|>";
            auto build = [&](size_t from) {
              std::string p;
              for (size_t i = from; i < turns.size(); ++i)
                p += "<|user|>" + turns[i].first + "<|assistant|>" + turns[i].second + "<|endoftext|>";
              return p + tail;
            };
            const int ctx = session->get_config().max_position_embeddings;
            size_t from = 0;
            std::string prompt = build(from);
            while (from < turns.size() && session->bpe_token_count(prompt) + max_new_tokens > ctx)
              prompt = build(++from);
            const int prompt_tokens = session->bpe_token_count(prompt);
            if (prompt_tokens >= ctx) {
              res.status = 400;
              json err;
              err["error"] = "message is " + std::to_string(prompt_tokens) + " tokens; the model's context is " +
                             std::to_string(ctx);
              res.set_content(err.dump(), "application/json");
              return;
            }
            const bool has_seed = req_json.contains("seed");
            const unsigned seed = has_seed ? req_json["seed"].get<unsigned>() : 0u;
            const bool want_ids = req_json.value("return_ids", false);
            const int turns_used = static_cast<int>(turns.size() - from);
            tinyllama::TinyLlamaSession* sess = session.get();
            // Generate under the session lock (held by the caller) and build
            // the reply object; cb, if set, sees each token as it is made.
            auto run = [=](std::function<void(int)> cb, double waited_ms) {
              if (has_seed) sess->set_seed(seed);
              sess->set_token_callback(std::move(cb));
              const std::string reply = sess->generate(prompt, max_new_tokens, temperature, top_k, top_p, "", false);
              sess->set_token_callback(nullptr);
              const tinyllama::TtftBreakdown& tb = sess->last_ttft();
              auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
              json res_json;
              res_json["reply"] = reply;
              res_json["stop_reason"] = sess->last_stop_reason();
              res_json["tokens"] = sess->last_generated_tokens();
              res_json["prompt_tokens"] = prompt_tokens;
              res_json["history_turns_used"] = turns_used;
              // Opt-in exact ids (preference data needs the sampled ids, not a
              // re-encoding of the stripped reply text).
              if (want_ids) {
                res_json["prompt_ids"] = sess->last_prompt_ids();
                res_json["ids"] = sess->last_generated_ids();
                res_json["stop_id"] = sess->last_stop_id();
              }
              res_json["ttft"] = {
                  {"first_token", tb.first_token},
                  {"queue_ms", waited_ms},
                  {"request_to_first_token_ms", tb.first_token ? ms(t_request, tb.t_first_token) : -1.0},
                  {"ttft_ms", tb.ttft_ms},
                  {"num_prompt_tokens", tb.num_prompt_tokens},
                  {"batch_prefill", tb.batch_prefill},
                  {"prefill_wall_ms", tb.prefill_wall_ms},
                  {"prefill_gpu_span_ms", tb.prefill_gpu_span_ms}};
              return res_json;
            };
            if (!req_json.value("stream", false)) {
              res.set_content(run(nullptr, queue_ms).dump(), "application/json");
              return;
            }
            // "stream": true -> server-sent events: {"token": text} as the
            // reply is generated (leading whitespace skipped, never a split
            // UTF-8 character), then the full reply object with "done": true.
            // The provider runs after this handler returns, so it takes the
            // session lock itself.
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider(
                "text/event-stream", [run, t_request, sess](size_t, httplib::DataSink& sink) {
                  std::lock_guard<std::mutex> stream_lock(session_mu);
                  const double waited = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t_request).count();
                  std::vector<int> ids;
                  size_t sent = 0;
                  auto send = [&sink](const json& j) {
                    const std::string ev = "data: " + j.dump() + "\n\n";
                    sink.write(ev.data(), ev.size());
                  };
                  json final_json = run(
                      [&](int id) {
                        ids.push_back(id);
                        const std::string full = sess->bpe_decode(ids);
                        const size_t end = utf8_complete_prefix(full);
                        const size_t start = full.find_first_not_of(" \t\n\v\f\r");
                        if (start == std::string::npos || start >= end) return;
                        if (sent < start) sent = start;
                        if (end > sent) {
                          send(json{{"token", full.substr(sent, end - sent)}});
                          sent = end;
                        }
                      },
                      waited);
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