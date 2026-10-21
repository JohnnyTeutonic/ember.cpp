#include "api.h"
#include "gguf_parser.h"
#include "model_macros.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>

#include "graph_ir.h"
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>  // header-only; names the prefill range for nsys (docs/TTFT.md)
#endif
#include <sstream>

#include "logger.h"
#include "model.h"
#include "safetensors_loader.h"
#include "tokenizer.h"

namespace tinyllama {


static void log_vector_summary_detailed(const std::string& name,
                                        const std::vector<float>& v,
                                        int current_pos, int current_layer,
                                        int N) {
  if (v.empty()) {
    Logger::info(name + " (pos=" + std::to_string(current_pos) + ", layer=" +
                 std::to_string(current_layer) + "): EMPTY VECTOR");
    return;
  }
  std::stringstream ss;
  ss << name << " (pos=" << std::to_string(current_pos)
     << ", layer=" << std::to_string(current_layer) << "): size=" << v.size();
  ss << ", first " << N << ": [";
  for (int i = 0; i < N && i < v.size(); ++i) {
    ss << std::fixed << std::setprecision(4) << v[i]
       << (i == N - 1 || i == v.size() - 1 ? "" : ", ");
  }
  ss << "]";
  float min_val = v[0], max_val = v[0], sum = 0.0f;
  bool all_finite = true;
  for (float val : v) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum += val;
    if (!std::isfinite(val)) all_finite = false;
  }
  ss << ", min=" << std::fixed << std::setprecision(4) << min_val;
  ss << ", max=" << std::fixed << std::setprecision(4) << max_val;
  ss << ", mean=" << std::fixed << std::setprecision(4) << (sum / v.size());
  ss << ", finite=" << (all_finite ? "yes" : "no");
  Logger::info(ss.str());
}

static std::string read_file_api(const std::string& path) {
  std::filesystem::path fs_path(path);
  std::ifstream file(fs_path, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open file: " + path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

static int argmax(const std::vector<float>& v) {
  if (v.empty()) {
    Logger::error("Cannot perform argmax on empty vector");
    return -1;
  }

  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// Prompt-lookup drafting for speculative decoding: the most recent earlier
// occurrence of the last 3 (then 2) tokens of the history, and up to k of
// the tokens that followed it. Empty if there is no such match.
static std::vector<int> prompt_lookup_draft(const std::vector<int>& h, int k) {
  const int n = static_cast<int>(h.size());
  for (int g = std::min(3, n - 1); g >= 2; --g) {
    for (int s = n - g - 1; s >= 0; --s) {
      if (std::equal(h.begin() + s, h.begin() + s + g, h.end() - g))
        return std::vector<int>(h.begin() + s + g, h.begin() + std::min(n, s + g + k));
    }
  }
  return {};
}

// CTRL-style repetition penalty over recently generated tokens. Tiny
// word-level models fall into short degenerate cycles ("you are very kind."
// x12, 2026-07-19); penalizing the last window of generated ids breaks the
// cycle while leaving fresh vocabulary untouched.
static void apply_repetition_penalty(std::vector<float>& logits,
                                     const std::vector<int>& generated,
                                     float penalty, size_t window = 64) {
  if (penalty <= 1.0f || generated.empty()) return;
  size_t start = generated.size() > window ? generated.size() - window : 0;
  for (size_t i = start; i < generated.size(); ++i) {
    int id = generated[i];
    if (id < 0 || static_cast<size_t>(id) >= logits.size()) continue;
    float& l = logits[id];
    l = (l > 0.0f) ? l / penalty : l * penalty;
  }
}

// TINYLLAMA_TOPK_DEBUG=1: dump the top-10 candidates (id, logit) at every
// sampled position to stderr. The decoder-side counterpart of the trainer's
// [TRACE] dumps — for asking "was the dropped word 2nd by a hair, or absent?"
static void debug_dump_topk(const std::vector<float>& logits) {
  // TINYLLAMA_LOGITS_DUMP=<path>: append the FULL logit vector at every
  // sampled position (one space-joined line each). The engine half of the
  // golden-batch GGUF-parity test — compare against the trainer's
  // golden_batch_test --dump rows with compare_golden_logits.py.
  static const char* dump_path = std::getenv("TINYLLAMA_LOGITS_DUMP");
  if (dump_path) {
    std::ofstream f(dump_path, std::ios::app);
    for (size_t i = 0; i < logits.size(); ++i) f << (i ? " " : "") << logits[i];
    f << "\n";
  }
  static const bool on = std::getenv("TINYLLAMA_TOPK_DEBUG") != nullptr;
  if (!on) return;
  std::vector<int> idx(logits.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(10, idx.size()),
                    idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  std::ostringstream os;
  os << "[TOPK]";
  for (size_t r = 0; r < std::min<size_t>(10, idx.size()); ++r)
    os << " " << idx[r] << ":" << logits[idx[r]];
  fprintf(stderr, "%s\n", os.str().c_str());
}

static int sample_top_k_top_p_temperature(const std::vector<float>& logits,
                                          float temperature, int top_k,
                                          float top_p, std::mt19937& rng) {
  if (logits.empty()) {
    throw std::runtime_error("Cannot sample from empty logits.");
  }

  // If temperature is very low, fall back to greedy sampling
  if (temperature < 0.05f) {
    return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
  }

  int vocab_size = logits.size();

  top_k = (std::min)(top_k, vocab_size);
  if (top_k <= 0) top_k = vocab_size;

  std::vector<float> scaled_logits(vocab_size);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float logit : logits) max_logit = (std::max)(max_logit, logit);

  // Scale logits to avoid numerical instability
  const float scale = 1.0f / temperature;
  for (int i = 0; i < vocab_size; ++i) {
    scaled_logits[i] = (logits[i] - max_logit) * scale;
  }

  std::vector<double> probs_double(vocab_size);
  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    probs_double[i] = std::exp(static_cast<double>(scaled_logits[i]));
    sum_exp += probs_double[i];
  }

  // Pure temperature (top_k <= 0 or >= vocab, top_p >= 1): draw from the
  // whole double-precision distribution, no sort and no float cumulative sum
  // that can reach 1.0 early and drop the tail. Same distribution as
  // paperkiln's sample_pure (charter 0001 §4); the RNG stream differs.
  if (top_k >= vocab_size && top_p >= 1.0f && sum_exp > 0.0) {
    std::discrete_distribution<int> pure(probs_double.begin(), probs_double.end());
    return pure(rng);
  }

  // Normalize probabilities
  if (sum_exp > 0.0) {
    for (int i = 0; i < vocab_size; ++i) {
      probs_double[i] /= sum_exp;
    }
  } else {
    // If all probabilities are zero, fall back to uniform distribution
    for (int i = 0; i < vocab_size; ++i) {
      probs_double[i] = 1.0 / vocab_size;
    }
  }

  std::vector<std::pair<float, int>> prob_idx(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    prob_idx[i] = {static_cast<float>(probs_double[i]), i};
  }

  // Only the top_k head is used, so sort just that head: the same elements
  // in the same order as a full sort (pairs are distinct, so the order is
  // total), without sorting the whole vocabulary on every token.
  if (top_k < vocab_size) {
    std::partial_sort(prob_idx.begin(), prob_idx.begin() + top_k, prob_idx.end(),
                      std::greater<std::pair<float, int>>());
    prob_idx.resize(top_k);
  } else {
    std::sort(prob_idx.begin(), prob_idx.end(), std::greater<std::pair<float, int>>());
  }

  float cumulative_prob = 0.0f;
  int last_idx = 0;
  for (int i = 0; i < prob_idx.size(); ++i) {
    cumulative_prob += prob_idx[i].first;
    last_idx = i;
    if (cumulative_prob >= top_p) {
      break;
    }
  }
  prob_idx.resize(last_idx + 1);

  float final_sum = 0.0f;
  for (const auto& pi : prob_idx) {
    final_sum += pi.first;
  }

  // Renormalize probabilities after top-k and top-p filtering
  std::vector<float> final_probs(prob_idx.size());
  if (final_sum > 0.0f) {
    for (size_t i = 0; i < prob_idx.size(); ++i) {
      final_probs[i] = prob_idx[i].first / final_sum;
    }
  } else {
    // If all probabilities are zero after filtering, use uniform distribution
    float uniform_prob = 1.0f / prob_idx.size();
    std::fill(final_probs.begin(), final_probs.end(), uniform_prob);
  }

  std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
  int sampled_idx_in_filtered = dist(rng);

  return prob_idx[sampled_idx_in_filtered].second;
}


TinyLlamaSession::TinyLlamaSession(const std::string& model_path_arg,
                                 const std::string& tokenizer_path_arg,
                                 int threads,
                                 int num_gpu_layers_from_cli,
                                 bool cli_use_mmap,
                                 bool use_kv_quant,
                                 bool use_batch_generation,
                                 int max_batch_size)
          : threads_(threads), use_batch_generation_(use_batch_generation), 
            max_batch_size_(max_batch_size), rng_(std::random_device{}()) {
  Logger::info("TinyLlamaSession constructor entered. Model path: " + model_path_arg +
               ", Tokenizer path: " + tokenizer_path_arg +
               ", Threads: " + std::to_string(threads) +
               ", Num GPU Layers (CLI): " + std::to_string(num_gpu_layers_from_cli) +
               ", Use MMAP (CLI): " + (cli_use_mmap ? "true" : "false") +
               ", Use KV Quant (CLI): " + (use_kv_quant ? "true" : "false"));

  std::string effective_model_file_path = model_path_arg; 
  std::string path_for_config_json = model_path_arg;

  ModelConfig initial_model_config_for_model_ctor;
  initial_model_config_for_model_ctor.use_mmap_for_gguf = cli_use_mmap;
  initial_model_config_for_model_ctor.use_kvcache_quantization = use_kv_quant;
  if (num_gpu_layers_from_cli < 0) { 
      initial_model_config_for_model_ctor.num_cpu_offload_layers = 0;
  } else {
      initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli;
  }

  std::filesystem::path fs_model_path(model_path_arg);
  bool is_dir = std::filesystem::is_directory(fs_model_path);

  if (is_dir) {
    Logger::info("Model path is a directory. Assuming SafeTensors model directory: " + model_path_arg);
    effective_model_file_path = (fs_model_path / "model.safetensors").string();
    std::string config_json_path_in_dir = (fs_model_path / "config.json").string(); 

    Logger::info("Derived SafeTensors model file path: " + effective_model_file_path);
    Logger::info("Path for loading config.json: " + config_json_path_in_dir);

    // Directly populate initial_model_config_for_model_ctor
    // load_model_config_from_json returns bool and populates the passed ModelConfig&
    bool st_config_loaded = SafeTensorsLoader::load_model_config_from_json(config_json_path_in_dir, initial_model_config_for_model_ctor);
    
    if (st_config_loaded) {
        Logger::info("Successfully loaded config.json directly into initial_model_config_for_model_ctor.");
        // Log tokenizer_family IMMEDIATELY after loading from config.json
        std::string family_after_json_load = "UNKNOWN_POST_JSON_LOAD_DIR_CASE";
        if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_after_json_load = "LLAMA_SENTENCEPIECE";
        else if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_after_json_load = "LLAMA3_TIKTOKEN";
        Logger::info("[API_CPP_POST_JSON_LOAD_DIR_CASE] Tokenizer family in initial_model_config_for_model_ctor: " + family_after_json_load);
        // tokenizer_family should now be set in initial_model_config_for_model_ctor
        // num_hidden_layers is also now set from config.json
    } else {
        Logger::warning("Failed to load config.json for SafeTensors. initial_model_config_for_model_ctor will have defaults/CLI overrides for some fields, tokenizer_family likely UNKNOWN.");
        // If config.json fails, num_hidden_layers might be 0 or default, which will affect cpu offload calculation.
        // It's crucial config.json loads for correct layer counts.
    }
    
    // Apply CLI overrides for mmap and GPU layers. GPU layer logic depends on total_hidden_layers from config.
    initial_model_config_for_model_ctor.use_mmap_for_gguf = cli_use_mmap; 

    int total_layers_from_config = initial_model_config_for_model_ctor.num_hidden_layers;
    if (total_layers_from_config <= 0 && st_config_loaded) {
        Logger::warning("config.json loaded but num_hidden_layers is <= 0. GPU offload logic might be incorrect.");
    } else if (total_layers_from_config <= 0 && !st_config_loaded) {
        Logger::warning("config.json NOT loaded and num_hidden_layers is <= 0 (default). GPU offload may not work as expected. Model load will likely fail.");
        // If config.json didn't load, total_layers_from_config is likely 0 from default ModelConfig.
        // The TinyLlamaModel constructor will ultimately use its own parsed config, but this intermediate step needs care.
    }

    if (num_gpu_layers_from_cli < 0) { // -1 signifies all layers on GPU
        initial_model_config_for_model_ctor.num_cpu_offload_layers = 0;
    } else if (num_gpu_layers_from_cli == 0) { // 0 signifies all layers on CPU
        initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config; // All layers offloaded to CPU
    } else { // N > 0 signifies N layers on GPU
        if (total_layers_from_config > 0) {
            initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config - num_gpu_layers_from_cli;
        } else {
            // Cannot determine actual GPU layer count if total_layers_from_config is unknown.
            // Pass the CLI hint, TinyLlamaModel ctor will deal with it against its own parsed config.
            initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli; 
            Logger::warning("Total hidden layers unknown from config.json before model load; passing num_gpu_layers_from_cli as num_cpu_offload_layers hint.");
        }
    }
    // Clamp num_cpu_offload_layers
    if (total_layers_from_config > 0) {
         initial_model_config_for_model_ctor.num_cpu_offload_layers = std::max(0, std::min(initial_model_config_for_model_ctor.num_cpu_offload_layers, total_layers_from_config));
    }

    initial_model_config_for_model_ctor.is_gguf_file_loaded = false;

    SafeTensorsLoader st_loader(effective_model_file_path); 
    model_ = std::make_unique<TinyLlamaModel>(initial_model_config_for_model_ctor, st_loader);
    config_ = model_->get_config(); 
    config_.is_gguf_file_loaded = false; // Ensure this is set for session's copy too

    config_.use_kvcache_quantization = use_kv_quant; // Re-apply CLI/constructor preference

    Logger::info("TinyLlamaSession: Finalizing ModelConfig for KVCache initialization. use_kvcache_quantization set to: " + 
                  std::string(config_.use_kvcache_quantization ? "true" : "false"));

    kv_cache_.initialize(config_, config_.num_hidden_layers, 
                         config_.num_hidden_layers - config_.num_cpu_offload_layers, 
                         config_.max_position_embeddings, 
                         config_.num_key_value_heads, 
                         config_.hidden_size / config_.num_attention_heads,
                         max_batch_size_);

  } else { // Not a directory, assume it's a file path and check extension
    std::string extension = fs_model_path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".gguf") {
      Logger::info("GGUF model type detected by extension for Session constructor: " + model_path_arg);
      // The GGUF constructor decodes num_cpu_offload_layers as the RAW
      // GPU-layer hint (<0 all GPU, 0 all CPU, >0 that many GPU layers).
      // The translation above turned -1 into 0, which that decoder reads as
      // "all CPU": every default GGUF run was CPU-only (docs/TTFT.md).
      initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli;
      model_ = std::make_unique<TinyLlamaModel>(initial_model_config_for_model_ctor, model_path_arg);
      config_ = model_->get_config(); 
    } else if (extension == ".safetensors") {
      Logger::info("SafeTensors model type detected by extension for Session constructor (file case): " + model_path_arg);
      effective_model_file_path = model_path_arg;
      
      bool st_config_loaded = SafeTensorsLoader::load_model_config_from_json(effective_model_file_path, initial_model_config_for_model_ctor);
      if (st_config_loaded) {
          Logger::info("Successfully loaded config.json for SafeTensors in Session ctor (file case).");
          // Log tokenizer_family IMMEDIATELY after loading from config.json (file case)
          std::string family_after_json_load_file_case = "UNKNOWN_POST_JSON_LOAD_FILE_CASE";
          if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_after_json_load_file_case = "LLAMA_SENTENCEPIECE";
          else if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_after_json_load_file_case = "LLAMA3_TIKTOKEN";
          Logger::info("[API_CPP_POST_JSON_LOAD_FILE_CASE] Tokenizer family in initial_model_config_for_model_ctor: " + family_after_json_load_file_case);
          // tokenizer_family and num_hidden_layers are now set in initial_model_config_for_model_ctor
      } else {
          Logger::warning("Failed to load config.json for SafeTensors in Session ctor (file case). Model will use defaults or CLI overrides.");
      }
      
      initial_model_config_for_model_ctor.use_mmap_for_gguf = cli_use_mmap;
      // Correctly calculate num_cpu_offload_layers based on total_layers_from_config
      int total_layers_from_config_file_case = initial_model_config_for_model_ctor.num_hidden_layers;
      if (num_gpu_layers_from_cli < 0) { 
          initial_model_config_for_model_ctor.num_cpu_offload_layers = 0;
      } else if (num_gpu_layers_from_cli == 0) { 
          initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config_file_case;
      } else { 
          if (total_layers_from_config_file_case > 0) {
            initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config_file_case - num_gpu_layers_from_cli;
          } else {
            initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli; 
            Logger::warning("Total hidden layers unknown from config.json (file case); passing num_gpu_layers_from_cli as num_cpu_offload_layers hint.");
          }
      }
      if (total_layers_from_config_file_case > 0) {
        initial_model_config_for_model_ctor.num_cpu_offload_layers = std::max(0, std::min(initial_model_config_for_model_ctor.num_cpu_offload_layers, total_layers_from_config_file_case));
      }

      initial_model_config_for_model_ctor.is_gguf_file_loaded = false;

      SafeTensorsLoader st_loader(effective_model_file_path);
      model_ = std::make_unique<TinyLlamaModel>(initial_model_config_for_model_ctor, st_loader);
      config_ = model_->get_config(); 
      config_.is_gguf_file_loaded = false; // Ensure this is set for session's copy too

      config_.use_kvcache_quantization = use_kv_quant; // Re-apply CLI/constructor preference

      Logger::info("TinyLlamaSession: Finalizing ModelConfig for KVCache initialization. use_kvcache_quantization set to: " + 
                    std::string(config_.use_kvcache_quantization ? "true" : "false"));

      // Initialize KVCache with potentially updated config_ (from model load) 
      // and the now-set use_kvcache_quantization flag.
      kv_cache_.initialize(config_, config_.num_hidden_layers, 
                           config_.num_hidden_layers - config_.num_cpu_offload_layers, 
                           config_.max_position_embeddings, 
                           config_.num_key_value_heads, 
                           config_.hidden_size / config_.num_attention_heads,
                           max_batch_size_);
    } else {
      throw std::runtime_error("Unsupported model file type or extension in Session constructor: " + model_path_arg +
                               ". Please provide a directory for SafeTensors, a .gguf file, or a .safetensors file.");
    }
  }

  if (!model_) {
      throw std::runtime_error("Model pointer is null after instantiation attempt in Session constructor.");
  }

  try {
    if (config_.is_gguf_file_loaded) {
      const GGUFData* gguf_data = model_->get_gguf_data();
      if (!gguf_data) {
          throw std::runtime_error("GGUF model loaded but GGUFData is null in Session constructor.");
      }
      tokenizer_ = std::make_unique<Tokenizer>(*gguf_data, config_);
      Logger::info("Tokenizer initialized from GGUF metadata.");
    } else { // SafeTensors (either from directory or direct .safetensors file)
      std::filesystem::path p_tokenizer_arg(tokenizer_path_arg);
      std::string tokenizer_dir = p_tokenizer_arg.parent_path().string();
      if (tokenizer_dir.empty()) { 
          tokenizer_dir = "."; 
      }
      
      std::string vocab_json_path = (std::filesystem::path(tokenizer_dir) / "tokenizer.json").string();
      // The model_path for the tokenizer constructor should be the actual sentencepiece model file, e.g. data/tokenizer.model
      // tokenizer_path_arg already holds this (e.g. data/tokenizer.model)
      std::string sp_model_path = tokenizer_path_arg; 

      Logger::info("Initializing Tokenizer for SafeTensors. Vocab JSON path: " + vocab_json_path + ", SP Model path: " + sp_model_path);
      // Log the tokenizer_family from the config_ that will be passed to the Tokenizer
      std::string family_to_log = "UNKNOWN_IN_API_CPP";
      if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_to_log = "LLAMA_SENTENCEPIECE";
      else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_to_log = "LLAMA3_TIKTOKEN";
      Logger::info("[API_CPP_TOKENIZER_INIT] Tokenizer family from session config for SafeTensors: " + family_to_log);

      tokenizer_ = std::make_unique<Tokenizer>(vocab_json_path, sp_model_path, config_); 
      Logger::info("Tokenizer initialized from external files for SafeTensors model.");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to initialize Tokenizer: ") + e.what());
  }
  
  if (!tokenizer_) {
      throw std::runtime_error("Tokenizer pointer is null after instantiation attempt.");
  }

  eos_token_id_ = config_.eos_token_id; // Use the session's config_ which is now model's config
  // paperkiln IR (LOCAL_LLM_STACK.md section 4): EMBER_IR=path runs the
  // interpreter; an IR embedded in the GGUF (`paperkiln.ir`) runs it when its
  // family is not llama (llama keeps the tuned path) or it declares sliding-
  // window attention (only the interpreter implements it). EMBER_INTERPRETER=1/0
  // forces it on or off.
  if (const GGUFData* gd = model_->get_gguf_data()) {
    const std::string ir = ember_ir::find_ir(*gd);
    const char* force = std::getenv("EMBER_INTERPRETER");
    if (model_->ir_only() && ir.empty()) throw std::runtime_error("IR-only model but no IR found");
    if (model_->ir_only() && force && std::string(force) == "0")
      throw std::runtime_error("EMBER_INTERPRETER=0, but this GGUF can only run through the IR interpreter");
    if (!ir.empty() && !(force && std::string(force) == "0")) {
      const std::string family = nlohmann::json::parse(ir).value("family_hint", "");
      if ((force && std::string(force) == "1") || std::getenv("EMBER_IR") || family != "llama" || model_->ir_only() ||
          ember_ir::has_windowed_attention(ir)) {
        // CUDA builds run the interpreter on the device when the CLI asked for
        // GPU layers (any nonzero count: the interpreter does not split
        // layers); EMBER_IR_DEVICE=cpu keeps it on the CPU. Every op the
        // interpreter has (recurrent state, gating and attention-residual ops
        // and sliding-window attention included) runs on either; an IR past
        // the device kernels' limits stays on the CPU (ember_ir::device_supported).
        bool ir_gpu = false;
#ifdef HAS_CUDA
        const char* dev = std::getenv("EMBER_IR_DEVICE");
        ir_gpu = num_gpu_layers_from_cli != 0 && !(dev && std::string(dev) == "cpu") &&
                 ember_ir::device_supported(ir);
#endif
        ir_ = std::make_unique<ember_ir::Interpreter>(ir, *gd, ir_gpu);
        Logger::warning("Running the model through the paperkiln IR interpreter (" +
                        std::string(ir_->on_gpu() ? "CUDA" : "CPU") + "; family_hint '" + family +
                        "'). The llama path's weights stay loaded but unused.");
      }
    }
  }
  // Small quantised models (a Q8_0 download for the browser): dequantise
  // once, so the CPU path runs at f32 speed on the quantised values.
  // EMBER_DEQUANT_AT_LOAD=0 keeps the per-call quantised kernels.
  {
    const char* dq = std::getenv("EMBER_DEQUANT_AT_LOAD");
    if (!(dq && std::string(dq) == "0") && model_->dequantize_all_if_small(64u * 1000 * 1000))
      Logger::info("Dequantised all weights to f32 at load (small model).");
  }

  // paperkiln chat models: their vocabulary ("bpe" with id-pair merges, or the llama.cpp-compatible
  // "gpt2"/"llama-bpe" form with the chat specials at ids 0-3) goes through the vendored paperkiln
  // tokenizers in chat_tokenizer.h (see is_bpe()); any other GGUF keeps ember's own tokenizers.
  {
    std::string ext = std::filesystem::path(model_path_arg).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".gguf") {
      microtorch::bpe::GgufTokenizer g = microtorch::bpe::read_gguf_tokenizer(model_path_arg);
      const size_t n_merges = g.merges.size();
      bpe_ = ember::make_chat_tokenizer(std::move(g));
      if (bpe_) {
        repetition_penalty_ = 1.0f;  // mtstudio chat applies none
        Logger::info(std::string("Chat tokenizer (") + bpe_->kind() + "): " + std::to_string(bpe_->size()) +
                     " tokens, " + std::to_string(n_merges) + " merges (chat mode)");
      }
    }
  }

  const ModelConfig& final_model_config = model_->get_config(); // Explicitly use model's config
  int total_model_layers = final_model_config.num_hidden_layers;
  
  int effective_cpu_offload_layers = final_model_config.num_cpu_offload_layers;
  int gpu_layers_for_kvcache = total_model_layers - effective_cpu_offload_layers;
  if (gpu_layers_for_kvcache < 0) gpu_layers_for_kvcache = 0; // Sanity check, should not happen if model ctor is correct
  if (gpu_layers_for_kvcache > total_model_layers) gpu_layers_for_kvcache = total_model_layers; // Sanity

  Logger::info("[Session KVCache Init] Total Layers: " + std::to_string(total_model_layers) +
               ", Effective CPU Offload by Model: " + std::to_string(effective_cpu_offload_layers) +
               ", GPU Layers for KVCache: " + std::to_string(gpu_layers_for_kvcache));

  if (total_model_layers <= 0) {
      throw std::runtime_error("Model config has zero or negative num_hidden_layers before KVCache init.");
  }
  if (final_model_config.num_attention_heads <= 0) {
      throw std::runtime_error("Model config has zero or negative num_attention_heads before KVCache init.");
  }

  int head_dim = final_model_config.hidden_size / final_model_config.num_attention_heads;
  
  kv_cache_.initialize(final_model_config,          // Total layers for CPU part of KVCache
                       total_model_layers, 
                       gpu_layers_for_kvcache,    // Actual GPU layers for device memory
                       final_model_config.max_position_embeddings,
                       final_model_config.num_key_value_heads, 
                       head_dim,
                       max_batch_size_);
  Logger::info("TinyLlamaSession initialization complete (after KVCache init).");
}

TinyLlamaSession::~TinyLlamaSession() {
  Logger::info("TinyLlamaSession: Destroyed.");
}

std::string TinyLlamaSession::generate(const std::string& user_prompt, int steps,
                                     float temperature,
                                     int top_k, float top_p,
                                     const std::string& system_prompt_arg, // Renamed for clarity inside function
                                     bool apply_q_a_format_cli_hint) { // Renamed for clarity
  auto t_start = std::chrono::high_resolution_clock::now(); // Start timing

  // TTFT instrumentation (docs/TTFT.md): each phase up to the first token.
  using ttft_clock = std::chrono::steady_clock;
  auto ms_between = [](ttft_clock::time_point a, ttft_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  last_ttft_ = TtftBreakdown{};
  last_ttft_.t_entry = ttft_clock::now();

  generated_text_for_api_return_.clear(); // Clear for new generation
  generated_stream_.str("");             // Clear for new generation
  generated_stream_.clear();          // Clear error flags

  Logger::info("[Generate API] User prompt: \"" + user_prompt + "\", System prompt: \"" + system_prompt_arg + "\", Steps: " + std::to_string(steps));

  if (!model_ || !tokenizer_) {
    throw std::runtime_error("Model or tokenizer not loaded.");
  }

  std::string final_prompt_for_encoding;
  bool used_chat_template = false;

  // Log conditions for chat template application
  if (tokenizer_) {
    bool gguf_template_empty = tokenizer_->get_gguf_chat_template().empty();
    Logger::info("[Generate API] GGUF chat template from tokenizer is empty: " + std::string(gguf_template_empty ? "true" : "false"));
    if (!gguf_template_empty) {
        Logger::info("[Generate API] GGUF Template Content (first 100 chars): " + tokenizer_->get_gguf_chat_template().substr(0, 100));
    }
  } else {
    Logger::warning("[Generate API] Tokenizer is null before checking chat template!");
  }
  std::string family_log_str = "UNKNOWN";
  if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_log_str = "LLAMA_SENTENCEPIECE";
  else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_log_str = "LLAMA3_TIKTOKEN";
  Logger::info("[Generate API] Configured tokenizer_family: " + family_log_str);

  // New Priority Logic:
  // Priority 0: WORD_LEVEL models (transformer_cpp exports) are raw word-level
  // LMs: "Q:"/"A:" wrappers are OOV noise that shift every position, and any
  // chat formatting ("user: ... assistant:") is the caller's contract.
  // BPE chat models: the caller (server.cpp) already built the exact SFT
  // template with its <|user|>/<|assistant|> markers; nothing is added.
  if (bpe_) {
    final_prompt_for_encoding = user_prompt;
    used_chat_template = false;
  } else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::WORD_LEVEL) {
    Logger::info("[Generate API] WORD_LEVEL tokenizer: using raw prompt (no Q/A or chat template).");
    final_prompt_for_encoding = user_prompt;
    used_chat_template = false;
  }
  // Priority 1 (NEW): Legacy Q/A formatting if CLI hint is true
  else if (apply_q_a_format_cli_hint) {
    Logger::info("[Generate API] Using legacy Q/A formatting (CLI Hint is true - Priority 1).");
    std::string temp_prompt = user_prompt;
    if (!system_prompt_arg.empty()) {
        temp_prompt = system_prompt_arg + "\\n\\nQ: " + user_prompt + "\\nA:";
    } else {
        temp_prompt = "Q: " + user_prompt + "\\nA:";
    }
    final_prompt_for_encoding = temp_prompt;
    used_chat_template = false; // Q/A is not a 'chat template' in the GGUF sense
  }
  // Priority 2 (WAS 1): GGUF Chat Template from Tokenizer (only if Q/A hint is false)
  else if (tokenizer_ && !tokenizer_->get_gguf_chat_template().empty()) {
    std::string gguf_template_content = tokenizer_->get_gguf_chat_template();
    bool is_llama_sentencepiece_family = (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE);
    bool looks_like_jinja = (gguf_template_content.find("{%") != std::string::npos);

    if (is_llama_sentencepiece_family && looks_like_jinja) {
        Logger::info("[Generate API] Detected LLAMA_SENTENCEPIECE model with a Jinja-like GGUF template. Forcing Q/A format to avoid C++ Jinja processing issues (Priority 2 Override).");
        std::string temp_prompt = user_prompt;
        if (!system_prompt_arg.empty()) {
            temp_prompt = system_prompt_arg + "\\\\n\\\\nQ: " + user_prompt + "\\\\nA:";
        } else {
            temp_prompt = "Q: " + user_prompt + "\\\\nA:";
        }
        final_prompt_for_encoding = temp_prompt;
        used_chat_template = false;
    } else {
        Logger::info("[Generate API] Using GGUF chat template from tokenizer (Q/A Hint false - Priority 2).");
        final_prompt_for_encoding = tokenizer_->apply_chat_template(user_prompt, system_prompt_arg, config_);
        used_chat_template = true;
    }
  } 
  // Priority 3 (WAS 2): Llama 3 family specific template (only if Q/A hint false and no GGUF template)
  else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
    Logger::info("[Generate API] Llama 3 tokenizer family detected, using apply_chat_template (Q/A Hint false, No GGUF template - Priority 3).");
    final_prompt_for_encoding = tokenizer_->apply_chat_template(user_prompt, system_prompt_arg, config_);
    used_chat_template = true;
  }
  // Priority 4 (WAS 3/4 depending on apply_q_a_format_cli_hint): Raw prompt (if all above are false)
  else {
    Logger::info("[Generate API] No applicable template/hint. Using user prompt as is (prepending system prompt if available - Priority 4).");
    if (!system_prompt_arg.empty()) {
        final_prompt_for_encoding = system_prompt_arg + "\\n\\n" + user_prompt;
    } else {
        final_prompt_for_encoding = user_prompt;
    }
    used_chat_template = false;
  }
  
  Logger::debug("[Generate API] Final prompt for encoding (first 100 chars): \\\"" + final_prompt_for_encoding.substr(0, 100) + "\\\"");

  // BPE chat models take no BOS (the SFT template starts at <|user|>).
  std::vector<int> tokens = bpe_ ? bpe_->encode(final_prompt_for_encoding)
                                 : tokenizer_->encode(final_prompt_for_encoding, true, false, Tokenizer::PreTokenizeMethod::DEFAULT); // add_bos=true, add_eos=false (EOS handled by loop)
  // Stop tokens: <|endoftext|> always; in BPE chat mode also a generated
  // <|user|> or <|assistant|> (the model starting the next turn).
  auto is_stop = [&](int id) {
    return bpe_ ? (id == 1 || id == 2 || id == 3) : id == eos_token_id_;
  };
  auto decode_one = [&](int id) {
    return bpe_ ? bpe_->decode({id}) : tokenizer_->decode({id}, false);
  };
  // Streaming: the per-token callback (set_token_callback) sees every
  // generated token except stop markers, in order.
  auto emit_token = [&](int id) {
    if (token_callback_ && !is_stop(id)) token_callback_(id);
  };
  bool stopped_at_first = false;
  last_stop_reason_ = "length";
  last_generated_tokens_ = 0;
  last_prompt_ids_ = tokens;
  last_generated_ids_.clear();
  last_stop_id_ = -1;

  if (tokens.empty()) {
    Logger::warning("Tokenization resulted in empty ID list for prompt: " +
                    final_prompt_for_encoding);
    return "";
  }

  int num_prompt_tokens = tokens.size();
  Logger::info("[Generate API] Number of prompt tokens: " + std::to_string(num_prompt_tokens));
  last_ttft_.num_prompt_tokens = num_prompt_tokens;
  last_ttft_.tokenize_ms = ms_between(last_ttft_.t_entry, ttft_clock::now());
  {
    // Prefix-cache opportunity against what the previous request cached.
    size_t lcp = 0;
    while (lcp < kv_token_history_.size() && lcp < tokens.size() &&
           kv_token_history_[lcp] == tokens[lcp]) {
      ++lcp;
    }
    last_ttft_.prefix_prev_cached_tokens = static_cast<int>(kv_token_history_.size());
    last_ttft_.prefix_reusable_tokens =
        static_cast<int>(std::min(lcp, tokens.size() - 1));
    last_ttft_.prefix_hit_rate =
        static_cast<double>(last_ttft_.prefix_reusable_tokens) / num_prompt_tokens;
  }
  // Prefix reuse (EMBER_PREFIX_REUSE, on by default): the KV cache still
  // holds the previous request's prompt and reply, so the shared leading
  // tokens are kept and only the rest is prefilled, from that offset. At
  // least the last prompt token is always recomputed, for its logits. Only
  // the batched prefill path reuses; the token-by-token path starts over.
  static const bool prefix_reuse_on = [] {
    const char* v = std::getenv("EMBER_PREFIX_REUSE");
    return !(v && std::string(v) == "0");
  }();
  int reuse = prefix_reuse_on ? last_ttft_.prefix_reusable_tokens : 0;
  // Recurrent IR state (linear attention, state-space scans) cannot drop
  // tokens: resume where the interpreter holds the state after the prefix
  // (the live state or a snapshot), else nearer the start or from 0.
  if (ir_ && ir_->has_recurrent()) reuse = ir_->resume_point(reuse);
  kv_token_history_.clear();  // refilled only if this request completes

  int total_steps = num_prompt_tokens + steps -1; // Max total tokens including prompt
  int generated_count = 0;
  int next_token_id = -1;

  std::vector<float> logits; // Declare logits here, to be populated by prefill or loop
  std::vector<int> generated_token_ids; // Track generated tokens separately

  // The /chat contract's repetition penalty (set_repeat_penalty): the window
  // is the last repeat_last_n_ ids of prompt + reply so far, the token being
  // sampled not included; each distinct id outside the stop set is scaled
  // once, before sampling.
  const bool repeat_on = repeat_penalty_ != 1.0f && repeat_last_n_ > 0;
  std::vector<char> repeat_seen(repeat_on ? config_.vocab_size : 0);
  auto apply_repeat_penalty = [&](std::vector<float>& lg) {
    if (!repeat_on) return;
    const size_t np = tokens.size(), total = np + generated_token_ids.size();
    const size_t first = total > static_cast<size_t>(repeat_last_n_) ? total - repeat_last_n_ : 0;
    if (repeat_seen.size() < lg.size()) repeat_seen.resize(lg.size());
    std::fill(repeat_seen.begin(), repeat_seen.end(), 0);
    for (size_t i = first; i < total; ++i) {
      const int id = i < np ? tokens[i] : generated_token_ids[i - np];
      if (id < 0 || static_cast<size_t>(id) >= lg.size() || repeat_seen[id] || is_stop(id)) continue;
      repeat_seen[id] = 1;
      float& l = lg[id];
      l = l > 0.0f ? l / repeat_penalty_ : l * repeat_penalty_;
    }
  };

  auto t_kv_reset = ttft_clock::now();
  if (reuse > 0) {
    // Keep positions [0, reuse); every later position is rewritten before
    // attention reads it (attention reads only positions <= the current one).
    kv_cache_.seq_len = reuse;
  } else {
    kv_cache_.clear_data(); // Clear K/V vector data for all layers
    kv_cache_.seq_len = 0;  // Reset KVCache logical sequence length for new sequence
  }
  last_ttft_.kv_reset_ms = ms_between(t_kv_reset, ttft_clock::now());

  std::vector<float> current_data_host; // To hold embedding or output of CPU layers
  int start_pos_for_loop = 0;

  // Batch prefill for every prompt length. The old 32-token threshold cost
  // short (chat-sized) prompts: on chat_sft_bpe, 20 prompts of 10-30 tokens,
  // p50 TTFT 294-338 ms -> 53-55 ms on CUDA and 983-1251 -> 592-650 ms on CPU,
  // with identical greedy replies (experiments/chat_2026-10-18/).
  // EMBER_PREFILL_MIN restores a threshold (measurement knob).
  static const int prefill_min = [] {
    const char* v = std::getenv("EMBER_PREFILL_MIN");
    return v && *v ? std::max(1, std::atoi(v)) : 1;
  }();
  bool prefill_enabled = num_prompt_tokens >= prefill_min;
  // Reused positions: [0, prefill_from) stay in the cache untouched.
  const int prefill_from = prefill_enabled ? reuse : 0;
  const int prefill_count = num_prompt_tokens - prefill_from;
  last_ttft_.prefix_reused_tokens = prefill_from;

if (prefill_enabled) {
      Logger::info("[Generate API] Prefill enabled. num_prompt_tokens: " + std::to_string(num_prompt_tokens) + 
                   ", num_cpu_offload_layers: " + std::to_string(config_.num_cpu_offload_layers) +
                   ", total_hidden_layers: " + std::to_string(config_.num_hidden_layers));
      last_ttft_.batch_prefill = true;
      auto t_embed = ttft_clock::now();

      std::vector<float> batch_initial_embeddings(ir_ ? 0 : prefill_count * config_.hidden_size);
      for (int i = 0; i < (ir_ ? 0 : prefill_count); ++i) {  // the interpreter embeds tokens itself
          std::vector<float> token_embedding = model_->lookup_embedding(tokens[prefill_from + i]);
          if (token_embedding.empty()) {
              Logger::error("Prefill: Embedding lookup returned empty vector for token ID: " + std::to_string(tokens[i]) + " at prompt pos " + std::to_string(i));
              return ""; // Or handle error appropriately
          }
          std::copy(token_embedding.begin(), token_embedding.end(), batch_initial_embeddings.begin() + i * config_.hidden_size);
      }
      auto t_prefill = ttft_clock::now();
      last_ttft_.embed_ms = ms_between(t_embed, t_prefill);

      if (ir_) {
        const std::vector<int> slice(tokens.begin() + prefill_from, tokens.begin() + prefill_from + prefill_count);
        logits = ir_->forward(slice, prefill_from);
      } else {
      // If there are CPU layers to process for prefill
      std::vector<float> cpu_processed_embeddings;
      if (config_.num_cpu_offload_layers > 0) {
          Logger::info("[Generate API] Prefill: Processing " + std::to_string(config_.num_cpu_offload_layers) + " CPU layers for the batch.");
          cpu_processed_embeddings = model_->forward_cpu_batch(batch_initial_embeddings, prefill_count, config_.num_cpu_offload_layers, prefill_from, &kv_cache_);
          if (cpu_processed_embeddings.empty()) {
               Logger::error("Prefill: forward_cpu_batch returned empty or failed.");
               return "";
          }
      } else {
          cpu_processed_embeddings = batch_initial_embeddings; // No CPU layers, pass embeddings directly
      }

      // If all layers are CPU layers (i.e., num_gpu_layers == 0)
      if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
          Logger::info("[Generate API] Prefill: All layers are on CPU. Getting logits from final CPU layer output.");
          std::vector<float> batch_logits = model_->forward_cpu_logits_batch(cpu_processed_embeddings, prefill_count);
          if (batch_logits.empty() || batch_logits.size() % config_.vocab_size != 0) {
              Logger::error("Prefill: forward_cpu_logits_batch returned invalid logits.");
              return "";
          }
          // Extract logits for the last token of the prompt
          logits.assign(batch_logits.begin() + (prefill_count - 1) * config_.vocab_size,
                        batch_logits.begin() + prefill_count * config_.vocab_size);
      } else { // GPU layers exist and need to be processed
#ifdef HAS_CUDA
          Logger::info("[Generate API] Prefill: Processing GPU layers for the batch.");
          // Copy the (potentially CPU-processed) embeddings to the device
          float* d_temp_batch_embeddings = nullptr;
          size_t batch_embeddings_size_bytes = cpu_processed_embeddings.size() * sizeof(float);

          if (batch_embeddings_size_bytes == 0) {
              Logger::error("Prefill: cpu_processed_embeddings is empty, cannot proceed with GPU batch prefill.");
              return ""; // Or handle error appropriately
          }

          gpuErrchk(cudaMalloc(&d_temp_batch_embeddings, batch_embeddings_size_bytes));
          if (!d_temp_batch_embeddings) {
              Logger::error("Prefill: cudaMalloc failed for d_temp_batch_embeddings.");
              return ""; // Or handle error appropriately
          }
          
          gpuErrchk(cudaMemcpy(d_temp_batch_embeddings, cpu_processed_embeddings.data(), 
                               batch_embeddings_size_bytes, cudaMemcpyHostToDevice));

          // Events on the default stream (the forward runs on stream 0).
          cudaEvent_t ev_fwd_start, ev_fwd_end;
          gpuErrchk(cudaEventCreate(&ev_fwd_start));
          gpuErrchk(cudaEventCreate(&ev_fwd_end));
          gpuErrchk(cudaEventRecord(ev_fwd_start, 0));
          nvtxRangePushA("ttft_prefill_forward");
          logits = model_->forward_device_batch_prefill(d_temp_batch_embeddings, prefill_count, prefill_from, &kv_cache_, 0);
          nvtxRangePop();
          gpuErrchk(cudaEventRecord(ev_fwd_end, 0));
          gpuErrchk(cudaEventSynchronize(ev_fwd_end));
          float fwd_span_ms = 0.0f;
          gpuErrchk(cudaEventElapsedTime(&fwd_span_ms, ev_fwd_start, ev_fwd_end));
          last_ttft_.prefill_gpu_span_ms = fwd_span_ms;
          last_ttft_.op_ms = model_->last_prefill_op_ms();
          gpuErrchk(cudaEventDestroy(ev_fwd_start));
          gpuErrchk(cudaEventDestroy(ev_fwd_end));

          if (d_temp_batch_embeddings) {
              gpuErrchk(cudaFree(d_temp_batch_embeddings));
          }
#else
          Logger::error("[Generate API] GPU layers requested but CUDA not available. Cannot proceed.");
          return "";
#endif
      }
      }  // llama path
      
      if (logits.empty()) {
          Logger::error("Prefill: Logits are empty after prefill processing.");
          return ""; // Critical error
      }
      auto t_sample = ttft_clock::now();
      last_ttft_.prefill_wall_ms = ms_between(t_prefill, t_sample);
      apply_repetition_penalty(logits, generated_token_ids, repetition_penalty_);
      apply_repeat_penalty(logits);
      debug_dump_topk(logits);
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      generated_token_ids.push_back(next_token_id); // Track generated token
      generated_count++;
      emit_token(next_token_id);

      // Stream the first generated token from prefill
      generated_stream_ << decode_one(next_token_id);
      generated_text_for_api_return_ += decode_one(next_token_id);
      last_ttft_.t_first_token = ttft_clock::now();
      last_ttft_.first_token = true;
      last_ttft_.sample_ms = ms_between(t_sample, last_ttft_.t_first_token);
      last_ttft_.ttft_ms = ms_between(last_ttft_.t_entry, last_ttft_.t_first_token);

      start_pos_for_loop = num_prompt_tokens; // Next token will be at num_prompt_tokens
      if (bpe_ && is_stop(next_token_id)) {
        stopped_at_first = true;  // the reply is empty; skip the decode loop
        total_steps = 0;
      }
      kv_cache_.seq_len = num_prompt_tokens; // KVCache is now filled up to num_prompt_tokens

      Logger::info("[Generate API] Prefill completed. next_token_id: " + std::to_string(next_token_id) + 
                   ", Decoded: \"" + decode_one(next_token_id) + "\"" + 
                   ", start_pos_for_loop set to: " + std::to_string(start_pos_for_loop));
  }

  // EMBER_DECODE_TIMING=1: wall time per decode phase, host side here and
  // GPU-forward side in TinyLlamaModel::forward_device, summarised on stderr
  // after the loop (per decode step).
  static const bool decode_timing = std::getenv("EMBER_DECODE_TIMING") != nullptr;
  std::map<std::string, double> host_phase_ms;
  int decode_steps = 0;
  auto t_phase = ttft_clock::now();
  auto host_phase = [&](const char* name) {
    if (!decode_timing) return;
    const auto now = ttft_clock::now();
    host_phase_ms[name] += ms_between(t_phase, now);
    t_phase = now;
  };
#ifdef HAS_CUDA
  if (decode_timing) model_->decode_phase_ms().clear();
#endif
  const auto t_decode_start = ttft_clock::now();
  // EMBER_SPEC=k: speculative decoding with prompt-lookup drafts of up to k
  // tokens, greedy (top_k == 1) only, all-GPU or all-CPU layers, and off
  // under the contract repetition penalty (its window moves with each row).
  static const int spec_k = [] {
    const char* e = std::getenv("EMBER_SPEC");
    return e ? std::max(0, std::atoi(e)) : 0;
  }();
  const bool spec_on = spec_k > 0 && top_k == 1 && !repeat_on &&
                       (config_.num_cpu_offload_layers == 0 ||
                        config_.num_cpu_offload_layers == config_.num_hidden_layers);

  for (int pos = start_pos_for_loop; pos < total_steps; ++pos) {
    t_phase = ttft_clock::now();
    ++decode_steps;
    if (pos >= config_.max_position_embeddings) {
      Logger::warning("Reached max sequence length (" +
                      std::to_string(config_.max_position_embeddings) +
                      "). Stopping.");
      break;
    }

    int input_token_id;

    if (pos == num_prompt_tokens && start_pos_for_loop == num_prompt_tokens) {
        // This is the first token *after* a successful prefill.
        // `next_token_id` was already sampled using prefill's logits from the *last prompt token*.
        // This `next_token_id` is the actual input for the current position `pos`.
        input_token_id = next_token_id; 
        Logger::debug("[Generate Loop] First token post-prefill. Using prefill's next_token_id: " + std::to_string(input_token_id) + " for pos " + std::to_string(pos));
    } else {
        // Standard iterative logic:
        // If prefill didn't run (start_pos_for_loop == 0):
        //   For pos < num_prompt_tokens: use prompt token.
        //   For pos >= num_prompt_tokens: use previously sampled next_token_id.
        // If prefill did run (start_pos_for_loop == num_prompt_tokens):
        //   This 'else' block is for pos > num_prompt_tokens, so use previously sampled next_token_id.
        input_token_id = (pos < num_prompt_tokens && start_pos_for_loop == 0) ? tokens[pos] : next_token_id;
        if (start_pos_for_loop == 0 && pos < num_prompt_tokens) {
             Logger::debug("[Generate Loop] No prefill, prompt token. Using tokens[" + std::to_string(pos) + "]: " + std::to_string(input_token_id) + " for pos " + std::to_string(pos));
        } else {
             Logger::debug("[Generate Loop] Standard generation. Using previously sampled next_token_id: " + std::to_string(input_token_id) + " for pos " + std::to_string(pos));
        }
    }
    
    // Speculative step: one batched forward over [input, draft...] from pos
    // gives the logits at every position; tokens are taken greedily while
    // they equal the draft, plus the first one that does not (or a stop).
    // KV entries written for rejected draft positions lie past the valid
    // length and are overwritten by the next steps. Recurrent IR state has
    // absorbed them all: the next forward, from pos + accepted + 1, restores
    // the interpreter's snapshot after that row (all_rows keeps every row).
    if (spec_on && pos >= num_prompt_tokens) {
      const int room = std::min({spec_k, steps - generated_count, total_steps - pos - 1,
                                 config_.max_position_embeddings - pos - 1});
      std::vector<int> hist(tokens.begin(), tokens.begin() + num_prompt_tokens);
      hist.insert(hist.end(), generated_token_ids.begin(), generated_token_ids.end());
      const std::vector<int> draft = room > 0 ? prompt_lookup_draft(hist, room) : std::vector<int>{};
      if (!draft.empty()) {
        std::vector<int> batch = {input_token_id};
        batch.insert(batch.end(), draft.begin(), draft.end());
        const int n = static_cast<int>(batch.size());
        const int hs = config_.hidden_size, vs = config_.vocab_size;
        std::vector<float> emb(ir_ ? 0 : (size_t)n * hs);
        for (int i = 0; i < (ir_ ? 0 : n); ++i) {  // the interpreter embeds tokens itself
          const std::vector<float> e = model_->lookup_embedding(batch[i]);
          std::copy(e.begin(), e.end(), emb.begin() + (size_t)i * hs);
        }
        std::vector<float> all;
        if (ir_) {
          all = ir_->forward(batch, pos, /*all_rows=*/true);
        } else if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
          all = model_->forward_cpu_logits_batch(
              model_->forward_cpu_batch(emb, n, config_.num_hidden_layers, pos, &kv_cache_), n);
        } else {
#ifdef HAS_CUDA
          // The graph-like verification path (about one decode step per pass);
          // the batched prefill when it does not apply.
          if (config_.num_cpu_offload_layers == 0) all = model_->forward_device_verify(batch, pos, &kv_cache_);
          if (all.empty()) {
          float* d_emb = nullptr;
          gpuErrchk(cudaMalloc(&d_emb, emb.size() * sizeof(float)));
          gpuErrchk(cudaMemcpy(d_emb, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice));
          all = model_->forward_device_batch_prefill(d_emb, n, pos, &kv_cache_, 0, /*all_logits=*/true);
          gpuErrchk(cudaFree(d_emb));
          }
#endif
        }
        if (all.size() != (size_t)n * vs) {
          Logger::error("Speculative verification returned " + std::to_string(all.size()) + " logits; stopping.");
          break;
        }
        int accepted = 0;
        bool stop = false;
        for (int i = 0; i < n; ++i) {
          std::vector<float> lg(all.begin() + (size_t)i * vs, all.begin() + (size_t)(i + 1) * vs);
          apply_repetition_penalty(lg, generated_token_ids, repetition_penalty_);
          const int t = sample_top_k_top_p_temperature(lg, temperature, top_k, top_p, rng_);
          generated_token_ids.push_back(t);
          generated_count++;
          emit_token(t);
          generated_stream_ << decode_one(t);
          generated_text_for_api_return_ += decode_one(t);
          next_token_id = t;
          if (is_stop(t) || generated_count >= steps) { stop = true; break; }
          if (i + 1 < n && t == batch[i + 1]) { ++accepted; continue; }
          break;
        }
        last_ttft_.spec_drafted += static_cast<int>(draft.size());
        last_ttft_.spec_accepted += accepted;
        last_ttft_.spec_passes += 1;
        kv_cache_.seq_len = pos + accepted + 1;  // input + accepted drafts are now cached
        host_phase("host.spec_step");
        if (stop) break;
        pos += accepted;  // the loop's ++pos then moves to the next input position
        continue;
      }
    }

    host_phase("host.loop_head");
    if (!ir_) current_data_host = model_->lookup_embedding(input_token_id);
    if (pos == 14 || pos == 15 || pos == 16) {
        log_vector_summary_detailed("[API_CPP GenLoop] current_data_host after lookup_embedding for input_token_id=" + std::to_string(input_token_id),
                                    current_data_host, pos, -100, 8);
    }
    host_phase("host.embed_lookup");

    if (!ir_ && current_data_host.empty()) {
        Logger::error("Embedding lookup returned empty vector for token ID: " + std::to_string(input_token_id) + " at pos " + std::to_string(pos));
        break; 
    }

    // Mixed-mode forward pass logic
    if (ir_) {
      logits = ir_->forward({input_token_id}, pos);
    } else if (config_.num_cpu_offload_layers > 0 && config_.num_cpu_offload_layers < config_.num_hidden_layers) {
#ifdef HAS_CUDA
      // Mixed CPU/GPU mode: First process CPU layers, then GPU layers
      Logger::debug("[Mixed Mode] Processing " + std::to_string(config_.num_cpu_offload_layers) + " CPU layers first");
      std::vector<float> intermediate_activations = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
      
      Logger::debug("[Mixed Mode] CPU layers complete, transferring to GPU for remaining layers");
      gpuErrchk(cudaMemcpy(model_->get_x_dev(), intermediate_activations.data(), intermediate_activations.size() * sizeof(float), cudaMemcpyHostToDevice));
      logits = model_->forward_device(model_->get_x_dev(), pos, &kv_cache_, nullptr);
#else
      Logger::error("[Mixed Mode] Mixed CPU/GPU mode requested but CUDA not available. Cannot proceed.");
      break;
#endif
    } else if (config_.num_cpu_offload_layers == 0) {
#ifdef HAS_CUDA
      // GPU-only mode: the captured decode graph when it applies, else the
      // per-launch forward.
      logits = model_->forward_device_graph(input_token_id, pos, &kv_cache_);
      host_phase("host.forward_device_graph");
      if (logits.empty()) {
        gpuErrchk(cudaMemcpy(model_->get_x_dev(), current_data_host.data(), current_data_host.size() * sizeof(float), cudaMemcpyHostToDevice));
        host_phase("host.embed_h2d");
        logits = model_->forward_device(model_->get_x_dev(), pos, &kv_cache_, nullptr);
        host_phase("host.forward_device(total)");
      }
#else
      Logger::error("[GPU-only Mode] GPU-only mode requested but CUDA not available. Cannot proceed.");
      break;
#endif
    } else {
      // CPU-only mode
      logits = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
    }

    // Sampling logic: Only sample if we're at the last prompt token or generating
    if (pos == num_prompt_tokens - 1 || pos >= num_prompt_tokens) {
      apply_repetition_penalty(logits, generated_token_ids, repetition_penalty_);
      apply_repeat_penalty(logits);
      debug_dump_topk(logits);
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      host_phase("host.sample");

      // Only add to generated tokens if we're actually generating (not just finishing prompt)
      if (pos >= num_prompt_tokens) {
        generated_token_ids.push_back(next_token_id);
        generated_count++;
        emit_token(next_token_id);
        
        // Stream the generated token
        generated_stream_ << decode_one(next_token_id);
        generated_text_for_api_return_ += decode_one(next_token_id);
      } else {
        // This is the first token sampled from the last prompt position
        generated_token_ids.push_back(next_token_id);
        generated_count++;
        emit_token(next_token_id);
        
        // Stream the first generated token
        generated_stream_ << decode_one(next_token_id);
        generated_text_for_api_return_ += decode_one(next_token_id);
        // Token-by-token path: only the total is recorded (batch_prefill=false).
        last_ttft_.t_first_token = ttft_clock::now();
        last_ttft_.first_token = true;
        last_ttft_.ttft_ms = ms_between(last_ttft_.t_entry, last_ttft_.t_first_token);

        Logger::info("[Generate API] First token sampled from prompt: " + std::to_string(next_token_id) + 
                     ", Decoded: \"" + decode_one(next_token_id) + "\"");
      }
    }

    if (is_stop(next_token_id) && pos >= (bpe_ ? num_prompt_tokens - 1 : num_prompt_tokens)) { // EOS only if we are generating
      Logger::info("EOS token (" + std::to_string(eos_token_id_) +
                     ") sampled at pos " + std::to_string(pos) + ". Stopping.");
      break;
    }

    if (generated_count >= steps) { // steps is max new tokens
      Logger::info("Reached max generation steps (" + std::to_string(steps) + "). Stopping.");
      break;
    }

    // KVCache seq_len update for single token pass (already handled by batch prefill for its tokens)
    // This needs to happen *after* the forward pass for the current 'pos' has updated the cache for 'pos'.
    // So, after processing 'pos', the cache now contains information up to and including 'pos'.
    // The length of the sequence in the cache is pos + 1.
    if (!prefill_enabled || pos >= num_prompt_tokens) { // only update if not prefill, or if we are past prompt token processing in prefill case
        kv_cache_.seq_len = pos + 1;
        // Logger::debug("[Generate Loop] KVCache seq_len updated to: " + std::to_string(kv_cache_.seq_len) + " after processing pos " + std::to_string(pos));
    }
    host_phase("host.decode_text_and_rest");
  }
  if (decode_timing && decode_steps > 0) {
    const double total = ms_between(t_decode_start, ttft_clock::now());
    std::fprintf(stderr, "[decode timing] %d steps, %.3f ms/step wall (timed run, phases sync)\n", decode_steps,
                 total / decode_steps);
    for (const auto& kv : host_phase_ms)
      std::fprintf(stderr, "  %-34s %8.3f ms/step\n", kv.first.c_str(), kv.second / decode_steps);
#ifdef HAS_CUDA
    for (const auto& kv : model_->decode_phase_ms())
      std::fprintf(stderr, "    gpu.%-30s %8.3f ms/step\n", kv.first.c_str(), kv.second / decode_steps);
#endif
  }

  // Log all generated IDs before decoding
  std::string generated_ids_str = "[Generated IDs Pre-Decode] ";
  for(int gen_id : generated_token_ids) {
    generated_ids_str += std::to_string(gen_id) + " ";
  }
  Logger::debug(generated_ids_str);

  std::string result;
  if (bpe_) {
    std::vector<int> out = generated_token_ids;
    if (!out.empty() && is_stop(out.back())) {
      last_stop_reason_ = out.back() == 1 ? "eos" : "turn";
      last_stop_id_ = out.back();
      out.pop_back();
    }
    last_generated_tokens_ = static_cast<int>(out.size());
    last_generated_ids_ = out;
    result = bpe_->decode(out);
    // Strip leading and trailing whitespace (the C isspace set), as mtstudio does.
    const char* ws = " \t\n\v\f\r";
    const auto first = result.find_first_not_of(ws);
    const auto last = result.find_last_not_of(ws);
    result = first == std::string::npos ? std::string() : result.substr(first, last - first + 1);
    (void)stopped_at_first;
  } else {
    const bool stopped = !generated_token_ids.empty() && is_stop(generated_token_ids.back());
    if (stopped) last_stop_reason_ = "eos";
    last_generated_tokens_ = static_cast<int>(generated_token_ids.size()) - (stopped ? 1 : 0);
    result = tokenizer_->decode(generated_token_ids, true);
  }
  Logger::info("Generated response: " + result);

  // What the KV cache now holds: the prompt plus every generated token that
  // was fed back through the model (all but the last one sampled).
  kv_token_history_ = tokens;
  if (!generated_token_ids.empty()) {
    kv_token_history_.insert(kv_token_history_.end(), generated_token_ids.begin(),
                             generated_token_ids.end() - 1);
  }

  auto t_end = std::chrono::high_resolution_clock::now(); // End timing
  double time_taken_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  
  // Create a string with the desired precision for the time
  std::ostringstream time_ss;
  time_ss << std::fixed << std::setprecision(4) << time_taken_ms;
  Logger::info("[INFO] Total generation processing time: " + time_ss.str() + " ms");

  return result;
}

std::vector<std::string> TinyLlamaSession::generate_batch(const std::vector<std::string>& prompts,
                                                         int steps,
                                                         float temperature,
                                                         int top_k, float top_p,
                                                         const std::string& system_prompt_arg,
                                                         bool apply_q_a_format_cli_hint) {
  auto t_start = std::chrono::high_resolution_clock::now();

  if (prompts.empty()) {
    throw std::runtime_error("Cannot process empty prompts vector for batch generation.");
  }

  if (static_cast<int>(prompts.size()) > max_batch_size_) {
    throw std::runtime_error("Batch size " + std::to_string(prompts.size()) + 
                             " exceeds maximum batch size " + std::to_string(max_batch_size_));
  }

  Logger::info("[Batch Generate API] Processing " + std::to_string(prompts.size()) + 
               " prompts in batch. Steps: " + std::to_string(steps));

  if (!model_ || !tokenizer_) {
    throw std::runtime_error("Model or tokenizer not loaded for batch generation.");
  }

  // Process each prompt to create final prompts and tokenize them
  std::vector<std::string> final_prompts(prompts.size());
  std::vector<std::vector<int>> all_tokens(prompts.size());
  std::vector<int> prompt_lengths(prompts.size());
  int max_prompt_length = 0;

  for (size_t i = 0; i < prompts.size(); ++i) {
    // Apply same prompt processing logic as single generate()
    std::string final_prompt_for_encoding;
    bool used_chat_template = false;

    // Same priority logic as single generate()
    if (apply_q_a_format_cli_hint) {
      Logger::info("[Batch Generate API] Using legacy Q/A formatting for prompt " + std::to_string(i));
      if (!system_prompt_arg.empty()) {
        final_prompt_for_encoding = system_prompt_arg + "\\n\\nQ: " + prompts[i] + "\\nA:";
      } else {
        final_prompt_for_encoding = "Q: " + prompts[i] + "\\nA:";
      }
    } else if (tokenizer_ && !tokenizer_->get_gguf_chat_template().empty()) {
      std::string gguf_template_content = tokenizer_->get_gguf_chat_template();
      bool is_llama_sentencepiece_family = (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE);
      bool looks_like_jinja = (gguf_template_content.find("{%") != std::string::npos);

      if (is_llama_sentencepiece_family && looks_like_jinja) {
        Logger::info("[Batch Generate API] Using Q/A format override for prompt " + std::to_string(i));
        if (!system_prompt_arg.empty()) {
          final_prompt_for_encoding = system_prompt_arg + "\\\\n\\\\nQ: " + prompts[i] + "\\\\nA:";
        } else {
          final_prompt_for_encoding = "Q: " + prompts[i] + "\\\\nA:";
        }
      } else {
        Logger::info("[Batch Generate API] Using GGUF chat template for prompt " + std::to_string(i));
        final_prompt_for_encoding = tokenizer_->apply_chat_template(prompts[i], system_prompt_arg, config_);
        used_chat_template = true;
      }
    } else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
      Logger::info("[Batch Generate API] Using Llama 3 chat template for prompt " + std::to_string(i));
      final_prompt_for_encoding = tokenizer_->apply_chat_template(prompts[i], system_prompt_arg, config_);
      used_chat_template = true;
    } else {
      Logger::info("[Batch Generate API] Using raw prompt for prompt " + std::to_string(i));
      if (!system_prompt_arg.empty()) {
        final_prompt_for_encoding = system_prompt_arg + "\\n\\n" + prompts[i];
      } else {
        final_prompt_for_encoding = prompts[i];
      }
    }

    final_prompts[i] = final_prompt_for_encoding;
    all_tokens[i] = tokenizer_->encode(final_prompt_for_encoding, true, false, Tokenizer::PreTokenizeMethod::DEFAULT);
    
    if (all_tokens[i].empty()) {
      Logger::warning("Batch tokenization resulted in empty ID list for prompt " + std::to_string(i));
      all_tokens[i].push_back(tokenizer_->bos_token_id()); // Fallback to BOS token
    }

    prompt_lengths[i] = all_tokens[i].size();
    max_prompt_length = std::max(max_prompt_length, prompt_lengths[i]);
    
    Logger::info("[Batch Generate API] Prompt " + std::to_string(i) + ": " + 
                 std::to_string(prompt_lengths[i]) + " tokens");
  }

  // Initialize batch mode
  kv_cache_.initialize_batch(static_cast<int>(prompts.size()));
  kv_cache_.clear_data();
  kv_token_history_.clear();  // the cache no longer holds the last generate()'s tokens

  std::vector<std::string> results(prompts.size());
  
  // Try parallel batch processing if enabled
  if (use_batch_generation_) {
    Logger::info("[Batch Generate API] Using parallel batch processing");
    
    Logger::info("[DEBUG] Initializing KV cache for batch mode");
    kv_cache_.initialize_batch(static_cast<int>(prompts.size()));
    Logger::info("[DEBUG] KV cache batch initialization completed");
    
    Logger::info("[DEBUG] Clearing KV cache data");
    kv_cache_.clear_data();
    kv_token_history_.clear();  // the cache no longer holds the last generate()'s tokens
    Logger::info("[DEBUG] KV cache clear completed");
    
    // Phase 1: Parallel Batch Prefill
    Logger::info("[DEBUG] About to call batch_prefill_parallel");
    Logger::info("[DEBUG] all_tokens.size()=" + std::to_string(all_tokens.size()) + ", prompt_lengths.size()=" + std::to_string(prompt_lengths.size()));
    
    std::vector<std::vector<float>> batch_final_logits;
    
    bool prefill_success = batch_prefill_parallel(all_tokens, prompt_lengths, batch_final_logits);
    
    if (prefill_success && batch_final_logits.size() == prompts.size()) {
      Logger::info("[Batch Generate API] Batch prefill successful, starting parallel generation");
      
      // Add safety check for batch_final_logits
      Logger::info("[DEBUG] Checking batch_final_logits integrity after prefill");
      for (size_t i = 0; i < batch_final_logits.size(); ++i) {
        if (batch_final_logits[i].empty()) {
          Logger::error("[DEBUG] batch_final_logits[" + std::to_string(i) + "] is empty!");
          goto fallback_sequential;
        }
        if (batch_final_logits[i].size() != static_cast<size_t>(config_.vocab_size)) {
          Logger::error("[DEBUG] batch_final_logits[" + std::to_string(i) + "] has wrong size: " + 
                       std::to_string(batch_final_logits[i].size()) + " vs expected " + std::to_string(config_.vocab_size));
          goto fallback_sequential;
        }
        // Check for NaN/Inf values
        for (size_t j = 0; j < std::min(static_cast<size_t>(10UL), batch_final_logits[i].size()); ++j) {
          if (!std::isfinite(batch_final_logits[i][j])) {
            Logger::error("[DEBUG] batch_final_logits[" + std::to_string(i) + "][" + std::to_string(j) + "] is not finite: " + std::to_string(batch_final_logits[i][j]));
            goto fallback_sequential;
          }
        }
      }
      Logger::info("[DEBUG] batch_final_logits integrity check passed");
      
      // Sample first tokens for all sequences
      std::vector<int> current_tokens(prompts.size());
      std::vector<std::vector<int>> all_generated_tokens(prompts.size());
      std::vector<int> sequence_positions(prompts.size());
      std::vector<bool> sequence_finished(prompts.size(), false);
      
      Logger::info("[DEBUG] Starting token sampling for " + std::to_string(prompts.size()) + " sequences");
      
      for (size_t i = 0; i < prompts.size(); ++i) {
        Logger::info("[DEBUG] Sampling token for sequence " + std::to_string(i));
        
        // Safety check before sampling
        if (i >= batch_final_logits.size()) {
          Logger::error("[DEBUG] Index " + std::to_string(i) + " out of bounds for batch_final_logits (size: " + std::to_string(batch_final_logits.size()) + ")");
          goto fallback_sequential;
        }
        
        try {
          current_tokens[i] = sample_top_k_top_p_temperature(batch_final_logits[i], temperature, top_k, top_p, rng_);
          Logger::info("[DEBUG] Sampled token " + std::to_string(current_tokens[i]) + " for sequence " + std::to_string(i));
        } catch (const std::exception& e) {
          Logger::error("[DEBUG] Exception during sampling for sequence " + std::to_string(i) + ": " + std::string(e.what()));
          goto fallback_sequential;
        }
        
        all_generated_tokens[i].push_back(current_tokens[i]);
        sequence_positions[i] = prompt_lengths[i]; // Position for next token
        
        // Check for EOS
        if (current_tokens[i] == eos_token_id_) {
          sequence_finished[i] = true;
          Logger::info("[DEBUG] Sequence " + std::to_string(i) + " finished with EOS token");
        }
      }
      
      Logger::info("[DEBUG] Token sampling completed, starting generation loop");
      
      // Phase 2: Parallel Batch Generation
      for (int step = 1; step < steps; ++step) {
        Logger::info("[DEBUG] Starting generation step " + std::to_string(step));
        
        // Check if all sequences are finished
        bool all_finished = true;
        for (bool finished : sequence_finished) {
          if (!finished) {
            all_finished = false;
            break;
          }
        }
        if (all_finished) {
          Logger::info("[Batch Generate API] All sequences finished at step " + std::to_string(step));
          break;
        }
        
        Logger::info("[DEBUG] Collecting active sequences for step " + std::to_string(step));
        
        // Collect active sequences
        std::vector<int> active_tokens;
        std::vector<int> active_positions;
        std::vector<int> active_sequence_indices;
        std::vector<int> batch_to_original_seq_mapping; // Map batch index to original sequence index
        
        for (size_t i = 0; i < prompts.size(); ++i) {
          if (!sequence_finished[i]) {
            active_tokens.push_back(current_tokens[i]);
            active_positions.push_back(sequence_positions[i]);
            active_sequence_indices.push_back(active_tokens.size() - 1); // Use contiguous 0-based index
            batch_to_original_seq_mapping.push_back(i); // Remember original sequence index
            Logger::info("[DEBUG] Active sequence " + std::to_string(i) + " mapped to batch index " + std::to_string(active_tokens.size() - 1) + 
                        ": token=" + std::to_string(current_tokens[i]) + ", pos=" + std::to_string(sequence_positions[i]));
          }
        }
        
        if (active_tokens.empty()) {
          Logger::info("[DEBUG] No active tokens, breaking from generation loop");
          break;
        }
        
        Logger::info("[DEBUG] About to call batch_generation_parallel with " + std::to_string(active_tokens.size()) + " active sequences");
        
        // Process active sequences in parallel
        std::vector<std::vector<float>> step_logits;
        bool generation_success = batch_generation_parallel(active_tokens, active_positions, batch_to_original_seq_mapping, step_logits);        
        Logger::info("[DEBUG] batch_generation_parallel returned: " + std::string(generation_success ? "success" : "failure"));
        
        if (!generation_success || step_logits.size() != active_tokens.size()) {
          Logger::warning("[Batch Generate API] Parallel generation failed at step " + std::to_string(step) + 
                         ", falling back to sequential processing");
          goto fallback_sequential;
        }
        
        Logger::info("[DEBUG] Starting token sampling for step " + std::to_string(step));
        
        // Sample next tokens for active sequences
        for (size_t active_idx = 0; active_idx < active_tokens.size(); ++active_idx) {
          size_t original_seq_idx = batch_to_original_seq_mapping[active_idx]; // Use mapping to get original sequence index
          Logger::info("[DEBUG] Sampling for active_idx=" + std::to_string(active_idx) + ", original_seq_idx=" + std::to_string(original_seq_idx));
          
          // Safety checks
          if (active_idx >= step_logits.size()) {
            Logger::error("[DEBUG] active_idx " + std::to_string(active_idx) + " out of bounds for step_logits (size: " + std::to_string(step_logits.size()) + ")");
            goto fallback_sequential;
          }
          if (original_seq_idx >= prompts.size()) {
            Logger::error("[DEBUG] original_seq_idx " + std::to_string(original_seq_idx) + " out of bounds for prompts (size: " + std::to_string(prompts.size()) + ")");
            goto fallback_sequential;
          }
          
          try {
            int next_token = sample_top_k_top_p_temperature(step_logits[active_idx], temperature, top_k, top_p, rng_);
            Logger::info("[DEBUG] Sampled next token " + std::to_string(next_token) + " for original_seq_idx " + std::to_string(original_seq_idx));
            
            current_tokens[original_seq_idx] = next_token;
            all_generated_tokens[original_seq_idx].push_back(next_token);
            sequence_positions[original_seq_idx]++;
            
            // Check for EOS
            if (next_token == eos_token_id_) {
              sequence_finished[original_seq_idx] = true;
              Logger::info("[DEBUG] Sequence " + std::to_string(original_seq_idx) + " finished with EOS at step " + std::to_string(step));
            }
          } catch (const std::exception& e) {
            Logger::error("[DEBUG] Exception during sampling at step " + std::to_string(step) + " for original_seq_idx " + std::to_string(original_seq_idx) + ": " + std::string(e.what()));
            goto fallback_sequential;
          }
        }
        
        Logger::info("[DEBUG] Completed generation step " + std::to_string(step));
      }
      
      // Decode results for all sequences
      for (size_t i = 0; i < prompts.size(); ++i) {
        results[i] = tokenizer_->decode(all_generated_tokens[i], true);
      }
      
      Logger::info("[Batch Generate API] Parallel batch processing completed successfully");
    } else {
      Logger::warning("[Batch Generate API] Batch prefill failed, falling back to sequential processing");
      goto fallback_sequential;
    }
  } else {
    fallback_sequential:
    Logger::info("[Batch Generate API] Using sequential processing");
    
    for (size_t i = 0; i < prompts.size(); ++i) {
      Logger::info("[Batch Generate API] Processing prompt " + std::to_string(i + 1) + 
                   "/" + std::to_string(prompts.size()));
      
      // Reset to single-sequence mode for this prompt
      kv_cache_.seq_len = 0;
      kv_token_history_.clear();  // the cache no longer holds the last generate()'s tokens
      
      // Use existing single-sequence generation logic
      std::string result = generate(prompts[i], steps, temperature, top_k, top_p, 
                                   system_prompt_arg, apply_q_a_format_cli_hint);
      results[i] = result;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double time_taken_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  
  std::ostringstream time_ss;
  time_ss << std::fixed << std::setprecision(4) << time_taken_ms;
  Logger::info("[Batch Generate API] Total batch processing time: " + time_ss.str() + " ms for " + 
               std::to_string(prompts.size()) + " prompts");

  return results;
}

bool TinyLlamaSession::batch_prefill_parallel(const std::vector<std::vector<int>>& all_tokens,
                                              const std::vector<int>& prompt_lengths,
                                              std::vector<std::vector<float>>& batch_final_logits) {
  Logger::info("[EMERGENCY_DEBUG] batch_prefill_parallel function entry - FIRST LINE");
  Logger::info("[DEBUG] Entering batch_prefill_parallel");
  // Calculate total tokens across all prompts for batch prefill
  int total_tokens_across_all_prompts = 0;
  for (int len : prompt_lengths) {
    total_tokens_across_all_prompts += len;
  }
  
  if (total_tokens_across_all_prompts == 0) {
    Logger::error("[Batch Prefill] No tokens to process in batch prefill.");
    return false;
  }

  Logger::info("[Batch Prefill] Processing " + std::to_string(all_tokens.size()) + 
               " sequences with total " + std::to_string(total_tokens_across_all_prompts) + " tokens");

  // Process all tokens for all sequences in batch
  Logger::info("[Batch Prefill] Preparing batch embeddings for " + 
               std::to_string(total_tokens_across_all_prompts) + " tokens");
  
  // Calculate required memory
  size_t required_memory_bytes = static_cast<size_t>(total_tokens_across_all_prompts) * config_.hidden_size * sizeof(float);
  Logger::info("[DEBUG] About to allocate " + std::to_string(required_memory_bytes) + " bytes (" + 
               std::to_string(required_memory_bytes / (1024*1024)) + " MB) for batch embeddings");
  
  std::vector<float> batch_embeddings(total_tokens_across_all_prompts * config_.hidden_size);
  Logger::info("[DEBUG] batch_embeddings allocation completed successfully");
  
  int token_offset = 0;
  
  // Add detailed logging for token processing
  Logger::info("[DEBUG] Starting token embedding processing for " + std::to_string(all_tokens.size()) + " sequences");
  
  for (size_t seq_idx = 0; seq_idx < all_tokens.size(); ++seq_idx) {
    Logger::info("[DEBUG] Processing sequence " + std::to_string(seq_idx) + " with " + std::to_string(prompt_lengths[seq_idx]) + " tokens");
    
    // Log first few tokens of this sequence
    std::string token_ids_str = "Token IDs: ";
    for (int i = 0; i < std::min(5, prompt_lengths[seq_idx]); ++i) {
      token_ids_str += std::to_string(all_tokens[seq_idx][i]) + " ";
    }
    if (prompt_lengths[seq_idx] > 5) token_ids_str += "...";
    Logger::info("[DEBUG] Sequence " + std::to_string(seq_idx) + " " + token_ids_str);
    
    for (int token_idx = 0; token_idx < prompt_lengths[seq_idx]; ++token_idx) {
      int current_token_id = all_tokens[seq_idx][token_idx];
      
      // Log token placement in batch
      if (seq_idx < 2 && token_idx < 3) { // Only log first few tokens of first two sequences
        Logger::info("[DEBUG] Placing token " + std::to_string(current_token_id) + 
                    " from seq " + std::to_string(seq_idx) + " pos " + std::to_string(token_idx) + 
                    " at batch offset " + std::to_string(token_offset));
      }
      
      std::vector<float> token_embedding = model_->lookup_embedding(current_token_id);
      if (token_embedding.empty() || token_embedding.size() != static_cast<size_t>(config_.hidden_size)) {
        Logger::error("[Batch Prefill] Embedding lookup failed for token " + 
                     std::to_string(current_token_id) + 
                     " in sequence " + std::to_string(seq_idx));
        return false;
      }
      
      // Ensure we don't write beyond bounds
      size_t target_offset = token_offset * config_.hidden_size;
      if (target_offset + config_.hidden_size > batch_embeddings.size()) {
        Logger::error("[Batch Prefill] Buffer overflow detected at token offset " + std::to_string(token_offset));
        return false;
      }
      
      std::copy(token_embedding.begin(), token_embedding.end(), 
               batch_embeddings.begin() + target_offset);
      token_offset++;
    }
    
    Logger::info("[DEBUG] Sequence " + std::to_string(seq_idx) + " complete. Next token_offset: " + std::to_string(token_offset));
  }

  // Process CPU layers if any
  std::vector<float> cpu_processed_embeddings;
  if (config_.num_cpu_offload_layers > 0) {
    Logger::info("[Batch Prefill] Processing " + std::to_string(config_.num_cpu_offload_layers) + 
                 " CPU layers for batch prefill");
    cpu_processed_embeddings = model_->forward_cpu_batch(batch_embeddings, 
                                                        total_tokens_across_all_prompts, 
                                                        config_.num_cpu_offload_layers, 
                                                        0, &kv_cache_,
                                                        prompt_lengths);

    if (cpu_processed_embeddings.empty()) {
      Logger::error("[Batch Prefill] CPU batch processing failed.");
      return false;
    }
  } else {
    cpu_processed_embeddings = batch_embeddings;
  }

  // Process GPU layers if any
  std::vector<float> final_batch_logits;
  
  if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
    // All CPU - get logits from CPU
    Logger::info("[Batch Prefill] All layers on CPU, computing logits");
    final_batch_logits = model_->forward_cpu_logits_batch(cpu_processed_embeddings, 
                                                         total_tokens_across_all_prompts);
  } else {
    // GPU layers exist - transfer to GPU and process
#ifdef HAS_CUDA
    Logger::info("[Batch Prefill] Processing GPU layers for batch prefill");
    
    Logger::info("[DEBUG] About to allocate GPU memory for batch prefill");
    float* d_batch_embeddings = nullptr;
    size_t batch_size_bytes = cpu_processed_embeddings.size() * sizeof(float);
    Logger::info("[DEBUG] GPU allocation size: " + std::to_string(batch_size_bytes) + " bytes (" + 
                 std::to_string(batch_size_bytes / (1024*1024)) + " MB)");
    
    Logger::info("[DEBUG] Calling cudaMalloc...");
    gpuErrchk(cudaMalloc(&d_batch_embeddings, batch_size_bytes));
    Logger::info("[DEBUG] cudaMalloc completed successfully");
    
    Logger::info("[DEBUG] Calling cudaMemcpy host to device...");
    gpuErrchk(cudaMemcpy(d_batch_embeddings, cpu_processed_embeddings.data(), 
                        batch_size_bytes, cudaMemcpyHostToDevice));
    Logger::info("[DEBUG] cudaMemcpy completed successfully");

    // Call forward_device_batch_prefill ONCE with all the batch data
    Logger::info("[DEBUG] Calling forward_device_batch_prefill with " + std::to_string(total_tokens_across_all_prompts) + " total tokens");
    std::vector<float> all_batch_logits = model_->forward_device_batch_prefill(
      d_batch_embeddings, total_tokens_across_all_prompts, 0, &kv_cache_, 0);
    
    Logger::info("[DEBUG] forward_device_batch_prefill completed, returned " + std::to_string(all_batch_logits.size()) + " total logits");
    
    gpuErrchk(cudaFree(d_batch_embeddings));
    
    final_batch_logits = all_batch_logits;
#else
    Logger::error("[Batch Prefill] GPU processing requested but CUDA not available.");
      return false;
#endif
  }

  Logger::info("[Batch Prefill] Successfully processed batch prefill for " + 
               std::to_string(all_tokens.size()) + " sequences");
  
  Logger::info("[DEBUG] About to return from batch_prefill_parallel");
  Logger::info("[DEBUG] batch_final_logits.size()=" + std::to_string(batch_final_logits.size()));
  for (size_t i = 0; i < batch_final_logits.size() && i < 3; ++i) {
    Logger::info("[DEBUG] batch_final_logits[" + std::to_string(i) + "].size()=" + std::to_string(batch_final_logits[i].size()));
  }
  
  // Extract logits for the last token of each sequence
  batch_final_logits.clear();
  batch_final_logits.resize(all_tokens.size());
  
  if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
    // For CPU-only, extract last token logits from the flat array
    if (final_batch_logits.size() != static_cast<size_t>(total_tokens_across_all_prompts * config_.vocab_size)) {
      Logger::error("[Batch Prefill] CPU logits size mismatch. Expected: " + 
                   std::to_string(total_tokens_across_all_prompts * config_.vocab_size) + 
                   ", got: " + std::to_string(final_batch_logits.size()));
      return false;
    }
    
    int token_offset = 0;
    for (size_t seq_idx = 0; seq_idx < all_tokens.size(); ++seq_idx) {
      int last_token_pos = token_offset + prompt_lengths[seq_idx] - 1;
      
      Logger::info("[DEBUG] Extracting logits for sequence " + std::to_string(seq_idx) + 
                  ": token_offset=" + std::to_string(token_offset) + 
                  ", prompt_length=" + std::to_string(prompt_lengths[seq_idx]) + 
                  ", last_token_pos=" + std::to_string(last_token_pos));
      
      batch_final_logits[seq_idx].resize(config_.vocab_size);
      
      // Bounds check before copying
      size_t src_start = last_token_pos * config_.vocab_size;
      size_t src_end = src_start + config_.vocab_size;
      if (src_end > final_batch_logits.size()) {
        Logger::error("[Batch Prefill] CPU logits bounds check failed for sequence " + std::to_string(seq_idx));
        return false;
      }
      
      std::copy(final_batch_logits.begin() + src_start,
               final_batch_logits.begin() + src_end,
               batch_final_logits[seq_idx].begin());
      
      // Log a few logit values for debugging
      if (seq_idx < 2) { // Only for first two sequences
        std::string logit_sample = "First 5 logits: ";
        for (int i = 0; i < 5 && i < config_.vocab_size; ++i) {
          logit_sample += std::to_string(batch_final_logits[seq_idx][i]) + " ";
        }
        Logger::info("[DEBUG] Sequence " + std::to_string(seq_idx) + " " + logit_sample);
      }
      
      token_offset += prompt_lengths[seq_idx];
    }
  } else {
    // For GPU, check if logits are for all tokens or just last tokens
    Logger::info("[DEBUG] GPU batch logits size: " + std::to_string(final_batch_logits.size()) + 
                 ", expected for all tokens: " + std::to_string(total_tokens_across_all_prompts * config_.vocab_size) +
                 ", expected for last tokens only: " + std::to_string(all_tokens.size() * config_.vocab_size));
    
    if (final_batch_logits.size() == static_cast<size_t>(total_tokens_across_all_prompts * config_.vocab_size)) {
      // GPU returned logits for all tokens, extract last token for each sequence
      Logger::info("[DEBUG] GPU returned logits for all tokens, extracting last token logits");
      int token_offset = 0;
      for (size_t seq_idx = 0; seq_idx < all_tokens.size(); ++seq_idx) {
        int last_token_pos = token_offset + prompt_lengths[seq_idx] - 1;
        
        Logger::info("[DEBUG] GPU: Extracting logits for sequence " + std::to_string(seq_idx) + 
                    ": token_offset=" + std::to_string(token_offset) + 
                    ", prompt_length=" + std::to_string(prompt_lengths[seq_idx]) + 
                    ", last_token_pos=" + std::to_string(last_token_pos));
        
        batch_final_logits[seq_idx].resize(config_.vocab_size);
        
        size_t src_start = last_token_pos * config_.vocab_size;
        size_t src_end = src_start + config_.vocab_size;
        if (src_end > final_batch_logits.size()) {
          Logger::error("[Batch Prefill] GPU logits bounds check failed for sequence " + std::to_string(seq_idx));
          return false;
        }
        
        std::copy(final_batch_logits.begin() + src_start,
                 final_batch_logits.begin() + src_end,
                 batch_final_logits[seq_idx].begin());
        
        // Log a few logit values for debugging
        if (seq_idx < 2) { // Only for first two sequences
          std::string logit_sample = "First 5 logits: ";
          for (int i = 0; i < 5 && i < config_.vocab_size; ++i) {
            logit_sample += std::to_string(batch_final_logits[seq_idx][i]) + " ";
          }
          Logger::info("[DEBUG] GPU Sequence " + std::to_string(seq_idx) + " " + logit_sample);
        }
        
        token_offset += prompt_lengths[seq_idx];
      }
    } else if (final_batch_logits.size() == static_cast<size_t>(all_tokens.size() * config_.vocab_size)) {
      // GPU returned logits for last tokens only
      Logger::info("[DEBUG] GPU returned logits for last tokens only");
      for (size_t seq_idx = 0; seq_idx < all_tokens.size(); ++seq_idx) {
        Logger::info("[DEBUG] GPU Last-Token-Only: Processing sequence " + std::to_string(seq_idx) + 
                    " at logit offset " + std::to_string(seq_idx * config_.vocab_size));
        
        batch_final_logits[seq_idx].resize(config_.vocab_size);
        
        size_t src_start = seq_idx * config_.vocab_size;
        size_t src_end = src_start + config_.vocab_size;
        if (src_end > final_batch_logits.size()) {
          Logger::error("[Batch Prefill] GPU logits bounds check failed for sequence " + std::to_string(seq_idx));
          return false;
        }
        
        std::copy(final_batch_logits.begin() + src_start,
                 final_batch_logits.begin() + src_end,
                 batch_final_logits[seq_idx].begin());
        
        // Log a few logit values for debugging
        if (seq_idx < 2) { // Only for first two sequences
          std::string logit_sample = "First 5 logits: ";
          for (int i = 0; i < 5 && i < config_.vocab_size; ++i) {
            logit_sample += std::to_string(batch_final_logits[seq_idx][i]) + " ";
          }
          Logger::info("[DEBUG] GPU Last-Token Sequence " + std::to_string(seq_idx) + " " + logit_sample);
        }
      }
    } else {
      Logger::error("[Batch Prefill] GPU logits size doesn't match expected patterns");
      return false;
    }
  }
  
  return true;
}

bool TinyLlamaSession::batch_generation_parallel(const std::vector<int>& current_tokens,
                                                 const std::vector<int>& token_positions,
                                                 const std::vector<int>& sequence_indices,
                                                 std::vector<std::vector<float>>& batch_logits) {
  Logger::info("[DEBUG] Entering batch_generation_parallel");
  
  int num_sequences = current_tokens.size();
  
  if (num_sequences == 0 || token_positions.size() != current_tokens.size()) {
    Logger::error("[Batch Generation] Invalid input sizes");
    return false;
  }

  Logger::info("[Batch Generation] Processing " + std::to_string(num_sequences) + 
               " sequences in parallel generation step");

  // Create batch embeddings for all sequences
  std::vector<float> batch_embeddings;
  batch_embeddings.reserve(num_sequences * config_.hidden_size);
  
  for (int i = 0; i < num_sequences; ++i) {
    std::vector<float> token_embedding = model_->lookup_embedding(current_tokens[i]);
    if (token_embedding.empty()) {
      Logger::error("[Batch Generation] Embedding lookup failed for token " + std::to_string(current_tokens[i]));
      return false;
    }
    batch_embeddings.insert(batch_embeddings.end(), token_embedding.begin(), token_embedding.end());
  }

  // Process through CPU layers if any
  if (config_.num_cpu_offload_layers > 0) {
    Logger::info("[Batch Generation] Processing " + std::to_string(config_.num_cpu_offload_layers) + 
                 " CPU layers for batch generation");
    
    std::vector<std::vector<float>> cpu_batch_logits = model_->forward_cpu_batch_generation(
      batch_embeddings, token_positions, sequence_indices, num_sequences, &kv_cache_);
    
    if (cpu_batch_logits.size() != static_cast<size_t>(num_sequences)) {
      Logger::error("[Batch Generation] CPU batch generation returned wrong number of results");
      return false;
    }

    // If all layers are on CPU, we're done
    if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
      batch_logits = cpu_batch_logits;
      Logger::info("[Batch Generation] All CPU layers processed, returning logits");
      return true;
    }

    // Convert CPU results back to batch embeddings for GPU processing
    batch_embeddings.clear();
    batch_embeddings.resize(num_sequences * config_.hidden_size);
    // Note: This would need the CPU layer output activations, not logits
    // For now, fall back to sequential processing if mixed CPU/GPU
    Logger::warning("[Batch Generation] Mixed CPU/GPU not yet implemented for batch generation");
    return false;
  }

  // GPU-only processing
  if (config_.num_cpu_offload_layers < config_.num_hidden_layers) {
#ifdef HAS_CUDA
    Logger::info("[Batch Generation] Processing GPU layers for batch generation");
    
    float* d_batch_embeddings = nullptr;
    size_t batch_size_bytes = batch_embeddings.size() * sizeof(float);
    
    gpuErrchk(cudaMalloc(&d_batch_embeddings, batch_size_bytes));
    gpuErrchk(cudaMemcpy(d_batch_embeddings, batch_embeddings.data(), 
                        batch_size_bytes, cudaMemcpyHostToDevice));

    std::vector<std::vector<float>> gpu_batch_logits = model_->forward_device_batch_generation(
      d_batch_embeddings, token_positions, sequence_indices, num_sequences, &kv_cache_, 0);
    
    gpuErrchk(cudaFree(d_batch_embeddings));
    
    if (gpu_batch_logits.size() != static_cast<size_t>(num_sequences)) {
      Logger::error("[Batch Generation] GPU batch generation returned wrong number of results");
      return false;
    }
    
    batch_logits = gpu_batch_logits;
    Logger::info("[Batch Generation] GPU batch generation completed successfully");
    return true;
#else
    Logger::error("[Batch Generation] GPU processing requested but CUDA not available.");
    return false;
#endif
  }
  
  Logger::error("[Batch Generation] No valid processing path found");
  return false;
}

}  // namespace tinyllama