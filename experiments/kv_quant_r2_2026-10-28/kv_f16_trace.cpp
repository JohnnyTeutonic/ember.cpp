// Diagnostic-only CPU/CUDA teacher forcing. All input IDs are explicitly frozen
// outside this tool.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

#include "gguf_parser.h"
#include "graph_ir.h"
#include "ir_digest.h"
using json = nlohmann::json;
namespace fs = std::filesystem;
std::string contents(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}
void atomic_json(const fs::path& path, const json& value) {
  auto temp = path;
  temp += ".partial";
  std::ofstream f(temp);
  f << value.dump(1) << "\n";
  f.close();
  if (!f) throw std::runtime_error("JSON write failed");
  fs::rename(temp, path);
}
json inventory(const fs::path& folder) {
  json result = json::object();
  for (const auto& entry : fs::recursive_directory_iterator(folder))
    if (entry.is_regular_file()) {
      const auto relative = fs::relative(entry.path(), folder).generic_string();
      if (relative == "case.json") continue;
      result[relative] = {
          {"bytes", entry.file_size()},
          {"sha256", ember_ir::sha256_hex_bytes(contents(entry.path()))}};
    }
  return result;
}
void trace_path(const fs::path& path) {
#ifdef _WIN32
  if (_putenv_s("EMBER_KV_TRACE_DIR", path.string().c_str()))
    throw std::runtime_error("trace env failed");
#else
  if (setenv("EMBER_KV_TRACE_DIR", path.string().c_str(), 1))
    throw std::runtime_error("trace env failed");
#endif
}
void logits(std::ofstream& f, const std::vector<float>& data) {
  for (float v : data)
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite diagnostic logits");
  f.write(reinterpret_cast<const char*>(data.data()),
          data.size() * sizeof(float));
  if (!f) throw std::runtime_error("logits write failed");
}
int main(int argc, char** argv) {
#ifndef EMBER_KV_DIAGNOSTICS
  std::cerr << "diagnostic compile flag is required\n";
  return 2;
#endif
  try {
    if (argc != 5)
      throw std::runtime_error(
          "usage: kv_f16_trace MODEL.gguf IDS.json OUT cpu|cuda");
    const std::string backend = argv[4];
    if (backend != "cpu" && backend != "cuda")
      throw std::runtime_error("invalid backend");
    const char* graph = std::getenv("EMBER_IR_GRAPH");
    if (!graph || std::string(graph) != "0")
      throw std::runtime_error(
          "trace requires EMBER_IR_GRAPH=0 before any forward");
    const char* model_sha = std::getenv("EMBER_KV_TRACE_MODEL_SHA256");
    if (!model_sha || std::string(model_sha).size() != 64 ||
        std::string(model_sha).find_first_not_of("0123456789abcdef") !=
            std::string::npos)
      throw std::runtime_error("externally verified model SHA256 is required");
    const auto ids = json::parse(contents(argv[2]));
    if (!ids.is_array() || ids.empty()) throw std::runtime_error("empty IDs");
    const auto gguf = load_gguf_meta(argv[1], true);
    const auto ir = ember_ir::find_ir(gguf);
    if (ir.empty()) throw std::runtime_error("missing IR");
    ember_ir::Interpreter model(ir, gguf, backend == "cuda");
    if (model.on_gpu() != (backend == "cuda"))
      throw std::runtime_error("actual backend differs from requested backend");
    const fs::path out = fs::absolute(argv[3]);
    fs::create_directories(out);
    const json fixed = {{"schema", 1},
                        {"backend", backend},
                        {"actual_gpu", model.on_gpu()},
                        {"format", model.kv_format()},
                        {"ids", ids},
                        {"vocab", model.vocab()},
                        {"ir_sha256", ember_ir::sha256_hex_bytes(ir)},
                        {"model_sha256", model_sha},
                        {"graph", false},
                        {"diagnostic_only", true}};
    if (fs::exists(out / "manifest.json")) {
      if (json::parse(contents(out / "manifest.json")) != fixed)
        throw std::runtime_error("manifest drift");
    } else
      atomic_json(out / "manifest.json", fixed);
    if (!fs::exists(out / "ir.json")) {
      std::ofstream f(out / "ir.json");
      f << ir << "\n";
    }
    int completed = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      const auto sequence = ids[i].get<std::vector<int>>();
      if (sequence.empty()) throw std::runtime_error("empty sequence");
      for (int token : sequence)
        if (token < 0 || token >= model.vocab())
          throw std::runtime_error("token outside vocabulary");
      std::ostringstream name;
      name << "case_" << std::setfill('0') << std::setw(4) << i;
      const auto folder = out / name.str();
      if (fs::exists(folder / "case.json")) {
        const auto saved = json::parse(contents(folder / "case.json"));
        if (saved.at("ids") != sequence ||
            saved.at("files") != inventory(folder))
          throw std::runtime_error("completed case drift");
        ++completed;
        continue;
      }
      if (fs::exists(folder))
        throw std::runtime_error(
            "incomplete case retained; explicit operational review required");
      fs::create_directory(folder);
      std::ofstream prefill(folder / "prefill.f32", std::ios::binary),
          decode(folder / "decode.f32", std::ios::binary);
      trace_path(folder / "prefill");
      const auto p = model.forward(sequence, 0, true);
      if (p.size() != sequence.size() * model.vocab())
        throw std::runtime_error("prefill shape mismatch");
      logits(prefill, p);
      for (size_t t = 0; t < sequence.size(); ++t) {
        std::ostringstream part;
        part << "decode_" << std::setfill('0') << std::setw(4) << t;
        trace_path(folder / part.str());
        const auto d = model.forward({sequence[t]}, static_cast<int>(t), false);
        if (d.size() != size_t(model.vocab()))
          throw std::runtime_error("decode shape mismatch");
        logits(decode, d);
      }
      prefill.close();
      decode.close();
      if (!prefill || !decode) throw std::runtime_error("logits flush failed");
      atomic_json(folder / "case.json",
                  {{"ids", sequence}, {"files", inventory(folder)}});
      ++completed;
      atomic_json(out / "progress.json", {{"complete", false},
                                          {"completed", completed},
                                          {"total", ids.size()}});
    }
    atomic_json(
        out / "progress.json",
        {{"complete", true}, {"completed", completed}, {"total", ids.size()}});
    std::cout << json{{"diagnostic_only", true}, {"complete", true},
                      {"backend", backend},      {"format", model.kv_format()},
                      {"prompts", completed},    {"vocab", model.vocab()}}
                     .dump()
              << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}