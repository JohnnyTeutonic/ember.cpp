#pragma once
// What a running server can say about what it serves, for evaluation receipts (GET /health's "provenance";
// docs/USAGE.md). The server hashes the weights file once at start, before it loads it, and refuses to start
// when the file's stamp (size, modification time and, on POSIX, device and inode) differs after the load:
// the digest then names the bytes the load read. The build half is compiled in (the git commit and whether
// the tree had uncommitted changes, from tools/ember_build_info.cmake; the backend, configuration and
// compiler), and the running executable is hashed as well, so a dirty or unknown commit is still identified.
//
// Trust: this is the server's own report. It guards against operator error (the wrong file, a stale server,
// a server restarted on other weights or another build) and not against a server built to lie.
#include <cstdint>
#include <string>
#include <vector>

namespace ember {

// A file's identity without reading it: equal stamps before and after a load mean the file was not replaced
// or rewritten in between (a rewrite that keeps size and modification time and, on POSIX, the inode is not
// detected).
struct FileStamp {
  bool ok = false;
  uint64_t size = 0;
  int64_t mtime_ns = 0;
  uint64_t dev = 0;
  uint64_t ino = 0;
};
bool same_stamp(const FileStamp& a, const FileStamp& b);
FileStamp file_stamp(const std::string& path);

struct FileDigest {
  bool ok = false;
  std::string sha256;  // 64 lowercase hex digits when ok
  uint64_t bytes = 0;
  std::string error;   // why not, when !ok
};
// SHA-256 of a file's bytes, read in 4 MiB pieces.
FileDigest sha256_file(const std::string& path);

struct BuildInfo {
  std::string git_commit;  // 40 hex digits, or "unknown" (a tree without git)
  std::string git_dirty;   // "true", "false" or "unknown"
  std::string backend;     // "cuda" or "cpu"
  std::string config;      // CMake configuration (Release, Debug, ...), "" when none was set
  std::string compiler;    // CMake's compiler id and version
};
BuildInfo build_info();

// The running executable's path ("" when the platform does not say).
std::string executable_path();

// Names (not values) of the EMBER_* environment variables set in this process, sorted: settings that change
// what the server runs (EMBER_IR, EMBER_INTERPRETER, EMBER_TEST_SCRIPTED_REPLIES, ...).
std::vector<std::string> ember_env_names();

// Weights the server can hash: a regular .gguf file (self-contained: weights, tokenizer and chat template).
// A SafeTensors directory or file also reads config.json and a tokenizer from elsewhere, so it is reported
// as "unsupported_format" rather than as a partial digest.
bool digestible_weights(const std::string& model_path);

// 32 random hex digits, new for each server process: a restart shows as a changed /health.
std::string new_instance_id();

}  // namespace ember
