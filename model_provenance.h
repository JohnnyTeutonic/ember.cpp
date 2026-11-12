#pragma once
// What a running server can say about what it serves, for evaluation receipts (GET /health's "provenance";
// docs/USAGE.md).
//
// What the weights digest means: the SHA-256 of the .gguf file's bytes as hashed at start, before the load.
// On POSIX the hash reads through one descriptor whose stamp (device, inode, size, modification time) must
// not change while hashing; after the load the loader's own descriptor and the path must both still carry
// that stamp, or the server refuses to start; and every /health fstat()s the loader's descriptor and reports
// "modified_since_load" (no digest) once the mapped file has changed. So the digest names the file the loader
// mapped, unchanged since it was hashed, as far as those stamps can tell: a rewrite that keeps size,
// modification time and inode is not seen. Under mmap most bytes are read lazily during inference, which is
// why the check repeats. On Windows the stamp has no inode and there is no per-request check (the loader's
// share mode blocks writers while the file is mapped).
//
// The build half is compiled in (git commit and dirty state from tools/ember_build_info.cmake; backend,
// configuration, compiler); the running executable is hashed (/proc/self/exe on Linux, before the weights),
// so a dirty or unknown commit is still identified. EMBER_* settings that change computation are reported
// with their values, side files they name (EMBER_IR, EMBER_DRAFT_TABLE, EMBER_TEST_SCRIPTED_REPLIES) by
// SHA-256, any other EMBER_* variable by name only.
//
// Trust: this is the server's own report. It guards against operator error (the wrong file, a stale server,
// a server restarted on other weights, settings or another build), not against a server built to lie.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ember {

// A file's identity without reading it.
struct FileStamp {
  bool ok = false;
  uint64_t size = 0;
  int64_t mtime_ns = 0;
  uint64_t dev = 0;
  uint64_t ino = 0;
};
bool same_stamp(const FileStamp& a, const FileStamp& b);
FileStamp file_stamp(const std::string& path);
// fstat() of an open descriptor (POSIX); !ok on Windows or for fd < 0.
FileStamp file_stamp_fd(int fd);

struct FileDigest {
  bool ok = false;
  std::string sha256;  // 64 lowercase hex digits when ok
  uint64_t bytes = 0;
  FileStamp stamp;     // POSIX: the descriptor's stamp, equal before and after hashing (else !ok)
  std::string error;   // why not, when !ok
};
// SHA-256 of a file's bytes, read in 4 MiB pieces. POSIX: through one descriptor, refused when its stamp
// changes while hashing.
FileDigest sha256_file(const std::string& path);

struct BuildInfo {
  std::string git_commit;  // hex object id, or "unknown" (a tree without git, or git refused it)
  std::string git_dirty;   // "true", "false" or "unknown" (tracked files only)
  std::string backend;     // "cuda" or "cpu"
  std::string config;      // CMake configuration (Release, Debug, ...), "" when none was set
  std::string compiler;    // CMake's compiler id and version
};
BuildInfo build_info();

// The running executable: its path ("" when the platform does not say; Linux may append " (deleted)") and
// its digest (Linux: /proc/self/exe, the running image even when the path was replaced or deleted).
struct ExecutableDigest {
  std::string path;
  FileDigest digest;
};
ExecutableDigest executable_digest();

// The EMBER_* environment: every name set; values for the settings ember reads (kValueSettings in the
// .cpp; values longer than 256 bytes are cut); SHA-256 of the side files named by EMBER_IR,
// EMBER_DRAFT_TABLE and EMBER_TEST_SCRIPTED_REPLIES. Any other EMBER_* variable: name only.
struct EnvRecord {
  std::vector<std::string> names;
  std::vector<std::pair<std::string, std::string>> values;
  std::vector<std::pair<std::string, FileDigest>> files;
};
EnvRecord ember_env_record();

// Weights the server can hash: a regular .gguf file (self-contained: weights, tokenizer and chat template).
// A SafeTensors directory or file also reads config.json and a tokenizer from elsewhere (the tokenizer from
// the working directory), so it is reported as "unsupported_format" rather than as a partial digest.
bool digestible_weights(const std::string& model_path);

// 32 hex digits, new for each server process (random_device mixed with the time and the process id): a
// restart shows as a changed /health.
std::string new_instance_id();

}  // namespace ember
