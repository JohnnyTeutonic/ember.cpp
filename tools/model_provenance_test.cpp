// model_provenance.h and the streaming SHA-256 behind it (ir_digest.h): FIPS 180-4 vectors fed in awkward
// pieces, agreement with sha256_hex_bytes at every length across the padding boundaries, file digests, file
// stamps that change when a file is rewritten or replaced, which model paths get a digest, the compiled-in
// build record, and the EMBER_* names without their values. Exit 1 on the first failure.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "ir_digest.h"
#include "model_provenance.h"

namespace fs = std::filesystem;
static int failures = 0;

static void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::cerr << "FAIL: " << what << "\n";
  }
}

static std::string streamed(const std::string& msg, size_t piece) {
  ember_ir::Sha256 h;
  for (size_t off = 0; off < msg.size(); off += piece) h.update(msg.data() + off, std::min(piece, msg.size() - off));
  return h.hex();
}

static bool is_hex(const std::string& s, size_t n) {
  return s.size() == n && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}

static void write_file(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

int main() {
  // FIPS 180-4 / NIST example vectors.
  const struct { std::string msg, hex; } vectors[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
      {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
  };
  for (const auto& v : vectors)
    for (size_t piece : {size_t(1), size_t(3), size_t(63), size_t(64), size_t(65), size_t(4096), size_t(1) << 20})
      check(streamed(v.msg, piece) == v.hex, "vector of length " + std::to_string(v.msg.size()) + " in pieces of " +
                                                std::to_string(piece));
  std::string msg;
  for (int n = 0; n <= 200; ++n) {
    check(streamed(msg, 7) == ember_ir::sha256_hex_bytes(msg), "streamed = one-shot at length " + std::to_string(n));
    msg += static_cast<char>((n * 131 + 7) & 0xFF);
  }

  const fs::path dir = fs::temp_directory_path() / ("ember_provenance_test_" + ember::new_instance_id());
  fs::create_directories(dir);
  const fs::path a = dir / "a.gguf";
  std::string bytes(5u << 20, '\0');  // over one 4 MiB read
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<char>((i * 2654435761u) >> 13);
  write_file(a, bytes);

  const ember::FileDigest d = ember::sha256_file(a.string());
  check(d.ok && d.bytes == bytes.size() && d.sha256 == ember_ir::sha256_hex_bytes(bytes), "sha256_file of a 5 MiB file");
  const ember::FileDigest missing = ember::sha256_file((dir / "absent.gguf").string());
  check(!missing.ok && missing.sha256.empty() && !missing.error.empty(), "sha256_file of a missing file refuses");

  const ember::FileStamp s0 = ember::file_stamp(a.string());
  check(s0.ok && s0.size == bytes.size(), "file_stamp reads the size");
  check(ember::same_stamp(s0, ember::file_stamp(a.string())), "an untouched file keeps its stamp");
  check(!ember::same_stamp(s0, ember::file_stamp((dir / "absent.gguf").string())), "a missing file never matches");
  write_file(a, bytes + "x");
  check(!ember::same_stamp(s0, ember::file_stamp(a.string())), "a rewrite that changes the size changes the stamp");
#ifndef _WIN32
  // Replaced by rename with the same size and the same modification time: the inode tells.
  write_file(a, bytes);
  const ember::FileStamp s1 = ember::file_stamp(a.string());
  const fs::path b = dir / "b.gguf";
  write_file(b, bytes);
  fs::last_write_time(b, fs::last_write_time(a));
  fs::rename(b, a);
  const ember::FileStamp s2 = ember::file_stamp(a.string());
  check(s2.size == s1.size && s2.mtime_ns == s1.mtime_ns, "the replacement keeps size and modification time");
  check(!ember::same_stamp(s1, s2), "a file replaced by rename changes the stamp (inode)");
#endif

  check(ember::digestible_weights(a.string()), "a .gguf file gets a digest");
  const fs::path upper = dir / "B.GGUF";
  write_file(upper, "x");
  check(ember::digestible_weights(upper.string()), "the extension is matched without case");
  const fs::path st = dir / "model.safetensors";
  write_file(st, "x");
  check(!ember::digestible_weights(st.string()), "a .safetensors file is not reported as a full digest");
  check(!ember::digestible_weights(dir.string()), "a model directory is not reported as a full digest");
  check(!ember::digestible_weights((dir / "absent.gguf").string()), "a missing file gets no digest");

  const ember::BuildInfo bi = ember::build_info();
  check(bi.git_commit == "unknown" || is_hex(bi.git_commit, 40), "git_commit is 40 hex digits or unknown: " + bi.git_commit);
  check(bi.git_dirty == "true" || bi.git_dirty == "false" || bi.git_dirty == "unknown", "git_dirty: " + bi.git_dirty);
  check(bi.backend == "cpu" || bi.backend == "cuda", "backend: " + bi.backend);
  check(!bi.compiler.empty(), "compiler recorded");
  std::cout << "build: commit " << bi.git_commit << " dirty " << bi.git_dirty << " backend " << bi.backend
            << " config '" << bi.config << "' compiler " << bi.compiler << "\n";

  const std::string exe = ember::executable_path();
#ifdef __linux__
  check(!exe.empty() && ember::sha256_file(exe).ok, "the running executable is found and hashed");
#endif

#ifdef _WIN32
  _putenv_s("EMBER_PROVENANCE_TEST", "secret-value");
#else
  setenv("EMBER_PROVENANCE_TEST", "secret-value", 1);
#endif
  bool named = false, leaked = false;
  for (const auto& n : ember::ember_env_names()) {
    named = named || n == "EMBER_PROVENANCE_TEST";
    leaked = leaked || n.find("secret") != std::string::npos || n.find('=') != std::string::npos;
  }
  check(named && !leaked, "EMBER_* variables are listed by name, without values");

  const std::string i1 = ember::new_instance_id(), i2 = ember::new_instance_id();
  check(is_hex(i1, 32) && i1 != i2, "instance ids are 32 hex digits and differ");

  std::error_code ec;
  fs::remove_all(dir, ec);
  if (failures) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "model_provenance_test: all checks passed\n";
  return 0;
}
