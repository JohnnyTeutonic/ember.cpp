// model_provenance.h and the streaming SHA-256 behind it (ir_digest.h): FIPS 180-4 vectors fed in awkward
// pieces, agreement with sha256_hex_bytes at every length across the padding boundaries, file digests, file
// stamps that change when a file is rewritten or replaced (by path and through a held descriptor), which
// model paths get a digest, the compiled-in build record, the executable digest, and the EMBER_* record
// (values, side-file digests, other names without values). Exit 1 when any check fails.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "ir_digest.h"
#include "model_provenance.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

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

#ifndef _WIN32
  {
    // The digest carries the stamp of the descriptor it read; a held descriptor sees an in-place rewrite
    // (the server's per-/health check), while a rename replacement shows only by path (its load-time check).
    write_file(a, bytes);
    const ember::FileDigest dd = ember::sha256_file(a.string());
    check(dd.ok && ember::same_stamp(dd.stamp, ember::file_stamp(a.string())), "the digest's stamp is the file's");
    const int fd = ::open(a.string().c_str(), O_RDONLY);
    check(ember::same_stamp(dd.stamp, ember::file_stamp_fd(fd)), "a held descriptor starts with the hashed stamp");
    {
      std::ofstream app(a, std::ios::binary | std::ios::app);
      app << "tail";
    }
    check(!ember::same_stamp(dd.stamp, ember::file_stamp_fd(fd)), "an in-place append shows on the held descriptor");
    const ember::FileStamp held = ember::file_stamp_fd(fd);
    const fs::path c = dir / "c.gguf";
    write_file(c, bytes);
    fs::rename(c, a);
    check(ember::same_stamp(held, ember::file_stamp_fd(fd)), "a rename replacement leaves the held descriptor alone");
    check(!ember::same_stamp(held, ember::file_stamp(a.string())), "and shows by path");
    ::close(fd);
    check(!ember::file_stamp_fd(-1).ok, "no descriptor, no stamp");
  }
#endif

  const ember::BuildInfo bi = ember::build_info();
  check(bi.git_commit == "unknown" || is_hex(bi.git_commit, 40) || is_hex(bi.git_commit, 64),
        "git_commit is a hex object id (SHA-1 or SHA-256) or unknown: " + bi.git_commit);
  check(bi.git_dirty == "true" || bi.git_dirty == "false" || bi.git_dirty == "unknown", "git_dirty: " + bi.git_dirty);
  check(bi.backend == "cpu" || bi.backend == "cuda", "backend: " + bi.backend);
  check(!bi.compiler.empty(), "compiler recorded");
  std::cout << "build: commit " << bi.git_commit << " dirty " << bi.git_dirty << " backend " << bi.backend
            << " config '" << bi.config << "' compiler " << bi.compiler << "\n";

  const ember::ExecutableDigest exe = ember::executable_digest();
#ifdef __linux__
  check(!exe.path.empty() && exe.digest.ok && exe.digest.sha256 == ember::sha256_file(exe.path).sha256,
        "the running executable is found and hashed (/proc/self/exe = its path)");
#endif

  // Settings: values for those ember reads, side files by digest, anything else by name only.
  const fs::path side = dir / "side.ir.json";
  write_file(side, "{\"ir\": 1}");
  auto set = [](const char* k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
#else
    setenv(k, v.c_str(), 1);
#endif
  };
  set("EMBER_PROVENANCE_TEST", "secret-value");
  set("EMBER_KV", "q8_0");
  set("EMBER_DEQUANT_AT_LOAD", "0");
  set("EMBER_IR", side.string());
  set("EMBER_TEST_SCRIPTED_REPLIES", (dir / "absent.json").string());
  const ember::EnvRecord env = ember::ember_env_record();
  bool named = false, leaked = false;
  for (const auto& n : env.names) {
    named = named || n == "EMBER_PROVENANCE_TEST";
    leaked = leaked || n.find("secret") != std::string::npos || n.find('=') != std::string::npos;
  }
  for (const auto& [k, v] : env.values) leaked = leaked || k == "EMBER_PROVENANCE_TEST" || v.find("secret") != std::string::npos;
  check(named && !leaked, "an unknown EMBER_* variable is listed by name, without its value");
  auto value = [&](const std::string& k) {
    for (const auto& [n, v] : env.values)
      if (n == k) return v;
    return std::string("<absent>");
  };
  check(value("EMBER_KV") == "q8_0" && value("EMBER_DEQUANT_AT_LOAD") == "0", "computation settings carry values");
  check(value("EMBER_IR") == "<absent>", "a side-file setting's path is not reported as a value");
  bool ir_ok = false, scripted_err = false;
  for (const auto& [n, fd] : env.files) {
    ir_ok = ir_ok || (n == "EMBER_IR" && fd.ok && fd.sha256 == ember_ir::sha256_hex_bytes("{\"ir\": 1}"));
    scripted_err = scripted_err || (n == "EMBER_TEST_SCRIPTED_REPLIES" && !fd.ok && !fd.error.empty());
  }
  check(ir_ok, "EMBER_IR's side file is reported by its SHA-256");
  check(scripted_err, "an unreadable side file is reported with an error, not skipped");

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
