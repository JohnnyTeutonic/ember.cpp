// Decode route evidence for the E10 PARTS4 gate (experiments/spec_r1_2026-10-28/parts4_route_20261102/).
//
// Off unless EMBER_ROUTE_TRACE names a file. The timing legs leave it unset, so there every added statement is
// behind the cached flag below. Tracing never changes an arithmetic operation: it reads the record and the logits
// row the decode step already produced, and before each replay fills the pick buffers with a sentinel (all bytes
// 0xFF) that the selector kernels overwrite completely. A record still equal to the sentinel after a replay shows
// that the replayed graph did not run its selector.
//
// EMBER_ROUTE_TRACE_FORCE=undecidable (honoured only while tracing) makes forward_device_graph hand api.cpp a copy
// of each record with its nonfinite flag set, so the greedy rule returns -1, the branch a near tie or a nonfinite
// row takes, and the host samples the row the same replay produced. The route record keeps the true record. It is
// fault injection for the route-proof legs, labelled in every event it affects; api.cpp's dispatch is unchanged.
#ifndef EMBER_DECODE_ROUTE_TRACE_H
#define EMBER_DECODE_ROUTE_TRACE_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace ember {
namespace route_trace {

inline const char* path() {
  static const char* p = [] {
    const char* v = std::getenv("EMBER_ROUTE_TRACE");
    return v && *v ? v : nullptr;
  }();
  return p;
}

inline bool on() { return path() != nullptr; }

inline bool force_undecidable() {
  static const bool f = [] {
    const char* v = std::getenv("EMBER_ROUTE_TRACE_FORCE");
    return on() && v && std::strcmp(v, "undecidable") == 0;
  }();
  return f;
}

// One JSON object per line, flushed at once so a killed server keeps every finished event.
inline void emit(const std::string& line) {
  static std::mutex m;
  static FILE* f = nullptr;
  std::lock_guard<std::mutex> g(m);
  if (!f) f = std::fopen(path(), "a");
  if (!f) return;  // the checker treats a missing or short trace as no route evidence
  std::fputs(line.c_str(), f);
  std::fputc('\n', f);
  std::fflush(f);
}

// ember_verify::RowPick's layout (verify_kernels.h), host-only so the checks below build without CUDA.
struct Record {
  float max;
  float second;
  int first;
  int last;
  int nan;
};
static_assert(sizeof(Record) == 20, "RowPick wire layout");

template <class P>
inline Record record_of(const P& p) {
  static_assert(sizeof(P) == sizeof(Record), "RowPick wire layout");
  Record r;
  std::memcpy(&r, &p, sizeof r);
  return r;
}

inline uint32_t bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return u;
}

// Every byte still 0xFF: the selector cannot write this (its nan field is 0 or 1).
inline bool unwritten(const Record& r) {
  unsigned char b[sizeof(Record)];
  std::memcpy(b, &r, sizeof b);
  return std::all_of(b, b + sizeof b, [](unsigned char c) { return c == 0xFF; });
}

// The record of row[begin, end): the rule of select_rows_kernel (verify_kernels.cu) over the whole row and of
// partition_rows_kernel (row_pick_partitioned.cuh) over one partition.
inline Record partition_oracle(const float* row, long long begin, long long end) {
  float mx = -INFINITY;
  int nan = 0;
  for (long long i = begin; i < end; ++i) {
    const float v = row[i];
    if (v != v) nan = 1;
    else mx = std::fmax(mx, v);
  }
  Record r{mx, -INFINITY, INT_MAX, -1, nan};
  for (long long i = begin; i < end; ++i) {
    const float v = row[i];
    if (v == mx) {
      r.first = std::min(r.first, static_cast<int>(i));
      r.last = std::max(r.last, static_cast<int>(i));
    } else if (v < mx) {
      r.second = std::fmax(r.second, v);
    }
  }
  return r;
}

// merge_parts (row_pick_partitioned.cuh) over count partition records.
inline Record merge_oracle(const Record* p, int count) {
  Record r{-INFINITY, -INFINITY, INT_MAX, -1, 0};
  for (int i = 0; i < count; ++i) {
    r.max = std::fmax(r.max, p[i].max);
    r.nan |= p[i].nan;
  }
  for (int i = 0; i < count; ++i) {
    if (p[i].max == r.max) {
      r.first = std::min(r.first, p[i].first);
      r.last = std::max(r.last, p[i].last);
      r.second = std::fmax(r.second, p[i].second);
    } else if (p[i].max < r.max) {
      r.second = std::fmax(r.second, p[i].max);
    }
  }
  return r;
}

// Partition p of parts over a row of vocab entries, as partition_rows_kernel splits it.
inline long long part_begin(int vocab, int p, int parts) { return static_cast<long long>(vocab) * p / parts; }

inline bool same_value(const Record& a, const Record& b) {
  return a.max == b.max && a.second == b.second && a.first == b.first && a.last == b.last && a.nan == b.nan;
}

inline bool same_bits(const Record& a, const Record& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// Why api.cpp's greedy_from_pick returns -1 for a record.
enum class Undecided { kNone, kNonfinite, kNearTie };

inline const char* undecided_name(Undecided u) {
  return u == Undecided::kNonfinite ? "nonfinite" : u == Undecided::kNearTie ? "near_tie" : "none";
}

// greedy_from_pick's decision without its RNG draw: the token, or -1 with the reason.
inline int decision(const Record& p, float temperature, Undecided* why) {
  *why = Undecided::kNone;
  if (p.nan || !std::isfinite(p.max) || p.first < 0 || p.last < 0) {
    *why = Undecided::kNonfinite;
    return -1;
  }
  if (temperature < 0.05f) return p.first;
  const float scaled = (p.second - p.max) * (1.0f / temperature);
  if (!(scaled < -1e-5f)) {
    *why = Undecided::kNearTie;
    return -1;
  }
  return p.last;
}

// The conditions under which api.cpp's plain decode loop requests a compact pick, one bit each; zero means
// eligible. Mirrors want_device_pick there.
enum Exclusion : unsigned {
  kIr = 1u << 0,
  kCpuOffload = 1u << 1,
  kPickOff = 1u << 2,
  kTopK = 1u << 3,
  kRepeatPenalty = 1u << 4,
  kRepetitionPenalty = 1u << 5,
  kPromptPosition = 1u << 6,
};

inline unsigned exclusions(bool ir, int cpu_offload_layers, bool pick_env, int top_k, bool repeat_on,
                           float repetition_penalty, int pos, int num_prompt_tokens) {
  unsigned m = 0;
  if (ir) m |= kIr;
  if (cpu_offload_layers != 0) m |= kCpuOffload;
  if (!pick_env) m |= kPickOff;
  if (top_k != 1) m |= kTopK;
  if (repeat_on) m |= kRepeatPenalty;
  if (!(repetition_penalty <= 1.0f)) m |= kRepetitionPenalty;
  if (!(pos >= num_prompt_tokens - 1)) m |= kPromptPosition;
  return m;
}

inline std::string exclusion_names(unsigned m) {
  static const char* const names[] = {"ir", "cpu_offload", "pick_off", "top_k", "repeat_penalty",
                                      "repetition_penalty", "prompt_position"};
  std::string s;
  for (int i = 0; i < 7; ++i)
    if (m & (1u << i)) s += (s.empty() ? "" : ",") + std::string(names[i]);
  return s;
}

// Where a sampled decode position's token came from.
inline const char* route_name(bool eligible, bool record_returned, bool debug_rows, bool forced, Undecided why) {
  if (!eligible) return "host_ineligible";
  if (!record_returned) return "per_launch_fallback";
  if (debug_rows) return "host_after_debug_rows";
  if (forced) return "host_after_forced_undecidable";
  if (why == Undecided::kNonfinite) return "host_after_nonfinite";
  if (why == Undecided::kNearTie) return "host_after_near_tie";
  return "compact";
}

// A replay's record checked against the row the same replay produced: the whole-row oracle, and for a
// partitioned selector each partition record and their merge.
struct ReplayCheck {
  bool record_written = false;
  bool record_value = false;   // same values as the whole-row oracle (+0 and -0 compare equal)
  bool record_bits = false;    // same bytes
  int parts_written = 0;
  int parts_value = 0;         // partition records with the values of their partition's oracle
  bool merge_value = false;    // merge_oracle of the partition records equals the record
};

inline ReplayCheck check_replay(const float* row, int vocab, const Record& record, const Record* parts_records,
                                int parts) {
  ReplayCheck c;
  c.record_written = !unwritten(record);
  const Record whole = partition_oracle(row, 0, vocab);
  c.record_value = same_value(record, whole);
  c.record_bits = same_bits(record, whole);
  if (parts > 1) {
    for (int p = 0; p < parts; ++p) {
      c.parts_written += !unwritten(parts_records[p]);
      c.parts_value += same_value(parts_records[p],
                                  partition_oracle(row, part_begin(vocab, p, parts), part_begin(vocab, p + 1, parts)));
    }
    c.merge_value = same_value(merge_oracle(parts_records, parts), record);
  }
  return c;
}

}  // namespace route_trace
}  // namespace ember

#endif  // EMBER_DECODE_ROUTE_TRACE_H
