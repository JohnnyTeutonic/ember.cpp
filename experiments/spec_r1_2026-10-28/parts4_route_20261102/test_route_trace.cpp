// CPU checks of decode_route_trace.h, the host half of the PARTS4 route evidence.
//   g++ -std=c++17 -O1 -I ../../.. test_route_trace.cpp -o test_route_trace && ./test_route_trace
// Prints one JSON summary line; exit status 0 only when every check holds.
#include "decode_route_trace.h"

#include <cstdio>
#include <random>
#include <vector>

namespace rt = ember::route_trace;

static int failures = 0, checks = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    ++checks;                                                             \
    if (!(cond)) {                                                        \
      ++failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                     \
  } while (0)

// api.cpp's want_device_pick, as written there (CUDA build, before the graph call).
static bool want_device_pick(bool ir, int offload, bool device_pick_on, int top_k, bool repeat_on, float rep, int pos,
                             int num_prompt_tokens) {
  return !ir && offload == 0 && device_pick_on && top_k == 1 && !repeat_on && rep <= 1.0f &&
         pos >= num_prompt_tokens - 1;
}

static void oracle_cases() {
  const float row[] = {1.0f, 3.0f, 2.0f, 3.0f, -1.0f};
  rt::Record r = rt::partition_oracle(row, 0, 5);
  CHECK(r.max == 3.0f && r.second == 2.0f && r.first == 1 && r.last == 3 && r.nan == 0);
  const float nan_row[] = {NAN, 0.5f, NAN};
  r = rt::partition_oracle(nan_row, 0, 3);
  CHECK(r.max == 0.5f && r.first == 1 && r.last == 1 && r.nan == 1 && r.second == -INFINITY);
  const float all_nan[] = {NAN, NAN};
  r = rt::partition_oracle(all_nan, 0, 2);
  CHECK(r.max == -INFINITY && r.first == INT_MAX && r.last == -1 && r.nan == 1);
  const float neg_inf[] = {-INFINITY, -INFINITY};
  r = rt::partition_oracle(neg_inf, 0, 2);
  CHECK(r.max == -INFINITY && r.first == 0 && r.last == 1 && r.second == -INFINITY);
  // The kernel's split: [vocab * p / parts, vocab * (p + 1) / parts).
  CHECK(rt::part_begin(16384, 0, 4) == 0 && rt::part_begin(16384, 1, 4) == 4096 &&
        rt::part_begin(16384, 3, 4) == 12288 && rt::part_begin(16384, 4, 4) == 16384);
  CHECK(rt::part_begin(10, 1, 4) == 2 && rt::part_begin(10, 2, 4) == 5 && rt::part_begin(10, 3, 4) == 7);
}

// merge_parts over the four partition records agrees in value with the whole-row record, for rows with ties
// across partition boundaries, NaNs, infinities, signed zeros and partitions that hold only NaNs.
static void merge_property() {
  std::mt19937 g(20261102);
  const int vocabs[] = {1, 2, 3, 4, 5, 7, 10, 64, 1000, 16384};
  int rows = 0;
  for (int vocab : vocabs) {
    for (int t = 0; t < 400; ++t) {
      std::vector<float> row(vocab);
      std::uniform_int_distribution<int> kind(0, 9), small(-3, 3);
      for (float& v : row) {
        const int k = kind(g);
        // Small integers make ties (within and across partitions) common.
        v = k == 0 ? NAN : k == 1 ? -INFINITY : k == 2 ? (g() & 1 ? 0.0f : -0.0f) : static_cast<float>(small(g));
      }
      if (t % 7 == 0)
        for (float& v : row) v = NAN;  // all-NaN row
      if (t % 11 == 0 && vocab >= 4)
        for (long long i = 0; i < rt::part_begin(vocab, 1, 4); ++i) row[i] = NAN;  // one all-NaN partition
      rt::Record parts[4];
      for (int p = 0; p < 4; ++p)
        parts[p] = rt::partition_oracle(row.data(), rt::part_begin(vocab, p, 4), rt::part_begin(vocab, p + 1, 4));
      const rt::Record whole = rt::partition_oracle(row.data(), 0, vocab);
      CHECK(rt::same_value(rt::merge_oracle(parts, 4), whole));
      const rt::ReplayCheck c = rt::check_replay(row.data(), vocab, whole, parts, 4);
      CHECK(c.record_written && c.record_value && c.parts_written == 4 && c.parts_value == 4 && c.merge_value);
      ++rows;
    }
  }
  CHECK(rows == 4000);
}

static void sentinel_cases() {
  rt::Record s;
  std::memset(&s, 0xFF, sizeof s);
  CHECK(rt::unwritten(s));
  const float row[] = {1.0f, 2.0f};
  const rt::Record r = rt::partition_oracle(row, 0, 2);
  CHECK(!rt::unwritten(r));
  rt::Record nan_rec = r;
  nan_rec.nan = 1;
  CHECK(!rt::unwritten(nan_rec));
  // A replay whose selector did not run leaves the sentinel in place: no part of the check passes.
  rt::Record sentinel_parts[4];
  std::memset(sentinel_parts, 0xFF, sizeof sentinel_parts);
  const rt::ReplayCheck c = rt::check_replay(row, 2, s, sentinel_parts, 4);
  CHECK(!c.record_written && !c.record_value && c.parts_written == 0 && c.parts_value == 0);
  // A record that differs from the row's oracle fails the value check while still counting as written.
  rt::Record wrong = r;
  wrong.first = 0;
  const rt::ReplayCheck w = rt::check_replay(row, 2, wrong, sentinel_parts, 0);
  CHECK(w.record_written && !w.record_value && !w.record_bits);
}

static void decision_cases() {
  rt::Undecided why;
  rt::Record r{5.0f, 4.0f, 2, 9, 0};
  CHECK(rt::decision(r, 0.0f, &why) == 2 && why == rt::Undecided::kNone);   // below 0.05: first index
  CHECK(rt::decision(r, 0.049f, &why) == 2 && why == rt::Undecided::kNone);
  CHECK(rt::decision(r, 1.0f, &why) == 9 && why == rt::Undecided::kNone);   // otherwise last index
  CHECK(rt::decision(r, 1.5f, &why) == 9);
  rt::Record tie{5.0f, 5.0f - 5e-6f, 2, 2, 0};
  CHECK(rt::decision(tie, 1.0f, &why) == -1 && why == rt::Undecided::kNearTie);
  CHECK(rt::decision(tie, 0.0f, &why) == 2);  // the near-tie rule applies only from 0.05
  rt::Record only{5.0f, -INFINITY, 3, 3, 0};
  CHECK(rt::decision(only, 1.0f, &why) == 3);
  rt::Record nanr{5.0f, 4.0f, 2, 2, 1};
  CHECK(rt::decision(nanr, 1.0f, &why) == -1 && why == rt::Undecided::kNonfinite);
  rt::Record inf{INFINITY, 4.0f, 2, 2, 0};
  CHECK(rt::decision(inf, 0.0f, &why) == -1 && why == rt::Undecided::kNonfinite);
  rt::Record empty{-INFINITY, -INFINITY, INT_MAX, -1, 1};
  CHECK(rt::decision(empty, 1.0f, &why) == -1 && why == rt::Undecided::kNonfinite);
}

static void exclusion_cases() {
  int combos = 0;
  for (int ir = 0; ir < 2; ++ir)
    for (int off : {0, 1, 16})
      for (int env = 0; env < 2; ++env)
        for (int k : {0, 1, 4})
          for (int rep_on = 0; rep_on < 2; ++rep_on)
            for (float rep : {1.0f, 0.9f, 1.2f})
              for (int pos : {0, 8, 9, 10, 40}) {
                const unsigned m = rt::exclusions(ir, off, env, k, rep_on, rep, pos, 10);
                CHECK((m == 0) == want_device_pick(ir, off, env, k, rep_on, rep, pos, 10));
                ++combos;
              }
  CHECK(combos == 2 * 3 * 2 * 3 * 2 * 3 * 5);
  CHECK(rt::exclusion_names(rt::kTopK | rt::kRepeatPenalty) == "top_k,repeat_penalty");
  CHECK(rt::exclusion_names(0).empty());
  CHECK(rt::exclusion_names(rt::kPromptPosition) == "prompt_position");
}

static void route_cases() {
  using U = rt::Undecided;
  CHECK(std::string(rt::route_name(false, false, false, false, U::kNone)) == "host_ineligible");
  CHECK(std::string(rt::route_name(true, false, false, false, U::kNone)) == "per_launch_fallback");
  CHECK(std::string(rt::route_name(true, true, true, false, U::kNone)) == "host_after_debug_rows");
  CHECK(std::string(rt::route_name(true, true, false, true, U::kNone)) == "host_after_forced_undecidable");
  CHECK(std::string(rt::route_name(true, true, false, false, U::kNonfinite)) == "host_after_nonfinite");
  CHECK(std::string(rt::route_name(true, true, false, false, U::kNearTie)) == "host_after_near_tie");
  CHECK(std::string(rt::route_name(true, true, false, false, U::kNone)) == "compact");
}

int main() {
  oracle_cases();
  merge_property();
  sentinel_cases();
  decision_cases();
  exclusion_cases();
  route_cases();
  std::printf("{\"checks\": %d, \"failures\": %d}\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
