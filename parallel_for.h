#pragma once
// ember::parallel_for(n, fn): runs fn(i) for every i in [0, n) once, in
// parallel. Native builds use OpenMP (schedule(static), exactly the loops
// this replaces); WebAssembly builds with pthreads (EMBER_WASM_THREADS in
// deploy/wasm/build.sh) use a small persistent thread pool, since Emscripten
// has no OpenMP; anything else runs serially. Each index is computed the same
// way whichever thread runs it, so results do not depend on the schedule.
#include <cstdint>
#include <functional>

namespace ember {

#if defined(__EMSCRIPTEN_PTHREADS__)
namespace detail {
// Splits [0, n) into contiguous chunks over the pool and the calling thread.
void pool_run(int64_t n, const std::function<void(int64_t, int64_t)>& body);
}  // namespace detail

template <class F>
void parallel_for(int64_t n, F&& fn) {
  if (n <= 1) {
    for (int64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  detail::pool_run(n, [&fn](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) fn(i);
  });
}
#else
template <class F>
void parallel_for(int64_t n, F&& fn) {
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) fn(i);
}
#endif

}  // namespace ember
