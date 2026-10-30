// Fixed-order FP64 accumulation for FP32 rows. AVX2 changes the instruction
// width, never the eight accumulation chains or their final reduction order.
// Bit identity is gated for finite FP32 inputs on x86-64; other targets retain
// the portable implementation. This TU must retain its no-contraction policy.
#pragma once

#if !defined(EMBER_CPU_DOT_PORTABLE_ONLY) && (defined(__GNUC__) || defined(__clang__)) && \
    defined(__x86_64__)
#include <immintrin.h>
#define EMBER_HAVE_AVX2_DOT 1
#endif

namespace ember {
inline double dot_f32_f64_portable(const float* a, const float* b, int n) {
  double acc[8] = {};
  int i = 0;
  for (; i + 8 <= n; i += 8)
    for (int k = 0; k < 8; ++k) acc[k] += static_cast<double>(a[i+k]) * static_cast<double>(b[i+k]);
  double tail = 0.0;
  for (; i < n; ++i) tail += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return ((acc[0]+acc[1])+(acc[2]+acc[3])) + ((acc[4]+acc[5])+(acc[6]+acc[7])) + tail;
}

inline bool has_avx2_dot() {
#ifdef EMBER_HAVE_AVX2_DOT
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

#ifdef EMBER_HAVE_AVX2_DOT
__attribute__((target("avx2")))
inline double dot_f32_f64_avx2(const float* a, const float* b, int n) {
  __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    lo = _mm256_add_pd(lo, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),
                                       _mm256_cvtps_pd(_mm_loadu_ps(b+i))));
    hi = _mm256_add_pd(hi, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),
                                       _mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));
  }
  double acc[8];
  _mm256_storeu_pd(acc, lo); _mm256_storeu_pd(acc+4, hi);
  double tail = 0.0;
  for (; i < n; ++i) tail += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return ((acc[0]+acc[1])+(acc[2]+acc[3])) + ((acc[4]+acc[5])+(acc[6]+acc[7])) + tail;
}
#endif

inline double dot_f32_f64(const float* a, const float* b, int n) {
  using Fn = double (*)(const float*, const float*, int);
#ifdef EMBER_HAVE_AVX2_DOT
  static const Fn fn = has_avx2_dot() ? dot_f32_f64_avx2 : dot_f32_f64_portable;
#else
  static const Fn fn = dot_f32_f64_portable;
#endif
  return fn(a, b, n);
}
}  // namespace ember
