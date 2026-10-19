// paperkiln side of the sliding-window attention check (tools/swa_attention_test.cpp).
// Builds random q, k, v per case, runs paperkiln's own ops::swa_attention
// (microtorch/src/ops.cpp) and writes inputs and outputs to a binary file:
//   int32 n_cases; per case: int32 T, window, sinks, dk; float scale;
//   float q[T*dk], k[T*dk], v[T*dk], y[T*dk] (row-major).
// Build against a paperkiln build (libmicrotorch.a + transformer_core), e.g.
//   g++ -O2 -std=c++17 -I PK/include -I JSON_INC swa_reference_dump.cpp PK/build/libmicrotorch.a TCORE.a -fopenmp
//   ./swa_reference_dump out.bin
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {
Matrix randn(size_t r, size_t c, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  Matrix m(r, c);
  for (size_t i = 0; i < r; ++i)
    for (size_t j = 0; j < c; ++j) m(i, j) = d(g);
  return m;
}

void put_matrix(std::FILE* f, const Matrix& m) {
  std::vector<float> buf(m.rows() * m.cols());
  for (size_t i = 0; i < m.rows(); ++i)
    for (size_t j = 0; j < m.cols(); ++j) buf[i * m.cols() + j] = m(i, j);
  std::fwrite(buf.data(), sizeof(float), buf.size(), f);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: swa_reference_dump out.bin\n");
    return 2;
  }
  // (T, window, sinks): windows past T, inside T, of 1, smaller than the
  // sink count, sinks 0, and long sequences where most queries are past the window.
  const int cases[][3] = {{16, 4, 1}, {16, 4, 0}, {16, 1, 3},  {16, 2, 5},  {20, 64, 0}, {20, 64, 4},
                          {33, 8, 4}, {40, 5, 8}, {24, 24, 2}, {24, 23, 1}, {64, 32, 4}, {48, 3, 3}};
  const int dk = 16;
  const float scale = 0.25f;
  std::FILE* f = std::fopen(argv[1], "wb");
  if (!f) return 1;
  const int32_t n = sizeof cases / sizeof cases[0];
  std::fwrite(&n, sizeof n, 1, f);
  for (int c = 0; c < n; ++c) {
    const int T = cases[c][0], W = cases[c][1], S = cases[c][2];
    Var q = make_var(randn(T, dk, 100 + 3 * c), false);
    Var k = make_var(randn(T, dk, 101 + 3 * c), false);
    Var v = make_var(randn(T, dk, 102 + 3 * c), false);
    Var y = ops::swa_attention(q, k, v, scale, W, S);
    const int32_t hdr[4] = {T, W, S, dk};
    std::fwrite(hdr, sizeof(int32_t), 4, f);
    std::fwrite(&scale, sizeof scale, 1, f);
    put_matrix(f, q->data);
    put_matrix(f, k->data);
    put_matrix(f, v->data);
    put_matrix(f, y->data);
  }
  std::fclose(f);
  std::printf("wrote %d cases to %s\n", n, argv[1]);
  return 0;
}
