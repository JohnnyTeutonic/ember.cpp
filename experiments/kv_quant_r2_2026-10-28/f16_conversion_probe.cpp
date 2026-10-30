// Independent CPU diagnostic: production helper vs compiler binary16
// conversion.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

#include "kv_quant.h"
uint16_t reference(float x) {
  _Float16 h = static_cast<_Float16>(x);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
float expand(uint16_t b) {
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return static_cast<float>(h);
}
int main() {
  uint64_t checked = 0, failures = 0;
  float first = 0;
  uint16_t got = 0, want = 0;
  auto check = [&](float x) {
    ++checked;
    auto a = ember_kvq::f32_to_f16(x), b = reference(x);
    bool nan = std::isnan(x);
    bool same = nan ? ((a & 0x7c00) == 0x7c00 && (a & 0x3ff)) : a == b;
    if (!same) {
      if (!failures) {
        first = x;
        got = a;
        want = b;
      }
      ++failures;
    }
  };
  for (unsigned b = 0; b < 65536; ++b) {
    float x = expand(b);
    check(x);
    float y = ember_kvq::f16_to_f32(b);
    if (std::isnan(x) ? !std::isnan(y) : std::memcmp(&x, &y, 4) != 0)
      ++failures;
  }
  for (unsigned b = 0; b < 0x7bff; ++b) {
    float lo = expand(b), hi = expand(b + 1), mid = (lo + hi) * 0.5f;
    for (float x :
         {std::nextafter(mid, -INFINITY), mid, std::nextafter(mid, INFINITY)}) {
      check(x);
      check(-x);
    }
  }
  for (float x : {65504.f, 65520.f, std::nextafter(65520.f, 0.f),
                  std::nextafter(65520.f, INFINITY)}) {
    check(x);
    check(-x);
  }
  std::mt19937 rng(98317);
  for (unsigned i = 0; i < 1000000; ++i) {
    uint32_t b = rng();
    float x;
    std::memcpy(&x, &b, 4);
    check(x);
  }
  float mid = 1.00048828125f, lo = std::nextafter(mid, -INFINITY),
        hi = std::nextafter(mid, INFINITY);
  std::cout << "{\"checked_f32_inputs\":" << checked
            << ",\"all_f16_expansions\":65536,\"failures\":" << failures
            << ",\"first_failure\":{\"input\":" << first
            << ",\"production\":" << got << ",\"reference\":" << want
            << "},\"midpoint_example\":{\"input_gap\":" << hi - lo
            << ",\"rounded_gap\":"
            << ember_kvq::f16_to_f32(ember_kvq::f32_to_f16(hi)) -
                   ember_kvq::f16_to_f32(ember_kvq::f32_to_f16(lo))
            << "}}\n";
  return failures ? 1 : 0;
}