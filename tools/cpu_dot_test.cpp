#include "cpu_dot.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

uint64_t bits(double x) { uint64_t v; std::memcpy(&v, &x, sizeof(v)); return v; }
// The exact pre-change eight-lane implementation, retained as a bit-level pin.
double previous(const float* a, const float* b, int n) {
  double acc[8] = {0,0,0,0,0,0,0,0}; int i=0;
  for (; i+8<=n; i+=8)
    for (int k=0;k<8;++k) acc[k] += static_cast<double>(a[i+k])*static_cast<double>(b[i+k]);
  double tail=0.0;
  for (;i<n;++i) tail+=static_cast<double>(a[i])*static_cast<double>(b[i]);
  return ((acc[0]+acc[1])+(acc[2]+acc[3]))+((acc[4]+acc[5])+(acc[6]+acc[7]))+tail;
}
int main() {
  std::mt19937 rng(20261030);
  std::vector<float> a(8200),b(8200);
  int checked=0;
  for (int mode=0;mode<5;++mode) {
    for (size_t i=0;i<a.size();++i) {
      auto sample=[&]() {
        uint32_t u=rng();
        // Finite raw FP32 bit patterns, including signed subnormals and extremes.
        if ((u & 0x7f800000u)==0x7f800000u) u &= ~0x00800000u;
        float f; std::memcpy(&f,&u,4); return f;
      };
      a[i]=sample(); b[i]=sample();
      if (mode==1) { a[i]=float(int(rng()%10001)-5000)/97.0f; b[i]=float(int(rng()%10001)-5000)/101.0f; }
      if (mode==2) { a[i]=(i%2 ? -1.0f:1.0f)*1e20f; b[i]=1e-20f; }
      if (mode==3) { a[i]=(i%2 ? -0.0f:0.0f); b[i]=std::numeric_limits<float>::max(); }
      if (mode==4) { a[i]=(i%2 ? -1:1)*std::numeric_limits<float>::denorm_min(); b[i]=float(i%17); }
    }
    for (int offset=0;offset<4;++offset) {
      for (int n=0;n<=8192;n+=(n<80 ? 1:73)) {
        const auto expected=bits(previous(a.data()+offset,b.data()+offset,n));
        if (bits(ember::dot_f32_f64_portable(a.data()+offset,b.data()+offset,n))!=expected ||
            bits(ember::dot_f32_f64(a.data()+offset,b.data()+offset,n))!=expected)
          throw std::runtime_error("portable/dispatched bit mismatch");
#ifdef EMBER_HAVE_AVX2_DOT
        if (ember::has_avx2_dot() && bits(ember::dot_f32_f64_avx2(a.data()+offset,b.data()+offset,n))!=expected)
          throw std::runtime_error("AVX2 bit mismatch");
#endif
        ++checked;
      }
    }
  }
  // Same finite workload, alternating order to expose a gross regression.
  for(size_t i=0;i<a.size();++i){a[i]=float(i%97)/97;b[i]=float(i%101)/101;}
  volatile double sink=0;
  double old_ms=0,new_ms=0;
  for(int trial=0;trial<4;++trial) {
    for(int slot=0;slot<2;++slot) {
      bool fast=(trial+slot)%2;
      auto start=std::chrono::steady_clock::now();
      for(int j=0;j<50000;++j) sink+= fast ? ember::dot_f32_f64(a.data(),b.data(),4096) : previous(a.data(),b.data(),4096);
      auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      (fast?new_ms:old_ms)+=ms;
    }
  }
  std::cout << "{\"checked\":"<<checked<<",\"avx2\":"<<(ember::has_avx2_dot()?"true":"false")
            <<",\"previous_ms\":"<<old_ms<<",\"dispatched_ms\":"<<new_ms
            <<",\"speedup\":"<<old_ms/new_ms<<",\"sink\":"<<sink<<"}\n";
}
