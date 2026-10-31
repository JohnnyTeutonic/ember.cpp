#pragma once
#include <cmath>
#include <stdexcept>
inline double checked_error(double observed,double reference){
 if(!std::isfinite(observed)||!std::isfinite(reference))throw std::runtime_error("nonfinite semantic output");
 return std::abs(observed-reference);
}
