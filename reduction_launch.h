#ifndef EMBER_REDUCTION_LAUNCH_H
#define EMBER_REDUCTION_LAUNCH_H
#ifdef __CUDACC__
#define EMBER_REDUCTION_HD __host__ __device__
#else
#define EMBER_REDUCTION_HD
#endif
namespace ember {
EMBER_REDUCTION_HD constexpr bool valid_single_block_reduction_launch(
    unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz) {
  return gx == 1 && gy == 1 && gz == 1 && by == 1 && bz == 1 &&
         bx >= 32 && bx <= 1024 && bx % 32 == 0;
}
#ifdef __CUDACC__
__device__ __forceinline__ void require_single_block_reduction_launch() {
  if (!valid_single_block_reduction_launch(gridDim.x, gridDim.y, gridDim.z,
                                            blockDim.x, blockDim.y, blockDim.z))
    __trap();  // Release-active: unlike assert(), never removed by NDEBUG.
}
#endif
}  // namespace ember
#undef EMBER_REDUCTION_HD
#endif
