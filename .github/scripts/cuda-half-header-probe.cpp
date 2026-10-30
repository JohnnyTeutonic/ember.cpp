// Host-only numerical/header probe: no CUDA device, driver or kernel execution.
#include "cuda_safe_headers.h"
#include <cstdint>
#include <cstring>
#include <type_traits>
static_assert(!std::is_same<__half, __nv_bfloat16>::value, "F16 and BF16 must remain distinct vendor types");
int main() {
    const __half h = __float2half_rn(1.0f);
    const __nv_bfloat16 b = __float2bfloat16_rn(1.0f);
    std::uint16_t hbits = 0, bbits = 0;
    static_assert(sizeof(h) == sizeof(hbits) && sizeof(b) == sizeof(bbits), "16-bit storage");
    std::memcpy(&hbits, &h, sizeof(hbits));
    std::memcpy(&bbits, &b, sizeof(bbits));
    return hbits == 0x3c00 && bbits == 0x3f80 && __half2float(h) == 1.0f &&
           __bfloat162float(b) == 1.0f ? 0 : 1;
}
