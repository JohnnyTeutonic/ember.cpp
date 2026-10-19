# WebAssembly SIMD, 20 Oct 2026

`deploy/wasm/build.sh` now compiles with `-msimd128` (WebAssembly SIMD, in every current
browser); `EMBER_WASM_SIMD=0` builds without it. No source changes: the CPU core's
vectorisable loops (the fp64-lane dot products, batch matmul) autovectorise.

Node 22 in WSL2, 40 frozen prompts, greedy, up to 60 tokens (`simd_ab.log`, two
interleaved rounds):

| Model | no SIMD | SIMD | replies = mtstudio |
|---|---|---|---|
| f32 `chat_sftfinal_v2` | 516-536 ms/reply | 248-277 ms | 40/40 both |
| `v2_q8all` (the Space) | 495-526 ms | 212-268 ms | 36/40 both |

Verification on the committed build.sh (ember.wasm 3a1c8c65): every reply identical,
one by one, to the non-SIMD build on f32 v2, q8all, q4_k_s and q4_0; `moe_tiny` 40/40 vs
mtstudio.

Threads (pthreads + SharedArrayBuffer) are not built. They need cross-origin isolation;
a static Space can send COOP/COEP through `custom_headers`, but the website embeds the
Space in an iframe on GitHub Pages, which cannot send them, so the embedded chat would
still need the single-threaded build.
