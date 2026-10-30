# CPU forward comparison — 30 October 2026

This bounded comparison measures single-thread FP32 forward latency, using the same existing F32 GGUF in Ember's CPU interpreter and Transformers' local GGUF loader. It is not a model-quality, kernel-only, end-to-end chat, or GPU benchmark.

Each model uses synthetic raw-token prompts of 16, 128 and 512 tokens and 32 output rows. The initial HF greedy continuation is frozen, including EOS. Before timing a shape, all 32 full vocabulary rows and argmax IDs must pass the existing ROUNDING bound (`max_abs_error / max(1, max_abs_reference_row) <= 1e-4`). A numerical failure is preserved and prevents timing that shape. No tolerance adjustment or selective prompt replacement is permitted.

Each shape has one unmeasured warm request per runtime, followed by two ABBA cycles (Ember, HF, HF, Ember). Timers surround only runtime forward calls: last-row-only prefill plus 31 cached single-token forwards. Tokenization, argmax, row writing, loading and process startup are excluded. Runtime dispatch, cache and output wrappers are included. FP32 describes weights and activations: Ember's reference linear kernels accumulate dot products in FP64 with eight fixed-order lanes, whereas Torch uses its own CPU kernels. Arithmetic strategies are therefore not identical. Ember preallocates KV while HF creates a DynamicCache per request, so the result does not isolate arithmetic kernels. Synthetic token workloads do not represent a natural-language distribution.

The binary is a Release CPU-only build (`-O3 -DNDEBUG`, default CMake flags, no extra native-architecture override), built on WSL ext4. `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are one, as are Torch intra/inter-op threads. `EMBER_IR_BUILD=1` enables native GGUF lowering and `EMBER_IR_CTX=1024` bounds cache allocation. No deterministic, KV or alternate interpreter override is used. The protocol records exact binary, source and weight hashes, library versions, CPU and environment. Other remote experiments continue, so shared-host interference remains a limitation; paired ordering helps but cannot eliminate it.

Every timing trial is fsynced to a hash-chained journal. Resume verifies the protocol, raw parity gates, trial identities, IDs and hashes; completed trials are never re-timed. HALTED results remain preserved. `benchmark.py --analyze --out RECEIPT_DIR` replays the complete gates and timing inventory without model calls.

Preflight: Transformers 4.57.0 cannot load the existing Pythia-160M GGUF (unsupported GPTNeoX GGUF architecture), and no native weights are cached. SmolLM2-135M and Gemma3-270M GGUF loaders passed shape/cache preflight; only the full parity gate establishes eligibility for timings. No weights were downloaded or copied.

Canonical receipts will live under `C:/ml_artifacts/benchmarks/hf_cpu_20261030/`. The original CPU comparison is complete for both supported models. The paired native old/new AVX2 comparison is also complete; its independent offline audit is recorded in `receipts/cpu_avx2_independent_audit_20261030.json`.


## Original portable CPU build versus Transformers

All 96 full vocabulary rows and greedy IDs pass for each model (32 rows at each of three prompt lengths). Maximum row-relative errors remain below 1e-4. Values below are medians of four trials per runtime and shape, in milliseconds. These are this laptop's measurements, not general model/engine rankings. The original portable Ember build is slower in every measured cell.

| Model | Prompt tokens | Ember prefill ms | HF prefill ms | Ember decode ms/token | HF decode ms/token |
|---|---:|---:|---:|---:|---:|
| smollm2-135m | 16 | 879.61 | 122.08 | 309.88 | 45.40 |
| smollm2-135m | 128 | 6627.86 | 374.05 | 333.72 | 41.70 |
| smollm2-135m | 512 | 34832.42 | 3216.86 | 412.75 | 71.63 |
| gemma3-270m | 16 | 861.55 | 125.36 | 159.26 | 69.67 |
| gemma3-270m | 128 | 6532.57 | 446.47 | 175.34 | 71.38 |
| gemma3-270m | 512 | 25202.29 | 2215.97 | 182.19 | 87.81 |

The candidate uses runtime AVX2 dispatch only on supported x86-64 GCC/Clang builds. It preserves eight FP64 accumulation lanes and the previous reduction order; other targets retain the portable implementation. Baseline, native-architecture, explicit-FMA and forced-portable builds each passed 3,840 finite/adversarial bit comparisons. Two CTest targets exercise automatic and forced-portable dispatch. This unit/microbenchmark evidence alone does not establish an end-to-end speedup.

`compare_native.py` checks the current CPU against the reference, loads old/new frozen binaries on the same weights, and requires every captured output bit to equal the retained HF-gated reference before paired ABBA timing. Successful duplicate raw captures are removed only after exact SHA equality with the retained reference; failed captures remain. Every completed trial is hash-chained and checked on resume. `--analyze --out DIR` replays its report without loading models.


## Paired old/new native CPU results

All 192 captured full-vocabulary rows and greedy IDs are bit-identical to the retained HF-gated references. Each entry is baseline/optimized median latency from four ABBA trials per build; values above one indicate a measured speedup.

| Model | Prompt tokens | Prefill speedup | Decode speedup |
|---|---:|---:|---:|
| smollm2-135m | 16 | 2.380x | 1.041x |
| smollm2-135m | 128 | 2.798x | 1.094x |
| smollm2-135m | 512 | 2.637x | 1.145x |
| gemma3-270m | 16 | 2.578x | 1.483x |
| gemma3-270m | 128 | 2.869x | 1.496x |
| gemma3-270m | 512 | 2.494x | 1.425x |

This improves the reference implementation while preserving its arithmetic order. It does not close the CPU gap to Transformers measured above; the optimized cells remain slower. A fresh source/binary-bound conformance run passed 554 cases with zero failures and 44 gradient skips. The independent audit replays all four saved timing analyses, verifies the two weight and two binary hashes and the saved conformance launch/report, and checks the 192 row/ID captures. Timing trials were not repeated to improve the result. GPU comparisons remain separate work.
