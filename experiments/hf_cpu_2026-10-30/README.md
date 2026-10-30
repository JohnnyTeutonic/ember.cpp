# CPU forward comparison — 30 October 2026

This bounded comparison measures single-thread FP32 forward latency, using the same existing F32 GGUF in Ember's CPU interpreter and Transformers' local GGUF loader. It is not a model-quality, kernel-only, end-to-end chat, or GPU benchmark.

Each model uses synthetic raw-token prompts of 16, 128 and 512 tokens and 32 output rows. The initial HF greedy continuation is frozen, including EOS. Before timing a shape, all 32 full vocabulary rows and argmax IDs must pass the existing ROUNDING bound (`max_abs_error / max(1, max_abs_reference_row) <= 1e-4`). A numerical failure is preserved and prevents timing that shape. No tolerance adjustment or selective prompt replacement is permitted.

Each shape has one unmeasured warm request per runtime, followed by two ABBA cycles (Ember, HF, HF, Ember). Timers surround only runtime forward calls: last-row-only prefill plus 31 cached single-token forwards. Tokenization, argmax, row writing, loading and process startup are excluded. Runtime dispatch, cache and output wrappers are included. Ember preallocates KV while HF creates a DynamicCache per request, so the result does not isolate arithmetic kernels. Synthetic token workloads do not represent a natural-language distribution.

The binary is a Release CPU-only build (`-O3 -DNDEBUG`, default CMake flags, no extra native-architecture override), built on WSL ext4. `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are one, as are Torch intra/inter-op threads. `EMBER_IR_BUILD=1` enables native GGUF lowering and `EMBER_IR_CTX=1024` bounds cache allocation. No deterministic, KV or alternate interpreter override is used. The protocol records exact binary, source and weight hashes, library versions, CPU and environment. Other remote experiments continue, so shared-host interference remains a limitation; paired ordering helps but cannot eliminate it.

Every timing trial is fsynced to a hash-chained journal. Resume verifies the protocol, raw parity gates, trial identities, IDs and hashes; completed trials are never re-timed. HALTED results remain preserved. `benchmark.py --analyze --out RECEIPT_DIR` replays the complete gates and timing inventory without model calls.

Preflight: Transformers 4.57.0 cannot load the existing Pythia-160M GGUF (unsupported GPTNeoX GGUF architecture), and no native weights are cached. SmolLM2-135M and Gemma3-270M GGUF loaders passed shape/cache preflight; only the full parity gate establishes eligibility for timings. No weights were downloaded or copied.

Canonical receipts will live under `C:/ml_artifacts/benchmarks/hf_cpu_20261030/`. Results are pending.
