# CUDA batch-op syncs, verification path, WASM threads, 20 Oct 2026

## Hidden syncs in the CUDA batch helpers (`sync_fix.log`)

`rmsnorm_batch_cuda`, `add_residual_batch_cuda`, `rope_batch_cuda`, `update_kv_cache_batch_cuda`,
`swiglu_batch_cuda` and others carried leftover debug probes: 22 `cudaMemcpyAsync` copies of a
few input values into pageable host vectors that nothing read. A copy to pageable memory blocks
the host until the GPU catches up, so every batch op was a full sync. They now run only with
`EMBER_KERNEL_DEBUG=1`; two unconditional per-layer `cudaStreamSynchronize` calls in the batch
prefill (single stream) are removed.

- First-token logits on chat_sftfinal_v3 (40 prompts, CUDA): byte-identical to release c843421.
- chat_sftfinal_v3 CUDA greedy vs mtstudio: 40/40.
- Forward of n new tokens after a cached 2048-token prefix (timing model, prefix reuse):
  1 token 56.4 -> 14.7 ms, 2: 51.3 -> 21.2, 5: 60.4 -> 20.8, 9: 35.6 -> 28.9, 17: 42.6 -> 23.3
  (medians of 3; the machine was loaded, so absolute values are noisy).

## Speculative verification path (`spec_and_wasm_threads.log`)

`TinyLlamaModel::forward_device_verify`: the draft rows go through FP32 GEMMs, the batch
kernels and, per row, the same split-K decode-attention kernel the graph decode uses (chunked
for the full cache, so attention matches it exactly); persistent buffers, no allocation per
pass. A pass of up to 9 tokens costs about 2.5 graph decode steps (instrumented: ~9 ms with
timing syncs vs ~2.6 ms per step), down from 10-20 steps through the batched prefill.

On the 40 chat prompts speculation stays exact (replies identical 40/40 in all three rounds,
k = 4 and 8) but is not faster (0.54-0.89x total wall): chat replies give 18-19 verification
passes in 636 tokens with 1.2 accepted drafts per pass, below the ~1.5 a pass needs to pay
for itself. It remains opt-in (`EMBER_SPEC=k`), for text-repeating workloads.

## WebAssembly threads (`spec_and_wasm_threads.log`)

`ember::parallel_for` (`parallel_for.h`): OpenMP `schedule(static)` in native builds (the
loops it replaces, unchanged), a persistent std::thread pool in Emscripten pthread builds
(Emscripten has no OpenMP). Used by the f32 matvec (every decode projection and the LM head),
the f32 batch matmul and CPU batch attention. `EMBER_WASM_THREADS=4 EMBER_WASM_NAME=ember-mt
bash deploy/wasm/build.sh OUT` builds `ember-mt.{js,wasm}`; `web/worker.js` loads it only on a
cross-origin-isolated page (COOP/COEP) and falls back to `ember.js` otherwise or if absent.

- 4-thread build vs single-threaded, node, reply by reply: q8all 40/40 identical, f32 v2 40/40
  identical; gpt2 (IR) and moe_tiny 40/40 vs mtstudio. Native CPU chat v3 40/40.
- Speed on an idle machine (`wasm_threads_quiet.log`, node, three interleaved rounds, load
  average 1.4-2.1, no GPU work): q8all (the Space's model) single 302-378 vs 4-thread 400-450
  ms/reply; f32 v2 single 195-347 vs 4-thread 187-308. Replies identical (40/40) on both. At
  d=384 the per-loop work is too small to repay the thread handoff, so threads do not help this
  model: the Space keeps the single-threaded build. ember-mt is for larger models.
