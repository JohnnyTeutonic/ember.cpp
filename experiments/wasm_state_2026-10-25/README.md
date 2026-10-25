# WebAssembly legs of the state-token and llama_attnres gates, 25 Oct 2026

ember's IR interpreter compiled to WebAssembly against ember's native CPU build on paperkiln's
state-token model state4 (`C:/ml_artifacts/state_gate/state4`: llama_state, d 64, 2 layers, vocab 400,
k 4, mask and eviction `turn_state`) and its Q8_0 file, the attention-residual model ar3
(`C:/ml_artifacts/attnres_gate/ar3`: llama_attnres), and the synthetic k = 4 model of the CPU gate
(`chat_state_k4.gguf`, `../state_tokens_2026-10-24/`). These are the CUDA leg's checks
(`../cuda_state_2026-10-25/`, `tools/state_device_gate`) with WebAssembly in the device's place, plus
state4 through the page's chat path. Sources: ember f4fc5e0, with the page changes of
`../wasm_ctx_2026-10-25/` for the chat path. Emscripten 6.0.6, node 24.19.0 (emsdk's).

## Method

- **`tools/state_wasm_gate.cpp`**, one source for both sides. It runs state_device_gate's sequence of
  calls on one engine and writes every logits row and every integer result to files:
  - one-call prefill with eviction on (the IR's setting) and off (`EMBER_EVICT=0`);
  - token-by-token decode, and turn-by-turn prefill (one call per turn);
  - free-running greedy from each prompt;
  - kv rows held after the context, on and off;
  - under eviction, rewinds into turns 0 and 1 (resume point, the refused direct forward, the re-prefill);
  - the session script (generation, a regenerated reply, an edited earlier message, a rewind into turn
    0), on and off and against a fresh prefill.
- **Builds** (`build.sh`). `native` is linked with `libtinyllama_core.a` of ember's CPU build.
  `state_wasm_gate.js` uses `deploy/wasm/build.sh`'s sources and flags (`-O3 -msimd128`,
  `-fexceptions`, Emscripten's Boost port), runs in node with NODERAWFS and reads the model without
  mmap, as the page does. The 1gb heap is the page's default.
- **Comparison** (`compare.py`). Both runs make the same calls in the same order, so the blocks line
  up. It reports, as max |dlogit| over the native row's range:
  - WebAssembly against native on every block, with the argmax at every row;
  - inside each engine: on against off, decode and turn by turn against prefill, calls against fresh;
  - greedy, teacher-forced and free-running;
  - every integer result equal: rows held, resume points, refusals, the block layout;
  - state4's rows held against mtstudio's (manifest `kv_rows_evict`, `kv_rows_keep`).
  The tolerance is 1e-4, the device gates' receipt bound.
- **Inputs.** `../cuda_state_2026-10-25/make_inputs.py`'s: state4's 40 `score_in` sequences (prompt +
  numpy's greedy continuation) with prompt lengths, user id 2 and a 12-call session script; the same
  sequences for Q8_0; ar3's 40 `score_in` sequences; the synthetic model's own 12-call script.
- **Random ids** (`tools/ir_wasm_gate`, as the CUDA leg ran `ir_device_gate`). state4: 40 prompts of 24
  ids, 32 greedy tokens. ar3: 40 x 8 ids, 8 greedy tokens (ctx 16).
- **The page's chat path** (`run_chat.sh`). The page serves state4 through `ember_chat`, because WebGPU
  takes llama-architecture files only and state4's is `paperkiln`. `chat_state4.py` writes the 40
  requests of `state4_gate.py`'s /chat leg: history and message from `prompts_text.jsonl`, greedy,
  `max_new_tokens` capped at the page's 64. The native server answers them
  (`../wasm_ctx_2026-10-25/collect_native.py`). `tools/wasm_chat_check.mjs` then replays each request
  on the page build (`deploy/wasm/build.sh`'s ember.js), through `ember_prepare` (prompt ids) and
  `ember_chat` (reply, token count, stop reason, prompt tokens, turns used).

## Results

| check | state4 | state4 Q8_0 | ar3 | synthetic k = 4 |
|---|---|---|---|---|
| WebAssembly vs native, worst over every block | 3.8e-7 | 2.5e-7 | 0 (bit-identical) | 2.4e-7 |
| top-1 vs native, every row of every block | 18,199/18,199 | 17,488/17,488 | 1,419/1,419 | 1,029/1,029 |
| greedy, teacher-forced / free-running (WebAssembly; native the same) | 40/40, 40/40 | 40/40, 40/40 | 40/40, 40/40 | |
| greedy ids equal to native's | yes | yes | yes | |
| eviction on vs off, decode vs prefill, turn by turn vs prefill, calls vs fresh (inside WebAssembly) | 0, bit-equal | 0, bit-equal | 0, bit-equal | 0, bit-equal |
| every integer result equal to native's | yes | yes | yes | yes |
| kv rows held, evicting / keeping: equal to native's (and for state4 to mtstudio's manifest) | 40/40; mean 25.35 / 76.2 | 40/40; 25.35 / 76.2 | | |
| rewinds into evicted turns: direct forward refused, re-prefill vs native | 80/80, 2.6e-7 | 80/80, 2.5e-7 | | |
| session script: 12 calls, direct rewinds refused, vs native | 12/12, 2/2, 3.8e-7 | | | 12/12, 2/2, 2.4e-7 |

Random ids (`random_state4.json`, `random_ar3.json`): state4 greedy 40/40 and top-1 960/960, worst
2.6e-7. ar3 40/40 and 320/320, bit-identical.

The chat path on state4 (`chat_state4.json`, `chat_state4_native.json`). The native server's prompt ids
equal `score_in`'s 40/40, and its reply ids plus stop id equal the continuation 40/40, as in
`state4_gate.py`. On the page build, `ember_prepare`'s prompt ids equal the server's 40/40, state blocks
included. `ember_chat`'s replies, token counts, stop reasons, prompt token counts and turns used equal
the server's 40/40 each (18 ms per reply in node). `ember_prepare` returns the ids `generate()` runs
(`bpe_prompt_ids`: the encoding with the state blocks inserted, as /chat runs it). Only the WebGPU
path calls it, and WebGPU does not take state4, so `ember_chat` is what serves state4 on a page.

So per-turn state tokens and attention residuals over depth run in WebAssembly as on the CPU. Logits
agree within 3.8e-7 of the range, at the level of the families gate (`../wasm_families_2026-10-25/`).
Eviction stays exact inside the WebAssembly engine, as on the CPU.

## Files

- `build.sh`, `run.sh` (interpreter legs), `run_chat.sh` (chat path), `compare.py`, `chat_state4.py`.
- Receipts: `state4.json`, `state4_q8_0.json`, `ar3.json`, `synth_k4.json`, `random_state4.json`,
  `random_ar3.json`, `chat_state4.json`, `chat_state4_native.json`. The logits (up to 30 MB per model)
  were deleted once each comparison was written.

Commands (WSL; each job runs under `scripts/cpu_budget.py`, one core, session ember):

```bash
SRC=~/wi/src BUILD=~/wi/build_cpu JOBS=2 bash build.sh ~/wi/sgbin
bash run.sh ~/wi/sgbin ~/wi/sgwork
bash run_chat.sh ~/wi/src ~/wi/build_cpu ~/wi/ctxwork/wasm1g ~/wi/chatwork
```
