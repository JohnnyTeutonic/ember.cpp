# The official GPT-2 (124M) through ember's IR builder, CPU, 25 Oct 2026

The checkpoint is OpenAI's GPT-2 124M (`openai-community/gpt2` at revision 607a30d7...), converted by
llama.cpp b6000 (4762ad73) `convert_hf_to_gguf.py --outtype f32`: `gpt2-124m-f32.gguf`, sha256
1c75f6c0... (`models.sha256`, `SOURCES.json`). The file has no `paperkiln.ir`, so ember builds the
graph from its metadata (`ember_ir::build_ir_from_gguf`, `graph_ir.cpp`) and the CPU interpreter
runs it. Binaries: ember at the commit that adds this folder (210f3b8 plus the `ir_version` fix
below), `-DHAS_CUDA=OFF`, built by `build.sh` in its own ext4 tree (`bin.sha256`). paperkiln's side:
`tools/gguf_ir.py` (sha256 25b1bd9b...) and `tools/ir_numpy.py` from paperkiln bc0e8124. Its
reference receipt is `microtorch/experiments/engine_2026-10-25/gpt2/` (`thirdparty_parity.py --raw
--n 40 --max-new 32`: top-1 40/40, greedy 40/40, worst first-token 4.2e-6 of the range).

## The graph

`ember_built.ir.json` equals `gguf_ir.build` (`gguf_ir.json`, byte-identical to paperkiln's
`gate_n40.jsonl.ir.json`) field by field except `_comment`. At ember d49c3e1 the two differed in one
field: ember's builder wrote `ir_version` 0 and gguf_ir.py writes 1, as it has from paperkiln
5349ee19 on (`check_d49c3e1.json`; every other field equal). IR_SPEC section 2.3 says ember's C++
builder follows that change, so `graph_ir.cpp` now writes 1. Engines read 0 and 1 as the same
document (IR_SPEC 2.1), so the interpreter runs the same graph either way: every number in
`check.json` is the same at both commits.

## Logits (`check.json`, `f64.json`)

`tools/thirdparty_ember_check.py --gguf ... --n 4 --len 32`: four seeded random sequences of 32 ids,
every position, ember's prefill and its token-by-token decode against paperkiln's numpy IR and HF
transformers (the native safetensors, f32). The float64 reference is the same IR in ir_numpy with
every float32 read as float64 (`f64.py`). "rel" is max |dlogit| over the reference's logit range.

| | worst rel | median rel | top-1 |
|---|---|---|---|
| ember (prefill and decode) vs numpy f32 | 6.2e-6 | | 128/128 |
| ember vs HF | 7.8e-6 | | 128/128 |
| numpy f32 vs HF | 7.1e-6 | | |
| ember decode vs ember prefill | 0 (bit-identical) | | 128/128 |
| ember vs float64 | 2.3e-6 | 5.5e-7 | 128/128 |
| numpy f32 vs float64 | 6.5e-6 | 3.6e-6 | 128/128 |
| HF vs float64 | 6.8e-6 | 3.7e-6 | 128/128 |

## Greedy on the raw prompts (`greedy.jsonl`, `greedy.summary.json`)

The first 40 prompts of paperkiln's `corpus/eval/chat_prompts_v1.jsonl`, encoded raw by HF's GPT-2
tokeniser (as `thirdparty_parity.py --raw`), 32 new tokens, stopping at `<|endoftext|>`. ember runs
through `tools/ir_adapter`: one prefill of the prompt, then one token at a time through its KV
cache, taking the argmax. HF's reference is the argmax of its own cached forward (never `generate`),
recomputed here; `greedy.py` compares each step's logits while the two agree.

- Greedy replies identical to HF 40/40, and their text equal to the paperkiln receipt's `hf_reply`
  40/40.
- Top-1 agrees at all 54 prompt positions and all 1,280 decode steps.
- First-token logits within 3.8e-6 of HF's range (all prompt positions 4.0e-6). Decode steps reach
  2.8e-5 of the range against HF's cached decode. The smallest HF top-1 margin on any step is 0.028.
- The 14 sequences (prompt plus reply, 462 positions) whose steps left HF by more than 1e-5, against
  float64 (`f64_greedy.json`, full forward): ember 6.8e-6 (median 4.6e-7), numpy f32 1.1e-5, HF
  1.1e-5; top-1 462/462 each. ember is the closest of the three to float64, and its decode equals its
  prefill bit for bit, so the larger step gap is mostly on HF's cached side.

CUDA and WASM are not gated on this checkpoint: CUDA needs Colab, and the tiny random GPT-2 already
runs CUDA against the CPU at 960/960 (`../thirdparty_2026-10-22/device_gate/`).

## Rerun

On WSL ext4, each heavy step through `scripts/cpu_budget.py` (2 cores): `prep.sh` (copies the
checkpoint and paperkiln's tools and receipt to `~/gpt2_gate`, checks the sha256s), `build.sh`,
`run.sh all` (check, f64, greedy, f64g). The checkpoint stays outside the repository
(`C:/ml_artifacts/thirdparty/gpt2-124m/`).
