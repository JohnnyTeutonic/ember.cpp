# pk_sft_v1 Q8_0 release gate for the browser Space, 28 Oct 2026

LOCAL_LLM_WORK.md E2.2. The gate for a quantised Space build (LOCAL_LLM_STACK.md section 3, point 1:
greedy parity, plus a measured logit error): Q8_0 ember against F32 ember, greedy 40/40 on the frozen
chat prompts. **Q8_0 fails it: 37/40.** The release was not staged and nothing was uploaded. F32
(441 MB) is the fallback, on Jonathan's word.

## Files (sha256 checked against LOCAL_LLM_STACK.md section 10, 25 Oct entry)

| file | sha256 | record |
|---|---|---|
| `C:/ml_artifacts/general_chat/sft/pk_sft_v1.gguf` (441,119,872 B) | 4d53c7ff236ecba3a2b34cd2d4769f08bdf0408bbc3655afa44b9ddc70929fdd | equal (full hash on record) |
| `pk_sft_v1_q8_0_all.gguf` (117,699,712 B) | 9eda55fd608222e5b108a8e62f64ed905b74ee0aa9015d071db3f6e1e54fa0cb | equal to the recorded prefix 9eda55fd |
| `pk_sft_v1.safetensors` (390,158,632 B) | eae61755dc6625eb87c54ce3152774596ed4299c65c0462ed3831b0f9d7ee80e | equal to the prefix eae61755 and to the full hash in paperkiln's `chat_general/receipts/general_chat_1025/evals2/manifest.json` |

## Setup

- ember 9d993f4 (`git archive`, so the working tree's uncommitted `ir_webgpu.js` is not in it), built
  on WSL ext4 in `~/e22rel` by `e22_build.sh`: native CPU build (`-DHAS_CUDA=OFF`, Release, g++ 13.3.0;
  `tinyllama_server` sha256 cdb0c12a...), the page's own WebAssembly build (`deploy/wasm/build.sh`,
  heap 1gb, Emscripten 6.0.6; `ember.wasm` b880bec5...), and `e22_page_gate` (the page's
  `ember_wasm.cpp` included whole, built natively and for node; prepared for the WebAssembly leg,
  not run because the Q8_0 leg failed first).
- Prompts: the first 40 of `tinychat_hf_prompts_v1.jsonl` (sha256 a906b25a..., the same file in
  `C:/ml_artifacts/chat_base/sft/` and paperkiln's `corpus/eval/`). Request: the F32 gate's
  (`tools/chat_parity.py`): 60 new tokens, temperature 1, top_k 1, top_p 1, seed 0, the paperkiln/1
  template the GGUF declares (`/health`: chat_template declared, context 512, vocab 16,384), with
  `return_ids`. `e22_chat_client.py` checks `/health` names the right file before any request.
- One CPU budget slot (`OMP_NUM_THREADS=1`): F32 333 s, Q8_0 458 s for the 40 replies.

## Results (`summary.json`)

**F32 ember against mtstudio.** On record: the 25 Oct gate (paperkiln's
`microtorch/experiments/chat_general/receipts/general_chat_1025/parity/`, ember aa50a61, CPU and
CUDA): 40/40 on reply, token count, stop reason and prompt tokens. This build (9d993f4) gives the
recorded mtstudio replies 40/40 on reply, token count and stop reason, prompt tokens equal to
llama3_bpe.py's 40/40, 40 distinct replies (`f32_server.jsonl`).

**Q8_0 against F32, greedy (`q8_server.jsonl`): 37/40.** Reply 37, ids 37, token count 38, stop
reason 40, prompt ids 40. Every reply ends at the end-of-text token in both files. The three that
differ part mid-reply at a near-tie in F32:

| prompt | first differing token | F32 / Q8_0 continuation | F32 top-1 margin there | Q8_0's margin for its own pick | max \|dlogit\| in that row |
|---:|---:|---|---:|---:|---:|
| 7 | 10 | "...when the water is so dark and cold." / "...when the water feels so heavy and heavy." | 0.0319 | 0.0189 | 0.082 |
| 10 | 11 | "...the color of leaves, isn't it?" / "...the color of leaves in unexpected ways." | 0.0258 | 0.0043 | 0.075 |
| 14 | 13 | "...especially with the new hospital being so busy." / "...especially with the fluorescent lights." | 0.0129 | 0.0362 | 0.068 |

In each case Q8_0's choice is F32's second choice (the F32 margin is the gap to it). Q8_0's replies
read as fluent, but prompt 7's repeats a word.

**Logit error, teacher-forced** (`e22_page_gate tf`: `score_rows` over prompt ids + F32's greedy
ids, every position that predicts a reply token or the stop: 733 positions over the 40 prompts,
`tf_f32.f32.json`, `tf_q8.f32.json`; the binaries were deleted after these numbers were taken):
max |dlogit| 0.181 (0.51% of that row's range; 0.42% of the global range), mean |dlogit| 0.0172
(0.057% of the row range on average). Top-1 agreement 730/733: the three misses are exactly the
three divergence points above. The F32 top-1 margin goes down to 0.0126 (1st percentile 0.022),
below the Q8_0 error in the same rows (up to about 0.09), so near-ties flip. The F32 argmax on this
path equals the served F32 greedy token at all 733 positions.

Under release_chat.py's older Q8 criterion (first-token top-1 40/40, max |dlogit| <= 1% of the
range) this file passes: all three misses are mid-reply. Under this release's whole-reply rule it
does not. Earlier Q8_0 files met the same pattern: pk_base_v1 36/40, chat_sftfinal_v2 q8all 36/40,
fw_moe 25/40.

**WebAssembly against native, WebGPU on SwiftShader:** not run. The gate stops at the Q8_0 leg;
the page bundle was built but not staged.

## Reproduce (WSL)

    python3 scripts/cpu_budget.py run --session ember --cores 1 --job ... --resume ... -- bash e22_build.sh
    python3 scripts/cpu_budget.py run --session ember --cores 1 --job ... --resume ... -- bash e22_gate.sh native
    python3 e22_compare.py ~/e22rel/run <paperkiln parity_cpu.jsonl> summary.json; python3 e22_check_tf.py

The scripts ran from the session scratchpad (`SP` in them); `e22_page_gate.cpp` is copied into the
source tree at build time.
