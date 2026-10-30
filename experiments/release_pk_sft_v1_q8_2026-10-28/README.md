# pk_sft_v1 release gate for the browser Space, 28 Oct 2026

Published 30 Oct 2026: Space commit `8e107b9706f81fec53749fa59da7602344e6c776`, website commit
`fdbaa1deded14881cfbe70c10526c1fe915c2c3d`. All 13 remote files match the staged manifest.
See `published_2026-10-30.json` for the publication verification; the measurements below are the prior gates.

LOCAL_LLM_WORK.md E2.2. The gate for a quantised Space build (LOCAL_LLM_STACK.md section 3, point 1:
greedy parity, plus a measured logit error): the file in ember against F32 ember, greedy 40/40 on the
frozen chat prompts. **Q8_0-all (118 MB) fails it: 37/40.** Of the files between it and F32 (441 MB),
every Q8_0 variant fails the same way and **F16 (221 MB) passes**, with WebAssembly = native and
WebGPU (SwiftShader) = WebAssembly 40/40. The selection rule (the smallest file that passes 40/40
within the contract) picks F16. The 28 Oct staged artifacts were published on 30 Oct
(sections "Variants", "The F16 file on the page's paths", "Release staging and publication").

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
  heap 1gb, Emscripten 6.0.6; `ember.wasm` b880bec5...; at 2gb by `e22_build2gb.sh`, `ember.wasm`
  23f8d86e...), and `e22_page_gate` (the page's `ember_wasm.cpp` included whole, built natively and
  for node with NODERAWFS, so both sides make the page's own calls).
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

**Q8_0-all against F32, greedy (`q8_server.jsonl`): 37/40.** Reply 37, ids 37, token count 38, stop
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

## Variants (`variants/`, `e22_variants.sh`, `e22_variant_cmp.py`)

Files between Q8_0-all and F32, made with existing tools from `pk_sft_v1.gguf` and put through the
same gate (greedy against F32's replies; teacher-forced logits on F32's greedy ids, 733 positions).
pk_sft_v1's embedding table and LM head are tied: the GGUF stores the same 16,384 x 768 table twice
(`token_embd.weight` = `output.weight` byte for byte; 110.1M stored values for 97.5M parameters).

| file | made by | bytes | greedy vs F32 (ids) | rows 7 / 10 / 14 | max \|dlogit\| (over row range) | mean \|dlogit\| | top-1 |
|---|---|---:|---:|---|---|---:|---:|
| Q8_0-all (embeddings, head Q8_0) | paperkiln export `gguf_q8_0_all` | 117,699,712 | 37/40 (7, 10, 14) | flip / flip / flip | 0.181 (0.51%) | 0.0172 | 730/733 |
| q8_head: Q8_0, head F32 | `tools/quantize_gguf.py --embed-q8` | 154,662,016 | 37/40 (7, 14, 31) | flip / same / flip | 0.165 (0.47%) | 0.0144 | 730/733 |
| q8emb_f32: Q8_0, embeddings and head F32 | `tools/quantize_gguf.py` (= paperkiln `gguf_q8_0`) | 191,624,320 | 37/40 (7, 14, 31) | flip / same / flip | 0.140 (0.42%) | 0.0141 | 730/733 |
| **f16: every matrix F16, norms F32** | llama.cpp b6000 (4762ad7) `llama-quantize ... F16` | **220,919,008** | **40/40** | same / same / same | **0.0068 (0.019%)** | **0.00065** | **733/733** |
| F32 | paperkiln export | 441,119,872 | reference | | | | |

sha256: q8_head 22e9907f..., q8emb_f32 a8d651f6..., f16 3916b3ae617f01b6610e8c6a8ebfab301227c83cad2d40b33d90d548a639b957
(`variants/*.sha256`). Keeping the head (and the embeddings) at F32 lowers the error by a fifth and
moves which near-ties flip (row 10 holds; row 31 flips at its token 13, "friends and family" for F32's
"friends and enjoying nature", where F32's top-1 margin is 0.0198), but the error that flips them
is the Q8_0 matrices inside the blocks. F16's mean |dlogit| (0.00065) is about a twentieth of theirs, and it keeps every greedy choice, including prompt 14's 0.0129 margin (0.0125 in F16).
Each reply ends at the end-of-text token in every file. The F16 file keeps every metadata field of the
F32 one (chat template, Jinja, `paperkiln.ir` and its digest), adding only `general.file_type` and
`general.quantization_version`; ember serves it on the tuned llama path, dequantised to f32 at load.
The Q8_0 variant files were deleted after recording; the commands above remake them.

**Selection** (the main session's rule: the smallest file that passes 40/40 within the contract):
F16, 221 MB. Its error is within every bound used before (release_chat.py's 1% of the range: 0.019%).

## The F16 file on the page's paths (`wasm_f16/`, `webgpu_f16.json`)

- **WebAssembly against native, the page's calls** (`e22_wasm_leg.sh`, `e22_wasm_cmp.py`):
  `e22_page_gate greedy` natively and in node (2gb), 40 prompts: reply, token count, stop reason,
  prompt ids, ids and stop id 40/40, and the native run equals the server's 40/40 on every field.
  The logits of every sampled position (733 rows) are not bit-equal: max |dlogit| 5.7e-6, 2.0e-7 of
  the row range (the f32 sums run in a different order under WebAssembly SIMD).
- **The Space's own ember.js** (`deploy/wasm/build.sh`, 2gb) through `tools/wasm_parity.mjs` against
  the native server's F16 replies: reply 40/40, token count 40/40, stop reason 40/40, 4,456 ms per
  reply in node on one loaded core.
- **Heap:** at the 1gb ceiling `ember_load` throws std::bad_alloc (the F16 matrices are dequantised
  to f32); at 2gb it loads and the WebAssembly memory is 1.48 GB after load (Q8_0-all: 0.82 GB).
  The staged page is the 2gb build.
- **First load in the page's ember.js** (node, one loaded core, `e22_load_time.mjs`): instantiate
  95 ms; the file into the filesystem and `ember_load` 26.5 s; the first reply (12 tokens) 4.0 s.
  Q8_0-all on the same build at 1gb: 14.6 s and 6.0 s. Download, computed from the size: 552 s at
  0.4 MB/s, 88 s at 2.5 MB/s (20 Mbit/s), 18 s at 12.5 MB/s (100 Mbit/s); Q8_0-all 294 / 47 / 9 s.
- **WebGPU on SwiftShader** (`e22_webgpu.py`: `tools/webgpu_parity/chat_test.mjs` from the 9d993f4
  tree through `swiftshader_run.py`, Dawn node 0.6.2, 1 core, the 2gb ember.js; the worker's own
  steps with `llama_webgpu.js`, which dequantises F16 to f32 storage buffers; `webgpu_f16.json`):
  WebGPU reply = WebAssembly reply 40/40 (reply, stop reason, token count), WebGPU = native server
  40/40, WebAssembly = native 40/40, 693 tokens, no WebGPU errors. Adapter `google | swiftshader |
  swiftshader-device-subzero- | SwiftShader driver 5.0.0`; the GPU-counter monitor checked 518 times
  and saw no context on a hardware adapter. 141 s per reply on one core (SwiftShader runs WGSL on the
  CPU; this says the path is correct, nothing about its speed on a GPU), 5,899 s in all.

## Release staging and publication

- `~/e22rel/space` (WSL ext4, 214 MB): the layout of `tools/build_wasm_space.sh` from the 9d993f4
  tree: `README.md` (the Space card, `staged/README_space.md`), `index.html` (`staged/index.html`),
  `chat.js`, `worker.js`, `webgpu/`, `style.css`, the 2gb `ember.js` + `ember.wasm`, `chat.gguf` (the
  F16 file), `.gitattributes` (LFS for .gguf and .wasm), `SOURCE.txt`, and `MANIFEST.sha256` (every
  file's sha256, `staged/MANIFEST.sha256`; checked by `publish_space.py` before any upload). Built by
  `e22_stage.sh`.
- The F16 file's home: `C:/ml_artifacts/general_chat/sft/pk_sft_v1_f16.gguf` (sha256 3916b3ae...);
  `~/e22rel/space/chat.gguf` is the upload's working copy.
- The site: `staged/chat.html` for `~/JohnnyTeutonic.github.io/chat.html` (and its source copy
  `website/chat.html`): pk_sft_v1's size, data, evaluation and licences; the download size 221 MB,
  16-bit (the live page says 71 MB, while the live Space serves a 19 MB model).
- `publish_space.py`: reads only the `HF_TOKEN=` line of the env file, never prints it, checks the
  manifest and that the account is johnnyteutonic, needs `--confirm ember-chat`, and makes one commit
  to the Space (files of the same name replaced, nothing deleted; the live Space gains `webgpu/`).

Publication commands and history: `PUBLISH.md`.

## Reproduce (WSL; every heavy step under scripts/cpu_budget.py, 1 core)

    bash e22_build.sh; bash e22_gate.sh native; python3 e22_compare.py ~/e22rel/run <paperkiln parity_cpu.jsonl> summary.json
    bash e22_variants.sh                     # q8_head q8emb_f32 f16
    bash e22_build2gb.sh; HEAP=2gb PAGE=~/e22rel/page2gb bash e22_wasm_leg.sh ~/e22rel/variants/pk_sft_v1_f16.gguf f16 ~/e22rel/variants/f16_server.jsonl
    python e22_webgpu.py                     # Windows
    bash e22_stage.sh

The scripts ran from the session scratchpad (`SP` in them); `e22_page_gate.cpp` is copied into the
source tree at build time.
