# Real k-quant blocks from paperkiln: pk_base_v1 Q4_K_M and Q4_K_S, 25 Oct 2026

paperkiln now stores real Q4_K, Q5_K and Q6_K blocks wherever ne0 is a multiple of 256 (paperkiln
255c9561, 3a3a8169); on pk_base_v1 its 4-bit files are byte-identical to llama.cpp b6000's
llama-quantize run on the F32 export. These are the first paperkiln files with K-quant tensors in
them (at d 384 every k-quant fell back to a legacy type, `../q4_2026-10-20/`).

| file (C:/ml_artifacts/next_base/base/) | tensors | sha256 |
|---|---|---|
| `pk_base_v1.gguf` (F32, d768 L12, 97.5M, 16k Llama 3-scheme vocabulary) | 111 F32 | 510d288c... (as `../pk_base_v1_2026-10-24/`) |
| `pk_base_v1_q4_k_m.gguf` (69.2 MB) | 73 Q4_K, 13 Q6_K, 25 F32 | 78b3482f3ba3bc050ba1c6d019803b898f6c480d5773e2b75a552613d2e8272f |
| `pk_base_v1_q4_k_s.gguf` (66.4 MB) | 80 Q4_K, 5 Q5_K, 1 Q6_K, 25 F32 | 7980aa549815a20c5626a51cf5ac4249be08bfa4c84b0366d9f026837d6105ca |

In both files `token_embd` is Q4_K and `output` Q6_K; Q5_K is `attn_v` of blocks 0-3 and `ffn_down` of
block 0 (q4_k_s).

## What changed in ember

Before: q4_k_m loaded; **q4_k_s did not** (`Unknown GGMLType in ggml_type_block_size: 13`, the process
exits): ember had no Q5_K at all, and `ggml_type_size` gave it 116 bytes, not 176.

- **Q5_K, native blocks** (`block_q5_K` {d, dmin, scales[12], qh[32], qs[128]}, `dequantize_q5_k`, a port of
  ggml-quants.c `dequantize_row_q5_K` with the same float operations in the same order). The llama path keeps
  Q5_K tensors in their blocks (`*_q5k`, any of the seven projections, the token embedding and the head) and
  dequantises them to the f32 copy before use, exactly as it does Q4_K and Q6_K.
- **The interpreter** reads Q4_K, Q5_K and Q6_K: rows stay in their 256-wide blocks for large models on the
  CPU and are dequantised row by row (`dequant_span_to_f32`), small models and CUDA dequantise at load
  (`dequant_legacy_to_f32`). Before, any K-quant tensor was refused at load.
- **Per-weight eviction.** The CPU forward frees the f32 copies of layer l-2 when they can be re-derived;
  the check looked only at `q_proj`, so a layer mixing sources could have lost an f32-only weight. It is now
  per weight (and knows Q5_K).
- **The CPU's batched LM head** (the prefill's last-token logits) now uses the f32 copy, as the decode step
  does; a Q5_K head has no kernel of its own. Before, a quantised head took its own kernel on a server's
  first request (the f32 copy did not exist yet) and the f32 copy on later ones. Those kernels accumulate in
  double as the f32 one does, and on pk_base_v1's Q8_0 and Q4_K_M files (Q8_0 and Q6_K heads) the old first
  request, a later request and 9238a86 give bit-identical first-token logits (`first_request.py`); no
  committed receipt moved (`regression_9238a86.txt`).
- `TINYLLAMA_LOGITS_DUMP` writes nine significant digits, so a dump reads back as the same float32 and two
  dumps compare bit for bit (it wrote six).
- New: `tools/kquant_test.cpp` and `tools/kquant_blocks.py` (below).

**How ember computes with K-quant weights:** it dequantises them (the llama path lazily, to an f32 copy;
the interpreter row by row or at load) and multiplies in F32 with the same kernels as an F32 file (double
accumulation on the CPU, cuBLAS f32 on CUDA). There are no in-block dot products on any path that serves
these files, which is why the logits below are bit-identical to an F32 file holding gguf-py's values.

## (1) Kernel test: ember's dequantisers = gguf-py's, bit for bit

`tools/kquant_test` against gguf-py from llama.cpp b6000 (4762ad73; `~/llamacpp_b6000/gguf-py`, the same
checkout cloned on the VM):

- **Random blocks** (`tools/kquant_blocks.py`, 4,096 per type, seed 20261025: every byte random except the
  f16 scales, drawn finite with zeros, negatives, subnormals and large values): 0 differing bits in
  1,048,576 values per type, through all three of ember's routes (the llama path's block routine,
  `dequant_span_to_f32`, `dequant_legacy_to_f32`), for Q4_K, Q5_K and Q6_K. A negative control (one
  reference bit flipped per type) fails as it should.
- **Every tensor of both files** against the file rewritten as F32 by gguf-py
  (`tools/quantize_gguf.py IN OUT --dequantize`; references sha256 471ce015... for q4_k_m, f7e7bda6... for
  q4_k_s): 86 quantised tensors each, 0 differing bits.

The same results on the Colab T4 VM's build (`cuda_t4/kquant_test_*.txt`).

## (2) Exactness: logits against gguf-py's dequantised F32 file

ember on the k-quant file against ember on the gguf-py-dequantised F32 file (the comparison chosen of the
two the gate allowed), next-token logits of the 40 frozen prompts (`tinychat_hf_prompts_v1.jsonl`, sha256
a906b25a...), 655,360 values per file:

| path | q4_k_m | q4_k_s |
|---|---|---|
| CPU, tuned llama path | **bit-identical** 655,360/655,360 | **bit-identical** 655,360/655,360 |
| CPU, IR interpreter (`EMBER_INTERPRETER=1`; K-quant rows kept in blocks) | bit-identical | bit-identical |
| CUDA (T4), tuned llama path | bit-identical | bit-identical |
| CUDA (T4), IR interpreter on the device | bit-identical | bit-identical |
| greedy replies, CPU and CUDA | 40/40 identical | 40/40 identical |

Worst difference 0: no float-summation gap, since ember multiplies the dequantised values with the F32
file's own kernels. For scale, the interpreter against the tuned path on the same file is 1.7e-7 of the
logit range (both files, and F32 itself: different kernels, ROUNDING).

## (3) Accuracy against the F32 model

First-token logits against ember's F32 model (CPU; the F32 model's own top-1 margins: smallest 0.0034,
10th percentile 0.053, median 0.44), and greedy replies (60 tokens) against mtstudio chat on the F32 model:

| | max abs dlogit | mean | max / logit range | first-token top-1 | replies = mtstudio F32 | replies = ember F32 |
|---|---|---|---|---|---|---|
| ember F32 | - | - | - | - | **40/40** (token counts, stops, prompt tokens 40/40) | - |
| q4_k_m | 1.28 | 0.173 | 4.6e-2 | 36/40 | 14/40 (token counts 15/40) | 14/40 |
| q4_k_s | 1.51 | 0.185 | 5.5e-2 | 36/40 | 14/40 (token counts 17/40) | 14/40 |

Prompt token counts equal paperkiln's tokeniser on 40/40 for every file. Each first-token flip is to the F32
model's second choice (`cuda_vs_cpu.json`): prompts 5, 9, 12, 22 (q4_k_m) and 5, 9, 12, 20 (q4_k_s), at F32
margins 0.003-0.13. The gguf-py-dequantised references give the same reply counts, as they must.

This base model is less robust to 4-bit than the d 384 chat model was (`../q4_2026-10-20/`: q4_k_m 0.50 max
dlogit, 38/40 first tokens, 23/40 replies): its logit range is wider in absolute terms and its top-1 margins
are small (a quarter of the first tokens within 0.18). The 4-bit files fail the Q8_0 gate's first-token
criterion (40/40) and are for size (69 / 66 MB against 118 MB for Q8_0 and 441 MB for F32), not fidelity.

**The mtstudio reference** is the 24 Oct record (`../pk_base_v1_2026-10-24/pk_base_v1_parity.jsonl`:
mtstudio chat, paperkiln c8f0b8af CPU build in `~/pk_cpu_1024`, on this sha256-identical F32 file, greedy).
It was not rerun whole (about 40 minutes of the budget); `mts_spot.py` reran the same mtstudio on the first
5 prompts today: 5/5 identical to the record (`mts_spot.jsonl`). ember F32 today: 40/40 against it.

## (4) CUDA: Colab T4 (sm_75), session `ember-kq-1025`, stopped by its driver

One CUDA build (the tree of this change, `COMMITS`: ember a4e7c8c plus this change's files, which equal the
committed ones line endings aside; the commits in between touched only the WASM gate's tool, README and
receipts) serves both columns: 0 GPU layers is the CPU path, -1 the device. T4 has no TF32.

| | q4_k_m | q4_k_s |
|---|---|---|
| CUDA vs the CPU, first-token logits | max 8.5e-6 (3.0e-7 of the range), top-1 40/40 | 6.7e-6 (2.4e-7), 40/40 |
| CUDA interpreter vs the CPU | 6.7e-6 (2.4e-7), 40/40 | 7.6e-6 (2.7e-7), 40/40 |
| CUDA vs gguf-py's dequantised F32 on CUDA | bit-identical | bit-identical |
| greedy replies, CUDA vs the CPU | **40/40** (token counts and stops 40/40) | **40/40** |
| greedy replies, CUDA vs mtstudio F32 | 14/40 | 14/40 |

The VM's CPU-path logits are bit-identical to the WSL CPU build's on both files, so its CUDA-vs-CPU
numbers are also CUDA against the receipts here.

## llama.cpp

No llama.cpp b6000 build exists locally (`~/llamacpp_b6000` holds sources and gguf-py only), so llama.cpp's
own numbers were skipped, as the gate allowed. paperkiln's byte-identity with llama-quantize stands for the
files themselves.

## Files

- `gate.py` (probe / replies / summary), `run_cpu.sh` (the CPU half, under `scripts/cpu_budget.py`, 2 cores),
  `summary.json` (every CPU comparison), `*_replies.jsonl` (CPU replies), `cuda_vs_cpu.py` / `.json`,
  `mts_spot.py` / `.jsonl`, `regression_9238a86.txt` (rerun_all.sh's gates, the parity replays, WASM and
  the IR adapter self-test 119/119 at 9238a86: no receipt moved by this change), `first_request.py` /
  `first_request_*.json`, `kquant_test_wsl.txt` (the kernel test and its negative control; "values
  compared" counts each K-quant tensor twice, once per route).
- `stage.sh`, `drive.sh`, `vm.sh` (the Colab session, path C of microtorch/AGENTS.md), `cuda_t4/` (its
  summary, replies, kernel tests, references' sha256, hardware, vm.log).
- Logit dumps (7.8 MB each) stayed in `~/kq/work` and were deleted after the summaries; the full VM output
  is in `C:/ml_artifacts/ember_kquant/ember-kq-1025/`.
