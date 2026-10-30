# KV-cache quantisation on the IR interpreter (R2, 28 Oct 2026)

`EMBER_KV=f32|f16|q8|q4` chooses how the IR interpreter stores its KV cache. Unset (or `f32`) is the default,
and every path runs as before. The receipts are from the official Qwen3-0.6B (Q8_0 file) and
Qwen2.5-0.5B-Instruct (f16 file) on a Colab L4 and on the CPU, over real long inputs at 2K, 4K and 8K
positions.

- **f16 pays on Qwen3-0.6B at long contexts, and its error does not register.** It halves the KV cache (1,792
  to 896 MiB at 8,192 positions) and decodes 1.08x, 1.14x and 1.24x faster at 2K, 4K and 8K. Logits stay within
  1.6e-3 to 3.9e-3 of the row range (mean 3-5e-5), top-1 agrees at 99.8-100% of positions, and perplexity
  moves by at most 0.01%. 200-token greedy replies equal F32's on 16 of 18 prompts; the other two part at
  tokens 69 and 88.
- **q8 and q4 do not hold up on these models.** With per-token blocks of 32 (llama.cpp's q8_0 and q4_1 rules),
  q8 moves perplexity by -0.03% to +3.8% and agrees on top-1 at 91-96% of positions. It leaves the F32 greedy
  path within 0-76 tokens on 17 of 18 prompts; the 18th is a Qwen2.5 8K prompt with room for 20 tokens. q4 multiplies perplexity 3.5 to 94 times. On Qwen3 at 8K they decode
  1.36x and 1.41x faster than F32, against f16's 1.24x.
- **Qwen2.5-0.5B gains nothing in any format.** Its 2 KV heads of 64 make an F32 cache of 192 MiB at 8K,
  against about 1.8 GiB of F32 weights read per token. Decode is 104-108 tok/s in every format.
- **The device kernels quantise as specified.** At 2K the CPU form (each stored row replaced by its
  dequantised value, the rounding checked against numpy) reproduces the L4's numbers: equal F32
  perplexities, q8 max error 0.238 and 0.338 of the range on both, top-1 within 8 positions of 2,048.
- **Not measured: the CUDA gates.** The default mode's byte-identity on the device, each format's run-to-run
  determinism and the device-against-CPU gate (`ir_device_gate`) have not run. Nor have the decode-path teacher
  forcing and Qwen2.5's bench at 8,064 positions. The session that ran them stalled before its first gate (see
  "Colab sessions"), and GPU work is stopped for the week. On the CPU the default mode is byte-identical to
  HEAD's (below).

## What is built

`kv_quant.h` / `kv_quant.cu` (new), wired into `graph_ir.cpp` / `graph_ir.h`:

| format | storage per element | rule |
|---|---|---|
| `f32` | 4 bytes | the existing cache, untouched |
| `f16` | 2 bytes | IEEE half, round to nearest even |
| `q8` | 1.0625 bytes | blocks of 32 channels of one position, inside a head: d = half(max\|x\| / 127), q = clamp(rint(x / d), ±127), x' = q d |
| `q4` | 0.625 bytes | blocks of 32, asymmetric: d = half((max - min) / 15), m = half(min), q = clamp(rint((x - m) / d), 0..15), x' = fma(q, d, m) |

Every division, product and fma is rounded once, with rint rounding half to even. So the CPU form and the CUDA
kernels quantise a given float to the same code. `f32_to_f16` equals numpy's float16 on 1,247,809 values
(every 4,099th bit pattern plus 200,000 normals), and the q8 and q4 round trips equal a numpy reference of the
rules on 4,096 blocks with an outlier channel.

- **Decode** (T = 1, the captured CUDA graph included): `decode_attn_tiled.cuh`'s split-K kernel with its
  cache loads replaced. It keeps the same chunking (32-key chunks, at most 64 per head), lane layout and
  fixed-order reductions, with no atomics, and reuses its merge (`tiled_merge_block`) unchanged. Each float4
  slice is dequantised as it is loaded. The position comes from device memory under the graph, as for F32.
- **Prefill**: the chunk's rows are quantised into the cache, then rows [0, start_pos + T) are dequantised into
  one F32 scratch shared by the layers, and the existing prefill kernel reads that. So prefill and decode both
  see what the cache holds.
- **CPU**: the interpreter keeps its F32 cache and replaces each written row by its dequantised value. That
  gives the format's numerics without its memory saving, and it is the reference for the device kernels.
- **Refused at load**: turn_state, windowed attention and depth_route on CUDA; `mla_attention` (its latent
  cache has its own write path) on both; q8 and q4 with a head_dim that is not a multiple of 32.
- **Memory**: the scratch is one layer's F32 K and V plus the decode partials, allocated only when a quantised
  format is set (67 MiB for Qwen3 at 8,448 positions, 8 MiB for Qwen2.5).
- **Not touched**: the tuned llama path, including its legacy INT8 cache (`use_kvcache_quantization`), which
  is off by default and kept out of graph decode.

## Results

### On the L4 (`receipts/colab_l4_eval`, session ember-r2c-3)

The eval tool is `tools/kv_quant_eval.cpp`: one process holds an F32 interpreter and one per format, all with
a kv capacity of 8,448 positions (`EMBER_IR_CTX`). The inputs (`tools/kvq_inputs.py`, tokenised through each
model's HF tokenizer, which equals ember's) are:
- "book": the first 8,448 tokens of *Pride and Prejudice* (Project Gutenberg #1342 from chapter I; source
  sha256 3f6bb9d6, excerpt 04dd13a7);
- "chat": whole UltraChat test conversations (`HuggingFaceH4/ultrachat_200k` test_sft) through the chat
  template, joined to 8,192 tokens;
- greedy prompts at each C: two questions over book excerpts (at offsets 0 and 12,000 tokens) and one
  multi-turn UltraChat conversation ending in a user turn.

Columns:
- **Errors**: teacher forcing through the input in 512-token prefill chunks (the cache filled chunk by chunk),
  |dlogit| as a fraction of the F32 row's range. The maximum over positions [0, C), and the mean.
- **Top-1 and perplexity**: over the same positions. Perplexity is of the input's next token, as a change
  from F32's.
- **Greedy**: 200 tokens after a prompt of about C tokens. The count equal to F32, with the first differing
  index per prompt in brackets ("-" when equal).
- **Decode**: the median of 120 steps after a C-token prefill.

**Qwen3-0.6B** (Q8_0 file; 28 layers, 8 KV heads x 128; F32 weights on the device)

| C | KV | max err / range (book, chat) | mean err / range | top-1 agree | ppl change | greedy = f32 | decode tok/s | KV MiB at C |
|---|---|---|---|---|---|---|---|---|
| 2048 | f32 | - | - | - | ppl 22.448 / 13.979 | - | 78.2 | 448 |
| 2048 | f16 | 1.6e-03, 3.3e-03 | 3.3e-05, 3.7e-05 | 99.9%, 100.0% | -0.01%, -0.00% | 3/3 (-,-,-) | 84.2 (1.08x) | 224 |
| 2048 | q8 | 2.4e-01, 3.4e-01 | 4.4e-03, 3.6e-03 | 93.3%, 95.2% | +1.25%, -0.03% | 0/3 (11,7,0) | 86.2 (1.10x) | 119 |
| 2048 | q4 | 6.4e-01, 7.3e-01 | 7.0e-02, 8.0e-02 | 24.6%, 26.0% | +1022%, +1470% | 0/3 (0,0,0) | 87.1 (1.11x) | 70 |
| 4096 | f32 | - | - | - | ppl 21.557 / 9.786 | - | 68.4 | 896 |
| 4096 | f16 | 1.7e-03, 3.3e-03 | 3.4e-05, 3.8e-05 | 99.8%, 100.0% | -0.00%, -0.00% | 3/3 (-,-,-) | 77.9 (1.14x) | 448 |
| 4096 | q8 | 2.7e-01, 3.9e-01 | 4.3e-03, 3.8e-03 | 93.3%, 95.5% | +1.20%, +0.27% | 0/3 (8,0,69) | 81.9 (1.20x) | 238 |
| 4096 | q4 | 6.4e-01, 8.0e-01 | 7.3e-02, 8.7e-02 | 21.2%, 24.1% | +1897%, +2845% | 0/3 (0,0,0) | 83.8 (1.22x) | 140 |
| 8192 | f32 | - | - | - | ppl 22.528 / 7.838 | - | 54.3 | 1792 |
| 8192 | f16 | 2.4e-03, 3.3e-03 | 3.6e-05, 3.9e-05 | 99.9%, 100.0% | -0.00%, -0.01% | 2/3 (69,-,-) | 67.6 (1.24x) | 896 |
| 8192 | q8 | 2.7e-01, 3.9e-01 | 3.9e-03, 3.5e-03 | 93.6%, 96.1% | +1.04%, +0.33% | 0/3 (5,0,60) | 74.0 (1.36x) | 476 |
| 8192 | q4 | 6.4e-01, 8.0e-01 | 7.6e-02, 9.2e-02 | 19.9%, 21.3% | +2274%, +4389% | 0/3 (0,0,0) | 76.3 (1.41x) | 280 |

**Qwen2.5-0.5B-Instruct** (f16 file; 24 layers, 2 KV heads x 64; F32 weights on the device). The file declares
8,192 positions (`qwen2.context_length`), so it has no 8K bench, and its 8K greedy prompts leave room for only
16-20 tokens.

| C | KV | max err / range (book, chat) | mean err / range | top-1 agree | ppl change | greedy = f32 | decode tok/s | KV MiB at C |
|---|---|---|---|---|---|---|---|---|
| 2048 | f32 | - | - | - | ppl 15.430 / 9.895 | - | 106.7 | 48 |
| 2048 | f16 | 3.9e-03, 3.7e-03 | 5.2e-05, 4.7e-05 | 99.9%, 100.0% | +0.01%, +0.00% | 3/3 (-,-,-) | 107.5 (1.01x) | 24 |
| 2048 | q8 | 3.8e-01, 3.6e-01 | 6.1e-03, 5.2e-03 | 91.4%, 93.4% | +3.09%, +2.72% | 0/3 (15,9,76) | 106.7 (1.00x) | 13 |
| 2048 | q4 | 5.7e-01, 7.7e-01 | 5.6e-02, 5.9e-02 | 46.1%, 50.7% | +253%, +300% | 0/3 (0,0,2) | 106.8 (1.00x) | 8 |
| 4096 | f32 | - | - | - | ppl 15.097 / 6.773 | - | 104.0 | 96 |
| 4096 | f16 | 3.9e-03, 3.7e-03 | 5.0e-05, 4.7e-05 | 99.9%, 99.9% | +0.01%, +0.00% | 2/3 (88,-,-) | 105.5 (1.01x) | 48 |
| 4096 | q8 | 4.9e-01, 4.5e-01 | 6.6e-03, 6.0e-03 | 91.3%, 93.8% | +3.75%, +2.15% | 0/3 (0,49,2) | 104.6 (1.01x) | 26 |
| 4096 | q4 | 9.3e-01, 7.7e-01 | 6.2e-02, 7.0e-02 | 34.6%, 32.2% | +951%, +4887% | 0/3 (0,0,0) | 105.0 (1.01x) | 15 |
| 8192 | f32 | - | - | - | ppl 15.957 / 5.218 | (16-20 tokens) | - | 192 |
| 8192 | f16 | 3.9e-03, 3.7e-03 | 4.9e-05, 4.4e-05 | 99.9%, 99.9% | +0.00%, +0.01% | 3/3 (-,-,-) | - | 96 |
| 8192 | q8 | 4.9e-01, 4.5e-01 | 6.7e-03, 6.3e-03 | 90.8%, 94.7% | +3.80%, +1.60% | 1/3 (0,-,0) | - | 51 |
| 8192 | q4 | 9.3e-01, 7.7e-01 | 6.5e-02, 7.8e-02 | 27.8%, 24.7% | +1427%, +9293% | 0/3 (0,0,2) | - | 30 |

Prefill time is unchanged by every format (Qwen3 at 8K: 5,924 ms F32, 5,888-5,988 ms quantised).

### On the CPU against the L4 (`receipts/cpu_2k`)

`kv_quant_eval --cpu 1`, one thread on WSL: the same inputs and phase at 2,048 positions (EMBER_IR_CTX 2048).
This is the format's rounding with no device kernel, as a reference for the kernels.

| model | KV | max err / range, CPU | L4 | top-1 of 2,048, CPU | L4 | ppl change, CPU | L4 |
|---|---|---|---|---|---|---|---|
| Qwen3 | f16 | 1.24e-03, 3.06e-03 | 1.55e-03, 3.32e-03 | 2045, 2046 | 2045, 2047 | -0.00%, -0.00% | -0.01%, -0.00% |
| Qwen3 | q8 | 2.38e-01, 3.37e-01 | 2.38e-01, 3.38e-01 | 1903, 1951 | 1911, 1949 | +1.22%, +0.23% | +1.25%, -0.03% |
| Qwen3 | q4 | 7.90e-01, 7.11e-01 | 6.40e-01, 7.29e-01 | 509, 547 | 503, 532 | +1111%, +1307% | +1022%, +1470% |
| Qwen2.5 | f16 | 3.79e-03, 4.28e-03 | 3.94e-03, 3.71e-03 | 2045, 2048 | 2046, 2047 | +0.01%, -0.01% | +0.01%, +0.00% |
| Qwen2.5 | q8 | 3.76e-01, 3.63e-01 | 3.75e-01, 3.59e-01 | 1870, 1920 | 1872, 1912 | +3.19%, +2.61% | +3.09%, +2.72% |
| Qwen2.5 | q4 | 5.60e-01, 7.64e-01 | 5.67e-01, 7.68e-01 | 938, 1051 | 944, 1039 | +261%, +284% | +253%, +300% |

The F32 perplexities are equal to four decimals (Qwen3 22.4482 and 13.9787, Qwen2.5 15.4304 and 9.8953). The
format rows differ by what F32-level differences upstream do to a quantiser: an element within about 1e-6
of a rounding boundary takes the other code.

### Why q8 fails here: Qwen's key channels (`kv_stats.py`, `receipts/kv_stats`)

The data is the CPU interpreter's F32 kv rows after a 512-position prefill of the book (`tools/kvq_dump_kv.cpp`).

| | Qwen3 K | Qwen3 V | Qwen2.5 K | Qwen2.5 V |
|---|---|---|---|---|
| largest channel / median channel (max \|x\| per head), median over heads | 10.4 | 2.2 | 3.2 | 1.9 |
| the same, worst head | 98.7 | 13.7 | 45.5 | 5.3 |
| relative error, q8 per-token blocks of 32 (shipped) | 0.98% | 0.60% | 0.75% | 0.64% |
| q8 one block per head and position | 1.75% | 0.78% | 0.90% | 0.76% |
| q8 one scale per channel (KIVI's key layout) | 0.49% | 0.90% | 0.46% | 0.81% |
| relative error, q4 per-token blocks of 32 (shipped) | 10.7% | 8.3% | 9.7% | 8.6% |
| q4 one scale per channel | 6.1% | 12.4% | 5.7% | 11.8% |

The relative error is ||X - X'|| / ||X|| per layer, the mean over layers. The keys have outlier channels, and
per-channel key scales halve the q8 key error and cut q4's by 40%. The values gain nothing from them. A q8 key
error of about 1% already moves these models by the figures above. Halving it is unlikely to bring q8 near
f16, which stores each element to half's 11 significant bits.

## Gates

| gate | result |
|---|---|
| CPU, default mode: ir_logits (prefill rows and token-by-token decode, 4 seeded sequences of 24 ids) from ember HEAD and from HEAD + R2, byte for byte | **byte-identical** on 9 IR families: Qwen2.5-0.5B, Qwen3-0.6B, Gemma-3-270m (windows), Pythia-160m, Mamba-130m, and E4's synthetic deepseek2 (MLA), qwen3moe, gemma2 (soft-capping) and phi3_longrope |
| CPU, `EMBER_KV=f32` against HEAD's default | byte-identical on all 9 |
| CPU, the formats in force (`EMBER_KV=q8` differs from f32 wherever there is an attention kv cache) | differs on Qwen2.5, Qwen3, Gemma-3 and Pythia; Mamba (no kv cache) equal; deepseek2 refused (`mla_attention`, under f16), qwen3moe refused (head_dim not a multiple of 32) |
| CUDA compile check of HEAD + R2 (nvcc `kv_quant.cu` for sm_89; g++ `graph_ir.cpp`, `kv_quant_eval.cpp` with HAS_CUDA) | passes |
| CUDA build and run of HEAD + R2 on an L4 (ember-r2c-3: `kv_quant_eval`, `ir_device_dump`, `ir_device_gate`, sm_89) | built; every eval above ran on it |
| CUDA, default mode byte-identical to HEAD (`ir_device_dump`, default and `EMBER_DETERMINISTIC=1`), each format run to run, device against the CPU form (`ir_device_gate`) | not run (GPU leg pending: `LOCAL_LLM_WORK.md` E9) |

On CUDA the default mode is unchanged by construction. Every new device statement sits behind
`kv_fmt_ != f32`, and the F32 allocation loop gains only that condition. The new kernels have fixed reduction
orders and no atomics. Neither is a measurement.

The CPU gate (`receipts/cpu_gate/gate.txt`, `cpu_gate.sh`, `cpu_gate2.sh`; one thread on WSL) builds ember
origin/main f393955 and the same tree with R2's patch, each CPU-only, and runs `ir_logits` at EMBER_IR_CTX
256 on four seeded sequences of 24 ids. The ids run from 100 to 50,000, or from 4 to 200 for the synthetic
files, whose vocabularies are smaller. The same gate passed on ember 43d6a33 before E4's two commits (Gemma 2,
Phi-3, DeepSeek-V3) landed.

## Against the papers

- **2604.04722** ("Don't Waste Bits!", adaptive KV quantisation for on-device LLMs): a controller picks 2, 4 or
  8 bits or FP16 per token, on SmolLM 135M, 360M and 1.7B. It reports 17.75% lower decode latency than static
  KV quantisation and stays within 0.30 points of FP16 on HellaSwag (SmolLM-360M). Its baseline is static
  per-token quantisation, the layout measured here. Qwen's 0.5-0.6B models lose 1-4% perplexity with it at
  8 bits.
- **2602.23200** (InnerQ): it groups along the inner dimension, mixes symmetric and asymmetric groups, keeps
  windows of recent and sink tokens at high precision, and normalises the key cache per channel (folded into
  the weights). It reports 1.3x over earlier KV quantisation and 2.7x over unquantised decode (Llama, Mistral).
  Qwen's key channels are the case its per-channel normalisation is for. On real keys a per-channel layout
  halves q8's key error (0.98% to 0.49% on Qwen3).
- **ember's ceiling is the weight bytes.** The device path holds F32 weights, about 2.2 GiB for Qwen3-0.6B,
  read every token. So removing the KV entirely at 8K could not pass about 1.8x, and f16 takes 1.24x of it.
  Both papers measure against FP16 weights, where KV is a larger share.

## Files

- `receipts/colab_l4_eval/`: bench, greedy, tf_prefill and meta per model, the inputs and their hashes
  (`models.sha256`), hardware (L4, driver 580.82.07, CUDA 13.0).
- `receipts/cpu_2k/`: the CPU runs (meta, tf_prefill) at 2,048 positions.
- `receipts/kv_stats/`: `kv_stats.py`'s output for both models.
- `receipts/cpu_gate/`: the CPU default-mode gate (`gate.txt`, with each output's sha256 prefix) and its two scripts.
- `kv_stats.py`: channel statistics and three layouts' reconstruction error on real kv rows.
- Tools: `tools/kv_quant_eval.cpp`, `tools/kvq_inputs.py`, `tools/kvq_report.py` (the tables from the receipts),
  `tools/kvq_dump_kv.cpp`, `tools/colab_kvq_vm.sh`, `tools/colab_kvq_drive.sh`.

## Colab sessions

Four L4 sessions, about 187 minutes in all (about 15 units at 4.8 units an hour):

| session | minutes | outcome |
|---|---|---|
| ember-r2 | about 23 | reclaimed while building; nothing measured |
| ember-r2-67 | about 25 | reclaimed during the Qwen3 eval; nothing came home |
| ember-r2c-3 | about 23 | reclaimed after both models' bench, greedy and tf_prefill; those receipts came home (the VM zips them every 2 minutes and the driver fetches them every poll) |
| ember-r2d-13 | 116 | the gates and the decode-path teacher forcing; it stalled before the first gate and ran to its deadline |

- **The reclaims.** The first three VMs were reclaimed 23-25 minutes in, each time as another session was
  created while the account held three L4s (pk-r3-bit, ember-e6-rec, pk-r4-base). From ember-r2d-13 on, the
  driver runs a trivial `exec` every poll.
- **The stall.** ember-r2d-13 ran `wait` before g1, meaning to wait for the base build, but a bare `wait` also
  waited for the receipt packer, which runs until the job ends. `colab_kvq_vm.sh` now waits on the build's pid.
- **Rerunning.** One session with
  `bash tools/colab_kvq_drive.sh <session> 115 <stage> g1,g2,g3,tfd_qwen3,tfd_qwen25 L4`, from a stage of
  ember HEAD and the inputs, reruns everything still open.

## E9 f16 gate campaign, 30 October 2026

The rerun selects only f16 (`KVQ_FORMATS=f16`), because q8/q4 are not adoption candidates. Existing
model/input hashes and the original 40 short prompts, two 3,072-id prompts, 32 generated tokens,
12-prompt CPU/device gate and 1e-4 tolerance remain fixed. Both models' decode teacher forcing is
compared against the corresponding stored prefill observations, without increasing their bounds.
The recorded prefill figures are measurements, so a larger decode error is reported as a failed
comparison, not silently accepted under a new tolerance.

`colab_kvq_vm.sh` now fails closed: dump/evaluation errors, byte differences or a failed CPU/device
process create GATE_FAILED; dump sizes must match the requested logits/id inventory; model hashes
are verified before use; `kvq_gate_check.py` checks all expected comparisons, inputs and JSON evidence
before GATE_DONE. Binary scratch is removed only after its byte count/hash and comparison are recorded.
Build concurrency is bounded (`KVQ_BUILD_JOBS`, default 4). Twelve offline validator tests pass.

The RunPod command-only job is `microtorch/tools/colab_chat/ember_e9_device_20261030.json`. The root
watchdog owns launch and the $3 cap. The wrapper always returns diagnostic artifacts, including an
explicit false gate summary on timeout; FINISHED means receipts are home, not that the science gate
passed. Inputs are staged from the original `C:/ml_artifacts/kvq/ember-r2c-3/` files. The 1.9 GB of
GGUFs are downloaded on the VM, then verified against the prior full SHA-256s. No local GPU is used.
This paragraph describes tested preparation; it is not a device result.

If a default-mode byte comparison differs, the old build is repeated with the same mode and prompt.
An unstable old control is recorded as BASELINE_UNSTABLE (unresolved), never as feature regression or
pass. A stable old control still leaves a differing new result as a failed byte-identity gate. The
wrapper retains this classification in gate_summary.json. Eighteen offline gate/wrapper tests pass.
