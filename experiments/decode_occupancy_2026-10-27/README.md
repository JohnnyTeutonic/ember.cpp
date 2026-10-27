# Decode attention at small contexts: the tiled split-K kernel (27 Oct 2026)

paperkiln's lead (`microtorch/experiments/exact_kv_skip_2026-10-27/NOTE.md`): at 384 keys one decode attention layer
of pick_e1_L16 reads 0.98 MB of K and V, 3.3 µs at an L4's 300 GB/s, against ~36 µs measured, on 10 thread blocks for
58 SMs. Measured here on two Colab L4s: the kernel itself takes that time, it is bound by latency and occupancy, and a
new kernel brings it to 5.5-6 µs. It is now the default: **the decode step is 1.53-1.59x faster on pick_e1_L16 and
1.76-1.80x on pick_e1_60m_L32 at 256-500 cached tokens**, with greedy ids unchanged on every request, the
deterministic mode bit-exact run to run, speculative verification bit-identical to plain decode under the mode, and
the CPU build byte-identical.

## 1. Diagnosis

**The old kernel** (`decode_attn_split_kernel`, now `EMBER_DECODE_ATTN=split`). Grid (heads, chunks): 256-key chunks
for the cache's *capacity*, not the current length. The picks' caches hold 512 positions, so every decode step ran
2 chunks per head, 10 blocks for pick_e1_L16 (5 heads) and 12 for pick_e1_60m_L32 (6), at any length. Inside a
block, 4 warps take one key each per pass (q·k by a 32-lane shuffle tree), and then 64 of the 128 threads walk the
chunk's keys one after another for P·V. Both loops are chains of dependent global loads, so a block's time grows with
its keys (up to 256) at roughly 0.14 µs per key, while 48 of the 58 SMs sit idle.

**In situ** (`receipts/cuda_s2/insitu/`, `receipts/insitu_summary.txt`): `occ_step` runs the engine's decode step
(the captured CUDA graph) at one cached length per process under `nsys --cuda-graph-trace=node`; each kernel's median
time inside the graph, per layer:

| keys | pick_e1_L16: split + merge -> tiled + merge (µs) | | pick_e1_60m_L32: split + merge -> tiled + merge (µs) | |
|---|---|---|---|---|
| 33 | 6.82 + 1.89 -> 3.23 + 2.11 | 1.6x | 6.82 + 1.86 -> 3.23 + 2.05 | 1.6x |
| 129 | 19.26 + 1.92 -> 3.20 + 2.08 | 4.0x | 26.18 + 1.86 -> 4.13 + 2.05 | 4.5x |
| 257 | 35.52 + 2.05 -> 3.30 + 2.24 | 6.8x | 67.39 + 2.02 -> 5.79 + 2.21 | 8.7x |
| 385 | 37.28 + 2.02 -> 3.68 + 2.37 | 6.5x | 68.43 + 1.98 -> 7.30 + 2.37 | 7.3x |
| 501 | 44.03 + 2.02 -> 4.26 + 2.40 | 6.9x | 68.58 + 1.98 -> 8.58 + 2.37 | 6.4x |

So the ~36 µs is the attention kernel's own time inside the graph (not launch gaps, not other kernels). On
pick_e1_60m_L32 it is 68 µs: the model's 292 MB of weights pass through the 48 MB L2 between one layer's attention and
the next step's, so its K and V come from DRAM (the synthetic bench's "cold" column gives the same 73 µs).

**Synthetic, per layer** (`occ_bench.cu`; `receipts/bench_summary.txt`, `receipts/cuda_s1/bench/`): the copied old
kernel and the tiled one on random caches of the picks' shapes (head_dim 64), a float64 CPU reference, two runs
compared bit for bit. "Graph" is the time per split + merge pair inside a CUDA graph of 64 pairs (CUDA events);
"hot" / "cold" are the two kernels' own times under nsys with the cache in L2 or after a 128 MB write that evicts it.
Floor: K + V bytes at the L4's 300 GB/s (the measured copy rate is 233 GB/s). Measured copy bandwidth and grid shape
are in each record.

| shape | keys / cache | old: blocks, graph µs, hot, cold | tiled (32, 64): blocks, graph µs, hot, cold | floor µs |
|---|---|---|---|---|
| pick_e1_L16, 5 heads | 128 / 512 | 5 of 10, 20.1, 18.6+1.9, 37.8+2.4 | 20 of 80, 4.9, 3.4+1.9, 7.9+2.3 | 1.09 |
| | **384 / 512** | 10 of 10, **36.9**, 35.3+2.0, 72.8+2.5 | 60 of 80, **5.45**, 3.7+2.3, 13.0+2.6 | **3.28** |
| | 128 / 4096 | 5 of 80, 20.9, 18.7+2.7, 37.8+3.7 | 20 of 320, 5.3, 3.8+1.9, 7.7+2.3 | 1.09 |
| | 1024 / 4096 | 20 of 80, 37.6, 35.3+2.7, 76.4+3.6 | 160 of 320, 6.7, 4.1+3.0, 20.0+4.3 | 8.74 |
| | 4096 / 4096 | 80 of 80, 38.7, 35.9+3.1, 101.6+3.4 | 320 of 320, 11.3, 7.5+4.3, 62.3+5.2 | 34.95 |
| pick_e1_60m_L32, 6 heads | 384 / 512 | 12 of 12, 36.9, 35.3+2.0, 73.2+2.3 | 72 of 96, 5.5, 3.7+2.2, 14.3+2.6 | 3.93 |
| | 4096 / 4096 | 96 of 96, 38.7, 36.0+3.1, 112.0+3.4 | 384 of 384, 12.1, 8.4+4.3, 74.4+5.0 | 41.94 |
| pk_base_v1, 12 heads | 384 / 512 | 24 of 24, 37.0, 35.3+2.0, 75.1+2.3 | 144 of 192, 5.9, 4.1+2.2, 18.8+3.5 | 7.86 |
| | 4096 / 4096 | 192 of 192, 40.2, 37.5+3.1, 151.9+3.4 | 768 of 768, 21.3, 17.4+4.7, 142.2+5.1 | 83.89 |

("5 of 10": blocks with keys, of blocks launched.) The old kernel's time is set by its longest block (256 keys:
~36 µs), whatever the length. Smaller chunks alone (the old kernel's code at 32 keys per chunk, `split_c32`) take
the 384-key case to 8.9 µs; the rest of the gain is inside the block. At 384 keys the tiled kernel is 1.7x the DRAM
floor in the graph (5.45 µs), its split kernel alone 1.1x (3.7 µs), from L2; with L2 cold it reads at about 76 GB/s
(13 µs). At 4096 keys with L2 cold it reaches 168 GB/s (62 µs against 45 µs at the copy rate). What remains at small
lengths is a fixed ~2 µs merge launch and ~3 µs of block latency.

## 2. The tiled kernel (`decode_attn_tiled.cuh`)

- **Chunks by the current length.** chunk = max(32, ceil(L / 64)), nchunks = ceil(L / chunk): 32-key chunks up to
  2,048 keys, then 64 chunks per head. The grid is sized for the cache's capacity (min(64, ceil(max_len / 32)) chunk
  blocks per head) so the CUDA graph captures it once; blocks past nchunks(L) return at once. L comes from device
  memory in the graph, as before.
- **Inside a block** (128 threads): q·k by quads of 4 lanes per key with float4 loads (32 keys per pass, a 2-step
  butterfly); the chunk's softmax by warp shuffles and per-warp slots added in warp order; P·V by groups of
  head_dim / 4 lanes (8 groups at head_dim 64), each adding its keys in order into a float4, then the groups' partial
  sums added in group order. The merge: the chunks' maximum, their weights e^{m_c - M} computed once, L by a fixed
  shuffle tree with explicit roundings, the chunks added in index order.
- **Fixed order everywhere**, so the result is the same run to run, and it depends only on L: the per-launch decode,
  the graph decode and `attention_decode_at_cuda` give the same bits at equal L (tested). The deterministic mode
  therefore keeps it.
- **Speculative verification** (`verify_kernels.cu`) runs the same block bodies per row (`attn_tiled_rows_kernel`),
  so a verification row is the decode step's arithmetic by construction; its scratch has room for either kernel.
- **Where it applies:** head_dim % 4 == 0 and <= 256, at most 128 heads, 16-byte-aligned q, K and V, up to 131,072
  positions. Anything else falls back to the 256-key kernel, and past that to the one-block-per-head kernel, as
  before. Selection: `EMBER_DECODE_ATTN` unset (tiled), `split` (the old kernel, unchanged), `legacy`;
  `EMBER_DECODE_ATTN_TILE=m,x` changes the chunking (default 32,64; `docs/USAGE.md`).
- **The chunking**: the bench's grid of (min chunk 8-128) x (max chunks 16-64) on three shapes and five lengths
  (`receipts/bench_summary.txt`): (32, 64) is within 0.6% of the best at 384 keys, within 4-7% at 128 and 1,024 keys
  (12% at 128 keys in a 4,096-position cache, where it launches 320 blocks for 20 with keys), and within 3-4% at 4,096
  keys on 5 and 6 heads; on 12 heads at 4,096 keys, longer chunks (8, 32) are 22% faster. In the decode step (`receipts/step_summary.txt`, second session), (16, 64) is within 0.8% of (32, 64) and
  (64, 64) 0.6-1.7% slower.

## 3. Gates

All CUDA legs on Colab L4s (sm_89, CUDA 13.0), the change built Release beside ember main 7526469 ("HEAD") from a git
archive; pick_e1_L16 (`C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf`, sha256 96f6afdc...) and
pick_e1_60m_L32 (`.../e1_60m/pick_e1_60m_L32/model.gguf`, 54a4180e...).

| gate | result | receipt |
|---|---|---|
| decode attention vs float64, 4 entry points, 6 shapes (head_dim 6-128, GQA, caches 256-8192, lengths across both kernels' chunk edges) | max \|err\| < 1e-4 everywhere for tiled, split, legacy, tiled under the mode, (16, 64) tiling; tiled: per-launch = devpos = decode_at bit for bit and run to run bit-identical at every length | `receipts/cuda_s2/dat/` |
| greedy, 360 requests (40 tinychat + 320 Carry-5), HEAD vs this build, default mode | replies, stop, ids **360/360** on both models; worst logit \|d\| at any sampled position 8.8e-6 (pick), 1.2e-5 (60M), at most 6.1e-7 of the row's logit range. HEAD against itself in default mode (pick): 1.0e-5 (the decode rmsnorm's atomics) | `receipts/cuda_s1/plain/cmp_*` |
| `EMBER_DETERMINISTIC=1`, this build twice | **bit-identical at every position**, 360/360 requests, 4,599 (pick) and 5,145 (60M) positions | `cmp_*_new_det_vs_new_det2.json` |
| the mode, HEAD vs this build with `EMBER_DECODE_ATTN=split` | bit-identical at every position, both models: the old kernel is reachable unchanged | `cmp_*_head_det_vs_new_det_split.json` |
| the mode, HEAD vs this build (tiled) | ids 360/360, worst \|d\| 8.2e-6 / 1.6e-5 | `cmp_*_head_det_vs_new_det.json` |
| speculative verification under the mode (all60; `EMBER_VERIFY_LINEAR` unset = rows), plain vs k = 2, 4, 8 and rows set | **every row bit-identical**: 1,409/1,409 (pick), 1,283/1,283 (60M), ids 60/60; default mode k = 4: ids 60/60 | `receipts/cuda_s1/vd/` |
| end-to-end ids (all140), tiled vs split, both modes | 140/140 on both models in every run | `receipts/cuda_s2/e2e/` |
| CUDA vs the CPU build, all140 | pick 140/140; 60M 139/140 with tiled and with split alike (one copy20 request, an existing CUDA/CPU difference) | `receipts/cpu/e2e/`, `receipts/gates_summary.txt` |
| interpreter (`ir_device_dump`, 16 local gate models), HEAD vs this build | prefill bit-identical on all 16; greedy 40/40 and decode argmax equal on every row of all 16; decode logits within 3.4e-6 of HEAD's; state4, state4_q8_0, mamba and mistral_swa (paths the change does not reach) bit-identical; decode vs prefill, ROUNDING ratio, on the 12 models it reaches: this build 2.3e-7 to 1.6e-6, HEAD 2.3e-7 to 1.5e-6 | `receipts/cuda_s2/dumps/` |
| SASS, tinyllama_server, kernel by kernel | all 104 of HEAD's kernels identical; 4 added (the tiled kernel and merge, their verification-row forms) | `receipts/cuda_s1/sass.json` |
| CPU build (WSL, HAS_CUDA=OFF): HEAD's tree, then this tree, in one build directory | tinyllama_server, ir_logits and ir_adapter **byte-identical** (ir_adapter sha256 a73412f6, keramos v1.3.4's receipt binary), so every CPU receipt stands | `receipts/cpu/cpu_identity.txt` |

The VM builds of one tree are not byte-reproducible across VMs (HEAD's tinyllama_server hashes differ between the two
sessions too), so the two sessions' binaries are compared by source: the second session's tree is the first's plus
`tools/decode_attention_test.cu`'s run-to-run check, now applied to the fixed-order kernels only (in the first session
the legacy kernel's default form, which adds with atomics, failed that check on 25 lengths, with every error under
1e-4; under the mode it passes).

## 4. Speed

**The decode step** (`occ_step`: the captured graph's replay, logits to pinned memory, sync; p50 of 300;
`receipts/step_summary.txt`). Tokens per second = 1000 / step ms, split -> tiled:

| keys | pick_e1_L16 (20M) | | pick_e1_60m_L32 (60M) | |
|---|---|---|---|---|
| 33 | 985 -> 1045 | 1.06x | 428 -> 449 | 1.05x |
| 129 | 826 -> 1045 | 1.27x | 333 -> 441 | 1.32x |
| 257 | 679 -> 1044 | 1.54x | 232 -> 418 | 1.80x |
| 385 | 679 -> 1036 | 1.53x | 226 -> 403 | 1.79x |
| 501 | 646 -> 1027 | 1.59x | 226 -> 396 | 1.76x |

Under `EMBER_DETERMINISTIC=1` the same within 1%. On pick_e1_L16 the step no longer grows with the context below 257
keys: attention is off its critical path there.

**End to end** (`vf_e2e.py` through tinyllama_server, greedy, each request twice and the second recorded; split and
tiled alternated, two runs each; `receipts/cuda_s2/e2e/`). Decode tokens per second, best of the two runs:

| set (prompt tokens) | pick_e1_L16 split -> tiled | | pick_e1_60m_L32 split -> tiled | |
|---|---|---|---|---|
| chat40 (13-22) | 724 -> 752 | 1.04x | 361 -> 377 | 1.05x |
| copy20 (97-145) | 653 -> 802 | 1.23x | 287 -> 381 | 1.33x |
| c12_40 (245-295) | 446 -> 627 | 1.41x | 178 -> 302 | 1.70x |
| long40 (320-430) | 471 -> 653 | 1.39x | 183 -> 310 | 1.69x |

The request sets are `occ_prompts.py`'s. The replies are short (mean 16-17 tokens on chat40, 27-36 on copy20, 6-8 on
c12_40 and long40), so per-request overheads weigh more here than in the step measurement.

## Files

- `decode_attn_tiled.cuh` (engine root), `cuda_kernels.cu` (`decode_attn_tiled`, `decode_attn_device`,
  `decode_attention_kind`, `decode_attention_tiles`), `verify_kernels.cu`, `tools/decode_attention_test.cu`.
- `occ_bench.cu` (synthetic kernel bench), `occ_step.cpp` (the decode step in situ; built as `tools/occ_step`),
  `occ_prompts.py` (request sets), `dump_compare.py` (interpreter dumps), `occ_summary.py`, `insitu_summary.py`.
- `build.sh` (WSL trees and compile checks), `stage.sh`, `drive.sh`, `vm.sh` (session `ember-occ-1027`, 34 min: its VM
  was reclaimed during the end-to-end leg, after legs (a)-(f) had come home), `stage2.sh`, `vm2.sh` (session
  `ember-occ2-1027`, 17 min: the end-to-end and interpreter legs again, the other tilings, the in-situ profile),
  `cpu_legs.sh` (CPU identity and CPU references).
- `receipts/cuda_s1/`, `receipts/cuda_s2/`: VM logs, driver logs, hardware, binaries' and models' sha256, every
  record; `receipts/cpu/`; the summaries `receipts/*_summary.txt`. Full VM output also in
  `C:/ml_artifacts/ember_occupancy/`.

Run (WSL): `git archive --prefix=ember/ <HEAD>` to `<in>/ember_head.tgz`; `bash stage.sh <in> <HEAD> ~/occ/stage &&
bash drive.sh <session> 115 ~/occ/stage`; `bash stage2.sh ~/occ/stage ~/occ/stage2 && bash drive.sh <session> 100
~/occ/stage2`; `bash cpu_legs.sh <in>/ember_head.tgz ~/occ/cpu`; `python3 occ_summary.py bench|step|gates <out>`.
