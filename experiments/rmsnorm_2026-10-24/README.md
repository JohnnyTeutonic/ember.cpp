# CUDA prefill rmsnorm at widths that are not a power of two (24 Oct 2026)

paperkiln's post job spot-checks ember against `mtstudio chat --kv-cache` (greedy, 40 Carry-5 items,
`tools/colab_chat/tiny_chat_run.py`). On the E1 L32 SFT model (`tc_sft_L32_r1.gguf`, sha256 1187e448...,
llama, d 192, 32 layers, 3 heads, head_dim 64, d_ff 832, llama3 vocabulary capped at 16384) ember at
cbdcb91 on an L4 (CUDA, gpu_layers -1, EMBER_GEMM_TF32=0) gave 0/40 items and 28/320 replies identical.
The E5 L8 models (d 448) gave 40/40.

## Cause

`rmsnorm_batch_cuda` (cuda_kernels.cu) launched one block of `hidden_size` threads per row when
`hidden_size <= 256`. The kernel's tree reduction halves `blockDim.x` at each step
(`for (s = blockDim.x / 2; s > 0; s >>= 1)`), so it needs a power of two. At d = 192 it runs s = 96, 48,
24, 12, 6, 3, 1, and at s = 3 the partial of thread 2 is never added: every element with index % 3 == 2,
a third of the sum of squares, drops out, while the mean still divides by 192. Every rmsnorm of the batch
prefill (attention norm, FFN norm and final norm of every prompt row) is wrong; the per-token decode path
(`rmsnorm_vector_cuda_optimized`, 256 threads) is right, but it attends to the prompt's wrong K/V. The
first reply token comes straight from the prefill's final norm, which is why the divergence shows at the
first reply with history_used 0. Speculative verify and the IR interpreter's group rmsnorm call the same
function.

Shapes: any width up to 256 that is not a power of two (96, 160, 192, 200, 255 fail in the kernel test).
d 320 (L16) and d 448 (L8) are above 256 and get 256 threads, so they were never affected; layer count,
head count and d_ff play no part.

**The fix** (cuda_kernels.cu, the launcher only): the block is `hidden_size` rounded up to a power of two,
capped at 256; the extra threads read nothing and contribute 0. Power-of-two widths up to 256 and every
width above 256 get exactly the launch they had, so their results are bit-identical.

**History.** Not a regression: the launcher and the kernel's reduction date from dc82798 / 532920e
(22-24 May 2025) and are unchanged since (`git log -L`). No ember commit served d 192 correctly on CUDA.
The kernel test fails identically on cuda_kernels.cu at b72ade8 (first BPE chat mode), ef4b30f (first
Llama 3-scheme chat mode), cbdcb91 (the pin) and 2886a97 (main before the fix). CPU builds never compile
this file.

## Evidence

Receipt: `rmsnorm_gate.json` (everything below), raw records in `receipts/`. CUDA runs: one Colab T4
(sm_75, session `ember-rms`, stopped at the end; `drive.sh`, `vm.sh`); the T4 at the pin reproduces the
post job's L4 ember records exactly (320/320 replies), so it is the same failure.

| check | pin cbdcb91 | 2886a97 | fix |
|---|---|---|---|
| kernel test (`tools/rmsnorm_batch_test.cu`, 18 widths, float64 reference) | 5 widths fail (worst rel. error 0.50) | same | pass (worst 1.3e-7) |
| kernel outputs vs 2886a97, bit for bit | | | equal at 32, 64, 128 and 256-1024; differ only at 96, 160, 192, 200, 255 |
| spot-check, exact SFT L32 on CUDA vs mtstudio's records (items, replies) | 0/40, 28/320 | 0/40, 28/320 | 40/40, 320/320 (also cbdcb91 + fix) |
| spot-check, exact SFT L32 on ember **CPU** (home) vs mtstudio's records | | 40/40, 320/320 | |
| 40 frozen tinychat prompts, SFT L32, CUDA vs mtstudio CPU | 5/40 | | 40/40 |
| 40 frozen tinychat prompts, sp_L16 (d 320), CUDA vs mtstudio CPU | 40/40 | | 40/40; ids equal to the pin 40/40 |
| ember CPU vs mtstudio CPU (home), 40 prompts: SFT L32, sp_L16, E1 L32 base, E5 L8 base | | 40/40 each | |

L8 on CUDA: the E5 post jobs' own spot-checks at the pin, on an L4 (post_e5_s0_r1, post_e5_s17_r1), are
40/40 items and 320/320 replies; d 448 gets the same launch before and after the fix.

**Repetition penalty: ruled out.** carry5_eval sends no repeat fields, so both engines use their defaults:
repeat_penalty 1.0 (off) in mtstudio and in ember at cbdcb91 and at 2886a97 (2886a97 changed only the
repeat_last_n default to min(64, ctx), and ctx 512 >= 64 makes it 64 at both commits); ember's older
CTRL-style penalty is 1.0 in chat mode. Every comparison above ran with the spot-check's body (`spot`)
and with explicit repeat_penalty 1.0, repeat_last_n 64 (`p1`): the counts are the same in both.

**Numerical sensitivity: ruled out.** `logit_gap.py`: ember's full logit rows (TINYLLAMA_LOGITS_DUMP,
9 digits; `dumpprec.py` is a VM-only print patch) against paperkiln's numpy reference (`tools/ir_numpy.py`
on the SFT GGUF and its IR) for the first four spot items' first replies:

| item | first divergent step (pin) | max abs logit diff vs numpy, pin CUDA | fix CUDA | ember CPU | top-1 minus top-2: numpy / pin CUDA |
|---|---|---|---|---|---|
| recall:0001 | 11 (numpy: end-of-text) | 3.83 (step 0: 4.05) | 6.4e-6 | 5.7e-6 | 0.368 / 0.138 |
| recall:0027 | 0 | 4.04 | 6.4e-6 | 7.6e-6 | 0.165 / 0.811 |
| recall:0053 | 2 | 2.75 (step 0: 1.73) | 6.7e-6 | 6.7e-6 | 0.217 / 0.009 |
| recall:0079 | 6 (numpy: end-of-text) | 2.11 (step 0: 2.89) | 6.2e-6 | 5.2e-6 | 0.326 / 0.373 |

The pinned CUDA logits are off by 1.7-4.1 already at the prefill row, where no token has diverged yet;
the fixed CUDA path and the CPU path are within 8e-6 of numpy (float32 rounding), and the reference's
margins at the divergent steps are 0.17-0.37, not near-ties.

**The mechanism on the CPU** (`emulate_bug.py`): numpy's reference with its rmsnorm changed for the
prompt rows only (elements with index % 3 == 2 left out of the sum, divided by 192) reproduces ember
CUDA's buggy first replies exactly on 8/8 spot items ("Hi Yuki, I'm Yuki. I'm Yuki.", "Bramble is a
tortoise that is a tortoise that is a tortoise.", ...), and the unmodified reference reproduces
mtstudio's on 8/8.

## Existing receipts

Unchanged. CPU and WebAssembly builds do not compile cuda_kernels.cu. On CUDA the launch differs only for
widths up to 256 that are not powers of two, and no model of an existing gate has one: the local GGUFs'
embedding widths are 32, 64, 128, 256, 384, 640, 768, 896 and 1024, and the IR group rmsnorm runs on
head dims 64-256 (powers of two). For those widths the kernel outputs are bit-equal before and after,
and sp_L16 (d 320) gives the same ids at the pin and with the fix on all 40 prompts.

## Files

- `tools/rmsnorm_batch_test.cu` (the regression test): `nvcc -O2 -std=c++17 -DHAS_CUDA -I.
  tools/rmsnorm_batch_test.cu cuda_kernels.cu logger.cpp -lcublas -o rmsnorm_batch_test`; exit 0 on pass;
  an optional argument dumps the raw outputs.
- `vm.sh`, `drive.sh`, `stage.sh`: the Colab run (builds cbdcb91, cbdcb91 + fix, 2886a97, 2886a97 + fix).
- `spot_cpu.sh`, `parity_cpu.sh`, `cmp_spot.py`, `sp_compare.py`: the spot-check and prompt comparisons.
- `make_specs.py`, `train.sh`: sp_L16 (shape_20m_L16's arch, 2 CPU steps of mtstudio; T 128, a 1 MB
  head of the DailyDialog text).
- `emulate_bug.py`, `logit_gap.py`, `analyze.py`, `finalize.py`: the analyses and this receipt.
- Logit dumps (21 MB) and the VM's full outputs: `/mnt/c/ml_artifacts/ember_rmsnorm/ember-rms/`.

## Open

- The baseline prefill attention kernel (`attention_batch_prefill_cuda_kernel`, used only when flash
  attention is off or head_dim is not a multiple of 32 or above 128) also reduces over a block of
  `head_dim` threads with the same halving loop, so a head_dim such as 80 would hit the same class of
  bug there. No current model has one; not changed here.
- Not rerun on an L4 (three L4s were busy with paperkiln's campaigns). The post job's next spot-check at
  the fixed commit is the L4 confirmation.
