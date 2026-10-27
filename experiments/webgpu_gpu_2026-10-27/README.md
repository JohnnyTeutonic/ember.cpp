# The browser's WebGPU path on a hardware GPU, 27-28 Oct 2026

E6's remote half: the page's WebGPU forward pass (`deploy/wasm/web/webgpu/`: the tuned `llama_webgpu.js` and
the IR interpreter `ir_webgpu.js`) in real headless Chrome on an NVIDIA L4, against ember CPU (the EXACT
reference) and against the page's WebAssembly. GPU work is remote only: every run here is on a Colab L4 VM; the
laptop ran no browser and no GPU code.

## The adapter

Headless Chrome 156.0.8078.47 (`--headless=new`, Vulkan, `--use-angle=vulkan`, `--enable-unsafe-webgpu`;
`tools/webgpu_parity/remote_chrome.py`) offers the GPU as its WebGPU adapter:

| | Colab L4 VM (all results below) |
|---|---|
| GPU, driver | NVIDIA L4, driver 580.82.07 (CUDA 13.0), Vulkan 1.4.312 |
| Chrome's GPU (DevTools SystemInfo) | ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA NVIDIA L4 (0x000027B8)), NVIDIA-580.82.7.0) |
| `GPUAdapterInfo` | vendor `nvidia`, architecture `lovelace`, isFallbackAdapter false, subgroups 32 |
| Limits asked for | maxStorageBufferBindingSize 2147483644, maxBufferSize 4294967292, workgroup storage 49152 |
| Fallback adapter (`forceFallbackAdapter`) | SwiftShader, never used: every page takes the default adapter, and the driver refuses a run whose adapter names SwiftShader, llvmpipe or lavapipe |
| VM CPU (WebAssembly and ember CPU) | Intel Xeon @ 2.20 GHz, 12 vCPUs (6 cores, 2 threads each), 52 GB |

`gpu_probe.html` checks a compute shader on the adapter (4M floats, every value right) before any gate.
Colab's runtime mounts only the compute and utility parts of the driver (`NVIDIA_DRIVER_CAPABILITIES=compute,utility`),
so the first probe design (the platform's libraries plus glvnd's EGL) found no Vulkan device; the second
(`vm_probe.sh` design 2) copies the graphics libraries of the running driver version from NVIDIA's own installer
(`NVIDIA-Linux-x86_64-580.82.07.run --extract-only`: libGLX_nvidia, libnvidia-glcore, -glvkspirv, -gpucomp,
EGL and the ICD manifest; the CUDA libraries are left alone), and Chrome then sees the L4. A first probe on a
RunPod L4 (`runpod_probe/`, before E6 moved to Colab) showed the other half: there the runtime mounts the
graphics libraries, but the NVIDIA ICD dlopens glvnd's `libEGL.so.1`, without which vkCreateInstance fails
(ERROR_INCOMPATIBLE_DRIVER) and Chrome falls back to llvmpipe; with `libegl1` installed Chrome 156 saw the L4
(driver 595.91.07).

## Gate 1: logits against ember CPU (`colab_l4/gates/`)

`tools/webgpu_parity/gate_page.html` is `ir_gate.mjs` in a browser: tools/ir_wasm_gate's CPU reference (40
seeded prompts of 24 ids and 32 greedy tokens; 8 + 8 for the families whose IR declares fewer than 64 positions;
EMBER_IR_CTX 1024) built on the VM from the same ember tree, the prompt one token per forward, each position's
logits against the CPU's row, the argmax at every position, then the greedy continuation. The document is the
one ember's CPU ran (`ir_logits` with EMBER_IR_PRINT, `colab_l4/docs/`). WebGPU is ROUNDING: the gate is greedy
40/40 and top-1 at every position, and the distance is reported as max |dlogit| over the row's range.

| Family (model) | Greedy | Top-1 | Worst \|dlogit\| / range | Median | Dispatches per token | ms per forward |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma-3-270m-it** f32 | 40/40 | 960/960 | 9.2e-05 | 1.5e-05 | 382 | 50.9 |
| **Qwen2.5-0.5B-Instruct** fp16 | 40/40 | 960/960 | 3.6e-05 | 9.7e-06 | 483 | 98.1 |
| **Qwen3-0.6B** Q8_0 | 40/40 | 960/960 | 1.0e-05 | 3.6e-06 | 535 | 114.3 |
| Pythia-160m f32 | 40/40 | 960/960 | 2.3e-04 | 1.6e-05 | 243 | 28.1 |
| Mamba-130m f32 | 40/40 | 960/960 | 1.4e-03 | 6.5e-05 | 411 | 39.1 |
| kimi (relabelled kimi_tiny) | 40/40 | 960/960 | 6.8e-07 | 3.5e-07 | 38 | 2.7 |
| srd | 40/40 | 960/960 | 4.1e-07 | 3.3e-07 | 68 | 2.8 |
| attnres | 40/40 | 960/960 | 4.7e-07 | 3.8e-07 | 46 | 2.7 |
| flex highway | 40/40 | 960/960 | 4.9e-07 | 3.6e-07 | 60 | 2.8 |
| paperkiln mamba | 40/40 | 960/960 | 6.2e-07 | 4.4e-07 | 34 | 2.6 |
| gpt2 | 40/40 | 960/960 | 4.6e-07 | 3.3e-07 | 40 | 2.7 |
| gpt2_swa (relabelled swa_tiny) | 40/40 | 960/960 | 4.0e-07 | 3.3e-07 | 40 | 2.7 |
| expr: gpt2_tiny's GELU | 40/40 | 960/960 | 4.3e-07 | 3.3e-07 | 40 | 2.7 |
| expr: + final norm | 40/40 | 960/960 | 4.7e-07 | 3.3e-07 | 41 | 2.7 |
| llama MoE through the IR | 40/40 | 960/960 | 4.6e-07 | 3.7e-07 | 39 | 2.7 |
| recursion b2x4 | 40/40 | 320/320 | 2.5e-07 | 1.3e-07 | 139 | 3.2 |
| recursion p1_b2x3_c1 | 40/40 | 320/320 | 6.2e-07 | 1.4e-07 | 139 | 3.2 |
| recursion p1_b2x3_c1, inject | 40/40 | 320/320 | 2.4e-07 | 1.5e-07 | 156 | 3.2 |
| Mixture-of-Recursions mor3 | 40/40 | 320/320 | 2.5e-07 | 1.3e-07 | 147 | 3.2 |
| Mixture-of-Recursions mor2p0 | 40/40 | 320/320 | 2.5e-07 | 1.5e-07 | 130 | 3.2 |
| state4 (turn_state) | 40/40 | 960/960 | 1.1e-06 | 4.0e-07 | 37 | 2.6 |
| state4 Q8_0 | 40/40 | 960/960 | 1.3e-06 | 4.1e-07 | 37 | 2.6 |
| llama_attnres ar3 | 40/40 | 320/320 | 2.7e-07 | 1.5e-07 | 69 | 2.8 |
| ar3 Q8_0 | 40/40 | 320/320 | 2.2e-07 | 1.4e-07 | 69 | 2.8 |
| GPT-2 (tiny random, builder) | 40/40 | 960/960 | 7.6e-07 | 5.7e-07 | 58 | 2.7 |
| Mistral sliding window (tiny random) | 40/40 | 960/960 | 2.9e-06 | 1.3e-06 | 54 | 2.7 |

The three official checkpoints E6 names pass at 40 prompts each, within the receipts' 1e-4 bound. Every
greedy continuation is identical and every argmax agrees on every family. The paperkiln and tiny third-party
families sit within 2.9e-6 of the range, as on SwiftShader (`../webgpu_ir_2026-10-26/`). Pythia-160m and
Mamba-130m are past the flat 1e-4 bound (`pass` false on the bound alone), as on SwiftShader (1.6e-4, 3.3e-4)
and on CUDA: deep f32 checkpoints accumulated in f32 on a GPU against ember CPU's double accumulation; for an
official checkpoint the IR's gate is greedy N/N and top-1 at every position, with closeness judged against
float64 (IR_SPEC 4.3). Mamba-130m's worst row is larger on the L4 than on SwiftShader (1.4e-3 against 3.3e-4,
median 6.5e-5); its file here is the VM's conversion of state-spaces/mamba-130m-hf at the laptop's revision
(1e76775f) with llama.cpp b6000, whose sha256 (99dff2ad) differs from the laptop's file (ff213c3c), so the two
runs are not on identical weights; reference and WebGPU here read the same file. The other checkpoints match
the laptop's and the 25 Oct VM's hashes (`colab_l4/setup/thirdparty.sha256`, revisions in
`thirdparty_revisions.json`).

## Gate 2: the page's chat path (`colab_l4/pages/`)

`tools/webgpu_parity/page_test.html` in Chrome on the L4: the Space page's own worker (`worker.js`, the Space's
`ember.js`: single-threaded, SIMD, 1gb heap) on WebGPU and on `?backend=wasm`, 40 greedy requests, against ember
CPU's /chat replies from the same tree (`vm_fixture.py`: `tinyllama_server`, CPU build, one thread;
`colab_l4/fixtures/`). Requests: the 40 frozen tinychat prompts (60 new tokens), state4's 40 multi-turn requests
with their state blocks (8 new tokens).

| Model | GPU path | WebGPU = WASM | WebGPU = ember CPU | WASM = ember CPU | Reply tokens |
|---|---|---:|---:|---:|---:|
| chat_sftfinal_v3 (tuned llama path) | llama_webgpu | 40/40 | 40/40 | 40/40 | 636 |
| state4, multi-turn with state blocks | ir_webgpu | 40/40 | 40/40 | 40/40 | 200 |
| tc_e1_mor_MoR3_r1 (Mixture-of-Recursions chat) | ir_webgpu | 40/40 | 40/40 | 40/40 | 576 |
| mamba_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 |
| srd_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2367 |
| attnres_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 |
| flex_highway_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 |
| gpt2_tiny | ir_webgpu | 40/40 | 40/40 | 40/40 | 1705 |
| kimi_tiny (relabelled) | ir_webgpu | 40/40 | 40/40 | 40/40 | 2188 |
| swa_tiny (relabelled) | ir_webgpu | 40/40 | 40/40 | 40/40 | 2171 |

chat_sftfinal_v3's 636 reply tokens are the 21 Oct receipt's count. swa_tiny matches ember CPU 40/40 here
because the fixture is ember CPU's own replies to these requests (on 26 Oct its fixture was mtstudio's for
another request set).

## Throughput (`colab_l4/bench/`)

`tools/webgpu_parity/bench_page.html` in Chrome: ember's WebAssembly loads the model and the WebGPU forward
pass is the one `worker.js` picks; WebGPU replies run `worker.js`'s chatGpu step for step (ember_prepare, the
GPU's logits, ember_sample, ember_finish), WebAssembly replies are ember_chat (the page's `?backend=wasm`).
Greedy. Per run: replies per second over the request set, prefill ms per prompt token (the time to the first
logits: WebGPU's prefill forward, WebAssembly's max_new_tokens 1 reply, over the prompt's ids) and decode ms per
token ((reply time - prefill time) / decode forwards). Medians over the runs; the first 2 requests (1 for the
official checkpoints) ran once on each backend before the runs and are not counted. WebGPU replies equal
WebAssembly's in every run (40/40 or 6/6). WebAssembly: the Space's single-threaded SIMD build (`ember.js`,
1gb heap; `ember_4gb.js`, the same build with a 4gb heap, for the official checkpoints); one thread of the VM's
Xeon. Requests: the tinychat prompts (prompt 17.6 ids on average for chat_sftfinal_v3, mamba_tiny; 18.1 for the
MoR model; 25 Gemma, 45 Qwen2.5, 24 Qwen3 with each model's chat template), max_new 60 (32 for the official
checkpoints, 6 requests); state4's multi-turn requests (71 prompt ids, max_new 8). These are measurements on
this VM, not claims about other hardware.

| Model (GPU path) | Requests x runs | Replies/s WebGPU | Replies/s WASM | ms per reply WebGPU | WASM | Prefill ms/token WebGPU | WASM | Decode ms/token WebGPU | WASM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_sftfinal_v3 (llama_webgpu) | 40 x 3 | 15.1 | 3.05 | 66 | 328 | 0.78 | 7.7 | 3.1 | 12.0 |
| tc_e1_mor_MoR3_r1 (ir_webgpu, MoR) | 40 x 3 | 2.19 | 1.26 | 456 | 793 | 13.3 | 19.2 | 14.8 | 31.0 |
| mamba_tiny (ir_webgpu) | 40 x 3 | 5.28 | 16.2 | 189 | 62 | 1.03 | 0.16 | 2.86 | 0.99 |
| state4 (ir_webgpu, state tokens) | 40 x 3 | 24.7 | 136 | 40 | 7.3 | 0.34 | 0.08 | 2.73 | 0.33 |
| Gemma-3-270m-it f32 (ir_webgpu) | 6 x 2 | 0.362 | 0.159 | 2763 | 6483 | 47.8 | 51.4 | 56.8 | 188 |
| Qwen2.5-0.5B-Instruct fp16 (ir_webgpu) | 6 x 2 | 0.135 | 0.039 | 7429 | 26488 | 94.8 | 100.5 | 102 | 708 |
| Qwen3-0.6B Q8_0 (ir_webgpu) | 6 x 2 | 0.159 | 0.045 | 6312 | 23694 | 110.7 | 219 | 118 | 595 |

The 8-thread WebAssembly build (`ember-mt.js`, EMBER_WASM_THREADS=8, the page cross-origin isolated) on the
same requests: chat_sftfinal_v3 6.32 replies/s (158 ms per reply, decode 7.4 ms/token), Qwen3-0.6B 0.160
replies/s (6462 ms, prefill 56 ms/token, decode 165 ms/token).

What the numbers show on this VM:
- The tuned llama path is 5.0x the single-threaded WebAssembly per reply on chat_sftfinal_v3 (2.4x the 8-thread
  build), as the 21 Oct laptop iGPU run showed (166 against 646 ms there; 66 against 328 here).
- The IR path is ahead from a few million parameters up: 1.7x on the 40M MoR chat model, 2.3x on Gemma-3-270m,
  3.6x and 3.8x on Qwen3-0.6B and Qwen2.5-0.5B (decode 5.0x and 6.9x).
- On the tiny IR models (mamba_tiny, state4, under 3M parameters) WebGPU is slower than WebAssembly: one forward
  costs about 2.7 ms of dispatch and readback whatever the model's size (`ms per forward` in gate 1 is 2.6-3.2 ms
  for every tiny family), while WebAssembly runs the whole token in well under 1 ms.
- The IR path's prefill is a token at a time (one command buffer per position, IR_SPEC 3.9), so its prefill
  ms per token equals its decode ms per token; WebAssembly's batched prefill is close to WebGPU's on Gemma and
  Qwen2.5 (51 and 101 ms per token against 48 and 95).
- On Qwen2.5-0.5B a decode step reads its 1.27 GB of fp16 weights in about 100 ms, near 13 GB/s against
  the L4's 300 GB/s: the kernels are one invocation per output row (correct, not tuned).

## Recurrence on the GPU

ember d65d546 compiles `recurrence` nodes to WGSL (`../webgpu_recurrence_2026-10-27/`, gated there on
SwiftShader). Its documents, gate 1 as above on a second L4 session at ember 5c2e68a (which holds d65d546), the
same adapter, driver and Chrome (`colab_l4_recurrence/`): the staged models of the recurrence receipt
(`staged_SHA256SUMS`, checked before staging), keramos's tp_mamba1 recurrence rewrite on its model and
Mamba-130m's rewrite on the converted checkpoint, both supplied as documents (EMBER_IR for the reference, the
same text as the page's document).

| Document (model) | Greedy | Top-1 | Worst \|dlogit\| / range | Median | Dispatches per token | ms per forward |
|---|---:|---:|---:|---:|---:|---:|
| pk_recurrence | 40/40 | 320/320 | 2.1e-07 | 1.1e-07 | 52 | 2.6 |
| pk_kimi as recurrence | 40/40 | 320/320 | 1.8e-07 | 1.1e-07 | 52 | 2.6 |
| pk_srd as recurrence | 40/40 | 320/320 | 1.7e-07 | 1.1e-07 | 82 | 2.7 |
| pk_mamba as recurrence | 40/40 | 320/320 | 1.7e-07 | 1.0e-07 | 40 | 2.5 |
| kimi_tiny rewritten | 40/40 | 960/960 | 5.9e-07 | 3.5e-07 | 52 | 2.6 |
| srd_tiny rewritten | 40/40 | 960/960 | 3.9e-07 | 3.3e-07 | 82 | 2.7 |
| mamba_tiny rewritten | 40/40 | 960/960 | 6.2e-07 | 4.4e-07 | 40 | 2.6 |
| kimi_tiny, custom recurrence | 40/40 | 960/960 | 6.2e-07 | 3.7e-07 | 56 | 2.6 |
| tp_mamba1 (keramos) rewritten | 40/40 | 960/960 | 4.1e-06 | 7.3e-07 | 61 | 2.6 |
| Mamba-130m rewritten | 40/40 | 960/960 | 9.7e-04 | 8.4e-05 | 699 | 39.6 |

Every greedy continuation is identical and every argmax agrees. Mamba-130m's rewrite is past the flat 1e-4
bound as its hand-written kernels are (1.4e-3 on the same file in gate 1); greedy 40/40 and top-1 960/960 are
the official-checkpoint gate. With this leg every IR document ember's interpreter runs has a WGSL path that is
gated on a hardware adapter.

## Runs and spend

| Session | GPU | Minutes | What |
|---|---|---:|---|
| RunPod `e6-webgpu-l4` (vp8qf47duqyqt0) | L4, EUR-IS-1 | 15.5 | the first probe (`runpod_probe/`); terminated by hand, $0.15, before E6 moved to Colab |
| Colab `ember-e6-gpu` | L4 | 12 | probe and builds passed; the stage check failed on its own list (SHA256SUMS hashed itself), so the session stopped (`colab_l4/session1_failed/`) |
| Colab `ember-e6-gpu2` | L4 | 95 | gates, page tests, throughput |
| Colab `ember-e6-rec` | L4 | 18 | recurrence (queued about 5 min for a slot) |

About 2.1 L4-hours of Colab in all (the driver records session minutes; Colab's unit counter is not read
here).

## Fixes found by the run

- The page's heap offset for the logits: `worker.js` (and `chat_test.mjs`, `bench_page.html`) wrote them at
  `logitsPtr >> 2`, a signed shift, so a 4gb WebAssembly build whose allocation lands past 2^31 got a negative
  offset (RangeError) on Gemma-3-270m and Qwen2.5; `>>> 2` since 5c2e68a. The Space's 1gb build never reaches
  2^31; its runs above are unaffected. The official checkpoints' benches ran after the fix
  (`colab_l4/bench/first_attempt/` holds the failed attempts and Qwen3's first run, which fitted under 2^31).

## Files

- `tools/webgpu_parity/`: `remote_chrome.py` (headless Chrome on a GPU VM; refuses to run without
  EMBER_WEBGPU_REMOTE=1 and refuses software adapters), `gpu_probe.html`, `gate_page.html`, `bench_page.html`,
  `page_test.html`, `e6_common.js`.
- Here: `drive.sh` (the local Colab driver, path C of microtorch/AGENTS.md), `vm_colab.sh`, `vm_probe.sh`,
  `vm_setup.sh` (ember CPU and the three WebAssembly builds, emsdk 6.0.6; the official checkpoints from Hugging
  Face), `vm_run.sh` (references, gates, page tests, benches), `vm_fixture.py`, `tables.py` (the tables from
  the receipts).
- Receipts: `colab_l4_recurrence/` (the recurrence leg), `colab_l4/` (probe, setup with the builds' sha256, docs, gates, fixtures, pages, bench,
  `e6_vm.log`), `runpod_probe/`.
