# What each capability cell rests on

The README's tables say which model families run on which backend. This page says what kind of
evidence stands behind each cell, so that a gate on a tiny synthetic model, or on a software
WebGPU adapter, is never read as validation of a released checkpoint or of a hardware GPU. Every
entry names its receipt folder under [`experiments/`](../experiments/); the receipt is the
authority where this summary and it differ.

## The categories

| Code | Evidence | What it does not show |
|---|---|---|
| **impl** | the code exists and is refused or accepted at load by rule; no gate | correctness |
| **par** | numerical parity: greedy replies identical and the top-1 equal at every position against the named reference, with the logit bound the receipt states | speed; models of another class |
| **ck** | parity on an official released checkpoint (named in the row) | other checkpoints of the family; long contexts unless stated |
| **pk** | parity on paperkiln-trained models only: tiny test models (d 32 to d 128, a few hundred to a few thousand steps) or the programme's chat models (named) | third-party checkpoints |
| **syn** | parity on synthetic or tiny random-weight models only | any released checkpoint |
| **perf** | a timing receipt on the named hardware | other hardware, other families, model quality |
| **fb** | a CUDA build serves the family on its CPU path (no device kernels for it) | any GPU execution |

Hardware: **L4** (NVIDIA L4, Colab or RunPod), **T4** (NVIDIA T4, Colab), **4060** (the laptop's
RTX 4060), **iGPU** (the laptop's Intel Gen12 integrated GPU, headless Edge), **SwS** (SwiftShader,
a software WebGPU adapter that runs on two CPU cores: correctness only, never a GPU result),
**node** (WebAssembly in node on a CPU), **page** (the chat page's own WebAssembly build, compared
by its replies). "GPU unnamed" means the receipt does not record the device.

References: the CPU column of paperkiln models is gated against paperkiln (`mtstudio` greedy
replies on 40 frozen prompts, and `mtstudio score` or paperkiln's numpy IR for logits within
ROUNDING, 1e-5 of max(1, max |ref|)). The CUDA, WebAssembly and WebGPU columns are gated against
ember's own CPU interpreter (greedy N/N, top-1 at every position, max |dlogit| over the row's range
within 1e-4). For official checkpoints the deciding rule is greedy and top-1; logit closeness is
judged against a float64 evaluation of the same graph (IR_SPEC 4.3), because float32 conditioning
grows with depth: Pythia-160m and Mamba-130m pass greedy and top-1 everywhere but exceed the flat
1e-4 on GPU and WebAssembly paths (up to 3.9e-4 on CUDA, 1.4e-3 for Mamba-130m on the L4 WebGPU run).

## paperkiln architectures and the tuned llama path

| Family | CPU | CUDA | WebAssembly | WebGPU |
|---|---|---|---|---|
| Llama 2/3, TinyLlama (tuned path) | **ck** SmolLM2-135M (llama architecture, tuned path, log-probabilities against HF; `score_2026-10-26`). TinyLlama 1.1B, Llama-2 7B and Llama-3 8B are listed as tested sources in `docs/USAGE.md` from the pre-rename project; no receipt for them is in `experiments/` | **pk** paperkiln llama chat models (below); **perf** L4 on a synthetic timing model at the TinyLlama-1.1B shape (`docs/TTFT.md`); no official llama checkpoint on the tuned CUDA path has a receipt here | **pk** only | **pk** only: chat_sftfinal_v3 f32 and Q8_0 on iGPU (`webgpu_2026-10-21`), L4 and SwS (`webgpu_gpu_2026-10-27`, `webgpu_ir_2026-10-26`) |
| paperkiln llama chat models | **pk** chat_sftfinal, sft5250, sft1500, pk_base_v1 and its chat models against mtstudio, 40/40 (sft1500 39/40 on CUDA at a documented near-tie) (`decode_2026-10-19`, `pk_base_v1_2026-10-24`) | **pk** 4060 (`decode_2026-10-19`), T4 (`kquant_2026-10-25`, `score_2026-10-26`); **perf** 4060 (`latency_2026-10-19`) | **pk** page (`webgpu_ir_2026-10-26`, `webgpu_gpu_2026-10-27`) | **pk** iGPU, L4, SwS; **perf** iGPU and L4 |
| Mixture of experts (llama.cpp layout) | **pk** fw_moe, 70M trained, 40/40 (`moe_fw_2026-10-21`) | **pk** fw_moe 40/40, GPU unnamed (`moe_fw_2026-10-21`) | **pk** node, 4 GB heap, 40/40 | **pk** fw_moe on iGPU (`webgpu_2026-10-21`); a tiny MoE through the IR on L4 and SwS |
| GPT-2 (layernorm, GELU, learned positions) | **pk** gpt2_tiny | **pk** GPU unnamed (`device_ops_2026-10-20`) | **pk** page | **pk** L4, SwS |
| Sliding window with sinks | **pk** gpt2_tiny weights with windowed IRs, swa_tiny (`swa_2026-10-20`) | **pk** GPU unnamed (`cuda_swa_2026-10-20`) | **pk** page (L4 VM, `webgpu_gpu_2026-10-27`) | **pk** L4, SwS |
| Kimi linear attention | **pk** kimi_tiny (`families_2026-10-20`) | **pk** GPU unnamed (`device_ops_2026-10-20`) | **pk** node | **pk** L4, SwS |
| Mamba-style scan | **pk** mamba_tiny (its 40 references are one repeated text, so first-token logits against numpy carry the gate) | **pk** GPU unnamed | **pk** node | **pk** L4, SwS |
| SRD, AttnRes, flex | **pk** srd_tiny, attnres_tiny, flex_highway_tiny, flex_plain_tiny | **pk** GPU unnamed | **pk** node | **pk** L4, SwS |
| `expr` | **pk** gpt2_tiny with its GELU as a formula (`expr_2026-10-22`) | **pk** 4060 (`thirdparty_2026-10-22/device_gate`), T4 (`mor_cuda_2026-10-25`) | **impl**: built from the same source, not gated | **pk** L4, SwS |
| `weight_index` (tied-block recursion) | **pk** recursion gate models (`recursion_2026-10-23`) | **pk** T4 (`mor_cuda_2026-10-25`) | **pk** page only: replies of the 40M MoR3 chat model, whose 8 applications share 4 stored layers, equal ember CPU's 40/40 (`webgpu_gpu_2026-10-27` gate 2); no logit-level WebAssembly gate | **pk** L4, SwS |
| `depth_route` (Mixture-of-Recursions) | **pk** mor3, mor2p0 and the MoR3 chat model | **pk** T4; **perf** T4 (MoR3 decode 1.41 ms/token against 24.2 ms on that VM's CPU) | **pk** page only, as above; **perf** the page's WebAssembly on the L4 VM's CPU | **pk** L4, SwS; **perf** L4 (Chrome) |
| `turn_state` | **pk** state4 (`state_tokens_2026-10-24`) | **pk** T4 (`cuda_state_2026-10-25`, `carry12_cuda_gates_2026-10-27`) | **pk** node (`wasm_state_2026-10-25`) | **pk** L4, SwS (every row kept: no eviction on WebGPU) |
| `llama_attnres` | **pk** ar3, ar_x5 at the 20m_L16 shape (`attnres_2026-10-24`, `attnres_x5_2026-10-25`) | **pk** T4 | **pk** node | **pk** L4, SwS |
| `llama_attnres` with `tensor_index` | **pk** ar_rec (d 32, 40 CPU steps) (`tensor_index_gate_2026-10-27`) | **pk** T4 | **pk** node | **pk** SwS only: no hardware-adapter run |
| `recurrence` | **pk** pk_recurrence and rewrites of kimi, srd and mamba models; Mamba-1's selective scan rewritten on Mamba-130m (`ir_rev11_2026-10-25`) | **fb** | **pk** node, pk_recurrence 40/40 (`ir_rev11_2026-10-25`) | **pk** L4, SwS; **ck** the Mamba-130m rewrite on L4, 40/40 (`webgpu_gpu_2026-10-27`, `webgpu_recurrence_2026-10-27`) |

## Third-party GGUFs (graph built from llama.cpp metadata)

| Family and file | CPU | CUDA | WebAssembly | WebGPU |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct f16 | **ck** greedy 40/40 against HF transformers with ember's tokeniser and template; top-1 against numpy | **ck** 4060, 40/40, 960/960; prompts of 558 to 1,289 tokens on L4 (`thirdparty_2026-10-22/device_gate`); **perf** none for CUDA | **ck** node, 10 prompts (240/240), 4 GB heap (`wasm_families_2026-10-25`) | **ck** L4 (Chrome), 40/40, 960/960; **perf** L4; SwS not run (memory) |
| Qwen3-0.6B Q8_0 | **ck** as Qwen2.5 | **ck** L4, 40/40 | **ck** node, 10 prompts | **ck** L4, 40/40; **perf** L4; SwS 1 prompt only |
| Gemma-3-270m-it f32 | **ck** top-1 against numpy at every position | **ck** 4060, 40/40; prompts of 794 to 1,324 tokens on L4; **perf** L4 against HF transformers (`hf_cuda_2026-10-30`) | **ck** node, 40/40 | **ck** L4, 40/40; **perf** L4; SwS not run (memory) |
| GPT-NeoX / Pythia-160m f32 | **ck** | **ck** 4060 (959/960 top-1: one near-tie flip), L4 (960/960) | **ck** node, 40/40, 4 GB heap | **ck** L4 and SwS, 40/40 |
| Mamba-1 130m f32 | **ck** | **ck** 4060, 40/40 | **ck** node, 40/40, 4 GB heap | **ck** SwS; L4 on a separately converted file whose sha256 differs from the laptop's, so the two WebGPU runs are not on identical weights |
| Mistral (sliding window) | **syn** tiny random model | **syn** | **syn** | **syn** L4, SwS |
| GPT-2 124M | **ck** OpenAI's checkpoint, greedy 40/40 against HF on raw prompts (`gpt2_2026-10-25`) | **syn** only | **syn** only | **syn** only (L4, SwS) |
| Gemma 2; Phi-3 LongRoPE and sliding window | **syn** (`e4_gemma2_phi3_2026-10-28`) | **syn** L4 (RunPod, `e4_moe_mla_2026-10-28/receipts/cuda_20261030.json`) | **syn** node | not supported: the page sends these to WebAssembly |
| Mixtral; Qwen2-MoE; Qwen3-MoE | **syn** | **syn** L4 (Mixtral through `EMBER_IR_BUILD=1`, its IR path) | **syn** node | not supported |
| DeepSeek2 and query-LoRA (MLA) | **syn** | **syn** L4 | **syn** node | not supported |
| DeepSeek2 dense/YaRN; DeepSeek3 | **syn** | **fb**: not credited as a CUDA result | **syn** node | not supported |

The synthetic E4 files establish the graphs and kernels, not support for full checkpoints or their
speed.

## Memory limits and fallbacks kept from the receipts

- The official checkpoints need the 4 GB WebAssembly heap (`EMBER_WASM_MAX_MEMORY=4gb`); Pythia and
  Mamba fail to load in the 1 GB default. The page caps the KV state at 2048 positions; uncapped,
  Qwen3-0.6B's declared context needs 9.4 GB of f32 KV (`wasm_ctx_2026-10-25`).
- A CUDA build serves `recurrence`, `position_bias` and the two DeepSeek variants above on its CPU
  path, and refuses `relative_kv` under `depth_route` on the device by name.
- On WebGPU the browser's memory limits depend on the device and build; the SwiftShader runs could
  not hold Gemma-3-270m or Qwen2.5-0.5B on the laptop.
- On the L4, WebGPU is slower than WebAssembly on models under about 3M parameters (dispatch-bound,
  about 2.7 ms per forward whatever the size) and faster on the 0.27B to 0.6B checkpoints
  (`webgpu_gpu_2026-10-27`).
