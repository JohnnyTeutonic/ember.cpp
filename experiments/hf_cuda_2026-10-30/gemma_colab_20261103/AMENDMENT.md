# Gemma3-270M on the frozen CUDA comparison: a Colab amendment

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. This runs the existing frozen comparison
(`../README.md`, controller `../benchmark.py`) for its second candidate input, Gemma3-270M, which the README records
as unmeasured. The Smol result and every Smol receipt stay as they are.

## What is unchanged

- **Controller.** `benchmark.py`, SHA-256 `ddd3104489dd2935…` (the same at ember `5bfda893` and at main
  `f80ede9`). It runs with the frozen arguments `--lengths 16 128 512 --steps 32 --cycles 2`.
- **Gates and schedule.** Every full-vocabulary row and greedy id passes
  `max_abs_error / max(1, max_abs_HF_row) <= 1e-4` before any timing. Each passed shape gets one warm request
  per engine, then two ABBA cycles, with hash-chained immutable trials. A failed row or a HALTED marker is kept;
  no tolerance change and no replacement prompts.
- **Environment pins** (`launch_contract_gemma.py` keeps the frozen contract's values):
  - versions: torch 2.4.1+cu124, numpy 1.26.4, transformers 4.57.0, tokenizers 0.22.1, gguf 0.19.0, safetensors
    0.6.2, huggingface-hub 0.35.3, sentencepiece 0.2.2, accelerate 1.13.0;
  - one thread, TF32 off, one visible device;
  - `EMBER_IR_BUILD=1`, `EMBER_IR_CTX=1024`, and no other `EMBER_*` override;
  - a native backend that must report CUDA, with an all-F32 tensor inventory.
- **Engine source.** Ember `5bfda893`, the compiled source of the Smol result. `ir_hf_cuda_bench` is built on the
  VM with the same CMake and make targets, and its SHA-256 is recorded. This is a new build; the archived Smol
  binary is not reused.
- **Canonical input.** The F32 GGUF must equal `0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa`
  byte for byte before any forward, or no forward is permitted.

## What differs, and why

- **Provider.** One Colab L4 (Claude's lane), not the RunPod image. The frozen contract's image-digest check
  cannot hold. The Colab image, driver, CUDA toolkit and Python are recorded instead, and the pinned packages
  are installed and verified exactly as the frozen contract lists them.
- **Vocabulary.** The backend check expects Gemma's 262,144 entries instead of Smol's 49,152. That literal is the
  only content change from `../launch_contract_v3.py`.
- **GGUF regeneration.** Verified locally on 3 Nov: the recipe below reproduces the canonical file byte for byte.
  - Converter: llama.cpp `convert_hf_to_gguf.py` from the pinned archive `converter.tgz` (SHA-256 `fa607fc3…`,
    the archive the Smol run used), with its bundled gguf-py.
  - Weights: `google/gemma-3-270m-it` revision `ac82b4e8…`. The repository is gated, so the VM takes
    `model.safetensors` (`700b710a…`) and `tokenizer.model` (`1299c11d…`) from the ungated `unsloth/gemma-3-270m-it`
    mirror, whose copies are byte-identical (Hub SHA-256 metadata), and checks both hashes. The six small files are
    Google's own and are uploaded from the local snapshot, with each git blob id matching Google's repository:
    `config.json`, `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`, `added_tokens.json`,
    `generation_config.json`. `chat_template.jinja` is left out: it reached the snapshot after the canonical
    conversion, and adding it changes the file.
  - The input folder must be named `gemma-3-270m-it`, because the converter derives `general.name`, `basename`,
    `finetune` and `size_label` from it.
  - `--outtype f32`.

## Outcomes

- **Pass:** every shape's rows and ids pass the gate, and timings are reported exactly as the frozen README
  reports Smol's.
- **Valid negative:** a gate failure is a numerical result, preserved, and the affected shapes are not timed.
- **Operational failure:** a session lost, a build failure, or a regeneration hash mismatch. It is recorded, and
  may be rerun once, unchanged.
- No model-quality or general engine-ranking claim follows, as in the frozen README.

## Budget and procedure

- One Colab L4, Claude's lane. The work deadline is 75 minutes; expected about 35 (build about 10, packages about
  5, regeneration about 2, three shapes of gating and timing).
- The run goes through `drive.py`, generalised from the reviewed selective-scan driver, with uploads added. Its
  deadline, cleanup, receipt and exit-code rules are unchanged. `vm_gemma.sh` writes `status.json` before `DONE`.
- Receipts go to `receipts/RUN_ID/`, and the raw archive to
  `/mnt/c/ml_artifacts/benchmarks/hf_cuda_gemma_colab_20261103/`.
