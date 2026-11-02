# Gemma3-270M on the frozen CUDA comparison: a Colab amendment

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. This runs the existing frozen comparison
(`../README.md`, controller `../benchmark.py`) for its second candidate input, Gemma3-270M, which the README records
as unmeasured. The Smol result and every Smol receipt stay as they are. Review: Codex methods review, kept in
the research workspace (`AI_ML/gemma_cuda_claude/reviews/`), not in this repository.

This is version 2. Version 1 (rounds 1-3 DO NOT RUN, round 4 RUN) ran once, as `g1-20261103` on 3 Nov. It failed
at its packages gate before the build, and the controller never started (`receipts/g1-20261103/`). Colab's image
had changed: Python 3.13.16 and only the CUDA 13.0 toolkit. Torch 2.4.1 and numpy 1.26.4 publish no Python 3.13
wheels, so an unchanged rerun would have failed the same way. Version 2 changes the interpreter and toolkit
installation (below) and nothing in the scientific protocol.

## What is unchanged

- **Controller.** `benchmark.py`, SHA-256 `ddd3104489dd2935…` (the same at ember `5bfda893` and at main
  `f80ede9`). It runs with the frozen arguments `--lengths 16 128 512 --steps 32 --cycles 2`, and its offline
  `--analyze` runs afterwards.
- **Gates and schedule.** Every full-vocabulary row and greedy id passes
  `max_abs_error / max(1, max_abs_HF_row) <= 1e-4` before any timing. Each passed shape gets one warm request
  per engine, then two ABBA cycles, with hash-chained immutable trials. A failed row or a HALTED marker is kept;
  no tolerance change and no replacement prompts.
- **Pinned top-level packages:** torch 2.4.1+cu124, numpy 1.26.4, transformers 4.57.0, tokenizers 0.22.1, gguf
  0.19.0, safetensors 0.6.2, huggingface-hub 0.35.3, sentencepiece 0.2.2, accelerate 1.13.0.
- **Runtime environment for the controller.** The nine variables and six unset overrides of the frozen wrapper are
  set after the packages are installed and before download and conversion, as the frozen wrapper did: one thread,
  TF32 off, one visible device, `EMBER_IR_BUILD=1`, `EMBER_IR_CTX=1024`.
- **Native backend.** It must report CUDA, one thread, a 1,024 context and an all-F32 tensor inventory.
- **Engine source.** Ember `5bfda893`, the compiled source of the Smol result.
- **Canonical input.** The F32 GGUF must equal `0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa`
  byte for byte before any forward, or no forward is permitted.

## What differs, and why

- **Provider.** One Colab L4 (Claude's lane), not the RunPod image. The frozen contract's image-digest check
  cannot hold. Instead the Colab image is recorded, and checked where this run depends on it: provider colab,
  exactly one NVIDIA L4, no provenance command failed. The record covers GPU, driver, every installed CUDA
  toolkit, Python, OS, CPU and the full `nvidia-smi` output.
- **Interpreter (v2).** The frozen pod ran CPython 3.11, and its README records that Python 3.12 changes the
  analyzer's float summation. The run therefore uses CPython 3.11.15, the build uv installs, not the image's own
  Python. The image's Python records provenance and installs uv 0.11.29 (wheel pinned by SHA-256 `eec03a8b…`).
  uv installs CPython 3.11.15 into a seeded virtual environment, which comes first on `PATH` for every later step:
  packages, contract, conversion, controller and analysis. The launch contract refuses any other interpreter.
- **Package installation.** Into that environment, pip installs torch 2.4.1 from the cu124 index and the other
  eight from PyPI, as in version 1. Every compiled pin has a cp311 x86-64 wheel. A fresh environment holds no image
  packages, so nothing is uninstalled. The import path the controller uses (torch with CUDA, transformers,
  `AutoModelForCausalLM`, `Gemma3ForCausalLM`) and the environment-only contract are checked immediately after
  installation, before the build or any download.
  - Top-level versions match the frozen recipe. Transitive packages and native libraries need not, and are
    recorded (`packages.json`, `interpreter.txt`).
  - Results are measurements in the recorded Colab environment. They are not an environment-controlled comparison
    with the Smol run.
- **CUDA toolkit (v2).** The image had only CUDA 13.0 on 3 Nov. Its own gate (`toolkit`, before the build) installs
  CUDA 12.6 from NVIDIA's ubuntu2404 repository. The repository keyring package (`cuda-keyring_1.1-1_all.deb`,
  SHA-256 `d2a6b11c…`) is checked before it is trusted. Then `cuda-minimal-build-12-6=12.6.3-1` (nvcc, cudart,
  CCCL) and `libcublas-dev-12-6=12.6.4.1-1` are installed, and the installed CUDA packages are recorded
  (`cuda_packages.txt`). The frozen pod's nvcc was 12.4. 12.4 has no Ubuntu 24.04 packages, and 12.6 is the
  toolkit under which ember `5bfda893` was verified to compile.
- **Engine build.** A new build, not the archived Smol executable.
  - Toolkit: `/usr/local/cuda-12.6`, whose `nvcc --version` must report release 12.6 and is recorded. Release,
    `CMAKE_CUDA_ARCHITECTURES=89`, target `ir_hf_cuda_bench`, as the frozen v1/v2 VM scripts built it.
  - The executable is archived in the evidence (`out/binaries/`) and its hash checked against the one that runs.
- **Launch contract** (`launch_contract_gemma.py`, against `../launch_contract_v3.py`):
  - provenance is recorded and checked as above instead of the image digest, so `--provenance` is required;
  - any `EMBER_*` variable other than the two required is refused (v3 refused only the six listed overrides);
  - the backend vocabulary is Gemma's 262,144 instead of Smol's 49,152;
  - without `--binary` it checks the environment alone (run right after the packages);
  - (v2) the interpreter running it must be CPython 3.11.15.
- **GGUF regeneration.** Verified locally on 3 Nov under Python 3.11 and 3.12: this recipe reproduces the canonical
  file byte for byte.
  - Converter: llama.cpp `convert_hf_to_gguf.py` from the pinned archive `converter.tgz` (SHA-256 `fa607fc3…`, the
    archive the Smol run used), with its bundled gguf-py.
  - Weights: `google/gemma-3-270m-it` revision `ac82b4e8…`. The repository is gated, so the VM takes
    `model.safetensors` (`700b710a…`) and `tokenizer.model` (`1299c11d…`) from the ungated `unsloth/gemma-3-270m-it`
    mirror at revision `23cf460f…`, whose copies are byte-identical (Hub SHA-256 metadata). Both hashes are checked
    (`weights.sha256`).
  - The six small files are Google's own, uploaded from the local snapshot and checked (`inputs.sha256`); each git
    blob id matches Google's repository. They are `config.json`, `tokenizer_config.json`, `tokenizer.json`,
    `special_tokens_map.json`, `added_tokens.json` and `generation_config.json`. `chat_template.jinja` is left out:
    it reached the snapshot after the canonical conversion, and adding it changes the file.
  - The input folder must be named `gemma-3-270m-it`, because the converter derives `general.name`, `basename`,
    `finetune` and `size_label` from it.
  - `--outtype f32`.
  - The driver checks all seven upload sources against `inputs.sha256` before it creates a session.

## Outcomes

- **Pass:** every shape's rows and ids pass the gate, and timings are reported exactly as the frozen README
  reports Smol's.
- **Valid negative:** reported only when the raw rows and ids show the gate failing, with the controller's HALTED
  text naming a parity failure. It is a numerical result: preserved, final, and the affected shapes are not timed.
  Exit codes alone, or a HALTED that names loading, a timeout or integrity, are not a valid negative.
  `failure_summary.json` keeps HALTED's text and the logs' tails. A failing `--analyze` after a genuine parity
  failure is expected, not evidence against it.
- **Operational failure.** Session lost, build or package failure, regeneration mismatch, or evidence not
  retrieved.
  - A fresh rerun, once and unchanged, is allowed only if the run did not reach the controller.
    `SCIENCE_STARTED` is written immediately before it starts, and `status.json` reports `science_started`.
    `g1-20261103` (version 1) did not reach it. Its failure was a property of the image, not a transient one, so
    the next run is version 2's first run, reviewed again before launch. Version 2 has its own single rerun.
  - A run that reached the controller is never repeated: its outputs are retrieved and reported as complete,
    incomplete or negative. A transport failure after that point makes the result incomplete; it does not make it
    eligible for another attempt.
- No model-quality, general engine-ranking or environment-controlled cross-model claim follows, as in the frozen
  README.

## Budget and procedure

- One Colab L4, Claude's lane; expected about 40 minutes (interpreter and packages about 6, toolkit about 3, build
  about 10, regeneration about 2, three shapes of gating and timing). `g1-20261103` used about 10 L4-minutes.
- The run goes through `drive.py`, generalised from the reviewed selective-scan driver.
  - Deadlines from the driver's start: polling ends at 60 minutes, retrieval at 75, shutdown attempts at 85, the
    final inventory at 90.
  - If polling ends without `DONE` (or a launch was attempted but not acknowledged), the driver sends SIGTERM to
    the VM script by its recorded PID, after checking that the PID is still that script.
  - The script starts the controller in its own process group (`setsid`). The native engine inherits that group,
    because the frozen controller starts it with a plain `Popen`. Containment is by that group: a process that left
    it on its own would not be seen.
  - A stop that arrives while the controller is being forked and registered is held, then delivered. Until no
    non-zombie process with the controller's PID or group is left, the stop handler signals both every second (so
    the window before the child's `setsid()` is covered). It escalates to SIGKILL after 60 seconds and records in
    `stop.json` whether anything is left. No wait is unbounded.
  - Only then does the EXIT trap write `status.json` and `DONE`; publication itself ignores further stop signals.
    An interrupted controller is benchmark gate 143. A member of the controller's group still running after the
    controller exits on its own is killed and recorded (`leftover_processes.txt`, gate 97), so nothing is still
    writing when the outputs are published.
  - (v2) Absence is never assumed: a failed or empty process listing counts as a live process. If anything
    survives SIGKILL, on either path, the script writes `QUIESCENCE_UNCONFIRMED`. The driver then treats the
    archive as possibly not final (exit at least 2), whatever the stop reply said.
  - The VM side answers the driver with one structured line: interrupted (DONE and a gone group), completed (DONE,
    no stop needed), unconfirmed, or not observed (no PID file after a minute). Anything but interrupted or
    completed means the archive may hold outputs still being written, so the run can exit no better than 2
    (transport or receipt failure).
  - Local tests (33) cover:
    - a stop during the controller, a stop between fork and registration, a stop before the controller's group
      exists, and a writer left behind by the controller, each checking that the writer's last write came before
      `status.json` was published;
    - the toolkit gate and a wrong nvcc release;
    - an unconfirmed quiescence.
  - Packing tolerates files changing while they are read, and pack and download are tried twice within the
    retrieval budget. A failed retrieval is logged as evidence lost before cleanup. In `g1-20261103` the first
    pack call hung until its 240-second limit and the second succeeded at once.
  - Accepted residual: archive extraction (local, seconds for the expected outputs) runs before VM cleanup and has
    no deadline of its own.
  - Cleanup, receipt and exit-code rules are the reviewed driver's. A validated success also needs the `DONE`
    marker.
  - Its local tests: `test_drive.py` (mock CLI) and `test_vm_gemma.py` (stub tools).
- The whole retrieved output tree, with the executable (`out/binaries/`), the controller, contract and inputs
  (`out/inputs/`), goes to `/mnt/c/ml_artifacts/benchmarks/hf_cuda_gemma_colab_20261103/RUN_ID/`. The top-level
  files of `out/` (status, gates, provenance, packages, logs, raw rows, trials, analysis, `stop.json` if any) are
  copied to `receipts/RUN_ID/` in the repository, with `archive.sha256` hashing every retrieved file, so the
  archived executable and inputs are pinned from the receipts. Each archive attempt is extracted into its own
  staging folder and promoted only when complete; receipts are written after the VM is stopped.
