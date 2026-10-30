"""Paired FP32 CUDA runtime forward timings, gated on complete logit/ID parity.

One GPU, one host thread; same GGUF weights, raw tokens, last-row-only heads, cache and fixed-length
teacher-forced continuations. No tokenizer, sampler, HTTP, startup or model
quality claim. Input-ID H2D and full final-row D2H are inside both forward timers.
The initial CUDA HF greedy continuation is frozen, including EOS.
Completed trials are journaled and never re-timed on resume.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import select
import statistics
import subprocess
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def validate_inputs(data, length, steps, vocab):
    if not isinstance(vocab, int) or vocab < 1:
        raise ValueError("invalid frozen vocabulary")
    expected = [(11 + 37 * i) % vocab for i in range(length)]
    if data.get("prompt_ids") != expected:
        raise ValueError("frozen prompt length/formula differs")
    continuation = data.get("continuation_ids", [])
    if len(continuation) != steps or any(
        type(token) is not int or not 0 <= token < vocab for token in continuation
    ):
        raise ValueError("invalid frozen continuation inventory/range")


def verify_files(expected):
    actual = {path: sha(path) for path in expected}
    if actual != expected:
        raise ValueError("input/binary/source integrity drift")
    return actual


def validate_native_metadata(ready, hf_vocab):
    inventory = ready.get("tensor_inventory", [])
    if not inventory or any(t.get("type") != 0 for t in inventory):
        raise ValueError("native GGUF inventory is not entirely F32")
    names = [t.get("name") for t in inventory]
    if len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("invalid native tensor inventory")
    if ready.get("vocab") != hf_vocab:
        raise ValueError("vocabulary differs")
    return {
        "vocab": hf_vocab,
        "max_len": ready["max_len"],
        "tensor_count": len(inventory),
        "tensor_type": 0,
        "tensor_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def assert_parity(hf, ember, hf_ids, ember_ids, tolerance=1e-4):
    if hf_ids != ember_ids or not hf_ids:
        raise ValueError("complete greedy ID inventory differs")
    if hf.shape != ember.shape or hf.ndim != 2 or hf.shape[0] != len(hf_ids):
        raise ValueError("logit row inventory differs")
    if not np.isfinite(hf).all() or not np.isfinite(ember).all():
        raise ValueError("nonfinite logits")
    error = np.max(np.abs(hf.astype(np.float64) - ember.astype(np.float64)), axis=1)
    relative = error / np.maximum(1.0, np.max(np.abs(hf), axis=1))
    if np.max(relative) > tolerance:
        raise ValueError(f"ROUNDING bound failed: {np.max(relative)}")
    return {
        "rows": len(hf_ids),
        "vocab": hf.shape[1],
        "max_abs": float(max(error)),
        "max_relative": float(max(relative)),
        "tolerance": tolerance,
    }


def validate_trial(row, expected_ids, steps):
    if row.get("argmax") != expected_ids or len(expected_ids) != steps:
        raise ValueError("trial changed the fixed greedy continuation")
    times = row.get("forward_ms", [])
    if len(times) != steps or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("invalid/incomplete forward timings")


def require_thread_budget(env):
    if any(
        env.get(key) != "1"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    ):
        raise ValueError("run through the shared one-core CUDA budget wrapper")


def digest_record(row):
    value = {k: v for k, v in row.items() if k != "record_sha256"}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trial_plan(protocol):
    return [
        (f"{length}_{cycle}_{slot}_{engine}", length, engine)
        for length in protocol["lengths"]
        for cycle in range(protocol["cycles"])
        for slot, engine in enumerate(protocol["sequence"])
    ]


def journal_rows(path, protocol, protocol_sha):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(
            "torn journal tail: preserve and repair explicitly before resume"
        )
    rows = [json.loads(line) for line in raw.splitlines()]
    plan = trial_plan(protocol)
    if len(rows) > len(plan):
        raise ValueError("extra timing records")
    previous = protocol_sha
    for row, (key, length, engine) in zip(rows, plan):
        if (row.get("key"), row.get("length"), row.get("engine")) != (
            key,
            length,
            engine,
        ):
            raise ValueError("journal order/identity differs")
        if row.get("previous_sha256") != previous or row.get(
            "record_sha256"
        ) != digest_record(row):
            raise ValueError("journal hash chain differs")
        previous = row["record_sha256"]
    return rows


def check_gate(out, length, steps, vocab):
    inputs = out / f"inputs_{length}.json"
    hf_dump, ember_dump = out / f"hf_{length}.f32", out / f"ember_{length}.f32"
    gate = json.loads((out / f"gate_{length}.json").read_text())
    if (
        gate["inputs_sha256"] != sha(inputs)
        or gate["hf_rows_sha256"] != sha(hf_dump)
        or gate["ember_rows_sha256"] != sha(ember_dump)
    ):
        raise ValueError("completed numerical-gate artifacts changed")
    data = json.loads(inputs.read_text())
    validate_inputs(data, length, steps, vocab)
    hf = np.fromfile(hf_dump, dtype="<f4").reshape(steps, -1)
    ember = np.fromfile(ember_dump, dtype="<f4").reshape(steps, -1)
    if hf.shape[1] != vocab or ember.shape[1] != vocab:
        raise ValueError("frozen vocabulary/row width differs")
    recomputed = assert_parity(
        hf, ember, data["continuation_ids"], np.argmax(ember, axis=1).tolist()
    )
    if np.argmax(hf, axis=1).tolist() != data["continuation_ids"] or any(
        gate[k] != v for k, v in recomputed.items()
    ):
        raise ValueError("numerical-gate report disagrees with raw rows")
    return data


def analyze(out):
    path = out / "protocol.json"
    protocol = json.loads(path.read_text())
    if sha(__file__) != protocol["controller_sha256"]:
        raise ValueError("controller source drift")
    journal = out / "trials.jsonl"
    rows = journal_rows(journal, protocol, sha(path))
    for length in protocol["lengths"]:
        data = check_gate(out, length, protocol["steps"], protocol["model_metadata"]["vocab"])
        for row in rows:
            if row["length"] == length:
                validate_trial(row, data["continuation_ids"], protocol["steps"])
                if row.get("inputs_sha256") != sha(out / f"inputs_{length}.json"):
                    raise ValueError("trial inputs changed")
    integrity = json.loads((out / "integrity.json").read_text())
    expected = protocol["integrity_files"]
    if integrity != {
        "protocol_sha256": sha(path), "journal_sha256": sha(journal),
        "files": expected,
        "inputs": {str(length): sha(out / f"inputs_{length}.json") for length in protocol["lengths"]},
        "status": "FINAL_INTEGRITY_VERIFIED",
    }:
        raise ValueError("final integrity receipt differs")
    result = {
        "status": "COMPLETE_PARITY_GATED_CUDA_MEASUREMENT",
        "protocol_sha256": sha(path),
        "journal_sha256": sha(journal),
        "scope": __doc__,
        "results": summarize(rows, protocol["lengths"], protocol["cycles"]),
    }
    summary = out / "summary.json"
    if summary.exists() and json.loads(summary.read_text()) != result:
        raise ValueError("saved summary disagrees with replay")
    return result


def summarize(rows, lengths, cycles):
    result = {}
    for length in lengths:
        point = {}
        for engine in ("ember", "hf"):
            selected = [
                r for r in rows if r["length"] == length and r["engine"] == engine
            ]
            if len(selected) != 2 * cycles:
                raise ValueError("timing inventory incomplete")
            prefill = [r["forward_ms"][0] for r in selected]
            decode = [
                sum(r["forward_ms"][1:]) / (len(r["forward_ms"]) - 1) for r in selected
            ]
            point[engine] = {
                "trials": len(selected),
                "prefill_ms_median": statistics.median(prefill),
                "decode_ms_per_token_median": statistics.median(decode),
                "prefill_ms_all": prefill,
                "decode_ms_per_token_all": decode,
            }
        point["hf_over_ember_prefill"] = (
            point["hf"]["prefill_ms_median"] / point["ember"]["prefill_ms_median"]
        )
        point["hf_over_ember_decode"] = (
            point["hf"]["decode_ms_per_token_median"]
            / point["ember"]["decode_ms_per_token_median"]
        )
        result[str(length)] = point
    return result


class Ember:
    def __init__(self, binary, gguf, out):
        self.log = open(out / "ember.stderr.log", "ab")
        self.proc = subprocess.Popen(
            [str(binary), str(gguf)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            bufsize=0,
        )
        try:
            self.ready = self.read()
            if (
                not self.ready.get("ready")
                or self.ready.get("backend") != "cuda"
                or self.ready.get("threads") != 1
            ):
                raise ValueError("not the required CUDA backend")
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not select.select(
                [self.proc.stdout], [], [], min(1.0, deadline - time.monotonic())
            )[0]:
                continue
            raw = self.proc.stdout.readline()
            if not raw:
                raise ValueError("Ember exited without a complete response")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                self.log.write(raw)
                self.log.flush()
        raise TimeoutError("Ember CUDA request exceeded180seconds")

    def run(self, key, prompt, continuation, dump=None):
        request = {"key": key, "prompt_ids": prompt, "continuation_ids": continuation}
        if dump:
            request["rows"] = str(dump)
        self.proc.stdin.write((json.dumps(request) + "\n").encode())
        self.proc.stdin.flush()
        result = self.read()
        if result.get("key") != key or result.get("backend") != "cuda":
            raise ValueError("Ember response identity/backend differs")
        return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.log.close()


class HF:
    def __init__(self, gguf):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("exactly one visible real CUDA device required")
        self.device = torch.device("cuda:0")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                str(gguf.parent),
                gguf_file=gguf.name,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                local_files_only=True,
            )
            .eval()
            .to(self.device)
        )
        if any(
            p.device.type != "cuda" or p.dtype != torch.float32
            for p in self.model.parameters()
        ):
            raise ValueError("HF must use CUDA FP32 weights")
        if "logits_to_keep" not in inspect.signature(self.model.forward).parameters:
            raise ValueError(
                "HF model cannot select the same last-row-only head workload"
            )

    def run(self, prompt, steps, continuation=None, dump=None):
        torch = self.torch
        past, observed, times, rows = None, [], [], []
        with torch.inference_mode():
            for step in range(steps):
                ids = (
                    prompt
                    if not step
                    else [
                        continuation[step - 1]
                        if continuation is not None
                        else observed[-1]
                    ]
                )
                tensor = torch.tensor([ids], dtype=torch.long)
                torch.cuda.synchronize(self.device)
                start = time.perf_counter_ns()
                tensor = tensor.to(self.device)
                output = self.model(
                    tensor, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                row = output.logits[0, -1].float().cpu()
                torch.cuda.synchronize(self.device)
                end = time.perf_counter_ns()
                past = output.past_key_values
                if past is None or output.logits.shape[:2] != (1, 1):
                    raise ValueError("HF cache/last-row workload unavailable")
                observed.append(int(row.argmax()))
                times.append((end - start) / 1e6)
                if dump:
                    rows.append(row.numpy().copy())
        if dump:
            np.asarray(rows, dtype="<f4").tofile(dump)
        return {"argmax": observed, "forward_ms": times}


def run(a):
    require_thread_budget(os.environ)
    import torch
    import transformers

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise ValueError("set NVIDIA_TF32_OVERRIDE=0 for both processes")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible real CUDA device required")

    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "HALTED").exists():
        raise ValueError("preserved HALTED receipt; no automatic retries")
    source = Path(__file__).resolve().parents[2]
    integrity_files = {
        str(path.resolve()): sha(path) for path in (
            a.gguf, a.binary, Path(__file__), source / "tools/ir_hf_cuda_bench.cpp",
            source / "graph_ir.cpp", source / "graph_ir.h", source / "CMakeLists.txt",
        )
    }
    protocol = {
        "integrity_files": integrity_files,
        "schema": 1,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
        },
        "cuda_runtime": torch.version.cuda,
        "tf32": False,
        "transfer_scope": "CPU input IDs through full CPU output logits; H2D and D2H included in both timers, no tokenization/argmax/dumps/loading/startup",
        "gguf": str(a.gguf),
        "gguf_sha256": sha(a.gguf),
        "binary": str(a.binary),
        "binary_sha256": sha(a.binary),
        "controller_sha256": sha(__file__),
        "harness_source_sha256": sha(
            Path(__file__).resolve().parents[2] / "tools/ir_hf_cuda_bench.cpp"
        ),
        "lengths": a.lengths,
        "steps": a.steps,
        "cycles": a.cycles,
        "sequence": ["ember", "hf", "hf", "ember"],
        "prompt_formula": "(11+37*i) modulo vocabulary; synthetic raw token IDs",
        "gate": "all rows and IDs; max_abs_error/max(1,max_abs_HF_row)<=1e-4",
        "workload": f"last-row-only prefill plus {a.steps - 1} cached single-token forwards for {a.steps} output rows; fixed teacher forcing including EOS",
        "timing_scope": "Synchronous CUDA runtime forward with CPU input-ID H2D and full final-row D2H included in both timers; HF Python dispatch/cache/output wrappers and differing KV allocation strategies included; Ember KV preallocated versus HF DynamicCache. No isolated kernel or general engine-ranking claim.",
        "threads": 1,
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "logical_cpus_available": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "EMBER_IR_CTX",
                "EMBER_IR",
                "EMBER_IR_BUILD",
                "EMBER_KV",
                "EMBER_DETERMINISTIC",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_TF32_OVERRIDE",
                "EMBER_DECODE_LINEAR",
                "EMBER_DECODE_PICK",
            )
        },
    }
    if any(
        os.environ.get(k)
        for k in (
            "EMBER_IR",
            "EMBER_KV",
            "EMBER_DETERMINISTIC",
            "EMBER_DECODE_LINEAR",
            "EMBER_DECODE_PICK",
        )
    ):
        raise ValueError(
            "unexpected interpreter overrides; use the default FP32 CUDA path"
        )
    verify_files(integrity_files)  # immediately before loading either implementation
    hf = HF(a.gguf)
    verify_files(integrity_files)
    ember = Ember(a.binary, a.gguf, a.out)
    try:
        verify_files(integrity_files)
        protocol["model_metadata"] = validate_native_metadata(ember.ready, hf.model.config.vocab_size)
        path = a.out / "protocol.json"
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError("protocol/input/source drift")
        else:
            atomic_json(path, protocol)
        journal = a.out / "trials.jsonl"
        done = journal_rows(journal, protocol, sha(path))
        for length in a.lengths:
            if length + a.steps > ember.ready["max_len"]:
                raise ValueError("requested context unavailable")
            prompt = [(11 + 37 * i) % ember.ready["vocab"] for i in range(length)]
            inputs, gate_path = (
                a.out / f"inputs_{length}.json",
                a.out / f"gate_{length}.json",
            )
            hf_dump, ember_dump = (
                a.out / f"hf_{length}.f32",
                a.out / f"ember_{length}.f32",
            )
            if gate_path.exists():
                data = check_gate(a.out, length, a.steps, protocol["model_metadata"]["vocab"])
                if data["prompt_ids"] != prompt:
                    raise ValueError("frozen prompt changed")
                continuation = data["continuation_ids"]
            else:
                seed = hf.run(prompt, a.steps, dump=hf_dump)
                continuation = seed["argmax"]
                if inputs.exists() and json.loads(inputs.read_text()) != {
                    "prompt_ids": prompt,
                    "continuation_ids": continuation,
                }:
                    raise ValueError("frozen continuation changed")
                atomic_json(
                    inputs, {"prompt_ids": prompt, "continuation_ids": continuation}
                )
                other = ember.run(f"gate_{length}", prompt, continuation, ember_dump)
                gate = assert_parity(
                    np.fromfile(hf_dump, dtype="<f4").reshape(a.steps, -1),
                    np.fromfile(ember_dump, dtype="<f4").reshape(a.steps, -1),
                    continuation,
                    other["argmax"],
                )
                gate.update(
                    inputs_sha256=sha(inputs),
                    hf_rows_sha256=sha(hf_dump),
                    ember_rows_sha256=sha(ember_dump),
                )
                atomic_json(gate_path, gate)
            verify_files(integrity_files)
            check_gate(a.out, length, a.steps, protocol["model_metadata"]["vocab"])
            for row in done:
                if row["length"] == length:
                    validate_trial(row, continuation, a.steps)
                    if row.get("inputs_sha256") != sha(inputs):
                        raise ValueError("trial inputs changed")
            # One unmeasured warm request per engine at this shape.
            validate_trial(hf.run(prompt, a.steps, continuation), continuation, a.steps)
            validate_trial(
                ember.run("warm", prompt, continuation), continuation, a.steps
            )
            for cycle in range(a.cycles):
                for slot, engine in enumerate(protocol["sequence"]):
                    key = f"{length}_{cycle}_{slot}_{engine}"
                    if key in {r["key"] for r in done}:
                        continue
                    verify_files(integrity_files)
                    check_gate(a.out, length, a.steps, protocol["model_metadata"]["vocab"])
                    row = (
                        hf.run(prompt, a.steps, continuation)
                        if engine == "hf"
                        else ember.run(key, prompt, continuation)
                    )
                    validate_trial(row, continuation, a.steps)
                    verify_files(integrity_files)
                    check_gate(a.out, length, a.steps, protocol["model_metadata"]["vocab"])
                    row.update(
                        key=key, length=length, engine=engine, inputs_sha256=sha(inputs)
                    )
                    row["previous_sha256"] = (
                        done[-1]["record_sha256"] if done else sha(path)
                    )
                    row["record_sha256"] = digest_record(row)
                    with journal.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    done.append(row)
                    print(
                        key,
                        "prefill",
                        round(row["forward_ms"][0], 2),
                        "decode",
                        round(statistics.mean(row["forward_ms"][1:]), 2),
                        flush=True,
                    )
        verify_files(integrity_files)
        for length in a.lengths:
            check_gate(a.out, length, a.steps, protocol["model_metadata"]["vocab"])
        atomic_json(a.out / "integrity.json", {
            "protocol_sha256": sha(path), "journal_sha256": sha(journal),
            "files": integrity_files,
            "inputs": {str(length): sha(a.out / f"inputs_{length}.json") for length in a.lengths},
            "status": "FINAL_INTEGRITY_VERIFIED",
        })
        summary = analyze(a.out)
        atomic_json(a.out / "summary.json", summary)
        print(json.dumps(summary["results"], indent=2))
    finally:
        ember.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 512])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument(
        "--analyze",
        action="store_true",
        help="replay all raw gates, journal and summary without loading a model",
    )
    a = p.parse_args()
    if a.analyze:
        print(json.dumps(analyze(a.out), indent=2))
        return
    if not a.gguf or not a.binary:
        p.error("--gguf and --binary required for run")
    if (
        a.steps < 2
        or a.cycles < 1
        or not a.lengths
        or min(a.lengths) < 1
        or len(set(a.lengths)) != len(a.lengths)
    ):
        p.error(
            "positive distinct lengths/cycles and at least two output rows required"
        )
    try:
        run(a)
    except BaseException as e:
        if a.out.exists() and not (a.out / "HALTED").exists():
            (a.out / "HALTED").write_text(type(e).__name__ + ": " + str(e) + "\n")
        raise


if __name__ == "__main__":
    main()
