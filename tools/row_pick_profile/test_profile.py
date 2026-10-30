"""CPU-only extraction, partition algebra and wire/fallback protocol checks."""

import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("prepare_profile", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def summary(values, offset=0):
    finite = [v for v in values if not math.isnan(v)]
    maximum = max(finite, default=-math.inf)
    indices = [offset + i for i, v in enumerate(values) if v == maximum]
    return (
        maximum,
        max((v for v in finite if v < maximum), default=-math.inf),
        min(indices, default=2**31 - 1),
        max(indices, default=-1),
        int(any(math.isnan(v) for v in values)),
    )


def combine(parts):
    maximum = max(p[0] for p in parts)
    matched = [p for p in parts if p[0] == maximum]
    second = max([p[1] if p[0] == maximum else p[0] for p in parts], default=-math.inf)
    return (
        maximum,
        second,
        min(p[2] for p in matched),
        max(p[3] for p in matched),
        int(any(p[4] for p in parts)),
    )


@pytest.mark.parametrize("parts", [4, 16, 64])
def test_partition_algebra(parts):
    rng = random.Random(20261031)
    for size in [1, 2, 31, 257, 4096, 16384]:
        cases = [[rng.uniform(-20, 20) for _ in range(size)] for _ in range(10)]
        cases += [[v] * size for v in [0.0, -0.0, -math.inf, math.inf, math.nan, 2.0]]
        for row in cases:
            assert combine(
                [
                    summary(
                        row[size * p // parts : size * (p + 1) // parts],
                        size * p // parts,
                    )
                    for p in range(parts)
                ]
            ) == summary(row)


def test_exact_extraction_and_only_range_delta(tmp_path):
    source = (ROOT / "verify_kernels.cu").read_text()
    original = prepare.extract(source)
    assert original.startswith("__global__ void select_rows_kernel(")
    assert original.count("for (int i = threadIdx.x; i < vocab; i += blockDim.x)") == 2
    subprocess.run(
        [
            sys.executable,
            str(HERE / "prepare.py"),
            "--root",
            str(ROOT),
            "--output",
            str(tmp_path),
        ],
        check=True,
    )
    header = (tmp_path / "selector.cuh").read_text()
    assert original in header
    assert header.count("(long long)vocab * blockIdx.y / parts") == 2
    assert "picks[blockIdx.x * parts + blockIdx.y] = p;" in header


@pytest.mark.parametrize("failure", ["exit", "timeout", "partial", "complete"])
def test_wrapper_terminal_contract(tmp_path, monkeypatch, failure):
    import json
    from types import SimpleNamespace

    wrapper_spec = importlib.util.spec_from_file_location(
        "profile_runner", HERE / "run.py"
    )
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper)
    output = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["run.py", "--output", str(output)])

    def fake(command, **kwargs):
        if command[0] == "nvidia-smi":
            kwargs["stdout"].write(
                "name, uuid, driver_version, memory.total\nNVIDIA L4, GPU-test, 550, 23034 MiB\n"
            )
            kwargs["stdout"].flush()
        if command[0] == str(output / "profile"):
            (output / "inputs.bin").write_bytes(b"fixed captured inputs")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 900)
            if failure == "exit":
                return SimpleNamespace(returncode=1)
            rows = complete_records()
            if failure == "partial":
                rows.pop()
            (output / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(wrapper.subprocess, "run", fake)
    wrapper.main()
    result = json.loads((output / "receipt.json").read_text())
    assert result["status"] == ("COMPLETE" if failure == "complete" else "HALTED")
    assert len(result["commands"]) == 5
    assert result["artifacts"]["inputs.bin"]["bytes"] == len(b"fixed captured inputs")
    assert not (output / "inputs.bin").exists()
    with pytest.raises(FileExistsError):
        wrapper.main()


def complete_records():
    rows = [{"kind": "device", "name": "NVIDIA L4", "sm": 89}]
    rows += [
        {"kind": "correctness", "vocab": v, "n": n}
        for v in [1, 31, 257, 4096, 16384, 32000, 32768, 65536, 128256]
        for n in [1, 4, 16]
    ]
    for v in [4096, 16384, 32000, 32768, 65536, 128256]:
        for n in [1, 4, 16]:
            for p in [0, 4, 16, 64]:
                for b in range(10):
                    rows.append(
                        {
                            "kind": "kernel_trial",
                            "vocab": v,
                            "n": n,
                            "parts": p,
                            "batch": b,
                            "microseconds": 2.0,
                            "iterations": 1000,
                            "raw_elapsed_ms": 2.0,
                        }
                    )
                    if n == 1:
                        for t in [0, 1]:
                            rows.append(
                                {
                                    "kind": "host_trial",
                                    "vocab": v,
                                    "n": n,
                                    "parts": p,
                                    "batch": b,
                                    "transfer_only": t,
                                    "microseconds": 5.0,
                                }
                            )
    rows.append({"kind": "COMPLETE", "checked_rows": 6804})
    return rows


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "missing",
        "nan",
        "zero",
        "wrong_gpu",
        "iterations",
        "wrong_identity",
    ],
)
def test_trial_inventory_refuses_damage(damage):
    spec = importlib.util.spec_from_file_location("runner_validator", HERE / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = complete_records()
    runner.validate_records(rows)
    kernel = next(r for r in rows if r["kind"] == "kernel_trial")
    if damage == "duplicate":
        rows.insert(-1, dict(kernel))
    elif damage == "missing":
        rows.remove(kernel)
    elif damage == "nan":
        kernel["microseconds"] = math.nan
    elif damage == "zero":
        kernel["microseconds"] = 0
    elif damage == "wrong_gpu":
        rows[0]["name"] = "NVIDIA T4"
    elif damage == "iterations":
        kernel["iterations"] = 1
    elif damage == "wrong_identity":
        kernel["vocab"] = 42
    with pytest.raises(ValueError):
        runner.validate_records(rows)


def test_baseline_inventory_independent_of_candidate():
    spec = importlib.util.spec_from_file_location("baseline_validator", HERE / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = [r for r in complete_records() if r.get("parts", 0) == 0]
    rows[-1]["checked_rows"] = 2268
    counts = runner.validate_records(rows, baseline=True)
    assert counts["kernel_trial"] == 180
    assert counts["host_trial"] == 120
    with pytest.raises(ValueError):
        runner.validate_records(rows)
    damaged = list(rows)
    damaged.insert(-1, {"kind": "MISMATCH"})
    with pytest.raises(ValueError):
        runner.validate_records(damaged, baseline=True)
