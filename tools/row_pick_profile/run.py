"""Fail-closed standalone selector experiment; no production edits or model calls."""

import argparse
import csv
import hashlib
import json
import os
import math
import shutil
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def validate_records(records, baseline=False):
    vocabularies = [4096, 16384, 32000, 32768, 65536, 128256]
    expected_kernel = {
        (v, n, p, b)
        for v in vocabularies
        for n in [1, 4, 16]
        for p in ([0] if baseline else [0, 4, 16, 64])
        for b in range(10)
    }
    expected_host = {
        (v, 1, p, b, t)
        for v in vocabularies
        for p in ([0] if baseline else [0, 4, 16, 64])
        for b in range(10)
        for t in [0, 1]
    }
    expected_correct = {
        (v, n)
        for v in [1, 31, 257, 4096, 16384, 32000, 32768, 65536, 128256]
        for n in [1, 4, 16]
    }
    seen = {"kernel_trial": set(), "host_trial": set(), "correctness": set()}
    counts = {}
    for row in records:
        kind = row.get("kind")
        counts[kind] = counts.get(kind, 0) + 1
        if kind in seen:
            fields = {
                "kernel_trial": ["vocab", "n", "parts", "batch"],
                "host_trial": ["vocab", "n", "parts", "batch", "transfer_only"],
                "correctness": ["vocab", "n"],
            }[kind]
            key = tuple(row[f] for f in fields)
            if key in seen[kind]:
                raise ValueError("duplicate trial identity")
            seen[kind].add(key)
        if kind in {"kernel_trial", "host_trial"}:
            if not math.isfinite(row["microseconds"]) or row["microseconds"] <= 0:
                raise ValueError("nonfinite/nonpositive timing")
        if kind == "kernel_trial":
            if (
                row.get("iterations") != 1000
                or row.get("raw_elapsed_ms") != row["microseconds"]
            ):
                raise ValueError("kernel iteration/raw elapsed inconsistency")
        if kind == "device" and (row.get("name") != "NVIDIA L4" or row.get("sm") != 89):
            raise ValueError("actual device not NVIDIA L4 SM89")
    if seen != {
        "kernel_trial": expected_kernel,
        "host_trial": expected_host,
        "correctness": expected_correct,
    }:
        raise ValueError("missing/unknown trial identities")
    if counts != {
        "device": 1,
        "correctness": 27,
        "kernel_trial": 180 if baseline else 720,
        "host_trial": 120 if baseline else 480,
        "COMPLETE": 1,
    }:
        raise ValueError("unexpected record inventory")
    if records[-1].get("kind") != "COMPLETE" or records[-1].get("checked_rows") != (
        2268 if baseline else 6804
    ):
        raise ValueError("incomplete correctness rows")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    for relative in [
        "tools/row_pick_profile/prepare.py",
        "tools/row_pick_profile/run.py",
        "tools/row_pick_profile/profile.cu",
        "verify_kernels.cu",
        "verify_kernels.h",
    ]:
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    started = time.time()
    receipt = {
        "status": "RUNNING",
        "baseline_only": args.baseline_only,
        "started_unix": started,
        "scope": "component only; production/defaults unchanged; full-model 15% gate not tested",
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                HERE / "prepare.py",
                HERE / "profile.cu",
                HERE / "run.py",
                ROOT / "verify_kernels.cu",
                ROOT / "verify_kernels.h",
            ]
        },
        "platform": platform.platform(),
        "commands": [],
    }
    atomic(out / "receipt.json", receipt)
    try:
        commands = [
            [
                sys.executable,
                str(HERE / "prepare.py"),
                "--root",
                str(ROOT),
                "--output",
                str(out),
            ],
            [args.nvcc, "--version"],
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv",
            ],
            [
                args.nvcc,
                "-std=c++17",
                "-O3",
                "-arch=sm_89",
                "-I" + str(out),
                str(HERE / "profile.cu"),
                "-o",
                str(out / "profile"),
            ],
            [str(out / "profile"), str(out / "trials.jsonl"), str(out / "inputs.bin")],
        ]
        if args.baseline_only:
            commands[-1].append("--baseline-only")
        for index, command in enumerate(commands):
            record = {"argv": command, "started_unix": time.time()}
            receipt["commands"].append(record)
            atomic(out / "receipt.json", receipt)
            with (out / f"command_{index}.log").open("w") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    check=False,
                )
            record.update(returncode=process.returncode, ended_unix=time.time())
            atomic(out / "receipt.json", receipt)
            if process.returncode:
                raise RuntimeError(f"command {index} failed; no retry")
            if index == 2:
                gpu_rows = list(
                    csv.reader((out / "command_2.log").read_text().splitlines())
                )
                if len(gpu_rows) != 2 or len(gpu_rows[1]) != 4:
                    raise RuntimeError("ambiguous GPU inventory")
                values = [value.strip() for value in gpu_rows[1]]
                if values[0] != "NVIDIA L4" or not values[1].startswith("GPU-"):
                    raise RuntimeError("GPU name/UUID not verified")
                receipt["gpu"] = dict(
                    zip(["name", "uuid", "driver", "memory_total"], values)
                )
        records = [
            json.loads(line) for line in (out / "trials.jsonl").read_text().splitlines()
        ]
        if not records or records[-1].get("kind") != "COMPLETE":
            raise RuntimeError("missing terminal COMPLETE")
        counts = validate_records(records, args.baseline_only)
        receipt.update(
            status="COMPLETE",
            record_counts=counts,
            checked_rows=records[-1]["checked_rows"],
        )
    except Exception as exc:
        receipt.update(status="HALTED", error=str(exc), no_retry=True)
    finally:
        receipt["ended_unix"] = time.time()
        receipt["artifacts"] = {
            str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in out.rglob("*")
            if p.is_file() and p.name not in {"receipt.json", "receipt.tmp"}
        }
        receipt["input_capture"] = {
            "retained": False,
            "regeneration": "standard mt19937 seed20261031; uint32 modulo65536 and exact power-of-two scale; source hash pinned",
        }
        atomic(out / "receipt.json", receipt)
        if (out / "inputs.bin").exists():
            (
                out / "inputs.bin"
            ).unlink()  # newly generated scratch; digest and size retained
    # A negative component result is durably home, not a request for automatic rerun.
    print(json.dumps({"status": receipt["status"], "output": str(out)}))


if __name__ == "__main__":
    main()
