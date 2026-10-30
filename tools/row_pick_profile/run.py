"""Fail-closed standalone selector experiment; no production edits or model calls."""

import argparse
import hashlib
import json
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    started = time.time()
    receipt = {
        "status": "RUNNING",
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
                "-arch=sm_89",
                "-I" + str(out),
                str(HERE / "profile.cu"),
                "-o",
                str(out / "profile"),
            ],
            [str(out / "profile"), str(out / "trials.jsonl"), str(out / "inputs.bin")],
        ]
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
        records = [
            json.loads(line) for line in (out / "trials.jsonl").read_text().splitlines()
        ]
        if not records or records[-1].get("kind") != "COMPLETE":
            raise RuntimeError("missing terminal COMPLETE")
        expected = {
            "device": 1,
            "correctness": 27,
            "kernel_trial": 720,
            "host_trial": 480,
            "COMPLETE": 1,
        }
        counts = {
            kind: sum(r.get("kind") == kind for r in records) for kind in expected
        }
        if counts != expected or any(r.get("kind") not in expected for r in records):
            raise RuntimeError(f"incomplete/unknown trial inventory: {counts}")
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
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in out.iterdir()
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
