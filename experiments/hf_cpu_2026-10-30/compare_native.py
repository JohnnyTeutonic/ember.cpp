"""Paired old/new native CPU timings, with bit-exact full-model output gate.

Uses completed HF-gated raw-token fixtures. No tokenizer, sampling or model/API
calls outside the two local native CPU processes. Completed trials are immutable.
"""

from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path
import statistics
from benchmark import (
    Ember,
    analyze as reference_analyze,
    atomic_json,
    sha,
    require_thread_budget,
    validate_trial,
    journal_rows,
    digest_record,
)


def plan(protocol):
    return [
        (f"{length}_{cycle}_{slot}_{engine}", length, engine)
        for length in protocol["lengths"]
        for cycle in range(protocol["cycles"])
        for slot, engine in enumerate(protocol["sequence"])
    ]


def analyze(out):
    protocol_path = out / "protocol.json"
    p = json.loads(protocol_path.read_text())
    if p["controller_sha256"] != sha(__file__) or p["shared_controller_sha256"] != sha(
        Path(__file__).with_name("benchmark.py")
    ):
        raise ValueError("controller drift")
    ref = Path(p["reference"])
    reference_analyze(ref)
    if (
        sha(ref / "protocol.json") != p["reference_protocol_sha256"]
        or sha(ref / "trials.jsonl") != p["reference_journal_sha256"]
    ):
        raise ValueError("reference changed")
    rows = journal_rows(out / "trials.jsonl", p, sha(protocol_path))
    if len(rows) != len(plan(p)):
        raise ValueError("incomplete timing inventory")
    result = {}
    for length in p["lengths"]:
        inputs = ref / f"inputs_{length}.json"
        data = json.loads(inputs.read_text())
        expected_sha = sha(ref / f"ember_{length}.f32")
        point = {}
        for engine in ("baseline", "optimized"):
            gate = json.loads((out / f"gate_{length}_{engine}.json").read_text())
            if gate != {
                "reference_rows_sha256": expected_sha,
                "actual_rows_sha256": expected_sha,
                "inputs_sha256": sha(inputs),
                "binary_sha256": p["binaries"][engine]["sha256"],
                "all_rows_bit_identical": True,
            }:
                raise ValueError("full-row bit gate differs")
            selected = [
                r for r in rows if r["length"] == length and r["engine"] == engine
            ]
            for r in selected:
                validate_trial(r, data["continuation_ids"], p["steps"])
                if r["inputs_sha256"] != sha(inputs):
                    raise ValueError("trial inputs changed")
            prefill = [r["forward_ms"][0] for r in selected]
            decode = [statistics.mean(r["forward_ms"][1:]) for r in selected]
            point[engine] = {
                "trials": len(selected),
                "prefill_ms_all": prefill,
                "decode_ms_per_token_all": decode,
                "prefill_ms_median": statistics.median(prefill),
                "decode_ms_per_token_median": statistics.median(decode),
            }
        point["prefill_speedup"] = (
            point["baseline"]["prefill_ms_median"]
            / point["optimized"]["prefill_ms_median"]
        )
        point["decode_speedup"] = (
            point["baseline"]["decode_ms_per_token_median"]
            / point["optimized"]["decode_ms_per_token_median"]
        )
        result[str(length)] = point
    summary = {
        "status": "COMPLETE_BIT_IDENTICAL_PAIRED_NATIVE_CPU",
        "protocol_sha256": sha(protocol_path),
        "journal_sha256": sha(out / "trials.jsonl"),
        "results": result,
    }
    if (out / "summary.json").exists() and json.loads(
        (out / "summary.json").read_text()
    ) != summary:
        raise ValueError("saved summary disagrees")
    return summary


def run(a):
    require_thread_budget(os.environ)
    reference_analyze(a.reference)
    old = json.loads((a.reference / "protocol.json").read_text())
    if sha(a.baseline) != old["binary_sha256"]:
        raise ValueError("wrong baseline binary")
    gguf = Path(old["gguf"])
    if sha(gguf) != old["gguf_sha256"]:
        raise ValueError("weights drift")
    if any(os.environ.get(k) for k in ("EMBER_IR", "EMBER_KV", "EMBER_DETERMINISTIC")):
        raise ValueError("unexpected overrides")
    for k in ("EMBER_IR_BUILD", "EMBER_IR_CTX"):
        if os.environ.get(k) != old["environment"][k]:
            raise ValueError("environment mismatch")
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    if cpu_model != old["cpu_model"]:
        raise ValueError("reference CPU differs from current CPU")
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "HALTED").exists():
        raise ValueError("preserved HALTED receipt")
    p = {
        "schema": 1,
        "controller_sha256": sha(__file__),
        "shared_controller_sha256": sha(Path(__file__).with_name("benchmark.py")),
        "reference": str(a.reference),
        "reference_protocol_sha256": sha(a.reference / "protocol.json"),
        "reference_journal_sha256": sha(a.reference / "trials.jsonl"),
        "gguf_sha256": sha(gguf),
        "lengths": old["lengths"],
        "steps": old["steps"],
        "cycles": 2,
        "sequence": ["baseline", "optimized", "optimized", "baseline"],
        "binaries": {
            k: {"path": str(v), "sha256": sha(v)}
            for k, v in [("baseline", a.baseline), ("optimized", a.optimized)]
        },
        "scope": "Single-thread native forward latency; identical FP32 weights/activations and fixed-order FP64 dot accumulation; cache/output wrappers included. Shared-host interference remains possible.",
        "environment": {k: os.environ.get(k) for k in old["environment"]},
        "cpu_model": cpu_model,
        "logical_cpus_available": len(os.sched_getaffinity(0)),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "native_sources": {
            name: sha(Path(__file__).resolve().parents[2] / name)
            for name in (
                "cpu_dot.h",
                "utils.cpp",
                "CMakeLists.txt",
                "tools/ir_hf_cpu_bench.cpp",
            )
        },
    }
    path = a.out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != p:
            raise ValueError("protocol drift")
    else:
        atomic_json(path, p)
    done = journal_rows(a.out / "trials.jsonl", p, sha(path))
    engines = {}
    try:
        for engine, binary in [("baseline", a.baseline), ("optimized", a.optimized)]:
            engines[engine] = Ember(binary, gguf, a.out)
        for length in p["lengths"]:
            inputs = a.reference / f"inputs_{length}.json"
            data = json.loads(inputs.read_text())
            ids = data["continuation_ids"]
            prompt = data["prompt_ids"]
            expected = sha(a.reference / f"ember_{length}.f32")
            for engine, instance in engines.items():
                gate_path = a.out / f"gate_{length}_{engine}.json"
                gate = {
                    "reference_rows_sha256": expected,
                    "actual_rows_sha256": expected,
                    "inputs_sha256": sha(inputs),
                    "binary_sha256": p["binaries"][engine]["sha256"],
                    "all_rows_bit_identical": True,
                }
                if gate_path.exists():
                    if json.loads(gate_path.read_text()) != gate:
                        raise ValueError("gate drift")
                else:
                    dump = a.out / f"gate_{length}_{engine}.f32"
                    row = instance.run(f"gate_{length}_{engine}", prompt, ids, dump)
                    validate_trial(row, ids, p["steps"])
                    if sha(dump) != expected:
                        raise ValueError(
                            "full native output bits changed; raw failure preserved"
                        )
                    atomic_json(gate_path, gate)
                    dump.unlink()  # Exact bytes already retained in the immutable reference; no duplicate raw matrix.
                validate_trial(instance.run("warm", prompt, ids), ids, p["steps"])
            for row in done:
                if row["length"] == length:
                    validate_trial(row, ids, p["steps"])
                    if row["inputs_sha256"] != sha(inputs):
                        raise ValueError("old trial inputs differ")
            for key, n, engine in plan(p):
                if n != length or key in {r["key"] for r in done}:
                    continue
                row = engines[engine].run(key, prompt, ids)
                validate_trial(row, ids, p["steps"])
                row.update(
                    key=key,
                    length=length,
                    engine=engine,
                    inputs_sha256=sha(inputs),
                    previous_sha256=done[-1]["record_sha256"] if done else sha(path),
                )
                row["record_sha256"] = digest_record(row)
                with (a.out / "trials.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                done.append(row)
                print(
                    key,
                    round(row["forward_ms"][0], 2),
                    round(statistics.mean(row["forward_ms"][1:]), 2),
                    flush=True,
                )
        summary = analyze(a.out)
        atomic_json(a.out / "summary.json", summary)
        print(json.dumps(summary["results"], indent=2))
    finally:
        for instance in engines.values():
            instance.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--optimized", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        print(json.dumps(analyze(a.out), indent=2))
        return
    if not all((a.reference, a.baseline, a.optimized)):
        ap.error("reference and both binaries required")
    try:
        run(a)
    except BaseException as e:
        if a.out.exists():
            (a.out / "HALTED").write_text(type(e).__name__ + ": " + str(e) + "\n")
        raise


if __name__ == "__main__":
    main()
