"""Offline native comparison replay: a complete pair or an explicit failure."""

import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_native as c  # noqa: E402


def fixture(tmp, monkeypatch):
    ref = tmp / "reference"
    ref.mkdir()
    out = tmp / "native"
    out.mkdir()
    (ref / "protocol.json").write_text("{}")
    (ref / "trials.jsonl").write_text("original trials\n")
    (ref / "inputs_16.json").write_text(
        json.dumps({"prompt_ids": [3], "continuation_ids": [1, 2]})
    )
    (ref / "ember_16.f32").write_bytes(bytes(16))
    monkeypatch.setattr(c, "reference_analyze", lambda path: {})
    p = {
        "controller_sha256": c.sha(c.__file__),
        "shared_controller_sha256": c.sha(Path(c.__file__).with_name("benchmark.py")),
        "reference": str(ref),
        "reference_protocol_sha256": c.sha(ref / "protocol.json"),
        "reference_journal_sha256": c.sha(ref / "trials.jsonl"),
        "lengths": [16],
        "cycles": 1,
        "steps": 2,
        "sequence": ["baseline", "optimized", "optimized", "baseline"],
        "binaries": {"baseline": {"sha256": "old"}, "optimized": {"sha256": "new"}},
    }
    c.atomic_json(out / "protocol.json", p)
    for engine in p["binaries"]:
        h = c.sha(ref / "ember_16.f32")
        c.atomic_json(
            out / f"gate_16_{engine}.json",
            {
                "reference_rows_sha256": h,
                "actual_rows_sha256": h,
                "inputs_sha256": c.sha(ref / "inputs_16.json"),
                "binary_sha256": p["binaries"][engine]["sha256"],
                "all_rows_bit_identical": True,
            },
        )
    rows = []
    previous = c.sha(out / "protocol.json")
    for key, length, engine in c.plan(p):
        row = {
            "key": key,
            "length": length,
            "engine": engine,
            "argmax": [1, 2],
            "forward_ms": [10.0, 20.0] if engine == "baseline" else [5.0, 10.0],
            "inputs_sha256": c.sha(ref / "inputs_16.json"),
            "previous_sha256": previous,
        }
        row["record_sha256"] = c.digest_record(row)
        previous = row["record_sha256"]
        rows.append(row)
    (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out, ref, p, rows


def test_complete_replay_and_summary_agree(tmp_path, monkeypatch):
    out, _, _, _ = fixture(tmp_path, monkeypatch)
    result = c.analyze(out)
    assert result["results"]["16"]["prefill_speedup"] == 2
    assert result["results"]["16"]["decode_speedup"] == 2
    c.atomic_json(out / "summary.json", result)
    assert c.analyze(out) == result


@pytest.mark.parametrize(
    "mutation", ["reference", "gate", "trial", "missing", "summary", "source"]
)
def test_tamper_or_incomplete_data_never_becomes_a_speedup(
    tmp_path, monkeypatch, mutation
):
    out, ref, p, rows = fixture(tmp_path, monkeypatch)
    if mutation == "reference":
        (ref / "trials.jsonl").write_text("changed\n")
    if mutation == "gate":
        path = out / "gate_16_optimized.json"
        g = json.loads(path.read_text())
        g["actual_rows_sha256"] = "different"
        c.atomic_json(path, g)
    if mutation == "trial":
        rows[0]["forward_ms"][0] = 0.001
        (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if mutation == "missing":
        (out / "trials.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows[:-1])
        )
    if mutation == "summary":
        c.atomic_json(out / "summary.json", {"invented": "speedup"})
    if mutation == "source":
        p["controller_sha256"] = "different"
        c.atomic_json(out / "protocol.json", p)
    with pytest.raises(ValueError):
        c.analyze(out)
