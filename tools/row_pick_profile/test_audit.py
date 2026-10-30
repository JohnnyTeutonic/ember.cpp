"""Independent raw-evidence audit refusal cases, without a GPU."""

import importlib.util
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_identity_inventory():
    audit = load("audit")
    rows = load("test_profile").complete_records()
    assert audit.audit_trials(rows, False)["counts"]["kernel_trial"] == 720
    baseline = [r for r in rows if r.get("parts", 0) == 0]
    baseline[-1]["checked_rows"] = 2268
    assert audit.audit_trials(baseline, True)["counts"]["host_trial"] == 120


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing", "wrong_identity", "nonfinite", "raw_ms"]
)
def test_corrupted_raw_evidence_refused(damage):
    audit = load("audit")
    rows = load("test_profile").complete_records()
    kernel = next(r for r in rows if r["kind"] == "kernel_trial")
    if damage == "duplicate":
        rows.insert(-1, dict(kernel))
    elif damage == "missing":
        rows.remove(kernel)
    elif damage == "wrong_identity":
        kernel["vocab"] = 42
    elif damage == "nonfinite":
        kernel["microseconds"] = math.inf
    elif damage == "raw_ms":
        kernel["raw_elapsed_ms"] = 0
    with pytest.raises(ValueError):
        audit.audit_trials(rows, False)


def test_halted_candidate_retains_explicit_mismatch_without_timing():
    audit = load("audit")
    rows = [
        {"kind": "device", "name": "NVIDIA L4", "sm": 89},
        {
            "kind": "MISMATCH",
            "vocab": 31,
            "n": 1,
            "fixture": 6,
            "parts": 4,
            "row": 0,
            "original": [0, 0, 0, 30, 0],
            "candidate": [2147483648, 0, 0, 30, 0],
        },
    ]
    report = audit.audit_trials(rows, False)
    assert not report["complete"]
    assert report["first_mismatch"]["fixture"] == 6
    rows.append(
        {
            "kind": "kernel_trial",
            "vocab": 4096,
            "n": 1,
            "parts": 0,
            "batch": 0,
            "iterations": 1000,
            "raw_elapsed_ms": 1.0,
            "microseconds": 1.0,
        }
    )
    with pytest.raises(ValueError, match="before complete parity"):
        audit.audit_trials(rows, False)
