"""Numerical/inventory and immutable timing-record gates; no model or GPU."""

import copy
import json
from pathlib import Path
import importlib.util
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "hf_cuda_benchmark_test", Path(__file__).with_name("benchmark.py")
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_complete_rounding_gate():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    got = ref + 1e-6
    assert b.assert_parity(ref, got, [1, 1], [1, 1])["rows"] == 2


@pytest.mark.parametrize("bad", ["ids", "missing_row", "nan", "drift"])
def test_numerical_failures_never_become_partial_comparisons(bad):
    ref = np.ones((2, 3), dtype=np.float32)
    got = ref.copy()
    ids = [0, 0]
    if bad == "ids":
        ids = [0, 1]
    if bad == "missing_row":
        got = got[:1]
    if bad == "nan":
        got[0, 0] = np.nan
    if bad == "drift":
        got[0, 0] += 0.01
    with pytest.raises(ValueError):
        b.assert_parity(ref, got, [0, 0], ids)


@pytest.mark.parametrize("times", [[1.0], [1.0, float("inf")], [1.0, 0.0], [-1.0, 2.0]])
def test_invalid_or_incomplete_timing_fails(times):
    with pytest.raises(ValueError):
        b.validate_trial({"argmax": [1, 2], "forward_ms": times}, [1, 2], 2)


def test_hash_chain_detects_changed_data_and_reordering(tmp_path):
    protocol = {
        "lengths": [16],
        "cycles": 1,
        "sequence": ["ember", "hf", "hf", "ember"],
    }
    previous = "protocol"
    rows = []
    for key, length, engine in b.trial_plan(protocol):
        row = {
            "key": key,
            "length": length,
            "engine": engine,
            "argmax": [1, 2],
            "forward_ms": [2.0, 3.0],
            "previous_sha256": previous,
        }
        row["record_sha256"] = b.digest_record(row)
        previous = row["record_sha256"]
        rows.append(row)
    path = tmp_path / "trials.jsonl"

    def save(rs):
        path.write_text("".join(json.dumps(r) + "\n" for r in rs))

    save(rows)
    assert len(b.journal_rows(path, protocol, "protocol")) == 4
    changed = copy.deepcopy(rows)
    changed[1]["forward_ms"][0] = 0.01
    save(changed)
    with pytest.raises(ValueError, match="hash chain"):
        b.journal_rows(path, protocol, "protocol")
    save(rows[::-1])
    with pytest.raises(ValueError, match="order"):
        b.journal_rows(path, protocol, "protocol")
    save(rows)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="torn"):
        b.journal_rows(path, protocol, "protocol")


def test_missing_paired_trial_does_not_produce_speedup():
    rows = [{"length": 16, "engine": "hf", "forward_ms": [1.0, 2.0]}]
    with pytest.raises(ValueError, match="incomplete"):
        b.summarize(rows, [16], 1)


def test_standalone_run_cannot_silently_give_ember_more_threads():
    env = {
        k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    b.require_thread_budget(env)
    for key in env:
        bad = {**env, key: "4"}
        with pytest.raises(ValueError, match="one-core"):
            b.require_thread_budget(bad)


def test_cuda_forward_timer_includes_input_and_output_transfers(monkeypatch):
    operations = []

    class Row:
        def float(self):
            return self

        def cpu(self):
            operations.append("D2H")
            return self

        def argmax(self):
            operations.append("argmax")
            return 1

    class Logits:
        shape = (1, 1, 2)

        def __getitem__(self, key):
            return Row()

    class Tensor:
        def to(self, device):
            operations.append("H2D")
            return self

    def tensor(ids, dtype):
        operations.append("CPU input")
        return Tensor()

    def forward(*args, **kwargs):
        operations.append("forward")
        return SimpleNamespace(logits=Logits(), past_key_values=object())

    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        tensor=tensor,
        long="int64",
        cuda=SimpleNamespace(synchronize=lambda device: operations.append("sync")),
    )
    hf = b.HF.__new__(b.HF)
    hf.torch = fake_torch
    hf.device = "cuda:0"
    hf.model = forward

    def clock():
        operations.append("timer")
        return len(operations) * 1000000

    monkeypatch.setattr(b.time, "perf_counter_ns", clock)
    result = hf.run([1, 2], 2, continuation=[1, 1])
    assert result["argmax"] == [1, 1]
    assert (
        operations
        == [
            "CPU input",
            "sync",
            "timer",
            "H2D",
            "forward",
            "D2H",
            "sync",
            "timer",
            "argmax",
        ]
        * 2
    )


@pytest.mark.parametrize("change", ["length", "formula", "range", "bool", "missing"])
def test_frozen_input_contract_rejects_drift(change):
    data = {"prompt_ids": [11, 48], "continuation_ids": [1, 2]}
    if change == "length":
        data["prompt_ids"].pop()
    elif change == "formula":
        data["prompt_ids"][0] = 12
    elif change == "range":
        data["continuation_ids"][0] = 64
    elif change == "bool":
        data["continuation_ids"][0] = True
    else:
        data["continuation_ids"].pop()
    with pytest.raises(ValueError, match="frozen"):
        b.validate_inputs(data, 2, 2, 64)


def test_source_binary_or_input_mutation_rejected(tmp_path):
    files = [tmp_path / n for n in ("model.gguf", "binary", "source.cpp")]
    for path in files:
        path.write_bytes(b"original")
    expected = {str(p): b.sha(p) for p in files}
    assert b.verify_files(expected) == expected
    for path in files:
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="integrity drift"):
            b.verify_files(expected)
        path.write_bytes(b"original")


@pytest.mark.parametrize("bad", [[], [{"name": "q", "type": 1}],
                                  [{"name": "q", "type": 0}, {"name": "q", "type": 0}]])
def test_non_f32_or_invalid_native_inventory_rejected(bad):
    with pytest.raises(ValueError):
        b.validate_native_metadata({"vocab": 64, "max_len": 128, "tensor_inventory": bad}, 64)


def test_native_inventory_digest_and_vocab_are_frozen():
    ready = {"vocab": 64, "max_len": 128,
             "tensor_inventory": [{"name": "w", "type": 0, "shape": [4, 4]}]}
    meta = b.validate_native_metadata(ready, 64)
    assert meta["tensor_count"] == 1 and meta["tensor_type"] == 0
    with pytest.raises(ValueError, match="vocabulary"):
        b.validate_native_metadata(ready, 65)
    altered = copy.deepcopy(ready)
    altered["tensor_inventory"][0]["shape"] = [2, 8]
    assert b.validate_native_metadata(altered, 64) != meta


def completed_fixture(out):
    protocol = {"controller_sha256": b.sha(b.__file__), "lengths": [2], "steps": 2,
                "cycles": 1, "sequence": ["ember", "hf", "hf", "ember"],
                "model_metadata": {"vocab": 3}, "integrity_files": {"binary": "frozen"}}
    b.atomic_json(out / "protocol.json", protocol)
    b.atomic_json(out / "inputs_2.json", {"prompt_ids": [2, 0], "continuation_ids": [1, 1]})
    rows = np.array([[1, 2, 0], [1, 2, 0]], dtype="<f4")
    rows.tofile(out / "hf_2.f32")
    rows.tofile(out / "ember_2.f32")
    gate = b.assert_parity(rows, rows, [1, 1], [1, 1])
    gate.update(inputs_sha256=b.sha(out / "inputs_2.json"),
                hf_rows_sha256=b.sha(out / "hf_2.f32"),
                ember_rows_sha256=b.sha(out / "ember_2.f32"))
    b.atomic_json(out / "gate_2.json", gate)
    previous = b.sha(out / "protocol.json")
    records = []
    for key, length, engine in b.trial_plan(protocol):
        row = {"key": key, "length": length, "engine": engine, "argmax": [1, 1],
               "forward_ms": [1., 1.], "inputs_sha256": gate["inputs_sha256"],
               "previous_sha256": previous}
        row["record_sha256"] = b.digest_record(row)
        previous = row["record_sha256"]
        records.append(row)
    (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    b.atomic_json(out / "integrity.json", {"protocol_sha256": b.sha(out / "protocol.json"),
        "journal_sha256": b.sha(out / "trials.jsonl"), "files": protocol["integrity_files"],
        "inputs": {"2": gate["inputs_sha256"]}, "status": "FINAL_INTEGRITY_VERIFIED"})


def test_offline_completion_requires_final_integrity(tmp_path):
    completed_fixture(tmp_path)
    assert b.analyze(tmp_path)["status"] == "COMPLETE_PARITY_GATED_CUDA_MEASUREMENT"
    integrity = tmp_path / "integrity.json"
    saved = integrity.read_bytes()
    integrity.unlink()
    with pytest.raises(FileNotFoundError):
        b.analyze(tmp_path)
    integrity.write_bytes(saved)
    data = json.loads(integrity.read_text())
    data["files"]["binary"] = "wrong"
    b.atomic_json(integrity, data)
    with pytest.raises(ValueError, match="final integrity"):
        b.analyze(tmp_path)


def test_offline_formula_drift_fails_even_with_updated_gate_hash(tmp_path):
    completed_fixture(tmp_path)
    inputs = tmp_path / "inputs_2.json"
    data = json.loads(inputs.read_text())
    data["prompt_ids"] = [0]
    b.atomic_json(inputs, data)
    gate_path = tmp_path / "gate_2.json"
    gate = json.loads(gate_path.read_text())
    gate["inputs_sha256"] = b.sha(inputs)
    b.atomic_json(gate_path, gate)
    with pytest.raises(ValueError, match="prompt length/formula"):
        b.analyze(tmp_path)


def test_existing_halted_diagnosis_is_not_overwritten(tmp_path, monkeypatch):
    import sys
    marker = tmp_path / "HALTED"
    marker.write_text("original gate failure\n")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--gguf", "dummy", "--binary", "dummy", "--out", str(tmp_path)])
    def refuse(args):
        raise ValueError("preserved HALTED receipt; no automatic retries")
    monkeypatch.setattr(b, "run", refuse)
    with pytest.raises(ValueError, match="preserved HALTED"):
        b.main()
    assert marker.read_text() == "original gate failure\n"
