"""Numerical/inventory and immutable timing-record gates; no model or GPU."""

import copy
import json
from pathlib import Path
import importlib.util
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("hf_cuda_benchmark_test", Path(__file__).with_name("benchmark.py"))
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
        def float(self): return self
        def cpu(self): operations.append("D2H"); return self
        def argmax(self): operations.append("argmax"); return 1
    class Logits:
        shape = (1, 1, 2)
        def __getitem__(self, key): return Row()
    class Tensor:
        def to(self, device): operations.append("H2D"); return self
    def tensor(ids, dtype): operations.append("CPU input"); return Tensor()
    def forward(*args, **kwargs):
        operations.append("forward")
        return SimpleNamespace(logits=Logits(),past_key_values=object())
    fake_torch = SimpleNamespace(inference_mode=nullcontext,tensor=tensor,long="int64",cuda=SimpleNamespace(synchronize=lambda device:operations.append("sync")))
    hf = b.HF.__new__(b.HF);hf.torch=fake_torch;hf.device="cuda:0";hf.model=forward
    def clock(): operations.append("timer"); return len(operations)*1000000
    monkeypatch.setattr(b.time,"perf_counter_ns",clock)
    result=hf.run([1,2],2,continuation=[1,1])
    assert result["argmax"] == [1,1]
    assert operations == ["CPU input","sync","timer","H2D","forward","D2H","sync","timer","argmax"]*2
