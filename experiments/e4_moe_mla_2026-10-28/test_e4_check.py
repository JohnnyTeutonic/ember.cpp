import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_check import FAMILIES, GPU, check


def populate(root):
    for family in FAMILIES:
        gpu = family in GPU
        row = {"cpu_greedy_equal_hf": True}
        r = {"cases": 40, "cpu_greedy_vs_hf": "40/40", "cpu_worst_first_rel_hf": 1e-6,
             "device_supported": gpu, "on_gpu": gpu, "vocab": 2, "rows_dumped": 640 if gpu else 0}
        if gpu:
            row.update(dev_greedy_equal_hf=True, dev_greedy_equal_cpu=True, dev_step_top1_cpu=16)
            r.update(dev_greedy_vs_hf="40/40", dev_greedy_vs_cpu="40/40", dev_step_top1_cpu="640/640",
                     dev_worst_first_rel_hf=1e-6, dev_worst_first_rel_cpu=1e-6, dev_worst_step_rel_cpu=1e-6)
        r["rows"] = [dict(copy.deepcopy(row), case=i) for i in range(40)]
        for mode in ("default", "det1", "det2"):
            (root / f"{family}_{mode}.json").write_text(json.dumps(r))
            if gpu:
                (root / f"{family}_{mode}.bin").write_bytes(bytes(640 * 2 * 4))
    (root / "suite_v1.6.1_cuda_build.json").write_text(json.dumps(
        {"counts": {"PASS": 554, "FAIL": 0, "SKIP": 44, "XFAIL": 0, "XPASS": 0}}))


def test_complete_receipts(tmp_path):
    populate(tmp_path)
    assert check(tmp_path)["pass"]


@pytest.mark.parametrize("field,value", [("cases", 0), ("on_gpu", False),
    ("device_supported", False), ("dev_greedy_vs_hf", "39/40"),
    ("dev_step_top1_cpu", "639/640"), ("dev_worst_first_rel_hf", float("nan")),
    ("dev_worst_step_rel_cpu", .01)])
def test_bad_device_report(tmp_path, field, value):
    populate(tmp_path)
    path = tmp_path / "mixtral_default.json"
    r = json.loads(path.read_text())
    r[field] = value
    path.write_text(json.dumps(r))
    assert not check(tmp_path)["pass"]


def test_missing_and_truncated_evidence(tmp_path):
    populate(tmp_path)
    (tmp_path / "phi3_longrope_default.json").unlink()
    (tmp_path / "mixtral_det1.bin").write_bytes(b"")
    assert len(check(tmp_path)["errors"]) >= 2


def test_nondeterministic_rows(tmp_path):
    populate(tmp_path)
    path = tmp_path / "mixtral_det2.bin"
    path.write_bytes(b"x" + path.read_bytes()[1:])
    assert not check(tmp_path)["pass"]


def test_suite_failure(tmp_path):
    populate(tmp_path)
    (tmp_path / "suite_v1.6.1_cuda_build.json").write_text(json.dumps(
        {"counts": {"PASS": 553, "FAIL": 1, "SKIP": 44, "XFAIL": 0, "XPASS": 0}}))
    assert not check(tmp_path)["pass"]
