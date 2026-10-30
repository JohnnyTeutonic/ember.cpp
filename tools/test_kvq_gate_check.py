"""Offline tests of the E9 fail-closed receipt validator."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("kvq_gate", TOOLS / "kvq_gate_check.py")
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_dump_rejects_truncation_and_empty_inventory(tmp_path):
    p = tmp_path / "rows.bin"
    dump_json(Path(str(p) + ".json"), {"prompts": 1, "vocab": 4})
    p.write_bytes(bytes(68))  # two prefill + two decode rows, one generated id
    assert G.check_dump(p, n=1, length=2, gen=1)["bytes"] == 68
    p.write_bytes(bytes(64))
    with pytest.raises(ValueError, match="truncated"):
        G.check_dump(p, n=1, length=2, gen=1)
    with pytest.raises(ValueError, match="empty"):
        G.check_dump(p, n=0, length=2, gen=1)


def valid_device():
    return {"pass": True, "on_gpu": True, "device_supported": True, "prompts": 12,
            "len": 24, "gen": 32, "tol": 1e-4, "worst_prefill_rel": 1e-5,
            "worst_decode_rel": 2e-5, "top1_agree": "288/288", "greedy_identical": "12/12",
            "rows": [{"i": i, "top1": 24, "greedy_first_diff": -1} for i in range(12)]}


@pytest.mark.parametrize("key,value", [("on_gpu", False), ("pass", False), ("prompts", 0),
                                       ("worst_decode_rel", 1.01e-4), ("tol", .01),
                                       ("worst_prefill_rel", float("nan"))])
def test_device_gate_rejects_incomplete_or_looser_evidence(key, value):
    record = valid_device()
    record[key] = value
    with pytest.raises(ValueError):
        G.check_device(record)


def test_device_gate_checks_individual_rows():
    record = valid_device()
    G.check_device(record)
    record["rows"][1]["i"] = 0
    with pytest.raises(ValueError, match="duplicated"):
        G.check_device(record)


def teacher_rows():
    rows = []
    for name in ("book", "chat"):
        contexts = {str(c): {"all": {"positions": c, "max_rel": .003, "mean_rel": .00003,
                                    "ppl_rel_diff": .0001}} for c in (2048, 4096, 8192)}
        rows.append({"name": name, "positions": 8192, "formats": {"f16": {"contexts": contexts}}})
    return rows


def test_teacher_forcing_is_complete_and_compared_without_relaxing_threshold():
    ref = teacher_rows()
    assert all(r["max_rel"]["within_prefill_observed"] for r in G.compare_teacher(ref, ref))
    measured = copy.deepcopy(ref)
    measured[0]["formats"]["f16"]["contexts"]["2048"]["all"]["max_rel"] = .0030001
    assert G.compare_teacher(measured, ref)[0]["max_rel"]["within_prefill_observed"] is False
    measured[0]["positions"] = 10
    with pytest.raises(ValueError, match="incomplete"):
        G.compare_teacher(measured, ref)


def valid_summary(tmp_path):
    reference = TOOLS.parent / "experiments/kv_quant_r2_2026-10-28/receipts/colab_l4_eval"
    for name in G.EXPECTED:
        if name.endswith(".json"):
            shutil.copyfile(reference / name, tmp_path / name)
    (tmp_path / "models.sha256").write_text("".join(f"{sha}  /frozen/{name}\n" for name, sha in G.EXPECTED.items()))
    phases, formats = ["g1", "g2", "g3"], ["f16"]
    labels = G.gate_labels(phases, formats)
    assert len(labels) == 14
    (tmp_path / "gates.txt").write_text("".join(f"{label} IDENTICAL {'a'*16} {'a'*16}\n" for label in labels))
    for phase, count in (("g1", 18), ("g2", 8)):
        for i in range(count):
            dump_json(tmp_path / phase / f"{i}.check.json", {"pass": True, "bytes": 68})
    for model in ("qwen3", "qwen25"):
        for fmt in ("f32", "f16"):
            dump_json(tmp_path / "g3" / f"{model}_{fmt}.json", valid_device())
    return phases, formats


def test_summary_rejects_missing_comparison_and_integrity_receipts(tmp_path):
    phases, formats = valid_summary(tmp_path)
    assert G.summary(tmp_path, phases, formats)["pass"] is True
    (tmp_path / "g1/0.check.json").unlink()
    assert G.summary(tmp_path, phases, formats)["pass"] is False


def test_summary_rejects_empty_or_changed_inputs(tmp_path):
    phases, formats = valid_summary(tmp_path)
    (tmp_path / "long_qwen3.json").write_text("[]")
    assert G.summary(tmp_path, phases, formats)["pass"] is False


def test_summary_rejects_different_byte_results(tmp_path):
    phases, formats = valid_summary(tmp_path)
    p = tmp_path / "gates.txt"
    p.write_text(p.read_text().replace("IDENTICAL", "DIFFER", 1))
    assert G.summary(tmp_path, phases, formats)["pass"] is False
