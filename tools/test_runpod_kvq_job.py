"""Remote wrapper fixtures; no GPU or external process launched."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("kvq_job", Path(__file__).with_name("runpod_kvq_job.py"))
J = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(J)


def stage(tmp_path):
    upload = tmp_path / "chat/upload"
    upload.mkdir(parents=True)
    manifest = {"formats": ["f16"], "deadline_minutes": 95, "baseline_ember_commit": "frozen",
                "phases": ["g1", "g2", "g3", "tfd_qwen3", "tfd_qwen25"], "files": {}}
    for name in J.FILES:
        p = upload / name
        p.write_text("frozen" if name == "HEAD" else "staged")
        manifest["files"][name] = {"bytes": p.stat().st_size, "sha256": J.sha(p)}
    (upload / "STAGING.json").write_text(json.dumps(manifest))
    return upload


@pytest.mark.parametrize("returncode,marker,reported,expected", [(0, True, True, True),
    (124, False, True, False), (0, False, True, False), (0, True, False, False)])
def test_wrapper_cannot_convert_incomplete_or_failed_gate_to_pass(tmp_path, monkeypatch, returncode,
                                                                 marker, reported, expected):
    stage(tmp_path)

    def remote(argv, **kwargs):
        assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=15s", "95m"]
        assert kwargs["env"]["KVQ_FORMATS"] == "f16"
        (tmp_path / "out/gate_summary.json").write_text(json.dumps({"pass": reported, "errors": []}))
        if marker:
            (tmp_path / "GATE_DONE").touch()
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(J.subprocess, "run", remote)
    out = tmp_path / "job"
    assert J.run(out, content=tmp_path) == 0
    assert json.loads((out / "gate_summary.json").read_text())["pass"] is expected
    assert json.loads((out / "execution.json").read_text())["gate_pass"] is expected
    assert (out / "out.zip").is_file() and (out / "vm.log").is_file()


def test_tampered_stage_is_rejected_before_launch(tmp_path, monkeypatch):
    upload = stage(tmp_path)
    (upload / "ember.tgz").write_text("altered")
    monkeypatch.setattr(J.subprocess, "run", lambda *a, **k: pytest.fail("must not launch invalid input"))
    out = tmp_path / "job"
    J.run(out, content=tmp_path)
    report = json.loads((out / "gate_summary.json").read_text())
    assert report["pass"] is False and "hash mismatch" in report["errors"][0]
