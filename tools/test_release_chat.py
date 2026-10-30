"""tools/release_chat.py without servers: the parity comparison, the Q8 criterion, the CPU lock
(waits for another holder, removes only its own line), resumable receipts, and publish refusing
without confirmation or with unpassed stages."""
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_chat as R  # noqa: E402


def test_compare_needs_reply_tokens_and_stop():
    rec = [{"ember": "a", "mtstudio": "a", "ember_tokens": 2, "mts_tokens": 2, "ember_stop": "eos", "mts_stop": "eos"}]
    assert R.compare(rec)["pass"]
    assert not R.compare([{**rec[0], "ember_stop": "length"}])["pass"]
    assert not R.compare([])["pass"]


def test_q8_check(tmp_path):
    f, q = tmp_path / "f.txt", tmp_path / "q.txt"
    f.write_text("0 10 1\n5 0 2\n")
    q.write_text("0 10.05 1\n5 0.05 2\n")
    r = R.q8_check(str(f), str(q))
    assert r["pass"] and r["top1"] == "2/2" and abs(r["max_over_range"] - 0.005) < 1e-9
    q.write_text("0 10 1\n0 5 2\n")
    assert not R.q8_check(str(f), str(q))["pass"]


def test_lock_waits_and_removes_only_its_own(tmp_path):
    lock = tmp_path / ".cpu_lock"
    lock.write_text("genetics: stage 3\n")
    got = []

    def take():
        with R.CpuLock("t", path=str(lock), wait_s=10, poll_s=0.05):
            got.append(lock.read_text())
    th = threading.Thread(target=take)
    th.start()
    time.sleep(0.2)
    assert not got                       # still waiting on the other session's lock
    lock.unlink()
    th.join(5)
    assert got and got[0].startswith("ember release_chat.py: t") and not lock.exists()
    with R.CpuLock("t", path=str(lock), wait_s=1, poll_s=0.01):
        lock.write_text("someone else\n")          # the file no longer holds our line
    assert lock.read_text() == "someone else\n"    # so it is left alone


def test_lock_times_out(tmp_path):
    lock = tmp_path / ".cpu_lock"
    lock.write_text("other\n")
    with pytest.raises(SystemExit):
        with R.CpuLock("t", path=str(lock), wait_s=0.1, poll_s=0.05):
            pass


def test_receipt_resumes(tmp_path):
    rc = R.Receipt(str(tmp_path))
    rc.put("export", True, files={})
    assert R.Receipt(str(tmp_path)).done("export") and not R.Receipt(str(tmp_path)).done("gate")


def test_publish_refuses(tmp_path):
    out = str(tmp_path)
    R.Receipt(out).put("export", True)
    args = R.main.__globals__["argparse"].Namespace(out=out, env_file="nope", confirm="")
    with pytest.raises(SystemExit, match="--confirm"):
        R.publish(args)
    args.confirm = "ember-chat"
    with pytest.raises(SystemExit, match="not passed"):
        R.publish(args)
    assert json.load(open(os.path.join(out, "release.json")))["stages"].get("publish") is None


@pytest.mark.parametrize("kind,compressed", [("f16", 220919008), ("q8", 117699712)])
def test_wasm_heap_uses_expanded_weights(kind, compressed):
    # This F16 release actually failed at 1 GB and passed at 2 GB.
    files = {"f32": {"bytes": 441119872}, kind: {"bytes": compressed}}
    assert R.select_wasm_heap(files, kind) == ("2gb", 441119872)


def test_wasm_heap_requires_uncompressed_reference():
    with pytest.raises(ValueError, match="F32"):
        R.select_wasm_heap({"q8": {"bytes": 100000000}}, "q8")
    assert R.select_wasm_heap({"f32": {"bytes": 58000000}}, "f32") == ("1gb", 58000000)


def test_wasm_stage_build_uses_expanded_heap(tmp_path, monkeypatch):
    from argparse import Namespace
    from subprocess import CompletedProcess

    rc = R.Receipt(str(tmp_path))
    rc.put("export", True, files={"f32": {"bytes": 441119872},
                                 "q8": {"bytes": 117699712, "path": "model.gguf"}})
    calls = []
    monkeypatch.setenv("CPU_BUDGET_CORES", "1")

    def fail_build(command, env):
        calls.append(env)
        return CompletedProcess(command, 1)

    monkeypatch.setattr(R, "run", fail_build)
    R.stage_wasm(Namespace(out=str(tmp_path), space_model="q8", json_inc="json", emsdk="emsdk"), rc)
    assert calls[0]["EMBER_WASM_MAX_MEMORY"] == "2gb"
    assert rc.data["stages"]["wasm"]["sizing_bytes"] == 441119872
    assert not rc.done("wasm")
