"""Tests of drive.py against a mock colab CLI (no network, no Colab). python3 -m unittest test_drive -v"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
GATES = ["analyze", "benchmark", "build", "checkout", "contract", "environment", "inputs", "packages", "provenance",
         "regenerate", "weights"]

MOCK = r'''#!/usr/bin/env python3
import json, os, re, sys, tarfile, time
from pathlib import Path
state_p = Path(os.environ["MOCK_STATE"]); content = Path(os.environ["MOCK_CONTENT"]); mode = os.environ["MOCK_MODE"]
st = json.loads(state_p.read_text()) if state_p.exists() else {"sessions": {}}
save = lambda: state_p.write_text(json.dumps(st))
args = sys.argv[1:]
cmd = args[0]
if cmd == "sessions":
    if mode == "sessions_fail" and st.get("created"):
        print("error: auth", file=sys.stderr); sys.exit(1)
    print("[colab] A new version of Colab CLI is available: 0.7.5 (current: 0.6.0)")
    if not st["sessions"]: print("[colab] No active sessions found on server.")
    for n, ep in st["sessions"].items(): print(f"[{n}] {ep} L4")
    sys.exit(0)
name = args[args.index("-s") + 1]
if cmd == "new":
    if mode == "orphan_new":
        st["sessions"]["?"] = "ep-orphan"; st["created"] = True; save(); sys.exit(1)
    st["sessions"][name] = "ep-" + name; st["created"] = True; save()
    if mode == "new_timeout": time.sleep(60)
    sys.exit(0)
if cmd == "stop":
    if mode != "stop_ignored": st["sessions"].pop(name, None); save()
    sys.exit(0)
if cmd == "exec":
    f = args[args.index("-f") + 1]
    m = re.match(r".*claude_scan_(.+)_(launch|poll|pack\d|mkdir|stop_workload)\.py$", f); run, label = m.group(1), m.group(2)
    if label == "mkdir":
        print("MADE"); sys.exit(0)
    if label == "stop_workload":
        st["stopped"] = True; save(); print("STOPPED"); sys.exit(0)
    if label.startswith("pack"): label = "pack"
    out = content / f"run_{run}" / "out"
    if label == "launch":
        out.mkdir(parents=True, exist_ok=True)
        gates = {g: 0 for g in os.environ["MOCK_GATES"].split()}
        if mode == "gate_fail": gates["strict_2"] = 1
        commit = os.environ["MOCK_COMMIT"] if mode != "wrong_commit" else "f" * 40
        s = {"run_id": run, "commit": commit, "head": commit, "exit_status": 0,
             "pass": mode != "gate_fail", "gates": gates}
        (out / "status.json").write_text(json.dumps(s))
        if mode != "never_done": (out / "DONE").write_text(run)
        print("LAUNCHED")
    elif label == "poll":
        print("DONE" if (out / "DONE").exists() else "RUNNING")
    else:
        with tarfile.open(content / f"scan_{run}.tgz", "w:gz") as t: t.add(out, arcname=f"run_{run}/out")
        print("PACKED")
    sys.exit(0)
if cmd == "upload":
    if mode == "upload_fail": sys.exit(1)
    st.setdefault("uploads", []).append(args[-1]); save(); sys.exit(0)
if cmd == "download":
    src, dst = args[-2], args[-1]
    Path(dst).write_bytes((content / Path(src).name).read_bytes()); sys.exit(0)
sys.exit(2)
'''


class DriveTests(unittest.TestCase):
    def run_drive(self, mode, run="t1", extra=None):
        tmp = Path(tempfile.mkdtemp())
        mock = tmp / "colab"
        mock.write_text(MOCK)
        mock.chmod(0o755)
        (tmp / "content").mkdir()
        (tmp / "upload").mkdir()
        for n in ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
                  "added_tokens.json", "generation_config.json", "converter.tgz"):
            (tmp / "upload" / n).write_text(n)
        env = dict(os.environ, DRIVE_TEST_HARNESS="1", DRIVE_TEST_COLAB=str(mock), DRIVE_TEST_CLI_PY=sys.executable,
                   DRIVE_TEST_REAP="true", DRIVE_TEST_ADOPT="true", DRIVE_TEST_ARTIFACTS=str(tmp / "art"),
                   DRIVE_TEST_RECEIPTS=str(tmp / "receipts"), DRIVE_TEST_UPLOAD_DIR=str(tmp / "upload"), DRIVE_TEST_MINUTE_SECONDS="0.3",
                   MOCK_STATE=str(tmp / "state.json"), MOCK_CONTENT=str(tmp / "content"), MOCK_MODE=mode,
                   MOCK_GATES=" ".join(GATES), MOCK_COMMIT=COMMIT, **(extra or {}))
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, run], env=env, capture_output=True,
                           text=True, timeout=120)
        state = json.loads((tmp / "state.json").read_text()) if (tmp / "state.json").exists() else {"sessions": {}}
        log = (tmp / "art" / run / "driver.log").read_text() if (tmp / "art" / run / "driver.log").exists() else ""
        return r.returncode, state, log, tmp, time.monotonic() - t0

    def test_normal_run_validates_and_stops(self):
        rc, state, log, tmp, _ = self.run_drive("normal")
        self.assertEqual(rc, 0, log)
        self.assertEqual(state["sessions"], {})
        self.assertIn("CLEANUP OK", log)
        self.assertIn("validation: PASS", log)
        self.assertTrue((tmp / "receipts" / "t1" / "status.json").is_file())
        self.assertTrue((tmp / "receipts" / "t1" / "driver.log").is_file())

    def test_failed_gate_is_exit_1_with_receipts(self):
        rc, state, log, tmp, _ = self.run_drive("gate_fail")
        self.assertEqual(rc, 1, log)
        self.assertEqual(state["sessions"], {})
        self.assertTrue((tmp / "receipts" / "t1" / "status.json").is_file())

    def test_wrong_commit_fails_validation(self):
        rc, _, log, _, _ = self.run_drive("wrong_commit")
        self.assertEqual(rc, 2, log)
        self.assertIn("identity mismatch", log)

    def test_unnamed_allocation_is_never_certified(self):
        rc, state, log, _, _ = self.run_drive("orphan_new")
        self.assertEqual(rc, 3, log)
        self.assertIn("CLEANUP UNRESOLVED", log)
        self.assertNotIn("CLEANUP OK", log)
        self.assertEqual(state["sessions"], {"?": "ep-orphan"})  # not ours by name or endpoint: not stopped

    def test_log_failure_does_not_stop_the_shutdown(self):
        rc, state, log, _, _ = self.run_drive("normal", extra={"DRIVE_TEST_LOG_FAIL_AFTER": "4"})
        self.assertEqual(state["sessions"], {})
        self.assertNotEqual(rc, 3)

    def test_cleanup_command_start_failure_is_retried(self):
        rc, state, log, _, _ = self.run_drive("normal", extra={"DRIVE_TEST_RAISE_ON": "stop"})
        self.assertEqual(state["sessions"], {}, log)
        self.assertIn("could not run", log)
        self.assertIn("CLEANUP OK", log)
        self.assertEqual(rc, 0, log)

    def test_test_hooks_refused_without_harness(self):
        env = dict(os.environ, DRIVE_TEST_COLAB="/bin/true")
        env.pop("DRIVE_TEST_HARNESS", None)
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), COMMIT, "t9"], env=env, capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing", r.stderr)

    def test_listing_failure_is_never_reported_as_clean(self):
        rc, state, log, _, _ = self.run_drive("sessions_fail")
        self.assertEqual(rc, 3, log)
        self.assertIn("CLEANUP UNRESOLVED", log)
        self.assertNotIn("CLEANUP OK", log)

    def test_ignored_stop_is_unresolved(self):
        rc, state, log, _, _ = self.run_drive("stop_ignored")
        self.assertEqual(rc, 3, log)
        self.assertIn("t1", json.dumps(state["sessions"]))
        self.assertIn("CLEANUP UNRESOLVED", log)

    def test_creation_timeout_still_stops_the_allocated_session(self):
        rc, state, log, _, elapsed = self.run_drive("new_timeout")
        self.assertEqual(state["sessions"], {}, log)  # the VM the timed-out call allocated was found and stopped
        self.assertIn("CLEANUP OK", log)
        self.assertIn(rc, (0, 2), log)
        self.assertLess(elapsed, 0.3 * 90 + 30)  # bounded by the shutdown deadline (plus inventory slack)

    def test_all_seven_files_are_uploaded(self):
        rc, state, log, _, _ = self.run_drive("normal")
        self.assertEqual(rc, 0, log)
        self.assertEqual(len(state.get("uploads", [])), 7)
        self.assertTrue(all("/content/upload_t1/" in u for u in state["uploads"]))

    def test_failed_upload_stops_before_launch_and_cleans_up(self):
        rc, state, log, _, _ = self.run_drive("upload_fail")
        self.assertIn("launched False", log)
        self.assertEqual(state["sessions"], {})
        self.assertIn("CLEANUP OK", log)
        self.assertEqual(rc, 2, log)

    def test_deadline_without_done_stops_the_workload_then_retrieves(self):
        rc, state, log, tmp, _ = self.run_drive("never_done")
        self.assertTrue(state.get("stopped"), log)
        self.assertIn("workload stop: ok", log)
        self.assertTrue((tmp / "receipts" / "t1" / "status.json").is_file())  # evidence came home
        self.assertEqual(state["sessions"], {})
        self.assertEqual(rc, 2, log)  # no DONE seen: never a validated success

    def test_bad_arguments_are_refused(self):
        r = subprocess.run([sys.executable, str(HERE / "drive.py"), "main", "x"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
