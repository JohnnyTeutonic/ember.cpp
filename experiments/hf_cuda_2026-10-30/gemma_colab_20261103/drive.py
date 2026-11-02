#!/usr/bin/env python3
"""Local driver for the Gemma CUDA comparison (AMENDMENT.md; microtorch/AGENTS.md path C). Derived from the reviewed
selective-scan driver (paperkiln experiments/mamba_parallel_scan_2026-11-03/drive.py): same deadlines, cleanup,
receipts and exit codes, plus an upload of the Google tokenizer/config files and converter.tgz before launch.
Run in a foreground WSL process from this ember checkout's root:

    python3 experiments/hf_cuda_2026-10-30/gemma_colab_20261103/drive.py FULL_COMMIT_SHA RUN_ID

Session name claude-gemma-RUN_ID. Outputs: /mnt/c/ml_artifacts/benchmarks/hf_cuda_gemma_colab_20261103/RUN_ID/
(driver.log, scan_out.tgz, run_RUN_ID/out/); the top-level files of out/ are copied to this folder's
receipts/RUN_ID/, with archive.sha256 hashing every retrieved file (out/binaries/, out/inputs/ included).

Deadlines from the driver's start: polling ends at 60 minutes, retrieval at 75, shutdown attempts at 85, the
final inventory at 90. Every external call has a timeout capped by the deadline that governs it; a call past its
deadline is not started; a timed-out child is killed.

Shutdown is certified only by an authoritative session listing that shows neither our session name, nor the
endpoint we owned, nor any endpoint that was not there before our creation call (a VM whose name the CLI lost
shows as [?]; it is never stopped unless it is ours by endpoint, but it does prevent certification).
Exit status:
  0  run validated: status.json matches run, commit and HEAD, exactly the required gates, all 0, VM exit 0
  1  the experiment failed (receipts preserved)
  2  transport, receipt or validation could not be completed
  3  cleanup unresolved: a VM may still be running (the log names it and how to stop it)
Test hooks (DRIVE_TEST_*) are honoured only with DRIVE_TEST_HARNESS=1, and refused otherwise.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TEST = os.environ.get("DRIVE_TEST_HARNESS") == "1"


def hook(name, default):
    return os.environ.get(name) if TEST and os.environ.get(name) else default


COL = hook("DRIVE_TEST_COLAB", os.path.expanduser("~/.local/bin/colab"))
CLI_PY = hook("DRIVE_TEST_CLI_PY", os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python"))
REAP = hook("DRIVE_TEST_REAP", f"{sys.executable} tools/colab_reap_orphans.py").split()
ADOPT = hook("DRIVE_TEST_ADOPT", f"{CLI_PY} tools/colab_adopt.py").split()
HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(hook("DRIVE_TEST_ARTIFACTS", "/mnt/c/ml_artifacts/benchmarks/hf_cuda_gemma_colab_20261103"))
UPLOAD_SOURCES = {name: "/mnt/c/ml_artifacts/thirdparty/gemma-3-270m-it/" + name for name in (
    "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "added_tokens.json",
    "generation_config.json")}
UPLOAD_SOURCES["converter.tgz"] = "/home/jonat/hf_cuda_smol_stage_20261030/converter.tgz"
UPLOAD_SOURCES = {k: hook("DRIVE_TEST_UPLOAD_DIR", None) and str(Path(hook("DRIVE_TEST_UPLOAD_DIR", "")) / k) or v
                  for k, v in UPLOAD_SOURCES.items()}
RECEIPTS = Path(hook("DRIVE_TEST_RECEIPTS", str(HERE / "receipts")))
MINUTE = float(hook("DRIVE_TEST_MINUTE_SECONDS", "60"))
LOG_FAIL_AFTER = int(hook("DRIVE_TEST_LOG_FAIL_AFTER", "0"))   # test: log writes fail after N lines
RAISE_ON = hook("DRIVE_TEST_RAISE_ON", "")                      # test: first call whose argv has this word raises
REQUIRED_GATES = {"analyze", "benchmark", "build", "checkout", "contract", "environment", "inputs", "packages",
                  "provenance", "regenerate", "toolkit", "weights"}


class Stop(Exception):
    pass


class Driver:
    def __init__(self, commit, run):
        self.commit, self.run = commit, run
        self.name = f"claude-gemma-{run}"
        self.out = ARTIFACTS / run
        self.receipts = RECEIPTS / run
        t = time.monotonic()
        self.work_end, self.retrieve_end, self.shutdown_end, self.inventory_end = (
            t + m * MINUTE for m in (60, 75, 85, 90))
        self.intent = False      # a creation call was made: cleanup must reconcile it
        self.endpoint = None     # the endpoint we own, once established
        self.before = None       # endpoints listed before the creation call
        self.cancel = False
        self.log_file = None
        self.log_lines = 0
        self.raised = False
        self.log_failed = False
        self.done = False
        self.stop_confirmed = None  # None: no stop was needed or no launch was attempted
        self.launch_attempted = False

    # ---- logging never raises: the file is the record, the terminal and stderr best effort
    def log(self, *parts):
        line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
        try:
            self.log_lines += 1
            if LOG_FAIL_AFTER and self.log_lines > LOG_FAIL_AFTER:
                raise OSError("test: log write failed")
            self.log_file.write(line + "\n")
            self.log_file.flush()
        except (OSError, ValueError, AttributeError):
            self.log_failed = True
            try:
                sys.stderr.write("[log file unavailable] " + line + "\n")
            except (OSError, ValueError):
                pass
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass

    def check(self):
        if self.cancel:
            raise Stop("cancelled by signal")

    def call(self, argv, want, deadline, label):
        """Run argv, timeout min(want, time to deadline). (rc, output), or (None, '') if not started, timed out
        (the child is killed) or failed to start. Never raises."""
        left = deadline - time.monotonic()
        if left <= 1:
            self.log(f"{label}: not started, deadline passed")
            return None, ""
        try:
            if RAISE_ON and RAISE_ON in argv and not self.raised:
                self.raised = True
                raise OSError("test: command failed to start")
            r = subprocess.run(argv, capture_output=True, text=True, timeout=min(want, left))
        except subprocess.TimeoutExpired:
            self.log(f"{label}: timed out, killed")
            return None, ""
        except OSError as e:
            self.log(f"{label}: could not run: {e}")
            return None, ""
        self.log(f"{label}: exit {r.returncode}")
        return r.returncode, r.stdout + r.stderr

    def sleep(self, seconds, deadline):
        end = min(time.monotonic() + seconds, deadline)
        while time.monotonic() < end and not self.cancel:
            time.sleep(min(1.0, max(0.0, end - time.monotonic())))

    def sessions(self, deadline):
        """Rows [(name, endpoint)] from an authoritative listing, or None if the listing failed or has a row
        that does not parse (fail closed). Names may repeat (several [?] orphans)."""
        rc, out = self.call([COL, "sessions"], 120, deadline, "sessions")
        if rc != 0:
            return None
        rows = []
        for line in out.splitlines():
            if not line.startswith("["):
                continue
            m = re.match(r"^\[([^\]]+)\]\s+(\S+)", line)
            if not m:
                self.log("unparsed session row:", line)
                return None
            if m.group(1) == "colab":  # the CLI's own banner and status lines, not sessions
                continue
            rows.append((m.group(1), m.group(2)))
        return rows

    # ---- phases
    def check_local_inputs(self):
        """The seven upload sources against the committed inputs.sha256, before any paid session exists."""
        import hashlib
        want = {}
        for line in (HERE / "inputs.sha256").read_text().splitlines():
            digest, name = line.split()
            want[name] = digest
        if set(want) != set(UPLOAD_SOURCES):
            raise Stop(f"upload set {sorted(UPLOAD_SOURCES)} differs from inputs.sha256 {sorted(want)}")
        for name, src in UPLOAD_SOURCES.items():
            try:
                h = hashlib.sha256(Path(src).read_bytes()).hexdigest()
            except OSError as e:
                raise Stop(f"preflight: cannot read {src}: {e}")
            if h != want[name] and not TEST:
                raise Stop(f"preflight: {src} is not the committed {name} ({h} != {want[name]})")
        self.log("local inputs match inputs.sha256" + (" (hash check skipped: test harness)" if TEST else ""))

    def preflight(self):
        self.check_local_inputs()
        rows = self.sessions(self.work_end)
        if rows is None:
            raise Stop("preflight: session listing failed; not creating anything")
        self.log("preflight sessions:", json.dumps(rows))
        if any(n == self.name for n, _ in rows):
            raise Stop(f"preflight: a session named {self.name} exists; not ours to touch")
        for argv, label in ((REAP, "orphans"), (ADOPT + ["--list"], "adopt --list")):
            rc, out = self.call(argv, 300, self.work_end, label)
            self.log(f"{label}:\n" + out.strip())
            if rc != 0:
                raise Stop(f"preflight: {label} failed")
        self.before = {ep for _, ep in rows}

    def create(self):
        self.intent = True
        self.call([COL, "new", "-s", self.name, "--gpu", "L4"], 600, self.work_end, "new")
        for _ in range(3):  # reconcile whatever the creation call returned
            rows = self.sessions(self.work_end)
            if rows is not None:
                mine = [ep for n, ep in rows if n == self.name]
                if len(mine) == 1:
                    self.endpoint = mine[0]
                    self.log("owned session", self.name, "endpoint", self.endpoint)
                    return
                new = sorted({ep for _, ep in rows} - self.before)
                if new:
                    self.log("endpoints new since creation, not under our name:", new)
            self.sleep(MINUTE / 3, self.work_end)
            self.check()
        raise Stop("creation not confirmed under our name")

    def exec_py(self, code, want, deadline, label):
        p = Path(f"/tmp/claude_scan_{self.run}_{label}.py")
        p.write_text(code)
        return self.call([COL, "exec", "-s", self.name, "-f", str(p), "--timeout", str(max(30, want - 60))],
                         want, deadline, label)

    def launch_state(self):
        """After a launch call without acknowledgement: 'started' when the VM script has claimed its run directory
        or published its PID within 90 s, 'absent' when neither appears, None when the probe itself keeps failing.
        The probe only reads, so it is retried."""
        code = ("import os, time\n"
                f"pid, run = '/content/vm_{self.run}.pid', '/content/run_{self.run}'\n"
                "for _ in range(90):\n"
                "    if os.path.exists(pid) or os.path.isdir(run):\n"
                "        print('STARTED'); raise SystemExit\n"
                "    time.sleep(1)\n"
                "print('ABSENT')\n")
        for attempt in (1, 2, 3):
            self.check()
            rc, out = self.exec_py(code, 200, self.work_end, f"launch_probe{attempt}")
            if rc == 0 and ("STARTED" in out.split() or "ABSENT" in out.split()):
                return "started" if "STARTED" in out.split() else "absent"
        return None

    def launch(self):
        """Fetch vm_gemma.sh at the commit and start it detached. The Colab CLI sometimes hangs on a call, so a launch
        call without acknowledgement is reconciled, never blindly repeated: if the script has started, the run goes
        on; only if it is shown absent is the launch tried once more. vm_gemma.sh claims its run directory with an
        atomic mkdir, so a delayed first launch and the retry cannot both run (the second exits before doing anything)."""
        url = (f"https://raw.githubusercontent.com/JohnnyTeutonic/ember.cpp/{self.commit}/"
               "experiments/hf_cuda_2026-10-30/gemma_colab_20261103/vm_gemma.sh")
        code = (
            "import subprocess, urllib.request\n"
            f"src = urllib.request.urlopen({url!r}).read()\n"
            f"open('/content/vm_run_{self.run}.sh', 'wb').write(src.replace(b'\\r\\n', b'\\n'))\n"
            f"subprocess.Popen('nohup bash /content/vm_run_{self.run}.sh {self.commit} {self.run} "
            f">> /content/vm_run_{self.run}.log 2>&1 &', shell=True)\n"
            "print('LAUNCHED')\n")
        self.launch_attempted = True  # from here the script may be running, whatever the acknowledgement says
        for label in ("launch", "launch_retry"):
            rc, out = self.exec_py(code, 200, self.work_end, label)
            if rc == 0 and "LAUNCHED" in out:
                return True
            state = self.launch_state()
            self.log(f"{label}: no acknowledgement; the VM script is {state or 'unknown'}")
            if state == "started":
                return True
            if state is None:
                return False  # unknown: never launch again; the stop path reconciles
        return False

    def upload(self):
        """The Google files and converter to /content/upload_RUN/; vm_gemma.sh checks them against inputs.sha256.
        Both steps are idempotent, so each is tried up to three times (the Colab CLI sometimes hangs on a call)."""
        code = f"import os\nos.makedirs('/content/upload_{self.run}', exist_ok=True)\nprint('MADE')\n"
        for attempt in (1, 2, 3):
            self.check()
            rc, out = self.exec_py(code, 120, self.work_end, f"mkdir{attempt}")
            if rc == 0 and "MADE" in out:
                break
        else:
            return False
        for name, src in UPLOAD_SOURCES.items():
            for attempt in (1, 2, 3):
                self.check()
                rc, _ = self.call([COL, "upload", "-s", self.name, src, f"/content/upload_{self.run}/{name}"], 600,
                                  self.work_end, f"upload {name} ({attempt})")
                if rc == 0:
                    break
            else:
                return False
        return True

    def poll(self):
        code = (f"import os\np = '/content/run_{self.run}/out/DONE'\n"
                f"print('DONE' if os.path.exists(p) and open(p).read().strip() == {self.run!r} else 'RUNNING')\n")
        while time.monotonic() < self.work_end and not self.cancel:
            self.sleep(2 * MINUTE, self.work_end)
            if self.cancel:
                break
            self.call(ADOPT + ["--refresh"], 180, self.work_end, "refresh")
            rc, out = self.exec_py(code, 200, self.work_end, "poll")
            if rc == 0 and "DONE" in out.split():
                return True
        return False

    def stop_workload(self):
        """Polling ended without DONE (or the launch acknowledgement was lost): send SIGTERM to the VM script's
        recorded PID, after checking that the PID is still that script. Its handler stops the controller's process
        group, waits (bounded) until it is gone, writes stop.json and exits through the trap that writes status.json
        and DONE. The VM side answers with one STOP_REPLY JSON line:
          interrupted  DONE holds our run id and stop.json says the controller group is gone or never started;
          completed    DONE holds our run id and there is no stop.json: the script finished on its own (its
                       controller and any leftover group member had already been stopped before publication);
          unconfirmed  anything else; not_observed  no PID file after waiting for one.
        Returns True (interrupted or completed), False (not confirmed, including a launch that was attempted but
        never observed) or None (no launch was attempted)."""
        if not self.launch_attempted:
            self.log("workload stop: no launch was attempted")
            return None
        code = (
            "import json, os, signal, time\n"
            f"run, script = {self.run!r}, '/content/vm_run_{self.run}.sh'\n"
            f"pid_file, done, stop = '/content/vm_{self.run}.pid', '/content/run_{self.run}/out/DONE', "
            f"'/content/run_{self.run}/out/stop.json'\n"
            "def reply(result, **kw):\n"
            "    print('STOP_REPLY ' + json.dumps(dict(result=result, **kw))); raise SystemExit\n"
            "def done_ok():\n"
            "    try:\n"
            "        return open(done).read().strip() == run\n"
            "    except OSError:\n"
            "        return False\n"
            "for _ in range(60):  # a script launched late publishes its PID within seconds of starting\n"
            "    if os.path.exists(pid_file):\n"
            "        break\n"
            "    time.sleep(1)\n"
            "else:\n"
            "    reply('not_observed')\n"
            "pid = int(open(pid_file).read())\n"
            "try:\n"
            "    ours = script.encode() in open(f'/proc/{pid}/cmdline', 'rb').read()\n"
            "except OSError:\n"
            "    ours = False\n"
            "if ours and not done_ok():\n"
            "    try:\n"
            "        os.kill(pid, signal.SIGTERM)\n"
            "    except ProcessLookupError:  # it finished in between; DONE and stop evidence decide below\n"
            "        pass\n"
            "for _ in range(120):\n"
            "    if done_ok():\n"
            "        break\n"
            "    time.sleep(1)\n"
            "s = json.load(open(stop)) if os.path.exists(stop) else None\n"
            "if not done_ok():\n"
            "    reply('unconfirmed', stop=s, signalled=ours)\n"
            "if s is None:\n"
            "    reply('completed', signalled=ours)\n"
            "gone = s.get('controller_group_gone') is True or s.get('controller_started') is False\n"
            "reply('interrupted' if gone else 'unconfirmed', stop=s, signalled=ours)\n")
        # idempotent (it signals only a live script without DONE), so a hung call is tried once more
        result, out = None, ""
        for attempt in (1, 2):
            rc, out = self.exec_py(code, 260, self.retrieve_end, f"stop_workload{attempt}")
            reply = None
            for line in out.splitlines():
                if line.startswith("STOP_REPLY "):
                    try:
                        reply = json.loads(line[len("STOP_REPLY "):])
                    except ValueError:
                        reply = None
            result = reply.get("result") if rc == 0 and isinstance(reply, dict) else None
            if result is not None:
                break
        confirmed = result in ("interrupted", "completed")
        self.log("workload stop:", f"confirmed ({result})" if confirmed else
                 f"NOT CONFIRMED ({result}): the archive may hold outputs still being written", out.strip()[-300:])
        return confirmed

    def retrieve(self):
        # GNU tar exits 1 when a file changed while it was read; the archive is still written and is evidence
        code = ("import subprocess\n"
                f"subprocess.run('cp /content/vm_run_{self.run}.log /content/run_{self.run}/out/ 2>/dev/null', shell=True)\n"
                f"r = subprocess.run('tar czf /content/scan_{self.run}.tgz --warning=no-file-changed -C /content "
                f"run_{self.run}/out', shell=True)\n"
                "print('PACKED' if r.returncode in (0, 1) else 'PACK_FAILED %d' % r.returncode)\n")
        archive = self.out / "scan_out.tgz"
        prefix = f"run_{self.run}/out"
        for attempt in (1, 2):
            archive.unlink(missing_ok=True)  # never validate a partial archive from an earlier attempt
            rc, out = self.exec_py(code, 300, self.retrieve_end, f"pack{attempt}")
            if rc != 0 or "PACKED" not in out:
                continue
            rc, _ = self.call([COL, "download", "-s", self.name, f"/content/scan_{self.run}.tgz", str(archive)],
                              600, self.retrieve_end, f"download {attempt}")
            if rc != 0 or not archive.is_file():
                continue
            # each attempt extracts into its own fresh staging folder; only a complete extraction is promoted
            stage = self.out / f"extract_{attempt}"
            try:
                shutil.rmtree(stage, ignore_errors=True)
                with tarfile.open(archive) as t:
                    members = t.getmembers()
                    bad = [m.name for m in members if not (m.isfile() or m.isdir())
                           or not (m.name == prefix or m.name.startswith(prefix + "/")) or ".." in Path(m.name).parts]
                    if bad:
                        self.log("refusing archive members", bad[:5])
                        continue
                    t.extractall(stage, members=members)
                (stage / f"run_{self.run}").rename(self.out / f"run_{self.run}")
                shutil.rmtree(stage, ignore_errors=True)
                return True
            except (tarfile.TarError, OSError) as e:
                self.log(f"archive attempt {attempt} unreadable:", e)
        self.log("EVIDENCE NOT RETRIEVED: the VM holds the only copy, and cleanup will stop it")
        return False

    def copy_receipts(self):
        src = self.out / f"run_{self.run}" / "out"
        try:
            self.receipts.mkdir(parents=True, exist_ok=False)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.receipts / f.name)
            # Everything retrieved, including out/binaries/ and out/inputs/, hashed into the receipts (in 1 MiB
            # chunks; this runs after VM cleanup, so local I/O never holds the VM).
            lines = []
            for f in sorted(p for p in src.rglob("*") if p.is_file()):
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                lines.append(f"{h.hexdigest()}  {f.relative_to(src).as_posix()}\n")
            tmp = self.receipts / "archive.sha256.tmp"
            tmp.write_text("".join(lines))
            tmp.rename(self.receipts / "archive.sha256")
            return True
        except Exception as e:  # never let receipt work prevent the exit-code rules
            self.log("receipt copy failed:", repr(e))
            return False

    def validate(self):
        """True: validated pass. False: a well-formed status of a failed experiment. None: missing or malformed."""
        try:
            s = json.loads((self.out / f"run_{self.run}" / "out" / "status.json").read_text())
        except (OSError, ValueError) as e:
            self.log("status.json missing or unreadable:", e)
            return None
        gates = s.get("gates") if isinstance(s, dict) else None
        if not isinstance(gates, dict) or not all(type(v) is int for v in gates.values()) \
                or type(s.get("exit_status")) is not int or not isinstance(s.get("pass"), bool) \
                or not isinstance(s.get("science_started"), bool):
            self.log("status.json malformed:", json.dumps(s)[:500])
            return None
        if s.get("run_id") != self.run or s.get("commit") != self.commit or s.get("head") != self.commit:
            self.log("status.json identity mismatch:", s.get("run_id"), s.get("commit"), s.get("head"))
            return None
        problems = []
        if set(gates) != REQUIRED_GATES:
            problems.append(f"gate set differs: missing {sorted(REQUIRED_GATES - set(gates))}, "
                            f"extra {sorted(set(gates) - REQUIRED_GATES)}")
        failed = sorted(k for k, v in gates.items() if v != 0)
        if failed:
            problems.append(f"failed gates {failed}")
        if s["exit_status"] != 0:
            problems.append(f"VM exit status {s['exit_status']}")
        if s["pass"] is not True:
            problems.append("pass is not true")
        if s["science_started"] is not True:
            problems.append("the controller never started")
        # a process the VM script could not take down (QUIESCENCE_UNCONFIRMED, or stop.json saying the group is not
        # gone) means the archive may not be a final snapshot, whatever the stop reply said
        out = self.out / f"run_{self.run}" / "out"
        try:
            stop = json.loads((out / "stop.json").read_text()) if (out / "stop.json").exists() else {}
        except (OSError, ValueError):
            stop = {"controller_group_gone": None}
        if (out / "QUIESCENCE_UNCONFIRMED").exists() or stop.get("controller_group_gone", True) is not True:
            self.log("quiescence not confirmed by the VM script: the archive may not be final")
            self.stop_confirmed = False
        self.log("status.json:", json.dumps(s))
        self.log("validation:", "PASS" if not problems else "; ".join(problems))
        # AMENDMENT.md, outcomes: a run that reached the controller may not be repeated, whatever happened next
        self.log("science started:", s.get("science_started"),
                 "(no rerun allowed)" if s.get("science_started") is not False else "(a rerun is allowed)")
        return not problems

    def cleanup(self):
        """True only when an authoritative listing shows neither our name, our endpoint, nor any endpoint new since
        before the creation call. Never raises."""
        if not self.intent:
            return True
        for attempt in range(1, 6):
            try:
                if time.monotonic() >= self.shutdown_end:
                    break
                self.call([COL, "stop", "-s", self.name], 120, self.shutdown_end, f"stop {attempt}")
                self.sleep(MINUTE / 6, self.shutdown_end)
                rows = self.sessions(self.shutdown_end)
                if rows is None:
                    continue
                names = {n for n, _ in rows}
                endpoints = {ep for _, ep in rows}
                new = endpoints - (self.before or set()) - ({self.endpoint} if self.endpoint else set())
                ours_listed = self.name in names or (self.endpoint is not None and self.endpoint in endpoints)
                if not ours_listed and not new:
                    self.log(f"CLEANUP OK: {self.name} ({self.endpoint}) and no new endpoint listed")
                    return True
                if self.endpoint is not None and self.endpoint in endpoints and self.name not in names:
                    # the CLI lost our name: re-adopt our own endpoint, then stop it on the next attempt
                    self.call(ADOPT + ["--adopt", f"{self.endpoint}={self.name}"], 180, self.shutdown_end, "adopt")
                if new:
                    self.log("endpoints new since our creation call, ownership not established:", sorted(new))
            except Exception as e:  # keep reconciling within the deadline
                self.log(f"cleanup attempt {attempt} error:", repr(e))
        self.log(f"CLEANUP UNRESOLVED: {self.name} (endpoint {self.endpoint}) may still be running, or a VM our "
                 f"creation call allocated is unnamed. Inspect `{COL} sessions`; stop ours with "
                 f"`{COL} stop -s {self.name}` or adopt the endpoint first (tools/colab_adopt.py --adopt EP=NAME).")
        return False

    def main(self):
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.log_file = open(self.out / "driver.log", "a")
        except OSError:
            self.log_file = None
        self.log(f"run {self.run} commit {self.commit} session {self.name}" + (" [TEST HARNESS]" if TEST else ""))
        transport = receipts = False
        valid = None
        try:
            self.preflight()
            self.check()
            self.create()
            self.check()
            launched = self.upload() and self.launch()
            done = self.poll() if launched else False
            self.done = done
            self.log("launched", launched, "done", done, "cancelled", self.cancel)
            if not done:  # also when the launch acknowledgement was lost: the script may be running
                self.stop_confirmed = self.stop_workload()
            transport = self.retrieve()  # also after a launch failure, deadline or signal: diagnostics are evidence
            valid = self.validate() if transport else None
        except Stop as e:
            self.log("stopped:", e)
        except Exception as e:
            self.log("error:", repr(e))
        finally:
            clean = False
            try:
                clean = self.cleanup()
            except Exception as e:  # cleanup() does not raise; this is the last line of defence
                self.log("cleanup error:", repr(e))
            for label, argv in (("final sessions", [COL, "sessions"]), ("final orphans", REAP)):
                rc, out = self.call(argv, 120, self.inventory_end, label)
                self.log(f"{label}:\n" + out.strip())
            receipts = transport and self.copy_receipts()  # local work only, after the VM is gone
            code =3 if not clean else 2 if not (transport and receipts and valid is not None) else 0 if valid else 1
            if code == 0 and not self.done:
                self.log("validated status but no matching DONE marker was seen: exit 2")
                code = 2
            if self.log_failed:
                code = max(code, 2)
            if self.stop_confirmed is False:  # the archive may not be a final snapshot
                self.log("the workload stop was not confirmed: exit at least 2")
                code = max(code, 2)
            self.log(f"exit {code} (transport {transport}, receipts {receipts}, valid {valid}, cleanup {clean})")
            try:
                if self.log_file:
                    self.log_file.close()
                if receipts:
                    shutil.copy2(self.out / "driver.log", self.receipts / "driver.log")
            except OSError:
                code = max(code, 2)
            return code


def main():
    if not TEST and any(k.startswith("DRIVE_TEST_") for k in os.environ):
        print("DRIVE_TEST_* is set but DRIVE_TEST_HARNESS is not: refusing a real run with test hooks",
              file=sys.stderr)
        return 2
    if len(sys.argv) != 3 or not re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]) \
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", sys.argv[2]):
        print(__doc__, file=sys.stderr)
        return 2
    d = Driver(sys.argv[1], sys.argv[2])

    def on_signal(signum, _frame):  # request cancellation; observed at controlled points
        d.cancel = True

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    return d.main()


if __name__ == "__main__":
    sys.exit(main())
