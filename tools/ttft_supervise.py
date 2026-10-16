#!/usr/bin/env python3
"""TTFT baseline on Colab through the repo's supervisor (docs/TTFT.md).

    python3 tools/ttft_supervise.py status   # banked / missing cells
    python3 tools/ttft_supervise.py run      # one L4 at a time until done
    python3 tools/ttft_supervise.py stop     # SIGTERM via this lane's pidfile

Path B of microtorch/AGENTS.md: a resumable job under
tools/colab_supervisor/supervisor.py. This file only DEFINES the job and
runs the supervisor's own loop with it alone; the shared JOBS list, pidfile,
status log and registry reaper are not touched (other lanes and other
sessions use them).

Why resumable: in Oct 2026 busy L4 sessions were reclaimed ~22 minutes after
creation. The unit of work is therefore one (model, run, length) cell, not a
run. Each VM gets a to-do list of the cells not yet banked, passes a quick
parity gate, and works through them; pull() copies the per-cell-flushed JSONL
home every tick, so a reclaim costs at most the cell in progress. A cell is
banked once its summary row exists. Done = every cell banked plus the nsys
kernel pass.

Only committed code is measured: the job ships `git archive HEAD` of
ember.cpp and paperkiln, and the VM script from HEAD.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
sys.path.insert(0, f"{REPO}/tools/colab_supervisor")
import supervisor as sup  # noqa: E402

EMBER = f"{REPO}/tinyllama.cpp"
PK = f"{REPO}/microtorch"
BANK = "/mnt/c/ml_artifacts/ttft/bank"          # <launch>/<model>/{raw,summary}.jsonl
LANE = os.path.expanduser("~/ttft_supervisor")  # this lane's pidfile + status log
STAGE = os.path.join(LANE, "stage")

RUNS = range(1, 6)
TIMING_LENGTHS = [128, 512, 1024, 2048, 4096]
NSYS_LENGTHS = [128, 1024]


def required():
    cells = {("story3k", r, 128) for r in RUNS}
    cells |= {("timing", r, n) for r in RUNS for n in TIMING_LENGTHS}
    return cells


def banked():
    """Cells with a summary row anywhere in the bank (any launch)."""
    have = set()
    for path in glob.glob(f"{BANK}/*/*/summary.jsonl"):
        model = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    have.add((model, row["run"], row["input_len"]))
    return have


def nsys_banked():
    """The kernel pass came home (full/<vm out dir>/nsys_timing_<n>), or the
    image had no nsys, which is recorded rather than retried forever."""
    if glob.glob(f"{BANK}/*/full/*/NSYS_UNAVAILABLE"):
        return True
    return all(glob.glob(f"{BANK}/*/full/*/nsys_timing_{n}") for n in NSYS_LENGTHS)


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


class TtftJob(sup.Job):
    """ember.cpp TTFT baseline: 30 cells over 5 fresh-server runs + nsys."""
    name = "ttft"
    gpu = "L4"
    priority = 0
    MAX_LAUNCHES = 6   # launches WITHOUT a newly banked cell before STALLED

    def __init__(self):
        super().__init__()
        os.makedirs(STAGE, exist_ok=True)
        os.makedirs(BANK, exist_ok=True)
        self.ec = git("-C", EMBER, "rev-parse", "--short=12", "HEAD").strip()
        self.pc = git("-C", PK, "rev-parse", "--short=12", "HEAD").strip()
        git("-C", EMBER, "archive", "--format=tar.gz", "--prefix=ember/",
            "-o", f"{STAGE}/ember.tgz", "HEAD")
        git("-C", PK, "archive", "--format=tar.gz", "--prefix=microtorch/",
            "-o", f"{STAGE}/paperkiln.tgz", "HEAD",
            "CMakeLists.txt", "include", "src", "third_party", "tools", "studio", "web",
            "python", "tests")
        with open(f"{STAGE}/colab_ttft_vm.sh", "w") as f:
            f.write(git("-C", EMBER, "show", "HEAD:tools/colab_ttft_vm.sh"))
        subprocess.run(["unzip", "-jo", "/mnt/c/ml_artifacts/paperkiln_story3k.zip",
                        "paperkiln_story3k/story3k.gguf", "-d", STAGE],
                       check=True, capture_output=True)
        with open(f"{STAGE}/boot.py", "w") as f:
            f.write(BOOT.replace("@EC@", self.ec).replace("@PC@", self.pc))
        self.vm_out = None
        sup.log(f"[ttft] staged ember {self.ec}, paperkiln {self.pc}")

    # -- what is left ------------------------------------------------------
    def missing(self):
        return sorted(required() - banked())

    def is_done(self):
        return not self.missing() and nsys_banked()

    def uploads(self):
        todo = [f"{m} {r}:{n}" for m, r, n in self.missing()]
        if not nsys_banked():
            todo.append("nsys")
        with open(f"{STAGE}/ttft_todo.txt", "w") as f:
            f.write("\n".join(todo) + "\n")
        self.vm_out = None
        sup.log(f"[ttft] launching with {len(todo)} to-do items")
        return [(f"{STAGE}/{name}", f"/content/{name}") for name in
                ("ember.tgz", "paperkiln.tgz", "story3k.gguf", "colab_ttft_vm.sh",
                 "ttft_todo.txt")]

    def boot(self):
        return f"{STAGE}/boot.py"

    # -- relay -------------------------------------------------------------
    def fetch(self, remote, local, timeout=180):
        """Download to a temp file; replace only if it did not shrink."""
        tmp = local + ".tmp"
        rc, _ = self.colab("download", "-s", self.name, remote, tmp, timeout=timeout)
        if rc != 0 or not os.path.exists(tmp):
            return False
        if os.path.getsize(tmp) >= (os.path.getsize(local) if os.path.exists(local) else 0):
            os.replace(tmp, local)
        else:
            os.remove(tmp)
        return True

    def pull(self):
        if self.vm_out is None:
            marker = f"{STAGE}/OUT_DIR"
            if self.fetch("/content/TTFT_OUT_DIR", marker, timeout=120):
                with open(marker) as f:
                    self.vm_out = f.read().strip()
                os.remove(marker)
        if self.vm_out is None:
            return
        dest = os.path.join(BANK, os.path.basename(self.vm_out))
        for model in ("story3k", "timing"):
            os.makedirs(f"{dest}/{model}", exist_ok=True)
            for name in ("raw.jsonl", "summary.jsonl"):
                self.fetch(f"{self.vm_out}/{model}/{name}", f"{dest}/{model}/{name}")
        self.fetch("/content/ttft_vm.log", f"{dest}/ttft_vm.log")
        rc, _ = self.colab("download", "-s", self.name, "/content/TTFT_DONE",
                           f"{STAGE}/TTFT_DONE", timeout=120)
        if rc == 0 and not os.path.exists(f"{dest}/full"):
            if self.fetch("/content/ttft_out.zip", f"{dest}/ttft_out.zip", timeout=900):
                shutil.unpack_archive(f"{dest}/ttft_out.zip", f"{dest}/full")

    def progress(self):
        need = required()
        return (f"{len(need & banked())}/{len(need)} cells, "
                f"nsys {'banked' if nsys_banked() else 'pending'}")

    def alive(self):
        # Liveness from the control-plane listing, not an exec probe: a busy
        # VM can stall exec for many minutes, and a reclaimed one vanishes
        # from the listing at once (colab_transfer_runner.py alive()).
        rc, out = self.colab("sessions", timeout=120)
        return rc != 0 or f"[{self.name}]" in out


# Idempotent detached launcher (Popen, fds on /dev/null, own session):
# re-running it on the same VM never starts a second job.
BOOT = r'''
import os, subprocess
if not os.path.exists("/content/ttft_vm.log"):
    subprocess.Popen("cd /content && bash colab_ttft_vm.sh resume @EC@ @PC@", shell=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
print("TTFT_BOOTED")
'''


def reap_own(active_names):
    """The shared reaper stops EVERY registered session not active here,
    which would include other lanes' and other agents' sessions. Reap only
    this lane's own snapshots."""
    for snap in sorted(glob.glob(os.path.join(sup.REG, "ttft-*.json"))):
        if "ttft" in active_names:
            continue
        sup.sh([sup.COL, "--config", snap, "stop", "-s", "ttft"], timeout=90)
        os.rename(snap, snap + ".reaped")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    os.makedirs(LANE, exist_ok=True)
    sup.PIDFILE = os.path.join(LANE, "supervisor.pid")
    sup.STATUS = os.path.join(LANE, "status.log")
    if cmd == "status":
        miss = sorted(required() - banked())
        print(f"banked {len(required()) - len(miss)}/{len(required())} cells; "
              f"nsys {'banked' if nsys_banked() else 'pending'}")
        for m in miss:
            print("  missing", *m)
        return
    if cmd == "stop":
        sup.stop()
        return
    if cmd != "run":
        sys.exit(__doc__)
    sup.JOBS = [TtftJob()]
    sup.PAUSED = set()
    sup.MAX_SESSIONS = 2      # one TTFT VM; leaves room for one other agent's session
    sup.TICK_S = 60           # a reclaim is seen within ~2 ticks, not ~10 min
    sup.reap_registry = reap_own
    sup.run()


if __name__ == "__main__":
    main()
