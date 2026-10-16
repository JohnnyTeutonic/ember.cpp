#!/usr/bin/env python3
"""TTFT measurements on Colab through the repo's supervisor (docs/TTFT.md).

    python3 tools/ttft_supervise.py status [--variant NAME]   # banked / missing
    python3 tools/ttft_supervise.py run    [--variant NAME]   # one L4 until done
    python3 tools/ttft_supervise.py stop   [--variant NAME]   # SIGTERM via pidfile

--variant picks the kernel under test (default: baseline). Each variant is
its own lane: own session name, bank, pidfile and status log. Run one lane
at a time (one Colab session at a time).

Path B of microtorch/AGENTS.md: a resumable job under
tools/colab_supervisor/supervisor.py. This file only DEFINES the job and
runs the supervisor's own loop with it alone; the shared JOBS list, pidfile,
status log and registry reaper are not touched (other lanes and other
sessions use them).

Why resumable: in Oct 2026 busy L4 sessions were reclaimed ~22 minutes after
creation. The unit of work is therefore one (model, run, length) cell, not a
run. Each VM gets a to-do list of what is not yet banked, passes a quick
parity gate, and works through it; pull() copies the per-cell-flushed JSONL
home every tick, so a reclaim costs at most the cell in progress. A cell is
banked once its summary row exists. Done = every cell banked, the nsys
kernel pass, and (for a variant) its one-time full parity gate.

Only committed code is measured: the job ships `git archive HEAD` of
ember.cpp and paperkiln, and the VM script from HEAD.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
sys.path.insert(0, f"{REPO}/tools/colab_supervisor")
import supervisor as sup  # noqa: E402

EMBER = f"{REPO}/tinyllama.cpp"
PK = f"{REPO}/microtorch"
ART = "/mnt/c/ml_artifacts/ttft"

# name -> session name, bank, server env selecting it, per-length request
# counts ("" = VM default 2048:2:5 4096:1:3, "none" = full protocol at every
# length), and whether a one-time full parity gate against the baseline
# kernel must be banked before the lane counts as done.
VARIANTS = {
    "baseline": dict(session="ttft", bank=f"{ART}/bank", env="", overrides="",
                     parity_gate=False),
    "flash_v1": dict(session="ttftv1", bank=f"{ART}/bank_flash_v1",
                     env="EMBER_PREFILL_ATTN=flash", overrides="none", parity_gate=True),
}
FLASH_PARITY_FILES = ("timing_baseline_vs_flash_full.jsonl", "story3k_baseline_vs_flash.jsonl")

RUNS = range(1, 6)
TIMING_LENGTHS = [128, 512, 1024, 2048, 4096]
NSYS_LENGTHS = [128, 1024]

V = None  # the selected variant's entry, set in main()


def required():
    cells = {("story3k", r, 128) for r in RUNS}
    cells |= {("timing", r, n) for r in RUNS for n in TIMING_LENGTHS}
    return cells


def banked():
    """Cells with a summary row anywhere in this lane's bank (any launch)."""
    have = set()
    for path in glob.glob(f"{V['bank']}/*/*/summary.jsonl"):
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
    if glob.glob(f"{V['bank']}/*/full/*/NSYS_UNAVAILABLE"):
        return True
    return all(glob.glob(f"{V['bank']}/*/full/*/nsys_timing_{n}") for n in NSYS_LENGTHS)


def parity_banked():
    """A variant's full gate: every row of both files passed, in one launch."""
    if not V["parity_gate"]:
        return True
    for launch in glob.glob(f"{V['bank']}/*/parity"):
        rows = []
        for name in FLASH_PARITY_FILES:
            path = os.path.join(launch, name)
            if not os.path.exists(path):
                break
            with open(path) as f:
                rows += [json.loads(line) for line in f if line.strip()]
        else:
            if rows and all(r["pass"] for r in rows):
                return True
    return False


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


class TtftJob(sup.Job):
    """ember.cpp TTFT lane: 30 cells over 5 fresh-server runs + nsys (+ gate)."""
    gpu = "L4"
    priority = 0
    MAX_LAUNCHES = 6   # launches WITHOUT newly banked work before STALLED

    def __init__(self, variant):
        self.name = VARIANTS[variant]["session"]
        super().__init__()
        self.variant = variant
        self.stage = os.path.expanduser(f"~/ttft_supervisor/{variant}/stage")
        os.makedirs(self.stage, exist_ok=True)
        os.makedirs(V["bank"], exist_ok=True)
        self.ec = git("-C", EMBER, "rev-parse", "--short=12", "HEAD").strip()
        self.pc = git("-C", PK, "rev-parse", "--short=12", "HEAD").strip()
        git("-C", EMBER, "archive", "--format=tar.gz", "--prefix=ember/",
            "-o", f"{self.stage}/ember.tgz", "HEAD")
        git("-C", PK, "archive", "--format=tar.gz", "--prefix=microtorch/",
            "-o", f"{self.stage}/paperkiln.tgz", "HEAD",
            "CMakeLists.txt", "include", "src", "third_party", "tools", "studio", "web",
            "python", "tests")
        with open(f"{self.stage}/colab_ttft_vm.sh", "w") as f:
            f.write(git("-C", EMBER, "show", "HEAD:tools/colab_ttft_vm.sh"))
        with open(f"{self.stage}/ttft_variant.txt", "w") as f:
            f.write(f"{variant}\n{V['env']}\n{V['overrides']}\n")
        subprocess.run(["unzip", "-jo", "/mnt/c/ml_artifacts/paperkiln_story3k.zip",
                        "paperkiln_story3k/story3k.gguf", "-d", self.stage],
                       check=True, capture_output=True)
        with open(f"{self.stage}/boot.py", "w") as f:
            f.write(BOOT.replace("@EC@", self.ec).replace("@PC@", self.pc))
        self.vm_out = None
        sup.log(f"[{self.name}] variant {variant}: staged ember {self.ec}, paperkiln {self.pc}")

    # -- what is left ------------------------------------------------------
    def todo(self):
        items = [f"{m} {r}:{n}" for m, r, n in sorted(required() - banked())]
        if not nsys_banked():
            items.append("nsys")
        if not parity_banked():
            items.append("parity_variant")
        return items

    def is_done(self):
        return not self.todo()

    def uploads(self):
        items = self.todo()
        with open(f"{self.stage}/ttft_todo.txt", "w") as f:
            f.write("\n".join(items) + "\n")
        self.vm_out = None
        sup.log(f"[{self.name}] launching with {len(items)} to-do items")
        return [(f"{self.stage}/{name}", f"/content/{name}") for name in
                ("ember.tgz", "paperkiln.tgz", "story3k.gguf", "colab_ttft_vm.sh",
                 "ttft_todo.txt", "ttft_variant.txt")]

    def boot(self):
        return f"{self.stage}/boot.py"

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
            marker = f"{self.stage}/OUT_DIR"
            if self.fetch("/content/TTFT_OUT_DIR", marker, timeout=120):
                with open(marker) as f:
                    self.vm_out = f.read().strip()
                os.remove(marker)
        if self.vm_out is None:
            return
        dest = os.path.join(V["bank"], os.path.basename(self.vm_out))
        for model in ("story3k", "timing"):
            os.makedirs(f"{dest}/{model}", exist_ok=True)
            for name in ("raw.jsonl", "summary.jsonl"):
                self.fetch(f"{self.vm_out}/{model}/{name}", f"{dest}/{model}/{name}")
        if V["parity_gate"] and not parity_banked():
            os.makedirs(f"{dest}/parity", exist_ok=True)
            for name in FLASH_PARITY_FILES:
                self.fetch(f"{self.vm_out}/parity/{name}", f"{dest}/parity/{name}")
        self.fetch("/content/ttft_vm.log", f"{dest}/ttft_vm.log")
        rc, _ = self.colab("download", "-s", self.name, "/content/TTFT_DONE",
                           f"{self.stage}/TTFT_DONE", timeout=120)
        if rc == 0 and not os.path.exists(f"{dest}/full"):
            if self.fetch("/content/ttft_out.zip", f"{dest}/ttft_out.zip", timeout=900):
                shutil.unpack_archive(f"{dest}/ttft_out.zip", f"{dest}/full")

    def progress(self):
        need = required()
        extra = "" if not V["parity_gate"] else \
            f", gate {'passed' if parity_banked() else 'pending'}"
        return (f"{len(need & banked())}/{len(need)} cells, "
                f"nsys {'banked' if nsys_banked() else 'pending'}{extra}")

    def progress_token(self):
        return (len(banked()), nsys_banked(), parity_banked())

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["status", "run", "stop"])
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    args = ap.parse_args()
    global V
    V = VARIANTS[args.variant]
    lane = os.path.expanduser(f"~/ttft_supervisor/{args.variant}")
    if args.variant == "baseline" and os.path.exists(os.path.expanduser("~/ttft_supervisor/supervisor.pid")):
        lane = os.path.expanduser("~/ttft_supervisor")   # the lane started before variants existed
    os.makedirs(lane, exist_ok=True)
    sup.PIDFILE = os.path.join(lane, "supervisor.pid")
    sup.STATUS = os.path.join(lane, "status.log")
    session = V["session"]

    if args.cmd == "status":
        miss = sorted(required() - banked())
        print(f"[{args.variant}] banked {len(required()) - len(miss)}/{len(required())} cells; "
              f"nsys {'banked' if nsys_banked() else 'pending'}; "
              f"gate {'passed' if parity_banked() else 'pending'}")
        for m in miss:
            print("  missing", *m)
        return
    if args.cmd == "stop":
        sup.stop()
        return

    def reap_own(active_names):
        """The shared reaper stops EVERY registered session not active here,
        which would include other lanes' and other agents' sessions. Reap
        only this lane's own snapshots."""
        if session in active_names:
            return
        for snap in sorted(glob.glob(os.path.join(sup.REG, f"{session}-*.json"))):
            sup.sh([sup.COL, "--config", snap, "stop", "-s", session], timeout=90)
            os.rename(snap, snap + ".reaped")

    sup.JOBS = [TtftJob(args.variant)]
    sup.PAUSED = set()
    sup.MAX_SESSIONS = 2      # one TTFT VM; leaves room for one other agent's session
    sup.TICK_S = 60           # a reclaim is seen within ~2 ticks, not ~10 min
    sup.reap_registry = reap_own
    sup.run()


if __name__ == "__main__":
    main()
