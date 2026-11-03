"""Stage the parts4_vm.py upload from a commit, then preflight it offline (no GPU, no provider call).

  python3 stage_upload.py --commit SHA --model MODEL.gguf --prompts eval100.jsonl --out UPLOAD_DIR

candidate.tgz is `git archive` of the commit; packet.json and parts4_vm.py come from the same commit. The model
and prompts are hard-linked when they share a filesystem, else symlinked to the canonical file; never copied.
Transport must send resolved regular files (the payload refuses links). The preflight repeats
every pre-device check parts4_vm.py makes, on the extracted archive: manifest bytes and hashes, registered input
pins, source pins, frozen collector, the packet tests and the trace header's C++ test. Prints the manifest SHA-256
for --manifest-sha256 and writes preflight.json beside the upload.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

PACKET = "experiments/spec_r1_2026-10-28/parts4_route_20261102"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def place(source, target):
    """A hard link on the same filesystem, else a symlink to the canonical file: never a second copy. Transport
    must send the resolved regular file (ordered_reductions TRANSPORT_REPAIRS.md, V3); the payload refuses links."""
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        os.symlink(source, target)
        return "symlink"


def stage(repo, commit, model, prompts, out):
    commit = git(repo, "rev-parse", commit + "^{commit}").decode().strip()
    out.mkdir(parents=True, exist_ok=False)
    archive = git(repo, "archive", "--format=tar", commit)
    with tarfile.open(out / "candidate.tgz", "w:gz") as tgz, tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                continue  # the payload refuses links; record what was left out
            tgz.addfile(member, tar.extractfile(member) if member.isfile() else None)
    for name in ("packet.json", "parts4_vm.py", "timing_gate.py"):
        (out / name).write_bytes(git(repo, "show", f"{commit}:{PACKET}/{name}"))
    placed = {"model.gguf": place(model, out / "model.gguf"), "eval100.jsonl": place(prompts, out / "eval100.jsonl")}
    files = {n: {"bytes": (out / n).stat().st_size, "sha256": sha(out / n)}
             for n in ("candidate.tgz", "model.gguf", "eval100.jsonl", "packet.json", "parts4_vm.py", "timing_gate.py")}
    manifest = {"schema": "ember.e10.parts4-upload.v1", "source_commit": commit, "files": files, "placed": placed}
    (out / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return commit, sha(out / "inputs.json")


def preflight(upload):
    inputs = json.loads((upload / "inputs.json").read_text())
    for name, rec in inputs["files"].items():
        p = upload / name
        if (p.is_symlink() and name not in ("model.gguf", "eval100.jsonl")) or not p.resolve().is_file() \
                or p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
            raise ValueError("input mismatch " + name)
    packet = json.loads((upload / "packet.json").read_text())
    if (inputs["files"]["model.gguf"]["sha256"] != packet["input_pins"]["model_sha256"]
            or inputs["files"]["eval100.jsonl"]["sha256"] != packet["input_pins"]["prompts_sha256"]):
        raise ValueError("registered input pins")
    report = {"source_commit": inputs["source_commit"], "manifest_sha256": sha(upload / "inputs.json")}
    # The payload's own helpers (extraction with its link and traversal refusals, the cleaned environment), from
    # the staged copy of parts4_vm.py, so the preflight cannot drift from what the VM will do.
    spec = importlib.util.spec_from_file_location("parts4_vm_staged", upload / "parts4_vm.py")
    vm = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True  # the upload holds exactly the manifest's files
    spec.loader.exec_module(vm)
    with tempfile.TemporaryDirectory(prefix="parts4_preflight_") as tmp:
        src = Path(tmp) / "src"
        vm.extract(upload / "candidate.tgz", src)
        for relative, expected in packet["source_files_sha256"].items():
            if sha(src / relative) != expected:
                raise ValueError("source pin mismatch " + relative)
        for name in ("parts4_vm.py", "packet.json"):
            if sha(src / PACKET / name) != sha(upload / name):
                raise ValueError("payload source/archive mismatch " + name)
        for relative, expected in packet["frozen_collector_sha256"].items():
            if sha(src / relative) != expected:
                raise ValueError("frozen collector drift " + relative)
        env = vm.clean_env()

        def pytest(args, cwd, want):
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", *args],
                               cwd=cwd, env=env, capture_output=True, text=True)
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr
            if r.returncode or not re.search(r"\b%d passed\b" % want, tail) or "skipped" in tail:
                raise ValueError(" ".join(args) + ": " + tail)
            return tail
        contracts = pytest([str(src / "tools/row_pick_profile/test_integration.py")], src, 12)
        tail = pytest(["test_source_reduction.py", "test_parts4_gate.py"], src / PACKET, packet["packet_test_count"])
        exe = Path(tmp) / "test_route_trace"
        subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(src), str(src / PACKET / "test_route_trace.cpp"), "-o",
                        str(exe)], check=True)
        header = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.strip()
        report.update(selector_contracts=contracts, packet_tests=tail, trace_header_test=json.loads(header),
                      source_pins="match",
                      frozen_collector="match", input_pins="match", archive_links="none")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    commit, manifest_sha = stage(a.repo, a.commit, a.model.resolve(), a.prompts.resolve(), a.out.resolve())
    report = preflight(a.out.resolve())
    (a.out.parent / (a.out.name + ".preflight.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
