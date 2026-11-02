"""Source-bound supervised VM payload for the E10 PARTS4 gate. Never allocates resources or changes defaults.

  python3 parts4_vm.py --upload DIR --manifest-sha256 SHA --out OUT

The procedure of ordered_reductions_2026-11-01/ordered_vm.py: an upload manifest checked byte for byte, a
cleaned environment for every command, one L4, a scientific deadline, a zip checkpoint every two minutes, and no
erased evidence or retry for a pass. Upload: inputs.json (the manifest), candidate.tgz (the instrumented source
at the packet's pinned commit), model.gguf, eval100.jsonl, packet.json and this file.

Steps: build tinyllama_server and decode_pick_rows from the archive; the CPU contracts (selector source
contracts, this packet's tests and the trace header's C++ test); the retained device checks (handcrafted
records, decode_pick_fallback.py under PARTS0 and PARTS4); the traced route matrix; then the gate's 17 legs in
plan order, each verified by the registered load_run as soon as it ends; parts4_gate.py analyze last.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import zipfile

PACKET = "experiments/spec_r1_2026-10-28/parts4_route_20261102"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def clean_env(base=None, **extra):
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(("EMBER_", "TINYLLAMA_")) or key == "CUBLAS_WORKSPACE_CONFIG":
            env.pop(key)
    env.update(EMBER_DECODE_PICK_PARTS="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.update(extra)
    return env


def extract(archive, destination):
    destination.mkdir()
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if not m.isfile() and not m.isdir():
                raise ValueError("links/special source archive member")
            if not (destination / m.name).resolve().is_relative_to(destination.resolve()):
                raise ValueError("source traversal")
        t.extractall(destination)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    upload, out = a.upload.resolve(), a.out.resolve()
    out.mkdir(exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("output already contains evidence")
    science = out / "science"
    science.mkdir()
    work = out.parent / (out.name + "_work")
    if work.exists():
        raise ValueError("work directory already exists")
    work.mkdir()
    summary = {"status": "INCOMPLETE", "complete": False, "steps": [], "legs": []}
    lock, stop = threading.Lock(), threading.Event()
    deadline = time.monotonic() + 9600

    def pack():
        with lock:
            atomic_json(out / "summary.json", summary)
            tmp = out / "out.zip.tmp"
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                                 strict_timestamps=False) as z:
                for p in sorted(science.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(out))
                z.write(out / "summary.json", "summary.json")
            os.replace(tmp, out / "out.zip")

    def periodic():
        while not stop.wait(120):
            try:
                pack()
            except Exception as e:
                (out / "vm.log").open("a").write("CHECKPOINT_ERROR " + str(e) + "\n")

    def run(cmd, log, timeout=1800, env=None, cwd=None, stderr_log=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("scientific deadline")
        with log.open("wb") as f, (stderr_log.open("wb") if stderr_log else open(os.devnull, "wb")) as e:
            result = subprocess.run(cmd, stdout=f, stderr=e if stderr_log else subprocess.STDOUT,
                                    env=clean_env() if env is None else env, timeout=min(timeout, remaining), cwd=cwd)
        summary["steps"].append({"log": log.name, "returncode": result.returncode})
        return result.returncode

    def terminate(sig, frame):
        raise TimeoutError("signal " + str(sig))

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    (out / "vm.log").write_text("E10 PARTS4 supervised payload starting\n")
    thread = threading.Thread(target=periodic, daemon=True)
    thread.start()
    try:
        manifest = upload / "inputs.json"
        if sha(manifest) != a.manifest_sha256:
            raise ValueError("manifest SHA mismatch")
        inputs = json.loads(manifest.read_text())
        for name, rec in inputs["files"].items():
            p = upload / name
            if not p.is_file() or p.is_symlink() or p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
                raise ValueError("input mismatch " + name)
        atomic_json(science / "inputs.json", inputs)
        packet = json.loads((upload / "packet.json").read_text())
        atomic_json(science / "packet.json", packet)
        if (inputs["files"]["model.gguf"]["sha256"] != packet["input_pins"]["model_sha256"]
                or inputs["files"]["eval100.jsonl"]["sha256"] != packet["input_pins"]["prompts_sha256"]):
            raise ValueError("registered input pins")
        if run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
               science / "device.log", 30):
            raise RuntimeError("device metadata")
        device = (science / "device.log").read_text().strip().splitlines()
        if len(device) != 1 or "L4" not in device[0]:
            raise ValueError("exactly one L4 required")
        if run(["apt-get", "update", "-qq"], science / "apt_update.log", 300):
            raise RuntimeError("build dependency index")
        if run(["apt-get", "install", "-y", "-q", "libboost-regex-dev", "python3-dev"], science / "apt_install.log", 300):
            raise RuntimeError("build dependencies")
        if run(["bash", "-lc", "nvcc --version; cmake --version; g++ --version; python3 --version; "
                "python3 -c 'import numpy, pytest; print(numpy.__version__, pytest.__version__)'; "
                "dpkg-query -W libboost-regex-dev python3-dev"], science / "toolchain.log", 60):
            raise RuntimeError("toolchain inventory (numpy and pytest are required)")

        src = work / "src"
        extract(upload / "candidate.tgz", src)
        for relative, expected in packet["source_files_sha256"].items():
            if sha(src / relative) != expected:
                raise ValueError("source pin mismatch " + relative)
        for name in ("parts4_vm.py", "packet.json"):
            if sha(src / PACKET / name) != sha(upload / name):
                raise ValueError("payload source/archive mismatch " + name)
        for relative, expected in packet["frozen_collector_sha256"].items():
            if sha(src / relative) != expected:
                raise ValueError("frozen collector drift " + relative)

        build = work / "build"
        if run(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DHAS_CUDA=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=89"], science / "configure.log", 600):
            raise RuntimeError("configure")
        if "CUDA toolkit and cuBLAS found" not in (science / "configure.log").read_text():
            raise ValueError("CUDA build not configured")
        if run(["cmake", "--build", str(build), "--target", "tinyllama_server", "decode_pick_rows", "-j2"],
               science / "build.log", 2400):
            raise RuntimeError("build")
        server, rows_probe = build / "tinyllama_server", build / "decode_pick_rows"
        binaries = {"tinyllama_server_sha256": sha(server), "decode_pick_rows_sha256": sha(rows_probe),
                    "source_archive_sha256": sha(upload / "candidate.tgz")}
        atomic_json(science / "binaries.json", binaries)

        # CPU contracts against the archived source: no device needed, run before any device evidence.
        if run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider",
                str(src / "tools/row_pick_profile/test_integration.py")], science / "cpu_selector_contracts.log", 300):
            raise RuntimeError("selector source contracts")
        if not re.search(r"\b12 passed\b", (science / "cpu_selector_contracts.log").read_text()):
            raise ValueError("selector source contract inventory incomplete")
        if run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider",
                "test_source_reduction.py", "test_parts4_gate.py"], science / "cpu_packet_tests.log", 300,
               cwd=src / PACKET):
            raise RuntimeError("packet tests")
        if not re.search(r"\b%d passed\b" % packet["packet_test_count"], (science / "cpu_packet_tests.log").read_text()):
            raise ValueError("packet test inventory incomplete")
        trace_test = work / "test_route_trace"
        if run(["g++", "-std=c++17", "-O1", "-I", str(src), str(src / PACKET / "test_route_trace.cpp"), "-o",
                str(trace_test)], science / "cpu_trace_header_build.log", 300):
            raise RuntimeError("trace header test build")
        if run([str(trace_test)], science / "cpu_trace_header.log", 120):
            raise RuntimeError("trace header test")

        # Retained device checks: handcrafted records, and the original fallback API gate under both partitions.
        if run([str(rows_probe)], science / "handcrafted.json", 120, stderr_log=science / "handcrafted.stderr.log"):
            raise RuntimeError("handcrafted record cases")
        if json.loads((science / "handcrafted.json").read_text()).get("pass") is not True:
            raise ValueError("handcrafted record cases failed")
        fallback = src / "experiments/spec_r1_2026-10-28/decode_pick_fallback.py"
        retained = {}
        for parts in ("0", "4"):
            dest = science / ("fallback_parts" + parts)
            if run([sys.executable, str(fallback), str(server), str(upload / "model.gguf"), str(dest)],
                   science / ("fallback_parts" + parts + ".log"), 900, clean_env(EMBER_DECODE_PICK_PARTS=parts)):
                raise RuntimeError("retained fallback gate PARTS" + parts)
            retained["fallback_parts" + parts] = json.loads((dest / "summary.json").read_text())
            if retained["fallback_parts" + parts].get("pass") is not True:
                raise ValueError("retained fallback gate failed PARTS" + parts)
        atomic_json(science / "retained.json", retained)

        matrix_tool = src / PACKET / "parts4_route_matrix.py"
        if run([sys.executable, str(matrix_tool), "run", str(server), str(upload / "model.gguf"),
                str(science / "matrix")], science / "matrix.log", 1800) not in (0, 1):
            raise RuntimeError("route matrix execution")

        gate = load(src / PACKET / "parts4_gate.py", "parts4_gate_frozen")
        plan = gate.plan(server, upload / "model.gguf", upload / "eval100.jsonl", science / "gate", 8080)
        if len(plan["request_keys"]) != packet["input_pins"]["requests"]:
            raise ValueError("request inventory")
        if [r["name"] for r in plan["runs"]] != packet["schedule"]:
            raise ValueError("schedule drift")
        registered = gate.registered_gate()
        for leg in plan["runs"]:
            log = science / ("leg_" + leg["name"] + ".log")
            rc = run(leg["command"], log, 1800)
            if rc:
                raise RuntimeError("leg execution " + leg["name"])
            # The registered predicates on the fixed inventory, at once; never a retry.
            registered.load_run(science / "gate", leg, plan)
            summary["legs"].append({"name": leg["name"], "returncode": rc, "log_sha256": sha(log)})
            atomic_json(science / "completed_legs.json", summary["legs"])
        result = gate.analyze(science / "gate", science / "matrix")
        atomic_json(science / "gate_summary.json", result)
        for name, rec in inputs["files"].items():
            if sha(upload / name) != rec["sha256"]:
                raise ValueError("final input drift " + name)
        if sha(server) != binaries["tinyllama_server_sha256"]:
            raise ValueError("final binary drift")
        summary.update(status=result["status"], complete=True, gate=result)
    except BaseException as exc:
        summary.update(status="ERROR_OR_INCOMPLETE", error=repr(exc), complete=False)
        (science / "exception.txt").write_text(traceback.format_exc())
        with (out / "vm.log").open("a") as f:
            f.write(repr(exc) + "\n")
    finally:
        stop.set()
        thread.join(timeout=5)
        pack()
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
