"""Actual CPU HTTP invariance check, never a GPU or speed measurement.

Run with CPU-only baseline/candidate servers; all server invocations explicitly
select zero GPU layers. Refuses an existing output directory. Only owned child
servers are terminated. Raw records, dumped logits and logs remain for replay.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def request(url, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as f:
        return json.load(f)


def run(baseline, candidate, model, output):
    out = Path(output).resolve()
    paths = {"baseline": str(Path(baseline).resolve()), "candidate": str(Path(candidate).resolve()),
             "model": str(Path(model).resolve())}
    frozen = {k: sha(v) for k, v in paths.items()}
    out.mkdir(exist_ok=False)
    records, dump_hashes = {}, {}
    for name, pick, parts, deterministic in (("baseline", "0", "0", "0"),
                                              ("candidate_off", "0", "0", "0"),
                                              ("candidate_option", "1", "4", "1")):
        work = out / name
        (work / "www").mkdir(parents=True)
        dump = work / "rows.txt"
        binary = paths["baseline"] if name == "baseline" else paths["candidate"]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   EMBER_DECODE_PICK=pick, EMBER_DECODE_PICK_PARTS=parts,
                   EMBER_DETERMINISTIC=deterministic, EMBER_SPEC="0", EMBER_DECODE_GRAPH="1",
                   TINYLLAMA_LOGITS_DUMP=str(dump))
        for k in ("TINYLLAMA_TOPK_DEBUG", "EMBER_DECODE_TIMING", "EMBER_IR"):
            env.pop(k, None)
        url = f"http://127.0.0.1:{port}"
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen([binary, paths["model"], str(port), "127.0.0.1", str(work / "www"), "0"],
                                       cwd=work, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=log)
            try:
                for _ in range(120):
                    try:
                        health = request(url + "/health")
                        if process.poll() is not None:
                            raise RuntimeError("owned server exited")
                        break
                    except OSError:
                        if process.poll() is not None:
                            raise RuntimeError("owned server exited before health")
                        time.sleep(1)
                else:
                    raise RuntimeError("owned server not ready")
                if health.get("model") != paths["model"]:
                    raise ValueError("wrong server model")
                (work / "health.json").write_text(json.dumps(health, indent=2) + "\n")
                rows = []
                for body in ({"user_input": "What is two plus two?"},
                             {"user_input": "What did I say my name was?", "history": [{"user": "My name is Ada.", "assistant": "Hello Ada."}]}):
                    body = dict(body, temperature=1.0, top_k=1, top_p=1.0, seed=13,
                                return_ids=True, max_new_tokens=12)
                    result = request(url + "/chat", body)
                    keys = ("reply", "ids", "prompt_ids", "stop_id", "tokens", "prompt_tokens", "stop_reason")
                    row = {k: result[k] for k in keys}
                    if not row["prompt_ids"] or not isinstance(row["ids"], list):
                        raise ValueError("missing actual model ids")
                    rows.append(row)
                records[name] = rows
                (work / "requests.json").write_text(json.dumps(rows, indent=2) + "\n")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        raw = dump.read_text().splitlines()
        values = [float(v) for line in raw for v in line.split()]
        if not raw or not values or not all(math.isfinite(v) for v in values):
            raise ValueError("empty or nonfinite model logits")
        packed = struct.pack(f"{len(values)}f", *values)
        (work / "rows.f32").write_bytes(packed)
        dump_hashes[name] = {"sha256": hashlib.sha256(packed).hexdigest(), "rows": len(raw), "values": len(values)}
    if not all(v == records["baseline"] for v in records.values()):
        raise ValueError("CPU HTTP output mismatch")
    if not all(v == dump_hashes["baseline"] for v in dump_hashes.values()):
        raise ValueError("CPU HTTP logit bits mismatch")
    if {k: sha(v) for k, v in paths.items()} != frozen:
        raise ValueError("model or server changed during CPU check")
    result = {"status": "PASS_CPU_ONLY", "paths": paths, "sha256": frozen,
              "runs": list(records), "requests_per_run": 2, "row_evidence": dump_hashes,
              "scope": "CPU HTTP behaviour and finite float32 row identity only; experimental CUDA dispatch was not executed"}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "model", "output"):
        p.add_argument(name)
    a = p.parse_args()
    run(a.baseline, a.candidate, a.model, a.output)
