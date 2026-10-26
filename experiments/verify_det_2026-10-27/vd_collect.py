"""Greedy /chat replies and every sampled row's logits from one ember tinyllama_server (verify_det_2026-10-27).

    python3 vd_collect.py BIN GGUF GPU_LAYERS PORT PROMPTS.jsonl OUTDIR [NAME=VALUE ...]

Starts the server with TINYLLAMA_LOGITS_DUMP and the given environment additions (e.g. EMBER_DETERMINISTIC=1
EMBER_SPEC=4) and sends every request of PROMPTS (vf_prompts.py's all60.jsonl) as vf_e2e.py does: greedy
(temperature 1, top_k 1, top_p 1, seed 0, return_ids), one request at a time, each twice; the second is the record.
The dump holds one full logit row per generated token: the prefill row, then every decode step's row, or under
EMBER_SPEC every verification row a token is taken from (api.cpp fetches it from the device for the dump; the
token is still the device record's pick). Nine significant digits, so each value reads back as the same float32.
OUTDIR/records.jsonl: per request the reply, ids, stop id, tokens, spec counters, the number of dumped rows, the
dump text's sha256 and row0 (its first row in OUTDIR/rows.f32, float32, vocab values per row)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

BIN, GGUF, GPUL, PORT, PROMPTS, OUT = sys.argv[1:7]
EXTRA = sys.argv[7:]
work = os.path.join(OUT, "run")
os.makedirs(os.path.join(work, "www"), exist_ok=True)
dump = os.path.join(work, "dump.txt")
dl = os.path.join(work, "debugging.log")
if not os.path.lexists(dl):
    os.symlink(os.devnull, dl)
env = dict(os.environ)
for kv in EXTRA:
    k, v = kv.split("=", 1)
    env[k] = v
env["TINYLLAMA_LOGITS_DUMP"] = dump
URL = f"http://127.0.0.1:{PORT}"
err = open(os.path.join(OUT, "server.log"), "w")
proc = subprocess.Popen([BIN, GGUF, PORT, "127.0.0.1", os.path.join(work, "www"), GPUL], cwd=work, env=env,
                        stdout=subprocess.DEVNULL, stderr=err, stdin=subprocess.DEVNULL)


def post(body):
    req = urllib.request.Request(URL + "/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


t0 = time.time()
recs, rows_total, vocab = [], 0, None
rows_path = os.path.join(OUT, "rows.f32")
try:
    health = None
    for _ in range(900):
        try:
            health = json.loads(urllib.request.urlopen(URL + "/health", timeout=2).read())
            break
        except OSError:
            if proc.poll() is not None:
                sys.exit("server exited before serving")
            time.sleep(1)
    if health is None:
        sys.exit("server did not come up")
    prompts = [json.loads(x) for x in open(PROMPTS, encoding="utf-8") if x.strip()]
    with open(rows_path, "wb") as fr:
        for i, r in enumerate(prompts):
            body = {"user_input": r["user_input"], "history": r.get("history", []), "temperature": 1.0, "top_k": 1,
                    "top_p": 1.0, "seed": 0, "max_new_tokens": r.get("max_new_tokens", 60), "return_ids": True}
            first = post(body)
            if os.path.exists(dump):
                os.remove(dump)
            e = post(body)
            text = open(dump, "rb").read() if os.path.exists(dump) else b""
            lines = [x for x in text.split(b"\n") if x.strip()]
            for x in lines:
                a = np.array(x.split(), dtype=np.float32)
                if vocab is None:
                    vocab = int(a.size)
                if a.size != vocab:
                    sys.exit(f"request {i}: a row of {a.size} values (vocab {vocab})")
                a.tofile(fr)
            tt = e.get("ttft", {})
            recs.append({"i": i, "set": r.get("set"), "reply": e["reply"], "ids": e.get("ids"),
                         "stop_id": e.get("stop_id"), "tokens": e.get("tokens"),
                         "drafted": tt.get("spec_drafted", 0), "accepted": tt.get("spec_accepted", 0),
                         "passes": tt.get("spec_passes", 0), "rows": len(lines), "row0": rows_total,
                         "dump_sha256": hashlib.sha256(text).hexdigest(),
                         "warm_pass_same": first["reply"] == e["reply"] and first.get("ids") == e.get("ids")})
            rows_total += len(lines)
finally:
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
with open(os.path.join(OUT, "records.jsonl"), "w", encoding="utf-8") as f:
    for x in recs:
        f.write(json.dumps(x) + "\n")
if os.path.exists(dump):
    os.remove(dump)
summary = {"requests": len(recs), "rows": rows_total, "vocab": vocab,
           "drafted": sum(x["drafted"] for x in recs), "accepted": sum(x["accepted"] for x in recs),
           "passes": sum(x["passes"] for x in recs), "warm_pass_same": sum(x["warm_pass_same"] for x in recs),
           "seconds": round(time.time() - t0, 1)}
json.dump({"bin": BIN, "gguf": GGUF, "gpu_layers": GPUL, "env": EXTRA, "health": health, "prompts": PROMPTS,
           "summary": summary}, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(json.dumps(summary))
