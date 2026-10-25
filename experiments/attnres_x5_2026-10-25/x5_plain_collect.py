"""Greedy /chat replies and every sampled position's logits from one ember tinyllama_server (the tiny-chat post
jobs' generation path: EMBER_GEMM_TF32=0, the given gpu_layers), for a bit-for-bit comparison of two builds.

    python3 x5_plain_collect.py BIN GGUF GPU_LAYERS PORT PROMPTS.jsonl CARRY5.jsonl OUTDIR

Requests, in this order, each body carry5_eval.py's (temperature 1, top_k 1, top_p 1, seed 7, return_ids):
  - tinychat: the first 40 prompts of PROMPTS (tinychat_hf_prompts_v1), first turn, max_new_tokens 48;
  - carry5:  the spot-check's 40 Carry-5 items (0, 26, ..., 1014), as carry5_eval.generate runs them:
             the five self turns (each with the replies so far as history), then gold, twin and noctx;
             max_new_tokens 24. 320 requests.
The server runs with TINYLLAMA_LOGITS_DUMP (nine significant digits: each value reads back as the same
float32). Per request: the reply, stop reason, ids, prompt ids, the number of dumped positions and the sha256
of the dump's text (every position's full logit vector); the first sampled position's logits go to
OUTDIR/first_logits.f32 (float32, one row per request). OUTDIR/records.jsonl holds the rest. With X5_KEEP_ALL=1
every sampled position's logits are also appended to OUTDIR/all_logits.f32 (record field "row0": the request's
first row there)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

BIN, GGUF, GPUL, PORT, PROMPTS, CARRY5, OUT = sys.argv[1:8]
os.makedirs(os.path.join(OUT, "run", "www"), exist_ok=True)
dump = os.path.join(OUT, "run", "dump.txt")
dl = os.path.join(OUT, "run", "debugging.log")
if not os.path.lexists(dl):
    os.symlink(os.devnull, dl)
URL = f"http://127.0.0.1:{PORT}"
KEEP = os.environ.get("X5_KEEP_ALL") == "1"
ROWS = 0
if KEEP and os.path.exists(os.path.join(OUT, "all_logits.f32")):
    os.remove(os.path.join(OUT, "all_logits.f32"))
env = {**os.environ, "EMBER_GEMM_TF32": "0", "TINYLLAMA_LOGITS_DUMP": dump}
env.pop("EMBER_IR_DEVICE", None)
err = open(os.path.join(OUT, "server.log"), "w")
proc = subprocess.Popen([BIN, GGUF, PORT, "127.0.0.1", os.path.join(OUT, "run", "www"), GPUL], cwd=os.path.join(OUT, "run"),
                        env=env, stdout=subprocess.DEVNULL, stderr=err, stdin=subprocess.DEVNULL)
health = None
for _ in range(600):
    try:
        health = json.loads(urllib.request.urlopen(URL + "/health", timeout=2).read())
        break
    except OSError:
        if proc.poll() is not None:
            sys.exit("server exited before serving")
        time.sleep(1)
if health is None:
    proc.kill()
    sys.exit("server did not come up")


def chat(history, user, max_new):
    body = {"user_input": user, "history": [{"user": u, "assistant": r} for u, r in history], "temperature": 1.0,
            "top_k": 1, "top_p": 1.0, "seed": 7, "max_new_tokens": max_new, "return_ids": True}
    if os.path.exists(dump):
        os.remove(dump)
    req = urllib.request.Request(URL + "/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        e = json.loads(r.read())
    text = open(dump, "rb").read() if os.path.exists(dump) else b""
    lines = [x for x in text.split(b"\n") if x.strip()]
    first = np.array(lines[0].split(), dtype=np.float32) if lines else np.zeros(0, np.float32)
    row0 = None
    if KEEP:
        global ROWS
        row0 = ROWS
        with open(os.path.join(OUT, "all_logits.f32"), "ab") as fa:
            for x in lines:
                np.array(x.split(), dtype=np.float32).tofile(fa)
        ROWS += len(lines)
    rec = {"reply": e["reply"].strip(), "stop": e.get("stop_reason"), "ids": e.get("ids"), "stop_id": e.get("stop_id"),
           "prompt_ids": e.get("prompt_ids"), "history_used": e.get("history_turns_used"), "positions": len(lines),
           "dump_sha256": hashlib.sha256(text).hexdigest()}
    if KEEP:
        rec["row0"] = row0
    return rec, first


t0 = time.time()
recs, firsts = [], []
try:
    ps = [json.loads(x)["prompt"] for x in open(PROMPTS, encoding="utf-8") if x.strip()][:40]
    for i, p in enumerate(ps):
        r, f = chat([], p, 48)
        recs.append({"set": "tinychat", "key": f"tinychat:{i}", **r})
        firsts.append(f)
    items = [json.loads(x) for x in open(CARRY5, encoding="utf-8") if x.strip()][::26][:40]
    for it in items:
        hist = []
        for k, t in enumerate(it["turns"]):
            r, f = chat(hist, t["user"], 24)
            recs.append({"set": "carry5", "key": f"{it['id']}|self|{k + 1}", **r})
            firsts.append(f)
            hist.append((t["user"], r["reply"]))
        tt = it["target_turn"]
        for mode, (h, u) in (("gold", ([(t["user"], t["gold"]) for t in it["turns"][:tt - 1]], it["turns"][tt - 1]["user"])),
                             ("twin", ([], it["twin"])), ("noctx", ([], it["noctx"]))):
            r, f = chat(h, u, 24)
            recs.append({"set": "carry5", "key": f"{it['id']}|{mode}", **r})
            firsts.append(f)
finally:
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
with open(os.path.join(OUT, "records.jsonl"), "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
np.concatenate(firsts).astype(np.float32).tofile(os.path.join(OUT, "first_logits.f32"))
if os.path.exists(dump):
    os.remove(dump)
json.dump({"bin": BIN, "gguf": GGUF, "gpu_layers": GPUL, "health": health, "requests": len(recs),
           "vocab": int(firsts[0].size), "seconds": round(time.time() - t0, 1)},
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(json.dumps({"requests": len(recs), "seconds": round(time.time() - t0, 1)}))
