"""tc_smoke_20m_X5 served through e115fd3's tinyllama_server /chat on the CPU (gpu_layers 0, EMBER_GEMM_TF32=0), the
served-template path with the GGUF's embedded Llama 3 vocabulary. Requests: the spot-check's 40 Carry-5 items
(carry5_v1 items 0, 26, ..., 1014) in carry5_eval.py's gold mode (the target user turn with the gold history), its
greedy body (temperature 1, top_k 1, top_p 1, seed 7, return_ids), max_new_tokens 8. Per item:
  - the server's prompt_ids equal to score_in's prompt_ids (the template and tokenizer path);
  - the generated ids equal to paperkiln's continuation up to the server's stop (the stop id included);
  - every sampled position's logits (TINYLLAMA_LOGITS_DUMP, nine significant digits) against ir_logits' decode rows
    at the same positions (WORK/cpu_rows.npy from x5s_gate_cpu.py): bit-equal, and max |d|.
    python3 x5s_chat.py BIN PORT CARRY5.jsonl OUT.json"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

D = "/mnt/c/ml_artifacts/attnres_gate_x5/smoke_20m_X5"
M = os.path.join(D, "tc_smoke_20m_X5.gguf")
WORK = os.path.expanduser("~/x5s/work")
BIN, PORT, CARRY5, OUTP = sys.argv[1:5]
run = os.path.join(WORK, "serve")
os.makedirs(os.path.join(run, "www"), exist_ok=True)
dump = os.path.join(run, "dump.txt")
dl = os.path.join(run, "debugging.log")
if not os.path.lexists(dl):
    os.symlink(os.devnull, dl)
URL = f"http://127.0.0.1:{PORT}"
env = {**os.environ, "EMBER_GEMM_TF32": "0", "TINYLLAMA_LOGITS_DUMP": dump}
env.pop("EMBER_IR", None)
err = open(os.path.join(WORK, "server.log"), "w")
proc = subprocess.Popen([BIN, M, PORT, "127.0.0.1", os.path.join(run, "www"), "0"], cwd=run, env=env,
                        stdout=subprocess.DEVNULL, stderr=err, stdin=subprocess.DEVNULL)
health = None
for _ in range(300):
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

ins = [json.loads(x) for x in open(os.path.join(D, "score_in.jsonl")) if x.strip()]
cpu = np.load(os.path.join(WORK, "cpu_rows.npy"))
items = [json.loads(x) for x in open(CARRY5, encoding="utf-8") if x.strip()][::26][:40]
recs = []
try:
    for k, (it, a) in enumerate(zip(items, ins)):
        tt = it["target_turn"]
        hist = [{"user": t["user"], "assistant": t["gold"]} for t in it["turns"][:tt - 1]]
        body = {"user_input": it["turns"][tt - 1]["user"], "history": hist, "temperature": 1.0, "top_k": 1,
                "top_p": 1.0, "seed": 7, "max_new_tokens": 8, "return_ids": True}
        if os.path.exists(dump):
            os.remove(dump)
        req = urllib.request.Request(URL + "/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            e = json.loads(r.read())
        lines = [x for x in open(dump, "rb").read().split(b"\n") if x.strip()] if os.path.exists(dump) else []
        lg = np.array([np.array(x.split(), dtype=np.float32) for x in lines], dtype=np.float32)
        ids = list(e.get("ids") or [])
        stop = e.get("stop_id")
        gen = ids + ([stop] if stop is not None and stop >= 0 and (not ids or ids[-1] != stop) else [])
        cont = a["continuation_ids"]
        n = min(len(lg), len(cont))
        ref = cpu[k][:n]
        recs.append({"i": k, "id": it["id"], "reply": e["reply"], "stop_reason": e.get("stop_reason"), "stop_id": stop,
                     "ids": ids, "history_turns_used": e.get("history_turns_used"),
                     "prompt_ids_equal": e.get("prompt_ids") == a["prompt_ids"],
                     "ids_equal_continuation_prefix": gen == cont[:len(gen)] and len(gen) > 0,
                     "positions": int(len(lg)), "positions_compared": int(n),
                     "logits_bit_equal": bool(np.array_equal(lg[:n], ref)),
                     "logits_max_abs": float(np.abs(lg[:n].astype(np.float64) - ref).max()) if n else None})
finally:
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
N = len(recs)
rep = {"bin": BIN, "gguf": M, "gpu_layers": 0, "health": health, "requests": N,
       "prompt_ids_equal_score_in": f"{sum(r['prompt_ids_equal'] for r in recs)}/{N}",
       "ids_equal_continuation_prefix": f"{sum(r['ids_equal_continuation_prefix'] for r in recs)}/{N}",
       "stop_reasons": sorted({str(r["stop_reason"]) for r in recs}),
       "positions_compared": int(sum(r["positions_compared"] for r in recs)),
       "logits_bit_equal_ir_logits_decode": f"{sum(r['logits_bit_equal'] for r in recs)}/{N}",
       "logits_max_abs_vs_ir_logits": max((r["logits_max_abs"] or 0.0) for r in recs),
       "items": recs}
rep["pass"] = all(r["prompt_ids_equal"] and r["ids_equal_continuation_prefix"] for r in recs) and N == 40
json.dump(rep, open(OUTP, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rep.items() if k != "items"}, indent=1))
for r in recs[:5]:
    print(r["id"], repr(r["reply"]), r["ids"], r["stop_id"])
sys.exit(0 if rep["pass"] else 1)
