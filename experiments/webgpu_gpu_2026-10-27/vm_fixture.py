"""ember CPU's /chat replies (the EXACT reference of the page test) on the VM, from the same ember tree as the page's
WebAssembly: tinyllama_server (CPU build, one OpenMP thread) per model, the 40 greedy requests, return_ids.
    python3 vm_fixture.py SERVER WWW MODEL.gguf PROMPTS OUT.json [--port 8099] [--n 40] [--max-new 60]
PROMPTS: tinychat_hf_prompts_v1.jsonl ({"prompt"} rows: sent as page_test.html's greedy request) or a .jsonl of
{"i", "request"} rows (sent as they are). OUT: {"prompts": [{i, reply, ids, prompt_ids, stop_id, stop_reason,
tokens}]} (page_test.html's greedy.json), and beside it OUT's .requests.jsonl (the requests sent) and .prompts.json.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("server")
ap.add_argument("www")
ap.add_argument("model")
ap.add_argument("prompts")
ap.add_argument("out")
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new", type=int, default=60)
a = ap.parse_args()
rows = [json.loads(x) for x in open(a.prompts, encoding="utf-8") if x.strip()][: a.n]
if "request" in rows[0]:
    reqs = [r["request"] for r in rows]
else:
    reqs = [{"user_input": r["prompt"], "max_new_tokens": a.max_new, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
            for r in rows]
base = os.path.splitext(a.out)[0]
with open(base + ".requests.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(reqs):
        f.write(json.dumps({"i": i, "request": r}) + "\n")
json.dump([{"user_input": r["user_input"]} for r in reqs], open(base + ".prompts.json", "w", encoding="utf-8"))
env = dict(os.environ, OMP_NUM_THREADS="1")
srv = subprocess.Popen([a.server, a.model, str(a.port), "127.0.0.1", a.www, "0"], env=env,
                       stdout=open(base + ".server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2).read()
            break
        except OSError:
            if srv.poll() is not None:
                raise SystemExit(f"server exited {srv.returncode}")
            time.sleep(1)
    out = []
    for i, r in enumerate(reqs):
        body = dict(r, return_ids=True)
        q = urllib.request.Request(f"http://127.0.0.1:{a.port}/chat", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(q, timeout=900))
        except urllib.error.HTTPError as e:   # a refused request is part of the reference (HTTP 400)
            d = {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"}
        out.append({"i": i, "reply": d.get("reply"), "ids": d.get("ids"), "prompt_ids": d.get("prompt_ids"),
                    "stop_id": d.get("stop_id"), "stop_reason": d.get("stop_reason"), "tokens": d.get("tokens"),
                    "error": d.get("error")})
    json.dump({"model": os.path.basename(a.model), "prompts": out}, open(a.out, "w", encoding="utf-8"))
    print(os.path.basename(a.model), len(out), "replies; tokens", sum(len(o["ids"] or []) for o in out),
          "errors", sum(1 for o in out if o["error"]), flush=True)
finally:
    srv.kill()
