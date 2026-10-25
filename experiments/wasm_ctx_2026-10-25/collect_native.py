"""ember's native CPU server on a request file: the references tools/wasm_chat_check.mjs compares the page's
WebAssembly chat path with (run in WSL, under the CPU budget).

    python3 collect_native.py SERVER GGUF REQUESTS.jsonl OUT.jsonl [KEY=VALUE ...]

Starts SERVER (tinyllama_server, CPU, no GPU layers) on GGUF with the KEY=VALUE environment (e.g.
EMBER_IR_CTX=2048), posts each row's request to /chat with return_ids, and writes {i, request, ember}
rows, `ember` being the server's reply object less its timings; the request is stored as sent, less
return_ids, so the page gets the same body."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

server, gguf, reqs, out = sys.argv[1:5]
extra = dict(kv.split("=", 1) for kv in sys.argv[5:])
WWW = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/www"
env = {k: v for k, v in os.environ.items() if not k.startswith("EMBER_")}
env.update(extra)
s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
log = open(out + ".server.log", "w")
srv = subprocess.Popen([server, gguf, str(port), "127.0.0.1", WWW, "0"], stdout=log, stderr=subprocess.STDOUT,
                       env=env, cwd=os.path.dirname(out) or ".")
try:
    for _ in range(1200):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)):
                break
        except Exception:
            if srv.poll() is not None:
                raise SystemExit("server exited: see " + out + ".server.log")
            time.sleep(0.5)
    health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5))
    rows = []
    for line in open(reqs, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        body = dict(r["request"], return_ids=True)
        q = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        e = json.load(urllib.request.urlopen(q, timeout=3600))
        e.pop("ttft", None)
        rows.append({"i": r["i"], "request": r["request"], "ember": e})
        print(r["i"], e["stop_reason"], e["tokens"], e["prompt_tokens"], e["history_turns_used"],
              json.dumps(e["reply"][:80]), flush=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    print(json.dumps({"rows": len(rows), "context": health.get("context"), "env": extra}))
finally:
    srv.terminate()
    srv.wait()
