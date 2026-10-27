"""Greedy /chat replies with exact ids from a running tinyllama_server (the parity request of
tools/chat_parity.py: 60 new tokens, temperature 1, top_k 1, top_p 1, seed 0), one jsonl row per
prompt: i, prompt, reply, tokens, stop_reason, prompt_tokens, prompt_ids, ids, stop_id, health.

    python3 e22_chat_client.py PORT PROMPTS.jsonl N OUT.jsonl EXPECT_MODEL_SUBSTRING
"""
import json
import sys
import urllib.request

port, prompts, n, out, expect = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30))
if expect not in str(health.get("model", "")):
    sys.exit(f"wrong model on {port}: {health.get('model')}")
rows = [json.loads(line) for line in open(prompts, encoding="utf-8")][:n]
with open(out + ".tmp", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        body = {"user_input": r["prompt"], "max_new_tokens": 60, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
                "seed": 0, "return_ids": True}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        f.write(json.dumps({"i": i, "prompt": r["prompt"], "reply": d["reply"], "tokens": d["tokens"],
                            "stop_reason": d["stop_reason"], "prompt_tokens": d.get("prompt_tokens"),
                            "prompt_ids": d["prompt_ids"], "ids": d["ids"], "stop_id": d["stop_id"]},
                           ensure_ascii=False) + "\n")
        print(i, d["tokens"], d["stop_reason"], flush=True)
with open(out + ".health.json", "w") as f:
    json.dump(health, f, indent=1)
import os  # noqa: E402

os.replace(out + ".tmp", out)
