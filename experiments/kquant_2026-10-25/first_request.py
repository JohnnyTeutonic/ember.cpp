#!/usr/bin/env python3
"""The first-request fix (9238a86), checked on a large Q8_0 file: ember's CPU prefill LM head used the Q8_0
kernel on a server's first request (before any decode step had made the f32 copy) and the f32 copy on every
later one. For prompt i of the 40 frozen prompts, three runs, each a fresh server, first-token logits dumped:
  before_first   parent build (bc94c04 with only 9238a86's nine-digit dump), prompt i as the first request
  before_later   parent build, prompt i after one warm-up request (prompt 39, 60 tokens)
  after_first    9238a86, prompt i as the first request
and the greedy reply of each. The fix is right if after_first = before_later, bit for bit.
    first_request.py GGUF OUT.json [i ...]     (default i = 0)"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
BUILDS = {"before": os.path.expanduser("~/kq2/par/tinyllama_server"), "after": os.path.expanduser("~/kq2/new/tinyllama_server")}
WORK = os.path.expanduser("~/kq2/first")
PORT = 8131
gguf, out = sys.argv[1], sys.argv[2]
idx = [int(x) for x in sys.argv[3:]] or [0]
ps = [json.loads(l)["prompt"] for l in open(PROMPTS, encoding="utf-8")][:40]
os.makedirs(WORK, exist_ok=True)


def call(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def greedy(p, n=60):
    return {"user_input": p, "max_new_tokens": n, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}


def run(build, i, warm):
    dump = f"{WORK}/{build}_{i}_{warm}.logits"
    if os.path.exists(dump):
        os.remove(dump)
    env = dict(os.environ, TINYLLAMA_LOGITS_DUMP=dump)
    p = subprocess.Popen([BUILDS[build], gguf, str(PORT), "127.0.0.1", "-", "0"], env=env, cwd=WORK,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        skip = 0
        if warm:
            call(greedy(ps[39]))
            skip = sum(1 for l in open(dump) if l.strip())  # the warm-up's dumped positions
        r = call(greedy(ps[i]))
    finally:
        p.terminate()
        p.wait(30)
    lines = [l for l in open(dump) if l.strip()]
    return r, np.array(lines[skip].split(), dtype=np.float32)  # the prompt's first-token logits


res = {}
for i in idx:
    (rb, lb), (rl, ll), (ra, la) = run("before", i, False), run("before", i, True), run("after", i, False)
    bits = lambda a, b: int((a.view(np.uint32) == b.view(np.uint32)).sum())
    res[i] = {"after_first_eq_before_later_bits": f"{bits(la, ll)}/{la.size}",
              "before_first_eq_before_later_bits": f"{bits(lb, ll)}/{la.size}",
              "max_abs_before_first_vs_later": float(np.abs(lb.astype(np.float64) - ll).max()),
              "reply_before_first": rb["reply"], "reply_before_later": rl["reply"], "reply_after_first": ra["reply"]}
    print(i, json.dumps(res[i]), flush=True)
json.dump({"gguf": gguf, "prompts": res}, open(out, "w"), indent=1)
