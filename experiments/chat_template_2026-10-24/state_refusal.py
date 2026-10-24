#!/usr/bin/env python3
"""Task 3: per-turn state tokens stay defined over paperkiln/1's markers; a state-token model whose template
is not paperkiln/1 is refused at load. On paperkiln's state4 (k 4, /mnt/c/ml_artifacts/state_gate/state4),
copies written by add_template_keys.py with a declared paperkiln.chat_template:
  - paperkiln/1 itself: loads; /health reports it ("declared"); /chat prompt ids and reply ids equal the
    original file's (which resolves by its specials) on 5 multi-turn requests;
  - a record with paperkiln/1's markers under another name ("custom/1"): refused at load, naming it;
  - chatml-qwen3/1: its stop tokens are not in state4's vocabulary, so chat mode is off and the file is
    refused at load (state tokens need the paperkiln/1 template).
Writes state_refusal.json beside this file."""
import json
import os
import subprocess

from common import CHECK, HERE, NEW, WORK, ember, greedy, post, stop, health

D = "/mnt/c/ml_artifacts/state_gate/state4"
GGUF = D + "/state4.gguf"
v1 = json.loads(subprocess.run([CHECK, "record", "paperkiln/1"], capture_output=True, text=True, check=True).stdout)
custom = json.loads(v1["paperkiln.chat_template"])
custom["name"] = "custom"
qwen3 = json.loads(subprocess.run([CHECK, "record", "chatml-qwen3/1"], capture_output=True, text=True, check=True).stdout)
variants = {"declared_v1": v1, "custom_name": {"paperkiln.chat_template": json.dumps(custom, separators=(",", ":"))},
            "chatml_qwen3": qwen3}
report = {"model": GGUF, "variants": {}}
def request(prompt):
    """A rendered paperkiln/1 prompt (prompts_text.jsonl) as a /chat request: history and message."""
    segs = prompt.split("<|user|>")[1:]
    hist = []
    for s in segs[:-1]:
        u, a = s.split("<|assistant|>")
        hist.append({"user": u, "assistant": a.replace("<|endoftext|>", "")})
    return {"user_input": segs[-1].replace("<|assistant|>", ""), "history": hist}


texts = [request(json.loads(x)["prompt"]) for x in open(D + "/prompts_text.jsonl", encoding="utf-8")][:5]


def run(gguf, port, tag):
    p, h = ember(NEW, gguf, port, "state_" + tag)
    try:
        out = []
        for body in texts:
            s, b = post(port, greedy(body, 16))
            out.append({"status": s, "prompt_ids": b.get("prompt_ids"), "ids": b.get("ids")})
        return h, out
    finally:
        stop(p)


_, base = run(GGUF, 8151, "original")
port = 8152
for name, keys in variants.items():
    dst = f"{WORK}/state4_{name}.gguf"
    kp = dst + ".keys.json"
    json.dump(keys, open(kp, "w"))
    rw = json.loads(subprocess.run(["python3", os.path.join(HERE, "add_template_keys.py"), GGUF, dst, kp],
                                   capture_output=True, text=True, check=True).stdout)
    row = {"tensor_data_equal": rw["tensor_data_equal"]}
    if name == "declared_v1":
        h, out = run(dst, port, name)
        row.update({"health_chat_template": h.get("chat_template"), "why": h.get("chat_template_why"),
                    "requests": len(out), "identical_to_original": sum(a == b for a, b in zip(out, base)),
                    "statuses": [x["status"] for x in out],
                    "prompt_tokens": [len(x["prompt_ids"] or []) for x in out],
                    "pass": h.get("chat_template_why") == "declared" and out == base and
                    all(x["status"] == 200 for x in out)})
    else:
        log = f"{WORK}/logs/state_{name}.log"
        logdir = f"{WORK}/refusal_{name}"
        os.makedirs(logdir, exist_ok=True)
        r = subprocess.run([NEW + "/tinyllama_server", dst, str(port), "127.0.0.1", "-", "0"], cwd=logdir,
                           capture_output=True, text=True, timeout=600)
        dbg = open(os.path.join(logdir, "debugging.log"), encoding="utf-8", errors="replace").read()
        msg = next((l for l in dbg.splitlines() if "Failed to load model" in l), "")
        row.update({"exit_code": r.returncode, "load_error": msg[msg.find("Failed"):] if msg else None,
                    "pass": r.returncode != 0 and "paperkiln/1 chat template" in msg})
    report["variants"][name] = row
    print(name, json.dumps(row), flush=True)
    port += 1
report["pass"] = all(v["pass"] for v in report["variants"].values())
json.dump(report, open(os.path.join(HERE, "state_refusal.json"), "w"), indent=1)
print("pass", report["pass"])
