"""Logit error of a quantised GGUF against its f32 source.

probe: send the first N eval prompts (greedy, one new token) to a server
started with TINYLLAMA_LOGITS_DUMP=<file>, so the file holds each prompt's
next-token logits, one line per prompt.
compare: max and mean |dlogit| and top-1 agreement of a dump against the
f32 dump from the same backend.
Usage: quant_gate.py probe PORT [N]
       quant_gate.py compare F32_DUMP QUANT_DUMP
"""
import json
import sys
import urllib.request

import numpy as np

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"


def probe(port, n):
    rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")][:n]
    for r in rows:
        body = {"user_input": r["prompt"], "max_new_tokens": 1, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=300).read()


def load(p):
    return np.array([np.array(l.split(), dtype=np.float64) for l in open(p) if l.strip()])


def compare(ref_path, q_path):
    a, b = load(ref_path), load(q_path)
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a - b)
    top1 = int((a.argmax(1) == b.argmax(1)).sum())
    # top-1 margin of the reference: how close the agreements were to flipping
    srt = np.sort(a, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    print(f"logits vs f32 over {len(a)} prompts: max |d| {d.max():.4f}, mean |d| {d.mean():.5f}, "
          f"max |d| / logit range {d.max() / (a.max() - a.min()):.2e}; top-1 agree {top1}/{len(a)} "
          f"(smallest f32 top-1 margin {margin.min():.4f})")


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        probe(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40)
    else:
        compare(sys.argv[2], sys.argv[3])
