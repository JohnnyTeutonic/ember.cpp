"""Speculative decoding check on an ember chat server: the 40 frozen eval
prompts, greedy, up to 60 new tokens. Writes one JSON line per prompt (reply,
tokens, wall ms, spec counters) and prints totals; `compare` reports reply
agreement and wall time between two runs (EMBER_SPEC off vs on).
Usage: spec_bench.py run PORT OUT.jsonl
       spec_bench.py compare BASE.jsonl SPEC.jsonl
"""
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"


def run(port, out):
    rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")][:40]
    cfg = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "max_new_tokens": 60, "seed": 0}
    recs = []
    for r in rows:
        body = json.dumps({"user_input": r["prompt"], **cfg}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        d = json.load(urllib.request.urlopen(req, timeout=300))
        wall = (time.perf_counter() - t0) * 1000
        tt = d["ttft"]
        recs.append({"reply": d["reply"], "tokens": d["tokens"], "wall_ms": wall,
                     "drafted": tt.get("spec_drafted", 0), "accepted": tt.get("spec_accepted", 0),
                     "passes": tt.get("spec_passes", 0)})
    with open(out, "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    dr, ac = sum(x["drafted"] for x in recs), sum(x["accepted"] for x in recs)
    print(f"n={len(recs)}: median {statistics.median(x['wall_ms'] for x in recs):.1f} ms/reply, "
          f"{sum(x['tokens'] for x in recs)} tokens; drafted {dr}, accepted {ac} "
          f"({100 * ac / dr if dr else 0:.0f}%), passes {sum(x['passes'] for x in recs)}")


def compare(a, b):
    ra = [json.loads(l) for l in open(a, encoding="utf-8")]
    rb = [json.loads(l) for l in open(b, encoding="utf-8")]
    same = sum(x["reply"] == y["reply"] for x, y in zip(ra, rb))
    ta, tb = sum(x["wall_ms"] for x in ra), sum(x["wall_ms"] for x in rb)
    print(f"replies identical {same}/{len(ra)}; total wall {ta:.0f} -> {tb:.0f} ms ({ta / tb:.2f}x)")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3])
