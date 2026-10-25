"""Replies of every VM run against the CPU build's (records only), per model:

    python3 vs_cpu.py VM_OUT CPU_WORK OUT.json

For each run under VM_OUT/plain (head, new off, new on): requests whose reply, stop and ids equal the CPU's, per
set, and the first differing requests with the first differing generated position. A Carry-5 self turn whose
reply differs changes the next turn's history, so later turns of that item compare different prompts."""
import json
import os
import sys

VM, CPU, OUT = sys.argv[1:4]


def recs(d):
    return [json.loads(x) for x in open(os.path.join(d, "records.jsonl"), encoding="utf-8") if x.strip()]


rep = {}
for run in sorted(os.listdir(os.path.join(VM, "plain"))):
    model = run.split("_")[0]
    cdir = os.path.join(CPU, "plain", model + "_cpu")
    if not os.path.exists(os.path.join(cdir, "records.jsonl")):
        continue
    a, b = recs(cdir), recs(os.path.join(VM, "plain", run))
    assert [r["key"] for r in a] == [r["key"] for r in b]
    d = {"equal": 0, "n": len(a), "tinychat_equal": 0, "carry5_equal": 0, "carry5_same_prompt_differs": 0,
         "first_divergences": []}
    for x, y in zip(a, b):
        same = x["reply"] == y["reply"] and x["stop"] == y["stop"] and x["ids"] == y["ids"]
        d["equal"] += same
        d[x["set"] + "_equal"] += same
        if not same:
            d.setdefault("differing_keys", []).append(x["key"])
        if not same:
            if x["set"] == "carry5" and x["prompt_ids"] == y["prompt_ids"]:
                d["carry5_same_prompt_differs"] += 1
            if x["prompt_ids"] == y["prompt_ids"] and len(d["first_divergences"]) < 10:
                k = next((i for i, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q), min(len(x["ids"]), len(y["ids"])))
                d["first_divergences"].append({"key": x["key"], "position": k, "cpu": x["reply"][:60], "cuda": y["reply"][:60]})
    rep[run] = d
json.dump(rep, open(OUT, "w"), indent=1)
for k, v in rep.items():
    print(k, v["equal"], "/", v["n"], "same-prompt Carry-5 differing:", v["carry5_same_prompt_differs"],
          [f"{e['key']}@{e['position']}" for e in v["first_divergences"]])
