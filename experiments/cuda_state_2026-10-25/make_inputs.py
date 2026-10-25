"""Inputs of tools/state_device_gate for the CUDA leg of the state-token and llama_attnres gates (run in WSL):
  state4_in.json    paperkiln's state4 (C:/ml_artifacts/state_gate/state4): the 40 score_in sequences (prompt +
                    numpy's greedy continuation), their prompt lengths, user id 2 (turn-by-turn calls, rewinds into
                    turns 0 and 1) and a session script in the shape of the synthetic CPU gate's (c) (generation,
                    a regenerated reply, an edited earlier message and a rewind into turn 0), sized for ctx 128;
  ar3_in.json       paperkiln's ar3 (C:/ml_artifacts/attnres_gate/ar3): its 40 score_in sequences and prompt lengths;
  synth_in.json     the synthetic CPU gate's own session script (state_tokens_gate.py (c), evict_script.json beside
                    chat_state_k4.gguf in ~/ember_wt/state_tokens), for that model.
Usage: python3 make_inputs.py OUT_DIR"""
import copy
import json
import os
import sys

import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def rows(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def seqs_of(d):
    ins = rows(os.path.join(d, "score_in.jsonl"))
    return [r["prompt_ids"] + r["continuation_ids"] for r in ins], [len(r["prompt_ids"]) for r in ins]


# state4's session script: the CPU gate's (c) structure with vocab 400 and shorter turns (ctx 128).
V, K, EOT, USER, ASST = 400, 4, 1, 2, 3
S = [V + j for j in range(K)]
rng = np.random.default_rng(7)


def text(n):
    return [int(x) for x in rng.integers(4, V, n)]


def exchange(u, a, closed=True):
    return [USER] + text(u) + [ASST] + text(a) + ([EOT] + S if closed else [])


t = [exchange(4, 5), exchange(5, 6), exchange(5, 4), exchange(3, 5), exchange(4, 3)]
conv = sum(t, [])
cut = lambda n: sum(t[:n], [])  # noqa: E731
calls = []
p0 = t[0][:t[0].index(ASST) + 1]
calls.append({"seq": p0, "from": 0})
calls.append({"seq": cut(1), "from": len(p0), "step": True})
for n in (1, 2):
    pn = cut(n) + t[n][:t[n].index(ASST) + 1]
    calls.append({"seq": pn, "from": len(cut(n))})
    calls.append({"seq": cut(n + 1), "from": len(pn), "step": True})
regen = cut(2) + t[2][:t[2].index(ASST) + 1] + text(3) + [EOT] + S
calls.append({"seq": regen, "from": len(cut(2)) + t[2].index(ASST) + 1, "step": True})
edited1 = copy.deepcopy(t[1])
edited1[4] = (edited1[4] + 1) % (V - 4) + 4
seq_e = cut(1) + edited1 + t[2] + [USER] + text(5) + [ASST]
calls.append({"seq": seq_e, "from": len(cut(1)) + 4})
calls.append({"seq": seq_e + text(5) + [EOT] + S, "from": len(seq_e), "step": True})
calls.append({"seq": conv + [USER] + text(4) + [ASST], "from": 0})
seq0 = t[0][:3] + text(5) + [ASST] + text(9) + [EOT] + S + t[1] + [USER] + text(3) + [ASST]
calls.append({"seq": seq0, "from": 3})
calls.append({"seq": seq0 + text(4), "from": len(seq0), "step": True})
assert max(len(c["seq"]) for c in calls) <= 128

s4, p4 = seqs_of("/mnt/c/ml_artifacts/state_gate/state4")
json.dump({"seqs": s4, "prompt_len": p4, "user_id": USER, "calls": calls}, open(os.path.join(out, "state4_in.json"), "w"))
json.dump({"seqs": s4, "prompt_len": p4, "user_id": USER}, open(os.path.join(out, "state4q8_in.json"), "w"))
a3, pa = seqs_of("/mnt/c/ml_artifacts/attnres_gate/ar3")
json.dump({"seqs": a3, "prompt_len": pa}, open(os.path.join(out, "ar3_in.json"), "w"))
synth = json.load(open(os.path.expanduser("~/ember_wt/state_tokens/evict_script.json")))
json.dump({"calls": synth["calls"]}, open(os.path.join(out, "synth_in.json"), "w"))
print(json.dumps({"state4": len(s4), "state4_calls": len(calls), "ar3": len(a3), "synth_calls": len(synth["calls"]),
                  "longest_state4_call": max(len(c["seq"]) for c in calls)}))
