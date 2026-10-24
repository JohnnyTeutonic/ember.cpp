#!/usr/bin/env python3
"""Writes system_set_v1.jsonl: /chat requests with a system message, built from the frozen parity prompts
(tinychat_hf_prompts_v1.jsonl, sha256 recorded below) and four system texts:
  - 40: prompt i (0-39) with system text i % 4 in the "system" field;
  - 8: a leading {"role": "system"} history entry, one exchange (prompt 40 + j and its reference reply),
    then prompt j;
  - 8: "system" and two exchanges (prompts 48-63 with their reference replies), then prompt 8 + j;
  - 4: "system" and six exchanges (prompts 64-87), then prompt 16 + j: long enough that a 256-token
    context with 60 new tokens drops whole exchanges, so the pinned system message is exercised.
Every request is a prompt conversation both engines render the same way (section 3, point 4)."""
import json
import os

from common import FROZEN, HERE, rows, sha256

SYSTEMS = ["You are a kind and patient friend.", "Answer in one short sentence.",
           "You are a helpful assistant. Be warm and encouraging.", "Keep your reply brief and friendly."]
P = rows(FROZEN)


def ex(k):
    return [{"role": "user", "content": P[k]["prompt"]}, {"role": "assistant", "content": P[k]["reference"]}]


out = [{"system": SYSTEMS[i % 4], "user_input": P[i]["prompt"]} for i in range(40)]
out += [{"history": [{"role": "system", "content": SYSTEMS[j % 4]}] + ex(40 + j), "user_input": P[j]["prompt"]}
        for j in range(8)]
out += [{"system": SYSTEMS[(j + 1) % 4], "history": ex(48 + 2 * j) + ex(49 + 2 * j), "user_input": P[8 + j]["prompt"]}
        for j in range(8)]
out += [{"system": SYSTEMS[(j + 2) % 4], "history": sum((ex(64 + 6 * j + t) for t in range(6)), []),
         "user_input": P[16 + j]["prompt"]} for j in range(4)]
path = os.path.join(HERE, "system_set_v1.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"n": len(out), "source": FROZEN, "source_sha256": sha256(FROZEN), "out_sha256": sha256(path)}))
