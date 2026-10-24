#!/usr/bin/env python3
"""G4, declared vs absent (LOCAL_LLM_STACK.md section 3, compatibility gates): a v1 model's GGUF with and
without the template keys serves the same /chat.

- chat_sftfinal_v3.gguf (no template keys: resolves to paperkiln/1 by its specials at ids 0-3) and a copy
  with paperkiln.chat_template and tokenizer.chat_template added as paperkiln's export writes them
  (chat_template_check record paperkiln/1; add_template_keys.py, tensor data byte-identical).
- Four servers: this change on the original and on the copy, and ember main 69d7ca2 (the pre-change
  engine of the tiny-chat campaigns, build_base.sh) on the original and on the copy (a new export served
  by the pinned engine). /health of the first two reports the template.
- The 40 frozen parity prompts, greedy (top_k 1), 60 new tokens, return_ids: prompt ids, reply ids,
  stop id, reply text, stop reason and token count must be identical across the four, 40/40.
Also writes the system-mode-none copy that conformance_live.py serves. Writes g4_declared.json beside
this file."""
import json
import os
import subprocess

from common import BASE, CHECK, FROZEN, HERE, NEW, SFT, WORK, ember, git_rev, greedy, post, rows, sha256, stop, ROOT


def copy_with(record_args, dst):
    keys = subprocess.run([CHECK, "record", *record_args], capture_output=True, text=True, check=True).stdout
    kp = dst + ".keys.json"
    open(kp, "w", encoding="utf-8").write(keys)
    r = subprocess.run(["python3", os.path.join(HERE, "add_template_keys.py"), SFT, dst, kp], capture_output=True,
                       text=True, check=True)
    return json.loads(keys), json.loads(r.stdout)


declared = WORK + "/chat_sftfinal_v3_declared.gguf"
keys, rewrite = copy_with(["paperkiln/1"], declared)
_, rewrite_none = copy_with(["paperkiln/1", "none"], WORK + "/chat_sftfinal_v3_sysnone.gguf")
prompts = rows(FROZEN, 40)
report = {"model": SFT, "model_sha256": sha256(SFT), "copy_sha256": sha256(declared), "keys": keys,
          "rewrite": rewrite, "rewrite_none": rewrite_none, "prompts": FROZEN, "prompts_sha256": sha256(FROZEN),
          "ember_new": git_rev(ROOT + "/ember.cpp") + " + working tree (this change)", "ember_base": "69d7ca2"}
servers = {"new_original": (NEW, SFT, 8111), "new_declared": (NEW, declared, 8112), "base_original": (BASE, SFT, 8113),
           "base_declared": (BASE, declared, 8114)}
procs, out = [], {}
try:
    for tag, (b, g, port) in servers.items():
        p, h = ember(b, g, port, "g4_" + tag)
        procs.append(p)
        report.setdefault("health", {})[tag] = h
    for tag, (_, _, port) in servers.items():
        out[tag] = []
        for r in prompts:
            status, body = post(port, greedy({"user_input": r["prompt"]}))
            assert status == 200, (tag, status, body)
            out[tag].append({k: body[k] for k in ("prompt_ids", "ids", "stop_id", "reply", "stop_reason", "tokens",
                                                  "prompt_tokens", "history_turns_used")})
finally:
    for p in procs:
        stop(p)
fields = ("prompt_ids", "ids", "stop_id", "reply", "stop_reason", "tokens")
same = {f: sum(len({json.dumps(out[t][i][f]) for t in servers}) == 1 for i in range(len(prompts))) for f in fields}
report["n"] = len(prompts)
report["identical_across_all"] = same
report["stops"] = {s: sum(x["stop_reason"] == s for x in out["new_original"]) for s in ("eos", "turn", "length")}
report["distinct_replies"] = len({x["reply"] for x in out["new_original"]})
report["pass"] = all(v == len(prompts) for v in same.values()) and rewrite["tensor_data_equal"]
with open(os.path.join(HERE, "g4_rows.jsonl"), "w", encoding="utf-8") as f:
    for i, r in enumerate(prompts):
        f.write(json.dumps({"i": i, "prompt": r["prompt"], **{t: out[t][i] for t in servers}}, ensure_ascii=False) + "\n")
print(json.dumps({k: report[k] for k in ("n", "identical_across_all", "stops", "distinct_replies", "pass")}, indent=1))
print(json.dumps(report["health"], indent=1))
json.dump(report, open(os.path.join(HERE, "g4_declared.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
