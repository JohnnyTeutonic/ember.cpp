#!/usr/bin/env python3
"""Every row's "prompt" (paperkiln's paperkiln/2 rendering) against ember's request path without the context limit
(`chat_template_check render GGUF`, ctx 2^30), and its ids against mtstudio's prompt_ids in paperkiln's reference.

    python3 prompt_check.py CHECK_BIN GATE.gguf PROMPTS.jsonl MTSTUDIO_GREEDY.jsonl OUT.json"""
import json
import subprocess
import sys

check, gguf, prompts, refp, out = sys.argv[1:6]
rows = [json.loads(x) for x in open(prompts, encoding="utf-8") if x.strip()]
ref = {r["id"]: r for r in (json.loads(x) for x in open(refp, encoding="utf-8") if x.strip())}
inp = "".join(json.dumps({"chat": r["request"], "max_new_tokens": 60}, ensure_ascii=False) + "\n" for r in rows)
res = subprocess.run([check, "render", gguf], input=inp.encode(), capture_output=True)
got = [json.loads(x) for x in res.stdout.decode().splitlines() if x.strip()][1:]
recs, text_ok, ids_ok, n_ids = [], 0, 0, 0
for r, g in zip(rows, got):
    t = g.get("prompt") == r["prompt"]
    m = ref.get(r["id"], {})
    i = None
    if "prompt_ids" in m:
        n_ids += 1
        i = g.get("ids") == m["prompt_ids"]
        ids_ok += i
    text_ok += t
    recs.append({"id": r["id"], "prompt_equal": t, "ids_equal_mtstudio": i, "prompt_tokens": g.get("prompt_tokens"),
                 "error": g.get("error")})
summary = {"rows": len(rows), "prompt_equal": text_ok, "mtstudio_rows_with_ids": n_ids, "ids_equal_mtstudio": ids_ok}
json.dump({"summary": summary, "rows": recs}, open(out, "w", encoding="utf-8"), indent=1)
print(json.dumps(summary))
