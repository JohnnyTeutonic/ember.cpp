"""Rows for tools/wasm_chat_check.mjs from utf8_server_check.py's output: the cut ks (and the ks a character
was split at) within the page's 64-token cap, each with the native /chat request and reply object."""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src, encoding="utf-8"))
s = d["summary"]
want = sorted(set(k for k in s["cut_ks"] + s["split_ks"] + [s["split_ks"][0] - 1 if s["split_ks"] else 1, 64]
                  if k <= 64))
recs = {r["k"]: r for r in d["records"]}
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for k in want:
        r = recs[k]
        req = dict(s["request"], max_new_tokens=k)
        e = {key: r[key] for key in ("reply", "stop_reason", "tokens", "prompt_tokens", "history_turns_used",
                                     "prompt_ids", "ids")}
        f.write(json.dumps({"i": k, "request": req, "ember": e, "vocab": s["health"]["vocab_size"]},
                           ensure_ascii=False) + "\n")
print("ref rows", want)
