"""The /chat requests of the context-cap gate (greedy: temperature 1, top_k 1, top_p 1, seed 0).

    python3 make_requests.py OUT_DIR

requests.jsonl: 10 single-turn requests from paperkiln's chat_prompts_v1 (the third-party gate's test
prompts): its distinct prompts of at most 600 characters (the page's limit on a message), 10 evenly
spaced in file order; the 4th and 8th carry a system message. 32 new tokens.
long_requests.jsonl: 3 requests for the context rule under a small cap (run at ctx 256): a system
message, 10 past exchanges (the other distinct prompts in file order, each with its reference reply cut
to 300 characters) and a new message, so the oldest exchanges must be dropped and the system message kept.
24 new tokens."""
import json
import os
import sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
src = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/eval/chat_prompts_v1.jsonl"
rows = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
seen, distinct = set(), []
for r in rows:
    if r["prompt"] not in seen and len(r["prompt"]) <= 600:
        seen.add(r["prompt"])
        distinct.append(r)
pick = [round(k * (len(distinct) - 1) / 9) for k in range(10)]
GREEDY = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
SYSTEM = "You are a helpful assistant. Answer briefly."
with open(os.path.join(out, "requests.jsonl"), "w", encoding="utf-8") as f:
    for i, k in enumerate(pick):
        req = {"user_input": distinct[k]["prompt"], "max_new_tokens": 32, **GREEDY}
        if i in (3, 7):
            req["system"] = SYSTEM
        f.write(json.dumps({"i": i, "src_index": k, "request": req}, ensure_ascii=False) + "\n")
rest = [r for j, r in enumerate(distinct) if j not in pick]
with open(os.path.join(out, "long_requests.jsonl"), "w", encoding="utf-8") as f:
    for i in range(3):
        block = rest[i * 11:(i + 1) * 11]
        hist = [{"user": r["prompt"], "assistant": r["reference"][:300]} for r in block[:10]]
        req = {"user_input": block[10]["prompt"], "system": SYSTEM, "history": hist, "max_new_tokens": 24, **GREEDY}
        f.write(json.dumps({"i": i, "request": req}, ensure_ascii=False) + "\n")
print(json.dumps({"distinct": len(distinct), "picked": pick}))
