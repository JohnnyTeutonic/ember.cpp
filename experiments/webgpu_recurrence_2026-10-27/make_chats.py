"""Multi-turn chat requests for the recurrence chat gate (chat_test.mjs .jsonl form {"i", "request"}): 20 requests
over the 40 frozen page prompts (PROMPTS.json, page_test.sh's), request j asking prompt (2j + 1) % 40 after a
history of two or three earlier turns (prompts with kimi_tiny's mtstudio greedy replies, first line, as the
assistant's turns). Greedy (top_k 1, seed 0), 40 new tokens. Each reply re-prefills the whole conversation from
position 0, so the recurrent state of every turn starts again from the initial state (init bodies included).

    python make_chats.py PROMPTS.json MTSTUDIO_GREEDY.jsonl OUT.jsonl
"""
import json
import sys

prompts = [p["user_input"] for p in json.load(open(sys.argv[1], encoding="utf-8"))]
replies = [json.loads(x)["mtstudio"].strip().split("\n")[0].strip() or "ok." for x in open(sys.argv[2], encoding="utf-8") if x.strip()]
with open(sys.argv[3], "w", encoding="utf-8", newline="\n") as f:
    for j in range(20):
        turns = 2 + j % 2
        hist = [{"user": prompts[(2 * j + 3 + 5 * t) % 40], "assistant": replies[(2 * j + 3 + 5 * t) % 40]} for t in range(turns)]
        req = {"user_input": prompts[(2 * j + 1) % 40], "history": hist, "max_new_tokens": 40, "temperature": 1.0,
               "top_k": 1, "top_p": 1.0, "seed": 0}
        f.write(json.dumps({"i": j, "request": req}) + "\n")
print(sys.argv[3], 20, "requests")
