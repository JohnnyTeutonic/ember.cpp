"""state4's /chat-equivalent requests for the page's chat path (run in WSL).

    python3 chat_state4.py requests OUT.jsonl     the 40 requests of ../state_tokens_2026-10-24/state4_gate.py's
                                                  /chat leg (history and message parsed from prompts_text.jsonl,
                                                  greedy, max_new_tokens min(len(continuation) + 2, 128 - len(prompt),
                                                  64), 64 being the page's cap)
    python3 chat_state4.py native REF.jsonl      the native server's replies (collect_native.py) against
                                                  score_in.jsonl: prompt ids equal, reply ids + stop id equal to
                                                  the continuation; a JSON line on stdout"""
import json
import re
import sys

D = "/mnt/c/ml_artifacts/state_gate/state4"


def rows(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def parse(text):  # state4_gate.py's parse
    parts = [p for p in re.split(r"(<\|user\|>|<\|assistant\|>|<\|endoftext\|>)", text) if p != ""]
    hist, user, reply, cur = [], None, None, None
    for p in parts:
        if p == "<|user|>":
            if user is not None and reply is not None:
                hist.append({"user": user, "assistant": reply})
            user, reply, cur = "", None, "u"
        elif p == "<|assistant|>":
            reply, cur = "", "a"
        elif p == "<|endoftext|>":
            cur = None
        elif cur == "u":
            user += p
        elif cur == "a":
            reply += p
    assert reply == "", text
    return hist, user


ins = rows(D + "/score_in.jsonl")
if sys.argv[1] == "requests":
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for i, (p, a) in enumerate(zip(rows(D + "/prompts_text.jsonl"), ins)):
            hist, user = parse(p["prompt"])
            mx = min(len(a["continuation_ids"]) + 2, 128 - len(a["prompt_ids"]), 64)
            req = {"user_input": user, "history": hist, "max_new_tokens": mx, "temperature": 1.0, "top_k": 1,
                   "top_p": 1.0, "seed": 0}
            f.write(json.dumps({"i": i, "request": req}, ensure_ascii=False) + "\n")
else:
    ref = rows(sys.argv[2])
    pid = sum(r["ember"]["prompt_ids"] == a["prompt_ids"] for r, a in zip(ref, ins))
    rep = sum(r["ember"]["ids"] + ([r["ember"]["stop_id"]] if r["ember"]["stop_id"] >= 0 else []) == a["continuation_ids"]
              for r, a in zip(ref, ins))
    print(json.dumps({"native_prompt_ids_equal_score_in": f"{pid}/{len(ins)}",
                      "native_reply_and_stop_equal_continuation": f"{rep}/{len(ins)}"}))
