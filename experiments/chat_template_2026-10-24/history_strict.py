#!/usr/bin/env python3
"""/chat role-form history pairs strictly (paperkiln 7a6c7659, as mtstudio's tests/test_mtstudio_bpe_chat.py
test_history_pairs_strictly), on this change's CPU server with chat_sftfinal_v3: a well-formed role history
gives the same reply (and prompt ids) as the same turns in pair form, with and without a leading system
entry; the four violations (an assistant without a user before it, two user messages in a row, a history
ending on a user message, an unknown role) and a system entry past the first are each a 400 naming the
rule, in mtstudio's words. Writes history_strict.json beside this file."""
import json
import os

from common import HERE, NEW, SFT, ember, post, stop

base = {"user_input": "hello", "top_k": 1, "max_new_tokens": 12, "temperature": 1.0, "top_p": 1.0, "seed": 0,
        "return_ids": True}
u, a = {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}
s = {"role": "system", "content": "Be kind."}
report = {"model": SFT, "same_as_pair": [], "refused": []}
p, h = ember(NEW, SFT, 8161, "history_strict")
try:
    for role_hist, pair_body in (([u, a], {"history": [{"user": "hi", "assistant": "hello there"}]}),
                                 ([s, u, a], {"system": "Be kind.", "history": [{"user": "hi", "assistant": "hello there"}]}),
                                 ([u, a, u, a], {"history": [{"user": "hi", "assistant": "hello there"}] * 2})):
        s1, r1 = post(8161, dict(base, history=role_hist))
        s2, r2 = post(8161, dict(base, **pair_body))
        same = s1 == s2 == 200 and all(r1[k] == r2[k] for k in ("reply", "prompt_ids", "ids", "stop_reason"))
        report["same_as_pair"].append({"role_history": role_hist, "status": [s1, s2], "reply": r1.get("reply"),
                                       "same": same})
    for hist, rule in (([a, u], "without a user message before it"), ([u, u, a], "two user messages"),
                       ([u, a, u], "ends with a user message"),
                       ([u, {"role": "narrator", "content": "x"}], "role"),
                       ([u, a, s], "first history entry")):
        st, e = post(8161, dict(base, history=hist))
        report["refused"].append({"history": hist, "rule": rule, "status": st, "error": e.get("error"),
                                  "ok": st == 400 and rule in (e.get("error") or "")})
finally:
    stop(p)
report["pass"] = all(x["same"] for x in report["same_as_pair"]) and all(x["ok"] for x in report["refused"])
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(HERE, "history_strict.json"), "w"), indent=1)
