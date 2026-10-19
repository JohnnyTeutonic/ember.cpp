"""Context-limit parity, ember vs mtstudio /chat (contract point 5 in
LOCAL_LLM_STACK.md section 3; mtstudio de33d142 or later): greedy replies
that run into the context, long histories whose oldest turns must be
dropped, and messages too long for the context (HTTP 400). Compares reply,
stop_reason, tokens and history_turns_used, or status and error text.
Usage: chat_ctx_parity.py EMBER_PORT MTS_PORT CTX [OUT.jsonl]
"""
import json
import sys
import urllib.error
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
FILLER = "and then we talked about the weather, the sea, our families and the small things of the day"


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except ValueError:
            return e.code, {}


def cases(ctx):
    prompts = [json.loads(l)["prompt"] for l in open(PROMPTS, encoding="utf-8")]
    greedy = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}
    words_per_token = 0.8  # rough: padding sizes below are targets, not exact
    out = []
    # Long replies: the message fills most of the context; max_new is large,
    # so a reply that does not stop by itself ends at the context limit.
    for i, frac in enumerate((0.5, 0.7, 0.85, 0.95)):
        pad = " ".join([FILLER] * max(1, int(ctx * frac * words_per_token / len(FILLER.split()))))
        out.append(("long_reply", {"user_input": prompts[i] + " " + pad, "max_new_tokens": 1024, **greedy}))
    for i in range(4):  # short messages, long max_new: the reply itself may reach the limit
        out.append(("long_max_new", {"user_input": prompts[10 + i], "max_new_tokens": 1024, **greedy}))
    # Long histories: n past turns; the oldest whole turns must be dropped.
    for j, n in enumerate((2, 4, 8, 12)):
        hist = [{"user": prompts[20 + (j * 3 + t) % 20], "assistant": FILLER} for t in range(n)]
        out.append((f"history_{n}", {"user_input": prompts[j], "history": hist, "max_new_tokens": 60, **greedy}))
    # Too long for the context on its own: HTTP 400.
    out.append(("oversize", {"user_input": " ".join([FILLER] * (ctx // 10 + 2)), "max_new_tokens": 10, **greedy}))
    return out


def main():
    ember, mts, ctx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    recs, agree = [], 0
    for name, body in cases(ctx):
        se, e = post(ember, body)
        sm, m = post(mts, body)
        if se == 200 and sm == 200:
            keys = ("reply", "stop_reason", "tokens", "history_turns_used")
            same = all(e.get(k) == m.get(k) for k in keys)
            detail = ", ".join(f"{k} {e.get(k)!r:.40}/{m.get(k)!r:.40}" for k in keys[1:])
        else:
            same = se == sm and e.get("error") == m.get("error")
            detail = f"status {se}/{sm}, error {e.get('error')!r} / {m.get('error')!r}"
        agree += same
        print(f"{'same' if same else 'DIFF'}  {name:13s} ember/mts: {detail}"
              + (f"; prompt_tokens {e.get('prompt_tokens')}" if se == 200 else ""))
        if not same and se == 200 and sm == 200:
            print(f"      ember {e.get('reply', '')[:100]!r}\n      mts   {m.get('reply', '')[:100]!r}")
        recs.append({"case": name, "ember_status": se, "mts_status": sm, "ember": e, "mtstudio": m, "same": same})
    print(f"n={len(recs)}: identical {agree}")
    if len(sys.argv) > 4:
        with open(sys.argv[4], "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
