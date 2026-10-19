"""OpenAI-compatible endpoints against /chat on the same server.

For the first N eval prompts, with a 1-turn history and the same seed and
sampling: the /v1/chat/completions reply must equal /chat's; the streamed
/v1 deltas must join to that reply; usage and finish_reason must be
consistent. Also /v1/models, error shapes, and the openai SDK if installed.
Usage: openai_check.py PORT [N]
"""
import json
import sys
import urllib.error
import urllib.request

port = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BASE = f"http://127.0.0.1:{port}"
rows = [json.loads(l) for l in open("/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl", encoding="utf-8")][:N + 1]


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


same = joined_ok = usage_ok = 0
for i in range(N):
    prev = rows[i + 1]
    samp = {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "seed": 300 + i}
    _, _, b = post("/chat", {"user_input": rows[i]["prompt"], "max_new_tokens": 64,
                             "history": [{"role": "user", "content": prev["prompt"]},
                                         {"role": "assistant", "content": prev["reference"]}], **samp})
    chat = json.loads(b)
    msgs = [{"role": "system", "content": "ignored"},
            {"role": "user", "content": prev["prompt"]}, {"role": "assistant", "content": prev["reference"]},
            {"role": "user", "content": rows[i]["prompt"]}]
    st, _, b = post("/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 64, **samp})
    oa = json.loads(b)
    reply = oa["choices"][0]["message"]["content"]
    same += reply == chat["reply"]
    u = oa["usage"]
    usage_ok += (u["completion_tokens"] == chat["tokens"] and u["prompt_tokens"] == chat["prompt_tokens"]
                 and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
                 and oa["choices"][0]["finish_reason"] == ("length" if chat["stop_reason"] == "length" else "stop"))
    st, ct, b = post("/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 64, "stream": True,
                                              "stream_options": {"include_usage": True}, **samp})
    events = [e[6:] for e in b.decode("utf-8").split("\n\n") if e.startswith("data: ")]
    assert ct.startswith("text/event-stream") and events[-1] == "[DONE]", (ct, events[-1:])
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    joined_ok += text.rstrip() == reply and finish == [oa["choices"][0]["finish_reason"]] and "usage" in chunks[-1]
    if reply != chat["reply"] or text.rstrip() != reply:
        print(f"[{i}] /chat {chat['reply']!r}\n     /v1  {reply!r}\n     strm {text!r}")
print(f"n={N}: /v1 reply == /chat reply {same}; usage+finish consistent {usage_ok}; stream joins to reply {joined_ok}")
with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
    print("/v1/models:", json.load(r)["data"][0]["id"])
for bad, why in [({"messages": []}, "empty messages"),
                 ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]}, "last=assistant"),
                 ({"nope": 1}, "no messages")]:
    st, _, b = post("/v1/chat/completions", bad)
    print(f"error ({why}): {st} {json.loads(b)['error']['type']}")
try:
    from openai import OpenAI
    c = OpenAI(base_url=BASE + "/v1", api_key="unused")
    r = c.chat.completions.create(model="x", messages=[{"role": "user", "content": "Hi! How are you today?"}],
                                  max_tokens=40, seed=1)
    print("openai SDK:", repr(r.choices[0].message.content), r.usage.total_tokens)
    s = c.chat.completions.create(model="x", messages=[{"role": "user", "content": "Hi! How are you today?"}],
                                  max_tokens=40, seed=1, stream=True)
    print("openai SDK stream:", repr("".join(ch.choices[0].delta.content or "" for ch in s if ch.choices)))
except ImportError:
    print("openai SDK not installed: skipped")
