#!/usr/bin/env python3
"""The schema v2 gate (LOCAL_LLM_STACK.md section 3, paperkiln/2): ember's server against `mtstudio chat` on
paperkiln's v2 gate GGUF and v2 parity prompt set, greedy, then stream = plain on ember (run in WSL; CPU).

    python3 gate_v2.py --ember SERVER_BIN --gguf GATE.gguf --prompts SET.jsonl --mts "CMD ... {port}" --out OUT.json
                       [--n 40] [--max-new 60] [--ember-port 8211] [--mts-port 8212]

--mts is the command that serves mtstudio's v2 /chat on {port} (e.g. "~/pk/build/mtstudio chat SPEC --port {port}").
Each prompt row is a /chat body ({"user_input"?, "system"?, "history"?, "tools"?}), or holds one under "request" or
"chat". Both servers get the body with temperature 1, top_k 1, top_p 1, seed 0 and max_new_tokens (chat_parity.py's
greedy request). Per prompt: reply, tool_calls, stop_reason, tokens and history_turns_used equal (prompt_tokens too
when mtstudio reports it). Then, on ember: /chat streamed (joined pieces, the final object) equals /chat plain, and
the same conversation as an OpenAI messages list gives the same content and tool_calls plain and streamed (content
deltas joined, one whole delta per call), finish_reason "tool_calls" exactly when the reply has calls and did not
stop at the length limit. The gate passes at n/n on every count."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--ember", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--mts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--ember-port", type=int, default=8211)
ap.add_argument("--mts-port", type=int, default=8212)
a = ap.parse_args()
WORK = os.path.expanduser("~/cv2/work")
os.makedirs(WORK, exist_ok=True)
GREEDY = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0, "max_new_tokens": a.max_new}


def up(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def start(argv, port, tag):
    if up(port):
        sys.exit(f"port {port} already answers; refusing to share it")
    cwd = tempfile.mkdtemp(dir=WORK)
    p = subprocess.Popen(argv, stdout=open(os.path.join(WORK, tag + ".log"), "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=cwd)
    for _ in range(1200):
        if p.poll() is not None:
            sys.exit(f"{tag} exited {p.returncode}; see {WORK}/{tag}.log")
        if up(port):
            return p
        time.sleep(0.5)
    p.kill()
    sys.exit(f"{tag} did not come up")


def post(port, path, body, raw=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            b = r.read()
            return r.status, (b if raw else json.loads(b))
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")}


def events(raw):
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        if block.strip():
            d = block[len("data: "):]
            out.append(d if d == "[DONE]" else json.loads(d))
    return out


def openai_messages(b):
    """The /chat body as an OpenAI request: system, history (pairs expanded), user_input; tools as given."""
    msgs = [{"role": "system", "content": b["system"]}] if b.get("system") else []
    for h in b.get("history", []):
        if "role" in h:
            m = {k: v for k, v in h.items() if k in ("role", "content", "tool_calls", "tool_call_id")}
            if m["role"] == "system" and not m.get("content"):
                continue
            msgs.append(m)
        else:
            msgs += [{"role": "user", "content": h.get("user", "")}, {"role": "assistant", "content": h.get("assistant", "")}]
    if "user_input" in b:
        msgs.append({"role": "user", "content": b["user_input"]})
    out = {"messages": msgs, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0, "max_tokens": a.max_new}
    if "tools" in b:
        out["tools"] = b["tools"]
    return out


rows = [json.loads(x) for x in open(a.prompts, encoding="utf-8") if x.strip()][:a.n]
bodies = [r.get("request") or r.get("chat") or r for r in rows]
pe = start([a.ember, a.gguf, str(a.ember_port), "127.0.0.1", "-", "0"], a.ember_port, "gate_ember")
pm = start([os.path.expanduser(x) for x in shlex.split(a.mts.format(port=a.mts_port))], a.mts_port, "gate_mtstudio")
recs = []
counts = {k: 0 for k in ("reply", "tool_calls", "stop_reason", "tokens", "history_turns_used", "prompt_tokens",
                         "chat_stream", "openai_plain", "openai_stream", "finish_reason")}
pt_reported = 0
try:
    for i, b in enumerate(bodies):
        body = {**b, **GREEDY}
        se, e = post(a.ember_port, "/chat", body)
        sm, m = post(a.mts_port, "/chat", body)
        rec = {"i": i, "request": b, "status": [se, sm], "ember": e, "mtstudio": m}
        if se == 200 and sm == 200:
            for k in ("reply", "tool_calls", "stop_reason", "tokens", "history_turns_used"):
                counts[k] += e.get(k) == m.get(k)
            if "prompt_tokens" in m:
                pt_reported += 1
                counts["prompt_tokens"] += e["prompt_tokens"] == m["prompt_tokens"]
            s2, raw = post(a.ember_port, "/chat", {**body, "stream": True}, raw=True)
            evs = events(raw) if s2 == 200 else []
            joined = "".join(x["token"] for x in evs if isinstance(x, dict) and "token" in x)
            final = next((x for x in evs if isinstance(x, dict) and x.get("done")), {})
            ok_stream = joined == e["reply"] and final.get("reply") == e["reply"] and final.get("tool_calls") == e.get("tool_calls")
            counts["chat_stream"] += ok_stream
            ob = openai_messages(b)
            so, o = post(a.ember_port, "/v1/chat/completions", ob)
            msg = o["choices"][0]["message"] if so == 200 else {}
            ok_oa = so == 200 and msg.get("content") == e["reply"] and msg.get("tool_calls") == e.get("tool_calls")
            counts["openai_plain"] += ok_oa
            want_finish = "length" if e["stop_reason"] == "length" else ("tool_calls" if e.get("tool_calls") else "stop")
            s3, oraw = post(a.ember_port, "/v1/chat/completions", {**ob, "stream": True}, raw=True)
            content, calls, finish = "", [], None
            for x in (events(oraw) if s3 == 200 else []):
                if x == "[DONE]" or not x.get("choices"):
                    continue
                d = x["choices"][0]["delta"]
                content += d.get("content") or ""
                calls += [{k: c[k] for k in ("id", "type", "function")} for c in d.get("tool_calls", [])]
                finish = x["choices"][0]["finish_reason"] or finish
            counts["openai_stream"] += content == msg.get("content") and calls == (msg.get("tool_calls") or [])
            counts["finish_reason"] += finish == want_finish and (o["choices"][0]["finish_reason"] if so == 200 else None) == want_finish
            rec.update({"chat_stream_equal": ok_stream, "openai_equal": ok_oa, "finish_reason": finish})
        recs.append(rec)
        print(f"[{i}] {'same' if se == sm == 200 and e.get('reply') == m.get('reply') and e.get('tool_calls') == m.get('tool_calls') else 'DIFF'}",
              flush=True)
finally:
    for p in (pe, pm):
        p.terminate()
        p.wait()
n = len(bodies)
summary = {"n": n, "gguf": a.gguf, "prompts": a.prompts, "ember": a.ember, "mts": a.mts, "counts": counts,
           "mtstudio_reports_prompt_tokens": pt_reported,
           "passed": all(v == n for k, v in counts.items() if k != "prompt_tokens") and counts["prompt_tokens"] == pt_reported}
with open(a.out, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "records": recs}, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(json.dumps(summary))
sys.exit(0 if summary["passed"] else 1)
