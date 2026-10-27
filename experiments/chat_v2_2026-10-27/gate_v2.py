#!/usr/bin/env python3
"""The schema v2 gate (LOCAL_LLM_STACK.md section 3, paperkiln/2): ember's server against `mtstudio chat` on
paperkiln's v2 gate GGUF and v2 parity prompt set, greedy, then stream = plain on ember (run in WSL; CPU).

    python3 gate_v2.py --ember SERVER_BIN --check CHECK_BIN --gguf GATE.gguf --prompts SET.jsonl
                       --mts "CMD ... {port}" --out OUT.json [--ref MTSTUDIO_GREEDY.jsonl]
                       [--n 40] [--max-new 60] [--seed 7] [--ember-port 8211] [--mts-port 8212]

--mts serves mtstudio's v2 /chat on {port} (an "ENV=VALUE" prefix is passed as environment). Each prompt row is
{"id", "request" (the /chat body), "prompt" (paperkiln's rendering)}, or a bare /chat body. Both servers get the
request plus temperature 1, top_k 1, top_p 1, seed, max_new_tokens and return_ids. Per row:
  - mtstudio: HTTP status, and on 200 reply, tool_calls, stop_reason, tokens, history_turns_used, prompt_ids, ids
    and stop_id equal (prompt_tokens too when mtstudio reports it); a 400 must meet a 400;
  - the rendering: `chat_template_check render` (the server's request path, with ember's context and max_new) gives
    the row's "prompt" text, and its ids are ember's prompt_ids;
  - --ref (paperkiln's own mtstudio replies, {"id", ...} or {"id", "error"}): the same fields against ember;
  - stream = plain on ember: /chat streamed (joined pieces, the final object) equals /chat plain, and the conversation
    as an OpenAI messages list gives the same content and tool_calls plain and streamed (one whole delta per call),
    finish_reason "tool_calls" exactly when the reply has calls and did not stop at the length limit.
The gate passes when every count is n/n."""
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
ap.add_argument("--check", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--mts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ref")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--ember-port", type=int, default=8211)
ap.add_argument("--mts-port", type=int, default=8212)
a = ap.parse_args()
WORK = os.path.expanduser("~/cv2/work")
os.makedirs(WORK, exist_ok=True)
GREEDY = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": a.seed, "max_new_tokens": a.max_new, "return_ids": True}
FIELDS = ("reply", "tool_calls", "stop_reason", "tokens", "history_turns_used", "prompt_ids", "ids", "stop_id")


def health(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except urllib.error.HTTPError:
        return {}
    except Exception:
        return None


def start(argv, port, tag, env=None):
    if health(port) is not None:
        sys.exit(f"port {port} already answers; refusing to share it")
    cwd = tempfile.mkdtemp(dir=WORK)
    p = subprocess.Popen(argv, stdout=open(os.path.join(WORK, tag + ".log"), "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=cwd, env={**os.environ, **(env or {})})
    for _ in range(1200):
        if p.poll() is not None:
            sys.exit(f"{tag} exited {p.returncode}; see {WORK}/{tag}.log")
        h = health(port)
        if h is not None:
            return p, h
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
        b = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(b)
        except ValueError:
            return e.code, {"error": b}


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
    out = {"messages": msgs, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": a.seed, "max_tokens": a.max_new}
    if "tools" in b:
        out["tools"] = b["tools"]
    return out


def compare(e_status, e, m_status, m):
    """Per field: True when equal (both 400: every field True)."""
    if e_status != 200 or m_status != 200:
        ok = e_status == m_status == 400
        return {k: ok for k in FIELDS}, ok
    eq = {k: e.get(k) == m.get(k) for k in FIELDS}
    return eq, all(eq.values())


rows = [json.loads(x) for x in open(a.prompts, encoding="utf-8") if x.strip()][:a.n]
bodies = [r.get("request") or r.get("chat") or r for r in rows]
ref = {}
if a.ref and os.path.exists(a.ref):
    for x in open(a.ref, encoding="utf-8"):
        if x.strip():
            r = json.loads(x)
            ref[r.get("id")] = r
pe, he = start([a.ember, a.gguf, str(a.ember_port), "127.0.0.1", "-", "0"], a.ember_port, "gate_ember")
words = shlex.split(a.mts.format(port=a.mts_port))
env = dict(w.split("=", 1) for w in words if "=" in w.split("/")[0] and not w.startswith("-"))
argv = [os.path.expanduser(w) for w in words if w.split("=", 1)[0] not in env]
pm, hm = start(argv, a.mts_port, "gate_mtstudio", env)
ctx = he.get("context")
# The rendering: chat_template_check render on every request, with ember's context and max_new.
inp = "".join(json.dumps({"chat": b, "ctx": ctx, "max_new_tokens": a.max_new}, ensure_ascii=False) + "\n" for b in bodies)
rr = subprocess.run([a.check, "render", a.gguf], input=inp.encode(), capture_output=True)
rendered = [json.loads(x) for x in rr.stdout.decode().splitlines() if x.strip()][1:]
recs = []
counts = {k: 0 for k in ("mtstudio_status", *("mtstudio_" + f for f in FIELDS), "mtstudio_all", "prompt_text", "prompt_ids_render",
                         "ref_all", "chat_stream", "openai_plain", "openai_stream", "finish_reason")}
pt_reported = pt_same = n_ref = 0
try:
    for i, (row, b) in enumerate(zip(rows, bodies)):
        body = {**b, **GREEDY}
        se, e = post(a.ember_port, "/chat", body)
        sm, m = post(a.mts_port, "/chat", body)
        eq, all_eq = compare(se, e, sm, m)
        counts["mtstudio_status"] += se == sm
        for k in FIELDS:
            counts["mtstudio_" + k] += eq[k]
        counts["mtstudio_all"] += all_eq
        if se == sm == 200 and "prompt_tokens" in m:
            pt_reported += 1
            pt_same += e["prompt_tokens"] == m["prompt_tokens"]
        rd = rendered[i] if i < len(rendered) else {"error": "no render row"}
        want_prompt = row.get("prompt")
        if "error" in rd:
            ok_text = se == 400 and want_prompt is None or se == 400
            ok_ids = se == 400
        else:
            ok_text = want_prompt is None or rd["prompt"] == want_prompt
            ok_ids = se == 200 and rd["ids"] == e.get("prompt_ids")
        counts["prompt_text"] += ok_text
        counts["prompt_ids_render"] += ok_ids
        rec = {"i": i, "id": row.get("id"), "status": [se, sm], "equal": eq, "ember": e, "mtstudio": m,
               "render": {k: rd.get(k) for k in ("prompt_tokens", "history_turns_used", "error")},
               "prompt_text_equal": ok_text, "prompt_ids_equal_render": ok_ids}
        if ref:
            r = ref.get(row.get("id"))
            if r is not None:
                n_ref += 1
                ok_ref = (("error" in r) == (se != 200)) and ("error" in r or all(e.get(k) == r.get(k) for k in FIELDS))
                counts["ref_all"] += ok_ref
                rec["ref_equal"] = ok_ref
        if se == 200:
            s2, raw = post(a.ember_port, "/chat", {**body, "stream": True}, raw=True)
            evs = events(raw) if s2 == 200 else []
            joined = "".join(x["token"] for x in evs if isinstance(x, dict) and "token" in x)
            final = next((x for x in evs if isinstance(x, dict) and x.get("done")), {})
            ok_stream = joined == e["reply"] and final.get("reply") == e["reply"] and final.get("tool_calls") == e.get("tool_calls")
            ob = openai_messages(b)
            so, o = post(a.ember_port, "/v1/chat/completions", ob)
            msg = o["choices"][0]["message"] if so == 200 else {}
            ok_oa = so == 200 and msg.get("content") == e["reply"] and msg.get("tool_calls") == e.get("tool_calls")
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
            ok_os = s3 == 200 and content == msg.get("content") and calls == (msg.get("tool_calls") or [])
            ok_fin = finish == want_finish and (o["choices"][0]["finish_reason"] if so == 200 else None) == want_finish
        else:  # a refused request: both endpoints refuse it (400), streamed or not
            so, _ = post(a.ember_port, "/v1/chat/completions", openai_messages(b))
            s2, _ = post(a.ember_port, "/chat", {**body, "stream": True})
            ok_stream = s2 == 400
            ok_oa = ok_os = ok_fin = so == 400
        counts["chat_stream"] += ok_stream
        counts["openai_plain"] += ok_oa
        counts["openai_stream"] += ok_os
        counts["finish_reason"] += ok_fin
        rec.update({"chat_stream_equal": ok_stream, "openai_equal": ok_oa, "openai_stream_equal": ok_os, "finish_ok": ok_fin})
        recs.append(rec)
        print(f"[{i}] {row.get('id')} {'same' if all_eq else 'DIFF'} status {se}/{sm}", flush=True)
finally:
    for p in (pe, pm):
        p.terminate()
        p.wait()
n = len(bodies)
need = [k for k in counts if not (k == "ref_all" and not ref)]
summary = {"n": n, "gguf": a.gguf, "prompts": a.prompts, "ember": a.ember, "mts": a.mts, "request": GREEDY,
           "ember_health": he, "mtstudio_health": hm, "counts": counts, "mtstudio_prompt_tokens": [pt_same, pt_reported],
           "ref": a.ref if ref else None, "ref_rows": n_ref,
           "passed": all(counts[k] == (n_ref if k == "ref_all" else n) for k in need) and pt_same == pt_reported}
with open(a.out, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "records": recs}, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(json.dumps(summary))
sys.exit(0 if summary["passed"] else 1)
