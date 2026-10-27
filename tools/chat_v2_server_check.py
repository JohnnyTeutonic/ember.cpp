#!/usr/bin/env python3
"""ember's server under conversation schema v2 (paperkiln/2; LOCAL_LLM_STACK.md section 3), end to end over HTTP.

    python3 tools/chat_v2_server_check.py SERVER_BIN CHECK_BIN TOY_DIR V1.gguf PORT OUT.json

SERVER_BIN is tinyllama_server, CHECK_BIN chat_template_check (same build); TOY_DIR holds
experiments/chat_v2_2026-10-27/make_toy_v2.py's outputs; V1.gguf is a paperkiln/1 file (the toys' base). Every
server runs on the CPU (gpu_layers 0) from a scratch directory under ~/cv2/work. Checks:
  load      toy_v2_nocontrol.gguf and toy_v2_badtype.gguf: the server refuses to load (exits non-zero, the log
            naming the control token).
  render    on toy_v2.gguf, /chat (return_ids) and /v1/chat/completions requests with tools, tool calls and
            results, user_input left out after a tool message, and histories past the 256-token context: the
            prompt ids, prompt_tokens and history_turns_used equal `chat_template_check render` on the same
            requests (the code the server runs, without HTTP), and OpenAI's usage.prompt_tokens equals it.
  refused   v2 request errors and the context refusal: HTTP 400 with the rule's words.
  stream    greedy replies of the toy model: /chat plain = /chat streamed (joined pieces and the final object) and
            /v1 plain = /v1 streamed (joined content deltas), tool_calls and finish reasons included.
  scripted  a server with EMBER_TEST_SCRIPTED_REPLIES (TinyLlamaSession::set_scripted_replies): replies with tool
            calls, an unparsed call, a spare token, a turn stop and a length stop inside a call, each through
            /chat plain and streamed and /v1 plain and streamed: the content, tool_calls (ids call_0 ..),
            stop_reason and finish_reason as the rules say; streamed = plain byte for byte; one whole delta per
            call; no content piece holds text after the first <|tool_call|> id.
  v1        a server on V1.gguf: tools, tool messages and tool_calls are HTTP 400 on both endpoints; "tools": []
            gives the same greedy reply as no tools.
Writes OUT.json; exit 1 when a check fails."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

SERVER, CHECK, TOY, V1, PORT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
WASM_REF = sys.argv[7] if len(sys.argv) > 7 else None  # rows for tools/wasm_v2_check.mjs
wasm_rows = []
WORK = os.path.expanduser("~/cv2/work")
os.makedirs(WORK, exist_ok=True)
GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 0}
CALL = 16385
fails = []
report = {}


def check(ok, what):
    if not ok:
        fails.append(what)
    return ok


def health():
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2))
    except Exception:
        return None


def start(gguf, tag, env=None, expect_exit=False):
    if health() is not None:
        sys.exit(f"port {PORT} already answers /health; refusing to share it")
    log = os.path.join(WORK, tag + ".log")
    cwd = tempfile.mkdtemp(dir=WORK)  # the logger writes debugging.log into the working directory
    p = subprocess.Popen([SERVER, gguf, str(PORT), "127.0.0.1", "-", "0"], stdout=open(log, "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})}, cwd=cwd)
    for _ in range(600):
        if p.poll() is not None:
            if expect_exit:  # stdout, then the logger's file
                text = open(log, encoding="utf-8", errors="replace").read()
                dl = os.path.join(cwd, "debugging.log")
                if os.path.exists(dl):
                    text += open(dl, encoding="utf-8", errors="replace").read()
                return p, None, text
            sys.exit(f"server on {gguf} exited {p.returncode}; see {log}")
        h = health()
        if h is not None:
            return p, h, None
        time.sleep(0.5)
    p.kill()
    sys.exit(f"server on {gguf} did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
    for _ in range(40):
        if health() is None:
            return
        time.sleep(0.25)


def post(path, body, raw=False):
    data = (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            b = r.read()
            return r.status, (b if raw else json.loads(b))
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b)
        except ValueError:
            return e.code, {"raw": b.decode("utf-8", "replace")}


def events(raw):
    """Server-sent events: each data line's JSON (or "[DONE]"); strict UTF-8 and JSON, or an exception."""
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), block[:80]
        d = block[len("data: "):]
        out.append(d if d == "[DONE]" else json.loads(d))
    return out


def render(gguf, lines):
    """chat_template_check render: per request {"prompt", "ids", "prompt_tokens", "history_turns_used"} or {"error"}."""
    inp = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    r = subprocess.run([CHECK, "render", gguf], input=inp.encode(), capture_output=True, check=False)
    rows = [json.loads(x) for x in r.stdout.decode().splitlines() if x.strip()]
    return rows[0], rows[1:]


def text_ids(gguf, texts):
    """The ids of each text, from the render path: <|user|> TEXT <|assistant|> with the markers cut."""
    _, rows = render(gguf, [{"chat": {"user_input": t}} for t in texts])
    return [r["ids"][1:-1] for r in rows]


def chat_all(body, path="/chat"):
    """The plain and streamed replies of one request."""
    s, plain = post(path, body)
    s2, raw = post(path, {**body, "stream": True}, raw=True)
    return s, plain, s2, events(raw) if s2 == 200 else raw


def chat_stream_view(evs):
    toks = [e["token"] for e in evs if isinstance(e, dict) and "token" in e]
    final = [e for e in evs if isinstance(e, dict) and e.get("done")]
    return "".join(toks), toks, final[0] if final else None


def oa_stream_view(evs):
    content, calls, finish, chunks = "", [], None, 0
    for e in evs:
        if e == "[DONE]" or not e.get("choices"):
            continue
        ch = e["choices"][0]
        d = ch["delta"]
        chunks += 1
        content += d.get("content") or ""
        for c in d.get("tool_calls", []):
            calls.append(c)
        if ch["finish_reason"] is not None:
            finish = ch["finish_reason"]
    return content, calls, finish


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "The weather now.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}},
         {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}]
CALL_HIST = [{"role": "user", "content": "What is the weather in Paris?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
             {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"}]
LONG = "Tell me about the history of the city, its rivers, its bridges and its markets, in some detail please. "


def chat_requests():
    long_hist = []
    for i in range(6):
        long_hist += [{"role": "user", "content": f"Question {i}: " + LONG},
                      {"role": "assistant", "content": f"Answer {i}: it is a long story with many turns. " * 2}]
    return [
        {"user_input": "Hello there!", "system": "You are a weather assistant.", "tools": TOOLS},
        {"history": [{"role": "system", "content": "Be brief."}] + CALL_HIST, "tools": TOOLS},
        {"history": CALL_HIST + [{"role": "assistant", "content": "18 C and clear."}], "user_input": "And Rome?", "tools": TOOLS},
        {"user_input": "What next?", "history": long_hist, "system": "Keep it short.", "tools": TOOLS[:1]},
        {"history": long_hist + CALL_HIST, "tools": TOOLS},
        {"user_input": "plain v2 with no system or tools", "history": [{"user": "hi", "assistant": "hello"}]},
    ]


def oa_requests():
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "developer", "content": "Use the tools."}]
    oa_hist = [dict(m) for m in CALL_HIST]
    oa_hist[1] = {**oa_hist[1], "content": None}
    return [
        {"messages": msgs + [{"role": "user", "content": "Weather in Oslo?"}], "tools": TOOLS},
        {"messages": msgs + oa_hist, "tools": TOOLS},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "parts "}, {"type": "text", "text": "joined"}]}]},
    ]


def main():
    toy = os.path.join(TOY, "toy_v2.gguf")
    # load: refusals
    lr = {}
    for bad in ("toy_v2_nocontrol.gguf", "toy_v2_badtype.gguf"):
        p, h, log = start(os.path.join(TOY, bad), "load_" + bad, expect_exit=True)
        ok = h is None and p.returncode != 0 and "control token" in (log or "")
        check(ok, f"load refused: {bad}")
        line = next((x for x in (log or "").splitlines() if "control token" in x), "")
        lr[bad] = {"refused": ok, "exit": p.returncode, "log_line": line.strip()[-300:]}
    report["load"] = lr

    p, h, _ = start(toy, "toy_v2")
    report["health"] = h
    check(h["chat_template"] == {"name": "paperkiln", "version": 2}, "/health chat_template paperkiln/2")
    ctx = h["context"]

    # render: the server's prompt ids against chat_template_check render
    head, want = render(toy, [{"chat": r, "ctx": ctx, "max_new_tokens": 8} for r in chat_requests()] +
                        [{"openai": r, "ctx": ctx, "max_new_tokens": 8} for r in oa_requests()])
    rr = []
    for i, r in enumerate(chat_requests()):
        s, out = post("/chat", {**r, **GREEDY, "max_new_tokens": 8, "return_ids": True})
        w = want[i]
        ok = s == 200 and "error" not in w and out["prompt_ids"] == w["ids"] and out["prompt_tokens"] == w["prompt_tokens"] \
            and out["history_turns_used"] == w["history_turns_used"]
        check(ok, f"render /chat {i}")
        wasm_rows.append({"kind": "chat", "i": i, "request": {**r, **GREEDY, "max_new_tokens": 8}, "ember": out})
        rr.append({"endpoint": "/chat", "i": i, "status": s, "prompt_tokens": out.get("prompt_tokens"),
                   "history_turns_used": out.get("history_turns_used"), "expect": {k: w.get(k) for k in ("prompt_tokens", "history_turns_used", "error")},
                   "ok": ok})
    for j, r in enumerate(oa_requests()):
        s, out = post("/v1/chat/completions", {**r, **GREEDY, "max_tokens": 8})
        w = want[len(chat_requests()) + j]
        ok = s == 200 and out["usage"]["prompt_tokens"] == w["prompt_tokens"]
        check(ok, f"render /v1 {j}")
        rr.append({"endpoint": "/v1/chat/completions", "i": j, "status": s, "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
                   "expect": w.get("prompt_tokens"), "ok": ok})
    report["render"] = {"template": head, "requests": rr,
                        "dropped": [x["history_turns_used"] for x in rr if x["endpoint"] == "/chat"]}

    # refused: request errors under paperkiln/2
    big = {"history": [{"role": "user", "content": LONG * 6}, {"role": "assistant", "content": "x"}]}
    bad = [
        ("/chat", {"history": CALL_HIST[:1] + [{"role": "assistant", "content": "x"}]}, "Missing 'user_input'"),
        ("/chat", {"history": CALL_HIST, "user_input": "and?"}, "expected assistant"),
        ("/chat", {"user_input": "a", "tools": {"x": 1}}, "tools must be a list"),
        ("/chat", {"user_input": "a", "tools": [{"x": 1.5}]}, "non-integer number"),
        ("/chat", {"user_input": "a", "history": [{"role": "narrator", "content": "x"}]}, "history role 'narrator'"),
        ("/chat", {"user_input": "a <|tool|> b"}, "reserved text '<|tool|>'"),
        ("/chat", {"history": big["history"], "user_input": LONG * 16, "system": "s"}, "the system turn and the newest exchange are"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
         "the last message must be from the user or a tool"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}], "tool_choice": "required"}, "tool_choice"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}, {"role": "tool", "content": "r"}]}, "expected assistant"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": LONG * 16}]}, "the newest exchange is"),
    ]
    rf = []
    for path, body, rule in bad:
        s, out = post(path, {**body, **GREEDY})
        msg = out.get("error", {}).get("message") if path != "/chat" and isinstance(out.get("error"), dict) else out.get("error")
        ok = s == 400 and rule in (msg or "")
        check(ok, f"refused {path}: {rule}")
        rf.append({"endpoint": path, "status": s, "error": msg, "rule": rule, "ok": ok})
    report["refused"] = rf

    # stream = plain on the toy model's own greedy replies
    sp = []
    for i, r in enumerate(chat_requests()[:3] + [{"user_input": "Say something long."}]):
        s, plain, s2, evs = chat_all({**r, **GREEDY, "max_new_tokens": 40})
        joined, toks, final = chat_stream_view(evs)
        ok = s == 200 and s2 == 200 and joined == plain["reply"] and final["reply"] == plain["reply"] and \
            final.get("tool_calls") == plain.get("tool_calls") and final["stop_reason"] == plain["stop_reason"]
        check(ok, f"stream = plain /chat {i}")
        sp.append({"endpoint": "/chat", "i": i, "reply": plain.get("reply"), "stop_reason": plain.get("stop_reason"),
                   "tokens": plain.get("tokens"), "pieces": len(toks), "tool_calls": plain.get("tool_calls"), "ok": ok})
    for j, r in enumerate(oa_requests()):
        s, plain, s2, evs = chat_all({**r, **GREEDY, "max_tokens": 40}, "/v1/chat/completions")
        content, calls, finish = oa_stream_view(evs)
        m = plain["choices"][0]["message"]
        ok = s == 200 and s2 == 200 and content == m["content"] and finish == plain["choices"][0]["finish_reason"] and \
            [{k: c[k] for k in ("id", "type", "function")} for c in calls] == m.get("tool_calls", [])
        check(ok, f"stream = plain /v1 {j}")
        sp.append({"endpoint": "/v1/chat/completions", "i": j, "content": m["content"], "finish_reason": finish, "ok": ok})
    report["stream"] = sp
    stop(p)

    # scripted: tool-call replies through the real handlers
    t = text_ids(toy, ["Let me check.", "get_weather\n{\"city\": \"Paris\"}", "get_time\n{}", "Sure ", "get_weather",
                       "x ", " y", "Café au lait,", "get_weather\n{\"city\": \"Par"])
    check_ids, args_ids, time_ids, sure, name, x_, y_, cafe, partial = t
    EOS, TOOLM = 1, 16386
    cases = [
        ("content then a call", check_ids + [CALL] + args_ids + [EOS], 64,
         {"reply": "Let me check.", "calls": [("get_weather", "{\"city\": \"Paris\"}")], "stop": "eos", "finish": "tool_calls"}),
        ("two calls", [CALL] + args_ids + [CALL] + time_ids + [EOS], 64,
         {"reply": "", "calls": [("get_weather", "{\"city\": \"Paris\"}"), ("get_time", "{}")], "stop": "eos", "finish": "tool_calls"}),
        ("a call without a newline is content", sure + [CALL] + name + [EOS], 64,
         {"reply": "Sure <|tool_call|>get_weather", "calls": [], "stop": "eos", "finish": "stop"}),
        ("a spare token is content", x_ + [16388] + y_ + [EOS], 64,
         {"reply": "x <|spare_4|> y", "calls": [], "stop": "eos", "finish": "stop"}),
        ("a turn stop after a call", cafe + [CALL] + args_ids + [TOOLM, 5, 6], 64,
         {"reply": "Café au lait,", "calls": [("get_weather", "{\"city\": \"Paris\"}")], "stop": "turn", "finish": "tool_calls"}),
        ("the length limit inside the arguments", [CALL] + partial + args_ids + [EOS], 1 + len(partial),
         {"reply": "", "calls": [("get_weather", "{\"city\": \"Par")], "stop": "length", "finish": "length"}),
    ]
    script = []
    for _, ids, _, _ in cases:
        script += [ids] * 4
    sfile = os.path.join(WORK, "scripted_replies.json")
    json.dump(script, open(sfile, "w"))
    p, _, _ = start(toy, "toy_v2_scripted", env={"EMBER_TEST_SCRIPTED_REPLIES": sfile})
    sr = []
    for cname, ids, max_new, e in cases:
        want_calls = [{"id": f"call_{k}", "type": "function", "function": {"name": n, "arguments": a}}
                      for k, (n, a) in enumerate(e["calls"])]
        body = {"history": CALL_HIST[:1] + [{"role": "assistant", "content": "One moment."}], "user_input": "Weather?",
                "tools": TOOLS, **GREEDY, "max_new_tokens": max_new, "return_ids": True}
        s, plain = post("/chat", body)
        s2, raw = post("/chat", {**body, "stream": True}, raw=True)
        joined, toks, final = chat_stream_view(events(raw))
        cut = ids[:max_new]
        stop_at = next((k for k, x in enumerate(cut) if x in (1, 2, 3, 16384, 16386)), None)
        gen = cut if stop_at is None else cut[:stop_at]
        ok_chat = s == 200 and plain["reply"] == e["reply"] and plain.get("tool_calls", []) == want_calls and \
            plain["stop_reason"] == e["stop"] and plain["ids"] == gen and joined == plain["reply"] and \
            final["reply"] == plain["reply"] and final.get("tool_calls", []) == want_calls and s2 == 200
        oa = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, **GREEDY, "max_tokens": max_new}
        s3, op = post("/v1/chat/completions", oa)
        s4, oraw = post("/v1/chat/completions", {**oa, "stream": True}, raw=True)
        oevs = events(oraw)
        content, calls, finish = oa_stream_view(oevs)
        m = op["choices"][0]["message"]
        call_chunks = [ev for ev in oevs if ev != "[DONE]" and ev.get("choices") and "tool_calls" in ev["choices"][0]["delta"]]
        whole = all(len(ev["choices"][0]["delta"]["tool_calls"]) == 1 for ev in call_chunks) and \
            [c["index"] for c in calls] == list(range(len(want_calls))) and len(call_chunks) == len(want_calls)
        ok_oa = s3 == 200 and s4 == 200 and m["content"] == e["reply"] and m.get("tool_calls", []) == want_calls and \
            op["choices"][0]["finish_reason"] == e["finish"] and content == m["content"] and finish == e["finish"] and \
            [{k: c[k] for k in ("id", "type", "function")} for c in calls] == want_calls and whole
        wasm_rows.append({"kind": "scripted", "case": cname, "request": {k: v for k, v in body.items() if k != "return_ids"},
                          "ids": ids, "max_new_tokens": max_new,
                          "expect": {"reply": e["reply"], "tool_calls": want_calls, "stop_reason": e["stop"]}})
        check(ok_chat, f"scripted /chat: {cname}")
        check(ok_oa, f"scripted /v1: {cname}")
        sr.append({"case": cname, "script": ids, "max_new_tokens": max_new, "chat": {
            "reply": plain.get("reply"), "tool_calls": plain.get("tool_calls"), "stop_reason": plain.get("stop_reason"),
            "ids": plain.get("ids"), "pieces": toks, "final_tool_calls": final.get("tool_calls") if final else None, "ok": ok_chat},
            "openai": {"content": m.get("content"), "tool_calls": m.get("tool_calls"),
                       "finish_reason": op["choices"][0]["finish_reason"], "stream_finish_reason": finish,
                       "stream_call_chunks": len(call_chunks), "ok": ok_oa}})
    report["scripted"] = sr
    stop(p)

    # v1: tools, tool messages and tool_calls stay HTTP 400
    p, h1, _ = start(V1, "v1")
    v1 = {"health": h1["chat_template"]}
    v1_bad = [
        ("/chat", {"user_input": "a", "tools": TOOLS}, "tools are not supported"),
        ("/chat", {"history": CALL_HIST[:2] + [{"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}], "user_input": "a"},
         "tool_calls are not supported"),
        ("/chat", {"history": [{"role": "tool", "content": "r"}], "user_input": "a"}, "history role 'tool'"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}], "tools": TOOLS}, "tools are not supported"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}, {"role": "tool", "content": "r"}]},
         "tool messages are not supported"),
        ("/v1/chat/completions", {"messages": CALL_HIST[:2] + [{"role": "user", "content": "b"}]}, "tool_calls are not supported"),
    ]
    rows = []
    for path, body, rule in v1_bad:
        s, out = post(path, {**body, **GREEDY})
        msg = out.get("error", {}).get("message") if path != "/chat" and isinstance(out.get("error"), dict) else out.get("error")
        ok = s == 400 and rule in (msg or "")
        check(ok, f"v1 refused {path}: {rule}")
        rows.append({"endpoint": path, "status": s, "error": msg, "rule": rule, "ok": ok})
    s, a = post("/chat", {"user_input": "Hello!", **GREEDY, "max_new_tokens": 20})
    s2, b = post("/chat", {"user_input": "Hello!", "tools": [], **GREEDY, "max_new_tokens": 20})
    s3, c = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello!"}], "tools": [], **GREEDY, "max_tokens": 20})
    ok = s == s2 == s3 == 200 and a["reply"] == b["reply"] == c["choices"][0]["message"]["content"] and "tool_calls" not in a \
        and "tool_calls" not in c["choices"][0]["message"] and c["choices"][0]["finish_reason"] in ("stop", "length")
    check(ok, "v1: tools [] is no tools")
    v1["refused"] = rows
    v1["empty_tools"] = {"reply": a.get("reply"), "ok": ok}
    report["v1"] = v1
    stop(p)

    if WASM_REF:
        with open(WASM_REF, "w", encoding="utf-8") as f:
            for row in wasm_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["failures"] = fails
    report["passed"] = not fails
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"failures": fails, "out": OUT}))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
