#!/usr/bin/env python3
"""Replies cut inside a UTF-8 character, on every endpoint that returns text (utf8_text.h).

    python3 tools/utf8_server_check.py SERVER_BIN MODEL.gguf PORT OUT.json [--k-max 80 | --ks K,K,...] [--bindings BUILD_DIR]

pk-bench2's IF-tiny job (paperkiln experiments/benchmarks_2026-10-26) stopped at its prompt 375 on
pick_e1_60m_L32 with HTTP 500: the greedy reply at max_new_tokens 64 ended on the lone lead byte 0xD7 and
nlohmann::json refused to serialise it. This starts SERVER_BIN on MODEL (CPU, gpu_layers 0) and sends that
request (IF-tiny's body: temperature 1, top_k 1, top_p 1, seed 7) at every max_new_tokens k from 1 to K_MAX
to the four text paths: /chat, /chat with "stream": true, /v1/chat/completions, and the same with "stream":
true. Every response must be HTTP 200 and every server-sent event strict UTF-8 JSON; per k, the /chat
reply, the joined /chat pieces, the stream's final reply, the /v1 content and the joined /v1 deltas must be
equal, byte for byte. Greedy decoding makes reply k the first k tokens of reply K_MAX, so the k whose reply
ends in U+FFFD are the cuts inside a character, and a k-1 cut that k completes is a character split across
two streamed tokens. --bindings: the Python bindings' generate (BUILD_DIR's tinyllama_bindings) on the same
prompt at every cut k, against the /chat reply. Exit 1 unless every check holds and a cut was seen."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

USER = 'Do not use the word "the" in your reply. Which animal barks?'  # IF-tiny prompt 375
SAMPLING = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 7}
FFFD = "�"


def request(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # a stream the server broke off, or a server that died (a build without the fix)
        return 0, f"{type(e).__name__}: {e}".encode()


def events(raw):
    """The server-sent events' JSON objects, each decoded as strict UTF-8 (a failure raises)."""
    out = []
    for ev in raw.split(b"\n\n"):
        if not ev.startswith(b"data: "):
            continue
        payload = ev[6:]
        if payload == b"[DONE]":
            continue
        out.append(json.loads(payload.decode("utf-8")))
    return out


def one(port, k):
    rec = {"k": k, "status": {}, "errors": []}
    chat = dict(SAMPLING, user_input=USER, max_new_tokens=k, return_ids=True)
    oa = dict(SAMPLING, messages=[{"role": "user", "content": USER}], max_tokens=k)
    s, b = request(port, "/chat", chat)
    rec["status"]["chat"] = s
    if s == 200:
        d = json.loads(b.decode("utf-8"))
        rec.update(reply=d["reply"], ids=d.get("ids"), stop_reason=d["stop_reason"], tokens=d["tokens"],
                   prompt_tokens=d["prompt_tokens"], history_turns_used=d["history_turns_used"],
                   prompt_ids=d.get("prompt_ids"))
    else:
        rec["errors"].append(f"/chat HTTP {s}: {b[:300].decode('utf-8', 'replace')}")
    s, b = request(port, "/chat", dict(chat, stream=True))
    rec["status"]["chat_stream"] = s
    try:
        evs = events(b) if s == 200 else []
        pieces = [e["token"] for e in evs if not e.get("done")]
        final = [e for e in evs if e.get("done")]
        rec["chat_pieces"] = pieces
        rec["chat_stream_final"] = final[0]["reply"] if final else None
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        rec["errors"].append(f"/chat stream event: {e}")
    s, b = request(port, "/v1/chat/completions", oa)
    rec["status"]["v1"] = s
    if s == 200:
        d = json.loads(b.decode("utf-8"))
        rec["v1_content"] = d["choices"][0]["message"]["content"]
        rec["v1_finish"] = d["choices"][0]["finish_reason"]
    s, b = request(port, "/v1/chat/completions", dict(oa, stream=True))
    rec["status"]["v1_stream"] = s
    try:
        evs = events(b) if s == 200 else []
        deltas = [c["delta"].get("content", "") for e in evs for c in e.get("choices", [])]
        rec["v1_deltas"] = [x for x in deltas if x]
        fin = [c["finish_reason"] for e in evs for c in e.get("choices", []) if c.get("finish_reason")]
        rec["v1_stream_finish"] = fin[-1] if fin else None
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        rec["errors"].append(f"/v1 stream event: {e}")
    for name, st in rec["status"].items():
        if st != 200:
            rec["errors"].append(f"{name}: HTTP {st}")
    if "reply" in rec:
        r = rec["reply"]
        checks = {"chat_stream_joined": "".join(rec.get("chat_pieces") or []),
                  "chat_stream_final": rec.get("chat_stream_final"),
                  "v1_content": rec.get("v1_content"),
                  "v1_stream_joined": "".join(rec.get("v1_deltas") or [])}
        for name, v in checks.items():
            if v != r:
                rec["errors"].append(f"{name} {v!r} != /chat reply {r!r}")
        if rec.get("v1_finish") != rec.get("v1_stream_finish"):
            rec["errors"].append(f"finish_reason {rec.get('v1_finish')} vs stream {rec.get('v1_stream_finish')}")
    return rec


def start(server, model, port, work):
    os.makedirs(work, exist_ok=True)
    log = open(os.path.join(work, f"server_{port}.log"), "w")
    p = subprocess.Popen([server, model, str(port), "127.0.0.1", os.path.join(work, "no_www"), "0"], cwd=work,
                         stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError(f"server exited with {p.returncode}")
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("server did not come up")


def bindings_check(build, model, recs):
    sys.path.insert(0, build)
    import tinyllama_bindings as tb
    s = tb.TinyLlamaSession(model, model, threads=2, n_gpu_layers=0)
    prompt = "<|user|>" + USER + "<|assistant|>"  # paperkiln/1, one user turn (chat_tokenizer.h)
    out = []
    for r in recs:
        if "reply" not in r or not r["reply"].endswith(FFFD):
            continue
        try:
            got = s.generate(prompt, steps=r["k"], temperature=1.0, top_k=1, top_p=1.0, system_prompt="",
                             apply_q_a_format=False)
            out.append({"k": r["k"], "equal": got == r["reply"], "reply": got})
        except UnicodeDecodeError as e:
            out.append({"k": r["k"], "equal": False, "error": str(e)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("port", type=int)
    ap.add_argument("out")
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--ks", help="only these max_new_tokens (a build without the fix: the before side)")
    ap.add_argument("--bindings")
    a = ap.parse_args()
    work = os.path.splitext(os.path.abspath(a.out))[0] + "_work"
    p = start(a.server, a.model, a.port, work)
    recs = []
    t0 = time.time()
    try:
        health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=10).read())
        for k in ([int(x) for x in a.ks.split(',')] if a.ks else range(1, a.k_max + 1)):
            recs.append(one(a.port, k))
    finally:
        p.terminate()
        p.wait()
    full = next((r.get("ids") for r in reversed(recs) if r.get("ids") is not None), None)
    prefix_ok = all(r.get("ids") == full[:len(r["ids"])] for r in recs if r.get("ids") is not None) if full else False
    cut = [r["k"] for r in recs if r.get("reply", "").endswith(FFFD)]
    split = [r["k"] for prev, r in zip(recs, recs[1:])
             if prev.get("reply", "").endswith(FFFD) and not r.get("reply", "").startswith(prev.get("reply", ""))]
    bad = [r for r in recs if r["errors"]]
    summary = {"server": a.server, "model": a.model, "health": health, "request": dict(SAMPLING, user_input=USER),
               "k": a.ks or [1, a.k_max], "requests": 4 * len(recs), "seconds": round(time.time() - t0, 1),
               "all_http_200": all(s == 200 for r in recs for s in r["status"].values()),
               "ks_with_errors": [r["k"] for r in bad], "ids_prefix_consistent": prefix_ok,
               "cut_ks": cut, "split_ks": split,
               "replies_at_cuts": {r["k"]: r["reply"][-12:] for r in recs if r["k"] in cut}}
    if a.bindings:
        b = bindings_check(a.bindings, a.model, recs)
        summary["bindings"] = {"cut_ks": [x["k"] for x in b], "equal": sum(x["equal"] for x in b), "rows": b}
    ok = not bad and summary["all_http_200"] and cut and (not a.bindings or
                                                          summary["bindings"]["equal"] == len(summary["bindings"]["rows"]))
    summary["ok"] = bool(ok)
    with open(a.out + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=1, ensure_ascii=False)
    os.replace(a.out + ".tmp", a.out)
    for r in bad[:5]:
        print(f"k={r['k']}: {'; '.join(r['errors'])[:400]}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("health", "request", "replies_at_cuts")},
                     ensure_ascii=False)[:1500])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
