"""End-to-end greedy replies and decode speed from one ember tinyllama_server process (verify_2026-10-27).

    python3 vf_e2e.py run BIN GGUF GPU_LAYERS PORT PROMPTS.jsonl OUT.jsonl [NAME=VALUE ...]
    python3 vf_e2e.py compare BASE.jsonl OTHER.jsonl

run: starts the server with the given environment additions (e.g. EMBER_SPEC=4), checks /health, sends every
request of PROMPTS (one JSON object per line: user_input, optional history [{user, assistant}], optional
max_new_tokens, default 60), greedy (temperature 1, top_k 1, top_p 1, seed 0, return_ids), one request at a time,
each prompt twice (the first pass warms the server and its captured graphs; the second is the record, with its
replies checked against the first). Per request: reply, ids, stop id, tokens, wall ms, request_to_first_token_ms
and the spec counters. Decode tokens per second = (tokens - 1) / (wall - first token) summed over requests.
compare: replies and ids identical, and totals (tokens, decode tokens per second, drafted/accepted/passes)."""
import json
import os
import subprocess
import sys
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def run(binary, gguf, gpul, port, prompts, out, extra):
    work = out + ".run"
    os.makedirs(os.path.join(work, "www"), exist_ok=True)
    dl = os.path.join(work, "debugging.log")
    if not os.path.lexists(dl):
        os.symlink(os.devnull, dl)
    env = dict(os.environ)
    for kv in extra:
        k, v = kv.split("=", 1)
        env[k] = v
    url = f"http://127.0.0.1:{port}"
    err = open(out + ".server.log", "w")
    proc = subprocess.Popen([binary, gguf, str(port), "127.0.0.1", os.path.join(work, "www"), str(gpul)], cwd=work,
                            env=env, stdout=subprocess.DEVNULL, stderr=err, stdin=subprocess.DEVNULL)
    try:
        health = None
        for _ in range(900):
            try:
                health = json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
                break
            except OSError:
                if proc.poll() is not None:
                    sys.exit("server exited before serving")
                time.sleep(1)
        if health is None:
            sys.exit("server did not come up")
        rows = [json.loads(l) for l in open(prompts, encoding="utf-8") if l.strip()]
        recs = []
        for i, r in enumerate(rows):
            body = {"user_input": r["user_input"], "history": r.get("history", []), "temperature": 1.0, "top_k": 1,
                    "top_p": 1.0, "seed": 0, "max_new_tokens": r.get("max_new_tokens", 60), "return_ids": True}
            first = post(url + "/chat", body)
            t0 = time.perf_counter()
            e = post(url + "/chat", body)
            wall = (time.perf_counter() - t0) * 1000.0
            tt = e.get("ttft", {})
            recs.append({"i": i, "set": r.get("set"), "reply": e["reply"], "ids": e.get("ids"), "stop_id": e.get("stop_id"),
                         "tokens": e.get("tokens"), "prompt_tokens": e.get("prompt_tokens"), "wall_ms": wall,
                         "first_token_ms": tt.get("request_to_first_token_ms"),
                         "drafted": tt.get("spec_drafted", 0), "accepted": tt.get("spec_accepted", 0),
                         "passes": tt.get("spec_passes", 0),
                         "warm_pass_same": first["reply"] == e["reply"] and first.get("ids") == e.get("ids")})
        with open(out, "w", encoding="utf-8") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        meta = {"bin": binary, "gguf": gguf, "gpu_layers": gpul, "env": extra, "health": health, "prompts": prompts,
                "summary": totals(recs)}
        json.dump(meta, open(out + ".meta.json", "w"), indent=1)
        print(json.dumps(meta["summary"]))
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def totals(recs):
    tok = sum(r["tokens"] or 0 for r in recs)
    dec_tok = sum(max(0, (r["tokens"] or 0) - 1) for r in recs)
    dec_ms = sum(max(0.0, r["wall_ms"] - (r["first_token_ms"] or 0.0)) for r in recs)
    dr, ac, ps = (sum(r[k] for r in recs) for k in ("drafted", "accepted", "passes"))
    return {"requests": len(recs), "tokens": tok, "wall_ms": sum(r["wall_ms"] for r in recs),
            "decode_tokens": dec_tok, "decode_ms": dec_ms, "decode_tok_per_s": dec_tok / dec_ms * 1000.0 if dec_ms else 0,
            "drafted": dr, "accepted": ac, "passes": ps, "acceptance": ac / dr if dr else None,
            "tokens_from_passes": ac + ps, "warm_pass_same": sum(r["warm_pass_same"] for r in recs)}


def compare(a, b):
    ra = [json.loads(l) for l in open(a, encoding="utf-8")]
    rb = [json.loads(l) for l in open(b, encoding="utf-8")]
    ta, tb = totals(ra), totals(rb)
    res = {"base": a, "other": b, "n": len(ra),
           "replies_identical": sum(x["reply"] == y["reply"] for x, y in zip(ra, rb)),
           "ids_identical": sum(x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"] for x, y in zip(ra, rb)),
           "differ": [x["i"] for x, y in zip(ra, rb) if x["ids"] != y["ids"]],
           "decode_tok_per_s": [ta["decode_tok_per_s"], tb["decode_tok_per_s"]],
           "decode_speedup": tb["decode_tok_per_s"] / ta["decode_tok_per_s"] if ta["decode_tok_per_s"] else None,
           "wall_ms": [ta["wall_ms"], tb["wall_ms"]], "other_spec": {k: tb[k] for k in ("drafted", "accepted", "passes", "acceptance")}}
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8:])
    else:
        compare(sys.argv[2], sys.argv[3])
