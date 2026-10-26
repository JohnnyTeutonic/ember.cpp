"""Greedy replies, ids and decode speed from one ember tinyllama_server process (draft_2026-10-27).

    python3 dr_e2e.py run BIN GGUF GPU_LAYERS PORT PROMPTS.jsonl OUT.jsonl [--once] [--dump] [NAME=VALUE ...]
    python3 dr_e2e.py compare BASE.jsonl OTHER.jsonl
    python3 dr_e2e.py bits BASE.jsonl OTHER.jsonl        (two --dump runs)

verify_2026-10-27's vf_e2e.py with these additions: the records keep the prompt ids and the drafter's own time
(spec_draft_ms, spec_draft_calls); a line with "turns" (a multi-turn set, dr_prompts.py) is served turn by turn,
each request carrying the earlier user turns and the model's own replies (stripped, as carry5_eval) as history,
one record per turn; --once sends every request once (the CPU collection for the drafting simulator) instead of
twice (warm pass, then the record, its reply checked against the warm pass's); --dump runs the server with
TINYLLAMA_LOGITS_DUMP and keeps every sampled row of each recorded request in OUT.jsonl.rows.f32 (float32, as
verify_det_2026-10-27's vd_collect.py), compared bit for bit by `bits`. Otherwise as vf_e2e.py: greedy
(temperature 1, top_k 1, top_p 1, seed 0, return_ids), one request at a time; decode tokens per second =
(tokens - 1) / (wall - first token) summed over requests; the spec counters per request."""
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
    once = "--once" in extra
    dump_rows = "--dump" in extra
    extra = [x for x in extra if x not in ("--once", "--dump")]
    work = out + ".run"
    os.makedirs(os.path.join(work, "www"), exist_ok=True)
    dl = os.path.join(work, "debugging.log")
    if not os.path.lexists(dl):
        os.symlink(os.devnull, dl)
    env = dict(os.environ)
    for kv in extra:
        k, v = kv.split("=", 1)
        env[k] = v
    dump = os.path.join(work, "dump.txt")
    rows_f = None
    if dump_rows:
        env["TINYLLAMA_LOGITS_DUMP"] = dump
        rows_f = open(out + ".rows.f32", "wb")
    state = {"rows": 0, "vocab": None}
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

        def one(i, r, history, user, turn):
            body = {"user_input": user, "history": history, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0,
                    "max_new_tokens": r.get("max_new_tokens", 60), "return_ids": True}
            first = None if once else post(url + "/chat", body)
            if dump_rows and os.path.exists(dump):
                os.remove(dump)
            t0 = time.perf_counter()
            e = post(url + "/chat", body)
            wall = (time.perf_counter() - t0) * 1000.0
            tt = e.get("ttft", {})
            extra_rec = {}
            if dump_rows:
                import hashlib
                import numpy as np
                text = open(dump, "rb").read() if os.path.exists(dump) else b""
                lines = [x for x in text.split(b"\n") if x.strip()]
                for x in lines:
                    a = np.array(x.split(), dtype=np.float32)
                    if state["vocab"] is None:
                        state["vocab"] = int(a.size)
                    if a.size != state["vocab"]:
                        sys.exit(f"request {i}: a row of {a.size} values (vocab {state['vocab']})")
                    a.tofile(rows_f)
                extra_rec = {"rows": len(lines), "row0": state["rows"], "dump_sha256": hashlib.sha256(text).hexdigest()}
                state["rows"] += len(lines)
            rec = {"i": i, "set": r.get("set"), "turn": turn, "reply": e["reply"], "ids": e.get("ids"),
                   "stop_id": e.get("stop_id"), "prompt_ids": e.get("prompt_ids"), "tokens": e.get("tokens"),
                   "prompt_tokens": e.get("prompt_tokens"), "wall_ms": wall,
                   "first_token_ms": tt.get("request_to_first_token_ms"),
                   "drafted": tt.get("spec_drafted", 0), "accepted": tt.get("spec_accepted", 0),
                   "passes": tt.get("spec_passes", 0), "draft_ms": tt.get("spec_draft_ms", 0.0),
                   "draft_calls": tt.get("spec_draft_calls", 0),
                   "warm_pass_same": True if once else (first["reply"] == e["reply"] and first.get("ids") == e.get("ids")),
                   **extra_rec}
            recs.append(rec)
            return e["reply"]

        for i, r in enumerate(rows):
            if "turns" in r:
                hist = []
                for k, u in enumerate(r["turns"]):
                    reply = one(i, r, hist, u, k + 1)
                    hist.append({"user": u, "assistant": reply.strip()})
            else:
                one(i, r, r.get("history", []), r["user_input"], 0)
        with open(out, "w", encoding="utf-8") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        meta = {"bin": binary, "gguf": gguf, "gpu_layers": gpul, "env": extra, "once": once, "health": health,
                "prompts": prompts, "summary": totals(recs), "dump_rows": state["rows"] if dump_rows else None,
                "vocab": state["vocab"]}
        json.dump(meta, open(out + ".meta.json", "w"), indent=1)
        print(json.dumps(meta["summary"]))
    finally:
        if rows_f:
            rows_f.close()
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
    dms, dcalls = sum(r.get("draft_ms", 0.0) for r in recs), sum(r.get("draft_calls", 0) for r in recs)
    return {"requests": len(recs), "tokens": tok, "wall_ms": sum(r["wall_ms"] for r in recs),
            "decode_tokens": dec_tok, "decode_ms": dec_ms, "decode_tok_per_s": dec_tok / dec_ms * 1000.0 if dec_ms else 0,
            "drafted": dr, "accepted": ac, "passes": ps, "acceptance": ac / dr if dr else None,
            "tokens_from_passes": ac + ps, "draft_ms": dms, "draft_calls": dcalls,
            "draft_us_per_call": dms * 1000.0 / dcalls if dcalls else None, "warm_pass_same": sum(r["warm_pass_same"] for r in recs)}


def compare(a, b):
    ra = [json.loads(l) for l in open(a, encoding="utf-8")]
    rb = [json.loads(l) for l in open(b, encoding="utf-8")]
    ta, tb = totals(ra), totals(rb)
    key = lambda x: (x["ids"], x["stop_id"], x.get("prompt_ids"))
    res = {"base": a, "other": b, "n": len(ra), "n_other": len(rb),
           "replies_identical": sum(x["reply"] == y["reply"] for x, y in zip(ra, rb)),
           "ids_identical": sum(key(x) == key(y) for x, y in zip(ra, rb)),
           "differ": [[x["i"], x["turn"]] for x, y in zip(ra, rb) if key(x) != key(y)],
           "decode_tok_per_s": [ta["decode_tok_per_s"], tb["decode_tok_per_s"]],
           "decode_speedup": tb["decode_tok_per_s"] / ta["decode_tok_per_s"] if ta["decode_tok_per_s"] else None,
           "wall_ms": [ta["wall_ms"], tb["wall_ms"]], "other_spec": {k: tb[k] for k in ("drafted", "accepted", "passes", "acceptance")}}
    print(json.dumps(res))
    return res


def bits(a, b):
    """Two --dump runs: per request ids equal, the same number of dumped rows, every row bit-identical."""
    import numpy as np
    out = {"base": a, "other": b}
    data = []
    for p in (a, b):
        recs = [json.loads(l) for l in open(p, encoding="utf-8")]
        meta = json.load(open(p + ".meta.json"))
        rows = np.fromfile(p + ".rows.f32", dtype=np.float32).reshape(-1, meta["vocab"])
        data.append((recs, meta, rows))
    (ra, ma, xa), (rb, mb, xb) = data
    res = {"base": a, "other": b, "base_env": ma["env"], "other_env": mb["env"], "requests": len(ra),
           "ids_identical": 0, "row_counts_equal": 0, "requests_all_rows_bit_identical": 0, "rows_compared": 0,
           "rows_bit_identical": 0, "max_abs_diff": 0.0, "differ": [],
           "other_spec": {k: mb["summary"][k] for k in ("drafted", "accepted", "passes")}}
    if len(ra) != len(rb) or ma["vocab"] != mb["vocab"]:
        res["error"] = "request counts or vocabularies differ"
        print(json.dumps(res))
        return res
    for x, y in zip(ra, rb):
        same = x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"] and x["prompt_ids"] == y["prompt_ids"]
        res["ids_identical"] += same
        if x["rows"] != y["rows"] or not same:
            res["differ"].append([x["i"], x["turn"]])
            continue
        res["row_counts_equal"] += 1
        pa = xa[x["row0"]:x["row0"] + x["rows"]]
        pb = xb[y["row0"]:y["row0"] + y["rows"]]
        eq = np.all(pa.view(np.uint32) == pb.view(np.uint32), axis=1)
        res["rows_compared"] += int(eq.size)
        res["rows_bit_identical"] += int(eq.sum())
        if eq.all():
            res["requests_all_rows_bit_identical"] += 1
        else:
            res["differ"].append([x["i"], x["turn"]])
            d = np.abs(pa[~eq].astype(np.float64) - pb[~eq].astype(np.float64))
            res["max_abs_diff"] = max(res["max_abs_diff"], float(np.nanmax(d)))
    res["bit_identical"] = res["ids_identical"] == len(ra) and res["requests_all_rows_bit_identical"] == len(ra)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8:])
    elif sys.argv[1] == "bits":
        bits(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3])
