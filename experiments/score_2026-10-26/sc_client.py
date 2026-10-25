"""ember /score gate driver (run in WSL).

  sc_client.py serve BIN MODEL PORT OUTDIR TAG REQ [REQ ...] [--chunk N] [--errors] [--chat]
      Starts BIN (tinyllama_server MODEL PORT 127.0.0.1 - GPU_LAYERS) with the caller's environment, waits for
      /health, posts each REQ file (JSONL of {key, prompt_ids, continuation_ids}) as one call and, with --chunk,
      again in calls of N; writes OUTDIR/TAG.<req>.jsonl (records in request order), OUTDIR/TAG.<req>.chunk.jsonl,
      OUTDIR/TAG.stats.json (health, per-call stats, wall times); --errors probes the refusals; --chat sends
      one greedy /chat before and after the scoring (the replies must be equal: /score leaves /chat alone).
      GPU_LAYERS from $SC_GPU_LAYERS (default 0).
  sc_client.py compare REF.jsonl GOT.jsonl [--label L] [--hf]
      Per key: over_ctx, logprobs (max |d|, max |d| / max(1, max |ref|) = ROUNDING, bit-equal), argmax, sum.
      --hf: REF rows carry "range" (HF's logit range per position): max |d| / range too (the real-model rule).
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def post(url, body, timeout=7200):
    data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        txt = e.read().decode()
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, {"raw": txt}


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def write_jsonl(p, rows):
    with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(p + ".tmp", p)


ERRORS = [  # (name, body); mtstudio score fails on each of these (or never sees it: not JSON / no list)
    ("not_json", b"{not json"),
    ("no_requests", {"reqs": []}),
    ("request_not_object", {"requests": [[1, 2]]}),
    ("empty_continuation", {"requests": [{"key": "a", "prompt_ids": [1, 2], "continuation_ids": []}]}),
    ("empty_prompt", {"requests": [{"key": "a", "prompt_ids": [], "continuation_ids": [5]}]}),
    ("prompt_id_negative", {"requests": [{"prompt_ids": [1, -1], "continuation_ids": [5]}]}),
    ("prompt_id_vocab", {"requests": [{"prompt_ids": [1, "VOCAB"], "continuation_ids": [5]}]}),
    ("continuation_id_vocab", {"requests": [{"prompt_ids": [1, 2], "continuation_ids": ["VOCAB"]}]}),
    ("ids_not_list", {"requests": [{"prompt_ids": "1 2", "continuation_ids": [5]}]}),
    ("ids_string_element", {"requests": [{"prompt_ids": [1, "x"], "continuation_ids": [5]}]}),
    ("text_fields_only", {"requests": [{"prompt": "Hello", "continuation": " world"}]}),
    ("good_then_bad", {"requests": [{"key": "ok", "prompt_ids": [1, 2], "continuation_ids": [5]},
                                    {"key": "bad", "prompt_ids": [1, 2], "continuation_ids": []}]}),
    ("empty_list", {"requests": []}),
]


def subst(body, vocab):
    if isinstance(body, bytes):
        return body
    s = json.dumps(body).replace('"VOCAB"', str(vocab))
    return json.loads(s)


def serve(argv):
    bin_, model, port, outdir, tag = argv[:5]
    rest = argv[5:]
    chunk = None
    if "--chunk" in rest:
        k = rest.index("--chunk")
        chunk = int(rest[k + 1])
        del rest[k:k + 2]
    errors = "--errors" in rest
    chat = "--chat" in rest
    reqs = [x for x in rest if not x.startswith("--")]
    os.makedirs(outdir, exist_ok=True)
    url = f"http://127.0.0.1:{port}"
    log = open(f"{outdir}/{tag}.server.log", "w")
    gl = os.environ.get("SC_GPU_LAYERS", "0")
    cwd = f"{outdir}/{tag}.cwd"  # the server's debugging.log lands here
    os.makedirs(cwd, exist_ok=True)
    p = subprocess.Popen([bin_, model, port, "127.0.0.1", "/nonexistent_www", gl], stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
    stats = {"tag": tag, "model": model, "bin": bin_, "env": {k: v for k, v in os.environ.items() if k.startswith(("EMBER_", "OMP_", "SC_"))}, "calls": []}
    try:
        t0 = time.time()
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    stats["health"] = json.loads(r.read())
                break
            except Exception:
                if p.poll() is not None or time.time() - t0 > 900:
                    raise SystemExit(f"server did not come up (rc {p.poll()})")
                time.sleep(1)
        stats["load_s"] = round(time.time() - t0, 2)
        vocab = stats["health"]["vocab_size"]
        chat_body = {"user_input": "What is the capital of France?", "max_new_tokens": 24, "temperature": 1.0,
                     "top_k": 1, "top_p": 1.0, "seed": 0, "return_ids": True}
        if chat:
            stats["chat_before"] = post(url + "/chat", chat_body)
        for rp in reqs:
            rows = read_jsonl(rp)
            name = os.path.basename(rp).replace(".jsonl", "")
            t = time.time()
            st, res = post(url + "/score", {"requests": rows})
            wall = time.time() - t
            if st != 200:
                raise SystemExit(f"{name}: HTTP {st}: {res}")
            write_jsonl(f"{outdir}/{tag}.{name}.jsonl", res["results"])
            stats["calls"].append({"req": name, "n": len(rows), "wall_s": round(wall, 3), "req_per_s": round(len(rows) / wall, 2),
                                   "stats": res["stats"]})
            print(f"{tag} {name}: {len(rows)} requests in {wall:.2f} s ({len(rows) / wall:.1f}/s) {res['stats']}", flush=True)
            if chunk:
                got, t = [], time.time()
                for i in range(0, len(rows), chunk):
                    st, res = post(url + "/score", {"requests": rows[i:i + chunk]})
                    if st != 200:
                        raise SystemExit(f"{name} chunk {i}: HTTP {st}: {res}")
                    for r in res["results"]:
                        r["i"] += i  # the index within its call
                    got += res["results"]
                write_jsonl(f"{outdir}/{tag}.{name}.chunk.jsonl", got)
                stats["calls"].append({"req": name + f".chunk{chunk}", "n": len(rows), "wall_s": round(time.time() - t, 3)})
        if errors:
            stats["errors"] = {}
            for nm, body in ERRORS:
                st, res = post(url + "/score", subst(body, vocab))
                stats["errors"][nm] = {"status": st, "body": res}
                print(f"{tag} error {nm}: {st} {json.dumps(res)[:160]}", flush=True)
        if chat:
            stats["chat_after"] = post(url + "/chat", chat_body)
            a, b = stats["chat_before"][1], stats["chat_after"][1]
            stats["chat_same"] = all(a.get(k) == b.get(k) for k in ("reply", "ids", "stop_id", "prompt_ids"))
            print(f"{tag} /chat before == after /score: {stats['chat_same']}", flush=True)
    finally:
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
    json.dump(stats, open(f"{outdir}/{tag}.stats.json", "w"), indent=1)


def compare(argv):
    ref_p, got_p = argv[0], argv[1]
    label = argv[argv.index("--label") + 1] if "--label" in argv else os.path.basename(got_p)
    hf = "--hf" in argv
    ref = {r["key"]: r for r in read_jsonl(ref_p)}
    got = {r["key"]: r for r in read_jsonl(got_p)}
    out = {"label": label, "ref": ref_p, "got": got_p, "keys_ref": len(ref), "keys_got": len(got),
           "keys_same": sorted(ref) == sorted(got)}
    n_over = over_same = argmax_pos = argmax_same = scored = bit_req = 0
    worst_abs = worst_rel = worst_sum = worst_range_rel = 0.0
    worst_key = None
    flips = []
    fields_same = 0
    for k, r in ref.items():
        g = got.get(k)
        if g is None:
            continue
        if bool(r.get("over_ctx")) == bool(g.get("over_ctx")):
            over_same += 1
        if r.get("over_ctx"):
            n_over += 1
            continue
        if g.get("over_ctx"):
            continue
        scored += 1
        if all(r.get(f) == g.get(f) for f in ("prompt_tokens", "continuation_ids")) or "continuation_ids" not in r:
            fields_same += 1
        lr, lg = r["logprobs"], g["logprobs"]
        if lr == lg and r["sum"] == g["sum"]:
            bit_req += 1
        scale = max(1.0, max(abs(x) for x in lr))
        for j, (a, b) in enumerate(zip(lr, lg)):
            d = abs(a - b)
            if d > worst_abs:
                worst_abs, worst_key = d, f"{k}@{j}"
            worst_rel = max(worst_rel, d / scale)
            if hf:
                worst_range_rel = max(worst_range_rel, d / r["range"][j])
        worst_sum = max(worst_sum, abs(r["sum"] - g["sum"]))
        for j, (a, b) in enumerate(zip(r["argmax"], g["argmax"])):
            argmax_pos += 1
            if a == b:
                argmax_same += 1
            else:
                flips.append({"key": k, "pos": j, "ref": a, "got": b, "ref_margin": r.get("margin", [None] * (j + 1))[j]})
    out.update({"over_ctx_ref": n_over, "over_ctx_same": over_same, "scored": scored, "fields_same": fields_same,
                "requests_bit_equal": bit_req, "logprob_worst_abs": worst_abs, "logprob_worst_at": worst_key,
                "logprob_worst_rounding": worst_rel, "sum_worst_abs": worst_sum,
                "argmax_same": argmax_same, "argmax_positions": argmax_pos, "argmax_flips": flips})
    if hf:
        out["logprob_worst_over_range"] = worst_range_rel
    print(json.dumps(out))


if __name__ == "__main__":
    {"serve": serve, "compare": compare}[sys.argv[1]](sys.argv[2:])
