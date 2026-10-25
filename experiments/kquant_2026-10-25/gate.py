#!/usr/bin/env python3
"""k-quant gate for paperkiln's pk_base_v1 Q4_K_M / Q4_K_S files (25 Oct 2026).

    gate.py probe TAG GGUF [VAR=VALUE ...]    first-token logits of the 40 frozen prompts -> WORK/TAG.logits
    gate.py replies TAG GGUF [VAR=VALUE ...]  greedy replies (60 tokens) to the 40 prompts -> OUT/TAG_replies.jsonl
    gate.py summary                           every comparison below -> OUT/summary.json (and printed)

Each run starts ember's tinyllama_server (EMBER_BIN, default ~/kq/build_cpu/tinyllama_server) on
GGUF with GPU_LAYERS (default 0: the CPU) and the given environment, checks /health, sends the
requests and stops it. A probe sets TINYLLAMA_LOGITS_DUMP (nine digits: the dump reads back as the
same float32) and asks for one greedy token per prompt, so each dump line is the logit vector of the
prompt's next token. Replies are written line by line and a rerun resumes after the last line.
The mtstudio reference is ../pk_base_v1_2026-10-24/pk_base_v1_parity.jsonl: mtstudio chat (paperkiln
c8f0b8af CPU build) on the same pk_base_v1.gguf (sha256 510d288c...), greedy, the same 40 prompts.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("KQ_WORK", os.path.expanduser("~/kq/work"))
OUT = os.environ.get("KQ_OUT", HERE)
EMBER = os.environ.get("EMBER_BIN", os.path.expanduser("~/kq/build_cpu/tinyllama_server"))
GPU_LAYERS = os.environ.get("GPU_LAYERS", "0")
PORT = int(os.environ.get("KQ_PORT", "8121"))
PROMPTS = os.environ.get("KQ_PROMPTS", "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl")
MTS_REF = os.path.join(HERE, "..", "pk_base_v1_2026-10-24", "pk_base_v1_parity.jsonl")
N = 40


def health():
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2))
    except Exception:
        return None


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def server(tag, gguf, env_pairs, extra_env):
    if health() is not None:
        sys.exit(f"port {PORT} already answers /health; refusing to share it")
    env = dict(os.environ, **dict(p.split("=", 1) for p in env_pairs), **extra_env)
    os.makedirs(WORK, exist_ok=True)
    log = open(os.path.join(WORK, tag + "_server.log"), "w")
    p = subprocess.Popen([EMBER, gguf, str(PORT), "127.0.0.1", "-", GPU_LAYERS], env=env, stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    for _ in range(900):
        if p.poll() is not None:
            sys.exit(f"server exited {p.returncode} on {gguf}; see {log.name}")
        if health() is not None:
            return p
        time.sleep(1)
    p.kill()
    sys.exit("server did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def prompts():
    return [json.loads(l)["prompt"] for l in open(PROMPTS, encoding="utf-8")][:N]


def body(prompt, max_new):
    return {"user_input": prompt, "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0}


def probe(tag, gguf, env_pairs):
    dump = os.path.join(WORK, tag + ".logits")
    if os.path.exists(dump) and sum(1 for _ in open(dump)) == N:
        print(f"{dump}: done already")
        return
    if os.path.exists(dump):
        os.remove(dump)
    t0 = time.time()
    p = server(tag, gguf, env_pairs, {"TINYLLAMA_LOGITS_DUMP": dump})
    try:
        for pr in prompts():
            post(body(pr, 1))
    finally:
        stop(p)
    print(f"probe {tag}: {N} prompts, {time.time() - t0:.0f} s")


def replies(tag, gguf, env_pairs):
    out = os.path.join(OUT, tag + "_replies.jsonl")
    done = [json.loads(l) for l in open(out, encoding="utf-8")] if os.path.exists(out) else []
    ps = prompts()
    if len(done) >= N:
        print(f"{out}: done already")
        return
    t0 = time.time()
    p = server(tag, gguf, env_pairs, {})
    try:
        with open(out, "a", encoding="utf-8") as f:
            for i in range(len(done), N):
                e = post(body(ps[i], 60))
                f.write(json.dumps({"i": i, "reply": e["reply"], "tokens": e["tokens"], "stop": e["stop_reason"],
                                    "prompt_tokens": e["prompt_tokens"]}) + "\n")
                f.flush()
    finally:
        stop(p)
    print(f"replies {tag}: {N - len(done)} prompts, {time.time() - t0:.0f} s")


def load_logits(tag):
    path = os.path.join(WORK, tag + ".logits")
    if not os.path.exists(path):
        return None
    rows = [np.array(l.split(), dtype=np.float32) for l in open(path) if l.strip()]
    return np.array(rows) if len(rows) == N else None  # a probe still running is skipped


def logit_cmp(a, b):
    """b against reference a (float32 arrays [prompts, vocab])."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    srt = np.sort(a, axis=1)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "max_over_range": float(d.max() / (float(a.max()) - float(a.min()))),
            "bit_identical": int((a.view(np.uint32) == b.view(np.uint32)).sum()), "values": int(a.size),
            "top1_agree": int((a.argmax(1) == b.argmax(1)).sum()),
            "ref_smallest_top1_margin": float((srt[:, -1] - srt[:, -2]).min())}


def load_replies(tag):
    path = os.path.join(OUT, tag + "_replies.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows if len(rows) == N else None


def reply_cmp(rows, ref, key="reply", tok="tokens"):
    same = [r["reply"] == x[key] for r, x in zip(rows, ref)]
    return {"reply_identical": int(sum(same)), "token_count_identical": int(sum(r["tokens"] == x[tok] for r, x in zip(rows, ref))),
            "differs_at": [r["i"] for r, s in zip(rows, same) if not s]}


def summary():
    s = {"logits": {}, "replies": {}}
    pairs = [("f32", "q4_k_m"), ("f32", "q4_k_s"), ("q4_k_m_ref", "q4_k_m"), ("q4_k_s_ref", "q4_k_s"),
             ("f32", "q4_k_m_ref"), ("f32", "q4_k_s_ref"),
             ("ir_q4_k_m_ref", "ir_q4_k_m"), ("ir_q4_k_s_ref", "ir_q4_k_s"), ("q4_k_m", "ir_q4_k_m"),
             ("q4_k_s", "ir_q4_k_s"), ("f32", "ir_f32"),
             ("cpu_q4_k_m", "cuda_q4_k_m"), ("cpu_q4_k_s", "cuda_q4_k_s"), ("cuda_q4_k_m_ref", "cuda_q4_k_m"),
             ("cuda_q4_k_s_ref", "cuda_q4_k_s"), ("cpu_q4_k_m", "cuda_ir_q4_k_m"), ("cpu_q4_k_s", "cuda_ir_q4_k_s"),
             ("cuda_ir_q4_k_m_ref", "cuda_ir_q4_k_m"), ("cuda_ir_q4_k_s_ref", "cuda_ir_q4_k_s")]
    for ra, rb in pairs:
        a, b = load_logits(ra), load_logits(rb)
        if a is not None and b is not None:
            s["logits"][f"{rb} vs {ra}"] = logit_cmp(a, b)
    mts = [json.loads(l) for l in open(MTS_REF, encoding="utf-8")]
    f32 = load_replies("f32")
    for tag in ["f32", "q4_k_m", "q4_k_m_ref", "q4_k_s", "q4_k_s_ref", "cuda_q4_k_m", "cuda_q4_k_m_ref",
                "cuda_q4_k_s", "cuda_q4_k_s_ref"]:
        rows = load_replies(tag)
        if rows is None:
            continue
        rec = {"vs_mtstudio_f32": reply_cmp(rows, mts, "mtstudio", "mts_tokens"),
               "prompt_tokens_equal_paperkiln_tokeniser": int(sum(r["prompt_tokens"] == x["bpe_py_prompt_tokens"]
                                                                  for r, x in zip(rows, mts)))}
        if f32 is not None and tag != "f32":
            rec["vs_ember_f32"] = reply_cmp(rows, f32)
        # its gguf-py reference's replies, and for a CUDA run the same file's CPU replies (the CPU receipts)
        for other in (tag + "_ref", tag[len("cuda_"):] if tag.startswith("cuda_") else None):
            o = load_replies(other) if other else None
            if o is not None:
                rec[f"vs_{other}"] = reply_cmp(rows, o)
        s["replies"][tag] = rec
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k, v in s["logits"].items():
        print(f"{k}: max |d| {v['max_abs']:.3g}, mean {v['mean_abs']:.3g}, max/range {v['max_over_range']:.2e}, "
              f"bit-identical {v['bit_identical']}/{v['values']}, top-1 {v['top1_agree']}/{N} "
              f"(ref smallest margin {v['ref_smallest_top1_margin']:.4f})")
    for k, v in s["replies"].items():
        print(k, json.dumps({a: (b if not isinstance(b, dict) else {c: d for c, d in b.items() if c != "differs_at"})
                             for a, b in v.items()}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":
        probe(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "replies":
        replies(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        summary()
