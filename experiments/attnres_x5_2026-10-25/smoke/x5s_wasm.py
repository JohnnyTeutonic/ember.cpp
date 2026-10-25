"""WebAssembly leg on tc_smoke_20m_X5: tools/state_wasm_gate built natively (against ~/x5s/build's libtinyllama_core.a)
and to WebAssembly (deploy/wasm/build.sh's flags, 1gb heap) by ../../wasm_state_2026-10-25/build.sh from e115fd3.
score_in's 40 sequences in four chunks of ten (each chunk's logits are ~190 MB per engine at a 16,384 vocabulary,
deleted once compared): per chunk ../../wasm_state_2026-10-25/compare.py (WebAssembly vs native at every block,
decode vs prefill inside each engine, teacher-forced and free-running greedy), and the WebAssembly prefill's
log-probs of the continuations against mtstudio score's. Aggregated to OUT.json.
    python3 x5s_wasm.py BIN_DIR COMPARE.py OUT.json"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

D = "/mnt/c/ml_artifacts/attnres_gate_x5/smoke_20m_X5"
M = os.path.join(D, "tc_smoke_20m_X5.gguf")
W = os.path.expanduser("~/x5s/wasm")
BIN, CMP, OUTP = sys.argv[1:4]
NODE = sorted(glob.glob(os.path.expanduser("~/emsdk/node/*/bin/node")))[0]
ins = [json.loads(x) for x in open(os.path.join(D, "score_in.jsonl")) if x.strip()]
outs = [json.loads(x) for x in open(os.path.join(D, "score_out.jsonl")) if x.strip()]
chunks = []
lp_worst, lp_greedy = 0.0, 0
for c in range(4):
    sl = list(range(10 * c, 10 * c + 10))
    d = os.path.join(W, f"c{c}")
    for e in ("native", "wasm"):
        os.makedirs(os.path.join(d, e), exist_ok=True)
    inp = os.path.join(d, "in.json")
    json.dump({"seqs": [ins[i]["prompt_ids"] + ins[i]["continuation_ids"] for i in sl],
               "prompt_len": [len(ins[i]["prompt_ids"]) for i in sl]}, open(inp, "w"))
    rn = subprocess.run([os.path.join(BIN, "state_wasm_gate"), M, inp, os.path.join(d, "native")], capture_output=True, text=True)
    rw = subprocess.run([NODE, os.path.join(BIN, "state_wasm_gate.js"), M, inp, os.path.join(d, "wasm")], capture_output=True, text=True)
    if rn.returncode or rw.returncode:
        sys.exit(f"chunk {c}: native rc {rn.returncode} {rn.stderr[-400:]}; wasm rc {rw.returncode} {rw.stderr[-400:]}")
    cr = subprocess.run([sys.executable, CMP, f"smoke_20m_X5_c{c}", inp, os.path.join(d, "native"), os.path.join(d, "wasm")],
                        capture_output=True, text=True)
    rep = json.loads(cr.stdout)
    run = json.load(open(os.path.join(d, "wasm", "run.json")))
    lg = np.fromfile(os.path.join(d, "wasm", "logits.f32"), dtype=np.float32).reshape(-1, run["vocab"])
    for r, i in zip(run["rows"], sl):
        o, n = r["prefill_on"]
        pre = lg[o:o + n].astype(np.float64)
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(ins[i]["prompt_ids"]), ins[i]["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(outs[i]["logprobs"])
        lp_worst = max(lp_worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        lp_greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
    chunks.append(rep)
    for e in ("native", "wasm"):
        os.remove(os.path.join(d, e, "logits.f32"))
    print(f"chunk {c}: pass {rep['pass']}, rows {rep['logit_rows']}, ms {rep['ms']}", flush=True)


def frac(key_path):
    a = b = 0
    for r in chunks:
        x = r
        for k in key_path:
            x = x[k]
        p, q = x.split("/")
        a, b = a + int(p), b + int(q)
    return f"{a}/{b}"


def agg(path):
    blocks = [r for r in chunks]
    for k in path:
        blocks = [b[k] for b in blocks]
    return {"worst_rel": max(b["worst_rel"] for b in blocks), "bit_equal": all(b["bit_equal"] for b in blocks),
            "top1": frac(list(path) + ["top1"])}


out = {"gguf": M, "engine_pair": "state_wasm_gate native (g++ -O3, libtinyllama_core.a) vs WebAssembly (em++ -O3 -msimd128, node)",
       "sequences": sum(r["sequences"] for r in chunks), "logit_rows": sum(r["logit_rows"] for r in chunks),
       "integer_results_equal": all(r["integer_results_equal"] for r in chunks),
       "wasm_vs_native": {k: agg(["wasm_vs_native", k]) for k in chunks[0]["wasm_vs_native"]},
       "inside": {e: {k: agg(["inside", e, k]) for k in chunks[0]["inside"][e]} for e in ("native", "wasm")},
       "greedy": {e: {g: frac(["greedy", e, g]) for g in ("teacher_forced", "free_running")} for e in ("native", "wasm")},
       "greedy_ids_equal": all(r["greedy_ids_equal"] for r in chunks),
       "wasm_logprob_worst_rel_vs_mtstudio": lp_worst, "wasm_greedy_vs_continuation": f"{lp_greedy}/{len(ins)}",
       "ms": {e: sum(r["ms"][e] for r in chunks) for e in ("native", "wasm")},
       "chunks_pass": [r["pass"] for r in chunks]}
out["pass"] = bool(all(out["chunks_pass"]) and lp_worst <= 1e-5 and lp_greedy == len(ins))
json.dump(out, open(OUTP, "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if out["pass"] else 1)
