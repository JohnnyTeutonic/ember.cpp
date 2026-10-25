"""CPU gate of ember on paperkiln's five third-party-ops training models (C:/ml_artifacts/thirdparty_train_gate):
tp_gqa (llama, 4/2 GQA), tp_qwen (NEOX, per-head q/k RMSNorm, theta 1e6, 4/2), tp_mqa_partial (1 kv head,
n_rot 4 of 8), tp_neox_partial (MHA, NEOX, n_rot 4 of 8), tp_mamba1 (Mamba-1, llama.cpp tensor names).
  - sha256 of every file against manifest.json's "files";
  - the path: tools/ir_path_check (graph_ir.h tuned_llama_match), tinyllama_server's own load log (gpu_layers 0),
    and tp_session_logits's session (built as the server builds it): "tuned" or "interpreter";
  - logits of the 40 score_in sequences on the served path (tp_session_logits, gpu_layers 0) and on the interpreter
    (tools/ir_logits): per-token log-probs of the continuation against mtstudio score's (score_out.jsonl), worst
    |d| / max(1, max |ref|) (ROUNDING 1e-5); argmax at every scored position = continuation (teacher-forced greedy)
    and = mtstudio's argmax; free-running greedy = continuation; decode vs prefill (bit-equal, else worst |d| over
    the row's range); the session's logits against ir_logits's (bit-equal when the interpreter serves);
  - the Q8_0 files: the same, greedy gated, log-probs reported.
    python3 tp_gate_cpu.py BUILD_DIR OUT.json          (writes the CPU logits to ~/tptrain/work for the CUDA leg)"""
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

D = "/mnt/c/ml_artifacts/thirdparty_train_gate"
NAMES = ["tp_gqa", "tp_qwen", "tp_mqa_partial", "tp_neox_partial", "tp_mamba1"]
B, OUT = sys.argv[1], sys.argv[2]
WORK = os.path.expanduser("~/tptrain/work")
os.makedirs(WORK, exist_ok=True)
ENV = {k: v for k, v in os.environ.items() if k not in ("EMBER_IR", "EMBER_INTERPRETER", "EMBER_IR_ONLY")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def rel(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    rng = ref.max(1) - ref.min(1)
    return float((np.abs(x - ref).max(1) / np.where(rng > 0, rng, 1)).max())


def score(em, ins, outs):
    worst, tf, tf_m, free, dec_bit, dec_rel = 0.0, 0, 0, 0, True, 0.0
    for k, (a, b) in enumerate(zip(ins, outs)):
        assert b["continuation_ids"] == a["continuation_ids"]
        pre = np.array(em[k]["prefill"], dtype=np.float64)
        dec = np.array(em[k]["decode"], dtype=np.float64)
        dec_bit = dec_bit and bool(np.array_equal(np.float32(dec), np.float32(pre)))
        dec_rel = max(dec_rel, rel(dec, pre))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        am = [int(pre[P - 1 + j].argmax()) for j in range(len(cont))]
        tf += am == cont
        tf_m += am == b["argmax"]
        if "free" in em[k]:
            free += em[k]["free"] == cont
    n = len(ins)
    r = {"logprob_worst_rel": worst, "greedy": f"{tf}/{n}", "argmax_eq_mtstudio": f"{tf_m}/{n}",
         "decode_bit_equal_prefill": dec_bit, "decode_vs_prefill_worst_rel": dec_rel}
    if "free" in em[0]:
        r["greedy_free_running"] = f"{free}/{n}"
    return r


def run(argv, inp):
    r = subprocess.run(argv, input=inp, capture_output=True, text=True, env=ENV, cwd=WORK)
    if r.returncode:
        return None, r.stderr.strip()[-800:]
    return r.stdout, json.loads(r.stdout)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def server_log(gguf):
    """tinyllama_server's load decision on gguf, gpu_layers 0: the IR lines of its log (ember's Logger writes
    debugging.log in the working directory) once /health answers."""
    port = free_port()
    sdir = os.path.join(WORK, "srv_" + os.path.basename(gguf))
    os.makedirs(sdir, exist_ok=True)
    logp = os.path.join(sdir, "debugging.log")
    with open(os.path.join(sdir, "stdout.log"), "w") as log:
        p = subprocess.Popen([os.path.join(B, "tinyllama_server"), gguf, str(port), "127.0.0.1", "-", "0"], env=ENV,
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=sdir)
        up = False
        for _ in range(120):
            if p.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                up = True
                break
            except Exception:
                time.sleep(0.5)
        p.terminate()
        try:
            p.wait(20)
        except subprocess.TimeoutExpired:
            p.kill()
    text = open(logp, errors="replace").read() + open(os.path.join(sdir, "stdout.log"), errors="replace").read()
    lines = [x.strip()[:300] for x in text.splitlines() if re.search(r"\[IR\]|IR interpreter|tuned llama", x)]
    shutil.rmtree(sdir)
    return {"health": up, "log_bytes": len(text), "ir_lines": lines,
            "path": "interpreter" if any("Running the model through the paperkiln IR interpreter" in x for x in lines)
            else "tuned"}


report = {"build": B, "models": {}}
for name in NAMES:
    d = os.path.join(D, name)
    man = json.load(open(os.path.join(d, "manifest.json")))
    m = {"sha256": {f: sha(os.path.join(d, f)) == h for f, h in man["files"].items()},
         "manifest_sha256": sha(os.path.join(d, "manifest.json"))}
    m["sha256_all_match"] = all(m["sha256"].values())
    if not m["sha256_all_match"]:
        report["models"][name] = m
        print(name, "sha256 mismatch: not gated")
        continue
    ins, outs = rows(os.path.join(d, "score_in.jsonl")), rows(os.path.join(d, "score_out.jsonl"))
    inp = json.dumps({"seqs": [r["prompt_ids"] + r["continuation_ids"] for r in ins],
                      "prompt_len": [len(r["prompt_ids"]) for r in ins]})
    seqs = json.dumps([r["prompt_ids"] + r["continuation_ids"] for r in ins])
    files = [f for f in (name + ".gguf", name + "_q8_0.gguf") if os.path.exists(os.path.join(d, f))]
    for f in files:
        g = os.path.join(d, f)
        tag = f[:-5]
        e = {}
        pc = subprocess.run([os.path.join(B, "ir_path_check"), g], capture_output=True, text=True, env=ENV)
        e["ir_path_check"] = json.loads(pc.stdout.splitlines()[0])
        e["server"] = server_log(g)
        st, sj = run([os.path.join(B, "tp_session_logits"), g, "0"], inp)
        it, ij = run([os.path.join(B, "ir_logits"), g], seqs)
        if st is None:
            e["session"] = {"error": sj}
        else:
            e["session"] = {"path": sj["path"], "device": sj["device"], "heads": sj["heads"], "kv_heads": sj["kv_heads"],
                            **score(sj["seqs"], ins, outs)}
            json.dump(sj["seqs"], open(os.path.join(WORK, f"cpu_session_{tag}.json"), "w"))
        if it is None:
            e["ir_logits"] = {"error": ij}
        else:
            e["ir_logits"] = score(ij, ins, outs)
            open(os.path.join(WORK, f"cpu_ir_{tag}.json"), "w").write(it)
        if st is not None and it is not None:
            e["session_vs_ir_logits"] = {
                "prefill_bit_equal": all(np.array_equal(np.float32(a["prefill"]), np.float32(b["prefill"]))
                                         for a, b in zip(sj["seqs"], ij)),
                "decode_bit_equal": all(np.array_equal(np.float32(a["decode"]), np.float32(b["decode"]))
                                        for a, b in zip(sj["seqs"], ij)),
                "prefill_worst_rel": max(rel(a["prefill"], b["prefill"]) for a, b in zip(sj["seqs"], ij))}
        paths = {e["ir_path_check"].get("path"), e["server"]["path"], e["session"].get("path")}
        e["path"] = paths.pop() if len(paths) == 1 else f"DISAGREE {sorted(map(str, paths))}"
        served = e["session"]
        q8 = f.endswith("_q8_0.gguf")
        e["pass"] = bool("error" not in served and served["greedy"] == "40/40" and served["greedy_free_running"] == "40/40"
                         and (q8 or served["logprob_worst_rel"] <= 1e-5)
                         and (served["decode_bit_equal_prefill"] or served["decode_vs_prefill_worst_rel"] <= 1e-5)
                         and e["server"]["health"] and not e["path"].startswith("DISAGREE"))
        m[tag] = e
        print(name, tag, "path", e["path"], "|", json.dumps({k: served.get(k) for k in (
            "logprob_worst_rel", "greedy", "greedy_free_running", "decode_bit_equal_prefill",
            "decode_vs_prefill_worst_rel")}), "| ir_logits", json.dumps(e.get("ir_logits")),
              "| session=ir", json.dumps(e.get("session_vs_ir_logits")), "pass", e["pass"], flush=True)
    m["pass"] = all(m[f[:-5]]["pass"] for f in files)
    report["models"][name] = m

# the critical check: cases 2-4 carry family_hint llama but are not plain llama: the interpreter must serve them
report["family_hint_llama_not_plain_on_interpreter"] = {
    n: {t: report["models"][n][t]["path"] for t in report["models"][n] if t.startswith(n)}
    for n in ("tp_qwen", "tp_mqa_partial", "tp_neox_partial")}
report["critical_check_pass"] = all(v == "interpreter" for x in report["family_hint_llama_not_plain_on_interpreter"].values()
                                    for v in x.values())
report["pass"] = report["critical_check_pass"] and all(report["models"][n].get("pass") for n in NAMES)
json.dump(report, open(OUT, "w"), indent=1)
print("critical check (cases 2-4 on the interpreter):", report["critical_check_pass"], "| pass", report["pass"])
sys.exit(0 if report["pass"] else 1)
