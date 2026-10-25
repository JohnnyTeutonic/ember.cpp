"""CPU gate of ember's IR interpreter on paperkiln's arm X5 model ar_x5 (llama_attnres, full form, scale "sources",
the tiny-chat programme's 20m_L16 shape: d 320, 16 layers, 5 heads, d_ff 864; ir_version 1), the method of
../attnres_2026-10-24/attnres_gate.py:
  - sha256 of every file against manifest.json's "files";
  - per build (tools/ir_logits on the GGUF's embedded IR): per-token log-probs of the 40 continuations against
    mtstudio score's (score_out.jsonl), worst |d| / max(1, max |ref|) (ROUNDING 1e-5); ember's argmax at every
    scored position equal to the continuation (greedy n/40); decode logits equal to prefill bit for bit;
  - ar_x5_q8_0.gguf: the same numbers (reported, not gated), pseudo-query tensor types;
  - blk.0.attnres_attn_q (written as zeros) is never read: a copy with it set to NaN gives bit-equal logits;
  - the builds' F32 outputs against each other, bit for bit.
A build that refuses the file (cbdcb915 reads ir_version 0 only) is recorded with its message.
    python3 x5_gate_cpu.py OUT.json NAME=BUILD_DIR [NAME=BUILD_DIR ...]     (the first build is gated)"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import gguf

D = "/mnt/c/ml_artifacts/attnres_gate_x5/ar_x5"
WORK = os.path.expanduser("~/x5g/work")
out_path = sys.argv[1]
builds = dict(a.split("=", 1) for a in sys.argv[2:])
os.makedirs(WORK, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()]


man = json.load(open(os.path.join(D, "manifest.json")))
report = {"dir": D, "manifest_sha256": sha(os.path.join(D, "manifest.json")), "sha256": {}, "builds": {}}
for f, h in man["files"].items():
    got = sha(os.path.join(D, f))
    report["sha256"][f] = {"sha256": got, "match": got == h}
report["sha256_all_match"] = all(v["match"] for v in report["sha256"].values())
if not report["sha256_all_match"]:
    print(json.dumps(report, indent=1))
    raise SystemExit("sha256 mismatch: not gating")
ins, outs = rows(os.path.join(D, "score_in.jsonl")), rows(os.path.join(D, "score_out.jsonl"))
seqs = [r["prompt_ids"] + r["continuation_ids"] for r in ins]
env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}


def ember(bindir, gguf_path):
    r = subprocess.run([os.path.join(bindir, "ir_logits"), gguf_path], input=json.dumps(seqs), capture_output=True,
                       text=True, env=env)
    if r.returncode:
        return None, r.stderr.strip()[-600:]
    return r.stdout, json.loads(r.stdout)


def score(em):
    worst, greedy, dec_eq = 0.0, 0, True
    for k, (a, b) in enumerate(zip(ins, outs)):
        assert b["continuation_ids"] == a["continuation_ids"]
        pre = np.array(em[k]["prefill"], dtype=np.float64)
        dec_eq = dec_eq and bool(np.array_equal(np.array(em[k]["decode"], dtype=np.float32),
                                                np.array(em[k]["prefill"], dtype=np.float32)))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
    return {"logprob_worst_rel": worst, "greedy": f"{greedy}/{len(ins)}", "decode_eq_prefill": dec_eq}


texts = {}
first = next(iter(builds))
for name, b in builds.items():
    rep = {"bin": os.path.join(b, "ir_logits")}
    text, em = ember(b, os.path.join(D, "ar_x5.gguf"))
    if text is None:
        rep["f32"] = {"refused": em}
        report["builds"][name] = rep
        continue
    texts[name] = text
    open(os.path.join(WORK, f"cpu_{name}_ar_x5.json"), "w").write(text)  # for the device and WASM comparisons
    g = score(em)
    g["pass"] = g["logprob_worst_rel"] <= 1e-5 and g["greedy"] == f"{len(ins)}/{len(ins)}" and g["decode_eq_prefill"]
    rep["f32"] = g
    tq, eq = ember(b, os.path.join(D, "ar_x5_q8_0.gguf"))
    rep["q8_0"] = {"refused": eq} if tq is None else score(eq)
    if tq is not None:
        open(os.path.join(WORK, f"cpu_{name}_ar_x5_q8_0.json"), "w").write(tq)
    report["builds"][name] = rep

r = gguf.GGUFReader(os.path.join(D, "ar_x5_q8_0.gguf"))
types = {t.name: t.tensor_type.name for t in r.tensors if "attnres" in t.name}
report["q8_0_pseudo_query_types"] = sorted(set(types.values()))
report["q8_0_pseudo_queries"] = len(types)
report["q8_0_pseudo_queries_f32"] = all(v == "F32" for v in types.values())
blk0 = np.asarray(next(t for t in gguf.GGUFReader(os.path.join(D, "ar_x5.gguf")).tensors
                       if t.name == "blk.0.attnres_attn_q.weight").data)
report["blk0_attn_q_all_zero"] = bool((blk0 == 0).all())

# blk.0.attnres_attn_q overwritten with NaN, first build: equal logits mean it is never read.
nan_path = os.path.join(WORK, "ar_x5_nan_blk0_attn_q.gguf")
shutil.copyfile(os.path.join(D, "ar_x5.gguf"), nan_path)
t = next(t for t in gguf.GGUFReader(nan_path).tensors if t.name == "blk.0.attnres_attn_q.weight")
with open(nan_path, "r+b") as f:
    f.seek(int(t.data_offset))
    f.write(np.full(int(t.n_elements), np.nan, dtype=np.float32).tobytes())
assert np.isnan(np.asarray(next(x for x in gguf.GGUFReader(nan_path).tensors
                                if x.name == "blk.0.attnres_attn_q.weight").data)).all()
text_nan, _ = ember(builds[first], nan_path)
report["blk0_attn_q_not_read"] = {"build": first, "nan_copy_logits_bit_equal": text_nan == texts.get(first)}
os.remove(nan_path)

report["f32_outputs_bit_equal_between_builds"] = {f"{first}={n}": texts[n] == texts[first] for n in texts if n != first}
g = report["builds"][first]
report["pass"] = bool(g["f32"].get("pass") and g["q8_0"].get("greedy") == f"{len(ins)}/{len(ins)}"
                      and g["q8_0"].get("decode_eq_prefill") and report["q8_0_pseudo_queries_f32"]
                      and report["blk0_attn_q_not_read"]["nan_copy_logits_bit_equal"])
print(json.dumps(report, indent=1))
json.dump(report, open(out_path, "w"), indent=1)
sys.exit(0 if report["pass"] else 1)
