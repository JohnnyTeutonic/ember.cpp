"""Gate of ember's CPU interpreter on paperkiln's first llama_attnres model (IR_SPEC_DRAFT.md section 2;
/mnt/c/ml_artifacts/attnres_gate/ar3, paperkiln c48ca962): 3 llama blocks, d 32, the full (sublayer) form
of attention residuals: depth_attend over a bank list, a per-layer affine after each depth_attend, the
head's depth_attend and affine. The IR is the one embedded in ar3.gguf.
  - sha256 of the four gate files against the expected prefixes;
  - per-token log-probs of the 40 continuations against mtstudio score's (score_out.jsonl), ROUNDING
    (1e-5 of max(1, max |ref|)); ember's argmax at every scored position equal to the continuation,
    40/40; decode logits equal to prefill bit for bit;
  - ar3_q8_0.gguf loads and runs (its [d] pseudo-queries are F32): numbers reported, not gated;
  - blk.0.attnres_attn_q is never read (its depth_attend has one source): a copy of ar3.gguf with that
    tensor overwritten by NaN gives logits equal bit for bit.
Writes attnres_gate_cpu.json beside this file."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import gguf

D = "/mnt/c/ml_artifacts/attnres_gate/ar3"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
EXPECT = {"ar3.gguf": "0e2aea08", "ar3.ir.json": "7428c891", "score_in.jsonl": "7c53d154", "score_out.jsonl": "81e677bb"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()]


report = {"dir": D, "bin": BIN, "sha256": {}}
for f, pre in EXPECT.items():
    h = sha(os.path.join(D, f))
    report["sha256"][f] = {"sha256": h, "expected_prefix": pre, "match": h.startswith(pre)}
if not all(v["match"] for v in report["sha256"].values()):
    print(json.dumps(report, indent=1))
    raise SystemExit("sha256 mismatch: not gating")
ins, outs = rows(os.path.join(D, "score_in.jsonl")), rows(os.path.join(D, "score_out.jsonl"))
seqs = [r["prompt_ids"] + r["continuation_ids"] for r in ins]
env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}


def ember(gguf_path):
    r = subprocess.run([BIN, gguf_path], input=json.dumps(seqs), capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(gguf_path + ": " + r.stderr[-1500:])
    return r.stdout, json.loads(r.stdout)


def score(em):
    worst, greedy, dec_eq = 0.0, 0, True
    for k, (a, b) in enumerate(zip(ins, outs)):
        assert b["continuation_ids"] == a["continuation_ids"]
        pre = np.array(em[k]["prefill"], dtype=np.float64)
        dec_eq = dec_eq and bool(np.array_equal(np.array(em[k]["decode"]), np.array(em[k]["prefill"])))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
    return {"logprob_worst_rel": worst, "greedy": f"{greedy}/{len(ins)}", "decode_eq_prefill": dec_eq}


text_f32, em = ember(os.path.join(D, "ar3.gguf"))
g = score(em)
g["pass"] = g["logprob_worst_rel"] <= 1e-5 and g["greedy"] == f"{len(ins)}/{len(ins)}" and g["decode_eq_prefill"]
report["f32"] = g

r = gguf.GGUFReader(os.path.join(D, "ar3_q8_0.gguf"))
report["q8_0"] = score(ember(os.path.join(D, "ar3_q8_0.gguf"))[1])
report["q8_0"]["pseudo_query_types"] = {t.name: t.tensor_type.name for t in r.tensors if "attnres" in t.name}
report["q8_0"]["pseudo_queries_f32"] = all(v == "F32" for v in report["q8_0"]["pseudo_query_types"].values())

# blk.0.attnres_attn_q overwritten with NaN: equal logits mean it is never read.
work = os.path.expanduser("~/ember_wt/attnres_gate")
os.makedirs(work, exist_ok=True)
nan_path = os.path.join(work, "ar3_nan_blk0_attn_q.gguf")
shutil.copyfile(os.path.join(D, "ar3.gguf"), nan_path)
t = next(t for t in gguf.GGUFReader(nan_path).tensors if t.name == "blk.0.attnres_attn_q.weight")
with open(nan_path, "r+b") as f:
    f.seek(int(t.data_offset))
    f.write(np.full(int(t.n_elements), np.nan, dtype=np.float32).tobytes())
assert np.isnan(np.asarray(next(x for x in gguf.GGUFReader(nan_path).tensors
                                if x.name == "blk.0.attnres_attn_q.weight").data)).all()
text_nan, _ = ember(nan_path)
report["blk0_attn_q_not_read"] = {"nan_copy_logits_bit_equal": text_nan == text_f32}
os.remove(nan_path)

report["pass"] = (g["pass"] and report["q8_0"]["pseudo_queries_f32"] and report["q8_0"]["decode_eq_prefill"]
                  and report["blk0_attn_q_not_read"]["nan_copy_logits_bit_equal"])
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(HERE, "attnres_gate_cpu.json"), "w"), indent=1)
sys.exit(0 if report["pass"] else 1)
