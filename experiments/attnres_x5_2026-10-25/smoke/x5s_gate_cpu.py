"""CPU gate of ember e115fd3's IR interpreter on paperkiln's arm X5 real-vocabulary smoke model tc_smoke_20m_X5
(llama_attnres, full form, scale "sources", 20m_L16: d 320, 16 layers, 5 heads, d_ff 864; the 16,384-entry Llama 3
vocabulary; 300 steps), the method of ../x5_gate_cpu.py:
  - sha256 of every file against manifest.json's "files";
  - tools/ir_logits on the GGUF's embedded IR, one process per sequence (a full [T, 16384] dump of all 40 at once
    would be gigabytes of JSON): per-token log-probs of the 40 continuations against mtstudio score's
    (score_out.jsonl), worst |d| / max(1, max |ref|) (ROUNDING 1e-5); ember's argmax at every scored position equal
    to the continuation (greedy n/40); decode logits equal to prefill bit for bit;
  - the GGUF's embedded IR equal to tc_smoke_20m_X5.ir.json (parsed JSON);
  - blk.0.attnres_attn_q: whether it is all zeros, and whether a copy set to NaN gives bit-equal logits (never read).
The decode rows at the scored positions (P-1 .. P+6) go to WORK/cpu_rows.npy for x5s_chat.py.
    python3 x5s_gate_cpu.py BUILD_DIR OUT.json"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import gguf

D = "/mnt/c/ml_artifacts/attnres_gate_x5/smoke_20m_X5"
M = os.path.join(D, "tc_smoke_20m_X5.gguf")
WORK = os.path.expanduser("~/x5s/work")
bindir, out_path = sys.argv[1:3]
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
report = {"dir": D, "bin": os.path.join(bindir, "ir_logits"), "manifest_sha256": sha(os.path.join(D, "manifest.json")),
          "sha256": {}}
for f, h in man["files"].items():
    got = sha(os.path.join(D, f))
    report["sha256"][f] = {"sha256": got, "match": got == h}
report["sha256_all_match"] = all(v["match"] for v in report["sha256"].values())
if not report["sha256_all_match"]:
    print(json.dumps(report, indent=1))
    raise SystemExit("sha256 mismatch: not gating")
ins, outs = rows(os.path.join(D, "score_in.jsonl")), rows(os.path.join(D, "score_out.jsonl"))
env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}

# The embedded IR (EMBER_IR_PRINT) against the .ir.json beside it.
pr = os.path.join(WORK, "embedded_ir.json")
r = subprocess.run([os.path.join(bindir, "ir_logits"), M], input="[[2]]", capture_output=True, text=True,
                   env={**env, "EMBER_IR_PRINT": pr})
assert r.returncode == 0, r.stderr[-600:]
report["embedded_ir_equals_ir_json"] = json.load(open(pr)) == json.load(open(os.path.join(D, "tc_smoke_20m_X5.ir.json")))
ir = json.load(open(pr))
report["ir"] = {"ir_version": ir.get("ir_version"), "family_hint": ir.get("family_hint")}


def ember(path, seq):
    r = subprocess.run([os.path.join(bindir, "ir_logits"), path], input=json.dumps([seq]), capture_output=True,
                       text=True, env=env)
    if r.returncode:
        raise SystemExit(r.stderr.strip()[-600:])
    return r.stdout, json.loads(r.stdout)[0]


worst, greedy, dec_eq, worst_dec_vs_pre = 0.0, 0, True, 0.0
per, keep, hashes = [], [], []
for k, (a, b) in enumerate(zip(ins, outs)):
    assert b["continuation_ids"] == a["continuation_ids"]
    seq = a["prompt_ids"] + a["continuation_ids"]
    text, em = ember(M, seq)
    hashes.append(hashlib.sha256(text.encode()).hexdigest())
    pre32 = np.array(em["prefill"], dtype=np.float32)
    dec32 = np.array(em["decode"], dtype=np.float32)
    eq = bool(np.array_equal(pre32, dec32))
    dec_eq = dec_eq and eq
    worst_dec_vs_pre = max(worst_dec_vs_pre, float(np.abs(pre32.astype(np.float64) - dec32).max()))
    pre = pre32.astype(np.float64)
    lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
    P, cont = len(a["prompt_ids"]), a["continuation_ids"]
    lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
    ref = np.array(b["logprobs"])
    e = float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max()))
    worst = max(worst, e)
    am = [int(pre[P - 1 + j].argmax()) for j in range(len(cont))]
    g = am == cont
    greedy += g
    per.append({"i": k, "prompt_len": P, "logprob_rel": e, "greedy": g, "decode_eq_prefill": eq,
                "argmax_eq_mtstudio_argmax": am == b["argmax"]})
    keep.append(dec32[P - 1:P - 1 + len(cont)])
np.save(os.path.join(WORK, "cpu_rows.npy"), np.stack(keep))
n = len(ins)
report["f32"] = {"logprob_worst_rel": worst, "greedy": f"{greedy}/{n}", "decode_eq_prefill": dec_eq,
                 "decode_vs_prefill_max_abs": worst_dec_vs_pre,
                 "argmax_eq_mtstudio_argmax": f"{sum(p['argmax_eq_mtstudio_argmax'] for p in per)}/{n}",
                 "positions_scored": int(sum(len(a["continuation_ids"]) for a in ins)),
                 "prompt_len_range": [min(p["prompt_len"] for p in per), max(p["prompt_len"] for p in per)]}
report["f32"]["pass"] = worst <= 1e-5 and greedy == n and dec_eq

g0 = gguf.GGUFReader(M)
blk0 = next((t for t in g0.tensors if t.name == "blk.0.attnres_attn_q.weight"), None)
if blk0 is not None:
    report["blk0_attn_q_all_zero"] = bool((np.asarray(blk0.data) == 0).all())
    nan_path = os.path.join(WORK, "nan_blk0_attn_q.gguf")
    shutil.copyfile(M, nan_path)
    t = next(t for t in gguf.GGUFReader(nan_path).tensors if t.name == "blk.0.attnres_attn_q.weight")
    with open(nan_path, "r+b") as f:
        f.seek(int(t.data_offset))
        f.write(np.full(int(t.n_elements), np.nan, dtype=np.float32).tobytes())
    eqn = all(hashlib.sha256(ember(nan_path, a["prompt_ids"] + a["continuation_ids"])[0].encode()).hexdigest() == h
              for a, h in zip(ins, hashes))
    report["blk0_attn_q_not_read"] = {"nan_copy_logits_bit_equal": eqn}
    os.remove(nan_path)
report["per_item"] = per
report["pass"] = bool(report["f32"]["pass"] and report["embedded_ir_equals_ir_json"])
json.dump(report, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k not in ("per_item", "sha256")}, indent=1))
sys.exit(0 if report["pass"] else 1)
