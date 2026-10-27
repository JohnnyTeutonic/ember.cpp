"""CPU gate of ember's IR interpreter on paperkiln's ar_rec (llama_attnres under fixed recursion, the node key
"tensor_index": "application" of IR_SPEC 7.2; C:/ml_artifacts/attnres_gate_rec/ar_rec, paperkiln b9d5ce06, keramos
v1.3.5): 1 + [2] x 2 + 1 applications over 4 stored layers (weight_index [0, 1, 2, 1, 2, 3]), d 32, per-application
pseudo-queries rec_app.{0..5}.attnres_{attn,ffn}_q. The method of ../attnres_x5_2026-10-25/x5_gate_cpu.py, plus:
  - sha256 of every file against manifest.json's "files"; the document embedded in both GGUFs equal to ar_rec.ir.json;
  - per variant (F32, Q8_0), tools/ir_logits on the embedded document over the 40 score_in sequences:
      log-probs of the continuations against mtstudio score's (score_out.jsonl), worst |d| / max(1, max |ref|);
      logits at every position against keramos v1.3.5's ir_numpy.forward over the same GGUF (its weights as the
      gguf package dequantises them), worst max |d| / the reference row's range;
      ember's argmax at every scored position equal to the continuation (n/40); decode logits = prefill, bits;
  - the untagged document (the same document with every "tensor_index" key removed, served through EMBER_IR):
      ember's logits against ir_numpy's of that document (the same bound), and the gap between the two readings in
      paperkiln's measure (max |logit difference| / max(1, max |tagged|) over the first 8 prompts; paperkiln's numpy
      figure is manifest["untagged_gap"]), also over all 40 sequences;
  - tensors only the tags read: rec_app.4.* and rec_app.5.* (applications 4 and 5 bind rec_app.{4,5} when tagged,
      rec_app.{2,3} when not) overwritten with NaN: the tagged run's logits are NaN and the untagged run's are bit-equal
      to its run on the original file; rec_app.0.attnres_attn_q (written as zeros; its depth_attend has one source)
      overwritten with NaN gives bit-equal tagged logits.
Gate (F32 and Q8_0 alike): logits within 1e-5 of the row range of ir_numpy's over the same file (tagged and
untagged), greedy 40/40, decode = prefill bit for bit; the untagged gap above 1e-3 (F32: within 1e-3 of paperkiln's
numpy figure, relative to it; paperkiln measured the F32 file only); the NaN checks as stated. F32 also: log-probs within 1e-5 of max(1, max |ref|) of mtstudio's
(mtstudio scored the F32 weights, so for Q8_0 that distance is the quantisation's and is reported, not gated).
    python3 tig_gate_cpu.py OUT.json BUILD_DIR"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import gguf

D = "/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec"
KERAMOS = os.path.expanduser(os.environ.get("KERAMOS_TOOLS", "~/tig/keramos135/tools"))
WORK = os.path.expanduser("~/tig/work")
out_path, BUILD = sys.argv[1], sys.argv[2]
BIN = os.path.join(BUILD, "ir_logits")
sys.path.insert(0, KERAMOS)
import ir_numpy  # noqa: E402

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
report = {"dir": D, "bin": BIN, "keramos_tools": KERAMOS, "manifest_sha256": sha(os.path.join(D, "manifest.json")),
          "sha256": {}}
for f, h in man["files"].items():
    got = sha(os.path.join(D, f))
    report["sha256"][f] = {"sha256": got, "match": got == h}
report["sha256_all_match"] = all(v["match"] for v in report["sha256"].values())
if not report["sha256_all_match"]:
    print(json.dumps(report, indent=1))
    raise SystemExit("sha256 mismatch: not gating")

doc = json.load(open(os.path.join(D, "ar_rec.ir.json")))
untagged = copy.deepcopy(doc)
for sect in ("embed", "block", "head"):
    for nd in untagged[sect]["nodes"]:
        nd.pop("tensor_index", None)
report["tagged_nodes"] = [f"block.{k}:{nd['op']}:{nd['weight']}" for k, nd in enumerate(doc["block"]["nodes"])
                          if "tensor_index" in nd]
report["weight_index"] = doc["block"]["weight_index"]["per_layer"]
untagged_path = os.path.join(WORK, "ar_rec_untagged.ir.json")
json.dump(untagged, open(untagged_path, "w"))
report["untagged_doc_sha256"] = sha(untagged_path)

ins, outs = rows(os.path.join(D, "score_in.jsonl")), rows(os.path.join(D, "score_out.jsonl"))
seqs = [r["prompt_ids"] + r["continuation_ids"] for r in ins]
N = len(ins)
base_env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}


def ember(gguf_path, ir_path=None):
    env = dict(base_env)
    if ir_path:
        env["EMBER_IR"] = ir_path
    r = subprocess.run([BIN, gguf_path], input=json.dumps(seqs), capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(gguf_path + ": " + r.stderr[-1500:])
    em = json.loads(r.stdout)
    arr = lambda x: np.array([[np.nan if v is None else v for v in row] for row in x], dtype=np.float32)
    return r.stdout, [{"prefill": arr(e["prefill"]), "decode": arr(e["decode"])} for e in em]


def embedded_doc(path):
    r = gguf.GGUFReader(path)
    f = r.fields["paperkiln.ir"]
    return json.loads(bytes(f.parts[f.data[0]]).decode("utf-8"))


def vs_numpy(em, d, W):
    worst = 0.0
    for k, ids in enumerate(seqs):
        ref = ir_numpy.forward(d, W, list(ids))
        rng = ref.max(axis=-1) - ref.min(axis=-1)
        worst = max(worst, float((np.abs(em[k]["prefill"] - ref).max(axis=-1) / rng).max()))
    return worst


def score(em):
    worst, greedy, dec_eq = 0.0, 0, True
    for k, (a, b) in enumerate(zip(ins, outs)):
        assert b["continuation_ids"] == a["continuation_ids"]
        pre = em[k]["prefill"].astype(np.float64)
        dec_eq = dec_eq and bool(np.array_equal(em[k]["decode"], em[k]["prefill"]))
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        P, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[P - 1 + j, t] - lse[P - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        greedy += [int(pre[P - 1 + j].argmax()) for j in range(len(cont))] == cont
    return {"logprob_worst_rel_vs_mtstudio": worst, "greedy": f"{greedy}/{N}", "decode_eq_prefill": dec_eq}


def gap(a, b, n=None, prompt_only=False):
    g = 0.0
    for k in range(n or N):
        P = len(ins[k]["prompt_ids"]) if prompt_only else len(seqs[k])
        x, y = a[k]["prefill"][:P].astype(np.float64), b[k]["prefill"][:P].astype(np.float64)
        g = max(g, float(np.abs(x - y).max() / max(1.0, float(np.abs(x).max()))))
    return g


def nan_copy(src, names, dst):
    shutil.copyfile(src, dst)
    r = gguf.GGUFReader(dst)
    with open(dst, "r+b") as f:
        for t in r.tensors:
            if t.name in names:
                assert t.tensor_type.name == "F32", (t.name, t.tensor_type.name)
                f.seek(int(t.data_offset))
                f.write(np.full(int(t.n_elements), np.nan, dtype=np.float32).tobytes())
    got = {t.name for t in gguf.GGUFReader(dst).tensors if t.name in names and np.isnan(np.asarray(t.data)).all()}
    assert got == set(names), (got, names)


variants = {"f32": "ar_rec.gguf", "q8_0": "ar_rec_q8_0.gguf"}
all_pass = True
for v, fname in variants.items():
    path = os.path.join(D, fname)
    rep = {"gguf": fname}
    rep["embedded_doc_equals_ir_json"] = embedded_doc(path) == doc
    r = gguf.GGUFReader(path)
    qt = {t.name: t.tensor_type.name for t in r.tensors if "attnres" in t.name}
    rep["pseudo_queries"] = len(qt)
    rep["pseudo_query_types"] = sorted(set(qt.values()))
    rep["matrix_types"] = sorted({t.tensor_type.name for t in r.tensors if len(t.shape) == 2})
    W = ir_numpy.load_weights(path)
    text_t, em_t = ember(path)
    text_u, em_u = ember(path, untagged_path)
    g = score(em_t)
    g["logits_worst_rel_vs_ir_numpy"] = vs_numpy(em_t, doc, W)
    rep["tagged"] = g
    u = score(em_u)
    u["logits_worst_rel_vs_ir_numpy_untagged"] = vs_numpy(em_u, untagged, W)
    rep["untagged"] = u
    rep["untagged_gap"] = {
        "ember_first8_prompts": gap(em_t, em_u, 8, prompt_only=True),
        "ember_all40_sequences": gap(em_t, em_u),
        "paperkiln_numpy_first8": man["untagged_gap"],
    }
    rel = abs(rep["untagged_gap"]["ember_first8_prompts"] - man["untagged_gap"]) / man["untagged_gap"]
    rep["untagged_gap"]["ember_vs_paperkiln_rel"] = rel
    # NaN in the tensors only the tags read (applications 4 and 5).
    only_tagged = [f"rec_app.{a}.attnres_{s}_q.weight" for a in (4, 5) for s in ("attn", "ffn")]
    p4 = os.path.join(WORK, f"ar_rec_{v}_nan_app45.gguf")
    nan_copy(path, only_tagged, p4)
    _, em_t4 = ember(p4)
    text_u4, _ = ember(p4, untagged_path)
    rep["nan_rec_app_4_5"] = {"tensors": only_tagged,
                              "tagged_all_nan": bool(all(np.isnan(e["prefill"]).all() for e in em_t4)),
                              "untagged_bit_equal": text_u4 == text_u}
    os.remove(p4)
    p0 = os.path.join(WORK, f"ar_rec_{v}_nan_app0_attn.gguf")
    nan_copy(path, ["rec_app.0.attnres_attn_q.weight"], p0)
    text_t0, _ = ember(p0)
    rep["rec_app0_attn_q_all_zero"] = bool((np.asarray(next(t for t in r.tensors
                                                            if t.name == "rec_app.0.attnres_attn_q.weight").data) == 0).all())
    rep["rec_app0_attn_q_not_read"] = {"nan_copy_tagged_bit_equal": text_t0 == text_t}
    os.remove(p0)
    open(os.path.join(WORK, f"cpu_{v}_tagged.json"), "w").write(text_t)  # for the device / WASM comparisons
    open(os.path.join(WORK, f"cpu_{v}_untagged.json"), "w").write(text_u)
    rep["pass"] = bool(
        rep["embedded_doc_equals_ir_json"] and rep["pseudo_query_types"] == ["F32"]
        and (v != "f32" or g["logprob_worst_rel_vs_mtstudio"] <= 1e-5) and g["logits_worst_rel_vs_ir_numpy"] <= 1e-5
        and g["greedy"] == f"{N}/{N}" and g["decode_eq_prefill"]
        and u["logits_worst_rel_vs_ir_numpy_untagged"] <= 1e-5 and u["decode_eq_prefill"]
        and rep["untagged_gap"]["ember_first8_prompts"] > 1e-3 and (v != "f32" or rel <= 1e-3)
        and rep["nan_rec_app_4_5"]["tagged_all_nan"] and rep["nan_rec_app_4_5"]["untagged_bit_equal"]
        and rep["rec_app0_attn_q_not_read"]["nan_copy_tagged_bit_equal"])
    all_pass = all_pass and rep["pass"]
    report[v] = rep
    print(v, json.dumps({k: rep[k] for k in ("tagged", "untagged", "untagged_gap", "pass")}), flush=True)

report["pass"] = all_pass
json.dump(report, open(out_path, "w"), indent=1)
print(json.dumps(report, indent=1))
sys.exit(0 if all_pass else 1)
