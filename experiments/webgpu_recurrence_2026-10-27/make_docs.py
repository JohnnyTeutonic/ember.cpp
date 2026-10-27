"""The recurrence documents of the WebGPU gate that the suite does not ship, into docs/:

- <fam>_rec.ir.json for kimi_tiny, srd_tiny and mamba_tiny (C:/ml_artifacts): each file's own document with its
  hand-written recurrent ops replaced by their recurrences (keramos tools/ir_recurrence.py `rewrite`, IR_SPEC
  6.10.6), the slot keeping its name and tensors;
- mamba-130m_rec.ir.json: the document ember builds for state-spaces/mamba-130m (bcheck_mamba-130m), rewritten
  the same way (conv1d_causal and selective_scan as recurrences, on a real checkpoint);
- kimi_tiny_custom.ir.json: kimi_tiny with its linear attention replaced by a recurrence of this experiment's own,
  for what the rewrites leave out at model level: init bodies over per-application weights ({i}), a decay
  from a weight, an offset read with zero padding, a sum, a state tensor of another shape, a bare index.

    python make_docs.py
"""
import copy
import json
import os
import subprocess
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "keramos", "tools"))
import ir_recurrence  # noqa: E402

A = r"C:\ml_artifacts"
U = r"\\wsl.localhost\Ubuntu\home\jonat"
D = os.path.join(H, "docs")
os.makedirs(D, exist_ok=True)


def gguf_doc(path):
    return json.loads(subprocess.run([sys.executable, os.path.join(H, "gguf_doc.py"), "get", path],
                                     capture_output=True, text=True, check=True, encoding="utf-8").stdout)


def save(name, doc):
    p = os.path.join(D, name + ".ir.json")
    json.dump(doc, open(p, "w", encoding="utf-8"), indent=1)
    ops = [n["op"] for n in doc["block"]["nodes"]]
    print(p, "recurrence nodes:", ops.count("recurrence"), "hand-written recurrent ops left:",
          sum(ops.count(o) for o in ("linear_attention", "ssm_scan", "conv1d_causal", "selective_scan")))


for fam in ("kimi_tiny", "srd_tiny", "mamba_tiny"):
    save(fam + "_rec", ir_recurrence.rewrite(gguf_doc(os.path.join(A, fam, fam + ".gguf"))))
save("mamba-130m_rec", ir_recurrence.rewrite(json.load(open(
    U + r"\ember_wt\bcheck_mamba-130m\mamba.ember_built.ir.json", encoding="utf-8"))))

doc = copy.deepcopy(gguf_doc(os.path.join(A, "kimi_tiny", "kimi_tiny.gguf")))
blk = doc["block"]
j = next(k for k, n in enumerate(blk["nodes"]) if n["op"] == "linear_attention")
la = blk["nodes"][j]
H_, E_ = la["heads"], la["head_dim"]
node = {
    "op": "recurrence", "in": la["in"], "state": la["state"], "out": la["out"],
    "index": {"h": H_, "e": E_, "f": E_},
    "inputs": ["q[h,e]", "k[h,e]", "v[h,e]"],
    "weights": {"h0[h,e]": "blk.{i}.attn_norm.bias", "g[h,e]": "blk.{i}.attn_norm.weight"},
    "init": ["num[h,e] = 0.1 * h0[h,e]", "den[h,e] = 1 + abs(h0[h,e])"],
    "let": ["pk[h,e] = where(k[h,e], k[h,e] + 1, exp(min(0, k[h,e]))) + 0.25 * relu(k[h,e - 1])",
            "a[h,e] = 0.5 + 0.5 * sigmoid(g[h,e])"],
    "update": ["num[h,e] = a[h,e] * num[h,e] + pk[h,e] * v[h,e]", "den[h,e] = a[h,e] * den[h,e] + pk[h,e]",
               "c[h] = 0.9 * c[h] + sum(f, pk[h,f] * q[h,f]) / %d" % E_],
    "readout": "y[h,e] = (where(q[h,e], q[h,e] + 1, exp(min(0, q[h,e]))) * num[h,e]) / max(den[h,e], 0.000001)"
               " + 0.01 * tanh(c[h]) + 0.001 * e",
}
blk["nodes"][j] = node
blk["state"] = [ir_recurrence.slot_for(node, s["name"]) if s["name"] == la["state"] else s for s in blk["state"]]
doc["family_hint"] = "kimi_recurrence"
doc["_comment"] = ("kimi_tiny with its linear attention replaced by a recurrence of the WebGPU recurrence gate "
                   "(ember experiments/webgpu_recurrence_2026-10-27/make_docs.py); not a trained model")
save("kimi_tiny_custom", doc)
