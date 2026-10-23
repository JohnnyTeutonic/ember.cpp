"""Smoke test of ember's block weight_index (tied-block recursion, paperkiln ir.hpp) on gpt2_tiny: the
IR's block applied 2 x its stored layers with weight_index [0, 1, .., L-1, 0, 1, .., L-1] (every
stored layer reused once more, each application with its own kv cache), and a schedule that
revisits the middle ([0] + [1 .. L-2] x 3 + [L-1], or [0, 0, 1, 1] at L = 2). Checks that ember
loads it, that prefill equals decode, that its logits equal paperkiln's ir_numpy over the same IR,
that the recursion changes the logits (the map is read), and that load refuses bad maps."""
import copy
import json
import os
import subprocess
import sys

import numpy as np

PK = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch"
GGUF = "/mnt/c/ml_artifacts/gpt2_tiny/gpt2_tiny.gguf"
BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
sys.path.insert(0, os.path.join(PK, "tools"))
import ir_numpy  # noqa: E402

base = json.load(open("/mnt/c/ml_artifacts/gpt2_tiny/gpt2_tiny.ir.json"))
L = base["block"]["repeat"]
W = ir_numpy.load_weights(GGUF)
seqs = [[5, 17, 300, 42, 9, 77, 128, 3], [11, 2, 250, 99, 1]]
middle = list(range(1, L - 1))
schedules = {"twice": list(range(L)) * 2,
             "middle_x3": [0] + middle * 3 + [L - 1] if L > 2 else [0, 0, 1, 1]}


def ember(doc, seqs_):
    path = os.path.expanduser("~/ember_wt/recursion_smoke.ir.json")
    json.dump(doc, open(path, "w"))
    r = subprocess.run([BIN, GGUF], input=json.dumps(seqs_), capture_output=True, text=True,
                       env={**os.environ, "EMBER_IR": path})
    return r


plain = json.loads(ember(base, seqs).stdout)
out = {"stored_layers": L}
for name, sched in schedules.items():
    doc = copy.deepcopy(base)
    doc["block"]["repeat"] = len(sched)
    doc["block"]["weight_index"] = {"per_layer": sched}
    r = ember(doc, seqs)
    if r.returncode:
        raise SystemExit(name + ": " + r.stderr[-1500:])
    em = json.loads(r.stdout)
    res = {"weight_index": sched}
    for k, ids in enumerate(seqs):
        ref = ir_numpy.forward(doc, W, ids)
        pre, dec = np.array(em[k]["prefill"]), np.array(em[k]["decode"])
        rng = ref.max(axis=-1) - ref.min(axis=-1)
        res[k] = {"vs_numpy_rel": float((np.abs(pre - ref).max(axis=-1) / rng).max()),
                  "decode_eq_prefill": bool(np.array_equal(dec, pre)),
                  "top1": f"{int((pre.argmax(1) == ref.argmax(1)).sum())}/{len(ids)}",
                  "changed_vs_plain": float(np.abs(pre - np.array(plain[k]["prefill"])).max())}
    out[name] = res

bad = {"wrong_length": {"per_layer": list(range(L))}, "negative": {"per_layer": [-1] + list(range(1, L)) * 2},
       "unused_layer": {"per_layer": [0] * (2 * L)}, "extra_stored": {"per_layer": list(range(L + 1)) * 2}}
refusals = {}
for name, wi in bad.items():
    doc = copy.deepcopy(base)
    doc["block"]["repeat"] = 2 * L if name != "extra_stored" else 2 * (L + 1)
    doc["block"]["weight_index"] = wi
    r = ember(doc, seqs[:1])
    msg = [ln for ln in r.stderr.splitlines() if "weight_index" in ln]
    refusals[name] = {"refused": r.returncode != 0, "message": msg[-1][-200:] if msg else r.stderr[-200:]}
out["refusals"] = refusals
print(json.dumps(out, indent=1))
