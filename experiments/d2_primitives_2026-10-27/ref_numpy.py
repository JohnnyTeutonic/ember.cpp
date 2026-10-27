"""A tools/ir_wasm_gate reference directory computed by keramos's numpy reference (tools/ir_numpy.py, run as a black
box through the runner's load_model), so the WebAssembly check and the WebGPU gate (tools/webgpu_parity/ir_gate.mjs)
compare against the reference itself rather than ember's native build.

    python3 ref_numpy.py KERAMOS_TREE MODEL_DIR NATIVE_REF_DIR OUT_DIR

Reads NATIVE_REF_DIR/ids.json (ir_wasm_gate ref's seeded prompts) and writes OUT_DIR/ids.json (the same prompts),
OUT_DIR/ref_<i>.bin (the reference's prefill logits at every position, float32 [len][vocab]) and OUT_DIR/ref.jsonl
(per prompt: the argmax per position and `gen` greedy tokens by plain argmax, as ir_wasm_gate's greedy). The greedy
run recomputes the whole prefix per token (the reference has no cache). Also prints the distance of the native
reference's logits to the numpy reference's (IR_SPEC 4.3's measure, max |d| / max(1, max |ref|), per prompt).
Resumes: a prompt with a row in ref.jsonl is skipped.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

tree, model_dir, native_dir, out_dir = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(tree / "tools"))
import ir_conformance  # noqa: E402  (the runner: load_model)
import ir_numpy  # noqa: E402  (the reference, a black box: forward(doc, W, ids))

doc, W, _meta, _ = ir_conformance.load_model(model_dir)
ids_doc = json.loads((native_dir / "ids.json").read_text())
V, L, G = ids_doc["vocab"], ids_doc["len"], ids_doc["gen"]
out_dir.mkdir(parents=True, exist_ok=True)
tmp = out_dir / "ids.json.tmp"
tmp.write_text(json.dumps(ids_doc))
os.replace(tmp, out_dir / "ids.json")
rows = out_dir / "ref.jsonl"
done = set()
if rows.exists():
    done = {json.loads(x)["i"] for x in rows.read_text().splitlines() if x.strip()}


def logits(ids):
    return np.asarray(ir_numpy.forward(doc, W, list(ids)), dtype=np.float32)


worst_native = 0.0
for i, ids in enumerate(ids_doc["ids"]):
    t0 = time.time()
    a = logits(ids)
    assert a.shape == (L, V), a.shape
    if i not in done:
        seq = list(ids)
        g = []
        lg = a
        for k in range(G):
            tok = int(np.argmax(lg[-1]))
            g.append(tok)
            if k + 1 < G:
                seq.append(tok)
                lg = logits(seq)
        b = out_dir / f"ref_{i}.bin"
        a.astype("<f4").tofile(str(b) + ".tmp")
        os.replace(str(b) + ".tmp", b)
        with rows.open("a") as f:
            f.write(json.dumps({"i": i, "top1": [int(x) for x in a.argmax(axis=1)], "decode_rel": 0.0, "greedy": g,
                                "ms": 1000 * (time.time() - t0)}) + "\n")
    nb = native_dir / f"ref_{i}.bin"
    if nb.exists():
        n = np.fromfile(nb, dtype="<f4").reshape(L, V)
        d = float(np.max(np.abs(n.astype(np.float64) - a)) / max(1.0, float(np.max(np.abs(a)))))
        worst_native = max(worst_native, d)
print(json.dumps({"model": str(model_dir), "prompts": len(ids_doc["ids"]), "len": L, "gen": G,
                  "native_vs_numpy_worst_of_max1": worst_native}))
