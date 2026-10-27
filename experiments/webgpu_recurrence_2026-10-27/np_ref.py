"""keramos's numpy reference (tools/ir_numpy.py, used as a black-box oracle) in ir_gate.mjs's reference format, on
the prompts of ember CPU's reference (ids.json of tools/ir_wasm_gate ref): per prompt the prefill logits at every
position (ref_<i>.bin, float32 [len][vocab]) and its greedy continuation (ref.jsonl {i, greedy, top1}), each
greedy step a fresh forward over the whole sequence. Resumes per prompt.

    python3 np_ref.py MODEL.gguf DOC.json|- EMBER_REF_DIR OUT_DIR K
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "keramos", "tools"))
import ir_numpy  # noqa: E402

gguf, doc_path, ref, out, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
if doc_path == "-":
    doc = json.loads(subprocess.run([sys.executable, os.path.join(H, "gguf_doc.py"), "get", gguf], capture_output=True,
                                    text=True, check=True).stdout)
else:
    doc = json.load(open(doc_path, encoding="utf-8"))
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(ref, "ids.json"), os.path.join(out, "ids.json"))
ids = json.load(open(os.path.join(out, "ids.json")))
W = ir_numpy.load_weights(gguf)
rows_path = os.path.join(out, "ref.jsonl")
done = set()
if os.path.exists(rows_path):
    done = {json.loads(x)["i"] for x in open(rows_path) if x.strip()}
for i in range(min(k, ids["n"])):
    if i in done:
        continue
    p = list(ids["ids"][i])
    lg = np.asarray(ir_numpy.forward(doc, W, p), np.float32)
    tmp = os.path.join(out, f"ref_{i}.bin.tmp")
    lg.tofile(tmp)
    os.replace(tmp, os.path.join(out, f"ref_{i}.bin"))
    seq, g = list(p), []
    last = lg[-1]
    for s in range(ids["gen"]):
        t = int(np.argmax(last))
        g.append(t)
        seq.append(t)
        if s + 1 < ids["gen"]:
            last = np.asarray(ir_numpy.forward(doc, W, seq, last_only=True), np.float32).reshape(-1)
    with open(rows_path, "a") as f:
        f.write(json.dumps({"i": i, "greedy": g, "top1": [int(x) for x in lg.argmax(1)]}) + "\n")
    print("numpy prompt", i, flush=True)
