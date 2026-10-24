"""The mechanism on the CPU: paperkiln's numpy reference (tools/ir_numpy.py) with its rmsnorm changed for the
prompt rows only to what ember's CUDA prefill computed at d = 192 (rmsnorm_batch_cuda's block of 192 threads:
the tree reduction drops the partial of thread 2 at s = 3, i.e. every element with index % 3 == 2, while
still dividing by d). Decode rows (ember's per-token CUDA path, a 256-thread block) keep the full sum. Since
each row's norm is its own and attention is causal, one full-sequence forward per step reproduces prefill +
cached decode. Greedy-decodes the first reply (history empty) of the spot-check items and compares the text
with the VM's ember CUDA records and mtstudio's.

    emulate_bug.py GGUF IR.json CARRY5_ITEMS SPOT_DIR N MAX_NEW > emulate.json
"""
import json
import os
import sys
import types

import numpy as np

TOOLS = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools"
sys.path.insert(0, TOOLS)
import llama3_bpe  # noqa: E402

src = open(os.path.join(TOOLS, "ir_numpy.py")).read()
old = 't = t / np.sqrt((t * t).mean(axis=2, keepdims=True) + np.float32(nd["eps"])) * wt'
assert src.count(old) == 1
src = src.replace(old, 't = t / np.sqrt(_ms(t) + np.float32(nd["eps"])) * wt')
mod = types.ModuleType("ir_numpy_bug")
mod.__file__ = os.path.join(TOOLS, "ir_numpy.py")
BUG = {"rows": 0, "on": False}


def _ms(t):
    ms = (t * t).mean(axis=2, keepdims=True)
    if BUG["on"] and BUG["rows"] > 0 and t.shape[2] == 192:
        keep = (np.arange(192) % 3 != 2)
        tt = t[:BUG["rows"]].astype(np.float64)
        ms_bug = ((tt * tt) * keep).sum(axis=2, keepdims=True) / 192.0
        ms = ms.copy()
        ms[:BUG["rows"]] = ms_bug.astype(np.float32)
    return ms


mod.__dict__["_ms"] = _ms
exec(compile(src, mod.__file__, "exec"), mod.__dict__)

gguf, irp, items_path, spot, n, max_new = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
doc = json.load(open(irp))
W = mod.load_weights(gguf)
tok = llama3_bpe.from_gguf(gguf)
EOT, USER, ASSISTANT = (tok.token_to_id(s) for s in ("<|endoftext|>", "<|user|>", "<|assistant|>"))
items = [json.loads(x) for x in open(items_path) if x.strip()][::26][:n]
rec = {}
for eng in ("mtstudio", "ember_default"):
    rec[eng] = {r["id"]: r for r in (json.loads(x) for x in open(os.path.join(spot, eng, "m", "gen.jsonl")) if x.strip())}


def greedy(prompt, bug):
    ids = tok.encode("<|user|>" + prompt + "<|assistant|>").ids
    P = len(ids)
    out = []
    BUG.update(rows=P, on=bug)
    for _ in range(max_new):
        lg = mod.forward(doc, W, ids + out, last_only=True)[-1]
        lg[0] = -np.inf  # the <unk> ban
        t = int(lg.argmax())
        if t in (EOT, USER, ASSISTANT):
            return tok.decode(out, skip_special_tokens=False).strip(), "eos" if t == EOT else "turn"
        out.append(t)
    return tok.decode(out, skip_special_tokens=False).strip(), "length"


res = {"n": 0, "bug_eq_ember_cuda": 0, "clean_eq_mtstudio": 0, "rows": []}
for it in items:
    u = it["turns"][0]["user"]
    rb, sb = greedy(u, True)
    rc, sc = greedy(u, False)
    ec = rec["ember_default"][it["id"]]["self"][0]
    mt = rec["mtstudio"][it["id"]]["self"][0]
    row = {"id": it["id"], "emulated_bug": rb, "ember_cuda": ec["reply"], "numpy_clean": rc, "mtstudio": mt["reply"],
           "bug_eq_ember_cuda": rb == ec["reply"] and sb == ec["stop"], "clean_eq_mtstudio": rc == mt["reply"] and sc == mt["stop"]}
    res["n"] += 1
    res["bug_eq_ember_cuda"] += row["bug_eq_ember_cuda"]
    res["clean_eq_mtstudio"] += row["clean_eq_mtstudio"]
    res["rows"].append(row)
    print(json.dumps(row), file=sys.stderr, flush=True)
print(json.dumps(res, indent=1))
