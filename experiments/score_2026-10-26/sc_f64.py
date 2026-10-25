"""Where ember and mtstudio score differ most, a float64 evaluation of the same graph decides which is nearer exact
arithmetic: paperkiln's ir_numpy (keramos) on gguf_ir.build's IR with every float32 read as float64, as
ember.cpp/experiments/gpt2_2026-10-25/f64.py does. For the N requests with the largest |ember - mtstudio|
log-prob difference: each side's max |logprob - float64 logprob| over the request's continuation.
    sc_f64.py MODEL.gguf REQ.jsonl MTS.jsonl EMBER.jsonl N OUT.json"""
import json
import os
import sys
import types

import numpy

g, req_p, mts_p, emb_p, n, out_p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
PK = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch"
spec = os.path.join(PK, "spec", "tools")
sys.path.insert(0, spec if os.path.isfile(os.path.join(spec, "gguf_ir.py")) else os.path.join(PK, "tools"))
import gguf_ir  # noqa: E402
import ir_numpy  # noqa: E402

rd = lambda p: {json.loads(x)["key"]: json.loads(x) for x in open(p) if x.strip()}  # noqa: E731
reqs, mts, emb = rd(req_p), rd(mts_p), rd(emb_p)
diff = []
for k, m in mts.items():
    e = emb.get(k)
    if m.get("over_ctx") or not e or e.get("over_ctx"):
        continue
    diff.append((max(abs(a - b) for a, b in zip(m["logprobs"], e["logprobs"])), k))
worst = [k for _, k in sorted(diff, reverse=True)[:n]]
meta, shapes = gguf_ir.read_meta(g)
doc = gguf_ir.build(meta, shapes)
W = {k: (v.astype(numpy.float64) if hasattr(v, "astype") else v) for k, v in ir_numpy.load_weights(g).items()}
shim = types.ModuleType("np64")
shim.__dict__.update({k: getattr(numpy, k) for k in dir(numpy)})
shim.float32 = numpy.float64
ir_numpy.np = shim
rows = []
for k in worst:
    p, c = reqs[k]["prompt_ids"], reqs[k]["continuation_ids"]
    lg = numpy.asarray(ir_numpy.forward(doc, W, p + c[:-1]), numpy.float64)[len(p) - 1:len(p) - 1 + len(c)]
    mx = lg.max(axis=1, keepdims=True)
    lp = (lg - mx - numpy.log(numpy.exp(lg - mx).sum(axis=1, keepdims=True)))[numpy.arange(len(c)), c]
    de = float(numpy.abs(numpy.array(emb[k]["logprobs"]) - lp).max())
    dm = float(numpy.abs(numpy.array(mts[k]["logprobs"]) - lp).max())
    rows.append({"key": k, "ember_vs_mtstudio": dict(diff)[k] if False else max(abs(a - b) for a, b in zip(mts[k]["logprobs"], emb[k]["logprobs"])),
                 "ember_vs_f64": de, "mtstudio_vs_f64": dm})
    print(rows[-1], flush=True)
out = {"gguf": os.path.basename(g), "requests": rows, "ember_closer": sum(r["ember_vs_f64"] < r["mtstudio_vs_f64"] for r in rows),
       "worst_ember_vs_f64": max(r["ember_vs_f64"] for r in rows), "worst_mtstudio_vs_f64": max(r["mtstudio_vs_f64"] for r in rows)}
json.dump(out, open(out_p, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "requests"}))
