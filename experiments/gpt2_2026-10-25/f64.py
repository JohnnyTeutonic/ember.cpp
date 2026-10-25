#!/usr/bin/env python3
"""ember, ir_numpy f32 and HF against a float64 evaluation of the same IR (ir_numpy with every float32
read as float64), on thirdparty_ember_check's sequences (same seed, count and length).

    python3 f64.py --gguf M.gguf --bin ir_logits --paperkiln PK --out f64.json [--n 4] [--len 32] [--seed 0]
    python3 f64.py ... --greedy greedy.jsonl [--min-rel 1e-5]   # greedy.py's prompt + reply sequences
                                                                # whose decode steps left HF by > min-rel
Rows compared: every position of every sequence (max |dlogit| over float64's range, median, top-1).
"""
import argparse
import json
import os
import subprocess
import sys
import types

import numpy


def rel_rows(a, b):
    a, b = numpy.asarray(a, numpy.float64), numpy.asarray(b, numpy.float64)
    return numpy.abs(a - b).max(axis=-1) / (b.max(axis=-1) - b.min(axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--paperkiln", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", help="gpt2_greedy.py's receipt: check its prompt + reply sequences instead")
    ap.add_argument("--min-rel", type=float, default=1e-5)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.paperkiln, "tools"))
    import gguf_ir
    import ir_numpy
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.set_grad_enabled(False)
    hf_dir = os.path.dirname(os.path.abspath(a.gguf))
    meta, shapes = gguf_ir.read_meta(a.gguf)
    doc = gguf_ir.build(meta, shapes)
    hi = min(doc["model"]["vocab"], AutoConfig.from_pretrained(hf_dir).vocab_size)
    rng = numpy.random.default_rng(a.seed)
    seqs = [rng.integers(0, hi, size=a.len).tolist() for _ in range(a.n)]
    picked = None
    if a.greedy:  # the greedy rows whose decode steps left HF by more than --min-rel: prompt + reply as fed
        rows = [json.loads(x) for x in open(a.greedy)]
        picked = [r["i"] for r in rows if r["steps_worst_rel"] > a.min_rel]
        seqs = [r["ids"] + r["ember_ids"][:-1] for r in rows if r["steps_worst_rel"] > a.min_rel]
    env = {k: v for k, v in os.environ.items() if k not in ("EMBER_IR", "EMBER_IR_BUILD")}
    r = subprocess.run([a.bin, a.gguf], input=json.dumps(seqs), capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(r.stderr[-1500:])
    ember = json.loads(r.stdout)
    hf = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32).eval()
    hfl = [hf(torch.tensor([ids])).logits[0].numpy()[:, : doc["model"]["vocab"]] for ids in seqs]
    del hf
    W32 = ir_numpy.load_weights(a.gguf)
    np32 = ir_numpy.np
    r32 = [ir_numpy.forward(doc, W32, ids) for ids in seqs]
    W64 = {k: (v.astype(numpy.float64) if hasattr(v, "astype") else v) for k, v in W32.items()}
    del W32
    shim = types.ModuleType("np64")
    shim.__dict__.update({k: getattr(numpy, k) for k in dir(numpy)})
    shim.float32 = numpy.float64
    ir_numpy.np = shim
    r64 = [ir_numpy.forward(doc, W64, ids) for ids in seqs]
    ir_numpy.np = np32
    sides = {"ember_prefill": [e["prefill"] for e in ember], "ember_decode": [e["decode"] for e in ember],
             "numpy_f32": r32, "hf": hfl}
    out = {"gguf": os.path.basename(a.gguf), "sequences": len(seqs), "positions": sum(map(len, seqs))}
    out.update({"greedy_prompts": picked, "min_rel": a.min_rel} if picked is not None else
               {"len": a.len, "seed": a.seed})
    for name, xs in sides.items():
        rr = numpy.concatenate([rel_rows(x, y) for x, y in zip(xs, r64)])
        top1 = sum(int((numpy.asarray(x).argmax(1) == y.argmax(1)).sum()) for x, y in zip(xs, r64))
        out[f"{name}_vs_f64"] = {"max": float(rr.max()), "median": float(numpy.median(rr)), "top1_same": top1}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
