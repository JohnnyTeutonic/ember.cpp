#!/usr/bin/env python3
"""ember's interpreter on a tiny random third-party model, against paperkiln's numpy IR and HF.

    python3 tools/thirdparty_ember_check.py --family mamba --paperkiln ~/paperkiln --llamacpp ~/llamacpp_b6000 \
        --bin build/ir_logits --work DIR [--n 4] [--len 40] [--seed 0]

A tiny randomly initialised HF transformers model of the family is converted by llama.cpp's own
convert_hf_to_gguf.py (paperkiln tools/thirdparty_synth.convert: f32, a placeholder vocabulary), its
IR built by paperkiln's tools/gguf_ir.py, and the same random token sequences run through:
  HF      the PyTorch forward (f32),
  numpy   paperkiln's tools/ir_numpy.py on the IR (the reference),
  ember   tools/ir_logits.cpp (EMBER_IR), as one prefill and token by token through the state.
For every position: max |dlogit| over the range and argmax agreement, ember against numpy and
against HF, prefill and decode. Writes DIR/check.json; exit 1 unless every argmax agrees and the
worst ember-vs-numpy difference is within ROUNDING (--tol, 1e-5).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def tiny_hf(family, seed):
    import torch
    if family in ("mistral_swa", "gemma3"):  # paperkiln's own synthetic models (tools/thirdparty_synth.py)
        import thirdparty_synth
        return thirdparty_synth.hf_model(family, seed)
    torch.manual_seed(seed)
    if family == "gptneox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        cfg = GPTNeoXConfig(vocab_size=512, hidden_size=64, intermediate_size=256, num_hidden_layers=3,
                            num_attention_heads=4, rotary_pct=0.25, rotary_emb_base=10000, max_position_embeddings=256,
                            use_parallel_residual=True, hidden_act="gelu", tie_word_embeddings=False,
                            initializer_range=0.2, attn_implementation="eager")
        model = GPTNeoXForCausalLM(cfg)
    elif family == "mamba":
        from transformers import MambaConfig, MambaForCausalLM
        cfg = MambaConfig(vocab_size=512, hidden_size=64, state_size=16, num_hidden_layers=3, expand=2, conv_kernel=4,
                          time_step_rank=8, use_bias=False, use_conv_bias=True, tie_word_embeddings=True,
                          initializer_range=0.2, layer_norm_epsilon=1e-5)
        model = MambaForCausalLM(cfg)
    else:
        raise SystemExit(f"thirdparty_ember_check: unknown family {family}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.add_(0.2 * torch.randn_like(p))
    return model.float().eval()


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((np.abs(a - b).max(axis=-1) / (b.max(axis=-1) - b.min(axis=-1))).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("mamba", "gptneox", "mistral_swa", "gemma3"))
    ap.add_argument("--paperkiln", required=True)
    ap.add_argument("--llamacpp", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.expanduser(a.paperkiln), "tools"))
    import gguf_ir
    import ir_numpy
    import thirdparty_synth
    import torch
    torch.set_grad_enabled(False)
    os.makedirs(a.work, exist_ok=True)
    model = tiny_hf(a.family, a.seed)
    gguf = thirdparty_synth.convert(model, a.family, os.path.expanduser(a.llamacpp), a.work)
    meta, shapes = gguf_ir.read_meta(gguf)
    doc = gguf_ir.build(meta, shapes)
    ir_path = os.path.join(a.work, f"{a.family}.ir.json")
    json.dump(doc, open(ir_path, "w"), indent=1)
    W = ir_numpy.load_weights(gguf)
    rng = np.random.default_rng(a.seed)
    seqs = [rng.integers(0, 512, size=a.len).tolist() for _ in range(a.n)]
    r = subprocess.run([a.bin, gguf], input=json.dumps(seqs), capture_output=True, text=True,
                       env={**os.environ, "EMBER_IR": ir_path})
    if r.returncode != 0:
        raise SystemExit(f"ir_logits failed: {r.stderr[-2000:]}")
    ember = json.loads(r.stdout)
    rows = []
    for k, ids in enumerate(seqs):
        hf = model(torch.tensor([ids])).logits[0].numpy()
        ref = ir_numpy.forward(doc, W, ids)
        pre, dec = np.array(ember[k]["prefill"]), np.array(ember[k]["decode"])
        row = {"i": k, "positions": len(ids), "numpy_vs_hf_rel": rel(ref, hf),
               "prefill_vs_numpy_rel": rel(pre, ref), "decode_vs_numpy_rel": rel(dec, ref),
               "prefill_vs_hf_rel": rel(pre, hf), "decode_vs_prefill_rel": rel(dec, pre),
               "top1_prefill_vs_numpy": int((pre.argmax(1) == ref.argmax(1)).sum()),
               "top1_decode_vs_numpy": int((dec.argmax(1) == ref.argmax(1)).sum()),
               "top1_prefill_vs_hf": int((pre.argmax(1) == hf.argmax(1)).sum())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    extra = {}
    if a.family == "gemma3":  # rmsnorm weight_offset: the same model with every norm read as w + 0.25
        doc2 = json.loads(json.dumps(doc))
        for part in ("block", "head"):
            for nd in doc2[part]["nodes"]:
                if nd["op"] == "rmsnorm":
                    nd["weight_offset"] = 0.25
        ir2 = os.path.join(a.work, f"{a.family}.offset.ir.json")
        json.dump(doc2, open(ir2, "w"), indent=1)
        r2 = subprocess.run([a.bin, gguf], input=json.dumps(seqs[:2]), capture_output=True, text=True,
                            env={**os.environ, "EMBER_IR": ir2})
        if r2.returncode != 0:
            raise SystemExit(f"ir_logits (weight_offset) failed: {r2.stderr[-2000:]}")
        e2 = json.loads(r2.stdout)
        rels = []
        for k, ids in enumerate(seqs[:2]):
            ref2 = ir_numpy.forward(doc2, W, ids)
            rels += [rel(np.array(e2[k]["prefill"]), ref2), rel(np.array(e2[k]["decode"]), ref2)]
            if not (np.array(e2[k]["prefill"]).argmax(1) == ref2.argmax(1)).all():
                rels.append(1.0)
        extra["weight_offset_0.25_vs_numpy_rel"] = max(rels)
        print(json.dumps(extra), flush=True)
    worst = max([max(x["prefill_vs_numpy_rel"], x["decode_vs_numpy_rel"]) for x in rows] + list(extra.values()))
    top1 = all(x["top1_prefill_vs_numpy"] == x["top1_decode_vs_numpy"] == x["positions"] for x in rows)
    summary = {"family": a.family, "gguf_arch": meta["general.architecture"], "sequences": len(rows), "len": a.len,
               "worst_ember_vs_numpy_rel": worst, "worst_ember_vs_hf_rel": max(x["prefill_vs_hf_rel"] for x in rows),
               "worst_numpy_vs_hf_rel": max(x["numpy_vs_hf_rel"] for x in rows), "all_top1_agree": top1,
               "pass": top1 and worst <= a.tol, **extra, "rows": rows}
    json.dump(summary, open(os.path.join(a.work, "check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
