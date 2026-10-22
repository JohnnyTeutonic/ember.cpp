#!/usr/bin/env python3
"""ember's interpreter on a tiny random third-party model, against paperkiln's numpy IR and HF.

    python3 tools/thirdparty_ember_check.py --family mamba --paperkiln ~/paperkiln --llamacpp ~/llamacpp_b6000 \
        --bin build/ir_logits --work DIR [--n 4] [--len 40] [--seed 0]
    python3 tools/thirdparty_ember_check.py --gguf pythia-160m-f32.gguf --paperkiln ~/paperkiln \
        --bin build/ir_logits --work DIR       # an official checkpoint, its HF files beside the GGUF

Without --gguf, a tiny randomly initialised HF transformers model of the family is converted by
llama.cpp's own convert_hf_to_gguf.py (paperkiln tools/thirdparty_synth.convert: f32, a placeholder
vocabulary). Its IR is built by paperkiln's tools/gguf_ir.py, and by ember's own builder
(ember_ir::find_ir with no EMBER_IR, as the server loads the file); the two must be equal, field by
field, except _comment. The same random token sequences then run through:
  HF      the PyTorch forward (f32),
  numpy   paperkiln's tools/ir_numpy.py on the IR (the reference),
  ember   tools/ir_logits.cpp on the IR it built, as one prefill and token by token through the state.
For every position: max |dlogit| over the range and argmax agreement, ember against numpy and
against HF, prefill and decode. Writes DIR/check.json; exit 1 unless the IRs are equal, every argmax
agrees and the worst ember-vs-numpy difference is within ROUNDING (--tol, 1e-5).

--ids FILE (tools/chat_prompt_ids.py: long chat prompts, past a window) checks those sequences on
their last row only (ember EMBER_IR_LAST=1, numpy last_only, HF logits_to_keep=1). For official
checkpoints the agreed bar is top-1 everywhere and greedy N/N, closeness judged against float64,
not the flat --tol (LOCAL_LLM_STACK section 5.3).
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
    elif family == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=512, n_embd=64, n_layer=3, n_head=4, n_positions=256, n_ctx=256, n_inner=256,
                         activation_function="gelu_new", tie_word_embeddings=True, initializer_range=0.2,
                         attn_implementation="eager")
        model = GPT2LMHeadModel(cfg)
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
    ap.add_argument("--family", choices=("mamba", "gptneox", "gpt2", "mistral_swa", "gemma3"))
    ap.add_argument("--gguf", help="an official checkpoint's GGUF, its HF config and weights in the same folder")
    ap.add_argument("--paperkiln", required=True)
    ap.add_argument("--llamacpp")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--ids", help="a JSON list of id lists (long prompts): last rows only")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.expanduser(a.paperkiln), "tools"))
    import gguf_ir
    import ir_numpy
    import thirdparty_synth
    import torch
    torch.set_grad_enabled(False)
    os.makedirs(a.work, exist_ok=True)
    if bool(a.family) == bool(a.gguf):
        raise SystemExit("thirdparty_ember_check: give --family (a synthetic model) or --gguf (a checkpoint)")
    if a.gguf:
        from transformers import AutoConfig, AutoModelForCausalLM
        gguf = os.path.expanduser(a.gguf)
        hf_dir = os.path.dirname(os.path.abspath(gguf))
        has_weights = any(f.endswith((".safetensors", ".bin")) for f in os.listdir(hf_dir))
        # Without the HF weights beside it (deleted for disk), ember is checked against numpy only.
        model = (AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32).eval() if has_weights
                 else None)
        hf_vocab = AutoConfig.from_pretrained(hf_dir).vocab_size
    else:
        model = tiny_hf(a.family, a.seed)
        gguf = thirdparty_synth.convert(model, a.family, os.path.expanduser(a.llamacpp), a.work)
    meta, shapes = gguf_ir.read_meta(gguf)
    doc = gguf_ir.build(meta, shapes)
    family = a.family or meta["general.architecture"]
    ir_path = os.path.join(a.work, f"{family}.ir.json")
    json.dump(doc, open(ir_path, "w"), indent=1)
    W = ir_numpy.load_weights(gguf)
    rng = np.random.default_rng(a.seed)
    hi = min(512, doc["model"]["vocab"]) if not a.gguf else min(doc["model"]["vocab"], hf_vocab)
    seqs = [rng.integers(0, hi, size=a.len).tolist() for _ in range(a.n)]
    if a.ids:
        seqs = json.load(open(a.ids))
    built_path = os.path.join(a.work, f"{family}.ember_built.ir.json")
    env = {k: v for k, v in os.environ.items() if k not in ("EMBER_IR", "EMBER_IR_BUILD")}
    r = subprocess.run([a.bin, gguf], input=json.dumps(seqs), capture_output=True, text=True,
                       env={**env, "EMBER_IR_PRINT": built_path, **({"EMBER_IR_LAST": "1"} if a.ids else {})})
    if r.returncode != 0:
        raise SystemExit(f"ir_logits failed: {r.stderr[-2000:]}")
    ember = json.loads(r.stdout)
    built = json.load(open(built_path))
    strip = lambda d: {k: v for k, v in d.items() if k != "_comment"}  # noqa: E731
    ir_equal = strip(built) == strip(doc)
    if not ir_equal:
        for k in sorted(set(built) | set(doc)):
            if k != "_comment" and built.get(k) != doc.get(k):
                print(f"IR field {k} differs:\n  ember  {json.dumps(built.get(k))[:600]}\n"
                      f"  python {json.dumps(doc.get(k))[:600]}", flush=True)
    print(json.dumps({"ember_built_ir_equals_gguf_ir": ir_equal}), flush=True)
    rows = []
    for k, ids in enumerate(seqs):
        if model is None:
            hf = None
        elif a.ids:
            try:
                hf = model(torch.tensor([ids]), logits_to_keep=1).logits[0, -1:].numpy()
            except TypeError:
                hf = model(torch.tensor([ids])).logits[0, -1:].numpy()
        else:
            hf = model(torch.tensor([ids])).logits[0].numpy()
        if hf is not None:
            hf = hf[:, :doc["model"]["vocab"]]
        ref = ir_numpy.forward(doc, W, ids, last_only=bool(a.ids))
        pre, dec = np.array(ember[k]["prefill"]), np.array(ember[k]["decode"])
        row = {"i": k, "positions": len(ref), "len": len(ids), "numpy_vs_hf_rel": rel(ref, hf) if hf is not None else None,
               "prefill_vs_numpy_rel": rel(pre, ref), "decode_vs_numpy_rel": rel(dec, ref),
               "prefill_vs_hf_rel": rel(pre, hf) if hf is not None else None, "decode_vs_prefill_rel": rel(dec, pre),
               "top1_prefill_vs_numpy": int((pre.argmax(1) == ref.argmax(1)).sum()),
               "top1_decode_vs_numpy": int((dec.argmax(1) == ref.argmax(1)).sum()),
               "top1_prefill_vs_hf": int((pre.argmax(1) == hf.argmax(1)).sum()) if hf is not None else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    extra = {}
    variants = []
    if family == "gemma3" and not a.ids:  # the variants dump every row: short sequences only  # rmsnorm weight_offset: the same model with every norm read as w + 0.25
        doc2 = json.loads(json.dumps(doc))
        for part in ("block", "head"):
            for nd in doc2[part]["nodes"]:
                if nd["op"] == "rmsnorm":
                    nd["weight_offset"] = 0.25
        variants.append(("weight_offset_0.25", doc2))
        # Linear RoPE scaling on the global layers (gemma3's rope.scaling.factor 4): per-layer freq_scale.
        doc3 = json.loads(json.dumps(doc))
        for nd in doc3["block"]["nodes"]:
            if nd["op"] == "rope":
                th = nd["theta"]["per_layer"]
                nd["freq_scale"] = {"per_layer": [1.0 if t == 10000.0 else 0.25 for t in th]}
        variants.append(("per_layer_freq_scale_0.25", doc3))
    for name, d2 in variants:
        ir2 = os.path.join(a.work, f"{family}.{name}.ir.json")
        json.dump(d2, open(ir2, "w"), indent=1)
        r2 = subprocess.run([a.bin, gguf], input=json.dumps(seqs[:2]), capture_output=True, text=True,
                            env={**env, "EMBER_IR": ir2})
        if r2.returncode != 0:
            raise SystemExit(f"ir_logits ({name}) failed: {r2.stderr[-2000:]}")
        e2 = json.loads(r2.stdout)
        rels = []
        for k, ids in enumerate(seqs[:2]):
            ref2 = ir_numpy.forward(d2, W, ids)
            rels += [rel(np.array(e2[k]["prefill"]), ref2), rel(np.array(e2[k]["decode"]), ref2)]
            if not (np.array(e2[k]["prefill"]).argmax(1) == ref2.argmax(1)).all():
                rels.append(1.0)
        extra[f"{name}_vs_numpy_rel"] = max(rels)
        print(json.dumps({f"{name}_vs_numpy_rel": max(rels)}), flush=True)
    worst = max([max(x["prefill_vs_numpy_rel"], x["decode_vs_numpy_rel"]) for x in rows] + list(extra.values()))
    top1 = all(x["top1_prefill_vs_numpy"] == x["top1_decode_vs_numpy"] == x["positions"] for x in rows)
    summary = {"family": family, "gguf": gguf if a.gguf else "synthetic", "gguf_arch": meta["general.architecture"],
               "ember_built_ir_equals_gguf_ir": ir_equal, "sequences": len(rows), "len": a.len,
               "worst_ember_vs_numpy_rel": worst,
               "worst_ember_vs_hf_rel": None if model is None else max(x["prefill_vs_hf_rel"] for x in rows),
               "worst_numpy_vs_hf_rel": None if model is None else max(x["numpy_vs_hf_rel"] for x in rows),
               "all_top1_agree": top1,
               "pass": ir_equal and top1 and worst <= a.tol, **extra, "rows": rows}
    json.dump(summary, open(os.path.join(a.work, "check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
