#!/usr/bin/env python3
"""HF transformers reference for ember's third-party GGUF gate (LOCAL_LLM_STACK.md section 7).

    python3 tools/thirdparty_hf_reference.py --hf-dir DIR --gguf FILE.gguf --out REF.jsonl \
        [--prompts chat_prompts_v1.jsonl] [--offset 0] [--n 40] [--max-new 32] [--logits-dir DIR]

Per prompt, from HF's own tokenizer and chat template (one user message, add_generation_prompt) and
the model loaded from the same GGUF (gguf_file=..., float32): the templated text and its ids, the
first-token logits (saved as LOGITS_DIR/first_{i}.f32, raw little-endian float32, when given) and the
greedy continuation of up to --max-new tokens. Greedy is plain argmax: the checkpoint's
generation_config sampling and repetition_penalty are overridden (do_sample=False,
repetition_penalty=1.0), and generation stops at the generation_config's eos ids. The receipt is
append-only JSONL, one row per finished prompt; a rerun resumes after the last row.

Also checks, once, that HF's template text equals ember's ChatML format (chat_tokenizer.h) for one
user message and for a two-turn history, and records both in REF.jsonl.template.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

QWEN25_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def ember_chatml(turns, user, system):
    """ember's ChatFormat::chatml(system).prompt(turns, 0, user), in Python."""
    p = "<|im_start|>system\n" + system + "<|im_end|>\n" if system else ""
    for u, a in turns:
        p += "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n" + a + "<|im_end|>\n"
    return p + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--gguf", required=True, help="file name inside --hf-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--offset", type=int, default=0, help="first prompt (0-based line of --prompts)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--logits-dir", default=None)
    ap.add_argument("--system", default=None, help="the default system text ember's format writes (auto: Qwen2.5's "
                                                   "when the template holds it)")
    a = ap.parse_args(argv)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    system = a.system
    if system is None:
        system = QWEN25_SYSTEM if QWEN25_SYSTEM in (tok.chat_template or "") else ""
    # The template, once: one message, and a history of two turns.
    checks = []
    for turns, user in (([], "Hi"), ([("What is 2+2?", "4."), ("And 3+3?", "6.")], "Thanks!")):
        msgs = []
        for u, r in turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        msgs.append({"role": "user", "content": user})
        hf_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        mine = ember_chatml(turns, user, system)
        checks.append({"turns": len(turns), "hf": hf_text, "ember_format": mine, "same": hf_text == mine})
    json.dump({"system": system, "checks": checks}, open(a.out + ".template.json", "w", encoding="utf-8"), indent=1)
    print("template checks:", [c["same"] for c in checks], flush=True)

    hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, gguf_file=a.gguf, torch_dtype=torch.float32).eval()
    gc = hf.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    print("generation_config:", gc.to_dict(), flush=True)
    prompts = [json.loads(x)["prompt"] for x in open(a.prompts, encoding="utf-8")][a.offset:a.offset + a.n]
    done = sum(1 for _ in open(a.out, encoding="utf-8")) if os.path.exists(a.out) else 0
    if a.logits_dir:
        os.makedirs(a.logits_dir, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            if i < done:
                continue
            t0 = time.time()
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True))
            x = torch.tensor([ids])
            logits = hf(x).logits[0, -1].numpy().astype(np.float32)
            if a.logits_dir:
                tmp = os.path.join(a.logits_dir, f"first_{i}.f32.tmp")
                logits.tofile(tmp)
                os.replace(tmp, os.path.join(a.logits_dir, f"first_{i}.f32"))
            out = hf.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.max_new, do_sample=False,
                              repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                              eos_token_id=eos, pad_token_id=gc.pad_token_id if gc.pad_token_id is not None else eos[0])
            new = out[0, len(ids):].tolist()
            row = {"i": i, "line": a.offset + i, "prompt": p, "text": text, "ember_format_same": text == ember_chatml([], p, system),
                   "prompt_ids": ids, "hf_ids": new, "hf_reply": tok.decode(new, skip_special_tokens=True),
                   "first_top1": int(logits.argmax()), "first_range": float(logits.max() - logits.min()),
                   "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{i}: {len(ids)} prompt ids, {len(new)} new, {row['seconds']} s: {row['hf_reply'][:60]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
