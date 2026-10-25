"""HF transformers reference for the /score gate (run in WSL, CPU, float32), as ember's
tools/thirdparty_hf_reference.py loads a third-party model: from the same GGUF (gguf_file=...).
    sc_hf.py HF_DIR GGUF_NAME REQ.jsonl OUT.jsonl
Per request: one forward over prompt + continuation[:-1] (no padding); for each continuation position, the
log-softmax in double, the argmax (mtstudio's banned names left out, none of which these vocabularies hold),
HF's logit range (max - min, the real-model rule's scale) and the top-1 margin. Resumes after the last row."""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

d, gguf, req_p, out_p = sys.argv[1:5]
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, gguf_file=gguf, torch_dtype=torch.float32).eval()
vocab = tok.get_vocab()
banned = sorted(vocab[s] for s in ("<unk>", "<s>", "</s>", "<pad>") if s in vocab)
reqs = [json.loads(x) for x in open(req_p, encoding="utf-8") if x.strip()]
done = set()
if os.path.exists(out_p):
    done = {json.loads(x)["key"] for x in open(out_p, encoding="utf-8") if x.strip()}
f = open(out_p, "a", encoding="utf-8", newline="\n")
for i, r in enumerate(reqs):
    if r["key"] in done:
        continue
    p, c = r["prompt_ids"], r["continuation_ids"]
    ids = torch.tensor([p + c[:-1]])
    lg = m(input_ids=ids).logits[0, len(p) - 1:len(p) - 1 + len(c)]
    rows = lg.double()
    lp = torch.log_softmax(rows, dim=-1)
    vals = [float(lp[j, c[j]]) for j in range(len(c))]
    rb = rows.clone()
    if banned:
        rb[:, banned] = -float("inf")
    top2 = torch.topk(rb, 2, dim=-1)
    rec = {"key": r["key"], "i": i, "prompt_tokens": len(p), "continuation_ids": c, "over_ctx": False,
           "logprobs": vals, "argmax": [int(x) for x in top2.indices[:, 0]], "sum": float(sum(vals)),
           "range": [float(x) for x in (lg.max(dim=-1).values - lg.min(dim=-1).values)],
           "margin": [float(x) for x in (top2.values[:, 0] - top2.values[:, 1])],
           "vocab_width": int(lg.shape[-1])}
    f.write(json.dumps(rec) + "\n")
    f.flush()
f.close()
print(f"{out_p}: {len(reqs)} requests, banned {banned}")
