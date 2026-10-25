#!/usr/bin/env python3
"""ember greedy on GPT-2's raw prompts through its IR interpreter (tools/ir_adapter: the IR ember
builds from the GGUF's metadata, KV-cached decode), against HF's own cached-forward argmax and the
paperkiln receipt's hf_reply text.

    python3 greedy.py --model DIR --adapter BIN --receipt gate_n40.jsonl --prompts chat_prompts_v1.jsonl \
        --out greedy.jsonl [--n 40] [--max-new 32]

Per prompt: HF tokenizer ids (raw, as thirdparty_parity --raw), the prompt prefill's logits at every
position (ember vs HF: max |dlogit| over HF's range, top-1), then greedy: ember argmax with its KV
cache vs HF argmax with its KV cache, and each step's logits compared while the two agree.
Append-only JSONL; a rerun resumes after the last row.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np


def rel_rows(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b).max(axis=-1) / (b.max(axis=-1) - b.min(axis=-1))


def margin(v):
    s = np.sort(np.asarray(v, np.float64))
    return float(s[-1] - s[-2])


class Adapter:
    def __init__(self, binary):
        self.p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "result" not in r:
            raise SystemExit(f"adapter {method}: {r}")
        return r["result"]

    @staticmethod
    def logits(res):
        t = res["logits"]
        return np.frombuffer(bytes.fromhex(t["f32"]), dtype=">f4").astype(np.float32).reshape(t["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=32)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    gguf = next(os.path.join(a.model, f) for f in os.listdir(a.model) if f.endswith(".gguf"))
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    eos = {tok.eos_token_id} | {hf.generation_config.eos_token_id}
    prompts = [json.loads(x)["prompt"] for x in open(a.prompts, encoding="utf-8")][: a.n]
    receipt = [json.loads(x) for x in open(a.receipt, encoding="utf-8")]
    done = sum(1 for _ in open(a.out)) if os.path.exists(a.out) else 0
    ad = Adapter(a.adapter)
    with open(a.out, "a") as f:
        for i in range(done, len(prompts)):
            t0 = time.time()
            ids = tok(prompts[i])["input_ids"]
            # HF: the prompt's rows, then argmax with its KV cache (no generate).
            out = hf(torch.tensor([ids]), use_cache=True)
            past = out.past_key_values
            hf_prompt = out.logits[0].numpy().copy()
            hf_steps, hf_new = [hf_prompt[-1]], []
            for _ in range(a.max_new):
                t = int(np.argmax(hf_steps[-1]))
                hf_new.append(t)
                if t in eos or len(hf_new) == a.max_new:
                    break
                out = hf(torch.tensor([[t]]), past_key_values=past, use_cache=True)
                past = out.past_key_values
                hf_steps.append(out.logits[0, -1].numpy().copy())
            # ember: one prefill of the prompt, then one token at a time through its KV state.
            s = ad.call("open", gguf=gguf, ir=None)["session"]
            em_prompt = Adapter.logits(ad.call("forward", session=s, ids=ids, report=["logits"]))
            em_steps, em_new = [em_prompt[-1]], []
            for _ in range(a.max_new):
                t = int(np.argmax(em_steps[-1]))
                em_new.append(t)
                if t in eos or len(em_new) == a.max_new:
                    break
                em_steps.append(Adapter.logits(ad.call("forward", session=s, ids=[t], report=["logits"]))[0])
            ad.call("close", session=s)
            agree = next((k for k, (x, y) in enumerate(zip(em_new, hf_new)) if x != y), min(len(em_new), len(hf_new)))
            k = min(agree + 1, len(em_steps), len(hf_steps))  # steps fed identical prefixes
            pr = rel_rows(em_prompt, hf_prompt)
            st = rel_rows(np.array(em_steps[:k]), np.array(hf_steps[:k]))
            text = tok.decode(em_new, skip_special_tokens=True)
            row = {"i": i, "ids": ids, "ember_ids": em_new, "hf_ids": hf_new,
                   "greedy_same_hf": em_new == hf_new, "first_diff": None if em_new == hf_new else agree,
                   "ember_text_same_receipt": text == receipt[i]["hf_reply"],
                   "hf_text_same_receipt": tok.decode(hf_new, skip_special_tokens=True) == receipt[i]["hf_reply"],
                   "prompt_positions": len(ids),
                   "prompt_top1_same": int((em_prompt.argmax(1) == hf_prompt.argmax(1)).sum()),
                   "prompt_worst_rel": float(pr.max()), "first_rel": float(pr[-1]),
                   "steps_compared": int(k), "steps_top1_same": int(sum(int(np.argmax(x) == np.argmax(y))
                                                                    for x, y in zip(em_steps[:k], hf_steps[:k]))),
                   "steps_worst_rel": float(st.max()), "steps_min_hf_margin": min(margin(x) for x in hf_steps[:k]),
                   "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{i}: greedy {row['greedy_same_hf']} receipt {row['ember_text_same_receipt']} "
                  f"prompt top1 {row['prompt_top1_same']}/{len(ids)} rel {row['prompt_worst_rel']:.2e} "
                  f"steps {row['steps_top1_same']}/{k} rel {row['steps_worst_rel']:.2e} ({row['seconds']} s)", flush=True)
    ad.p.stdin.close()
    ad.p.wait()
    rows = [json.loads(x) for x in open(a.out)]
    summ = {"prompts": len(rows), "greedy_same_hf": sum(r["greedy_same_hf"] for r in rows),
            "ember_text_same_receipt_hf_reply": sum(r["ember_text_same_receipt"] for r in rows),
            "hf_text_same_receipt_hf_reply": sum(r["hf_text_same_receipt"] for r in rows),
            "prompt_positions": sum(r["prompt_positions"] for r in rows),
            "prompt_top1_same": sum(r["prompt_top1_same"] for r in rows),
            "steps_compared": sum(r["steps_compared"] for r in rows),
            "steps_top1_same": sum(r["steps_top1_same"] for r in rows),
            "worst_first_rel": max(r["first_rel"] for r in rows),
            "worst_prompt_rel": max(r["prompt_worst_rel"] for r in rows),
            "worst_step_rel": max(r["steps_worst_rel"] for r in rows),
            "min_hf_margin": min(r["steps_min_hf_margin"] for r in rows),
            "max_new": a.max_new, "hf": "native safetensors f32, argmax of its cached forward (no generate)",
            "ember": "tools/ir_adapter (CPU interpreter, IR built from the GGUF), prefill then KV-cached decode"}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
