"""Specs for the shape parity check: chat specs over the two pretrained tiny_chat models on disk (E1 L32,
E5 L8), and a train spec for a briefly trained L16 (shape_20m_L16.json's arch) on ext4."""
import json
import os
import sys

W = os.path.expanduser("~/ember_wt/shape_parity")
VOCAB = "/mnt/c/ml_artifacts/capacity/vocab/vocab_llama3_bpe.gguf"
SPECS = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/experiments/tiny_chat_2026-10/specs"
os.makedirs(W, exist_ok=True)
for name, src in (("tc_e1_L32_r1", "/mnt/c/ml_artifacts/tiny_chat/e1/pt_e1_L32_r1"),
                  ("tc_e5_s0_r1", "/mnt/c/ml_artifacts/tiny_chat/e5/pt_e5_s0_r1")):
    s = json.load(open(os.path.join(src, f"{name.replace('tc_', 'pt_')}_spec.json")))
    s["data"]["vocab"] = VOCAB
    s["data"]["corpus"] = ""
    s["out_dir"] = src
    json.dump(s, open(os.path.join(W, f"{name}.chat.json"), "w"), indent=1)
for L in ("16", "32", "8"):
    s = json.load(open(os.path.join(SPECS, f"shape_20m_L{L}.json")))
    s["name"] = f"sp_L{L}"
    s["data"].update(corpus=os.path.join(W, "dd_head.txt"), vocab=VOCAB, T=128)
    s["train"].update(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 60, batch=8, accum=1, lr=0.002,
                      eval_every=1000, eval_windows=4, checkpoint_every=20,
                      lr_schedule={"warmup_steps": 5, "decay": "cosine", "min_lr_frac": 0.1})
    s.pop("_shape", None)
    s["out_dir"] = os.path.join(W, f"sp_L{L}")
    json.dump(s, open(os.path.join(W, f"sp_L{L}.json"), "w"), indent=1)
src = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/data/human_dailydialog_v1/text.txt"
data = open(src, encoding="utf-8").read()
cut = data.rfind("<|endoftext|>", 0, 1_000_000) + len("<|endoftext|>")
open(os.path.join(W, "dd_head.txt"), "w", encoding="utf-8").write(data[:cut])
print("specs in", W, "corpus chars", cut)
