#!/usr/bin/env python3
"""Long real inputs for tools/kv_quant_eval (R2, experiments/kv_quant_r2_2026-10-28), through a Hugging Face
tokenizer (ember's Qwen tokeniser equals HF's, LOCAL_LLM_STACK section 5.3).

    python3 tools/kvq_inputs.py HF_DIR BOOK.txt OUT.json [--qwen3] [--long OUT_LONG.json]

BOOK.txt: a public-domain novel (Project Gutenberg #1342, Pride and Prejudice, from its first chapter).
Chat transcripts: HuggingFaceH4/ultrachat_200k's test_sft split, conversations taken in file order.

OUT.json:
  tf      "book": the book's first 8,448 token ids, raw text, no template;
          "chat": whole UltraChat conversations, each through the chat template, concatenated to 8,192 ids;
  greedy  for each context C in 2048, 4096, 8192: two book questions over an excerpt (offsets 0 and 12,000
          tokens into the book) and one multi-turn chat whose history is UltraChat conversations joined
          into one conversation, each ending in a user turn with the generation prompt, at most C ids.
--qwen3 renders with enable_thinking=False (Qwen3's empty think block). --long writes two 3,072-id
prefixes (book, chat) as a list of id lists for ir_device_dump --ids.
"""
import argparse
import json

QUESTIONS = [
    "Summarise what has happened in this passage in a few sentences.",
    "List the characters who appear in this passage and say how they are related to each other.",
]


def template(tok, msgs, gen, qwen3):
    kw = {"enable_thinking": False} if qwen3 else {}
    ids = tok.apply_chat_template(msgs, add_generation_prompt=gen, **kw)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def ultrachat():
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "HuggingFaceH4/ultrachat_200k"
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("data/test_sft"))
    df = pd.read_parquet(hf_hub_download(repo, files[0], repo_type="dataset"))
    convs = []
    for conv in df["messages"]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in conv]
        if len(msgs) >= 2 and msgs[0]["role"] == "user":
            convs.append(msgs)
    return files[0], convs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("book")
    ap.add_argument("out")
    ap.add_argument("--qwen3", action="store_true")
    ap.add_argument("--long")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    text = open(a.book, encoding="utf-8").read()
    book = tok(text)["input_ids"]
    src, convs = ultrachat()

    chat = []
    for msgs in convs:
        chat += template(tok, msgs, False, a.qwen3)
        if len(chat) >= 8192:
            break
    tf = [{"name": "book", "ids": book[:8448]}, {"name": "chat", "ids": chat[:8192]}]

    greedy = []
    for C in (2048, 4096, 8192):
        for qi, (q, off) in enumerate(zip(QUESTIONS, (0, 12000))):
            budget = C - 64
            for _ in range(4):  # shrink the excerpt until the rendered prompt fits C
                excerpt = tok.decode(book[off:off + budget])
                ids = template(tok, [{"role": "user", "content": excerpt + "\n\n" + q}], True, a.qwen3)
                if len(ids) <= C:
                    break
                budget -= len(ids) - C + 8
            greedy.append({"name": f"book_q{qi + 1}_{C}", "ctx": C, "ids": ids})
        # one long conversation: UltraChat (user, assistant) turn pairs from conversation 200 on as history,
        # then the next pair's user turn, as many pairs as fit C
        pairs = [msgs[i:i + 2] for msgs in convs[200:260] for i in range(0, len(msgs) - 1, 2)
                 if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"]
        hist, best = [], None
        for pair in pairs:
            cand = template(tok, hist + [pair[0]], True, a.qwen3)
            if len(cand) > C:
                break
            best = cand
            hist = hist + pair
        greedy.append({"name": f"chat_{C}", "ctx": C, "ids": best})

    json.dump({"tokenizer": a.hf_dir, "book": a.book, "ultrachat": src, "tf": tf, "greedy": greedy}, open(a.out, "w"))
    for x in tf + greedy:
        print(x["name"], len(x["ids"]))
    if a.long:
        json.dump([book[:3072], chat[:3072]], open(a.long, "w"))


if __name__ == "__main__":
    main()
