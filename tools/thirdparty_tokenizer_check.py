#!/usr/bin/env python3
"""ember's Qwen tokeniser and ChatML prompt against HF's, without the model (LOCAL_LLM_STACK.md section 7).

    python3 tools/thirdparty_tokenizer_check.py MODEL.gguf HF_DIR BIN PROMPTS.jsonl OUT.json [ARCH]

BIN is tools/llama3_vocab_check.cpp built; ARCH (default qwen2) selects ember's ChatML format. Texts:
hard cases (digits, contractions, whitespace runs, non-ASCII, the specials glued to words), every
prompt raw, and every prompt in ember's ChatML template. Each is encoded by ember and by HF's
tokenizer (raw text: tok(text, add_special_tokens=False); the templated prompt:
tok.apply_chat_template([user], add_generation_prompt=True), whose text must also equal ember's).
Writes the counts and the first differences to OUT.json; exit 1 on any difference.
"""
import json
import subprocess
import sys

HARD = [
    "Hello world", " leading space", "trailing space ", "two  spaces", "tabs\tand\nnewlines\n\n\nthree",
    "It's they're we've I'm you'll he'd SHE'S", "12345678901 3.14159 1,000,000 2026-10-21",
    "café naïve Zürich São Paulo Øresund", "日本語のテキスト", "Ελληνικά κείμενα", "emoji 🙂👍🏽 done",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>",
    "x<|endoftext|>y", "<|im_start|> spaced <|im_end|>", "code: def f(x): return x**2  # comment",
    "URL https://example.com/a?b=c&d=e", "punctuation!!! ... ??? ---", "\r\nwindows\r\nlines",
    "<tool_call>{\"a\": 1}</tool_call>", "   \n  \n", "١٢٣ ½ 二〇二六",
]
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def chatml(user, system):
    p = "<|im_start|>system\n" + system + "<|im_end|>\n" if system else ""
    return p + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"


def main(argv):
    gguf, hf_dir, binary, prompts_path, out = argv[:5]
    arch = argv[5] if len(argv) > 5 else "qwen2"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    system = SYSTEM if SYSTEM in (tok.chat_template or "") else ""
    prompts = [json.loads(x)["prompt"] for x in open(prompts_path, encoding="utf-8")]
    cases = [("hard", t, tok(t, add_special_tokens=False)["input_ids"]) for t in HARD]
    cases += [("raw", p, tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    text_same = 0
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = chatml(p, system)
        text_same += tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == text
        cases.append(("chat", text, list(tok.apply_chat_template(msgs, add_generation_prompt=True))))
    r = subprocess.run([binary, gguf, arch], input="\0".join(t for _, t, _ in cases).encode("utf-8"),
                       capture_output=True, timeout=3600)
    if r.returncode != 0:
        sys.exit(r.stderr.decode())
    lines = r.stdout.decode().split("\n")[:len(cases)]
    diffs, same = [], {"hard": 0, "raw": 0, "chat": 0}
    for (kind, text, ref), line in zip(cases, lines):
        mine = [int(x) for x in line.split()]
        if mine == ref:
            same[kind] += 1
        elif len(diffs) < 20:
            k = next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), min(len(mine), len(ref)))
            diffs.append({"kind": kind, "text": text[:200], "at": k, "ember": mine[max(0, k - 3):k + 5],
                          "hf": ref[max(0, k - 3):k + 5]})
    summ = {"gguf": gguf, "arch": arch, "system": system, "cases": {k: sum(1 for c in cases if c[0] == k) for k in same},
            "ids_same": same, "template_text_same": text_same, "prompts": len(prompts), "differences": diffs,
            "stderr": r.stderr.decode().strip()[-300:]}
    json.dump(summ, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summ[k] for k in ("cases", "ids_same", "template_text_same", "prompts")}))
    for d in diffs[:5]:
        print(d)
    return 0 if not diffs and text_same == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
