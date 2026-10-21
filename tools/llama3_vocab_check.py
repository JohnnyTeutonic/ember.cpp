#!/usr/bin/env python3
"""ember's vendored Llama 3 encoder against HF tokenizers on a llama3-scheme paperkiln vocabulary.

    python3 tools/llama3_vocab_check.py VOCAB.gguf TOKENIZER.json BIN [TEXT_FILE ...]

BIN is tools/llama3_vocab_check.cpp built. Texts: a fixed set of hard cases (digits, contractions,
newlines, non-ASCII letters, the chat specials, the chat template) plus up to 300 non-empty lines of
each TEXT_FILE (cut at document separators). Prints the agreement and the first differences; exit 1
on any difference. This checks ember's tokeniser before a model trained on the vocabulary exists;
the model's 40/40 parity gate comes after.
"""
import subprocess
import sys

from tokenizers import Tokenizer

HARD = [
    "Hello world", " leading space", "trailing space ", "two  spaces", "tabs\tand\nnewlines\n\n\nthree",
    "It's they're we've I'm you'll he'd SHE'S", "12345678901 3.14159 1,000,000 2026-10-21",
    "café naïve Zürich São Paulo Øresund", "日本語のテキスト", "Ελληνικά κείμενα", "emoji 🙂👍🏽 done",
    "<|user|>What is the capital of France?<|assistant|>Paris.<|endoftext|>",
    "x<|endoftext|>y", "<|user|> spaced <|assistant|>", "code: def f(x): return x**2  # comment",
    "URL https://example.com/a?b=c&d=e", "punctuation!!! ... ??? ---", "\r\nwindows\r\nlines",
]


def main(argv):
    gguf, tj, binary, files = argv[0], argv[1], argv[2], argv[3:]
    tok = Tokenizer.from_file(tj)
    texts = list(HARD)
    for f in files:
        with open(f, encoding="utf-8", errors="strict") as fh:
            raw = fh.read(4_000_000)
        texts += [x for x in raw.replace("<|endoftext|>\n", "\n").split("\n") if x.strip()][:300]
    blob = "\0".join(texts).encode("utf-8")
    r = subprocess.run([binary, gguf], input=blob, capture_output=True)
    if r.returncode != 0:
        sys.exit(r.stderr.decode(errors="replace"))
    sys.stderr.write(r.stderr.decode(errors="replace"))
    ember = [[int(x) for x in line.split()] for line in r.stdout.decode().split("\n")[:len(texts)]]
    hf = [tok.encode(t).ids for t in texts]
    diffs = [(t, a, b) for t, a, b in zip(texts, ember, hf) if a != b]
    print(f"{len(texts) - len(diffs)}/{len(texts)} texts encode identically (ember vs HF tokenizers); "
          f"{sum(len(h) for h in hf)} tokens")
    for t, a, b in diffs[:5]:
        k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        print(f"  DIFF at token {k}: {t[:80]!r}\n    ember {a[k:k + 6]}\n    hf    {b[k:k + 6]}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
