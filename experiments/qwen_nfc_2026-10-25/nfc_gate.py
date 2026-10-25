#!/usr/bin/env python3
"""The Qwen NFC gate (LOCAL_LLM_STACK.md section 11, item 4.6): ember's NFC and Qwen token ids against HF.

    python3 nfc_gate.py nfc NFC_CHECK OUT.json
    python3 nfc_gate.py tokens VOCAB_CHECK OLD_VOCAB_CHECK OUT_DIR

nfc: unicode_nfc.h (tools/nfc_check) against HF tokenizers' NFC normaliser, string for string, on
  - every code point alone (U+0001-U+10FFFF, surrogates excepted);
  - every canonical decomposition, decomposed (NFD), and every two-element one as its pair;
  - every combining mark (Python's 922) after a base, before and after one mark of every class;
  - Hangul: every syllable decomposed, every LV syllable + every T, every L + V, and L + mark + V;
  - 200,000 seeded random strings of 1-8 code points from those pools, ASCII and other scripts.
  The same strings against Python's unicodedata NFC are reported: they differ only where HF's older
  Unicode tables do (the code points tools/gen_nfc_tables.py drops), and the count of those is given.
tokens: tools/llama3_vocab_check (this build, VOCAB_CHECK; and OLD_VOCAB_CHECK, built from ember
  aa50a61's chat_tokenizer.h, before NFC) against HF's tokenizer for Qwen2.5-0.5B-Instruct and
  Qwen3-0.6B, on nfc_set_v1.jsonl: each text raw (tok(text, add_special_tokens=False)) and, for texts
  holding no added token's text, as one user message in the chat template
  (apply_chat_template(..., add_generation_prompt=True), whose text must equal ember's ChatML).
Exit 1 on any difference of this build from HF.
"""
import json
import os
import random
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
A = "/mnt/c/ml_artifacts/thirdparty"
MODELS = {
    "qwen2.5-0.5b-instruct": (A + "/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf",
                              A + "/qwen2.5-0.5b-instruct", "qwen2"),
    "qwen3-0.6b": (A + "/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf", A + "/qwen3-0.6b", "qwen3"),
}
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def cps():
    return [c for c in range(1, 0x110000) if not 0xD800 <= c <= 0xDFFF]


def run_nul(binary, texts, argv=()):
    r = subprocess.run([binary, *argv], input="\0".join(texts).encode("utf-8"), capture_output=True, timeout=7200)
    if r.returncode != 0:
        sys.exit(r.stderr.decode())
    return r.stdout, r.stderr.decode().strip()


def suites():
    allc = cps()
    marks = [c for c in allc if unicodedata.combining(chr(c))]
    by_class = {}
    for m in marks:
        by_class.setdefault(unicodedata.combining(chr(m)), m)
    canon = {}
    for c in allc:
        d = unicodedata.decomposition(chr(c))
        if d and not d.startswith("<"):
            canon[c] = [int(x, 16) for x in d.split()]
    S = {}
    S["single"] = [chr(c) for c in allc]
    S["decomposition"] = [unicodedata.normalize("NFD", chr(c)) for c in canon] + \
        ["".join(chr(x) for x in d) for d in canon.values() if len(d) == 2]
    S["marks"] = []
    for base in ("q", "a", chr(0x3B1)):
        for m in marks:
            for p in by_class.values():
                S["marks"] += [base + chr(m) + chr(p), base + chr(p) + chr(m)]
    hangul = [unicodedata.normalize("NFD", chr(c)) for c in range(0xAC00, 0xD7A4)]
    lv = [chr(0xAC00 + 28 * k) for k in range(399)]
    hangul += [s + chr(t) for s in lv for t in range(0x11A8, 0x11C3)]
    hangul += [chr(l) + chr(v) for l in range(0x1100, 0x1113) for v in range(0x1161, 0x1176)]
    hangul += [chr(l) + chr(0x301) + chr(v) for l in range(0x1100, 0x1113) for v in range(0x1161, 0x1176)]
    S["hangul"] = hangul
    rng = random.Random(20261025)
    pool = (list(range(0x41, 0x7B)) + [d[0] for d in canon.values()] + marks + list(canon) +
            list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11A7, 0x11C3)) +
            [0xAC00 + 28 * k for k in range(0, 399, 7)] + [0x3C, 0x3E, 0x3D, 0x338, 0x20, 0x4E00, 0x1F642])
    S["random"] = ["".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 8))) for _ in range(200000)]
    return S


def cmd_nfc(binary, out):
    from tokenizers import normalizers
    import tokenizers
    hf = normalizers.NFC().normalize_str
    res = {"tokenizers": tokenizers.__version__, "python_unicodedata": unicodedata.unidata_version, "suites": {}}
    ok = True
    for name, texts in suites().items():
        raw, err = run_nul(binary, texts)
        mine = raw.decode("utf-8").split("\0")[:-1]
        assert len(mine) == len(texts), (name, len(mine), len(texts))
        same_hf = same_py = changed = 0
        diffs = []
        for t, m in zip(texts, mine):
            h = hf(t)
            p = unicodedata.normalize("NFC", t)
            changed += h != t
            same_hf += m == h
            same_py += m == p
            if m != h and len(diffs) < 10:
                diffs.append({"text": [f"{ord(c):04X}" for c in t], "ember": [f"{ord(c):04X}" for c in m],
                              "hf": [f"{ord(c):04X}" for c in h]})
        ok &= same_hf == len(texts)
        res["suites"][name] = {"strings": len(texts), "changed_by_nfc": changed, "ember_eq_hf": same_hf,
                               "ember_eq_python": same_py, "differences_from_hf": diffs}
        print(name, res["suites"][name]["strings"], "eq hf", same_hf, "eq python", same_py, flush=True)
    res["nfc_check_stderr"] = err
    res["pass"] = ok
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    return 0 if ok else 1


def chatml(user, system):
    p = "<|im_start|>system\n" + system + "<|im_end|>\n" if system else ""
    return p + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"


def cmd_tokens(binary, old_binary, out_dir):
    from transformers import AutoTokenizer
    rows = [json.loads(x) for x in open(os.path.join(HERE, "nfc_set_v1.jsonl"), encoding="utf-8")]
    ok = True
    for model, (gguf, hf_dir, arch) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf_dir)
        added = [t.content for t in tok.added_tokens_decoder.values()]
        system = SYSTEM if SYSTEM in (tok.chat_template or "") else ""
        cases = [("raw", r, r["text"], tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
        text_same = n_chat = 0
        for r in rows:
            if any(a in r["text"] for a in added):
                continue
            msgs = [{"role": "user", "content": r["text"]}]
            text = chatml(r["text"], system)
            n_chat += 1
            text_same += tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == text
            cases.append(("chat", r, text, list(tok.apply_chat_template(msgs, add_generation_prompt=True))))
        summ = {"model": model, "gguf": gguf, "hf": hf_dir, "strings": len(rows), "chat_cases": n_chat,
                "template_text_same": text_same}
        for label, b in (("ember", binary), ("ember_before_nfc", old_binary)):
            raw, err = run_nul(b, [c[2] for c in cases], (gguf, arch))
            lines = raw.decode().split("\n")[:len(cases)]
            same, by_cat, diffs = {"raw": 0, "chat": 0}, {}, []
            for (kind, r, text, ref), line in zip(cases, lines):
                mine = [int(x) for x in line.split()]
                c = by_cat.setdefault(kind + ":" + r["category"], [0, 0])
                c[1] += 1
                if mine == ref:
                    same[kind] += 1
                    c[0] += 1
                elif len(diffs) < 10:
                    k = next((i for i, (a, b2) in enumerate(zip(mine, ref)) if a != b2), min(len(mine), len(ref)))
                    diffs.append({"i": r["i"], "kind": kind, "category": r["category"], "at": k,
                                  "ember": mine[max(0, k - 3):k + 5], "hf": ref[max(0, k - 3):k + 5]})
            summ[label] = {"ids_same": same, "by_category": {k: f"{a}/{n}" for k, (a, n) in sorted(by_cat.items())},
                           "differences": diffs, "stderr": err[-200:]}
        e = summ["ember"]["ids_same"]
        good = e["raw"] == len(rows) and e["chat"] == n_chat and text_same == n_chat
        summ["pass"] = good
        ok &= good
        json.dump(summ, open(os.path.join(out_dir, f"tokens_{model}.json"), "w", encoding="utf-8"), indent=1)
        print(model, "ember", e, "before NFC", summ["ember_before_nfc"]["ids_same"], "of raw", len(rows),
              "chat", n_chat, "template text", text_same, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "nfc":
        sys.exit(cmd_nfc(sys.argv[2], sys.argv[3]))
    sys.exit(cmd_tokens(sys.argv[2], sys.argv[3], sys.argv[4]))
