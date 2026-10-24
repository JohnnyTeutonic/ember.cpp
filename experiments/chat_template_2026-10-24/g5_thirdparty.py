#!/usr/bin/env python3
"""G5 for the third-party records (LOCAL_LLM_STACK.md section 3, gates): ember's rendered prompt ids against
HF transformers' apply_chat_template, with no model.

- ember: tools/chat_template_check render GGUF, which loads the GGUF's tokenizer and resolves its template
  as the session does (chat_gguf.h, make_chat_setup) and renders each request through chat_request.h's
  /chat or OpenAI mapping and the context rule (the server's and WebAssembly's code); its ids are the
  vocabulary's own encoding (Qwen: paperkiln's llama.cpp port with the qwen2 pre-tokeniser; Gemma 3:
  spm_tokenizer.h, llama.cpp b6000's SentencePiece session).
- HF: AutoTokenizer.from_pretrained(the model's HF directory: tokenizer_config.json's template; Gemma 3's
  chat_template.jinja), apply_chat_template(messages, add_generation_prompt=True[, enable_thinking]), ids
  and text.
- Conversations: the 200 test prompts of chat_prompts_v1.jsonl (one user message); 25 tokeniser hard
  cases (non-ASCII scripts, emoji, whitespace and newline runs, CRLF, digits, added-token text), and 4
  with Unicode whitespace at a message's edges (Gemma 3's trim, Python str.strip() in both HF's Jinja and
  the record since paperkiln 7a6c7659); 40 with a system
  message (half as "system", half as a leading history entry; on the OpenAI path, half as developer
  messages); 40 with two exchanges of history (the prompts' reference replies; for Qwen3 half of them
  carry <think> reasoning, the case strip_think handles); 20 with a system message and two exchanges;
  20 with whitespace around every message (Gemma 3's trim); 40 through the OpenAI mapping. Qwen3:
  thinking default (HF's: on), on and off (enable_thinking False appends the empty think block) on
  the single-turn and history sets; Qwen2.5 and Gemma 3: thinking false on 20 prompts, which their
  templates ignore.
Writes g5_thirdparty.json beside this file (counts and the first differences per set)."""
import json
import os
import subprocess

from common import A, CHAT_PROMPTS, CHECK, HERE, WORK, rows, sha256

MODELS = {
    "qwen2.5-0.5b-instruct": ("qwen2.5-0.5b-instruct-fp16.gguf", "chatml-qwen2.5/1"),
    "qwen3-0.6b": ("Qwen3-0.6B-Q8_0.gguf", "chatml-qwen3/1"),
    "gemma-3-270m-it": ("gemma-3-270m-it-f32.gguf", "gemma3/1"),
}
SYSTEMS = ["You are a kind and patient friend.", "Answer in one short sentence.",
           "You are a helpful assistant. Be warm and encouraging.", "Keep your reply brief and friendly."]
P = rows(CHAT_PROMPTS)
# Tokeniser hard cases (tools/thirdparty_tokenizer_check.py's, minus the ones holding a template's own
# markers, which every template refuses), plus whitespace runs and other added-token text.
HARD = [
    "Hello world", " leading space", "trailing space ", "two  spaces", "tabs\tand\nnewlines\n\n\nthree",
    "It's they're we've I'm you'll he'd SHE'S", "12345678901 3.14159 1,000,000 2026-10-21",
    "café naïve Zürich São Paulo Øresund", "日本語のテキスト", "Ελληνικά κείμενα", "emoji 🙂👍🏽 done",
    "code: def f(x): return x**2  # comment", "URL https://example.com/a?b=c&d=e", "punctuation!!! ... ??? ---",
    "\r\nwindows\r\nlines", "<tool_call>{\"a\": 1}</tool_call>", "١٢٣ ½ 二〇二六", "many      spaces and\t\t\ttabs",
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nthirty-four newlines", "<unk> <mask> <pad> text",
    "x<unused0>y", "▁already▁escaped", "zero​width", "in between nbsp", "a　ideographic space",
]
# Unicode (non-ASCII) whitespace at a message's edges: HF's Jinja trim (Gemma 3) strips it, and since
# paperkiln 7a6c7659 the record's content "trim" strips exactly what Python's str.strip() strips (gated).
UNICODE_EDGE = [" nbsp ", "　ideographic　", " em space", "x "]


def think(i, reply):
    return f"<think>\nThe user wrote: {P[i]['prompt']}\nI should answer kindly.\n</think>\n\n{reply}"


def conversations(model):
    """(set name, messages, thinking or None, path, system as a history entry)."""
    out = []
    for i, p in enumerate(P):
        out.append(("single200", [{"role": "user", "content": p["prompt"]}], None, "chat", False))
    for h in HARD:
        out.append(("hard25", [{"role": "user", "content": h}], None, "chat", False))
    for h in UNICODE_EDGE:
        out.append(("unicode_edge_whitespace4", [{"role": "user", "content": h}], None, "chat", False))
    for i in range(40):
        out.append(("system40", [{"role": "system", "content": SYSTEMS[i % 4]},
                                 {"role": "user", "content": P[i]["prompt"]}], None, "chat", i % 2 == 1))
    for i in range(40):
        a, b, c = P[40 + i], P[80 + i], P[120 + i]
        ra, rb = a["reference"], b["reference"]
        if model == "qwen3-0.6b" and i % 2 == 0:
            ra, rb = think(40 + i, ra), think(80 + i, rb)
        out.append(("history40", [{"role": "user", "content": a["prompt"]}, {"role": "assistant", "content": ra},
                                  {"role": "user", "content": b["prompt"]}, {"role": "assistant", "content": rb},
                                  {"role": "user", "content": c["prompt"]}], None, "chat", False))
    for i in range(20):
        a, b, c = P[160 + i], P[180 + i], P[i]
        ra, rb = a["reference"], b["reference"]
        if model == "qwen3-0.6b":
            ra = think(160 + i, ra)
        out.append(("system_history20", [{"role": "system", "content": SYSTEMS[(i + 1) % 4]},
                                         {"role": "user", "content": a["prompt"]}, {"role": "assistant", "content": ra},
                                         {"role": "user", "content": b["prompt"]}, {"role": "assistant", "content": rb},
                                         {"role": "user", "content": c["prompt"]}], None, "chat", i % 2 == 1))
    for i in range(20):
        out.append(("whitespace20", [{"role": "system", "content": " " + SYSTEMS[i % 4] + " "},
                                     {"role": "user", "content": "  " + P[i]["prompt"] + " \n"},
                                     {"role": "assistant", "content": "\n" + P[i]["reference"] + "  "},
                                     {"role": "user", "content": "\t" + P[20 + i]["prompt"] + "\n\n"}], None, "chat",
                    False))
    for i in range(40):
        m = [{"role": "user", "content": P[i]["prompt"]}]
        if i % 4 == 1:
            m.insert(0, {"role": "developer" if i % 8 == 1 else "system", "content": SYSTEMS[i % 4]})
        if i % 4 >= 2:
            m = [{"role": "user", "content": P[100 + i]["prompt"]},
                 {"role": "assistant", "content": think(100 + i, P[100 + i]["reference"]) if model == "qwen3-0.6b"
                  else P[100 + i]["reference"]}] + m
        out.append(("openai40", m, None, "openai", False))
    if model == "qwen3-0.6b":
        for name, msgs, _, path, hs in list(out):
            if name in ("single200", "history40", "system_history20"):
                out.append((name + "_thinking_on", msgs, True, path, hs))
                out.append((name + "_thinking_off", msgs, False, path, hs))
    else:
        for i in range(20):
            out.append(("thinking_false_ignored20", [{"role": "user", "content": P[i]["prompt"]}], False, "chat", False))
    return out


def ember_request(msgs, thinking, path, system_in_history):
    if path == "openai":
        body = {"messages": msgs}
        if thinking is not None:
            body["thinking"] = thinking
        return {"openai": body}
    m = list(msgs)
    req = {}
    if m[0]["role"] == "system":
        s = m.pop(0)["content"]
        if system_in_history:
            req["history"] = [{"role": "system", "content": s}]
        else:
            req["system"] = s
    req.setdefault("history", []).extend({"role": x["role"], "content": x["content"]} for x in m[:-1])
    req["user_input"] = m[-1]["content"]
    if thinking is not None:
        req["thinking"] = thinking
    return {"chat": req}


def hf_messages(msgs):
    return [{"role": "system" if x["role"] == "developer" else x["role"], "content": x["content"]} for x in msgs]


def main():
    from transformers import AutoTokenizer
    import transformers
    report = {"prompts": CHAT_PROMPTS, "prompts_sha256": sha256(CHAT_PROMPTS), "transformers": transformers.__version__,
              "models": {}}
    for model, (gguf_name, want) in MODELS.items():
        d = f"{A}/thirdparty/{model}"
        gguf = f"{d}/{gguf_name}"
        tok = AutoTokenizer.from_pretrained(d)
        convs = conversations(model)
        reqs = [ember_request(m, t, p, h) for _, m, t, p, h in convs]
        inp = f"{WORK}/g5tp_{model}_in.jsonl"
        with open(inp, "w", encoding="utf-8") as f:
            for r in reqs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        r = subprocess.run([CHECK, "render", gguf], stdin=open(inp, encoding="utf-8"), capture_output=True, text=True)
        # split on newlines only: str.splitlines also splits on U+2028, which a case holds.
        lines = [json.loads(x) for x in r.stdout.split("\n") if x.strip()]
        head, outs = lines[0], lines[1:]
        assert len(outs) == len(convs), (model, r.returncode, r.stderr[-1000:])
        sets = {}
        for (name, msgs, thinking, path, hs), e in zip(convs, outs):
            kw = {} if thinking is None else {"enable_thinking": thinking}
            hm = hf_messages(msgs)
            text = tok.apply_chat_template(hm, tokenize=False, add_generation_prompt=True, **kw)
            ids = list(tok.apply_chat_template(hm, tokenize=True, add_generation_prompt=True, **kw))
            s = sets.setdefault(name, {"n": 0, "text_same": 0, "ids_same": 0, "errors": 0, "first_differences": []})
            s["n"] += 1
            if "error" in e:
                s["errors"] += 1
                if len(s["first_differences"]) < 3:
                    s["first_differences"].append({"messages": msgs, "thinking": thinking, "ember_error": e["error"]})
                continue
            s["text_same"] += e["prompt"] == text
            s["ids_same"] += e["ids"] == ids
            if (e["prompt"] != text or e["ids"] != ids) and len(s["first_differences"]) < 3:
                k = next((j for j, (x, y) in enumerate(zip(e["ids"], ids)) if x != y), min(len(e["ids"]), len(ids)))
                s["first_differences"].append({"messages": msgs, "thinking": thinking, "ember_text": e["prompt"],
                                               "hf_text": text, "first_id_difference": k,
                                               "ember_ids": e["ids"][max(0, k - 3):k + 5],
                                               "hf_ids": ids[max(0, k - 3):k + 5]})
        total = {k: sum(s[k] for name, s in sets.items() if not name.endswith("not_gated"))
                 for k in ("n", "text_same", "ids_same", "errors")}
        report["models"][model] = {"gguf": gguf, "gguf_sha256": sha256(gguf), "ember": head,
                                   "template_expected": want, "template_resolved_ok": head.get("template") == want,
                                   "sets": sets, "total": total,
                                   "pass": head.get("template") == want and total["n"] == total["ids_same"] ==
                                   total["text_same"]}
        print(model, json.dumps(head), json.dumps(total), flush=True)
    report["pass"] = all(m["pass"] for m in report["models"].values())
    json.dump(report, open(os.path.join(HERE, "g5_thirdparty.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print("pass", report["pass"])


if __name__ == "__main__":
    main()
