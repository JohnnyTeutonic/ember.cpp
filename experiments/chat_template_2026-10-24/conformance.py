#!/usr/bin/env python3
"""Gate (a): paperkiln's conformance file (microtorch/tests/chat_template_cases.json) against ember.

1. Offline: tools/chat_template_check cases (the vendored chat_template.h against every check of paperkiln's
   tests/test_chat_template.cpp, then each edge case through ember's request paths, chat_request.h:
   /chat with the system message as "system" and as a leading history entry, and the OpenAI mapping).
   Its report is conformance_cases.json.
2. Live: every request that report built, posted to a running tinyllama_server (this change, CPU)
   serving a model with the case's template: chat_sftfinal_v3 (paperkiln/1), its copy declaring
   paperkiln/1 with system mode none (g4_declared.py writes it), Qwen2.5-0.5B-Instruct, Qwen3-0.6B and
   Gemma-3-270m-it. /chat (greedy, 1 new token, return_ids): a path the offline run refused must be a
   400; a rendered one must give prompt ids equal to the reference tokeniser's encoding of the offline
   text (paperkiln's tools/bpe.py for the 8k vocabulary, HF's tokenizer for the third-party models).
   /v1/chat/completions: refused -> 400, rendered -> usage.prompt_tokens equal to that encoding's
   length. paperkiln/legacy-word is not served on a chat path with history (the word-level server path
   wraps one message), so its case runs offline only.
Writes conformance_cases.json and conformance_live.json beside this file."""
import json
import os
import subprocess
import sys

from common import A, CHECK, HERE, PK, SFT, WORK, ember, post, sha256, stop

CASES = PK + "/tests/chat_template_cases.json"
r = subprocess.run([CHECK, "cases", CASES], capture_output=True, text=True)
offline = json.loads(r.stdout)
offline["cases_file"], offline["cases_sha256"], offline["rc"] = CASES, sha256(CASES), r.returncode
json.dump(offline, open(os.path.join(HERE, "conformance_cases.json"), "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
print("offline:", json.dumps(offline["header"]), json.dumps(offline["paths_summary"]), flush=True)
if r.returncode != 0:
    sys.exit("the vendored header fails the conformance file")

cases = {c["name"]: c for c in json.load(open(CASES, encoding="utf-8"))["cases"]}
TP = A + "/thirdparty"
SERVERS = {  # template key -> (gguf, reference tokeniser, env)
    "paperkiln/1": (SFT, "bpe", {}),
    "paperkiln/1 none": (WORK + "/chat_sftfinal_v3_sysnone.gguf", "bpe", {}),
    "chatml-qwen2.5/1": (TP + "/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf", TP + "/qwen2.5-0.5b-instruct",
                         {"EMBER_IR_CTX": "1024"}),
    "chatml-qwen3/1": (TP + "/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf", TP + "/qwen3-0.6b", {"EMBER_IR_CTX": "1024"}),
    "gemma3/1": (TP + "/gemma-3-270m-it/gemma-3-270m-it-f32.gguf", TP + "/gemma-3-270m-it", {"EMBER_IR_CTX": "1024"}),
}


def key(row):
    t = cases[row["name"]]["template"]
    return t if isinstance(t, str) else row["template"] + (" none" if t["system"]["mode"] == "none" else "")


def encoder(ref):
    if ref == "bpe":
        sys.path.insert(0, PK + "/tools")
        import bpe
        return bpe.read_gguf(SFT).encode
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ref)
    return lambda s: tok(s, add_special_tokens=False)["input_ids"]


live = {"servers": {}, "rows": []}
port = 8131
for k, (gguf, ref, env) in SERVERS.items():
    todo = [row for row in offline["paths"] if key(row) == k and row["paths"]]
    if not todo:
        continue
    enc = encoder(ref)
    p, h = ember(os.path.expanduser("~/ember_wt/build_cpu"), gguf, port, "conf_" + k.replace("/", "_").replace(" ", "_"),
                 env)
    live["servers"][k] = {"gguf": gguf, "health": h}
    try:
        for row in todo:
            for path, v in row["paths"].items():
                if not isinstance(v, dict):
                    continue
                req = dict(v["request"])
                off = v["result"]
                if path == "openai":
                    req.update({"max_tokens": 1, "temperature": 1.0, "top_k": 1, "seed": 0})
                    status, body = post(port, req, "/v1/chat/completions")
                    got = body.get("usage", {}).get("prompt_tokens") if status == 200 else None
                    want = len(enc(off["text"])) if "text" in off else None
                else:
                    req.update({"max_new_tokens": 1, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0,
                                "return_ids": True})
                    status, body = post(port, req)
                    got = body.get("prompt_ids") if status == 200 else None
                    want = enc(off["text"]) if "text" in off else None
                ok = (status == 400) if "refused" in off else (status == 200 and got == want)
                live["rows"].append({"case": row["name"], "template": k, "path": path, "offline": off,
                                     "same_as_python_twin": v["same"], "status": status,
                                     "error": body.get("error") if status != 200 else None, "live_matches_offline": ok})
    finally:
        stop(p)
    port += 1
skipped = sorted({row["name"] for row in offline["paths"] if key(row) not in SERVERS and row["paths"]})
live["offline_only"] = {n: "paperkiln/legacy-word: no chat path with history serves it" for n in skipped}
live["n"] = len(live["rows"])
live["live_matches_offline"] = sum(x["live_matches_offline"] for x in live["rows"])
live["pass"] = live["n"] == live["live_matches_offline"]
print("live:", live["n"], "requests,", live["live_matches_offline"], "match the offline path; offline only:", skipped)
for x in live["rows"]:
    if not x["live_matches_offline"]:
        print("MISMATCH", json.dumps(x, ensure_ascii=False))
json.dump(live, open(os.path.join(HERE, "conformance_live.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
