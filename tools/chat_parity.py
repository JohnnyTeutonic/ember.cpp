"""Greedy parity: ember tinyllama_server (BPE chat mode) vs mtstudio chat.

Both servers get the same /chat requests with top_k=1 (argmax in both
samplers). Compares reply text, generated token count and stop reason, and
checks ember's prompt token count against tools/bpe.py on the same template.
"""
import json
import os
import sys
import urllib.request

# paperkiln's tools/bpe.py: PAPERKILN_TOOLS if set, else the sibling checkout
# under either folder name (paperkiln, or microtorch before the rename).
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_tools = os.environ.get("PAPERKILN_TOOLS") or next(
    (p for p in (os.path.join(_root, n, "tools") for n in ("paperkiln", "microtorch"))
     if os.path.exists(os.path.join(p, "bpe.py"))), None)
if _tools is None:
    sys.exit("chat_parity.py: paperkiln's tools/bpe.py not found; set PAPERKILN_TOOLS")
sys.path.insert(0, _tools)
import bpe  # noqa: E402

# Ports and model from the environment (PARITY_EMBER_PORT, PARITY_MTS_PORT,
# PARITY_GGUF); the defaults are the 18 Oct 2026 receipts' setup.
EMBER = f"http://127.0.0.1:{os.environ.get('PARITY_EMBER_PORT', '8090')}/chat"
MTS = f"http://127.0.0.1:{os.environ.get('PARITY_MTS_PORT', '8091')}/chat"
GGUF = os.environ.get("PARITY_GGUF", "/home/jonat/ember_ttft/chat/chat_sft_bpe.gguf")
PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else None


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


tok = bpe.read_gguf(GGUF)
rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")][:N]
same_text = same_tok = same_stop = tok_ok = 0
records = []
for i, r in enumerate(rows):
    body = {"user_input": r["prompt"], "max_new_tokens": 60, "temperature": 1.0,
            "top_k": 1, "top_p": 1.0, "seed": 0}
    e = post(EMBER, body)
    m = post(MTS, body)
    py_n = len(tok.encode("<|user|>" + r["prompt"] + "<|assistant|>"))
    rec = {"i": i, "prompt": r["prompt"], "ember": e["reply"], "mtstudio": m["reply"],
           "ember_tokens": e["tokens"], "mts_tokens": m["tokens"],
           "ember_stop": e["stop_reason"], "mts_stop": m["stop_reason"],
           "ember_prompt_tokens": e["prompt_tokens"], "bpe_py_prompt_tokens": py_n}
    records.append(rec)
    same_text += e["reply"] == m["reply"]
    same_tok += e["tokens"] == m["tokens"]
    same_stop += e["stop_reason"] == m["stop_reason"]
    tok_ok += e["prompt_tokens"] == py_n
    if e["reply"] != m["reply"]:
        print(f"[{i}] DIFF\n  ember: {e['reply']!r}\n  mts:   {m['reply']!r}")
print(f"n={len(rows)} reply identical {same_text}, token count {same_tok}, "
      f"stop reason {same_stop}, prompt tokens == bpe.py {tok_ok}")
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
