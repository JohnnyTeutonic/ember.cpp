"""Multi-turn conversations for the KV prefix-reuse check.

Runs C conversations of T turns each against a server, greedy (top_k 1),
feeding each reply back as history (as the chat page does), and saves the
replies, server TTFT and prefix_reused_tokens per turn. Run once against a
server with prefix reuse (default) and once with EMBER_PREFIX_REUSE=0, then
compare with --compare A.json B.json.
Usage: chat_prefix_check.py PORT OUT.json [C] [T]
       chat_prefix_check.py --compare REUSE.json NOREUSE.json
"""
import json
import statistics
import sys
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"


def run(port, out, C, T):
    rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]
    convs = []
    for c in range(C):
        hist, turns = [], []
        for t in range(T):
            msg = rows[(c * T + t) % len(rows)]["prompt"]
            body = {"user_input": msg, "history": hist, "max_new_tokens": 48, "temperature": 1.0,
                    "top_k": 1, "top_p": 1.0, "seed": 0}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=300))
            turns.append({"reply": d["reply"], "ttft_ms": d["ttft"]["ttft_ms"], "prompt_tokens": d["prompt_tokens"],
                          "reused": d["ttft"].get("prefix_reused_tokens", 0)})
            hist += [{"role": "user", "content": msg}, {"role": "assistant", "content": d["reply"]}]
        convs.append(turns)
    json.dump(convs, open(out, "w"), indent=1)


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    T = len(a[0])
    same = [0] * T
    first_diff = []
    for ca, cb in zip(a, b):
        diverged = False
        for t in range(T):
            ok = ca[t]["reply"] == cb[t]["reply"]
            same[t] += ok
            if not ok and not diverged:
                first_diff.append(t)
                diverged = True
    print("replies identical per turn (reuse vs no reuse):", same, f"of {len(a)} conversations")
    for t in range(T):
        ra = statistics.median(c[t]["ttft_ms"] for c in a)
        rb = statistics.median(c[t]["ttft_ms"] for c in b)
        reused = statistics.median(c[t]["reused"] for c in a)
        prompt = statistics.median(c[t]["prompt_tokens"] for c in a)
        print(f"turn {t}: prompt {prompt:.0f} tok, reused {reused:.0f} | TTFT p50 reuse {ra:.1f} ms vs no reuse {rb:.1f} ms")
    if first_diff:
        print("first divergent turn per diverging conversation:", first_diff)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8,
            int(sys.argv[4]) if len(sys.argv) > 4 else 5)
