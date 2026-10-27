"""The README's tables from this folder's receipts (markdown on stdout): the model gates (<name>/cpu, <name>/numpy),
the op gate (op_gate.json), the state checks (state_check.json), the chats (chat/, regress/chat/), the regression
gates (regress/<family>/) and the HEAD comparison (regress/same_as_head.json).

    python tables.py
"""
import glob
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
e = lambda x: f"{x:.1e}" if isinstance(x, (int, float)) else str(x)  # noqa: E731

MODELS = ["pk_recurrence", "pk_kimi_rec", "pk_srd_rec", "pk_mamba_rec", "tp_mamba1_rec", "kimi_tiny_rec", "srd_tiny_rec",
          "mamba_tiny_rec", "kimi_tiny_custom", "mamba-130m_rec"]
print("| Document | Reference | Prompts x ids + greedy | Greedy | Top-1 | Worst |dlogit| / range | Median | Dispatches per token | s per prompt |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|")
for m in MODELS:
    for kind in ("cpu", "numpy"):
        p = os.path.join(H, m, kind, "summary.json")
        if not os.path.exists(p):
            continue
        s = J(p)
        print(f"| {m} | {'ember CPU' if kind == 'cpu' else 'numpy'} | {s['prompts']} x {s['len']} + {s['gen']} | {s['greedy_identical']} | "
              f"{s['top1_agree']} | {e(s['worst_rel'])} | {e(s['median_rel'])} | {s['dispatches_per_token']} | {s['ms_per_prompt'] / 1000:.1f} |")
print()
p = os.path.join(H, "op_gate.json")
if os.path.exists(p):
    print("| Op case | Class | vs numpy | bit-equal | vs ember CPU | bit-equal |")
    print("|---|---|---:|---:|---:|---:|")
    for r in J(p)["rows"]:
        a, b = r["vs_numpy"], r.get("vs_ember_cpu", {})
        print(f"| {r['name']} | {r['class']} | {e(a['rel'])} | {a['same']}/{a['n']} | {e(b.get('rel'))} | {b.get('same')}/{b.get('n')} |")
    print()
p = os.path.join(H, "state_check.json")
if os.path.exists(p):
    print("| Document | tokenwise | chunks | reset | turns | refusal | trace | trace logits | trace state |")
    print("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in J(p)["results"]:
        s, t = r.get("seq") or {}, r.get("trace") or {}
        f = lambda k: f"{s[k]}/{s['n']}" if s else ""  # noqa: E731
        print(f"| {r['name']} | {f('whole_tokenwise')} | {f('chunks')} | {f('reset')} | {f('turns')} | {f('refusal')} | "
              f"{t.get('file', '')} | {e(t['worst_rel']) if t else ''} | {e(t['worst_state_rel']) if t else ''} |")
    print()
for where in ("chat", os.path.join("regress", "chat")):
    rows = sorted(glob.glob(os.path.join(H, where, "*.json")))
    if not rows:
        continue
    print(f"| Model ({where}) | GPU path | WebGPU = WASM | WebGPU = CPU fixture | WASM = CPU fixture | Reply tokens | s per reply |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for p in rows:
        s = J(p)
        n = s["n"]
        fx = lambda k: f"{s[k]}/{n}" if s.get(k) is not None else "no fixture"  # noqa: E731
        print(f"| {os.path.basename(p)[:-5]} | {s['backend']} | {s['webgpu_vs_wasm']}/{n} | {fx('webgpu_vs_ember_cpu')} | "
              f"{fx('wasm_vs_ember_cpu')} | {s['tokens']} | {s['ms_per_reply']['webgpu_swiftshader'] / 1000:.1f} |")
    print()
rows = sorted(glob.glob(os.path.join(H, "regress", "*", "summary.json")))
if rows:
    print("| Family (regression) | Prompts | Greedy | Top-1 | Worst | Median | Dispatches per token | s per prompt |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in rows:
        s = J(p)
        print(f"| {os.path.basename(os.path.dirname(p))} | {s['prompts']} | {s['greedy_identical']} | {s['top1_agree']} | {e(s['worst_rel'])} | "
              f"{e(s['median_rel'])} | {s['dispatches_per_token']} | {s['ms_per_prompt'] / 1000:.1f} |")
