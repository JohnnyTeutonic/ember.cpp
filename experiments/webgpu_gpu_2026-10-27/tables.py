"""The README's tables from the receipts in this folder (gates/, pages/, bench/, probe/).
    python tables.py [DIR]
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def res(name, sub):
    r = load(os.path.join(D, sub, name + ".json"))
    return (r or {}).get("result") or {}, r or {}


print("## gates\n")
print("| Family | Path | Greedy | Top-1 | Worst |dlogit| / range | Median | Dispatches/token | ms per forward | pass |")
print("|---|---|---:|---:|---:|---:|---:|---:|---|")
for f in sorted(x[:-5] for x in os.listdir(os.path.join(D, "gates")) if x.endswith(".json")) if os.path.isdir(os.path.join(D, "gates")) else []:
    s, r = res(f, "gates")
    if not s:
        print(f"| {f} | | | | | | | | {r.get('error', 'no result')[:120]} |")
        continue
    print(f"| {f} | {s.get('family_hint')} | {s.get('greedy_identical')} | {s.get('top1_agree')} | {s.get('worst_rel', 0):.1e} | "
          f"{s.get('median_rel') or 0:.1e} | {s.get('dispatches_per_token')} | {s.get('ms_per_forward_median') or 0:.2f} | "
          f"{s.get('pass')} |")

print("\n## pages\n")
print("| Model | WebGPU = WASM | WebGPU = CPU | WASM = CPU | Reply tokens | ms/reply WebGPU | ms/reply WASM | errors | pass |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
pd = os.path.join(D, "pages")
for f in sorted(x[:-5] for x in os.listdir(pd) if x.endswith(".json")) if os.path.isdir(pd) else []:
    s, r = res(f, "pages")
    n = s.get("n")
    ms = s.get("ms_per_reply") or {}
    print(f"| {f} | {s.get('webgpu_vs_wasm')}/{n} | {s.get('webgpu_vs_ember_cpu')}/{n} | {s.get('wasm_vs_ember_cpu')}/{n} | "
          f"{s.get('tokens')} | {ms.get('webgpu', 0):.0f} | {ms.get('wasm', 0):.0f} | {s.get('errors')} | "
          f"{s.get('pass')} {r.get('error') or s.get('error') or ''} |")

print("\n## bench (medians over runs)\n")
print("| Model | GPU path | Build | n x runs | replies/s WebGPU | replies/s WASM | ms/reply WebGPU | ms/reply WASM | "
      "prefill ms/token WebGPU | prefill ms/token WASM | decode ms/token WebGPU | decode ms/token WASM | same replies |")
print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
bd = os.path.join(D, "bench")
for f in sorted(x[:-5] for x in os.listdir(bd) if x.endswith(".json") and "ember_ir_doc" not in x) if os.path.isdir(bd) else []:
    s, r = res(f, "bench")
    if not s or s.get("error"):
        print(f"| {f} | {(r.get('error') or s.get('error') or '')[:150]} |")
        continue
    g, w = (s.get("median") or {}).get("webgpu") or {}, (s.get("median") or {}).get("wasm") or {}
    v = lambda d, k, fmt: (fmt % d[k]) if d.get(k) is not None else ""  # noqa: E731
    print(f"| {f} | {s.get('gpu_path') or '-'} | {s.get('ember_js')} | {s.get('n')} x {s.get('runs')} | "
          f"{v(g, 'replies_per_s', '%.2f')} | {v(w, 'replies_per_s', '%.2f')} | {v(g, 'ms_per_reply', '%.0f')} | "
          f"{v(w, 'ms_per_reply', '%.0f')} | {v(g, 'prefill_ms_per_token', '%.2f')} | {v(w, 'prefill_ms_per_token', '%.2f')} | "
          f"{v(g, 'decode_ms_per_token', '%.2f')} | {v(w, 'decode_ms_per_token', '%.2f')} | {s.get('webgpu_vs_wasm_same')} |")
