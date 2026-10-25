"""paperkiln llama safetensors from its F32 GGUF: the inverse of export_gguf_llama (paperkiln src/gguf.cpp) with
weights_in_out (projections [in][out]; norms [1, d]; embed_tokens [vocab, d]; lm_head [d, vocab] when the file has
output.weight). F32 values are copied unchanged. pick_e1_L16's directory holds only the AdamW state under the name
model.safetensors, so mtstudio score needs this.
    sc_gguf2st.py IN.gguf OUT.safetensors [CHECK.safetensors]   (CHECK: compare every tensor bit for bit)"""
import re
import sys

import numpy as np
from gguf import GGUFReader
from safetensors.numpy import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
PROJ = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
out = {}
for t in r.tensors:
    if t.tensor_type.name != "F32":
        sys.exit(f"{t.name}: {t.tensor_type.name}, not F32")
    a = np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", t.name)
    if t.name == "token_embd.weight":
        out["embed_tokens.weight"] = a
    elif t.name == "output_norm.weight":
        out["norm.weight"] = a.reshape(1, -1)
    elif t.name == "output.weight":
        out["lm_head.weight"] = np.ascontiguousarray(a.T)
    elif m and m.group(2) in PROJ:
        out[f"layers.{m.group(1)}.{PROJ[m.group(2)]}.weight"] = np.ascontiguousarray(a.T)
    elif m and m.group(2) in NORM:
        out[f"layers.{m.group(1)}.{NORM[m.group(2)]}.weight"] = a.reshape(1, -1)
    else:
        sys.exit(f"unmapped tensor {t.name}")
# A tied head: the exporter writes output.weight as a copy of token_embd (the dict has no lm_head).
if "lm_head.weight" in out and out["lm_head.weight"].T.tobytes() == out["embed_tokens.weight"].tobytes():
    del out["lm_head.weight"]
    print("output.weight == token_embd.weight: tied, no lm_head")
save_file(out, dst)
print(f"{len(out)} tensors -> {dst}")
if len(sys.argv) > 3:
    ref = load_file(sys.argv[3])
    bad = [k for k in ref if k not in out or out[k].shape != ref[k].shape or out[k].tobytes() != ref[k].tobytes()]
    extra = [k for k in out if k not in ref]
    print(f"check against {sys.argv[3]}: {len(ref)} tensors, {len(bad)} differ, {len(extra)} extra", bad[:5], extra[:5])
    sys.exit(1 if bad or extra else 0)
