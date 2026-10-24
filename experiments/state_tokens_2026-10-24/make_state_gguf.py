"""Writes the synthetic state-token test model: paperkiln's chat_sftfinal_v3 (llama, d 384, 6 layers,
vocab 8192, ctx 256; /mnt/c/ml_artifacts/chat_base/sftfinal/) with every tensor and metadata key
copied byte for byte, plus
  - state_embd.weight [k, d], k = 4: seeded normal rows with the token embedding's standard deviation
    (untrained: the test is the engine's arithmetic and bookkeeping, not a model's behaviour);
  - paperkiln.state_tokens.k = 4 (u32);
  - paperkiln.ir: the file's IR with the extended embedding ("state_weight"), attention "mask":
    "turn_state" and kv "evict": "turn_state" (family_hint stays "llama": the engine must route state
    tokens to the interpreter by itself).
Also writes the IR variants the tests use beside it: ts.ir.json (the embedded one), ts_noevict.ir.json,
ts_causal.ir.json (state_weight with the causal mask: shows the mask is read) and plain.ir.json (the
original IR). Output directory: argv[1] (default ~/ember_wt/state_tokens)."""
import copy
import json
import os
import sys

import numpy as np
import gguf

SRC = "/mnt/c/ml_artifacts/chat_base/sftfinal/chat_sftfinal_v3.gguf"
K = 4
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ember_wt/state_tokens")
os.makedirs(out_dir, exist_ok=True)
dst = os.path.join(out_dir, "chat_state_k4.gguf")

r = gguf.GGUFReader(SRC)
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
plain = json.loads(bytes(r.fields["paperkiln.ir"].parts[-1]).decode())

ts = copy.deepcopy(plain)
ts["_comment"] = ("chat_sftfinal_v3 with per-turn state tokens (paperkiln IR_SPEC_DRAFT.md section 1), "
                  "synthetic: state_embd.weight is untrained; ember state_tokens_2026-10-24")
for n in ts["embed"]["nodes"]:
    if n["op"] == "embedding":
        n["state_weight"] = "state_embd.weight"
for n in ts["block"]["nodes"]:
    if n["op"] == "attention":
        n["mask"] = "turn_state"
ts["block"]["state"][0]["evict"] = "turn_state"
noevict = copy.deepcopy(ts)
del noevict["block"]["state"][0]["evict"]
causal = copy.deepcopy(noevict)
for n in causal["block"]["nodes"]:
    if n["op"] == "attention":
        del n["mask"]
for name, doc in {"ts": ts, "ts_noevict": noevict, "ts_causal": causal, "plain": plain}.items():
    json.dump(doc, open(os.path.join(out_dir, name + ".ir.json"), "w"), indent=1)

w = gguf.GGUFWriter(dst, arch)
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name in ("general.architecture", "general.alignment", "paperkiln.ir"):
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        sub = f.types[-1]
        if sub == gguf.GGUFValueType.STRING:
            val = [bytes(f.parts[i]) for i in f.data]  # raw bytes: paperkiln's tokens are byte strings
        else:
            val = [f.parts[i].tolist()[0] for i in f.data]
        w.add_key_value(name, val, vt, sub_type=sub)
    elif vt == gguf.GGUFValueType.STRING:
        w.add_key_value(name, bytes(f.parts[-1]), vt)
    else:
        w.add_key_value(name, f.parts[-1].tolist()[0], vt)
w.add_key_value("paperkiln.ir", json.dumps(ts), gguf.GGUFValueType.STRING)
w.add_key_value("paperkiln.state_tokens.k", K, gguf.GGUFValueType.UINT32)

emb = None
for t in r.tensors:
    a = np.asarray(t.data)
    shape = [int(x) for x in reversed(t.shape.tolist())]
    assert t.tensor_type == gguf.GGMLQuantizationType.F32
    a = a.reshape(shape).astype(np.float32)
    if t.name == "token_embd.weight":
        emb = a
    w.add_tensor(t.name, a)
rng = np.random.default_rng(20261024)
state = (rng.standard_normal((K, emb.shape[1])) * emb.std()).astype(np.float32)
w.add_tensor("state_embd.weight", state)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(json.dumps({"gguf": dst, "k": K, "d": int(emb.shape[1]), "vocab": int(emb.shape[0]),
                  "state_std": float(state.std()), "emb_std": float(emb.std())}))
