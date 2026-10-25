"""Copies of cases 2-4 (and the mamba file) relabelled as general.architecture "llama": every "paperkiln.<k>" key that
names a llama hyperparameter written as "llama.<k>", everything else (paperkiln.ir included) and every tensor
byte-for-byte. The structural match (graph_ir.h tuned_llama_match) then reads the same dimensions as the IR, so it
must refuse these documents on their ops (NEOX pairing, n_rot, q/k norms, MQA), not on a missing embedding_length.
    python3 tp_relabel.py SRC.gguf DST.gguf"""
import sys

import gguf

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, "llama")
KEEP = ("paperkiln.ir", "paperkiln.ir_sha256", "paperkiln.ir_version")
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k in ("general.architecture",):
        continue
    t = f.types[0]
    name = k
    if k.startswith("paperkiln.") and k not in KEEP:
        name = "llama." + k[len("paperkiln."):]
    if t == gguf.GGUFValueType.ARRAY:
        st = f.types[-1]
        if st == gguf.GGUFValueType.STRING:
            vals = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        else:
            vals = [f.parts[i].tolist()[0] for i in f.data]
        w.add_key_value(name, vals, t, sub_type=st)
    elif t == gguf.GGUFValueType.STRING:
        w.add_key_value(name, bytes(f.parts[-1]).decode("utf-8"), t)
    else:
        w.add_key_value(name, f.parts[-1].tolist()[0], t)
for tn in r.tensors:  # the F32 files only
    assert tn.tensor_type == gguf.GGMLQuantizationType.F32, tn.name
    w.add_tensor(tn.name, tn.data.copy())
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", dst)
