"""Quantise the 2-D weight tensors of an f32 GGUF to Q8_0.

Every metadata field and every other tensor is copied unchanged. By default
the attention and MLP weights are quantised and token_embd / output / norms
stay f32; --output-q8 also quantises output.weight (the LM head), as
llama.cpp does, and --embed-q8 the token embedding table. Uses llama.cpp's gguf-py (pip install gguf) for reading,
writing and the Q8_0 block format. A weight row width that is not a multiple
of 32 is an error (Q8_0 blocks are 32 wide).
--dequantize goes the other way: every quantised tensor is written back as F32
with gguf-py's reference dequantiser, which gives the f32 reference a quantised
file should reproduce in ember (ember dequantises the legacy formats at load).
Usage: quantize_gguf.py IN.gguf OUT.gguf [--output-q8] [--embed-q8]
       quantize_gguf.py IN.gguf OUT.gguf --dequantize
"""
import argparse

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q, GGUFValueType as V


def field_value(f):
    # Strings stay raw bytes: byte-level BPE token strings are not UTF-8.
    t = f.types[0]
    if t == V.STRING:
        return bytes(f.parts[f.data[0]])
    if t == V.ARRAY:
        sub = f.types[1]
        if sub == V.STRING:
            return [bytes(f.parts[i]) for i in f.data]
        return [f.parts[i][0].item() for i in f.data]
    return f.parts[f.data[0]][0].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--output-q8", action="store_true", help="also quantise output.weight")
    ap.add_argument("--embed-q8", action="store_true", help="also quantise token_embd.weight")
    ap.add_argument("--dequantize", action="store_true", help="write every quantised tensor as F32 instead")
    a = ap.parse_args()
    r = gguf.GGUFReader(a.src)
    arch = field_value(r.fields["general.architecture"]).decode("utf-8")
    w = gguf.GGUFWriter(a.dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types[0]
        val = field_value(f)
        if t == V.ARRAY:
            w.add_key_value(name, val, V.ARRAY, sub_type=f.types[1])
        else:
            w.add_key_value(name, val, t)
    # paperkiln exports before the dims fix recorded 2-D shapes as [out, in]
    # where GGUF/ggml lists ne0 (the fastest, `in`) first; the bytes were
    # already in ggml order. Detected from token_embd (ne0 must be the width);
    # such files are written back with ggml-order dims (bytes unchanged).
    n_embd = field_value(r.fields[f"{arch}.embedding_length"])
    embd = next(t for t in r.tensors if t.name == "token_embd.weight")
    old_dims = int(embd.shape[0]) != int(n_embd)
    if old_dims:
        print("recorded 2-D dims are [out, in]; writing ggml order")
    n_q = 0
    for t in r.tensors:
        data = np.array(t.data)
        if a.dequantize:
            if t.tensor_type not in (Q.F32, Q.F16, Q.BF16):
                data = gguf.quants.dequantize(data, t.tensor_type).astype(np.float32)
                w.add_tensor(t.name, data, raw_dtype=Q.F32)
                n_q += 1
            else:
                w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
            continue
        if old_dims and data.ndim == 2:
            data = data.reshape(tuple(int(x) for x in t.shape))  # numpy (out, in) -> ne [in, out]
        quantise = (t.tensor_type == Q.F32 and data.ndim == 2 and
                    (t.name.startswith("blk.") or (a.output_q8 and t.name == "output.weight") or
                     (a.embed_q8 and t.name == "token_embd.weight")))
        if quantise and data.shape[-1] % 32 != 0:
            raise SystemExit(f"{t.name}: row width {data.shape[-1]} is not a multiple of 32 (Q8_0 block width)")
        if quantise:
            w.add_tensor(t.name, gguf.quants.quantize(data.astype(np.float32), Q.Q8_0), raw_dtype=Q.Q8_0)
            n_q += 1
        else:
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.dst}: {n_q} tensors " + ("dequantised to F32" if a.dequantize else "quantised to Q8_0"))


if __name__ == "__main__":
    main()
