#!/usr/bin/env python3
"""Toy paperkiln/2 GGUFs for ember's v2 server tests (run in WSL; the gguf package from llama.cpp).

    python3 make_toy_v2.py BASE.gguf RECORD.json OUT_DIR [PAPERKILN_V2_VOCAB.gguf]

BASE is a llama GGUF with paperkiln's 16k Llama 3-scheme vocabulary (gate_llama3.gguf: 16,384 tokens,
tokenizer.ggml.model "gpt2", pre "llama-bpe"). RECORD.json is {"paperkiln.chat_template": ...,
"tokenizer.chat_template": ...}, the strings paperkiln's export writes for paperkiln/2
(`chat_template_check record paperkiln/2`). Writes, refusing to overwrite:
  toy_v2.gguf           the eight control tokens (<|system|>, <|tool_call|>, <|tool|>, <|spare_3|> ..
                        <|spare_7|>) appended as ids 16384-16391 with token type 3, scores 0; every row
                        of the tensors whose first dimension is the vocabulary (token_embd, output) gets
                        8 rows, each the mean of the old rows (paperkiln's chat_v2_vocab.py extend);
                        llama.vocab_size 16392; the two template keys; paperkiln.ir and
                        paperkiln.ir_version left out (the IR document names 16,384 rows), so ember runs
                        the file on its llama path. Every other key and tensor is copied unchanged.
  toy_v2_nocontrol.gguf the template keys on the base vocabulary (no control tokens): ember refuses it.
  toy_v2_badtype.gguf   the control tokens appended with token type 1 (normal): ember refuses it.
With PAPERKILN_V2_VOCAB (paperkiln's tools/chat_v2_vocab.py vocab output) the appended tokens and their
types are checked against paperkiln's, and the first 16,384 tokens against the base's. Prints the
sha256 of every output."""
import hashlib
import json
import os
import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

CONTROL = ["<|system|>", "<|tool_call|>", "<|tool|>", "<|spare_3|>", "<|spare_4|>", "<|spare_5|>",
           "<|spare_6|>", "<|spare_7|>"]
SKIP = {"paperkiln.ir", "paperkiln.ir_version", "general.architecture"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def field_value(f):
    t = GGUFValueType(f.types[0])
    if t == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]), t, None
    if t == GGUFValueType.ARRAY:
        st = GGUFValueType(f.types[1])
        if st == GGUFValueType.STRING:
            return [bytes(f.parts[i]) for i in f.data], t, st
        return [f.parts[i][0].item() for i in f.data], t, st
    return f.parts[f.data[0]][0].item(), t, None


def strings(reader, key):
    f = reader.fields[key]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def ints(reader, key):
    f = reader.fields[key]
    return [int(f.parts[i][0]) for i in f.data]


def write(base, out, record, n_extra, extra_type):
    if os.path.exists(out):
        sys.exit(f"{out} exists; refusing to overwrite")
    r = GGUFReader(base)
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    n_vocab = len(r.fields["tokenizer.ggml.tokens"].data)
    w = GGUFWriter(out, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name in SKIP or name in record:
            continue
        val, t, st = field_value(f)
        if n_extra and name == "tokenizer.ggml.tokens":
            val = val + [c.encode() for c in CONTROL[:n_extra]]
        elif n_extra and name == "tokenizer.ggml.token_type":
            val = val + [extra_type] * n_extra
        elif n_extra and name == "tokenizer.ggml.scores":
            val = val + [0.0] * n_extra
        elif n_extra and name == f"{arch}.vocab_size":
            val = val + n_extra
        w.add_key_value(name, val, t, st)
    for k, v in record.items():
        w.add_string(k, v)
    for t in r.tensors:
        a = np.array(t.data)
        if t.tensor_type.name != "F32":
            sys.exit(f"{t.name}: {t.tensor_type.name} (this tool copies F32 tensors only)")
        if n_extra and a.shape[0] == n_vocab:
            a = np.concatenate([a, np.repeat(a.mean(axis=0, keepdims=True), n_extra, axis=0)]).astype(np.float32)
        w.add_tensor(t.name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(out, sha256(out))


def main():
    base, record_path, out_dir = sys.argv[1:4]
    record = json.load(open(record_path, encoding="utf-8"))
    assert json.loads(record["paperkiln.chat_template"])["version"] == 2
    os.makedirs(out_dir, exist_ok=True)
    print(base, sha256(base))
    write(base, os.path.join(out_dir, "toy_v2.gguf"), record, 8, 3)
    write(base, os.path.join(out_dir, "toy_v2_nocontrol.gguf"), record, 0, 3)
    write(base, os.path.join(out_dir, "toy_v2_badtype.gguf"), record, 8, 1)
    if len(sys.argv) > 4:
        pk = GGUFReader(sys.argv[4])
        toy = GGUFReader(os.path.join(out_dir, "toy_v2.gguf"))
        b = GGUFReader(base)
        n = len(b.fields["tokenizer.ggml.tokens"].data)
        same_new = (strings(pk, "tokenizer.ggml.tokens")[n:] == strings(toy, "tokenizer.ggml.tokens")[n:]
                    and ints(pk, "tokenizer.ggml.token_type")[n:] == ints(toy, "tokenizer.ggml.token_type")[n:])
        same_old = strings(pk, "tokenizer.ggml.tokens")[:n] == strings(b, "tokenizer.ggml.tokens")
        print(json.dumps({"paperkiln_vocab": sys.argv[4], "sha256": sha256(sys.argv[4]),
                          "appended_tokens_and_types_equal": same_new, "base_tokens_equal": same_old,
                          "paperkiln_tokens": len(pk.fields["tokenizer.ggml.tokens"].data),
                          "toy_tokens": len(toy.fields["tokenizer.ggml.tokens"].data)}))


if __name__ == "__main__":
    main()
