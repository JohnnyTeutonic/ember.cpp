#!/usr/bin/env python3
"""Copy a GGUF with string keys added to its metadata, everything else byte for byte.

    add_template_keys.py SRC.gguf DST.gguf KEYS.json

KEYS.json is {key: string value} (chat_template_check record's output: paperkiln.chat_template and
tokenizer.chat_template). The copy is the header with the key count raised, the original key/value
bytes, the new keys after them, the original tensor-info bytes, padding to the file's alignment and
the original tensor data bytes unchanged (tensor offsets are relative to the data section, so they
hold). A key already present is refused. Prints the sha256 of the source's and the copy's tensor
data sections, which must be equal.
"""
import hashlib
import json
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def parse(buf):
    p = 0

    def u32():
        nonlocal p
        v = struct.unpack_from("<I", buf, p)[0]
        p += 4
        return v

    def u64():
        nonlocal p
        v = struct.unpack_from("<Q", buf, p)[0]
        p += 8
        return v

    def string():
        nonlocal p
        n = u64()
        s = bytes(buf[p:p + n])
        p += n
        return s

    def skip(t):
        nonlocal p
        if t == 8:
            string()
        elif t == 9:
            et, n = u32(), u64()
            for _ in range(n):
                skip(et)
        else:
            p += SIZES[t]

    assert bytes(buf[:4]) == b"GGUF", "not GGUF"
    p = 4
    version = u32()
    n_tensors, n_kv = u64(), u64()
    kv_start = p
    keys, alignment = [], 32
    for _ in range(n_kv):
        k = string().decode("utf-8")
        t = u32()
        if k == "general.alignment" and t == 4:
            alignment = struct.unpack_from("<I", buf, p)[0]
        keys.append(k)
        skip(t)
    kv_end = p
    for _ in range(n_tensors):
        string()
        nd = u32()
        p += 8 * nd + 4 + 8
    ti_end = p
    data_start = (ti_end + alignment - 1) // alignment * alignment
    return version, n_tensors, n_kv, kv_start, kv_end, ti_end, data_start, alignment, keys


def kv_string(key, value):
    k, v = key.encode("utf-8"), value.encode("utf-8")
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def main(src, dst, keys_path):
    buf = memoryview(open(src, "rb").read())
    version, n_tensors, n_kv, kv_start, kv_end, ti_end, data_start, alignment, keys = parse(buf)
    new = json.load(open(keys_path, encoding="utf-8"))
    for k in new:
        if k in keys:
            sys.exit(f"{src} already has {k}")
    head = b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", n_tensors, n_kv + len(new))
    body = bytes(buf[kv_start:kv_end]) + b"".join(kv_string(k, v) for k, v in new.items()) + bytes(buf[kv_end:ti_end])
    out = head + body
    out += b"\0" * ((-len(out)) % alignment)
    data = bytes(buf[data_start:])
    with open(dst, "wb") as f:
        f.write(out)
        f.write(data)
    chk = parse(memoryview(open(dst, "rb").read()))
    copy_data = open(dst, "rb").read()[chk[6]:]
    a, b = hashlib.sha256(data).hexdigest(), hashlib.sha256(copy_data).hexdigest()
    print(json.dumps({"src": src, "dst": dst, "added": list(new), "tensor_data_sha256_src": a,
                      "tensor_data_sha256_dst": b, "tensor_data_equal": a == b,
                      "kv_count": [n_kv, chk[2]], "alignment": alignment}))
    if a != b:
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
