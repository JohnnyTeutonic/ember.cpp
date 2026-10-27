"""A GGUF's IR document (paperkiln.ir), read or replaced: the chat gate needs a file whose own document is the
recurrence one, since the page's WebAssembly reads the document from the file (ember_load; no EMBER_IR there).

    python gguf_doc.py get IN.gguf [OUT.json]
    python gguf_doc.py put IN.gguf DOC.json OUT.gguf     every other key, tensor info and tensor byte as IN's

GGUF v2/v3: the key-value pairs, then the tensor infos (offsets relative to the data section), then the data
at the next multiple of general.alignment (default 32); `put` rewrites the one string value, re-pads and
copies the data section unchanged.
"""
import json
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.take("<Q")
        s = self.b[self.o:self.o + n]
        self.o += n
        return s.decode("utf-8", "surrogateescape")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("<I"), self.take("<Q")
            return [self.value(et) for _ in range(n)]
        return self.take(SCALAR[t])


def parse(b):
    r = Reader(b)
    if b[:4] != b"GGUF":
        sys.exit("not a GGUF")
    r.o = 4
    ver, nt, nkv = r.take("<I"), r.take("<Q"), r.take("<Q")
    kvs = []   # (key, type, start, end) byte spans of each pair
    meta = {}
    for _ in range(nkv):
        s = r.o
        k = r.string()
        t = r.take("<I")
        meta[k] = r.value(t)
        kvs.append((k, t, s, r.o))
    ti0 = r.o
    for _ in range(nt):
        r.string()
        nd = r.take("<I")
        r.o += 8 * nd + 4 + 8
    align = int(meta.get("general.alignment", 32))
    data = (r.o + align - 1) // align * align
    return ver, meta, kvs, ti0, r.o, data, align


def main():
    cmd = sys.argv[1]
    b = open(sys.argv[2], "rb").read()
    ver, meta, kvs, ti0, ti1, data, align = parse(b)
    if cmd == "get":
        doc = meta["paperkiln.ir"]
        if len(sys.argv) > 3:
            open(sys.argv[3], "w", encoding="utf-8").write(doc)
        else:
            print(doc)
        return
    doc = json.dumps(json.load(open(sys.argv[3], encoding="utf-8")), separators=(",", ":"))
    out = bytearray(b[:24])
    for k, t, s, e in kvs:
        if k == "paperkiln.ir":
            enc, key = doc.encode("utf-8"), k.encode("utf-8")
            out += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + struct.pack("<Q", len(enc)) + enc
        else:
            out += b[s:e]
    out += b[ti0:ti1]
    out += b"\0" * ((-len(out)) % align)
    out += b[data:]
    open(sys.argv[4], "wb").write(out)
    _, m2, _, _, _, d2, _ = parse(bytes(out))
    assert m2["paperkiln.ir"] == doc and bytes(out[d2:]) == b[data:]
    print(sys.argv[4], len(out), "bytes")


if __name__ == "__main__":
    main()
