"""Print a GGUF's IR model.ctx (paperkiln.ir), or its <arch>.context_length, without numpy."""
import json
import struct
import sys

f = open(sys.argv[1], "rb")
magic, ver, nt, nkv = struct.unpack("<IIQQ", f.read(24))


def rd(t):
    if t == 8:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        return [rd(et) for _ in range(n)]
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


meta = {}
for _ in range(nkv):
    k = rd(8)
    t = struct.unpack("<I", f.read(4))[0]
    meta[k] = rd(t)
if "paperkiln.ir" in meta:
    print(json.loads(meta["paperkiln.ir"])["model"]["ctx"])
else:
    a = meta.get("general.architecture", "")
    print(meta.get(a + ".context_length", 2048))
