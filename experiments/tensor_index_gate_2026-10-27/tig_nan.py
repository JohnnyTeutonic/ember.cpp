"""Copy a GGUF with the named F32 tensors overwritten by NaN (tig_gate_cpu.py's nan_copy, for the device leg).
    python3 tig_nan.py IN.gguf OUT.gguf NAME [NAME ...]"""
import shutil
import sys

import numpy as np
import gguf

src, dst, names = sys.argv[1], sys.argv[2], set(sys.argv[3:])
shutil.copyfile(src, dst)
r = gguf.GGUFReader(dst)
with open(dst, "r+b") as f:
    for t in r.tensors:
        if t.name in names:
            assert t.tensor_type.name == "F32", (t.name, t.tensor_type.name)
            f.seek(int(t.data_offset))
            f.write(np.full(int(t.n_elements), np.nan, dtype=np.float32).tobytes())
got = {t.name for t in gguf.GGUFReader(dst).tensors if t.name in names and np.isnan(np.asarray(t.data)).all()}
assert got == names, (got, names)
print(dst, sorted(got))
