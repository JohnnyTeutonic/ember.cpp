"""Random Q4_K / Q5_K / Q6_K blocks and gguf-py's dequantised values, for tools/kquant_test.cpp.

Every byte of each block is random except its f16 scales (d, and dmin for Q4_K / Q5_K), which
are drawn from finite values: normal, subnormal, zero and negative ones included. The values come
from gguf-py's reference dequantiser (gguf.quants.dequantize); PYTHONPATH picks the gguf-py
(llama.cpp b6000's gguf-py for the 25 Oct 2026 gate).

Usage: kquant_blocks.py OUT.bin [N_BLOCKS] [SEED]
Record: int32 ggml type, int32 block count n, n raw blocks, n * 256 float32 values.
"""
import struct
import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q

# block bytes and the byte offsets of the f16 scales in each block (ggml-common.h)
LAYOUT = {Q.Q4_K: (144, (0, 2)), Q.Q5_K: (176, (0, 2)), Q.Q6_K: (210, (208,))}


def scales(rng, n):
    """Finite f16 scales: mostly the magnitudes real files have, plus edge cases."""
    v = rng.normal(0.0, 1e-3, n).astype(np.float16)
    kind = rng.integers(0, 16, n)
    v[kind == 0] = 0.0
    v[kind == 1] = -np.abs(v[kind == 1])
    v[kind == 2] = rng.uniform(1e-7, 6e-5, int((kind == 2).sum())).astype(np.float16)  # subnormal f16
    v[kind == 3] = rng.uniform(1.0, 1000.0, int((kind == 3).sum())).astype(np.float16)
    return v


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 20261025)
    with open(out, "wb") as f:
        for qt, (bsz, offs) in LAYOUT.items():
            blocks = rng.integers(0, 256, (n, bsz), dtype=np.uint8)
            for o in offs:
                blocks[:, o:o + 2] = scales(rng, n).view(np.uint8).reshape(n, 2)
            ref = gguf.quants.dequantize(blocks, qt).astype(np.float32).reshape(-1)
            assert ref.size == n * 256 and np.isfinite(ref).all()
            f.write(struct.pack("<ii", int(qt), n))
            f.write(blocks.tobytes())
            f.write(ref.tobytes())
            print(f"{qt.name}: {n} blocks, values in [{ref.min():.4g}, {ref.max():.4g}]")
    print("gguf-py:", gguf.__file__)


if __name__ == "__main__":
    main()
