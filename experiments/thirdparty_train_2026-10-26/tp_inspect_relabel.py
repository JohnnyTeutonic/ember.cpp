"""The relabelled copies' llama keys (what the structural match reads), for the receipt."""
import glob
import sys

import gguf

for p in sorted(glob.glob(sys.argv[1] + "/*_as_llama.gguf")):
    f = gguf.GGUFReader(p).fields
    keys = {k: f[k].parts[-1].tolist()[0] for k in f if k.startswith("llama.")}
    print(p.split("/")[-1], bytes(f["general.architecture"].parts[-1]).decode(), keys)
