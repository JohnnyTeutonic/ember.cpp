import json
import sys

import numpy as np

a, b, name = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
eq = all(np.array_equal(np.float32(x[k]), np.float32(y[k])) for x, y in zip(a["seqs"], b) for k in ("prefill", "decode")) \
    and all(x["free"] == y["free"] for x, y in zip(a["seqs"], b))
print(json.dumps({"relabelled": name, "served_path": a["path"], "device": a["device"],
                  "logits_and_greedy_bit_equal_to_original_file": eq}))
