"""VM-only debug patch: TINYLLAMA_LOGITS_DUMP writes 9 significant digits (float32 round trip) instead of 6."""
import sys
p = sys.argv[1]
s = open(p).read()
old = "std::ofstream f(dump_path, std::ios::app);"
assert s.count(old) == 1, "dump line not found"
open(p, "w").write(s.replace(old, old + " f.precision(9);"))
