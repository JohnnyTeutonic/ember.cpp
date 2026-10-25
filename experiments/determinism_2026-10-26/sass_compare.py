"""Compare two `cuobjdump -sass` listings of a binary, kernel by kernel.

    python3 sass_compare.py HEAD.txt NEW.txt      (prints JSON)

A kernel's text is every line from its "Function : <mangled name>" header to the next header or the next fatbin
section (instructions with their encodings). Kernels are matched by name and architecture, with the anonymous-
namespace hash normalised (below); "identical" counts the head kernels whose SASS the new binary has character for
character."""
import json
import re
import sys

FUNC = re.compile(r"^\s*Function : (\S+)\s*$")
ARCH = re.compile(r"^\s*arch = (sm_\d+)\s*$")
BREAK = re.compile(r"^(Fatbin |\s*code for sm_)")
# nvcc names anonymous-namespace kernels after a hash of the translation unit and its path
# (_GLOBAL__N__<8 hex>_<len>_<file>_cu_<8 hex>): two source trees give different hashes for the same kernel.
ANON = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_(\d+_\w+?_cu)_[0-9a-f]{8}")


def norm(s):
    return ANON.sub(r"_GLOBAL__N__H_\1_H", s)


def kernels(path):
    out, arches = {}, set()
    name, arch, buf = None, None, []

    def flush():
        if name is not None:
            out.setdefault((arch, name), []).append("".join(buf))

    for line in open(path, errors="replace"):
        line = norm(line)
        m = ARCH.match(line)
        if m:
            arch = m.group(1)
            arches.add(arch)
        m = FUNC.match(line)
        if m:
            flush()
            name, buf = m.group(1), []
            continue
        if BREAK.match(line):
            flush()
            name, buf = None, []
            continue
        if name is not None:
            buf.append(line.rstrip() + "\n")
    flush()
    return out, arches


head, ah = kernels(sys.argv[1])
new, an = kernels(sys.argv[2])
same = [k for k in head if k in new and head[k] == new[k]]
differ = [k for k in head if k in new and head[k] != new[k]]
only_head = [k for k in head if k not in new]
only_new = [k for k in new if k not in head]
rep = {"head": sys.argv[1], "new": sys.argv[2], "arches": sorted(ah | an),
       "head_functions": len(head), "new_functions": len(new), "identical": len(same), "differ": len(differ),
       "only_in_head": len(only_head), "only_in_new": len(only_new),
       "differing": [f"{a} {n}" for a, n in differ], "only_in_head_names": [f"{a} {n}" for a, n in only_head],
       "only_in_new_names": [f"{a} {n}" for a, n in only_new],
       "head_instructions": sum(t.count("/*0") for v in head.values() for t in v)}
rep["every_head_kernel_identical"] = len(same) == len(head) and len(head) > 0
print(json.dumps(rep, indent=1))
