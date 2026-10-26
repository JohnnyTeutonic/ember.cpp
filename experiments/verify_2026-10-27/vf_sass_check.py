"""SASS of two tinyllama_server builds, kernel by kernel, ignoring blank lines (verify_2026-10-27).

    python3 vf_sass_check.py HEAD.txt NEW.txt      (prints JSON)

../determinism_2026-10-26/sass_compare.py takes a kernel's text up to the next header or fatbin section, blank
lines included. The last kernel of a translation unit's fatbin then carries the blank lines before whatever section
follows it; with verify_kernels.cu linked after graph_ir_cuda.cu, graph_ir_cuda's last kernel (embedding_kernel)
reads as "differ" there with every instruction equal. This is the same parse (its regular expressions, copied)
with blank lines dropped."""
import json
import re
import sys

FUNC = re.compile(r"^\s*Function : (\S+)\s*$")
ARCH = re.compile(r"^\s*arch = (sm_\d+)\s*$")
BREAK = re.compile(r"^(Fatbin |\s*code for sm_)")
ANON = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_(\d+_\w+?_cu)_[0-9a-f]{8}")


def kernels(path):
    out, name, arch, buf = {}, None, None, []

    def flush():
        if name is not None:
            out.setdefault(f"{arch} {name}", []).append("".join(buf))

    for line in open(path, errors="replace"):
        line = ANON.sub(r"_GLOBAL__N__H_\1_H", line)
        m = ARCH.match(line)
        if m:
            arch = m.group(1)
        m = FUNC.match(line)
        if m:
            flush()
            name, buf = m.group(1), []
            continue
        if BREAK.match(line):
            flush()
            name, buf = None, []
            continue
        if name is not None and line.strip():
            buf.append(line.rstrip() + "\n")
    flush()
    return out


a, b = kernels(sys.argv[1]), kernels(sys.argv[2])
print(json.dumps({"head_kernels": len(a), "new_kernels": len(b),
                  "head_identical_in_new": sum(1 for n in a if b.get(n) == a[n]),
                  "differ": [n for n in a if n in b and a[n] != b[n]], "only_in_head": [n for n in a if n not in b],
                  "only_in_new": sorted(n for n in b if n not in a)}, indent=1))
