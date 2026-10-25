"""This work's patch against ember HEAD: whole-file diffs of the files only it touches, api.h's hunks
without the WebGPU agent's (the ones naming ir_doc_ / ir_document), and the new files.
    python sc_mkpatch.py EMBER_DIR OUT.patch"""
import re
import subprocess
import sys

ember, out = sys.argv[1], sys.argv[2]
OWN = ["CMakeLists.txt", "graph_ir.cpp", "graph_ir.h", "server.cpp", "model.h", "model.cpp", "docs/USAGE.md", "README.md"]
NEW = ["score.h", "score.cpp"]
SHARED = {"api.h": ("ir_doc", "ir_document")}


def git(*a, ok=(0,)):
    r = subprocess.run(["git", "-C", ember, *a], capture_output=True)
    if r.returncode not in ok:
        sys.exit(r.stderr.decode())
    return r.stdout.decode("utf-8")


parts = [git("diff", "--", *OWN)]
for f, foreign in SHARED.items():
    d = git("diff", "--", f)
    if not d:
        continue
    head, *hunks = re.split(r"(?m)^(?=@@ )", d)
    keep = [h for h in hunks if not any(w in h for w in foreign)]
    if keep:
        parts.append(head + "".join(keep))
for f in NEW:
    d = git("diff", "--no-index", "--", "/dev/null", f, ok=(0, 1))
    parts.append(d.replace("b/" + f, f if False else "b/" + f))
text = "".join(parts)
open(out, "w", encoding="utf-8", newline="\n").write(text)
print(f"{out}: {text.count(chr(10))} lines, files:", re.findall(r"(?m)^\+\+\+ b/(\S+)", text))
