"""Give an older ember tree (cbdcb915) the nine-digit TINYLLAMA_LOGITS_DUMP of 9238a86 (one line in api.cpp's
debug_dump_topk), so its dumps read back as the same float32 and compare bit for bit. Nothing else changes.
    python3 x5_dump9.py SRC_DIR"""
import sys
p = sys.argv[1] + "/api.cpp"
s = open(p).read()
anchor = "    std::ofstream f(dump_path, std::ios::app);\n"
assert s.count(anchor) == 1 and "#include <iomanip>" in s
if "f << std::setprecision(9);" not in s:
    s = s.replace(anchor, anchor + "    f << std::setprecision(9);\n")
    open(p, "w").write(s)
print("dump9 applied to", p)
