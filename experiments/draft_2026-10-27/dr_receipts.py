"""Lean receipts of draft_2026-10-27 from the full outputs (which stay in C:/ml_artifacts/ember_draft/).

    python3 dr_receipts.py VM_OUT WORK RECEIPTS

Per end-to-end run (VM_OUT/e2e, VM_OUT/bits; WORK/cpu_e2e): the meta file and a records file without the reply text
and the prompt ids; plain decode's records keep their ids, the other runs keep each request's ids as a sha256 (the
compare files show them equal to plain decode's), gzipped. Also: the compare and bits files, the VM log, hardware and
sha256 files; the CPU records the simulator replays (WORK/collect, gzipped, prompt ids and ids kept); the CPU
regression's receipts' comparison lines; the tables' sha256 and build parameters."""
import glob
import gzip
import hashlib
import io
import json
import os
import shutil
import sys


def slim(src, dst, keep_ids):
    with gzip.GzipFile(dst + ".gz", "wb", mtime=0) as g, io.TextIOWrapper(g, encoding="utf-8", newline="\n") as f:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            r.pop("reply", None)
            r.pop("prompt_ids", None)
            if not keep_ids:
                r["ids_sha256"] = hashlib.sha256(json.dumps([r.pop("ids"), r.get("stop_id")]).encode()).hexdigest()
            f.write(json.dumps(r) + "\n")


def runs(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src_dir, "*.jsonl"))):
        name = os.path.basename(p)
        tag = name[:-6]
        plain = tag.endswith("_off") or tag.endswith("_off2") or tag.endswith("_det_plain")
        slim(p, os.path.join(dst_dir, name), plain)
        for extra in (p + ".meta.json",):
            if os.path.exists(extra):
                shutil.copy(extra, dst_dir)
    for p in glob.glob(os.path.join(src_dir, "*_vs_*.json")):
        shutil.copy(p, dst_dir)


def main(vm, work, rec):
    c = os.path.join(rec, "cuda")
    runs(os.path.join(vm, "e2e"), os.path.join(c, "e2e"))
    runs(os.path.join(vm, "bits"), os.path.join(c, "bits"))
    for f in ("hardware.txt", "gpu.txt", "bins.sha256", "big.sha256", "rows.sha256"):
        if os.path.exists(os.path.join(vm, f)):
            shutil.copy(os.path.join(vm, f), c)
    vml = os.path.join(os.path.dirname(vm), "vm.log")
    if os.path.exists(vml):
        shutil.copy(vml, os.path.join(c, "vm.log"))
    cpu = os.path.join(rec, "cpu")
    runs(os.path.join(work, "cpu_e2e"), os.path.join(cpu, "e2e"))
    col = os.path.join(cpu, "collect")
    os.makedirs(col, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(work, "collect", "*.jsonl"))):
        with open(p, "rb") as src, gzip.GzipFile(os.path.join(col, os.path.basename(p) + ".gz"), "wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
        if os.path.exists(p + ".meta.json"):
            shutil.copy(p + ".meta.json", col)
    tabs = {}
    for p in sorted(glob.glob(os.path.join(work, "tables", "*.bin"))):
        tabs[os.path.basename(p)] = {"bytes": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    for p in sorted(glob.glob(os.path.join(work, "corpus", "*.u16"))):
        tabs[os.path.basename(p)] = {"tokens": os.path.getsize(p) // 2, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    json.dump(tabs, open(os.path.join(rec, "tables.json"), "w"), indent=1)
    print(json.dumps({"receipts": rec, "tables": list(tabs)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
