"""Compile-only isolation check for the existing SM89 verification kernels.

Only the compiler-generated anonymous-namespace identifier is normalised. Every
instruction, address, encoding and header remains in the comparison. This does
not execute a GPU or establish full-model numeric/performance parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return re.sub(r"(?<=GLOBAL__N__)[0-9a-f]{8}(?=_17_verify_kernels_cu_)", "SOURCEID", value)


def functions(text):
    if set(re.findall(r"arch = (\w+)", text)) != {"sm_89"}:
        raise ValueError("expected only SM89 compilation")
    found = re.findall(r"Function : (\S+)\n(.*?)(?=\n\s*Function :|\nFatbin ptx code:|\Z)", text, re.S)
    result = {}
    for name, code in found:
        key = canonical(name)
        if key in result:
            raise ValueError("duplicate canonical kernel")
        result[key] = canonical(code).strip()
    if not result:
        raise ValueError("no functions in disassembly")
    return result


def compare(baseline, candidate):
    before, after = functions(baseline), functions(candidate)
    missing = sorted(before.keys() - after.keys())
    different = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    added = sorted(after.keys() - before.keys())
    if missing or different:
        raise ValueError("existing SASS changed: " + repr({"missing": missing, "different": different}))
    if len(added) != 2 or not any("partition_rows_kernel" in k for k in added) or not any("merge_parts" in k for k in added):
        raise ValueError("expected only partition and merge kernels to be added")
    return {"status": "PASS_COMPILE_ONLY", "existing_functions_equal": len(before), "added_functions": added,
            "normalization": "compiler anonymous-namespace identifier only; no instruction changes masked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline.read_text(), args.candidate.read_text())
    result["inputs"] = {"baseline": sha(args.baseline), "candidate": sha(args.candidate)}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
