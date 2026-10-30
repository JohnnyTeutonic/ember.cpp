"""Extract the unmodified production selector into a standalone benchmark header."""

import argparse
import hashlib
import json
from pathlib import Path


def extract(source):
    start = source.index("__global__ void select_rows_kernel(")
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.root / "verify_kernels.cu").read_bytes()
    original = extract(source.decode())
    # Partition the identical two-pass reduction, preserving absolute vocab indices.
    candidate = original.replace("select_rows_kernel", "partition_rows_kernel")
    candidate = candidate.replace(
        "RowPick* __restrict__ picks)", "RowPick* __restrict__ picks, int parts)"
    )
    candidate = candidate.replace(
        "for (int i = threadIdx.x; i < vocab; i += blockDim.x)",
        "for (int i = ((long long)vocab * blockIdx.y / parts) + threadIdx.x; i < (long long)vocab * (blockIdx.y + 1) / parts; i += blockDim.x)",
    )
    candidate = candidate.replace(
        "picks[blockIdx.x] = p;", "picks[blockIdx.x * parts + blockIdx.y] = p;"
    )
    assert candidate.count("(long long)vocab * blockIdx.y") == 2
    args.output.mkdir(parents=True, exist_ok=True)
    header = "constexpr int kSelThreads = 256;\n" + original + "\n" + candidate + "\n"
    (args.output / "selector.cuh").write_text(header)
    manifest = {
        "production_source_sha256": hashlib.sha256(source).hexdigest(),
        "original_kernel_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_header_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "candidate_delta": "same partition-local reduction, absolute indices; separate exact top-two merge",
    }
    (args.output / "extraction.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
