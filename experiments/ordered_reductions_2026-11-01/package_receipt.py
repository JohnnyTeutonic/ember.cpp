"""Package small scientific receipts only. Run ONLY after canonical ZIP is HOME.

This is an integrity/packaging helper, not an independent numerical gate.
The full ZIP, model, prompts and raw arrays are required for full replay.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

SMALL_NAMES = {
    'summary.json', 'science/inputs.json', 'science/protocol.json',
    'science/baseline_binary.json', 'science/candidate_binary.json',
    'science/device.log', 'science/toolchain.log', 'science/candidate_analysis.json',
    'science/baseline_comparison.json', 'science/completed_legs.json',
    'science/kernel_timing_comparison.json', 'science/baseline_scope.json',
    'science/baseline/plan.json', 'science/candidate/plan.json',
}
SMALL_LIMIT = 1024 * 1024
KERNEL_LIMIT = 64 * 1024 * 1024
RAW_FIELDS = {'input', 'weights', 'input_q', 'input_k', 'input_v', 'first_output', 'reference'}


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def small_name(name):
    return (name in SMALL_NAMES
            or re.fullmatch(r'science/(baseline|candidate)/[^/]+\.jsonl\.meta\.json', name)
            or re.fullmatch(r'science/trap_(rms|softmax)_[012]\.log', name)
            or re.fullmatch(r'science/storage[^/]*\.(json|txt)', name))


def compact_kernel(data):
    record = json.loads(data.decode('utf-8').splitlines()[-1])
    return {**{k: v for k, v in record.items() if k != 'records'},
            'records': [{k: v for k, v in row.items() if k not in RAW_FIELDS}
                        for row in record['records']]}


def package(archive, expected_sha256, output, audit=None):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError('refusing existing publication receipt')
    before = archive.stat()
    actual_sha = digest_file(archive)
    if actual_sha != expected_sha256:
        raise ValueError('canonical full ZIP SHA mismatch; do not package partial transfer')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ordered-receipt-', dir=output.parent) as scratch:
        target = Path(scratch) / 'receipt'
        target.mkdir()
        manifest, copied, kernels = [], [], []
        with zipfile.ZipFile(archive) as z:
            names = [i.filename for i in z.infolist()]
            if len(set(names)) != len(names):
                raise ValueError('duplicate ZIP member')
            for info in z.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                    raise ValueError('unsafe ZIP member')
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('ZIP symlink member')
                if info.is_dir():
                    continue
                h, size = hashlib.sha256(), 0
                with z.open(info) as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(block)
                        size += len(block)
                if size != info.file_size:
                    raise ValueError('ZIP size mismatch')
                manifest.append({'member': info.filename, 'bytes': size,
                                 'sha256': h.hexdigest(), 'crc32': f'{info.CRC:08x}'})
                if small_name(info.filename):
                    if size > SMALL_LIMIT:
                        raise ValueError('unexpected oversized small receipt ' + info.filename)
                    dest = target / info.filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(info))
                    copied.append(info.filename)
                elif re.fullmatch(r'science/kernel_(unset|0|1)_[0-2]_[0-3]_(baseline|candidate)\.log', info.filename):
                    if size > KERNEL_LIMIT:
                        raise ValueError('unexpected oversized kernel invocation')
                    kernels.append({'raw_member': info.filename, 'raw_sha256': h.hexdigest(),
                                    'summary': compact_kernel(z.read(info))})
        after = archive.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('archive changed while packaging')
        def write(name, value):
            (target / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        write('raw_manifest.json', {'archive_sha256': actual_sha, 'members': manifest})
        write('kernel_invocations_compact.json', {'scope': 'derived metrics and timings; full semantic arrays remain in raw ZIP', 'invocations': kernels})
        if audit is not None:
            audit = Path(audit)
            # Copy the separately authored audit unchanged; do not reinterpret its verdict here.
            json.loads(audit.read_text())
            (target / 'independent_audit.json').write_bytes(audit.read_bytes())
        write('packaging.json', {'schema': 1, 'canonical_local_zip': str(archive),
              'archive_bytes': before.st_size, 'archive_sha256': actual_sha,
              'all_member_crc_and_hash_reads_completed': True,
              'copied_small_members': copied, 'kernel_invocations_compacted': len(kernels),
              'packager_sha256': digest_file(Path(__file__)),
              'independent_numerical_verdict': 'NOT_ESTABLISHED_BY_PACKAGER',
              'supplied_audit_sha256': digest_file(audit) if audit is not None else None,
              'standalone_public_full_replay': False})
        (target / 'README.md').write_text("""This compact publication receipt is cryptographically bound to the canonical local full ZIP in packaging.json. raw_manifest.json gives every archive member hash and size; every member was streamed through CRC verification. This packaging step is not a numerical replay.

Small metadata, source/binary/input identities, guards and component timing/error summaries are included. Full float rows, component inputs/outputs, source archives, models and corpora are not copied here. Derived kernel summaries bind their raw invocation hashes; they cannot independently recheck the omitted semantic arrays. The separately supplied independent audit, if present, is copied unchanged.

Full replay requires the complete local ZIP, exact registered model and eval100 prompts, the pinned checker and its bound binary/input paths. This compact public directory alone is not a standalone full-replay package. The original candidate14/baseline10 distinction, PARTS=0 scope and historical failed verdicts remain unchanged.

The storage-only amendment placed future outputs on local scratch. All sixteen timing legs share local placement; the first baseline dump/off-repeat pair has mixed storage. Full .f32 link bytes are included in the canonical ZIP, while ephemeral linked per-request .run debugging/www/last-dump subtrees are omitted. See the experiment README and archived storage amendment. Neither omitted debug files nor this compact package should be represented as complete public raw rows or complete RNG-state proof.
""")
        target.rename(output)
    return actual_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', dest='archive', type=Path, required=True)
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--audit', type=Path)
    a = ap.parse_args()
    print(package(a.archive, a.expected_sha256, a.output, a.audit))


if __name__ == '__main__':
    main()
