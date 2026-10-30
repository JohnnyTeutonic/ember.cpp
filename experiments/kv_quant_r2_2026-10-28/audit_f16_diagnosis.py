"""Independent offline replay of one canonical remote E9 diagnostic archive."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import f16_trace_analysis as analysis
from run_f16_diagnosis import MODEL_SHA, validate_component

IMAGE = 'sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5'


def extract_checked(archive, folder):
    seen = set()
    total = 0
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in item.filename
                    or ':' in item.filename or item.filename in seen
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('unsafe/duplicate archive entry')
            seen.add(item.filename)
            total += item.file_size
            if total > 8 << 30:
                raise ValueError('diagnostic archive exceeds frozen 8GiB audit allowance')
            destination = folder.joinpath(*name.parts)
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as reader, destination.open('wb') as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)
                # Reading every member to EOF verifies its CRC without a second big pass.
                if destination.stat().st_size != item.file_size:
                    raise ValueError('extracted member size differs')
    return {'members': len(seen), 'uncompressed_bytes': total, 'crc': 'PASS'}


def check_leg(folder, expected):
    record = json.loads((folder/'leg.json').read_text())
    if (json.loads((folder/'stdout.json').read_text()) != expected or record['result'] != expected
            or record['files'] != {k:v for k,v in analysis.inventory(folder).items() if k!='leg.json'}):
        raise ValueError('raw diagnostic leg identity differs')
    return record


def bind_matrix_inventory(root, replay, ids):
    if (len(ids) != 12 or any(len(sequence) != 24 for sequence in ids)
            or replay.get('ids') != ids or replay.get('prompts') != 12
            or replay.get('model_sha256') != MODEL_SHA):
        raise ValueError('replayed prompt/model inventory differs from frozen input')
    expected = {f'case_{index:04d}' for index in range(12)}
    for fmt in ('f32', 'f16'):
        for backend in ('cpu', 'cuda'):
            folder = root/'diagnosis'/f'{fmt}_{backend}'
            actual = {p.name for p in folder.glob('case_*') if p.is_dir()}
            if actual != expected:
                raise ValueError('trace case inventory is not exactly 12 per matrix leg')


def validate_control(control, returncode, ids):
    if (control.get('on_gpu') is not True or control.get('device_supported') is not True
            or control.get('prompts') != 12 or control.get('len') != 24 or control.get('gen') != 32
            or control.get('tol') != 1e-4 or control.get('ids_file') is not None
            or control.get('positions') not in (None, 288) or returncode not in (0, 1)
            or control.get('pass') is not (returncode == 0)):
        raise ValueError('control registered inventory/backend/status differs')
    def fraction(value, denominator):
        parts = str(value).split('/')
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError('invalid control agreement inventory')
        numerator, total = map(int, parts)
        if total != denominator or not 0 <= numerator <= total:
            raise ValueError('control agreement denominator differs')
        return numerator
    top = fraction(control.get('top1_agree'), 288)
    greedy = fraction(control.get('greedy_identical'), 12)
    rows = control.get('rows', [])
    if len(rows) != 12 or [row.get('i') for row in rows] != list(range(12)):
        raise ValueError('control must contain every original prompt row')
    for row in rows:
        if (not isinstance(row.get('top1'), int) or not 0 <= row['top1'] <= 24
                or row.get('greedy_first_diff') not in range(-1, 32)
                or any(not isinstance(row.get(k), (int,float)) or not math.isfinite(row[k]) or row[k] < 0
                       for k in ('prefill_rel','decode_rel'))
                or ('ids' in row and row['ids'] != ids[row['i']])):
            raise ValueError('invalid per-prompt control evidence')
    if (sum(row['top1'] for row in rows) != top
            or sum(row['greedy_first_diff'] == -1 for row in rows) != greedy
            or max(row['prefill_rel'] for row in rows) != control.get('worst_prefill_rel')
            or max(row['decode_rel'] for row in rows) != control.get('worst_decode_rel')):
        raise ValueError('control aggregate/per-prompt evidence differs')
    expected = top == 288 and greedy == 12 and max(control['worst_prefill_rel'],control['worst_decode_rel']) <= 1e-4
    if control['pass'] is not expected:
        raise ValueError('control pass disagrees with fixed registered gate')


def audit(archive, outer_report, readiness, scratch):
    execution = json.loads((outer_report.parent/'execution.json').read_text())
    if execution.get('complete') is not True or execution.get('exit_code') != 0:
        raise ValueError('terminal operational execution is incomplete')
    archive_sha = analysis.sha(archive)
    plan = json.loads(readiness.read_text())
    with tempfile.TemporaryDirectory(prefix='e9-independent-audit-', dir=scratch) as temporary:
        root = Path(temporary)
        zip_receipt = extract_checked(archive, root)
        if (root/'diagnosis/diagnosis_report.json').read_bytes() != outer_report.read_bytes():
            raise ValueError('inner/outer report byte mismatch')
        reported = json.loads(outer_report.read_text())
        if reported.get('status') != 'DIAGNOSTIC_COMPLETE' or reported.get('e9_closed') is not False:
            raise ValueError('incomplete or incorrectly scoped diagnostic receipt')
        for name in ('COMMITS', 'INPUT_SHA256SUMS'):
            expected = plan['files'][name]
            if analysis.sha(root/name) != expected['sha256'] or (root/name).stat().st_size != expected['bytes']:
                raise ValueError('staging scope/manifest binding differs')
        commits = json.loads((root/'COMMITS').read_text())
        if commits['source'] != plan['source'] or commits['runner'] != plan['runner']:
            raise ValueError('frozen source/runner differs')
        runtime = json.loads((root/'runtime.json').read_text())
        if (runtime.get('torch') != '2.4.1+cu124' or runtime.get('numpy') != '1.26.4'
                or runtime.get('provenance', {}).get('image_digest') != IMAGE):
            raise ValueError('runtime/image contract differs')
        model = json.loads((root/'model_verification.json').read_text())
        if model.get('sha256') != MODEL_SHA:
            raise ValueError('canonical model verification differs')
        proof = json.loads((root/'normal_object_proof.json').read_text())
        if (proof.get('object_identical') is not True or proof.get('source_restored') is not True
                or proof['original_object_sha256'] != proof['normal_object_sha256']
                or proof['diagnostic_source_sha256'] != analysis.sha(Path(__file__).parents[2]/'graph_ir.cpp')
                or proof['normal_binary_sha256'] != analysis.sha(root/'binaries/ir_device_gate')):
            raise ValueError('normal CUDA object proof failed')
        # Never import code from the archive. Require local replay helper to match the frozen snapshot.
        if analysis.sha(root/'f16_trace_analysis.py') != analysis.sha(Path(analysis.__file__)):
            raise ValueError('local analyzer differs from frozen remote helper')
        integrity = reported['final_input_integrity']
        for remote, expected in integrity.items():
            name = Path(remote).name
            if name == 'e9_qwen3.gguf':
                if expected['sha256'] != MODEL_SHA or expected['bytes'] != model['bytes']:
                    raise ValueError('final model inventory differs')
                continue
            local = root/'binaries'/name if name.startswith(('ir_device_gate', 'kv_f16_')) else root/name
            if (not local.is_file() or local.stat().st_size != expected['bytes']
                    or analysis.sha(local) != expected['sha256']):
                raise ValueError('final executable/controller/ID binding differs: '+name)
        component = root/'diagnosis/component'
        if json.loads((component/'stdout.json').read_text()) != reported['component']:
            raise ValueError('raw primitive result differs')
        component_leg = check_leg(component, reported['component'])
        validate_component(reported['component'], component_leg['returncode'])
        if analysis.sha(root/'ids_seed0.json') != plan['ids_sha256']:
            raise ValueError('original seeded prompt inventory differs')
        replay = analysis.analyze(root/'diagnosis')
        bind_matrix_inventory(root, replay, json.loads((root/'ids_seed0.json').read_text()))
        for key, value in replay.items():
            if key != 'numpy_version' and reported.get(key) != value:
                raise ValueError('independent raw replay differs: '+key)
        if set(reported['controls']) != {'f32_graph0','f32_graph1','f16_graph0','f16_graph1'}:
            raise ValueError('missing/extra uninstrumented control')
        for name, control in reported['controls'].items():
            folder = root/'diagnosis'/('control_'+name)
            record = check_leg(folder, control)
            validate_control(control, record['returncode'], replay['ids'])
        if archive_sha != analysis.sha(archive):
            raise ValueError('canonical ZIP changed during audit')
        return {'status':'RAW_DIAGNOSIS_REPLAY_VERIFIED', 'e9_closed':False,
                'archive_sha256':archive_sha,'archive_bytes':archive.stat().st_size,
                'source':plan['source'],'runner':plan['runner'],'zip':zip_receipt,
                'model_sha256':MODEL_SHA,'ids_sha256':plan['ids_sha256'],'execution':execution,
                'remote_numpy':reported['numpy_version'],'replay_numpy':replay['numpy_version'],
                'controls':reported['controls'],'storage':replay['storage'],'logits':replay['logits'],
                'limitations':['Active trace scheduling differs; compare uninstrumented controls.',
                               'Rounding correlations alone do not establish causality.']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','report','readiness','output','scratch'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('audit receipt is immutable')
    result=audit(args.archive,args.report,args.readiness,args.scratch)
    args.output.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'e9_closed':False}))