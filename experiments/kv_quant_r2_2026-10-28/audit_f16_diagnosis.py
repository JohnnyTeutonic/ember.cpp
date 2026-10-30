"""Independent offline replay of one canonical remote E9 diagnostic archive."""
import argparse
import json
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


def audit(archive, outer_report, readiness, scratch):
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
        for key, value in replay.items():
            if key != 'numpy_version' and reported.get(key) != value:
                raise ValueError('independent raw replay differs: '+key)
        if set(reported['controls']) != {'f32_graph0','f32_graph1','f16_graph0','f16_graph1'}:
            raise ValueError('missing/extra uninstrumented control')
        for name, control in reported['controls'].items():
            folder = root/'diagnosis'/('control_'+name)
            record = check_leg(folder, control)
            if record['returncode'] != (0 if control['pass'] else 1) or control.get('on_gpu') is not True:
                raise ValueError('uninstrumented control status/backend differs')
        if archive_sha != analysis.sha(archive):
            raise ValueError('canonical ZIP changed during audit')
        return {'status':'RAW_DIAGNOSIS_REPLAY_VERIFIED', 'e9_closed':False,
                'archive_sha256':archive_sha,'archive_bytes':archive.stat().st_size,
                'source':plan['source'],'runner':plan['runner'],'zip':zip_receipt,
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