"""Fail before workloads when the frozen CUDA launch contract is incomplete."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

REQUIRED_ENV = {
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'EMBER_IR_BUILD': '1', 'EMBER_IR_CTX': '1024',
    'NVIDIA_TF32_OVERRIDE': '0', 'CUDA_VISIBLE_DEVICES': '0',
    'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
}
UNSET_ENV = ('EMBER_IR', 'EMBER_KV', 'EMBER_DETERMINISTIC', 'EMBER_DECODE_LINEAR',
             'EMBER_DECODE_OPT', 'EMBER_DECODE_PICK')
VERSIONS = {'torch':'2.4.1+cu124','numpy':'1.26.4','transformers':'4.57.0',
            'tokenizers':'0.22.1','gguf':'0.19.0','safetensors':'0.6.2',
            'huggingface-hub':'0.35.3','sentencepiece':'0.2.2','accelerate':'1.13.0'}
IMAGE_DIGEST = 'sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5'


def validate_environment(environment, versions, provenance):
    for key, expected in REQUIRED_ENV.items():
        if environment.get(key) != expected:
            raise ValueError('launch environment mismatch: ' + key)
    for key in UNSET_ENV:
        if key in environment:
            raise ValueError('unexpected inference override: ' + key)
    if versions != VERSIONS:
        raise ValueError('installed library versions differ from frozen image recipe')
    if provenance.get('image_digest') != IMAGE_DIGEST:
        raise ValueError('container image digest differs')
    return {'environment':{k:environment[k] for k in REQUIRED_ENV},
            'unset_overrides':list(UNSET_ENV),'versions':versions,'image_digest':IMAGE_DIGEST}


def validate_backend(ready):
    if (ready.get('ready') is not True or ready.get('backend') != 'cuda'
            or ready.get('threads') != 1 or ready.get('max_len') != 1024
            or ready.get('vocab') != 49152):
        raise ValueError('native loader/context/backend contract failed')
    inventory = ready.get('tensor_inventory')
    if not inventory or any(t.get('type') != 0 for t in inventory):
        raise ValueError('native tensor inventory must be entirely F32')
    return ready


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--provenance', default='/content/chat/provenance.json')
    parser.add_argument('--binary')
    parser.add_argument('--gguf')
    args = parser.parse_args()
    versions={k:importlib.metadata.version(k) for k in VERSIONS}
    result=validate_environment(os.environ, versions, json.loads(Path(args.provenance).read_text()))
    if bool(args.binary) != bool(args.gguf):
        raise ValueError('binary and GGUF must be supplied together')
    if args.binary:
        process=subprocess.run([args.binary,args.gguf],input='',capture_output=True,text=True,timeout=180)
        if process.returncode:
            raise ValueError('native readiness failed: '+process.stderr)
        rows=[json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
        if len(rows) != 1:
            raise ValueError('missing or extra native readiness inventory')
        result['native_ready']=validate_backend(rows[0])
    result['status']='LAUNCH_CONTRACT_PASS'
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
