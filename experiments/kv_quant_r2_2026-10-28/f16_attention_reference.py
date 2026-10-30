"""Posthoc f64 reconstruction of actual Qwen3 attention; no adoption or causal verdict."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import f16_trace_analysis as trace
from audit_f16_diagnosis import extract_checked


def reference(q, keys, values, hq, hkv, hd, scale, start):
    q=np.asarray(q,dtype=np.float64).reshape(-1,hq,hd)
    keys=np.asarray(keys,dtype=np.float64).reshape(-1,hkv,hd)
    values=np.asarray(values,dtype=np.float64).reshape(-1,hkv,hd)
    if (hq<=0 or hkv<=0 or hd<=0 or hq%hkv or keys.shape!=values.shape
            or start<0 or start+len(q)>len(keys)):
        raise ValueError('invalid causal GQA reference geometry')
    keys=np.repeat(keys,hq//hkv,axis=1)
    values=np.repeat(values,hq//hkv,axis=1)
    result=np.empty_like(q)
    for token in range(len(q)):
        length=start+token+1
        scores=np.einsum('hd,lhd->hl',q[token],keys[:length],optimize=False)*scale
        weights=np.exp(scores-scores.max(axis=1,keepdims=True))
        weights/=weights.sum(axis=1,keepdims=True)
        result[token]=np.einsum('hl,lhd->hd',weights,values[:length],optimize=False)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite f64 reference')
    return result.reshape(len(q),hq*hd)


def record(result, records, history, start, case, method):
    for layer,row in enumerate(records):
        meta,arrays=row['meta'],row['arrays']
        if method=='prefill':
            keys,values=arrays['k_cache'],arrays['v_cache']
        else:
            # Decode history is assembled ONLY from previous actual decode writes.
            state=history.setdefault(layer,{'k':[],'v':[]})
            if len(state['k'])!=start:
                raise ValueError('missing/reordered actual decode cache row')
            state['k'].append(arrays['k_cache'].copy())
            state['v'].append(arrays['v_cache'].copy())
            keys,values=np.concatenate(state['k']),np.concatenate(state['v'])
        expected=reference(arrays['q'],keys,values,meta['hq'],meta['hkv'],meta['hd'],meta['scale'],start)
        metric=trace.difference(arrays['attention'],expected)
        metric.pop('top1_equal')  # Hidden-activation argmax is not a vocabulary/token result.
        aggregate=result.setdefault(str(layer),{'rows':0,'worst_absolute':0.0,'worst_relative':0.0})
        if metric['worst_relative']>aggregate['worst_relative']:
            aggregate['worst_location']={'case':case,'method':method,'start':start}
        trace.merge_metric(aggregate,metric)


def analyze_extracted(root):
    report={'scope':'Posthoc actual-Q/K/V f64 causal GQA reconstruction; descriptive only',
            'e9_closed':False,'adoption_verdict':None,'causal_claim':False,'results':{},
            'decode_cache_assumption':'Previously recorded new-row cache writes persist; past rows are not resnapshotted each step.',
            'residual_scope':'Attention/read arithmetic or unobserved past-cache mutation; not a causal attribution.'}
    for fmt in ('f32','f16'):
        for backend in ('cpu','cuda'):
            folder=root/'diagnosis'/f'{fmt}_{backend}'
            manifest=json.loads((folder/'manifest.json').read_text())
            ir=json.loads((folder/'ir.json').read_text())
            attention=[node for node in ir['block']['nodes'] if node.get('op')=='attention']
            if (ir.get('family_hint')!='qwen3' or len(attention)!=1
                    or attention[0].get('causal') is not True
                    or any(attention[0].get(k,0) for k in ('window','sinks','softcap'))):
                raise ValueError('reference supports only standard unwindowed causal Qwen3 attention')
            layers=ir['block']['repeat']
            for case,ids in enumerate(manifest['ids']):
                part=folder/f'case_{case:04d}'
                prefill=trace.load_forward(part/'prefill',layers,'gpu' if backend=='cuda' else 'cpu',fmt,len(ids),0)
                record(report['results'].setdefault(f'{fmt}:{backend}:prefill',{}),prefill,{},0,case,'prefill')
                history={}
                for position in range(len(ids)):
                    decoded=trace.load_forward(part/f'decode_{position:04d}',layers,'gpu' if backend=='cuda' else 'cpu',fmt,1,position)
                    record(report['results'].setdefault(f'{fmt}:{backend}:decode',{}),decoded,history,position,case,'decode')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','audit','scratch','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('posthoc receipt is immutable')
    audited=json.loads(args.audit.read_text())
    source_before={str(p):trace.sha(p) for p in (Path(__file__),Path(trace.__file__))}
    before=trace.sha(args.archive)
    if audited.get('status')!='RAW_DIAGNOSIS_REPLAY_VERIFIED' or before!=audited['archive_sha256']:
        raise ValueError('verified raw archive is required before posthoc reference')
    with tempfile.TemporaryDirectory(prefix='e9-attention-reference-',dir=args.scratch) as temporary:
        root=Path(temporary)
        extracted=extract_checked(args.archive,root)
        result=analyze_extracted(root)
    if source_before != {str(Path(p)):trace.sha(Path(p)) for p in source_before}:
        raise ValueError('posthoc source changed during reference')
    if trace.sha(args.archive)!=before:
        raise ValueError('raw archive changed during posthoc replay')
    result.update(archive_sha256=before,audit_receipt_sha256=trace.sha(args.audit),source=audited['source'],
                  posthoc_source_sha256=trace.sha(Path(__file__)),frozen_helper_sha256=trace.sha(Path(trace.__file__)),
                  numpy_version=np.__version__,zip=extracted)
    args.output.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps({'posthoc_complete':True,'e9_closed':False}))