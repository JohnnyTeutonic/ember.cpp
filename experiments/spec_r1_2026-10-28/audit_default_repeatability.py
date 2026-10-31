"""Read-only historical default-repeatability diagnosis; no GPU run or gate revision."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zipfile
import numpy as np

HISTORICAL='ef34236697fe82c0cf2a1822b99a8eeb5dc82b10'
CURRENT='106fb0c'
EXPECTED_ARCHIVE='617759b2cc66c4fab41c148921ae7587bd783d9dafa1841202544b3a087402f0'

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return f'0x{struct.unpack("<I",struct.pack("<f",x))[0]:08x}'
def sum_in_order(values,order):
    total=0.0
    for i in order:total=f32(total+values[i])
    return total

def order_example():
    # Realizable squared warp partials: x[0]=1, x[32]=x[64]=2^-12,
    # all other inputs zero, any n>=65 (registered model n=320).
    partials=[1.0,2**-24,2**-24,0.0,0.0,0.0,0.0,0.0]
    orders=[list(range(8)),[1,2,0,3,4,5,6,7]]
    sums=[sum_in_order(partials,order) for order in orders]
    assert bits(sums[0])=='0x3f800000' and bits(sums[1])=='0x3f800001'
    return {'positive_warp_partials':partials,'orders':orders,'sum_f32':sums,'sum_bits':[bits(x) for x in sums],
            'scope':'Offline IEEE binary32 permitted-order counterexample only; not actual captured model activations or a measured CUDA schedule/rsqrt output.'}

def body(source,needle):
    start=source.index(needle);begin=source.index('{',start);depth=1;end=begin+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def audit(root,archive,output):
    root,archive,output=map(Path,(root,archive,output))
    if output.exists():raise FileExistsError('diagnostic receipts are immutable')
    def git(revision,name):return subprocess.check_output(['git','-C',str(root),'show',revision+':'+name],text=True)
    current=subprocess.check_output(['git','-C',str(root),'rev-parse',CURRENT],text=True).strip()
    sources=[]
    for symbol in ['__global__ void rmsnorm_optimized_kernel(','void rmsnorm_vector_cuda_optimized(']:
        before=body(git(HISTORICAL,'cuda_kernels.cu'),symbol);now=body(git(current,'cuda_kernels.cu'),symbol)
        if before!=now:raise ValueError('source body changed: '+symbol)
        sources.append({'file':'cuda_kernels.cu','symbol':symbol,'historical_equals_current':True,'body_sha256':hashlib.sha256(before.encode()).hexdigest()})
    kernel=body(git(current,'cuda_kernels.cu'),'__global__ void rmsnorm_optimized_kernel(')
    if 'atomicAdd(&s_block_sum_sq, thread_partial_sum_sq);' not in kernel:raise ValueError('unordered reduction premise changed')
    archive_hash=digest(archive)
    if archive_hash!=EXPECTED_ARCHIVE:raise ValueError('historical archive hash mismatch')
    stems=['default_rows_off','default_rows_off_repeat']
    hashes=[hashlib.sha256(),hashlib.sha256()];equal=[];worst=0.;different_values=0;top1=0
    with zipfile.ZipFile(archive) as z:
        plan=json.loads(z.read('out/pick/plan.json'))
        meta=[json.loads(z.read('out/pick/'+stem+'.jsonl.meta.json')) for stem in stems]
        records=[[json.loads(line) for line in z.read('out/pick/'+stem+'.jsonl').splitlines() if line.strip()] for stem in stems]
        keys=plan['request_keys']
        for rows in records:
            if [[r['i'],r['turn']] for r in rows]!=keys:raise ValueError('request inventory mismatch')
        fields=('i','turn','ids','prompt_ids','stop_id','tokens','rows','row0')
        if any(any(a[k]!=b[k] for k in fields) for a,b in zip(*records)):raise ValueError('controls do not share IDs/row mapping')
        if len(keys)!=260 or meta[0]['env']!=meta[1]['env'] or 'EMBER_DETERMINISTIC=0' not in meta[0]['env']:raise ValueError('wrong default controls')
        vocab=meta[0]['vocab'];count=meta[0]['dump_rows'];width=vocab*4
        if (vocab,count)!=(16384,3829) or any((m['vocab'],m['dump_rows'])!=(vocab,count) for m in meta):raise ValueError('row geometry changed')
        expected_offset=0
        for row in records[0]:
            if row['row0']!=expected_offset or row['rows']<=0:raise ValueError('noncontiguous dump rows')
            expected_offset+=row['rows']
        if expected_offset!=count:raise ValueError('row mapping incomplete')
        with z.open('out/pick/'+stems[0]+'.jsonl.rows.f32') as a,z.open('out/pick/'+stems[1]+'.jsonl.rows.f32') as b:
            done=0
            while done<count:
                n=min(8,count-done);aa=a.read(n*width);bb=b.read(n*width)
                if len(aa)!=n*width or len(bb)!=n*width:raise ValueError('short row data')
                hashes[0].update(aa);hashes[1].update(bb)
                au=np.frombuffer(aa,dtype='<u4').reshape(n,vocab);bu=np.frombuffer(bb,dtype='<u4').reshape(n,vocab)
                af=au.view('<f4');bf=bu.view('<f4')
                if not np.isfinite(af).all() or not np.isfinite(bf).all():raise ValueError('nonfinite historical logits')
                equal.extend(np.all(au==bu,axis=1).tolist())
                different_values+=int(np.count_nonzero(au!=bu));top1+=int(np.count_nonzero(np.argmax(af,axis=1)==np.argmax(bf,axis=1)))
                worst=max(worst,float(np.max(np.abs(af.astype(np.float64)-bf.astype(np.float64)))))
                done+=n
            if a.read(1) or b.read(1):raise ValueError('extra row data')
    first=[r['row0'] for r in records[0]];fresh=[r['row0'] for r in records[0] if r['turn']==0];history=[r['row0'] for r in records[0] if r['turn']!=0]
    receipt={'status':'KNOWN_DEFAULT_NONDETERMINISM_NOT_FIXED','historical_source_commit':HISTORICAL,'current_source_commit':current,'source_bodies':sources,'archive':{'path':str(archive),'sha256':archive_hash,'bytes':archive.stat().st_size,'validation':'Full archive hash independently matched; selected row/JSON members fully read and CRC checked, not all-member CRC replay'},'registered_model_sha256':plan['sha256']['model'],'binary_sha256':plan['sha256']['binary'],'prompts_sha256':plan['sha256']['prompts'],'default_control_env':meta[0]['env'],'geometry':meta[0]['health'],'raw_recomparison':{'requests':len(keys),'ids_prompt_stop_row_mapping_equal':True,'row_count':count,'vocab':vocab,'row_sha256':[h.hexdigest() for h in hashes],'bit_equal_rows':sum(equal),'bit_different_rows':len(equal)-sum(equal),'bit_different_values':different_values,'worst_absolute_float64_difference':worst,'top1_equal_rows':top1,'first_sampled_row_equal':sum(equal[i] for i in first),'first_sampled_row_total':len(first),'turn_zero_first_row_equal':sum(equal[i] for i in fresh),'turn_zero_first_row_total':len(fresh),'history_turn_first_row_equal':sum(equal[i] for i in history),'history_turn_first_row_total':len(history)},'permitted_order_example':order_example(),'diagnosis':'The historical and current tuned decode share the exact unordered atomicAdd RMSNorm body and wrapper. Default controls explicitly select that path. Existing deterministic-mode evidence and this source mechanism explain why bit-repeatability is not a default guarantee; this audit does not uniquely attribute every historical logit delta to a captured per-layer event.','gate_interpretation':'The frozen default bit gate remains a valid strict adoption requirement and remains unmet. It cannot isolate selector-induced changes when unchanged off/off model rows already differ. A repeated attempt until scheduling happens to match would not establish reproducibility. No criterion is waived or retroactively changed.','historical_verdict':'BASELINE_UNSTABLE preserved; deterministic paired speeds1.159246560127609 and1.141462409508764 still fail the requirement that both reach1.15.','current_four_partition_scope':'The explicitly experimental PARTS=4 path requires deterministic mode; its synthetic actual-GPU pass is separate from this pre-four-part registered gate.','prospective_minimum_if_needed':['For selector-only default-mode correctness, teacher-force the registered model and capture each current forward row once, then run original/four-part selectors and full sampler against that immutable same row. This avoids conflating upstream nondeterminism, but cannot close historical repeatability or speed requirements.','For unique cause attribution, capture actual same-token RMSNorm input and warp partials at the first divergent stage, and compare permitted atomic accumulation orders against ordered control. Do not infer that causal proof from this toy arithmetic example.','To promise whole-model bit-stable default execution requires a substantive default-path design/change and fresh registered parity/performance gates; simply rerunning or changing tolerances is not a remedy.'],'changes':'No production/default/arithmetic/tolerance change and no GPU/API execution. NumPy is used only for streamed offline raw-row comparisons.','numpy_version':np.__version__}
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.partial');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(output)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=audit(a.root,a.archive,a.output);print(json.dumps(r['raw_recomparison'],indent=2))
