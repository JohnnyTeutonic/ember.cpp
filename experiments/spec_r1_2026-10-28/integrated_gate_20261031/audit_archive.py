"""Post-run independent ZIP/source/full-raw replay. No GPU calls or gate changes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import zipfile


def digest_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()

def digest(path):
    with Path(path).open('rb') as stream:return digest_stream(stream)

def require(value,message):
    if not value:raise ValueError(message)

def audit(archive,stage,output):
    archive,stage,output=map(Path,(archive,stage,output))
    require(not output.exists(),'fresh audit output required')
    output.mkdir()
    manifest=json.loads((stage/'e10_integrated_manifest.json').read_text())
    require(manifest['packet_commit']=='153f0e9352e14cfbd2b6c7328cb32e8d0f6a7ce5','wrong frozen V2 packet')
    require(digest(stage/'e10_integrated_manifest.json')=='21f2295458d4a00ee5d4de7fdaaff7ee10326065fa92ef914f004c970284cfd5','manifest substitution')
    require(digest(stage/'e10_integrated_source.tar')==manifest['archive_sha256'],'source archive hash mismatch')
    root=stage/'extracted/ember'
    packet=root/'experiments/spec_r1_2026-10-28/integrated_gate_20261031'
    for name,expected in manifest['source_files'].items():
        require(digest(root/name)==expected,'frozen source extraction drift: '+name)
    spec=importlib.util.spec_from_file_location('frozen_e10_check',packet/'check.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    archive_sha=digest(archive)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names)),'duplicate ZIP members refused')
        bad=z.testzip();require(bad is None,'ZIP CRC failure: '+str(bad))
        require(json.loads(z.read('manifest.json'))==manifest,'outer source manifest mismatch')
        require(json.loads(z.read('science/inputs.json'))==manifest,'scientific source manifest mismatch')
        report=json.loads(z.read('science/report.json'))
        execution=json.loads(z.read('science/execution.json'))
        transport=json.loads(z.read('transport_summary.json'))
        require(execution['complete'] is True and execution['scientific_pass'] is True,'VM scientific completion missing')
        require(transport['complete'] is True and transport['scientific_pass'] is True,'transport completion missing')
        with z.open('science/work/synthetic.gguf') as stream:
            model_sha=digest_stream(stream)
        require(model_sha==checker.MODEL==report['model_sha256'],'fixture hash mismatch')
        for member,key in [('science_build/integrated_pick_probe','binary_sha256'),('science_build/ember/tinyllama_server','server_sha256')]:
            with z.open(member) as stream:actual=digest_stream(stream)
            require(actual==report[key],'actual archived binary mismatch: '+member)
        extracted=0
        for info in z.infolist():
            if not info.filename.startswith('science/') or info.is_dir() or info.filename.startswith('science/work/'):
                continue
            require(not stat.S_ISLNK(info.external_attr>>16),'archive symlink refused')
            target=(output/info.filename).resolve()
            require(target.is_relative_to(output.resolve()),'archive path escape')
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix(target.suffix+'.partial')
            with z.open(info) as source,temp.open('wb') as destination:shutil.copyfileobj(source,destination,1048576)
            temp.replace(target);extracted+=1
        hardware=z.read('hardware.log').decode(errors='replace')
    replay=checker.audit(output/'science')
    require(replay['kernel_cases']==96 and replay['graph_legs']==30 and replay['full_rows']==240,'registered result counts changed')
    graphs=[]
    for row in report['graphs']:
        graphs.append({k:row[k] for k in ('mode','repeat','prefix','row_file','row_sha256','successful_graph_calls','graph_has_pick','graph_pick_parts','scratch_nonnull','scratch_stable')})
        graphs[-1]['transitions']=row['transitions']
        if 'transition_file' in row:graphs[-1]['transition_sha256']=digest(output/'science'/row['transition_file'])
        graphs[-1]['selected_ids']=[{'step':x['step'],'temperature':x['temperature'],'id':x['full_id'],'rng_sha256':hashlib.sha256(x['full_rng'].encode()).hexdigest()} for x in row['sampler']]
    kernels=[]
    for row in report['kernels']:
        kernels.append({**{k:row[k] for k in ('vocab','rows','case')},'input_sha256':digest(output/'science'/row['input']),'original_sha256':digest(output/'science'/row['original']),'partition4_sha256':digest(output/'science'/row['partition4'])})
    raw_files={str(f.relative_to(output)):digest(f) for f in (output/'science').rglob('*') if f.is_file()}
    receipt={'raw_file_sha256':raw_files,'compact_public_limit':'Full float rows and complete RNG states are not included in the compact public receipt. They remain in the canonical LOCAL ZIP at the published SHA; the compact receipt alone cannot replay every raw comparison.','status':'PASS_SYNTHETIC_INTEGRATED_CORRECTNESS_ONLY','archive':{'path':str(archive),'sha256':archive_sha,'bytes':archive.stat().st_size,'members':len(z.infolist()),'crc':'ALL_MEMBERS_PASS'},'source_packet_commit':manifest['packet_commit'],'production_source_commit':manifest['production_source_commit'],'source_archive_sha256':manifest['archive_sha256'],'manifest_sha256':digest(stage/'e10_integrated_manifest.json'),'protocol_sha256':digest(packet/'protocol.json'),'checker_sha256':digest(packet/'check.py'),'probe_source_sha256':digest(packet/'integrated_probe.cpp'),'controller_source_sha256':digest(packet/'vm.py'),'fixture_sha256':model_sha,'binary_sha256':report['binary_sha256'],'server_sha256':report['server_sha256'],'device':report['device'],'replay':replay,'raw_extract_scope':'Science only; model streamed for hash, work and complete build tree not extracted','extracted_science_files':extracted,'full_sampler_comparisons':sum(len(x['sampler']) for x in report['graphs']),'api_fallback_requests':48,'graphs':graphs,'kernels':kernels,'hardware_log':hardware,'prior_failure':{'archive_sha256':'24f37c62d69f65f665f4ff5efa1c7915904bcf45bcf4459bb4d8e5f9cfe1f724','status':'PRESERVED_ENVIRONMENT_FAILURE_BEFORE_MODEL_ROWS'},'limitations':['Synthetic F32 Llama only, not registered pick_e1_L16.','No speed/adoption gate was run.','Historical default BASELINE_UNSTABLE and deterministic speed-pair failure remain unchanged.','No production arithmetic/default/tolerance change.','Bounded exercised capture/fallback paths, not universal lifetime or model coverage.']}
    target=output/'independent_audit.json';temp=target.with_suffix('.partial');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(target)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('stage',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();r=audit(a.archive,a.stage,a.output);print(json.dumps({k:r[k] for k in ('status','archive','replay','full_sampler_comparisons','api_fallback_requests')},indent=2))
