import importlib.util
import json
from pathlib import Path
import struct
import zipfile
GATE=Path('/home/jonat/ember_ordered_default_20261101/experiments/spec_r1_2026-10-28/decode_pick_gate.py')
spec=importlib.util.spec_from_file_location('storage_gate',GATE);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def test_frozen_client_paths_and_pack_follow_rows_link(tmp_path):
    science=tmp_path/'out/science';science.mkdir(parents=True);scratch=tmp_path/'owned_scratch';scratch.mkdir()
    out=science/'default_rows_off.jsonl';run=Path(str(out)+'.run');target=scratch/'run';target.mkdir();run.symlink_to(target,target_is_directory=True)
    rows=Path(str(out)+'.rows.f32');raw=scratch/'rows.f32';raw.write_bytes(b'');rows.symlink_to(raw)
    # Exact frozen client mkdir/open/remove operations through directory/file links.
    (run/'www').mkdir(exist_ok=True);dump=run/'dump.txt';dump.write_bytes(b'1 2 3 4\n')
    assert dump.read_bytes()==(target/'dump.txt').read_bytes()
    dump.unlink();dump.write_bytes(b'5 6 7 8\n')
    data=struct.pack('<4f',5,6,7,8)
    with rows.open('wb') as f:f.write(data)
    record={'i':0,'turn':0,'reply':'x','ids':[1],'stop_id':2,'prompt_ids':[3],'tokens':1,'warm_pass_same':True,'rows':1,'row0':0}
    out.write_text(json.dumps(record)+'\n')
    inputs={'binary':'/exact/bin','model':'/exact/model','prompts':'/exact/prompts'}
    leg={'name':'default_rows_off','env':['EMBER_DETERMINISTIC=0'],'dump':True}
    meta={'env':leg['env'],'once':False,'gpu_layers':999,'bin':inputs['binary'],'gguf':inputs['model'],'prompts':inputs['prompts'],'health':{'model':inputs['model'],'deterministic':False},'dump_rows':1,'vocab':4}
    Path(str(out)+'.meta.json').write_text(json.dumps(meta));(science/'outside.server.log').write_text('retained server log')
    assert gate.load_run(science,leg,{'request_keys':[[0,0]],'inputs':inputs})[0]==[record]
    zpath=tmp_path/'out.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,strict_timestamps=False) as z:
        for f in sorted(science.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(science.parent))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        assert z.read('science/default_rows_off.jsonl.rows.f32')==data
        assert z.read('science/outside.server.log')==b'retained server log'
        # pathlib rglob does not descend linked directory; ephemeral dump.txt omitted.
        assert not any('.run/' in n for n in z.namelist())
    assert rows.is_symlink() and raw.read_bytes()==data
