import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('pack',HERE/'package_receipt.py');pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)

def test_compact_pack_retains_hashes_not_arrays_or_account_snapshots(tmp_path):
    archive=tmp_path/'raw.zip'
    kernel={'device':'L4','pass':True,'records':[{'kernel':'rms','n':1,'input':[1],'weights':[1],'first_output':[1],'reference':[1],'error':0,'repeat_equal':True,'milliseconds':[.1],'pass':True}]}
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('summary.json',json.dumps({'status':'PASS'}));z.writestr('science/candidate/default_rows_off.jsonl.rows.f32',b'raw float rows')
        z.writestr('science/kernel_unset_0_0_baseline.log',json.dumps(kernel));z.writestr('science/account_balance.json','private account snapshot')
    sha=pack.digest_file(archive);out=tmp_path/'public';pack.package(archive,sha,out)
    manifest=json.loads((out/'raw_manifest.json').read_text());assert len(manifest['members'])==4
    assert not (out/'science/account_balance.json').exists() and not list(out.rglob('*.f32'))
    records=json.loads((out/'kernel_invocations_compact.json').read_text())['invocations'][0]['summary']['records']
    assert records[0]['error']==0 and records[0]['milliseconds']==[.1] and 'input' not in records[0]
    assert json.loads((out/'packaging.json').read_text())['independent_numerical_verdict']=='NOT_ESTABLISHED_BY_PACKAGER'
    with pytest.raises(ValueError):pack.package(archive,sha,out)

def test_partial_wrong_hash_and_unsafe_zip_refused_without_public_output(tmp_path):
    archive=tmp_path/'raw.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape','x')
    with pytest.raises(ValueError):pack.package(archive,'wrong',tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()
    with pytest.raises(ValueError):pack.package(archive,pack.digest_file(archive),tmp_path/'unsafe')
    assert not (tmp_path/'unsafe').exists() and not list(tmp_path.glob('.ordered-receipt-*'))
