import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('launch_contract',Path(__file__).with_name('launch_contract_v3.py'))
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


@pytest.mark.parametrize('key',list(contract.REQUIRED_ENV))
def test_missing_required_environment_rejected(key):
    env=dict(contract.REQUIRED_ENV)
    del env[key]
    with pytest.raises(ValueError,match=key):
        contract.validate_environment(env,contract.VERSIONS,{'image_digest':contract.IMAGE_DIGEST})


def test_frozen_environment_accepted():
    assert contract.validate_environment(contract.REQUIRED_ENV,contract.VERSIONS,
        {'image_digest':contract.IMAGE_DIGEST})['versions']==contract.VERSIONS


@pytest.mark.parametrize('key',contract.UNSET_ENV)
def test_inherited_override_rejected(key):
    env=dict(contract.REQUIRED_ENV)
    env[key]='1'
    with pytest.raises(ValueError,match=key):
        contract.validate_environment(env,contract.VERSIONS,{'image_digest':contract.IMAGE_DIGEST})


@pytest.mark.parametrize('field,value',[('backend','cpu'),('max_len',8192),('ready',False),('vocab',16384),('threads',8)])
def test_native_fallback_or_context_drift_rejected(field,value):
    ready={'ready':True,'backend':'cuda','max_len':1024,'vocab':49152,'threads':1,'tensor_inventory':[{'type':0}]}
    ready[field]=value
    with pytest.raises(ValueError):
        contract.validate_backend(ready)


def test_wrong_image_or_patch_version_rejected():
    for versions,provenance in [(dict(contract.VERSIONS,torch='2.4.0'),{'image_digest':contract.IMAGE_DIGEST}),
                               (contract.VERSIONS,{'image_digest':'different'})]:
        with pytest.raises(ValueError):
            contract.validate_environment(contract.REQUIRED_ENV,versions,provenance)
