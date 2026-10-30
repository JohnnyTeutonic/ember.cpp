from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
import f16_attention_reference as attention


def test_hand_solvable_causal_gqa_means_and_last_decode():
    q=np.zeros((3,4))
    k=np.zeros((3,2))
    v=np.array([[2,10],[4,20],[8,40]])
    expected=np.array([[2,2,10,10],[3,3,15,15],[14/3,14/3,70/3,70/3]])
    np.testing.assert_allclose(attention.reference(q,k,v,4,2,1,1,0),expected,rtol=0,atol=1e-14)
    np.testing.assert_allclose(attention.reference(q[-1:],k,v,4,2,1,1,2),expected[-1:],rtol=0,atol=1e-14)


def test_hand_solvable_nonzero_softmax_and_head_mapping():
    q=np.array([[0,0],[np.log(3),-np.log(3)]])
    result=attention.reference(q,np.array([[0],[1]]),np.array([[0],[4]]),2,1,1,1,0)
    np.testing.assert_allclose(result,np.array([[0,0],[3,1]]),rtol=0,atol=1e-14)


def test_decode_assembles_its_own_cache_rows():
    def row(value,observed):
        return {'meta':{'hq':2,'hkv':1,'hd':1,'scale':1},
                'arrays':{'q':np.zeros((1,2)),'k_cache':np.zeros((1,1)),
                          'v_cache':np.array([[value]]),'attention':np.array([[observed,observed]])}}
    history={}
    result={}
    attention.record(result,[row(2,2)],history,0,0,'decode')
    attention.record(result,[row(6,4)],history,1,0,'decode')
    assert result['0']['rows']==2
    assert result['0']['worst_absolute']==0
    assert [float(x[0,0]) for x in history[0]['v']]==[2,6]
    with pytest.raises(ValueError,match='missing/reordered'):
        attention.record(result,[row(8,4)],{},1,0,'decode')


def test_invalid_gqa_grouping_is_rejected():
    with pytest.raises(ValueError,match='geometry'):
        attention.reference(np.zeros((1,3)),np.zeros((1,2)),np.zeros((1,2)),3,2,1,1,0)