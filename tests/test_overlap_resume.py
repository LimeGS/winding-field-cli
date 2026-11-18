import json
from pathlib import Path

import numpy as np
import pytest

from winding_field.flatten import iter_overlap_pairs
from winding_field.overlap_audit import audit_uv


def crowded():
    rng=np.random.default_rng(7)
    triangles=rng.uniform(0,4,size=(300,3,2))+np.linspace(0,1.5,300)[:,None,None]
    uv=triangles.reshape(-1,2);faces=np.arange(len(uv)).reshape(-1,3)
    return uv,faces


def flags(pairs,count):
    involved=np.zeros(count,bool)
    for i,j in pairs:involved[[i,j]]=True
    return involved


def test_resume_audit_covers_every_pair_once(tmp_path):
    uv,faces=crowded()
    expected=list(iter_overlap_pairs(uv,faces));assert expected
    full=audit_uv(uv,faces,tmp_path/'full',guard=lambda:None,commit_tests=1)
    assert full['status']=='COMPLETE' and full['pairs']==len(expected)
    np.testing.assert_array_equal(full['involved'],flags(expected,len(faces)))
    again=audit_uv(uv,faces,tmp_path/'full',guard=lambda:None)
    assert again['pairs']==full['pairs'] and again['narrow_tests']==full['narrow_tests']

    for stop_at in (64,128,192):
        out=tmp_path/f'resume-{stop_at}'
        def stop():
            path=out/'AUDIT.json'
            if path.exists() and json.loads(path.read_text())['cursor']>=stop_at:
                raise InterruptedError('commit boundary')
        with pytest.raises(InterruptedError):
            audit_uv(uv,faces,out,guard=stop,commit_tests=1)
        state=json.loads((out/'AUDIT.json').read_text())
        assert state['status']=='AUDIT_PENDING' and state['cursor']==stop_at
        resumed=audit_uv(uv,faces,out,guard=lambda:None,commit_tests=1)
        assert resumed['status']=='COMPLETE'
        assert resumed['pairs']==full['pairs'] and resumed['narrow_tests']==full['narrow_tests']
        np.testing.assert_array_equal(resumed['involved'],full['involved'])


def test_mutated_input_or_state_rejects_before_completion(tmp_path):
    uv,faces=crowded();base=tmp_path/'base'
    audit_uv(uv,faces,base,guard=lambda:None)
    mutated_faces=faces.copy();mutated_faces[0]=np.roll(faces[0],1)
    with pytest.raises(ValueError,match='changed'):
        audit_uv(uv+1,faces,base,guard=lambda:None)
    with pytest.raises(ValueError,match='changed'):
        audit_uv(uv,mutated_faces,base,guard=lambda:None)
    state=json.loads((base/'AUDIT.json').read_text())
    for key,value,message in (('cursor',len(faces)+1,'cursor'),('involved_bits','AAAA','bits')):
        path=tmp_path/f'corrupt-{key}';path.mkdir()
        corrupt=dict(state,**{key:value})
        (path/'AUDIT.json').write_text(json.dumps(corrupt))
        with pytest.raises(ValueError,match=message):
            audit_uv(uv,faces,path,guard=lambda:None)


def test_empty_mesh_audit_is_complete(tmp_path):
    receipt=audit_uv(np.empty((0,2)),np.empty((0,3),np.int64),tmp_path,guard=lambda:None)
    assert receipt['status']=='COMPLETE' and receipt['pairs']==0 and not receipt['involved'].any()
