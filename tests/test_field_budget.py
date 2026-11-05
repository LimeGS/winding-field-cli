import json

import numpy as np
import pytest

from block_solver import solve_blocks
from field_store import BlockStore


def checkpoint(tmp_path):
    root=tmp_path/'field'
    def observations(lo,hi):
        shape=tuple(hi-lo);target=np.zeros((*shape,3),'f4');target[...,0]=1
        return target,np.ones(shape,'f4'),np.ones(shape,bool),{}
    options=dict(shape=(4,4,4),chunk=4,identity={'test':'bounded'},load_observations=observations,
                 sweeps=1,halo=1,reserve_bytes=0,origin_xyz=[0.]*3,spacing_xyz=[1.]*3,order=[(0,0,0)])
    solve_blocks(root,**options)
    return root,options


def test_extension_preserves_field_and_core_resumes_next_sweep(tmp_path):
    from winding_field.field_budget import effective_budget,extend
    root,options=checkpoint(tmp_path)
    before=json.loads((root/'CHECKPOINT.json').read_bytes())
    extend(root,2,1,reserve_bytes=0)
    after=json.loads((root/'CHECKPOINT.json').read_bytes())
    assert after['progress']==before['progress'] and after['chunks']==before['chunks']
    assert dict(after['identity'],sweeps=1)==before['identity']
    assert effective_budget(root,1)==2
    solve_blocks(root,**dict(options,sweeps=2),resume=True)
    current=json.loads((root/'CHECKPOINT.json').read_bytes())
    assert len(current['progress']['completed_sweeps'])==2
    assert current['sweep_extensions']==after['sweep_extensions']
    unchanged=(root/'CHECKPOINT.json').read_bytes()
    extend(root,2,1,reserve_bytes=0)
    assert (root/'CHECKPOINT.json').read_bytes()==unchanged
    with pytest.raises(ValueError):extend(root,1,1,reserve_bytes=0)


@pytest.mark.parametrize('after_publish',[False,True])
def test_absolute_target_recovers_from_publication_error(tmp_path,monkeypatch,after_publish):
    from winding_field.field_budget import extend,effective_budget
    root,_=checkpoint(tmp_path);original=BlockStore._publish
    def crash(self,manifest,payload=None):
        if after_publish:original(self,manifest,payload)
        raise OSError('injected publication failure')
    with monkeypatch.context() as m:
        m.setattr(BlockStore,'_publish',crash)
        with pytest.raises(OSError):extend(root,2,1,reserve_bytes=0)
    assert effective_budget(root,1)==(2 if after_publish else 1)
    extend(root,2,1,reserve_bytes=0)
    assert len(json.loads((root/'CHECKPOINT.json').read_bytes())['sweep_extensions'])==1


@pytest.mark.parametrize('damage',['published','downstream','payload','history','incomplete','cursor','writer'])
def test_extension_rejects_unsafe_checkpoint(tmp_path,damage):
    from winding_field.field_budget import extend
    root,_=checkpoint(tmp_path);path=root/'CHECKPOINT.json';before=path.read_bytes()
    data=json.loads(before)
    if damage=='published':(tmp_path/'FIELD.json').write_text('{}')
    elif damage=='downstream':(tmp_path/'meshes').mkdir()
    elif damage=='payload':(root/next(iter(data['chunks'].values()))['path']).write_bytes(b'bad')
    elif damage=='history':data['identity']['sweeps']=2
    elif damage=='incomplete':data['chunks']={}
    elif damage=='cursor':data['progress']['next_block']=0
    if damage in ('history','incomplete','cursor'):path.write_text(json.dumps(data))
    if damage=='writer':
        with BlockStore(root,data['shape'],data['chunk'],data['identity'],resume=True):
            with pytest.raises(RuntimeError,match='writer'):extend(root,2,1,reserve_bytes=0)
    else:
        with pytest.raises((ValueError,OSError)):extend(root,2,1,reserve_bytes=0)
