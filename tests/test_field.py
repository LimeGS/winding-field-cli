import json

import numpy as np
import pytest


def prepared_run(root,shape=(6,6,24),chunk=8,variable=False):
    from winding_field.artifacts import Tiles
    from winding_field.fibres import trace
    from winding_field.util import atomic_json
    root.mkdir()
    maps=Tiles(root/'maps',shape=shape,chunk=chunk,identity={'test':'analytic physical layers'})
    for key in maps.keys():
        lo,hi=maps.box(key); local=tuple(hi-lo)
        density=np.full(local,1/16,np.float32)
        if variable:
            density=np.broadcast_to(np.where(np.arange(lo[2],hi[2])<12,1/8,1/16),local).astype(np.float32).copy()
        normals=np.zeros(local+(3,),np.float32); normals[...,0]=1
        tangent=np.zeros_like(normals); tangent[...,2]=1
        maps.write(key,dict(density=density,normals=normals,tangents=tangent,support=np.ones(local,bool),
                            presence=np.ones(local,np.float32),ct=np.full(local,128,np.uint8)))
    meta=dict(shape_zyx=list(shape),spacing_xyz=[2.,3.,4.],origin_xyz=[10.,20.,30.],frame='test',
              tiles_sha256=maps.identity(),prediction='analytic-test',physical_accuracy='synthetic')
    atomic_json(root/'PREPARED.json',meta); maps.close()
    trace(root,{'seed_stride':4})
    return root


def test_physical_planar_field_and_fibre_constraints(tmp_path):
    from winding_field.field import solve,open_field
    run=prepared_run(tmp_path/'run')
    receipt=solve(run,{'sweeps':2,'field_chunk':8})
    assert receipt['status']=='FIELD_READY'
    assert receipt['fibre_constraint_rows']>0
    with open_field(run) as source:
        field,valid=source.read([0,0,0],[6,6,24])
    expected=np.broadcast_to(np.arange(24)/8,field.shape)
    np.testing.assert_allclose(field,expected,atol=2e-4)
    assert valid.all()
    assert receipt['physical_acceptance'] is False


def test_nonconstant_layer_separation_is_not_replaced_by_mean(tmp_path):
    from winding_field.field import solve,open_field
    run=prepared_run(tmp_path/'run',shape=(6,6,32),chunk=32,variable=True)
    solve(run,{'sweeps':1,'field_chunk':32})
    with open_field(run) as source:
        field,_=source.read([0,0,0],[6,6,32])
    delta=np.diff(field[3,3])
    assert np.mean(delta[2:10])==pytest.approx(.25,abs=.002)
    assert np.mean(delta[16:28])==pytest.approx(.125,abs=.002)


def test_interrupted_solver_resumes_exact_problem(tmp_path):
    from winding_field.field import solve,open_field
    run=prepared_run(tmp_path/'run')
    calls=0
    def stop():
        nonlocal calls
        calls+=1
        return calls>4
    receipt=solve(run,{'sweeps':2,'field_chunk':8},stop=stop)
    assert receipt['status']=='PAUSED'
    assert not (run/'FIELD.json').exists()
    with pytest.raises(ValueError,match='identity|differ'):
        solve(run,{'sweeps':3,'field_chunk':8})
    assert solve(run,{'sweeps':2,'field_chunk':8})['status']=='FIELD_READY'
    with open_field(run) as source:
        field,_=source.read([0,0,0],[6,6,24])
    np.testing.assert_allclose(field[3,3],np.arange(24)/8,atol=2e-4)


def test_failed_local_convergence_does_not_publish_field(tmp_path):
    from winding_field.field import solve
    run=prepared_run(tmp_path/'run',shape=(6,6,32),chunk=32,variable=True)
    with pytest.raises(RuntimeError,match='converge'):
        solve(run,{'sweeps':1,'field_chunk':32,'maxiter':0})
    assert not (run/'FIELD.json').exists()


@pytest.mark.parametrize('limit',[0,1])
def test_field_initialization_stops_at_block_boundary_and_resumes(tmp_path,limit):
    from winding_field.field import solve
    run=prepared_run(tmp_path/'run'); checkpoint=run/'field/CHECKPOINT.json'
    def stop():
        return checkpoint.exists() and len(json.loads(checkpoint.read_bytes())['chunks'])>=limit
    config={'sweeps':1,'field_chunk':8}
    assert solve(run,config,stop=stop)['status']=='PAUSED'
    saved=json.loads(checkpoint.read_bytes())
    assert len(saved['chunks'])==limit
    assert not saved['progress'].get('initialization_done')
    assert not (run/'FIELD.json').exists()
    assert solve(run,config)['status']=='FIELD_READY'


def test_field_initialization_preserves_disk_reserve(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import winding_field.field as field
    from winding_field.field import solve
    from winding_field._core.field_store import StorageBudgetError
    run=prepared_run(tmp_path/'run'); checkpoint=run/'field/CHECKPOINT.json'
    reserve=16<<20
    initializing=False; Store=field.BlockStore
    def initialization_store(*args,**kwargs):
        nonlocal initializing
        store=Store(*args,**kwargs)
        initializing=True
        return store
    monkeypatch.setattr(field,'BlockStore',initialization_store)
    monkeypatch.setattr('winding_field._core.field_store.shutil.disk_usage',
        lambda path:SimpleNamespace(free=reserve if initializing else reserve*10))
    try:
        solve(run,{'sweeps':1,'field_chunk':8,'reserve_bytes':reserve})
    except StorageBudgetError:
        pass
    assert json.loads(checkpoint.read_bytes())['chunks']=={}
    assert not (run/'FIELD.json').exists()


def test_resumed_fibre_archive_uses_indexed_tile_lookup(tmp_path):
    import sqlite3
    from winding_field.fibres import trace
    run=prepared_run(tmp_path/'run')
    count=json.loads((run/'FIBRES.json').read_bytes())['count']
    with sqlite3.connect(run/'fibres/fibres.sqlite') as db:
        db.execute('DROP INDEX IF EXISTS curves_tile')
    (run/'FIBRES.json').unlink()
    assert trace(run,{'seed_stride':4})['count']==count
    with sqlite3.connect(run/'fibres/fibres.sqlite') as db:
        plan=db.execute('EXPLAIN QUERY PLAN SELECT points FROM curves WHERE tile=?',('0_0_0',)).fetchall()
        assert all('SCAN curves' not in row[-1] for row in plan),plan
        assert db.execute('SELECT count(*) FROM curves WHERE tile=?',('0_0_0',)).fetchone()[0]>0
