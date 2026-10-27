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
