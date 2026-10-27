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


def test_initial_checkpoint_space_failure_can_resume_when_space_recovers(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from winding_field.field import solve
    from winding_field._core.field_store import StorageBudgetError
    run=prepared_run(tmp_path/'run'); free=0
    monkeypatch.setattr('winding_field._core.field_store.shutil.disk_usage',lambda p:SimpleNamespace(free=free))
    config=dict(sweeps=1,field_chunk=8,reserve_bytes=16<<20)
    with pytest.raises(StorageBudgetError):
        solve(run,config)
    assert not (run/'FIELD.json').exists()
    free=1<<30
    assert solve(run,config)['status']=='FIELD_READY'


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


@pytest.mark.parametrize('pitch',[8.65,17.,130.,300.])
def test_exact_field_pitch_retains_fine_prior_and_strict_extent(tmp_path,pitch):
    from winding_field.config import field_geometry
    meta=dict(shape_zyx=[83,92,101],spacing_xyz=[8.64,9.,10.],origin_xyz=[13.,27.,45.],frame='test')
    geometry=field_geometry(meta,pitch)
    np.testing.assert_array_equal(geometry['spacing_xyz'],[pitch]*3)
    np.testing.assert_array_equal(geometry['origin_xyz'],meta['origin_xyz'])
    extent=(np.array(meta['shape_zyx'])[::-1]-1)*meta['spacing_xyz']
    np.testing.assert_array_equal(geometry['shape_zyx'],(np.floor(extent/pitch).astype(int)+1)[::-1])
    assert np.all((np.array(geometry['shape_zyx'])[::-1]-1)*pitch<=extent)
    assert meta['spacing_xyz']==[8.64,9.,10.]


@pytest.mark.parametrize('spacing',[8.64,8.65])
def test_explicit_native_pitch_retains_roundoff_boundary_plane(spacing):
    from winding_field.config import field_geometry
    meta=dict(shape_zyx=[128]*3,spacing_xyz=[spacing]*3,origin_xyz=[0.]*3,frame='test')
    assert field_geometry(meta,spacing)['shape_zyx']==[128]*3


def test_coarse_field_uses_physical_constraints_and_mesh_geometry(tmp_path):
    from winding_field.field import solve,open_field
    from winding_field.meshes import mesh
    run=prepared_run(tmp_path/'run',shape=(12,12,32),chunk=4)
    config=dict(sweeps=1,field_chunk=32,field_spacing_um=7.,seed_stride=4)
    receipt=solve(run,config)
    assert receipt['geometry']['spacing_xyz']==[7.]*3
    assert receipt['geometry']['shape_zyx']==[7,5,9]
    assert receipt['fibre_constraint_rows']>0
    with open_field(run) as src:
        field,valid=src.read([0]*3,list(src.shape))
        np.testing.assert_allclose(field,np.broadcast_to(np.arange(9)*7/16,field.shape),atol=2e-4)
        assert valid.all()
    result=mesh(run,dict(config,mesh_tile=4))
    assert result['status']=='MESH_READY'
    with pytest.raises(ValueError,match='identity|differ'):
        solve(run,dict(config,field_spacing_um=8.))


def test_physical_sampling_orients_directors_and_preserves_support_holes(tmp_path):
    from winding_field.artifacts import Tiles
    from winding_field.field import sample_prior
    maps=Tiles(tmp_path/'maps',shape=(4,4,12),chunk=4,identity={})
    for key in maps.keys():
        lo,hi=maps.box(key); shape=tuple(hi-lo)
        normals=np.zeros(shape+(3,),np.float32); normals[...,0]=(-1)**key[2]
        support=np.ones(shape,bool)
        if key[2]==1: support[:,:,1]=False
        maps.write(key,dict(normals=normals,density=np.full(shape,1/16,np.float32),support=support))
    meta=dict(shape_zyx=[4,4,12],origin_xyz=[10.,20.,30.],spacing_xyz=[2.,3.,4.])
    geometry=dict(origin_xyz=meta['origin_xyz'],spacing_xyz=[3.]*3,shape_zyx=[4,4,8])
    cal=dict(mode='planar',axis_xyz=[1.,0.,0.])
    target,weights,valid=sample_prior(maps,meta,geometry,cal,[0,0,0],[4,4,8])
    assert not valid[:,:,3].any()  # x=4.5 interpolates an unsupported fine node.
    assert valid[:,:,:3].all() and valid[:,:,4:].all()
    np.testing.assert_allclose(target[valid][:,0],1/16)
    assert not target[~valid].any()
    maps.close()


@pytest.mark.parametrize('pitch',[17.,130.])
@pytest.mark.parametrize('point_count',[2,80])
def test_long_qual_spline_constrains_blocks_away_from_its_origin(tmp_path,monkeypatch,pitch,point_count):
    import io
    import sqlite3
    import winding_field.field as field
    from winding_field.artifacts import Tiles
    from winding_field.util import atomic_json,file_sha
    run=tmp_path/'run'; shape=(80,24,24)
    maps=Tiles(run/'maps',shape=shape,chunk=8,identity={})
    for key in maps.keys():
        lo,hi=maps.box(key); local=tuple(hi-lo)
        normals=np.zeros(local+(3,),np.float32); normals[...,0]=1
        maps.write(key,dict(density=np.full(local,1/100,np.float32),normals=normals,support=np.ones(local,bool)))
    meta=dict(shape_zyx=list(shape),spacing_xyz=[9.]*3,origin_xyz=[10.,20.,30.],frame='test',
              prediction='qual-v2',tiles_sha256=maps.identity())
    maps.close(); atomic_json(run/'PREPARED.json',meta)
    (run/'fibres').mkdir()
    points=np.column_stack([np.full(80,41.5),np.full(80,51.5),30+np.arange(80)*9.])
    if point_count==2:
        points=points[68:70]
    blob=io.BytesIO(); np.save(blob,points,allow_pickle=False)
    with sqlite3.connect(run/'fibres/fibres.sqlite') as db:
        db.executescript('CREATE TABLE curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT);'
                        'CREATE VIRTUAL TABLE curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
        db.execute('INSERT INTO curves VALUES (1,?,?,?)',('qual',blob.getvalue(),'V'))
        db.execute('INSERT INTO curve_bounds VALUES (1,41.5,41.5,51.5,51.5,?,?)',(points[:,2].min(),points[:,2].max()))
    atomic_json(run/'FIBRES.json',dict(sha256=file_sha(run/'fibres/fibres.sqlite')))
    calls=[]; pair=field.pair_constraints
    def constraints(a,b,origin,*args,**kwargs):
        calls.append((origin.copy(),a.copy(),b.copy()))
        return pair(a,b,origin,*args,**kwargs)
    monkeypatch.setattr(field,'pair_constraints',constraints)
    result=field.solve(run,dict(sweeps=1,field_chunk=4,field_spacing_um=pitch))
    assert result['fibre_constraint_rows']>0
    assert any(origin[2]>30 and len(a) for origin,a,b in calls)
    for origin,a,b in calls:
        assert len(np.unique(np.c_[a,b],axis=0))==len(a)
    with field.open_field(run) as source:
        values,support=source.read([0]*3,list(source.shape))
        expected=np.broadcast_to(np.arange(source.shape[2])*pitch/100,values.shape)
        np.testing.assert_allclose(values,expected,atol=2e-4)
        assert support.all()
