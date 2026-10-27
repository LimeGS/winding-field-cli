import importlib.util
import io
import json
import os
import sqlite3
import sys

import numpy as np
import pytest


def adapter():
    assert importlib.util.find_spec('winding_field.splines'), 'Qual spline adapter missing'
    from winding_field import splines
    return splines


def prepared(root,shape=(8,8,24),*,probabilities=None,gap=None,chunk=8):
    from winding_field.artifacts import Tiles
    from winding_field.util import atomic_json
    maps=Tiles(root/'maps',shape=shape,chunk=chunk,identity={'fixture':'qual'})
    if probabilities is None:
        probabilities=np.empty(shape+(2,),np.float32)
        probabilities[...,0]=.2; probabilities[...,1]=.7
    for key in maps.keys():
        lo,hi=maps.box(key); region=tuple(slice(a,b) for a,b in zip(lo,hi))
        support=np.ones(tuple(hi-lo),bool)
        if gap is not None:
            support[...,np.arange(lo[2],hi[2])==gap]=False
        maps.write(key,dict(probabilities=probabilities[region],support=support))
    meta=dict(shape_zyx=list(shape),spacing_xyz=[9.]*3,origin_xyz=[100.,200.,300.],
              frame='fixture',prediction='qual-v2',tiles_sha256=maps.identity(),
              source=dict(source='fixture',level=0,shape=list(shape),spacing_xyz=[9.]*3,origin_xyz=[100.,200.,300.]))
    atomic_json(root/'PREPARED.json',meta)
    return maps,meta


def test_probability_materialization_resumes_and_preserves_completed_input(tmp_path):
    splines=adapter()
    maps,meta=prepared(tmp_path,gap=10)
    root=tmp_path/'fibres'
    def stop():
        path=root/'input.sqlite'
        if not path.exists(): return False
        with sqlite3.connect(path) as db:
            return db.execute('SELECT count(*) FROM tiles').fetchone()[0]>=1
    assert splines.materialize_predictions(maps,root,stop=stop) is None
    path=splines.materialize_predictions(maps,root)
    data=np.load(path,mmap_mode='r',allow_pickle=False)
    assert data.shape==(2,8,8,24) and data.dtype==np.float32
    np.testing.assert_allclose(data[0,:,:,:10],.2)
    np.testing.assert_allclose(data[1,:,:,:10],.7)
    assert not data[:,:,:,10].any()
    before=path.stat().st_mtime_ns
    assert splines.materialize_predictions(maps,root)==path
    assert path.stat().st_mtime_ns==before
    changed=np.load(path,mmap_mode='r+',allow_pickle=False); changed[0,0,0,0]=.3; changed.flush()
    with pytest.raises(ValueError,match='changed|corrupt'):
        splines.materialize_predictions(maps,root)
    maps.close()


def test_materialization_checks_disk_reserve_before_allocating(tmp_path,monkeypatch):
    splines=adapter()
    from types import SimpleNamespace
    maps,_=prepared(tmp_path)
    monkeypatch.setattr(splines.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024))
    with pytest.raises(OSError,match='space|reserve'):
        splines.materialize_predictions(maps,tmp_path/'fibres',reserve_bytes=1024)
    assert not list((tmp_path/'fibres').glob('*.npy'))
    maps.close()


def test_stop_reaps_generator_child(tmp_path):
    splines=adapter()
    ready=tmp_path/'ready'
    command=[sys.executable,'-c',
             'import os,time; from pathlib import Path; Path('+repr(str(ready))+').write_text(str(os.getpid())); time.sleep(30)']
    assert splines.run_child(command,tmp_path/'child.log',lambda:ready.exists()) is False
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()),0)


def upstream_curves(path,shape,curves):
    from avf_spline_generator.afv import export_afv
    export_afv(((family,(p-.5)/2) for family,p in curves),path,voxel_um=9.,origin_xyz=[0]*3,volume_shape=shape)


def test_import_multiblock_curves_splits_support_gap_and_resumes(tmp_path):
    splines=adapter()
    maps,meta=prepared(tmp_path,shape=(8,8,600),gap=300,chunk=32)
    p=np.column_stack([np.arange(300)*2+.5,np.full(300,3.5),np.full(300,3.5)])
    upstream_curves(tmp_path/'upstream.afv',maps.shape,[(2,p),(1,p[:5]+[0,1,0])])
    root=tmp_path/'fibres'; root.mkdir()
    def stop():
        path=root/'fibres.sqlite'
        if not path.exists(): return False
        with sqlite3.connect(path) as db:
            return db.execute('SELECT count(*) FROM done').fetchone()[0]>=1
    assert splines.import_curves(tmp_path/'upstream.afv',root,maps,meta,stop=stop) is False
    assert splines.import_curves(tmp_path/'upstream.afv',root,maps,meta) is True
    with sqlite3.connect(root/'fibres.sqlite') as db:
        curves=[(f,np.load(io.BytesIO(blob),allow_pickle=False)) for f,blob in db.execute('SELECT family,points FROM curves ORDER BY id')]
        assert len(curves)==3 and {f for f,p in curves}=={'H','V'}
        assert db.execute('SELECT count(*) FROM curve_bounds').fetchone()==(3,)
    np.testing.assert_allclose(curves[0][1][0],[104.5,231.5,331.5])
    for family,q in curves:
        assert not (q[:,0].min()<2800 and q[:,0].max()>2800)
    before=(root/'fibres.sqlite').read_bytes()
    assert splines.import_curves(tmp_path/'upstream.afv',root,maps,meta) is True
    assert (root/'fibres.sqlite').read_bytes()==before
    maps.close()


def test_support_split_checks_between_endpoints(tmp_path):
    splines=adapter()
    maps,_=prepared(tmp_path,gap=10)
    p=np.array([[1.,3.,3.],[3.,3.,3.],[18.,3.,3.],[22.,3.,3.]])
    pieces=list(splines.supported_pieces(p,maps))
    assert len(pieces)==2
    np.testing.assert_array_equal(pieces[0],p[:2])
    np.testing.assert_array_equal(pieces[1],p[2:])
    maps.close()


def test_import_rejects_discontinuous_afv_blocks(tmp_path):
    splines=adapter()
    maps,meta=prepared(tmp_path,shape=(8,8,600),chunk=32)
    p=np.column_stack([np.arange(300)*2+.5,np.full(300,3.5),np.full(300,3.5)])
    path=tmp_path/'upstream.afv'; upstream_curves(path,maps.shape,[(2,p)])
    with sqlite3.connect(path) as db:
        db.execute('UPDATE blocks SET first_segment=255 WHERE first_segment=256')
    root=tmp_path/'fibres'; root.mkdir()
    with pytest.raises(ValueError,match='block|continu'):
        splines.import_curves(path,root,maps,meta)
    maps.close()


def test_official_cpu_generator_produces_family_curves_across_tiles(tmp_path):
    splines=adapter()
    z,y,x=np.indices((24,32,96))
    probabilities=np.stack([np.exp(-((z-7)**2+(y-7)**2)/8),
                            np.exp(-((z-17)**2+(y-24)**2)/8)],axis=-1).astype(np.float32)
    probabilities[:,:,:4]=0; probabilities[:,:,-4:]=0
    maps,meta=prepared(tmp_path,shape=z.shape,probabilities=probabilities,chunk=16)
    maps.close()
    from winding_field.fibres import trace
    result=trace(tmp_path,dict(threads=2,reserve_bytes=0))
    assert result['status']=='FIBRES_READY' and result['count']>=2
    with sqlite3.connect(tmp_path/'fibres/fibres.sqlite') as db:
        curves=[(family,np.load(io.BytesIO(blob),allow_pickle=False)) for family,blob in db.execute('SELECT family,points FROM curves')]
    assert {family for family,p in curves}=={'V','H'}
    assert max(np.ptp(p[:,0]) for family,p in curves)>32*9
    assert trace(tmp_path,dict(threads=2,reserve_bytes=0))==result


def test_runtime_identity_includes_model_json(tmp_path,monkeypatch):
    import winding_field.workflow as workflow
    package=tmp_path/'package'; package.mkdir()
    (package/'workflow.py').write_text('fixture')
    plans=package/'qual_plans.json'; plans.write_text('{"version":1}')
    monkeypatch.setattr(workflow,'__file__',str(package/'workflow.py'))
    before=workflow.implementation_sha()
    plans.write_text('{"version":2}')
    assert workflow.implementation_sha()!=before
