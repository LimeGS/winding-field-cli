import json
import sqlite3

import numpy as np
import pytest

from test_field import prepared_run


def export_fixture(root):
    prepared_run(root)
    path=root/'PREPARED.json'; meta=json.loads(path.read_bytes())
    meta['source']=dict(source='test-ct',level=0,shape=meta['shape_zyx'],
                        origin_xyz=meta['origin_xyz'],spacing_xyz=meta['spacing_xyz'])
    path.write_text(json.dumps(meta))
    return root


def test_lasagna_stops_after_one_chunk_resumes_and_checks_retained_bytes(tmp_path,monkeypatch):
    import winding_field.exports as exports
    root=export_fixture(tmp_path/'run'); writes=[]; original=exports.atomic_bytes
    def write(path,data):
        writes.append(path)
        original(path,data)
    monkeypatch.setattr(exports,'atomic_bytes',write)
    assert exports.export_lasagna(root,stop=lambda:len(writes)>=1) is None
    assert len(writes)==1
    first=writes[0]; before=first.stat().st_mtime_ns
    assert not (root/'lasagna/fibres.lasagna.json').exists()
    assert exports.export_lasagna(root) is True
    assert len(writes)==21
    assert first.stat().st_mtime_ns==before
    first.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='export|corrupt'):
        exports.export_lasagna(root)


def test_afv_pauses_at_record_and_resumes_without_reinserting(tmp_path):
    from winding_field.exports import export_afv
    from winding_field.util import file_sha
    root=export_fixture(tmp_path/'run'); pending=root/'fibres/fibres.partial.afv'
    def stop():
        if not pending.exists(): return False
        with sqlite3.connect(pending) as db:
            return db.execute('SELECT count(*) FROM fibers').fetchone()[0]>=1
    assert export_afv(root,stop) is None
    assert not (root/'fibres/fibres.afv').exists()
    with sqlite3.connect(pending) as db:
        assert db.execute('SELECT count(*) FROM fibers').fetchone()[0]==1
        retained=db.execute('SELECT points FROM blocks').fetchone()[0]
        # A resumed exporter must not delete/reinsert the committed record.
        db.executescript("CREATE TRIGGER keep_fibre BEFORE DELETE ON fibers BEGIN SELECT RAISE(ABORT,'lost progress'); END;")
    sha=export_afv(root)
    assert sha==file_sha(root/'fibres/fibres.afv')
    assert not pending.exists()
    with sqlite3.connect(root/'fibres/fibres.afv') as db, sqlite3.connect(root/'fibres/fibres.sqlite') as src:
        assert db.execute('SELECT count(*) FROM fibers').fetchone()==src.execute('SELECT count(*) FROM curves').fetchone()
        assert db.execute('SELECT points FROM blocks ORDER BY id LIMIT 1').fetchone()[0]==retained
        assert json.loads(db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()[0]) is True


def test_interrupted_afv_rejects_changed_source(tmp_path):
    from winding_field.exports import export_afv
    root=export_fixture(tmp_path/'run')
    assert export_afv(root,lambda:True) is None
    with sqlite3.connect(root/'fibres/fibres.sqlite') as db:
        db.execute("UPDATE curves SET tile='changed' WHERE id=1")
    with pytest.raises(ValueError,match='changed|identity'):
        export_afv(root)


@pytest.mark.parametrize('mutation',[
    'UPDATE blocks SET points=zeroblob(length(points)) WHERE id=1',
    'DELETE FROM fibers WHERE id=2',
    'UPDATE fibers SET length=0 WHERE id=1',
    'UPDATE block_bounds SET min_x=0,max_x=0 WHERE id=1',
])
def test_afv_resume_rejects_corrupted_retained_inventory(tmp_path,mutation):
    from winding_field.exports import export_afv
    root=export_fixture(tmp_path/'run'); pending=root/'fibres/fibres.partial.afv'
    def stop():
        if not pending.exists(): return False
        with sqlite3.connect(pending) as db:
            return db.execute('SELECT count(*) FROM fibers').fetchone()[0]>=2
    assert export_afv(root,stop) is None
    with sqlite3.connect(pending) as db:
        db.execute(mutation)
    with pytest.raises(ValueError,match='corrupt|changed|inventory'):
        export_afv(root)
    assert not (root/'fibres/fibres.afv').exists()


def test_workflow_export_stop_does_not_complete_prepare_stage(tmp_path,monkeypatch):
    from winding_field.demo import synthetic
    from winding_field.workflow import run
    import winding_field.exports as exports
    root=tmp_path/'run'; source=synthetic(tmp_path/'ct.zarr')
    original=exports.atomic_bytes; calls=0
    def interrupt(path,data):
        nonlocal calls
        original(path,data); calls+=1
        (root/'STOP').touch()
    monkeypatch.setattr(exports,'atomic_bytes',interrupt)
    config=dict(source=source,output=str(root),prediction='classical',cache=str(tmp_path/'cache'),reserve_bytes=0)
    state=run(config,until='prepare')
    assert state['status']=='PAUSED' and not state['stages'] and calls==1
    monkeypatch.setattr(exports,'atomic_bytes',original)
    state=run(state['config'],resume=True,until='prepare')
    assert 'prepare' in state['stages']


def qual_export_fixture(root):
    from test_splines import prepared,upstream_curves
    from winding_field.splines import import_curves
    from winding_field.util import atomic_json,file_sha
    maps,meta=prepared(root,shape=(8,8,600),chunk=32)
    # Observation pitch 9 µm, selected-source pitch 4.5 µm, with a nonzero ROI.
    meta['source'].update(spacing_xyz=[4.5]*3,origin_xyz=[10.,20.,30.],shape=[100,100,1400],level=1)
    meta['origin_xyz']=[100.,200.,300.]
    atomic_json(root/'PREPARED.json',meta)
    p=np.column_stack([np.arange(300)*2+.5,np.full(300,3.5),np.full(300,3.5)])
    upstream_curves(root/'upstream.afv',maps.shape,[(2,p),(1,p[:5]+[0,1,0])])
    (root/'fibres').mkdir()
    assert import_curves(root/'upstream.afv',root/'fibres',maps,meta)
    maps.close()
    atomic_json(root/'FIBRES.json',dict(sha256=file_sha(root/'fibres/fibres.sqlite')))
    return root


def test_qual_afv_preserves_families_transform_and_bounded_blocks(tmp_path):
    from winding_field.exports import export_afv
    from winding_field.splines import afv_points
    root=qual_export_fixture(tmp_path/'run')
    pending=root/'fibres/fibres.partial.afv'
    def stop():
        if not pending.exists(): return False
        with sqlite3.connect(pending) as db:
            return db.execute('SELECT count(*) FROM fibers').fetchone()[0]>=1
    assert export_afv(root,stop) is None
    with sqlite3.connect(pending) as db:
        assert db.execute('SELECT count(*) FROM blocks').fetchone()==(2,)
    sha=export_afv(root)
    assert export_afv(root)==sha
    with sqlite3.connect(root/'fibres/fibres.afv') as db:
        assert db.execute('PRAGMA user_version').fetchone()==(1,)
        assert db.execute('SELECT family FROM fibers ORDER BY id').fetchall()==[('H',),('V',)]
        points=afv_points(db,1,300)
        np.testing.assert_allclose(points[0],[21.,47.,67.])
        assert max(row[0] for row in db.execute('SELECT length(points)/24 FROM blocks'))<=257
        metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
        assert metadata['frame']['level']==1
        assert metadata['root']['coordinate_base_shape_zyx']==[100,100,1400]
        annotation=json.loads(db.execute('SELECT annotation FROM fibers WHERE id=1').fetchone()[0])
        assert annotation['hv_classification']['manual_tag']=='H'
        assert annotation['length_mm']==pytest.approx(598*9/1000)


def test_qual_afv_resume_checks_later_blocks(tmp_path):
    from winding_field.exports import export_afv
    root=qual_export_fixture(tmp_path/'run'); pending=root/'fibres/fibres.partial.afv'
    def stop():
        if not pending.exists(): return False
        with sqlite3.connect(pending) as db:
            return db.execute('SELECT count(*) FROM fibers').fetchone()[0]>=1
    assert export_afv(root,stop) is None
    with sqlite3.connect(pending) as db:
        assert db.execute('SELECT count(*) FROM blocks').fetchone()==(2,)
        db.execute('UPDATE blocks SET points=zeroblob(length(points)) WHERE first_segment=256')
    with pytest.raises(ValueError,match='changed|corrupt'):
        export_afv(root)


def test_export_density_converts_world_gradient_to_anisotropic_index_space(tmp_path):
    from winding_field.artifacts import Tiles
    from winding_field.exports import export_lasagna
    from winding_field.io import Volume
    root=export_fixture(tmp_path/'run'); maps=Tiles(root/'maps')
    for key in maps.keys():
        data=maps.load(key); data['density'][:]=.01
        data['normals'][:]=[2**-.5,2**-.5,0]
        maps.write(key,data)
    meta=json.loads((root/'PREPARED.json').read_bytes())
    meta.update(spacing_xyz=[2.,4.,8.],tiles_sha256=maps.identity()); maps.close()
    (root/'PREPARED.json').write_text(json.dumps(meta))
    export_lasagna(root)
    manifest=json.loads((root/'lasagna/surfaces.lasagna.json').read_bytes())
    assert 'grad_mag' in manifest['groups']
    volume=Volume(root/'lasagna/surface_grad_mag.ome.zarr',cache=tmp_path/'cache')
    try: np.testing.assert_allclose(volume.read([0,0,0],[1,1,1]),np.sqrt(10)*10,rtol=1e-6)
    finally: volume.close()
