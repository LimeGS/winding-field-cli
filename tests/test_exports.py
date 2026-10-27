import json
import sqlite3

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
    assert len(writes)==18
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
