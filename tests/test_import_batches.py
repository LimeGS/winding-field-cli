from contextlib import closing
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ct_fixture import write_ct
from test_afv_batching import many
from test_exports import export_fixture
from winding_field import afv, exports
from winding_field.io import Volume


def commits(monkeypatch,suffix,*,fail=None):
    connect=sqlite3.connect;writes=[]
    class Tracked(sqlite3.Connection):
        def commit(self):
            if self.in_transaction:
                writes.append(1)
                if len(writes)==fail:raise OSError('injected batch commit failure')
            return super().commit()
    def opened(database,*args,**kwargs):
        if str(database).endswith(suffix):kwargs['factory']=Tracked
        return connect(database,*args,**kwargs)
    monkeypatch.setattr(sqlite3,'connect',opened)
    return writes


def test_chunk_batch_reduces_commits_without_accepting_changed_bytes(tmp_path,monkeypatch):
    source=write_ct(tmp_path/'ct',np.ones((4,4,20),np.uint8),chunk=4)
    writes=commits(monkeypatch,'source.sqlite')
    with closing(Volume(source,cache=tmp_path/'cache')) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite')
        with volume.chunk_batch():
            volume.read([0]*3,[4,4,20])
            assert volume._ledger.execute('SELECT count(*) FROM chunks').fetchone()[0]==5
            assert not writes
            (source/'0/0.0.0').write_bytes(bytes(64))
            with pytest.raises(ValueError,match='changed'):volume.read([0]*3,[4]*3)
        assert len(writes)==1


def test_chunk_batch_rolls_back_only_pending_pins(tmp_path):
    source=write_ct(tmp_path/'ct',np.ones((4,4,8),np.uint8),chunk=4)
    with closing(Volume(source,cache=tmp_path/'cache')) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite')
        volume.read([0]*3,[4]*3)
        with pytest.raises(RuntimeError):
            with volume.chunk_batch():
                volume.read([0,0,4],[4,4,8])
                raise RuntimeError('stop')
        assert volume._ledger.execute('SELECT count(*) FROM chunks').fetchone()[0]==1


def test_chunk_batch_flushes_at_128_chunks(tmp_path):
    source=write_ct(tmp_path/'ct',np.ones((4,4,516),np.uint8),chunk=4)
    with closing(Volume(source,cache=tmp_path/'cache')) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite')
        with volume.chunk_batch():
            volume.read([0]*3,[4,4,516])
            with sqlite3.connect(tmp_path/'source.sqlite') as reader:
                assert reader.execute('SELECT count(*) FROM chunks').fetchone()[0]==128
        with sqlite3.connect(tmp_path/'source.sqlite') as reader:
            assert reader.execute('SELECT count(*) FROM chunks').fetchone()[0]==129


def test_chunk_batch_flushes_after_two_seconds(tmp_path,monkeypatch):
    import winding_field.io as io
    source=write_ct(tmp_path/'ct',np.ones((4,4,8),np.uint8),chunk=4)
    with closing(Volume(source,cache=tmp_path/'cache')) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite')
        clock=SimpleNamespace(value=0.)
        monkeypatch.setattr(io,'time',SimpleNamespace(monotonic=lambda:clock.value))
        with volume.chunk_batch():
            volume.read([0]*3,[4]*3)
            clock.value=2.1
            volume.read([0,0,4],[4,4,8])
            with sqlite3.connect(tmp_path/'source.sqlite') as reader:
                assert reader.execute('SELECT count(*) FROM chunks').fetchone()[0]==2


def test_afv_export_batches_curve_records(tmp_path,monkeypatch):
    config=many(tmp_path/'run',200);afv.import_fibres(tmp_path/'run',config)
    writes=commits(monkeypatch,'fibres.partial.afv')
    monkeypatch.setattr(exports,'time',SimpleNamespace(monotonic=lambda:0.),raising=False)
    assert exports.export_afv(tmp_path/'run')
    assert len(writes)<=3
    with sqlite3.connect(tmp_path/'run/fibres/fibres.afv') as db:
        assert db.execute('SELECT count(*) FROM fibers').fetchone()[0]==200
        assert db.execute('SELECT count(*) FROM block_bounds').fetchone()[0]==200


def test_failed_export_batch_never_advances_durable_frontier(tmp_path,monkeypatch):
    config=many(tmp_path/'run',5);afv.import_fibres(tmp_path/'run',config)
    with monkeypatch.context() as patch:
        commits(patch,'fibres.partial.afv',fail=2)
        patch.setattr(exports,'time',SimpleNamespace(monotonic=lambda:0.),raising=False)
        with pytest.raises(OSError,match='commit failure'):exports.export_afv(tmp_path/'run')
    with sqlite3.connect(tmp_path/'run/fibres/fibres.partial.afv') as db:
        assert db.execute('SELECT count(*) FROM fibers').fetchone()[0]==0
        assert not db.execute("SELECT value FROM metadata WHERE key='export_last_id'").fetchone()
    assert exports.export_afv(tmp_path/'run')


def test_lasagna_export_batches_complete_files(tmp_path,monkeypatch):
    root=export_fixture(tmp_path/'run')
    writes=commits(monkeypatch,'exports.sqlite')
    monkeypatch.setattr(exports,'time',SimpleNamespace(monotonic=lambda:0.),raising=False)
    assert exports.export_lasagna(root)
    assert len(writes)<=3
    with sqlite3.connect(root/'exports.sqlite') as db:
        assert db.execute('SELECT count(*) FROM files').fetchone()[0]==21


def test_worker_pin_capture_includes_cache_hits_and_rejects_conflicting_parent_pins(tmp_path):
    source=write_ct(tmp_path/'ct',np.ones((4,4,4),np.uint8),chunk=4)
    with closing(Volume(source,cache=tmp_path/'cache',decoded_cache_bytes=1<<20)) as child:
        child.bind_ledger(tmp_path/'child.sqlite')
        with child.verified_chunks():
            with child.capture_pins() as first:child.read([0]*3,[4]*3)
            with child.capture_pins() as second:child.read([0]*3,[4]*3)
            assert first==second and list(first)==['0/0.0.0']
        with closing(Volume(source,cache=tmp_path/'parent-cache')) as parent:
            parent.bind_ledger(tmp_path/'parent.sqlite')
            parent.merge_pins(first);parent.validate_used()
            with pytest.raises(ValueError,match='changed'):parent.merge_pins({'0/0.0.0':'0'*64})
            for key in ('../escape','0/1.0.0','0/0.0.-1'):
                with pytest.raises(ValueError):parent.merge_pins({key:'0'*64})


def test_pin_capture_is_bounded_and_does_not_weaken_metadata_checks(tmp_path):
    from winding_field.io import PinBudgetExceeded
    import json
    source=write_ct(tmp_path/'ct',np.ones((4,4,4),np.uint8),chunk=4)
    with closing(Volume(source,cache=tmp_path/'cache')) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite')
        with pytest.raises(PinBudgetExceeded):
            with volume.capture_pins(dict(remaining=1)):volume.read([0]*3,[4]*3)
        with pytest.raises(ValueError,match='metadata'):
            with volume.verified_chunks():
                volume.read([0]*3,[4]*3)
                path=source/'0/.zarray';data=json.loads(path.read_bytes());data['order']='F'
                path.write_text(json.dumps(data))


def test_hard_exit_keeps_only_durable_source_pin_batches(tmp_path):
    source=write_ct(tmp_path/'ct',np.ones((4,4,516),np.uint8),chunk=4)
    script=f"""
import os
from winding_field.io import Volume
v=Volume({str(source)!r},cache={str(tmp_path/'cache')!r})
v.bind_ledger({str(tmp_path/'source.sqlite')!r})
with v.chunk_batch():
    v.read([0,0,0],[4,4,516])
    os._exit(23)
"""
    child=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert child.returncode==23,child.stderr
    with sqlite3.connect(tmp_path/'source.sqlite') as db:
        assert db.execute('SELECT count(*) FROM chunks').fetchone()[0]==128
    with closing(Volume(source,cache=tmp_path/'cache')) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite');volume.validate_used()
        with volume.chunk_batch():volume.read([0]*3,[4,4,516])
    with sqlite3.connect(tmp_path/'source.sqlite') as db:
        assert db.execute('SELECT count(*) FROM chunks').fetchone()[0]==129


def test_hard_exit_export_keeps_atomic_curves_bounds_and_cursor(tmp_path):
    config=many(tmp_path/'run',1002);afv.import_fibres(tmp_path/'run',config)
    script=f"""
import os,sqlite3
from types import SimpleNamespace
from winding_field import exports
exports.time=SimpleNamespace(monotonic=lambda:0.)
connect=sqlite3.connect
class Crash(sqlite3.Connection):
    def execute(self,sql,args=()):
        if sql.startswith('INSERT INTO fibers') and args[0]==1002:os._exit(24)
        return super().execute(sql,args)
def opened(path,*args,**kwargs):
    if str(path).endswith('fibres.partial.afv'):kwargs['factory']=Crash
    return connect(path,*args,**kwargs)
sqlite3.connect=opened
exports.export_afv({str(tmp_path/'run')!r})
"""
    child=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert child.returncode==24,child.stderr
    with sqlite3.connect(tmp_path/'run/fibres/fibres.partial.afv') as db:
        assert db.execute('SELECT count(*) FROM fibers').fetchone()[0]==1000
        assert db.execute('SELECT count(*) FROM block_bounds').fetchone()[0]==1000
        assert db.execute("SELECT value FROM metadata WHERE key='export_last_id'").fetchone()[0]=='1000'
    assert exports.export_afv(tmp_path/'run')
    with sqlite3.connect(tmp_path/'run/fibres/fibres.afv') as db:
        assert db.execute('SELECT count(*) FROM fibers').fetchone()[0]==1002
