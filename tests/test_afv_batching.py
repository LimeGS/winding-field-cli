"""Atomic AFV block batches, using real SQLite and official AFV/map fixtures."""
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_afv_import import fixture
from test_splines import upstream_curves
from winding_field import afv


def many(root, n, *, excluded=False):
    _, config=fixture(root)
    path=Path(config['fibres']);path.unlink()
    x=30. if excluded else 1.
    points=np.array([[x,2.,3.],[x+1,2.,3.]],dtype='<f8')
    upstream_curves(path,(8,8,64),((1,points) for _ in range(n)))
    with sqlite3.connect(config['fibres']) as db:
        for fid in range(1,n+1):
            note=json.dumps(dict(hv_classification=dict(manual_tag='H'),original_annotation=fid))
            db.execute('UPDATE fibers SET family=?,annotation=? WHERE id=?',('unknown',note,fid))
    return config


def inventory(root):
    with sqlite3.connect(root/'fibres/fibres.sqlite') as db:
        curves=list(db.execute('SELECT id,points,family,source_id,source_block,annotation FROM curves ORDER BY id'))
        done=list(db.execute('SELECT * FROM done ORDER BY fid,block'))
        bounds=list(db.execute('SELECT * FROM curve_bounds ORDER BY id'))
        assert db.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
    assert len(curves)==len(bounds)
    assert {r[0] for r in curves}=={r[0] for r in bounds}
    assert {(r[3],r[4]) for r in curves}<={(r[0],r[1]) for r in done}
    return curves,done,bounds


def tracked(monkeypatch, *, fail_commit=None):
    connect=sqlite3.connect;commits=[];attempts=[]
    class Connection(sqlite3.Connection):
        def commit(self):
            if not self.in_transaction:return super().commit()
            attempts.append(1)
            if len(attempts)==fail_commit:raise OSError('injected commit failure')
            super().commit()
            commits.append(self.execute('SELECT count(*) FROM done').fetchone()[0])
    def opened(database,*args,**kwargs):
        if str(database).endswith('/fibres.sqlite'):
            kwargs['factory']=Connection
        return connect(database,*args,**kwargs)
    monkeypatch.setattr(afv.sqlite3,'connect',opened)
    return commits


@pytest.mark.parametrize('excluded',[False,True])
def test_complete_and_excluded_blocks_share_bounded_commits_and_final_partial(tmp_path,monkeypatch,excluded):
    config=many(tmp_path,7,excluded=excluded)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',3,raising=False)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',100.,raising=False)
    commits=tracked(monkeypatch)
    result=afv.import_fibres(tmp_path,config)
    assert commits==[3,6,7]
    curves,done,bounds=inventory(tmp_path)
    assert len(done)==7 and len(curves)==len(bounds)==(0 if excluded else 7)
    assert result['input_count']==7 and result['count']==len(curves)
    for _,blob,family,fid,_,note in curves:
        assert family=='H' and json.loads(note)['original_annotation']==fid
        np.testing.assert_array_equal(np.load(io.BytesIO(blob)),[[109.,218.,327.],[118.,218.,327.]])


def test_time_limit_flushes_at_complete_block_boundaries(tmp_path,monkeypatch):
    config=many(tmp_path,3)
    clock=SimpleNamespace(value=0.)
    original=afv._pieces
    def pieces(*args,**kwargs):
        yield from original(*args,**kwargs)
        clock.value+=2.1
    monkeypatch.setattr(afv,'_pieces',pieces)
    monkeypatch.setattr(afv,'time',SimpleNamespace(monotonic=lambda:clock.value,time=lambda:1700000000.+clock.value),raising=False)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',1000,raising=False)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',2.,raising=False)
    commits=tracked(monkeypatch)
    afv.import_fibres(tmp_path,config)
    assert commits==[1,2,3]


@pytest.mark.parametrize('failure',['stop_inside','stop_between','exception'])
def test_pending_batch_rolls_back_and_resume_never_duplicates(tmp_path,monkeypatch,failure):
    config=many(tmp_path,5)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',2,raising=False)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',100.,raising=False)
    original=afv._pieces;calls=[0]
    def pieces(*args,**kwargs):
        if failure=='stop_between':
            yield from original(*args,**kwargs)
            calls[0]+=1
            return
        calls[0]+=1
        yield from original(*args,**kwargs)
        if calls[0]==3 and failure!='stop_between':
            if failure=='exception':raise RuntimeError('injected clipping failure')
            raise InterruptedError('injected stop inside block')
    monkeypatch.setattr(afv,'_pieces',pieces)
    if failure=='exception':
        with pytest.raises(RuntimeError,match='clipping'):afv.import_fibres(tmp_path,config)
    else:
        assert afv.import_fibres(tmp_path,config,stop=lambda:calls[0]>=3 if failure=='stop_between' else False)==dict(status='PAUSED')
    first=inventory(tmp_path)
    assert len(first[1])==2
    assert not (tmp_path/'FIBRES.json').exists()
    monkeypatch.setattr(afv,'_pieces',original)
    assert afv.import_fibres(tmp_path,config)['status']=='FIBRES_READY'
    final=inventory(tmp_path)
    assert final[0][:2]==first[0] and len(final[1])==5 and len(final[0])==5
    assert len({(c[3],c[4]) for c in final[0]})==5


def test_commit_failure_leaves_only_prior_batch_durable(tmp_path,monkeypatch):
    config=many(tmp_path,5)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',2,raising=False)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',100.,raising=False)
    with monkeypatch.context() as patch:
        commits=tracked(patch,fail_commit=2)
        with pytest.raises(OSError,match='commit failure'):afv.import_fibres(tmp_path,config)
    assert commits==[2] and len(inventory(tmp_path)[1])==2
    assert afv.import_fibres(tmp_path,config)['status']=='FIBRES_READY'
    assert len(inventory(tmp_path)[1])==5


def test_hard_process_exit_replays_only_uncommitted_writes(tmp_path):
    config=many(tmp_path,5)
    script=r'''
import json,os,sqlite3
from pathlib import Path
from winding_field import afv
afv._BATCH_BLOCKS=2;afv._BATCH_SECONDS=100.
connect=sqlite3.connect
class Crash(sqlite3.Connection):
    def execute(self,sql,params=()):
        result=super().execute(sql,params)
        if sql.startswith('INSERT INTO curves(') and params[3]==3:os._exit(23)
        return result
def opened(path,*a,**kw):
    if str(path).endswith('/fibres.sqlite'):kw['factory']=Crash
    return connect(path,*a,**kw)
afv.sqlite3.connect=opened
afv.import_fibres(Path(ROOT),CONFIG)
'''
    script='ROOT='+repr(str(tmp_path))+'\nCONFIG='+repr(config)+'\n'+script
    child=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert child.returncode==23,child.stdout+child.stderr
    assert len(inventory(tmp_path)[1])==2
    assert not (tmp_path/'FIBRES.json').exists()
    assert afv.import_fibres(tmp_path,config)['status']=='FIBRES_READY'
    assert len(inventory(tmp_path)[0])==5


def test_progress_distinguishes_processed_pending_and_durable_blocks(tmp_path,monkeypatch):
    config=many(tmp_path,3)
    clock=SimpleNamespace(value=0.)
    monkeypatch.setattr(afv,'time',SimpleNamespace(monotonic=lambda:clock.value,time=lambda:1700000000.+clock.value))
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',1000)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',100.)
    monkeypatch.setattr(afv,'_PROGRESS_SECONDS',2.,raising=False)
    events=[];atomic=afv.atomic_json
    def publish(path,value):
        atomic(path,value)
        if Path(path).name=='STATUS.json':
            with sqlite3.connect(tmp_path/'fibres/fibres.sqlite') as db:
                assert value['blocks_committed']==db.execute('SELECT count(*) FROM done').fetchone()[0]
                assert db.execute('PRAGMA synchronous').fetchone()==(2,)
                assert db.execute('PRAGMA journal_mode').fetchone()==('delete',)
            events.append(dict(value))
    monkeypatch.setattr(afv,'atomic_json',publish)
    original=afv._pieces
    def pieces(*args,**kwargs):
        yield from original(*args,**kwargs)
        clock.value+=2.1
    monkeypatch.setattr(afv,'_pieces',pieces)
    afv.import_fibres(tmp_path,config)
    assert any(e['blocks_processed']==1 and e['blocks_committed']==0 and e['blocks_pending']==1 for e in events)
    last=events[-1]
    assert last['status']=='FIBRES_READY'
    assert last['blocks_processed']==last['blocks_committed']==last['blocks_total']==3
    assert last['blocks_pending']==0 and last['commits_this_invocation']==1
    assert last['last_batch_committed_utc']==1700000000.+clock.value
    assert last['blocks_per_second']==pytest.approx(3/6.3)


def test_progress_is_throttled_independently_of_commit_count(tmp_path,monkeypatch):
    config=many(tmp_path,12)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',1)
    monkeypatch.setattr(afv,'time',SimpleNamespace(monotonic=lambda:1.,time=lambda:1700000000.))
    writes=[];atomic=afv.atomic_json
    def publish(path,value):
        if Path(path).name=='STATUS.json':writes.append(value['status'])
        atomic(path,value)
    monkeypatch.setattr(afv,'atomic_json',publish)
    afv.import_fibres(tmp_path,config)
    assert writes==['IMPORTING','VALIDATING','FIBRES_READY']


def test_failed_or_stale_progress_is_not_a_recovery_authority(tmp_path,monkeypatch):
    config=many(tmp_path,5)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',2)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',100.)
    original=afv._pieces;calls=[0]
    def pieces(*args,**kwargs):
        calls[0]+=1
        yield from original(*args,**kwargs)
        if calls[0]==3:raise RuntimeError('original geometry failure')
    monkeypatch.setattr(afv,'_pieces',pieces)
    with pytest.raises(RuntimeError):afv.import_fibres(tmp_path,config)
    progress=json.loads((tmp_path/'fibres/STATUS.json').read_bytes())
    assert progress['status']=='FAILED' and progress['blocks_committed']==2 and progress['blocks_pending']==0
    (tmp_path/'fibres/STATUS.json').write_text('{"blocks_committed":999999999}')
    monkeypatch.setattr(afv,'_pieces',original)
    afv.import_fibres(tmp_path,config)
    assert len(inventory(tmp_path)[1])==5
    assert json.loads((tmp_path/'fibres/STATUS.json').read_bytes())['blocks_committed']==5


def test_progress_write_failure_preserves_last_committed_batch(tmp_path,monkeypatch):
    config=many(tmp_path,3)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',2)
    monkeypatch.setattr(afv,'_BATCH_SECONDS',100.)
    monkeypatch.setattr(afv,'_PROGRESS_SECONDS',0.,raising=False)
    atomic=afv.atomic_json
    def publish(path,value):
        if Path(path).name=='STATUS.json' and value['blocks_committed']>=2:
            raise OSError('injected progress write failure')
        atomic(path,value)
    monkeypatch.setattr(afv,'atomic_json',publish)
    with pytest.raises(OSError,match='progress write failure'):afv.import_fibres(tmp_path,config)
    assert len(inventory(tmp_path)[1])==2
    monkeypatch.setattr(afv,'atomic_json',atomic)
    assert afv.import_fibres(tmp_path,config)['status']=='FIBRES_READY'


def test_cli_status_exposes_advisory_afv_progress_without_changing_run_fields(tmp_path,capsys):
    from winding_field.cli import main
    state=dict(status='RUNNING',config=dict(surface_output='atlas'))
    (tmp_path/'RUN.json').write_text(json.dumps(state))
    (tmp_path/'fibres').mkdir()
    progress=dict(schema='afv-import-progress-v1',status='IMPORTING',blocks_processed=100,
                  blocks_committed=0,blocks_total=200)
    (tmp_path/'fibres/STATUS.json').write_text(json.dumps(progress))
    assert main(['status',str(tmp_path)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['afv_progress']==progress
    assert {k:result[k] for k in state}==state


def test_excluded_stream_memory_does_not_scale_with_dataset_and_default_batch_is_bounded(tmp_path,monkeypatch):
    import tracemalloc
    monkeypatch.setattr(afv,'_BATCH_SECONDS',1000.)
    peaks=[]
    for n in (250,2500):
        root=tmp_path/str(n)
        config=many(root,n,excluded=True)
        with monkeypatch.context() as patch:
            commits=tracked(patch)
            tracemalloc.start()
            try:
                assert afv.import_fibres(root,config)['count']==0
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:tracemalloc.stop()
        assert commits==([250] if n==250 else [1000,2000,2500])
        assert len(inventory(root)[1])==n
    assert peaks[1]<peaks[0]+2*1024**2


@pytest.mark.parametrize('text',['{','[]','{"schema":"other"}',
    '{"schema":"afv-import-progress-v1","elapsed_s":NaN}'])
def test_advisory_progress_damage_does_not_break_cli_status(tmp_path,capsys,text):
    from winding_field.cli import main
    state=dict(status='RUNNING',config=dict(surface_output='atlas'))
    (tmp_path/'RUN.json').write_text(json.dumps(state))
    (tmp_path/'fibres').mkdir();(tmp_path/'fibres/STATUS.json').write_text(text)
    (tmp_path/'surfaces').mkdir();(tmp_path/'surfaces/STATUS.json').write_text('{"status":"RUNNING"}')
    assert main(['status',str(tmp_path)])==0
    result=json.loads(capsys.readouterr().out)
    assert result==dict(state,native_progress=dict(status='RUNNING'))
