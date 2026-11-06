import numpy as np
import pytest

from winding_field.bounded_journal import CurveJournal


def curve(x):
    points=np.array([[x,2,3],[x+1,2,3]],dtype='f4')
    return points,np.array([[1,0,0],[1,0,0]],dtype='f4')


def test_seed_commit_is_atomic_and_reopen_preserves_family_and_commit_order(tmp_path):
    path=tmp_path/'curves.sqlite'
    with CurveJournal(path,{'source':'483A'},quota_bytes=1<<20) as journal:
        journal.commit(1,0,[curve(5),curve(2)])
        with pytest.raises(ValueError,match='finite'):
            journal.commit(1,1,[curve(7),(np.full((2,3),np.nan,'f4'),curve(1)[1])])
        assert journal.next_seed(1)==1
        journal.commit(2,0,[curve(9)])
    with CurveJournal(path,{'source':'483A'},quota_bytes=1<<20) as journal:
        assert journal.next_seed(1)==1 and journal.next_seed(2)==1
        curves=list(journal.curves(1,[0,0,0],[10,10,10]))
        assert [p[0,0] for p,t in curves]==[5,2]
        assert list(journal.curves(1,[0,0,20],[10,10,30]))==[]
        journal.commit(1,1,[])
        assert journal.next_seed(1)==2
        with pytest.raises(ValueError,match='sequence'):journal.commit(1,1,[])
    with pytest.raises(ValueError,match='identity'):
        CurveJournal(path,{'source':'another'},quota_bytes=1<<20)


def test_quota_failure_does_not_advance_progress(tmp_path):
    path=tmp_path/'curves.sqlite'
    with CurveJournal(path,{},quota_bytes=1<<20) as journal:
        journal.quota_bytes=path.stat().st_size
        with pytest.raises(OSError,match='quota'):
            journal.commit(1,0,[curve(5)])
        with pytest.raises(OSError,match='quota'):
            journal.commit(1,0,[])
        assert journal.next_seed(1)==0
        assert list(journal.curves(1,[0,0,0],[10,10,10]))==[]


def test_killed_seed_transaction_is_replayed_not_partially_visible(tmp_path):
    import os
    import subprocess
    import sys
    path=tmp_path/'curves.sqlite'
    with CurveJournal(path,{},quota_bytes=1<<20) as journal:
        journal.commit(1,0,[curve(1)])
    code='''
import os,sys,numpy as np
from winding_field.bounded_journal import CurveJournal
with CurveJournal(sys.argv[1],{},quota_bytes=1<<20) as journal:
    def interrupted():
        yield np.ones((2,3),'f4'),np.ones((2,3),'f4')
        os._exit(77)
    journal.commit(1,1,interrupted())
'''
    result=subprocess.run([sys.executable,'-c',code,str(path)],env=os.environ.copy())
    assert result.returncode==77
    with CurveJournal(path,{},quota_bytes=1<<20) as journal:
        assert journal.next_seed(1)==1
        assert len(list(journal.curves(1,[0,0,0],[10,10,10])))==1


def test_trace_resume_rebuilds_coverage_from_committed_global_curves(tmp_path):
    from winding_field.bounded_qual import PagedField,trace_seeds
    shape=(16,16,64)
    fields={name:PagedField(shape,dtype,
        lambda lo,hi,value=value,dtype=dtype:np.full(tuple(hi-lo),value,dtype),chunk=4,cache_bytes=128)
        for name,value,dtype in [('centrality',10000,np.uint16),('presence',255,np.uint8),
                                ('nx',255,np.uint8),('ny',128,np.uint8)]}
    source=[(0,(8,8,10),2),(1,(8,8,50),2)]
    with CurveJournal(tmp_path/'curves.sqlite',{},quota_bytes=1<<20) as journal:
        result=trace_seeds(source,1,fields,journal,chunk=4,cache_bytes=128,
                           stop=lambda:journal.next_seed(1)==1)
        assert result=='PAUSED'
        assert len(list(journal.curves(1,[0,0,0],shape)))==1
    with CurveJournal(tmp_path/'curves.sqlite',{},quota_bytes=1<<20) as journal:
        assert trace_seeds(source,1,fields,journal,chunk=4,cache_bytes=128)=='COMPLETE'
        assert journal.next_seed(1)==2
        assert len(list(journal.curves(1,[0,0,0],shape)))==1


def test_trace_checks_stop_before_advancing_seed_iterator(tmp_path):
    from winding_field.bounded_qual import trace_seeds
    from types import SimpleNamespace
    def source():
        raise AssertionError('stopped work must not enumerate a region')
        yield
    with CurveJournal(tmp_path/'curves.sqlite',{},quota_bytes=1<<20) as journal:
        assert trace_seeds(source(),1,{'presence':SimpleNamespace(shape=(8,8,8))},journal,stop=lambda:True)=='PAUSED'


@pytest.mark.parametrize('change',[
    'DELETE FROM bounds','UPDATE bounds SET x0=50,x1=51',
    'UPDATE curves SET family=2','UPDATE curves SET seed=7','DELETE FROM progress',
    'DELETE FROM curves','DELETE FROM commits WHERE seed=1'])
def test_reopen_rejects_changed_replay_inventory_and_empty_commits(tmp_path,change):
    import sqlite3
    path=tmp_path/'curves.sqlite'
    with CurveJournal(path,{},quota_bytes=1<<20) as journal:
        journal.commit(1,0,[curve(5)])
        journal.commit(1,1,[])
    with sqlite3.connect(path) as db:db.execute(change)
    with pytest.raises(ValueError,match='inventory'):
        CurveJournal(path,{},quota_bytes=1<<20)
