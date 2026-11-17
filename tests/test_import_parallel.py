from contextlib import closing
import multiprocessing as mp
import os
import json
from pathlib import Path
import shutil
import time

import pytest


def pid(value):
    return value,os.getpid()


def fail(value):
    raise ValueError('worker rejected input')


def die(value):
    os._exit(42)


def environment(value):
    return {name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES')}


def cooperative(path):
    from winding_field.import_parallel import worker_stopped
    Path(path).touch()
    while not worker_stopped():time.sleep(.01)
    return None


def ready(event):
    global _READY
    _READY=event


def out_of_order(value):
    if value==0:assert _READY.wait(10)
    else:_READY.set()
    return value


def resources(monkeypatch, *, cpus=8, known=True, host=8<<30, effective=8<<30, disk=8<<30):
    from winding_field import import_parallel
    monkeypatch.setattr(import_parallel,'cpu_limit',lambda:cpus)
    monkeypatch.setattr(import_parallel,'memory_report',lambda:dict(
        known=known,host_available_bytes=host,effective_headroom_bytes=effective))
    monkeypatch.setattr(import_parallel.shutil,'disk_usage',lambda path:shutil._ntuple_diskusage(disk*2,0,disk))


def test_worker_admission_respects_quotas_and_unknown_memory(tmp_path,monkeypatch):
    from winding_field.import_parallel import limits
    resources(monkeypatch,cpus=2)
    assert limits(0,output=tmp_path,reserve_bytes=0,worker_bytes=256<<20)['workers']==1
    assert limits(8,output=tmp_path,reserve_bytes=0,worker_bytes=256<<20)['workers']==1
    resources(monkeypatch,known=False)
    assert limits(4,output=tmp_path,reserve_bytes=0,worker_bytes=256<<20)['workers']==1


@pytest.mark.parametrize('low',['host','effective','disk'])
def test_unsafe_admission_fails_before_output_creation(tmp_path,monkeypatch,low):
    from winding_field.import_parallel import limits
    resources(monkeypatch,**{low:1<<20})
    output=tmp_path/'new'
    with pytest.raises(OSError):limits(4,output=output,reserve_bytes=1<<20,worker_bytes=256<<20)
    assert not output.exists()


def test_ordered_map_bounds_unconsumed_work_and_restores_environment():
    from winding_field.import_parallel import ordered_map
    submitted=[];old=dict(os.environ)
    def jobs():
        for value in range(18):
            submitted.append(value);yield value
    with closing(ordered_map(pid,jobs(),workers=2,max_inflight=4)) as results:
        for index,(value,process) in enumerate(results):
            assert value==index and process!=os.getpid()
            assert len(submitted)<=index+4
    assert dict(os.environ)==old
    values=list(ordered_map(environment,[0,1],workers=2,max_inflight=4))
    assert all(v==dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='') for v in values)


def test_later_worker_completion_cannot_reorder_publication():
    from winding_field.import_parallel import ordered_map
    event=mp.get_context('spawn').Event()
    assert list(ordered_map(out_of_order,[0,1],workers=2,max_inflight=4,
                            initializer=ready,initargs=(event,)))==[0,1]


@pytest.mark.parametrize('fn',[fail,die])
def test_worker_errors_are_not_silently_replayed_and_leave_no_children(fn):
    from winding_field.import_parallel import ordered_map
    previous={p.pid for p in mp.active_children()}
    with pytest.raises(Exception):
        list(ordered_map(fn,[1,2],workers=2,max_inflight=4))
    assert {p.pid for p in mp.active_children()}<=previous


def test_stop_signals_and_reaps_owned_workers(tmp_path):
    from winding_field.import_parallel import ordered_map
    previous={p.pid for p in mp.active_children()};marker=tmp_path/'started'
    with pytest.raises(InterruptedError):
        list(ordered_map(cooperative,[str(marker)]*4,workers=2,max_inflight=4,stop=marker.exists))
    assert marker.exists() and {p.pid for p in mp.active_children()}<=previous


@pytest.mark.parametrize('workers',[2,4])
def test_parallel_afv_has_identical_ordered_rows_and_clipping(tmp_path,monkeypatch,workers):
    from test_afv_batching import many,inventory
    from winding_field import afv
    config=many(tmp_path/'serial',20)
    shutil.copytree(tmp_path/'serial',tmp_path/'parallel')
    resources(monkeypatch)
    afv.import_fibres(tmp_path/'serial',dict(config,import_workers=1))
    afv.import_fibres(tmp_path/'parallel',dict(config,import_workers=workers))
    assert inventory(tmp_path/'parallel')==inventory(tmp_path/'serial')
    execution=json.loads((tmp_path/'parallel/fibres/STATUS.json').read_bytes())['execution']
    assert execution['workers']==workers
    assert execution['worker_pids'] and os.getpid() not in execution['worker_pids']


def test_parallel_afv_stop_preserves_frontier_and_resume_never_duplicates(tmp_path,monkeypatch):
    import sqlite3
    from test_afv_batching import many,inventory
    from winding_field import afv
    config=dict(many(tmp_path/'run',24),import_workers=2);resources(monkeypatch)
    monkeypatch.setattr(afv,'_BATCH_BLOCKS',2)
    previous={p.pid for p in mp.active_children()}
    def stop():
        path=tmp_path/'run/fibres/fibres.sqlite'
        if not path.exists():return False
        with sqlite3.connect(path) as db:
            return db.execute('SELECT count(*) FROM done').fetchone()[0]>=2
    assert afv.import_fibres(tmp_path/'run',config,stop)['status']=='PAUSED'
    first=inventory(tmp_path/'run');assert len(first[1])==2
    assert {p.pid for p in mp.active_children()}<=previous
    assert afv.import_fibres(tmp_path/'run',config)['status']=='FIBRES_READY'
    final=inventory(tmp_path/'run')
    assert len(final[1])==len(final[0])==24 and final[0][:2]==first[0]


def test_large_worker_result_requests_streaming_replay_instead_of_truncation(tmp_path,monkeypatch):
    import numpy as np
    from test_afv_benchmark import tool
    from winding_field import afv,import_parallel
    tool().generated_input(tmp_path/'run',1,1)
    afv._clip_initialize(str(tmp_path/'run/maps'),1<<20)
    try:
        monkeypatch.setattr(import_parallel,'RESULT_BYTES',1)
        job=(1,'H',0,'{}',np.array([[1.,2.,3.],[20.,2.,3.]]))
        returned,fragments,stats=afv._clip_worker(job)
        assert returned is job and fragments is None
    finally:
        afv._WORKER_SUPPORT.close();afv._WORKER_SUPPORT.maps.close()


def test_worker_maps_cannot_write_or_recreate_missing_indexes(tmp_path):
    import sqlite3
    from test_afv_benchmark import tool
    from winding_field.artifacts import Tiles
    tool().generated_input(tmp_path/'run',1,1)
    with closing(Tiles(tmp_path/'run/maps',readonly=True)) as maps:
        with pytest.raises(sqlite3.OperationalError,match='readonly'):
            maps.db.execute('DELETE FROM tiles')
    (tmp_path/'run/maps/tiles.sqlite').unlink()
    with pytest.raises(sqlite3.OperationalError):Tiles(tmp_path/'run/maps',readonly=True)
    assert not (tmp_path/'run/maps/tiles.sqlite').exists()
