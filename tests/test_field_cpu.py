import json
from contextlib import closing
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from winding_field import field
from test_field import prepared_run


def reversed_prepare(spec,lo,hi,build_rows=True):
    from winding_field.field_cpu import prepare_static_block
    marker=Path(spec['run'])/'second-worker-finished'
    if lo[2]==0:
        deadline=time.monotonic()+10
        while not marker.exists():
            if time.monotonic()>deadline:raise TimeoutError('second worker never finished')
            time.sleep(.01)
    block=prepare_static_block(spec,lo,hi,build_rows)
    if lo[2]>0:marker.touch()
    block.worker_info['finished_at']=time.monotonic()
    return block


def crashed_prepare(*args):
    os._exit(31)


def blocked_prepare(spec,*args):
    (Path(spec['run'])/'worker-waiting').touch()
    time.sleep(30)
    raise AssertionError('blocked test worker was not cancelled')


def pool_spec(run):
    from winding_field import field_cpu
    meta=json.loads((run/'PREPARED.json').read_bytes())
    with closing(field_cpu.Tiles(run/'maps',readonly=True)) as maps:
        cal=field.calibration(maps,meta)
    geometry=field.field_geometry(meta)
    geometry.update(voxel_um=1.,sheet_step=1.)
    return dict(run=str(run.resolve()),meta=meta,fibres=json.loads((run/'FIBRES.json').read_bytes()),
                geometry=geometry,calibration=cal,weight=1.,boxes=[((0,0,0),(6,6,10)),((0,0,6),(6,6,18))])


@pytest.fixture(autouse=True)
def single_parent_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('options',[
    dict(field_cpu_workers=True,field_cpu_buffer_bytes=1<<20),
    dict(field_cpu_workers=-1,field_cpu_buffer_bytes=1<<20),
    dict(field_cpu_workers=1,field_cpu_buffer_bytes=0),
    dict(field_cpu_workers=0,field_cpu_buffer_bytes=1<<20),
    dict(field_cpu_workers=1,field_cpu_buffer_bytes=True),
    dict(field_cpu_workers=1,field_cpu_buffer_bytes=1<<20,bounded_preparation=True),
])
def test_invalid_cpu_pool_options_reject_before_io(tmp_path,options):
    with pytest.raises(ValueError,match='CPU|cpu'):
        field.solve(tmp_path/'absent',options)
    assert not (tmp_path/'absent').exists()


def test_parallel_preparation_preserves_serial_field_and_commit_order(tmp_path,monkeypatch):
    from winding_field import field_cpu
    from field_store import BlockStore
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    monkeypatch.setattr(field_cpu,'prepare_static_block',reversed_prepare)
    validated=[];validate=field_cpu.validate_block
    def record(*args,**kwargs):
        result=validate(*args,**kwargs);validated.append(result);return result
    monkeypatch.setattr(field_cpu,'validate_block',record)
    original=BlockStore.commit;commits=[]
    def commit(store,key,*args,**kwargs):
        commits.append(tuple(key))
        return original(store,key,*args,**kwargs)
    monkeypatch.setattr(BlockStore,'commit',commit)
    results=[];orders=[];digests=[]
    for workers in (0,2):
        run=prepared_run(tmp_path/str(workers),shape=(6,6,32),variable=True)
        config=dict(sweeps=3,field_chunk=8,constraint_cache_bytes=1<<20)
        if workers:config.update(field_cpu_workers=workers,field_cpu_buffer_bytes=16<<20)
        commits.clear();result=field.solve(run,config);orders.append(list(commits))
        with field.open_field(run) as source:results.append(source.read([0]*3,[6,6,32]))
        cp=json.loads((run/'field/CHECKPOINT.json').read_bytes())
        digests.append([v['sha256'] for _,v in sorted(cp['chunks'].items())])
        if workers:
            stats=result['field_cpu_stats']
            assert stats['completed']==12
            assert stats['peak_buffer_bytes']<=16<<20
            assert stats['workers']==2
            assert stats['constraint_builds']==4
            assert all(not info['cuda_initialized'] and info['threads']==1
                       for info in stats['worker_info'].values())
    assert orders[0]==orders[1] and digests[0]==digests[1]
    assert validated[0].worker_info['finished_at']>validated[1].worker_info['finished_at']
    for a,b in zip(results[0],results[1]):np.testing.assert_array_equal(a,b)


def test_cpu_pool_stop_resume_preserves_field(tmp_path,monkeypatch):
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    config=dict(sweeps=2,field_chunk=8,field_cpu_workers=2,field_cpu_buffer_bytes=16<<20)
    def stop():
        cp=run/'field/CHECKPOINT.json'
        return cp.exists() and json.loads(cp.read_bytes())['progress'].get('next_block')==1
    assert field.solve(run,config,stop)['status']=='PAUSED'
    assert field.solve(run,config)['status']=='FIELD_READY'
    ref=prepared_run(tmp_path/'ref',shape=(6,6,32),variable=True)
    field.solve(ref,dict(sweeps=2,field_chunk=8))
    with field.open_field(run) as a,field.open_field(ref) as b:
        np.testing.assert_array_equal(a.read([0]*3,[6,6,32])[0],b.read([0]*3,[6,6,32])[0])


@pytest.mark.parametrize('stop_kind',['field-stop','callback'])
def test_stop_during_worker_wait_pauses_without_checkpoint_changes(tmp_path,monkeypatch,stop_kind):
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    monkeypatch.setattr(field_cpu,'prepare_static_block',blocked_prepare)
    monkeypatch.setattr(field_cpu.CPUFieldInputs,'timeout_seconds',1.5)
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    before={child.pid for child in mp.active_children()};saved={}
    checkpoint=run/'field/CHECKPOINT.json'
    def stop():
        if not (run/'worker-waiting').exists():return False
        if not saved:
            saved['checkpoint']=checkpoint.read_bytes()
            saved['chunks']={str(p.relative_to(run/'field')):p.read_bytes() for p in (run/'field').rglob('*.npz')}
        if stop_kind=='field-stop':
            (run/'field/STOP').touch()
            return False
        return True
    started=time.monotonic()
    result=field.solve(run,dict(sweeps=2,field_chunk=8,field_cpu_workers=1,
                               field_cpu_buffer_bytes=16<<20),stop=stop)
    assert result['status']=='PAUSED' and time.monotonic()-started<8
    assert saved and checkpoint.read_bytes()==saved['checkpoint']
    assert saved['chunks']
    assert {str(p.relative_to(run/'field')):p.read_bytes() for p in (run/'field').rglob('*.npz')}==saved['chunks']
    status=json.loads((run/'field/STATUS.json').read_bytes())
    assert status['status']=='CHECKPOINTED_STOP' and 'error' not in status
    assert {child.pid for child in mp.active_children()}<=before
    assert not (run/'FIELD.json').exists()
    monkeypatch.undo()
    (run/'field/STOP').unlink(missing_ok=True)
    assert field.solve(run,dict(sweeps=2,field_chunk=8))['status']=='FIELD_READY'


def test_pool_admission_reserves_parent_threads_and_memory(monkeypatch):
    from field_runtime import ResourceUnavailable
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'cpu_limit',lambda:4)
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    assert field_cpu.admit_workers(8,16<<20,0,parent_threads=3)==1
    with pytest.raises(ResourceUnavailable):field_cpu.admit_workers(2,16<<20,0,parent_threads=4)
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:1<<20)
    with pytest.raises(ResourceUnavailable):field_cpu.admit_workers(2,16<<20,0,parent_threads=1)


def test_spline_pair_budget_rejects_before_large_interpolation(monkeypatch):
    from field_runtime import ResourceUnavailable
    calls=[];original=np.linspace
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(np,'linspace',counted)
    with pytest.raises(ResourceUnavailable):
        field.spline_pairs(np.array([[0.,0.,0.],[1.,0.,0.]]),
                           np.zeros(3),np.ones(3),1e-8,max_pairs=10)
    assert not calls


@pytest.mark.parametrize('failure',['bounds','identity','dtype','partial','bytes','rows'])
def test_malformed_payload_rejected_before_consumption(tmp_path,monkeypatch,failure):
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    pool=field_cpu.CPUFieldInputs(pool_spec(run),workers=1,max_bytes=16<<20)
    try:
        lo,hi=pool.boxes[0];pool.load(lo,hi);block=pool.active
        if failure=='bounds':block.lo=(0,0,1)
        elif failure=='identity':block.identity='wrong'
        elif failure=='dtype':
            block.observations=(block.observations[0].astype(np.float64),*block.observations[1:])
            block.retained_bytes=field_cpu.retained_bytes(block.observations,block.rows)
        elif failure=='partial':block.observations=block.observations[:2]
        elif failure=='bytes':block.retained_bytes+=1
        elif failure=='rows':
            block.rows.diagonal=block.rows.diagonal.copy();block.rows.diagonal[0]+=1
        with pytest.raises(ValueError,match='CPU'):
            field_cpu.validate_block(block,pool.spec,lo,hi,build_rows=True)
    finally:pool.close()


@pytest.mark.parametrize('failure',['crash','timeout','stop'])
def test_worker_failure_or_stop_cleans_only_owned_children(tmp_path,monkeypatch,failure):
    from concurrent.futures import TimeoutError
    from concurrent.futures.process import BrokenProcessPool
    from winding_field import field_cpu
    import subprocess
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    before={child.pid for child in mp.active_children()}
    unrelated=subprocess.Popen([os.sys.executable,'-c','import time;time.sleep(30)'])
    pool=field_cpu.CPUFieldInputs(pool_spec(run),workers=2,max_bytes=16<<20,stop=lambda:failure=='stop')
    if failure=='crash':monkeypatch.setattr(field_cpu,'prepare_static_block',crashed_prepare)
    if failure=='timeout':pool.timeout_seconds=.001
    try:
        expected={'crash':BrokenProcessPool,'timeout':TimeoutError,'stop':InterruptedError}[failure]
        with pytest.raises(expected):pool.load(*pool.boxes[0])
    finally:
        pool.close();pool.close()
        try:
            assert unrelated.poll() is None
            assert {child.pid for child in mp.active_children()}<=before
        finally:unrelated.terminate();unrelated.wait()
    assert not (run/'FIELD.json').exists()


def test_small_buffer_rejects_without_spawning(tmp_path,monkeypatch):
    from field_runtime import ResourceUnavailable
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32))
    before={child.pid for child in mp.active_children()}
    with pytest.raises(ResourceUnavailable):
        field_cpu.CPUFieldInputs(pool_spec(run),workers=1,max_bytes=4096)
    assert {child.pid for child in mp.active_children()}==before


def test_buffer_admission_checks_observation_shape_before_spawn(tmp_path,monkeypatch):
    from field_runtime import ResourceUnavailable
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32))
    spec=pool_spec(run);spec['boxes']=[((0,0,0),(256,256,256))]
    with pytest.raises(ResourceUnavailable,match='CPU'):
        field_cpu.CPUFieldInputs(spec,workers=1,max_bytes=16<<20)


@pytest.mark.parametrize('cached',[False,True])
def test_pool_empty_constraints_and_parent_cache(tmp_path,monkeypatch,cached):
    from winding_field import field_cpu
    from winding_field.util import file_sha,atomic_json
    import sqlite3
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32))
    with sqlite3.connect(run/'fibres/fibres.sqlite') as db:db.execute('DELETE FROM curves')
    atomic_json(run/'FIBRES.json',dict(sha256=file_sha(run/'fibres/fibres.sqlite')))
    result=field.solve(run,dict(sweeps=2,field_chunk=8,field_cpu_workers=2,
                               field_cpu_buffer_bytes=16<<20,constraint_cache_bytes=(1<<20) if cached else 0))
    assert result['status']=='FIELD_READY' and result['fibre_constraint_rows']==0
    assert result['field_cpu_stats']['constraint_builds']==(4 if cached else 8)


def test_automatic_pool_rejects_before_io(tmp_path):
    with pytest.raises(ValueError,match='CPU'):
        field.solve(tmp_path/'absent',dict(field_cpu_workers=1,field_cpu_buffer_bytes=16<<20),
                    convergence=object())
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('mode',['legacy','coarse','qual'])
def test_worker_provider_arrays_and_constraints_are_exact(tmp_path,monkeypatch,mode):
    from winding_field import field_cpu
    from winding_field.util import atomic_json,file_sha
    import io
    import sqlite3
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    spec=pool_spec(run)
    if mode=='qual':
        spec['meta']['prediction']='qual-v2';atomic_json(run/'PREPARED.json',spec['meta'])
        with sqlite3.connect(run/'fibres/fibres.sqlite') as db:
            db.execute('CREATE VIRTUAL TABLE curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z)')
            for index,raw in db.execute('SELECT id,points FROM curves').fetchall():
                points=np.load(io.BytesIO(raw),allow_pickle=False)
                box=tuple(v for pair in zip(points.min(0),points.max(0)) for v in pair)
                db.execute('INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)',(index,*box))
        spec['fibres']['sha256']=file_sha(run/'fibres/fibres.sqlite')
        atomic_json(run/'FIBRES.json',spec['fibres'])
    if mode=='coarse':
        spec['geometry']=field.field_geometry(spec['meta'],4.)
        spec['geometry'].update(voxel_um=1.,sheet_step=1.)
        spec['boxes']=[((0,0,0),tuple(spec['geometry']['shape_zyx']))]
    pool=field_cpu.CPUFieldInputs(spec,workers=1,max_bytes=16<<20)
    try:
        for lo,hi in pool.boxes:
            actual=pool.load(lo,hi);rows=pool.constraints(lo,hi,None)
            with closing(field_cpu.Tiles(run/'maps',readonly=True)) as maps,closing(
                    sqlite3.connect((run/'fibres/fibres.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
                expected=field.sample_prior(maps,spec['meta'],spec['geometry'],spec['calibration'],
                                            np.array(lo),np.array(hi))
                expected_rows=field._constraint_rows(db,spec['meta'],spec['fibres'],spec['geometry'],maps,
                                                      np.array(lo),np.array(hi),1.)
            for a,b in zip(actual[:3],expected):
                assert a.dtype==b.dtype and a.tobytes()==b.tobytes()
            assert (rows is None)==(expected_rows is None)
            if rows is not None:
                assert rows.digest==expected_rows.digest
                for a,b in zip(rows._structural_arrays(),expected_rows._structural_arrays()):
                    np.testing.assert_array_equal(a,b)
    finally:pool.close()


@pytest.mark.parametrize('kind',['oversized','nonfinite'])
def test_worker_rejects_invalid_or_oversized_curve_before_publication(tmp_path,monkeypatch,kind):
    from field_runtime import ResourceUnavailable
    from winding_field import field_cpu
    from winding_field.util import atomic_json,file_sha
    import io
    import sqlite3
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    run=prepared_run(tmp_path/'run',shape=(6,6,32))
    points=np.zeros((60000 if kind=='oversized' else 4,3),np.float64)
    if kind=='nonfinite':points[0,0]=np.nan
    raw=io.BytesIO();np.save(raw,points,allow_pickle=False)
    with sqlite3.connect(run/'fibres/fibres.sqlite') as db:
        db.execute('UPDATE curves SET points=?',(raw.getvalue(),))
    atomic_json(run/'FIBRES.json',dict(sha256=file_sha(run/'fibres/fibres.sqlite')))
    pool=field_cpu.CPUFieldInputs(pool_spec(run),workers=1,max_bytes=16<<20)
    try:
        with pytest.raises(ResourceUnavailable if kind=='oversized' else ValueError):
            pool.load(*pool.boxes[0])
    finally:pool.close()
    assert not (run/'FIELD.json').exists()


def test_cli_and_config_pool_flag_parity(tmp_path,monkeypatch):
    from winding_field.cli import main
    from winding_field import workflow
    from winding_field.config import validate_config
    from winding_field.util import atomic_json
    received=[]
    monkeypatch.setattr(workflow,'plan',lambda config:received.append(config.copy()) or {})
    flags=['plan','--source',str(tmp_path/'ct'),'--output',str(tmp_path/'out'),'--device','cpu']
    assert main([*flags,'--field-cpu-workers','2','--field-cpu-buffer-bytes',str(16<<20)])==0
    settings=tmp_path/'config.json'
    atomic_json(settings,dict(field_cpu_workers=2,field_cpu_buffer_bytes=16<<20))
    assert main([*flags,'--cli-config',str(settings)])==0
    assert received[0]==received[1]
    with pytest.raises(ValueError,match='CPU'):
        validate_config(dict(field_cpu_workers=2,field_cpu_buffer_bytes=0))


def test_cli_and_config_cache_profiles_match(tmp_path,monkeypatch):
    from winding_field.cli import main
    from winding_field import workflow
    from winding_field.config import validate_config
    from winding_field.util import atomic_json
    received=[]
    monkeypatch.setattr(workflow,'plan',lambda config:received.append(config.copy()) or {})
    profiles=[('cache64-serial',['--constraint-cache-bytes',str(64<<20)],
               dict(constraint_cache_bytes=64<<20)),
              ('cache0-pool4',['--constraint-cache-bytes','0','--field-cpu-workers','4',
                               '--field-cpu-buffer-bytes',str(512<<20)],
               dict(constraint_cache_bytes=0,field_cpu_workers=4,field_cpu_buffer_bytes=512<<20)),
              ('cache16-pool4',['--constraint-cache-bytes',str(16<<20),'--field-cpu-workers','4',
                                '--field-cpu-buffer-bytes',str(512<<20)],
               dict(constraint_cache_bytes=16<<20,field_cpu_workers=4,field_cpu_buffer_bytes=512<<20))]
    flags=['plan','--source',str(tmp_path/'ct'),'--output',str(tmp_path/'out'),'--device','cpu']
    for name,arguments,config in profiles:
        validate_config(config)
        assert main([*flags,*arguments])==0
        settings=tmp_path/f'{name}.json'
        atomic_json(settings,config)
        assert main([*flags,'--cli-config',str(settings)])==0
    assert received[0::2]==received[1::2]
    assert [profile['constraint_cache_bytes'] for profile in received[0::2]]==[64<<20,0,16<<20]


@pytest.mark.parametrize('constraint_budget',[0,1<<20])
def test_pool_skips_boxes_with_retained_observations_and_rows(tmp_path,monkeypatch,constraint_budget):
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    reference=prepared_run(tmp_path/'reference',shape=(6,6,32),variable=True)
    field.solve(reference,dict(sweeps=3,field_chunk=8))
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    result=field.solve(run,dict(sweeps=3,field_chunk=8,constraint_cache_bytes=constraint_budget,
                                observation_cache_bytes=1<<20,field_cpu_workers=2,
                                field_cpu_buffer_bytes=16<<20))
    digests=[]
    for root in (reference,run):
        cp=json.loads((root/'field/CHECKPOINT.json').read_bytes())
        digests.append([v['sha256'] for _,v in sorted(cp['chunks'].items())])
    assert digests[0]==digests[1]
    pool=result['field_cpu_stats'];observed=result['observation_cache_stats']
    if constraint_budget:
        # Only the first sweep reaches workers; later sweeps reuse both static inputs.
        assert pool['completed']==pool['submitted']==4
        assert observed['hits']==8 and observed['misses']==4
    else:
        # Rows are rebuilt each sweep, so workers still run and observations are never served stale.
        assert pool['completed']==12 and pool['skipped']==0
        assert observed['hits']==0 and observed['misses']==12 and observed['peak_entries']==4


def test_cli_and_config_new_cache_flags_match(tmp_path,monkeypatch):
    from winding_field.cli import main
    from winding_field import workflow
    from winding_field.util import atomic_json
    received=[]
    monkeypatch.setattr(workflow,'plan',lambda config:received.append(config.copy()) or {})
    flags=['plan','--source',str(tmp_path/'ct'),'--output',str(tmp_path/'out'),'--device','cpu']
    assert main(flags)==0
    assert 'observation_cache_bytes' not in received[0] and 'field_cache_bytes' not in received[0]
    assert main([*flags,'--observation-cache-bytes',str(8<<30),'--field-cache-bytes',str(1<<30)])==0
    settings=tmp_path/'config.json'
    atomic_json(settings,dict(observation_cache_bytes=8<<30,field_cache_bytes=1<<30))
    assert main([*flags,'--cli-config',str(settings)])==0
    assert received[1]==received[2]
    assert received[1]['observation_cache_bytes']==8<<30 and received[1]['field_cache_bytes']==1<<30


def test_pool_lookahead_skips_retained_box_after_uncached_boxes(tmp_path,monkeypatch):
    from winding_field import field_cpu
    monkeypatch.setattr(field_cpu,'headroom_bytes',lambda:8<<30)
    reference=prepared_run(tmp_path/'reference',shape=(6,6,32),variable=True)
    field.solve(reference,dict(sweeps=3,field_chunk=8))
    run=prepared_run(tmp_path/'run',shape=(6,6,32),variable=True)
    # Boxes are 360/432/432/360 nodes: first-fit retains boxes 0 and 3 only.
    result=field.solve(run,dict(sweeps=3,field_chunk=8,constraint_cache_bytes=1<<20,
                                observation_cache_bytes=2*(4096+360*17),field_cpu_workers=2,
                                field_cpu_buffer_bytes=16<<20))
    digests=[]
    for root in (reference,run):
        cp=json.loads((root/'field/CHECKPOINT.json').read_bytes())
        digests.append([v['sha256'] for _,v in sorted(cp['chunks'].items())])
    assert digests[0]==digests[1]
    pool=result['field_cpu_stats'];observed=result['observation_cache_stats']
    assert observed['peak_entries']==2 and observed['hits']==4 and observed['misses']==8
    assert pool['skipped']==2 and pool['completed']==pool['submitted']==8
