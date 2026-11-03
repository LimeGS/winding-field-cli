import json
import os
import sys
import time

import numpy as np
import pytest


def field_case(tmp_path,levels=(.5,1.5)):
    from test_field import prepared_run
    from ct_fixture import write_ct
    from winding_field.field import solve
    run=prepared_run(tmp_path/'run')
    solve(run,{'sweeps':1,'field_chunk':8})
    z,y,_=np.indices((6,6,24))
    ct=write_ct(tmp_path/'ct.zarr',(128+y+3*z).astype(np.uint8),spacing=(2.,3.,4.),origin=(10.,20.,30.))
    config=dict(source=str(ct),output=str(run),cache=str(tmp_path/'cache'),reserve_bytes=0,
                levels=list(levels),surface_output='native',coverage_target=1.,surface_time_limit_s=30.,
                chart_faces=32,mesh_tile=4,export_tifxyz=False)
    return run,config


def test_native_levels_publish_independently_and_resume_is_readonly(tmp_path):
    from winding_field.native_pipeline import work
    run,config=field_case(tmp_path)
    field_before=(run/'field/CHECKPOINT.json').read_bytes()
    paused=work(run,config,until='mesh')
    assert paused['status']=='PAUSED'
    assert (run/'surfaces/level-0000/meshes/MESH.json').exists()
    result=work(run,config)
    assert result['status']=='COMPLETE'
    assert len(result['levels'])==2
    assert all(r['target_met'] for r in result['levels'])
    assert all(r['coverage_fraction']==1. for r in result['levels'])
    for record in result['levels']:
        assert all(record['timings_s'][phase]>0 for phase in ('mesh','flatten','assemble','rasterize','sample_ct','publish'))
        assert sum(record['timings_s'].values())<=record['elapsed_s']+.1
    files=list((run/'surfaces').glob('level-*/component-*/sheet.png'))
    assert len(files)==2
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
    assert work(run,config)['status']=='COMPLETE'
    assert before=={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
    assert (run/'field/CHECKPOINT.json').read_bytes()==field_before


def test_cancel_then_resume_native_retains_completed_geometry(tmp_path):
    from winding_field.native_pipeline import work
    run,config=field_case(tmp_path,levels=(.5,))
    def stop():
        return any((run/'surfaces').glob('level-*/charts/*.npz'))
    assert work(run,config,stop=stop)['status']=='PAUSED'
    charts=list((run/'surfaces').glob('level-*/charts/*.npz'))
    assert charts
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in charts}
    assert work(run,config)['status']=='COMPLETE'
    assert before=={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in charts}


def test_supervisor_can_kill_unresponsive_owned_worker_and_recover_checkpoint(tmp_path):
    from winding_field.native_pipeline import _supervise
    from winding_field.native_store import Journal
    root=tmp_path/'surface';root.mkdir()
    script=tmp_path/'blocked.py'
    script.write_text('''import signal,time,sys
from pathlib import Path
import numpy as np
from winding_field.native_store import Journal
signal.signal(signal.SIGTERM,signal.SIG_IGN)
with Journal(Path(sys.argv[1])/"level-0000",{"source":"fixed"}) as store:
    store.save_arrays("saved.npz",points=np.array([[1.,2.,3.]]))
    time.sleep(60)
''')
    began=time.monotonic()
    def stop():
        if not (root/'level-0000/saved.npz').exists():
            return False
        assert json.loads((root/'STATUS.json').read_bytes())['status']=='RUNNING'
        return True
    result=_supervise([sys.executable,'-B',str(script),str(root)],root,seconds=30,
                     stop=stop)
    assert result['status']=='PAUSED'
    assert result['reason']=='STOP'
    assert time.monotonic()-began<10
    with Journal(root/'level-0000',{'source':'fixed'}) as store:
        np.testing.assert_array_equal(store.load_arrays('saved.npz')['points'],[[1,2,3]])


def test_time_budget_is_cumulative_and_extension_does_not_relabel_speed(tmp_path):
    from winding_field.native_pipeline import run_native
    run,config=field_case(tmp_path,levels=(.5,))
    config['surface_time_limit_s']=.05
    first=run_native(run,config)
    assert first['status']=='PAUSED'
    assert first['reason']=='TIME_LIMIT'
    assert run_native(run,config)['status']=='PAUSED'
    result=run_native(run,config,extra_seconds=30)
    assert result['status']=='COMPLETE',result
    assert result['levels'][0]['original_deadline_met'] is False


def test_completed_publication_can_be_verified_after_budget_is_spent(tmp_path):
    from winding_field.native_pipeline import work,run_native
    run,config=field_case(tmp_path,levels=(.5,))
    assert work(run,config)['status']=='COMPLETE'
    (run/'surfaces/level-0000/TIME.json').write_text(json.dumps({'elapsed_s':config['surface_time_limit_s']}))
    assert run_native(run,config)['status']=='COMPLETE'


def test_changed_level_catalog_cannot_omit_a_requested_level_on_resume(tmp_path):
    from winding_field.native_pipeline import work
    run,config=field_case(tmp_path)
    assert work(run,config,until='mesh')['status']=='PAUSED'
    (run/'surfaces/LEVELS.json').write_text('[0.5]')
    with pytest.raises(ValueError,match='level|catalog'):
        work(run,config)
    assert not (run/'SURFACES.json').exists()


def test_native_output_symlink_is_rejected_before_writing_outside_run(tmp_path):
    from winding_field.native_pipeline import run_native
    run,config=field_case(tmp_path)
    outside=tmp_path/'external'; outside.mkdir()
    (run/'surfaces').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        run_native(run,config)
    assert list(outside.iterdir())==[]


def test_unregistered_level_receipt_is_recovered_after_publication_stop(tmp_path,monkeypatch):
    import winding_field.native_pipeline as pipeline
    from winding_field.native_store import Paused
    run,config=field_case(tmp_path,levels=(.5,))
    original=pipeline.atomic_json
    def interrupted(path,value):
        original(path,value)
        if path.name=='LEVEL.json':
            raise Paused('PUBLICATION_STOP')
    monkeypatch.setattr(pipeline,'atomic_json',interrupted)
    assert pipeline.work(run,config)['status']=='PAUSED'
    level=run/'surfaces/level-0000'
    assert (level/'LEVEL.json').exists()
    with __import__('sqlite3').connect(level/'work.sqlite') as db:
        assert db.execute("SELECT 1 FROM files WHERE name='LEVEL.json'").fetchone() is None
    saved={str(p):p.read_bytes() for p in level.glob('component-*/sheet.png')}
    monkeypatch.setattr(pipeline,'atomic_json',original)
    assert pipeline.work(run,config)['status']=='COMPLETE'
    assert saved=={str(p):p.read_bytes() for p in level.glob('component-*/sheet.png')}


def test_resume_charges_startup_and_journal_validation_exactly_once(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import winding_field.native_pipeline as pipeline
    import winding_field.native_store as storage
    run,config=field_case(tmp_path,levels=(.5,))
    clock=[1000.]
    fake=SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0])
    original=storage.Journal
    class DelayedJournal(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            clock[0]+=5.
    monkeypatch.setattr(pipeline,'time',fake)
    monkeypatch.setattr(storage,'time',fake)
    monkeypatch.setattr(pipeline,'Journal',DelayedJournal)
    result=pipeline.work(run,config,started_epoch=998.)
    assert result['status']=='COMPLETE'
    assert result['levels'][0]['elapsed_s']==7.
