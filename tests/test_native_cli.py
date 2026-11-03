import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from test_cli import invoke
from test_native_runtime import field_case


@pytest.mark.parametrize('value',[0.,-1.,1.1,float('nan'),True])
def test_invalid_native_coverage_is_rejected_before_io(value):
    from winding_field.config import validate_config
    with pytest.raises(ValueError,match='coverage'):
        validate_config({'surface_output':'native','coverage_target':value})


@pytest.mark.parametrize('value',[0.,-1.,float('inf'),float('nan'),True])
def test_invalid_native_time_is_rejected_before_io(value):
    from winding_field.config import validate_config
    with pytest.raises(ValueError,match='time'):
        validate_config({'surface_output':'native','surface_time_limit_s':value})


def test_native_stage_path_preserves_legacy_and_direct_routes():
    from winding_field.workflow import stage_path
    assert [s for s,_ in stage_path({'surface_output':'native'})]==['prepare','fibres','field','surfaces']
    assert [s for s,_ in stage_path({})]==['prepare','fibres','field','mesh','flatten']
    assert [s for s,_ in stage_path({'surface_output':'tifxyz'})]==['prepare','fibres','field','tifxyz']


def test_missing_optional_tiff_dependency_fails_before_creating_run(tmp_path,monkeypatch):
    import builtins
    from winding_field.workflow import run
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name=='tifffile':
            raise ImportError('not installed')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='tifffile'):
        run({'source':str(tmp_path/'missing-ct'),'output':str(output),'cache':str(tmp_path/'cache'),
             'surface_output':'native','export_tifxyz':True})
    assert not output.exists()


def test_new_cli_defaults_native_and_records_requested_controls(tmp_path):
    from ct_fixture import write_ct
    import numpy as np
    ct=write_ct(tmp_path/'ct.zarr',np.full((8,8,16),128,np.uint8))
    out=tmp_path/'out'
    result=invoke('run','--source',ct,'--output',out,'--cache',tmp_path/'cache','--prediction','classical','--reserve-gib','0',
                  '--until','prepare','--coverage-target','.9','--surface-time-limit','20',cwd=tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    config=json.loads((out/'RUN.json').read_bytes())['config']
    assert config['surface_output']=='native'
    assert config['coverage_target']==.9
    assert config['surface_time_limit_s']==20


def test_installed_cli_native_renders_and_resumes_from_other_directory(tmp_path):
    from winding_field.demo import synthetic
    source=synthetic(tmp_path/'ct.zarr'); out=tmp_path/'out'
    result=invoke('run','--source',source,'--output',out,'--cache',tmp_path/'cache','--prediction','classical','--reserve-gib','0',
                  '--levels','.5','--chart-faces','128','--surface-time-limit','60',cwd=tmp_path)
    assert result.returncode in (0,2),result.stdout+result.stderr
    state=json.loads((out/'RUN.json').read_bytes())
    assert state['status'] in ('COMPLETE','PARTIAL')
    assert state['stages']['surfaces']['receipt']=='SURFACES.json'
    images=list((out/'surfaces').glob('level-*/component-*/sheet.png'))
    assert images
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in images}
    again=invoke('resume',out,cwd=tmp_path)
    assert again.returncode==result.returncode,again.stdout+again.stderr
    assert before=={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in images}
    images[0].write_bytes(b'corrupt')
    broken=invoke('resume',out,cwd=tmp_path)
    assert broken.returncode==1
    assert 'changed' in broken.stdout+broken.stderr


@pytest.mark.parametrize('method',['sigint','stop-file'])
def test_actual_cli_cancellation_preserves_checkpoints_and_resumes(tmp_path,method):
    from winding_field.demo import synthetic
    source=synthetic(tmp_path/'ct.zarr'); out=tmp_path/'out'
    prepared=invoke('run','--source',source,'--output',out,'--cache',tmp_path/'cache','--prediction','classical','--reserve-gib','0',
                    '--levels','.5','--chart-faces','8','--surface-time-limit','90','--until','field',cwd=tmp_path)
    assert prepared.returncode==0,prepared.stdout+prepared.stderr
    field_before=(out/'field/CHECKPOINT.json').read_bytes()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    with (tmp_path/'resume.log').open('w+') as log:
        process=subprocess.Popen([sys.executable,'-B','-m','winding_field','resume',str(out)],cwd=tmp_path,env=env,
                                 stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        try:
            while not list((out/'surfaces').glob('level-*/charts/*.npz')):
                assert process.poll() is None
                assert time.monotonic()<deadline
                time.sleep(.01)
            if method=='sigint':
                process.send_signal(signal.SIGINT)
            else:
                (out/'STOP').touch()
            process.wait(timeout=10)
            log.seek(0); output=log.read()
            assert process.returncode==0,output
        finally:
            if process.poll() is None:
                process.kill();process.wait()
    state=json.loads((out/'RUN.json').read_bytes())
    assert state['status']=='PAUSED'
    assert 'surfaces' not in state['stages']
    root=out/'surfaces/level-0000'
    with sqlite3.connect(root/'work.sqlite') as db:
        names=[r[0] for r in db.execute("SELECT name FROM files WHERE name LIKE 'charts/%'")]
    assert names
    before={name:((root/name).read_bytes(),(root/name).stat().st_mtime_ns) for name in names}
    result=invoke('resume',out,cwd=tmp_path)
    assert result.returncode in (0,2),result.stdout+result.stderr
    assert json.loads((out/'RUN.json').read_bytes())['status'] in ('COMPLETE','PARTIAL')
    assert before=={name:((root/name).read_bytes(),(root/name).stat().st_mtime_ns) for name in names}
    assert (out/'field/CHECKPOINT.json').read_bytes()==field_before


def test_completed_cli_resume_revalidates_native_ct_snapshots(tmp_path):
    from winding_field.demo import synthetic
    source=synthetic(tmp_path/'ct.zarr'); out=tmp_path/'out'
    result=invoke('run','--source',source,'--output',out,'--cache',tmp_path/'cache','--prediction','classical',
                  '--reserve-gib','0','--levels','.5','--chart-faces','128','--surface-time-limit','60',cwd=tmp_path)
    assert result.returncode in (0,2),result.stdout+result.stderr
    with sqlite3.connect(out/'surfaces/level-0000/CT-SNAPSHOT.sqlite') as snapshot:
        key=snapshot.execute('SELECT key FROM chunks LIMIT 1').fetchone()[0]
    # A native-only source record must remain protected even if working ledgers omit it.
    for ledger in (out/'source.sqlite',out/'surfaces/source.sqlite'):
        with sqlite3.connect(ledger) as db:
            db.execute('DELETE FROM chunks WHERE key=?',(key,)); db.commit()
    path=tmp_path/'ct.zarr'/key
    path.write_bytes(bytes([251])*path.stat().st_size)
    result=invoke('resume',out,cwd=tmp_path)
    assert result.returncode==1,result.stdout+result.stderr
    assert 'source chunk changed' in result.stdout+result.stderr
