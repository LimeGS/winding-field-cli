import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def invoke(*args,cwd):
    env=dict(os.environ,OMP_NUM_THREADS='2')
    env.pop('PYTHONPATH',None)
    return subprocess.run([sys.executable,'-m','winding_field',*map(str,args)],cwd=cwd,env=env,
                          capture_output=True,text=True,timeout=180)


def test_installed_end_to_end_demo_and_resume_from_other_directory(tmp_path):
    run=tmp_path/'demo'
    result=invoke('demo','--output',run,cwd=tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads((run/'RUN.json').read_text())
    assert receipt['status']=='COMPLETE'
    assert receipt['physical_acceptance'] is False
    assert (run/'index.html').exists()
    assert list((run/'flat').glob('*.png'))
    assert list((run/'meshes').glob('*.obj'))
    assert (run/'fibres/fibres.afv').exists()
    assert (run/'lasagna/fibres.lasagna.json').exists()
    before=(run/'field/CHECKPOINT.json').read_bytes()
    result=invoke('resume',run,cwd=tmp_path)
    assert result.returncode==0,result.stderr
    assert before==(run/'field/CHECKPOINT.json').read_bytes()
    next((run/'meshes').glob('*.obj')).unlink()
    result=invoke('resume',run,cwd=tmp_path)
    assert result.returncode!=0
    assert 'export' in result.stderr or 'product' in result.stderr


def test_until_then_resume_and_refuse_changed_input(tmp_path):
    from ct_fixture import write_ct
    x=np.indices((12,16,32))[2]
    source=write_ct(tmp_path/'ct.zarr',(20+210*(np.cos(2*np.pi*x/8)>.4)).astype(np.uint8))
    run=tmp_path/'run'
    result=invoke('run','--source',source,'--output',run,'--prediction','classical','--reserve-gib','0','--until','prepare',cwd=tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((run/'RUN.json').read_text())['status']=='PAUSED'
    assert not (run/'FIELD.json').exists()
    (source/'0/0.0.0').write_bytes(bytes([10])*32**3)
    result=invoke('resume',run,cwd=tmp_path)
    assert result.returncode!=0
    assert 'changed' in result.stderr


def test_empty_ct_records_failure_not_a_completed_reconstruction(tmp_path):
    from ct_fixture import write_ct
    source=write_ct(tmp_path/'empty.zarr',np.zeros((8,8,16),np.uint8))
    run=tmp_path/'run'
    result=invoke('run','--source',source,'--output',run,'--prediction','classical','--reserve-gib','0',cwd=tmp_path)
    assert result.returncode!=0
    receipt=json.loads((run/'RUN.json').read_text())
    assert receipt['status']=='FAILED'
    assert 'supported' in receipt['error']
    assert not (run/'FIELD.json').exists()


def test_resume_admits_remaining_products_and_completed_runs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from winding_field.demo import synthetic
    from winding_field.workflow import run
    source=synthetic(tmp_path/'ct.zarr'); output=tmp_path/'run'
    config=dict(source=source,output=str(output),prediction='classical',chunk=64,
                field_chunk=64,sweeps=1,seed_stride=4,reserve_bytes=4096,cache=str(tmp_path/'cache'))
    state=run(config,until='prepare')
    total=24*32*56*100
    written=sum(p.stat().st_size for name in ('maps','lasagna') for p in (output/name).rglob('*') if p.is_file())
    free=total-min(written,total//2)+4096
    assert free<total+4096
    monkeypatch.setattr('winding_field.workflow.shutil.disk_usage',lambda p:SimpleNamespace(free=free))
    assert run(state['config'],resume=True,until='fibres')['status']=='PAUSED'
    free=1<<30  # Mesh extraction separately reserves its transactional working space.
    assert run(state['config'],resume=True)['status']=='COMPLETE'
    free=8192
    assert run(state['config'],resume=True)['status']=='COMPLETE'
