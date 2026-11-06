import argparse
import importlib.util
import json
import sys

import numpy as np
import pytest

from winding_field.util import atomic_json,file_sha


def runner():
    assert importlib.util.find_spec('winding_field.flatboi_checkpoint'), 'installed checkpoint runner missing'
    from winding_field import flatboi_checkpoint
    return flatboi_checkpoint


def inputs(tmp_path):
    v=np.array([[0.,0,0],[1.,0,0],[1.,1,0],[0.,1,0]])
    f=np.array([[0,1,2],[0,2,3]])
    text='v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n'
    obj=tmp_path/'input.obj';obj.write_text(text)
    mapped=tmp_path/'initial.npz';np.savez(mapped,world=v,faces=f)
    return v,f,obj,mapped


def test_durable_checkpoint_rejects_corruption_and_does_not_advance_on_cancel(tmp_path):
    r=runner();v,f,obj,_=inputs(tmp_path);identity={'source':'test'}
    batch=tmp_path/'batch-0';batch.mkdir()
    record=r.publish_checkpoint(tmp_path,batch,obj,v,f,identity,10,8.,False,lambda:None)
    before=(tmp_path/'LATEST.json').read_bytes()
    partial=tmp_path/'partial';partial.mkdir();(partial/'checkpoint.obj').write_text('incomplete')
    assert r.load_checkpoint(tmp_path,identity)[0]==record
    def stop():raise InterruptedError('stop')
    with pytest.raises(InterruptedError):r.publish_checkpoint(tmp_path,partial,obj,v,f,identity,20,7.,False,stop)
    assert (tmp_path/'LATEST.json').read_bytes()==before
    with pytest.raises(ValueError):r.load_checkpoint(tmp_path,{'source':'changed'})
    (tmp_path/record['obj']).write_text('corrupt')
    with pytest.raises(ValueError,match='hash'):r.load_checkpoint(tmp_path,identity)


@pytest.mark.parametrize('replacement',[('v 1 0 0','v 2 0 0'),('vt 0 1','vt 0 -1')])
def test_bad_geometry_and_uv_never_publish(tmp_path,replacement):
    r=runner();v,f,obj,_=inputs(tmp_path);obj.write_text(obj.read_text().replace(*replacement))
    batch=tmp_path/'batch';batch.mkdir()
    with pytest.raises(ValueError):r.publish_checkpoint(tmp_path,batch,obj,v,f,{},10,8.,False,lambda:None)
    assert not (tmp_path/'LATEST.json').exists()


def fake_native(tmp_path,monkeypatch,r):
    monkeypatch.setattr(r,'require_cgroup',lambda:None)
    app=tmp_path/'AppRun';app.write_text(f'''#!{sys.executable}
import os,sys,shutil
from pathlib import Path
assert sys.argv[1]=='flatboi' and sys.argv[4:]==['symmetric_dirichlet','--tol=1e-6']
assert os.environ['OMP_NUM_THREADS']==os.environ['PASTIX_NUM_THREADS']=='2'
p=Path(sys.argv[2]);n=int(sys.argv[3])
shutil.copyfile(p,p.with_name('inputmesh_flatboi.obj'))
p.with_name('inputmesh_energies_flatboi.txt').write_text('8\\n'*(n+1))
print(f'PROGRESS {{n}}/{{n}}',flush=True)
''');app.chmod(0o755)
    monkeypatch.setattr(r,'check_flatboi',lambda path,recipe:dict(app_run=str(app),launcher_sha256=file_sha(app),binary_sha256='b'*64,recipe=recipe))
    return app


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_subprocess_resume_and_legacy_import_preserve_original_identity(tmp_path,monkeypatch):
    r=runner();v,f,obj,mapped=inputs(tmp_path);app=fake_native(tmp_path,monkeypatch,r)
    args=argparse.Namespace(output=tmp_path/'run',source_map=mapped,input=obj,app_run=app,native_recipe='installed-2f89bfc9',iterations=10,batch_iterations=10,hours=1.)
    r.run(args)
    status=r.observe(args.output);assert status['saved']==status['calculated']==10
    args.iterations=20;r.run(args)
    latest=json.loads((args.output/'LATEST.json').read_text());assert latest['iterations']==20
    # Simulate the original operational runner identity, preserving all other bindings.
    old=dict(latest['identity'],runner_sha256='a'*64)
    latest['identity']=old
    atomic_json(args.output/'LATEST.json',latest);atomic_json(args.output/'IDENTITY.json',old)
    atomic_json((args.output/latest['obj']).parent/'CHECKPOINT.json',latest)
    original=(args.output/'LATEST.json').read_bytes()
    with pytest.raises(ValueError,match='identity'):
        r.import_checkpoint(args.output,tmp_path/'wrong',mapped,obj,app,'c'*64)
    result=r.import_checkpoint(args.output,tmp_path/'import',mapped,obj,app,'a'*64)
    assert result['identity']['runner_sha256']!=old['runner_sha256']
    receipt=json.loads((tmp_path/'import'/'IMPORTED.json').read_text())
    assert receipt['checkpoint']==latest and receipt['source_latest_sha256']==file_sha(args.output/'LATEST.json')
    assert (args.output/'LATEST.json').read_bytes()==original
    args.output=tmp_path/'import';args.iterations=30;r.run(args)
    assert r.observe(args.output)['saved']==30
    obj.write_text(obj.read_text()+'# changed\n')
    with pytest.raises(ValueError,match='identity'):r.run(args)


def test_status_distinguishes_live_calculation_from_durable_progress(tmp_path):
    r=runner();batch=tmp_path/'batch-000010-test';batch.mkdir()
    atomic_json(tmp_path/'STATUS.json',dict(status='RUNNING',target=200,started=1))
    atomic_json(tmp_path/'LATEST.json',dict(iterations=10,energy=8))
    atomic_json(batch/'COMMAND.json',dict(prior_iterations=10))
    (batch/'stdout.log').write_text('PROGRESS 7/10\n[it 7] E = 7.5\n')
    data=r.observe(tmp_path)
    assert (data['saved'],data['calculated'],data['energy'])==(10,17,7.5)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_native_run_binds_preparation_and_import_rejects_symlink_ancestor(tmp_path,monkeypatch):
    r=runner();v,f,obj,mapped=inputs(tmp_path);app=fake_native(tmp_path,monkeypatch,r)
    prep=tmp_path/'PREPARED.json';atomic_json(prep,dict(centerline=[[0,0,0],[0,0,1]]))
    args=argparse.Namespace(output=tmp_path/'run',source_map=mapped,input=obj,app_run=app,native_recipe='installed-2f89bfc9',iterations=10,batch_iterations=10,hours=1.)
    r.run(args)
    latest=json.loads((args.output/'LATEST.json').read_text())
    assert latest['identity'].get('source_prepared_sha256')==file_sha(prep)
    target=tmp_path/'target';target.mkdir();alias=tmp_path/'alias';alias.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        r.import_checkpoint(args.output,alias/'new',mapped,obj,app,file_sha(r.__file__))
    assert not (target/'new').exists()
    atomic_json(prep,dict(centerline=[[1,0,0],[1,0,1]]))
    with pytest.raises(ValueError,match='identity'):r.run(args)
