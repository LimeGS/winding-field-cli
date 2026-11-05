"""Bounded native scheduling must preserve serial DFS ownership and saved UVs."""
import json
import threading
import time

import numpy as np
import pytest

from winding_field import patch_pipeline as p
from test_patch_pipeline import grid


@pytest.mark.parametrize('budget', [1, 5, 4096])
def test_parallel_failure_tree_matches_serial(tmp_path, budget):
    v,f=grid();active=peak=0;lock=threading.Lock()
    def flatten(v,f,path):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        time.sleep(.003 if v[0,0]>12 else .01)
        with lock:active-=1
        return (None if len(f)>8 else v[:,:2].copy()),dict(status='fixture',size=len(f))
    kw=dict(flatten=flatten,max_patch_faces=32,max_depth=3,max_charts=budget)
    serial=p.build_map(v,f,tmp_path/'serial',**kw)
    peak=0
    parallel=p.build_map(v,f,tmp_path/'parallel',workers=4,**kw)
    for key in serial:
        if isinstance(serial[key],np.ndarray):np.testing.assert_array_equal(parallel[key],serial[key])
    assert parallel['report']==serial['report']
    assert peak<=4
    if budget==4096:assert peak>1


def test_worker_settings_and_affinity_contract():
    cfg=dict(output_frame=dict(id='test',voxel_um=9.6,origin_xyz_um=[0,0,0]),app_run='/missing')
    assert p.settings(cfg)['workers']==1
    assert p.settings(dict(cfg,workers=4,native_cpus=[0,1,2,3]))['workers']==4
    for bad in (0,True,5):
        with pytest.raises(ValueError,match='workers'):p.settings(dict(cfg,workers=bad))
    with pytest.raises(ValueError,match='CPU'):p.settings(dict(cfg,workers=4,native_cpus=[1]))


def test_cancel_native_kills_owned_group(tmp_path):
    from winding_field.surface_export import flatten_native
    app=tmp_path/'AppRun';app.write_text('#!/bin/sh\nsleep 60 &\necho $! > child.pid\nwait\n');app.chmod(0o700)
    started=time.monotonic()
    def guard():
        if (tmp_path/'native/child.pid').exists():raise InterruptedError('cancel test')
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
    with pytest.raises(InterruptedError,match='cancel test'):
        flatten_native(v,f,app,tmp_path/'native',guard=guard,timeout=3)
    assert time.monotonic()-started<3
    receipt=json.loads((tmp_path/'native/receipt.json').read_text())
    assert receipt['returncode'] is not None and receipt['returncode']!=0
    import subprocess
    pid=int((tmp_path/'native/child.pid').read_text())
    status=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
    assert not status or status.startswith('Z')


def test_guard_aborts_parallel_without_orphan_threads(tmp_path):
    v,f=grid();cancel=threading.Event();running=0;lock=threading.Lock()
    def flatten(v,f,path):
        nonlocal running
        with lock:running+=1
        try:
            cancel.wait(.1)
            return v[:,:2].copy(),{}
        finally:
            with lock:running-=1
    calls=0
    def guard():
        nonlocal calls
        calls+=1
        if calls>4:cancel.set();raise InterruptedError('stop')
    with pytest.raises(InterruptedError):p.build_map(v,f,tmp_path,flatten=flatten,workers=4,max_patch_faces=8,guard=guard)
    assert running==0


def cache_fixture(tmp_path):
    from winding_field.surface_export import flatten_native
    from winding_field.util import file_sha
    from test_surface_export import native_fixture,native_plane
    app=native_fixture(tmp_path/'AppRun');v,f=native_plane()
    root=tmp_path/'old';patch=root/'level-0000/patches/patch-000000';patch.parent.mkdir(parents=True)
    uv,receipt=flatten_native(v,f,app,patch,expected_sha256=file_sha(app))
    identity=dict(source={'fixture':True},mesh_sha256='mesh',settings=dict(output_frame={'test':True},max_patch_faces=4096,max_depth=3,max_charts=4096),
                  native=dict(launcher_sha256=file_sha(app),binary_sha256='binary',iterations=200,energy='symmetric_dirichlet',relative_energy_tolerance=1e-6))
    (root/'IDENTITY.json').write_text(json.dumps(identity))
    return root,identity,v,f,uv,patch


def test_reuse_saved_uv_and_copy_binding(tmp_path):
    from winding_field.native_patches import NativeReuse
    from winding_field.util import file_sha
    root,identity,v,f,uv,patch=cache_fixture(tmp_path)
    reuse=NativeReuse(root,file_sha(root/'IDENTITY.json'),identity)
    out=tmp_path/'new';saved,receipt=reuse.get(0,v,f,out)
    np.testing.assert_array_equal(saved,uv)
    assert receipt['reused_from']['receipt_sha256']==file_sha(patch/'receipt.json')
    assert file_sha(out/'inputmesh_flatboi.obj')==file_sha(patch/'inputmesh_flatboi.obj')
    assert reuse.get(0,v+1,f,tmp_path/'wrong') is None
    assert reuse.count==1


@pytest.mark.parametrize('tamper',['inputmesh.obj','inputmesh_flatboi.obj','stdout.log','identity','config','native','receipt'])
def test_reuse_rejects_tampered_cache(tmp_path,tamper):
    from winding_field.native_patches import NativeReuse
    from winding_field.util import file_sha
    root,identity,v,f,uv,patch=cache_fixture(tmp_path)
    digest=file_sha(root/'IDENTITY.json')
    if tamper=='identity':(root/'IDENTITY.json').write_text('{}')
    elif tamper=='config':identity['settings']['max_depth']=2
    elif tamper=='native':identity['native']['binary_sha256']='other'
    elif tamper=='receipt':
        data=json.loads((patch/'receipt.json').read_text());data['iterations']=199
        (patch/'receipt.json').write_text(json.dumps(data))
    else:
        with (patch/tamper).open('ab') as stream:stream.write(b'\n# changed\n')
    with pytest.raises(ValueError):
        reuse=NativeReuse(root,digest,identity);reuse.get(0,v,f,tmp_path/'new')


def test_reuse_incomplete_and_failed_are_not_success(tmp_path):
    from winding_field.native_patches import NativeReuse
    from winding_field.util import file_sha
    root,identity,v,f,uv,patch=cache_fixture(tmp_path)
    data=json.loads((patch/'receipt.json').read_text());data['returncode']=7;data['status']='FAIL'
    (patch/'receipt.json').write_text(json.dumps(data))
    (patch.parent/'patch-000001').mkdir()
    reuse=NativeReuse(root,file_sha(root/'IDENTITY.json'),identity)
    assert reuse.get(0,v,f,tmp_path/'new') is None
    assert reuse.inventory['failed']==1 and reuse.inventory['incomplete']==1


def test_resource_guards_stop_at_cap_and_reserve(tmp_path,monkeypatch):
    from winding_field import native_patches as n
    from types import SimpleNamespace
    monkeypatch.setattr(n,'process_tree_rss',lambda:12*2**30)
    with pytest.raises(InterruptedError,match='RSS'):n.ResourceGuard(tmp_path,11,8)()
    monkeypatch.setattr(n,'process_tree_rss',lambda:1)
    monkeypatch.setattr(n.shutil,'disk_usage',lambda path:SimpleNamespace(free=7*2**30))
    with pytest.raises(InterruptedError,match='disk'):n.ResourceGuard(tmp_path,11,8)()


def test_parallel_saved_uvs_preserve_raster(tmp_path):
    from winding_field.surface_export import flatten_native
    from winding_field.util import file_sha
    from test_surface_export import native_fixture
    from winding_field.native_patches import NativeReuse
    from PIL import Image
    v,f=grid();app=native_fixture(tmp_path/'AppRun');root=tmp_path/'cache'
    target=root/'level-0000/patches'
    p.build_map(v,f,target,flatten=lambda v,f,path:flatten_native(v,f,app,path,expected_sha256=file_sha(app)),max_patch_faces=32)
    identity=dict(source={},mesh_sha256='fixture',settings=dict(output_frame={},max_patch_faces=32,max_depth=3,max_charts=4096),
                  native=dict(launcher_sha256=file_sha(app),binary_sha256='fixture',iterations=200,energy='symmetric_dirichlet',relative_energy_tolerance=1e-6))
    (root/'IDENTITY.json').write_text(json.dumps(identity))
    maps=[]
    for workers in (1,4):
        cache=NativeReuse(root,file_sha(root/'IDENTITY.json'),identity)
        result=p.build_map(v,f,tmp_path/f'map-{workers}',flatten=lambda v,f,path:cache.get(0,v,f,path),workers=workers,max_patch_faces=32)
        maps.append(result)
        p.render_map(result,tmp_path/f'render-{workers}',lambda xyz:(xyz[:,0]+xyz[:,1],np.ones(len(xyz),bool)),
                     dict(id='test',voxel_um=9.6,origin_xyz_um=[0,0,0]),pitch_um=9.6,tile_size=8)
    for key in maps[0]:
        if isinstance(maps[0][key],np.ndarray):np.testing.assert_array_equal(maps[0][key],maps[1][key])
    for image in (tmp_path/'render-1').rglob('*.png'):
        other=tmp_path/'render-4'/image.relative_to(tmp_path/'render-1')
        np.testing.assert_array_equal(np.asarray(Image.open(image)),np.asarray(Image.open(other)))


def test_native_single_thread_includes_pastix(tmp_path):
    from test_surface_export import native_fixture,native_plane
    from winding_field.surface_export import flatten_native
    app=native_fixture(tmp_path/'AppRun')
    app.write_text(app.read_text().replace("action =", "import os\nPath('thread-env.json').write_text(json.dumps({k:os.environ.get(k) for k in ('PASTIX_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))\naction =",1))
    out=tmp_path/'native';uv,_=flatten_native(*native_plane(),app,out,threads=1)
    assert uv is not None
    assert set(json.loads((out/'thread-env.json').read_text()).values())=={'1'}


def test_reuse_rejects_ambiguous_saved_outputs(tmp_path):
    import shutil
    from winding_field.native_patches import NativeReuse
    from winding_field.util import file_sha
    root,identity,v,f,uv,patch=cache_fixture(tmp_path)
    other=patch.with_name('patch-000001');shutil.copytree(patch,other)
    data=json.loads((other/'receipt.json').read_text());data['output_obj_sha256']='a'*64
    (other/'receipt.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='ambiguous'):
        reuse=NativeReuse(root,file_sha(root/'IDENTITY.json'),identity);reuse.get(0,v,f,tmp_path/'new')
