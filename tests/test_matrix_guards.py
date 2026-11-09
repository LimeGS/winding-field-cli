"""Resume must check authoritative CT; nested workers must honor export guards."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
import sys

import numpy as np
import pytest

from winding_field.util import atomic_json
from test_flattening_matrix import request, source_bundle


@pytest.mark.parametrize('was_missing',[False,True])
def test_remote_resume_detects_change_despite_retained_native_cache(tmp_path,was_missing,monkeypatch):
    from types import SimpleNamespace
    # This tiny fixture tests source mutation, not the host's free disk space.
    monkeypatch.setattr('shutil.disk_usage',lambda path:SimpleNamespace(free=16<<30))
    from winding_field.patch_pipeline import PatchCT
    from winding_field.flattening_matrix import settings
    _,source=source_bundle(tmp_path);chunk=source/'0.0.0';raw=chunk.read_bytes()
    if was_missing:chunk.unlink()
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(source)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    cfg=settings(request(tmp_path),'lscm','sheet');out=tmp_path/'ledger';out.mkdir()
    try:
        ct=PatchCT(url,out,cfg,voxel_um=9.6)
        try:
            _,valid=ct.sample(np.array([[5.,3.,5.]]),cfg['output_frame'])
            assert valid[0] is not None and bool(valid[0]) is not was_missing
        finally:ct.close()
        chunk.write_bytes(raw if was_missing else bytes(32**3))
        ct=PatchCT(url,out,cfg,voxel_um=9.6,resume=True)
        try:
            with pytest.raises(ValueError,match='CT chunk changed'):ct.validate_used()
        finally:ct.close()
    finally:server.shutdown();server.server_close();thread.join()


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_field_extraction_honors_export_stop_and_reserve(tmp_path,monkeypatch):
    from winding_field.cli import main
    from winding_field import surface_mesh
    field=tmp_path/'field.npy';np.save(field,np.broadcast_to(np.arange(7.),(7,7,7)))
    from winding_field.util import file_sha
    descriptor=dict(field=dict(kind='npy',path=str(field),sha256=file_sha(field)),geometry=dict(
        frame='test',voxel_um=9.6,origin_xyz=[0,0,0],spacing_xyz=[1,1,1],shape_zyx=[7]*3,sheet_step=1.))
    atomic_json(tmp_path/'field.json',descriptor);atomic_json(tmp_path/'config.json',request(tmp_path))
    out=tmp_path/'out';real=surface_mesh.check_resources;reserves=[]
    def resources(path,**kwargs):
        reserves.append(kwargs.get('reserve_bytes',0))
        (out/'STOP').touch()
        return real(path,**kwargs)
    monkeypatch.setattr(surface_mesh,'check_resources',resources)
    assert main(['export-sheets','--field-config',str(tmp_path/'field.json'),'--levels','3.5',
        '--source','unused','--flattening-solver','lscm','--scope','sheet',
        '--flattening-config',str(tmp_path/'config.json'),'--output',str(out)])==1
    assert reserves and min(reserves)>=8*2**30
    assert not (out/'meshes/MESH.json').exists()


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_matrix_guard_interrupts_native_before_next_checkpoint(tmp_path,monkeypatch):
    from winding_field import flatboi_checkpoint as native
    from winding_field.flattening_matrix import build_map, settings
    from test_flatboi_checkpoint import fake_native
    app=fake_native(tmp_path,monkeypatch,native)
    cfg=settings(dict(request(tmp_path),app_run=str(app)),'slim','sheet')
    v=np.array([[1.,1,1],[3.,1,1],[1.,1,3]]);f=np.array([[0,1,2]],np.int64)
    out=tmp_path/'map';root=out/'domain-000000/native'
    def guard():
        if (root/'LATEST.json').exists():raise InterruptedError('outer export stop fixture')
    with pytest.raises(InterruptedError,match='outer export stop'):
        build_map(v,f,out,cfg,guard=guard)
    from winding_field.sheet_atlas import _json
    assert _json(root/'LATEST.json')['iterations']==10
    assert len(list(root.glob('batch-*')))==1
    assert not (out/'MAP.json').exists()
