import json,shutil
from pathlib import Path
import numpy as np
import pytest
from winding_field import patch_pipeline as p
from winding_field.util import atomic_json,file_sha


def fixture(tmp_path):
    uv=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    mapped=dict(world=np.column_stack((uv,np.zeros(4))),uv=uv,faces=np.array([[0,1,2],[0,2,3]]),
        display=np.ones(2,bool),source_face_rows=np.array([17,29]))
    root=tmp_path/'old';level=root/'level-0000';level.mkdir(parents=True)
    atomic_json(root/'IDENTITY.json',dict(test=True))
    (level/'map.npz').write_bytes(b'bound-map')
    sample=lambda xyz:(xyz[:,0]+xyz[:,1],np.ones(len(xyz),bool))
    result=p.render_map(mapped,level/'render',sample,dict(voxel_um=1.),pitch_um=1.,tile_size=4)
    return root,level,mapped,sample,result


def reuse(root,level):
    from winding_field.patch_continue import TileReuse
    return TileReuse(level,identity_sha256=file_sha(root/'IDENTITY.json'),map_sha256=file_sha(level/'map.npz'))


def test_verified_partial_continuation_preserves_tiles_and_avoids_resampling(tmp_path):
    root,level,mapped,sample,original=fixture(tmp_path)
    saved=original['tiles'][:3]
    for tile in original['tiles'][3:]:shutil.rmtree(level/'render'/tile['directory'])
    inventory=reuse(root,level)
    continued=p.render_map(mapped,tmp_path/'new',sample,dict(voxel_um=1.),pitch_um=1.,tile_size=4,reuse=inventory,raster_workers=2)
    assert [t['directory'] for t in continued['tiles']]==[t['directory'] for t in original['tiles']]
    assert sum('reused_from' in t for t in continued['tiles'])==3
    for expected,actual in zip(original['tiles'],continued['tiles']):
        for name in ('image.png','mask.png','mapping.npz'):
            if name.endswith('.npz'):
                with np.load(tmp_path/'new'/actual['directory']/name) as z:
                    assert set(z.files)=={'xyz','mask','source_face_rows'}
            else:assert actual['files'][name]==expected['files'][name]
    # All newly completed tiles can be reused without invoking CT at all.
    shutil.copytree(tmp_path/'new',tmp_path/'all'/'level-0000'/'render')
    shutil.copyfile(root/'IDENTITY.json',tmp_path/'all'/'IDENTITY.json')
    shutil.copyfile(level/'map.npz',tmp_path/'all'/'level-0000'/'map.npz')
    def fail(xyz):pytest.fail('reused tile sampled CT')
    p.render_map(mapped,tmp_path/'again',fail,dict(voxel_um=1.),pitch_um=1.,tile_size=4,
        reuse=reuse(tmp_path/'all',tmp_path/'all'/'level-0000'))


@pytest.mark.parametrize('change',['payload','escape','symlink','render_symlink','canvas','identity','map'])
def test_reuse_rejects_corrupt_or_incompatible_source(tmp_path,change):
    root,level,mapped,sample,result=fixture(tmp_path)
    inventory=reuse(root,level);tile=level/'render'/result['tiles'][0]['directory']
    if change=='payload':(tile/'image.png').write_bytes(b'bad')
    elif change=='escape':
        d=json.loads((tile/'TILE.json').read_text());d['files']['../../outside']='0'*64;atomic_json(tile/'TILE.json',d)
    elif change=='symlink':
        (tile/'image.png').unlink();(tile/'image.png').symlink_to(tile/'mask.png')
    elif change=='render_symlink':
        (level/'render').rename(tmp_path/'redirected');(level/'render').symlink_to(tmp_path/'redirected',target_is_directory=True)
    elif change=='canvas':
        d=json.loads((tile/'TILE.json').read_text());d['canvas']['pitch']=2.;atomic_json(tile/'TILE.json',d)
    elif change=='identity':atomic_json(root/'IDENTITY.json',dict(changed=True))
    elif change=='map':(level/'map.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):
        p.render_map(mapped,tmp_path/'new',sample,dict(voxel_um=1.),pitch_um=1.,tile_size=4,reuse=inventory)


@pytest.mark.parametrize('sampler',[None,'scalar','native-float64'])
def test_continue_export_reuses_map_and_inherits_ct_ledger(tmp_path,monkeypatch,sampler):
    import sqlite3
    from test_patch_pipeline import grid,planar
    from winding_field.surface_mesh import import_mesh
    from winding_field import patch_continue as c
    assert hasattr(c,'continue_export')
    v,f=grid();mesh=tmp_path/'mesh';source=tmp_path/'input.npz';np.savez(source,vertices_xyz=v,faces=f)
    import_mesh(dict(path=str(source),sha256=file_sha(source)),mesh,
        dict(frame='test',voxel_um=9.6,origin_xyz=[0,0,0],spacing_xyz=[32]*3,shape_zyx=[10]*3,sheet_step=1.))
    cfg=p.settings(dict(output_frame=dict(id='test',voxel_um=9.6,origin_xyz_um=[0,0,0]),app_run='/unused',
        pitch_um=9.6,tile_size=4,disk_reserve_gib=.001))
    if sampler is None:cfg.pop('ct_sampler')
    else:cfg['ct_sampler']=sampler
    monkeypatch.setattr(p,'check_flatboi',lambda *args:dict(test=True))
    monkeypatch.setattr(p,'flatten_native',lambda v,f,app,path,**kw:planar(v,f,path))
    class CT:
        identity=dict(source='synthetic',metadata_sha256='a'*64,origin_xyz=[0,0,0],spacing_xyz=[9.6]*3,shape=[10]*3,level=0)
        chunks=(2,2,2);dtype=np.dtype('u1')
        def sample(self,xyz,frame):return xyz[:,0]+2*xyz[:,1],np.ones(len(xyz),bool)
    if sampler=='native-float64':
        CT.identity=dict(CT.identity,sampler='vc_reader-float64-support-v1',sampler_sha256='s'*64,
            extension_sha256='e'*64,reader_libraries_sha256={'native':'n'*64})
    old=tmp_path/'old';old.mkdir();receipt=p.export_mesh(mesh,old/'patches',CT(),cfg)
    if sampler is None:
        identity=json.loads((old/'patches/IDENTITY.json').read_text())
        identity['settings'].pop('ct_sampler')
        atomic_json(old/'patches/IDENTITY.json',identity)
    # The active Paris4 gate persists this before rasterization; LEVEL is absent until completion.
    (old/'patches/level-0000/MAP-REPORT.json').write_text(json.dumps(receipt['levels'][0]['mapping'],separators=(',',':')))
    (old/'patches/level-0000/LEVEL.json').unlink();(old/'patches/PATCHES.json').unlink()
    db=sqlite3.connect(old/'ct-chunks.sqlite');db.execute('CREATE TABLE chunks(key TEXT PRIMARY KEY,sha TEXT)')
    db.execute('INSERT INTO chunks VALUES (?,?)',('0,0,0','b'*64));db.commit();db.close()
    tiles=receipt['levels'][0]['rendering']['tiles'];remove=tiles[-1]['directory']
    with np.load(old/'patches/level-0000/render'/remove/'mapping.npz') as z:expected_mapping={k:z[k] for k in z.files}
    shutil.rmtree(old/'patches/level-0000/render'/remove)
    class NewCT(CT):
        def __init__(self,source,output,cfg,**kwargs):
            assert cfg['ct_sampler']==(sampler or 'scalar')
            self.db=sqlite3.connect(output/'ct-chunks.sqlite');self.db.execute('CREATE TABLE chunks(key TEXT PRIMARY KEY,sha TEXT)')
        def sample(self,xyz,frame):
            assert self.db.execute('SELECT sha FROM chunks WHERE key=?',('0,0,0',)).fetchone()==('b'*64,)
            return super().sample(xyz,frame)
        def close(self):self.db.commit();self.db.close()
    monkeypatch.setattr(p,'PatchCT',NewCT)
    def no_flatten(*a,**kw):pytest.fail('continuation repeated flattening')
    monkeypatch.setattr(p,'flatten_native',no_flatten)
    if sampler=='native-float64':
        NewCT.identity=dict(CT.identity,extension_sha256='changed')
        with pytest.raises(ValueError,match='sampler implementation'):
            c.continue_export(old/'patches',mesh,tmp_path/'changed-reader',
                identity_sha256=file_sha(old/'patches/IDENTITY.json'),overrides={})
        NewCT.identity=CT.identity
    out=tmp_path/'continued'
    result=c.continue_export(old/'patches',mesh,out,identity_sha256=file_sha(old/'patches/IDENTITY.json'),overrides={})
    assert result['status']=='PATCHES_READY_DIAGNOSTIC'
    rendered=result['levels'][0]['rendering']['tiles']
    assert sum('reused_from' in t for t in rendered)==len(tiles)-1
    assert result['ct_ledger_sha256']==file_sha(out/'ct-chunks.sqlite')
    assert (out/'patches/level-0000/MAP-REPORT.json').read_bytes()==(old/'patches/level-0000/MAP-REPORT.json').read_bytes()
    with np.load(old/'patches/level-0000/map.npz') as a,np.load(out/'patches/level-0000/map.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    for before,after in zip(tiles,rendered):
        for name in ('image.png','mask.png'):assert before['files'][name]==after['files'][name]
    with np.load(out/'patches/level-0000/render'/remove/'mapping.npz') as z:
        for key,value in expected_mapping.items():np.testing.assert_array_equal(z[key],value)
    with pytest.raises(ValueError,match='settings'):
        c.continue_export(old/'patches',mesh,tmp_path/'wrong',identity_sha256=file_sha(old/'patches/IDENTITY.json'),overrides={'pitch_um':1.})
    cancelled=[False];cancel_out=tmp_path/'cancel'
    def hashing(path):
        path=Path(path)
        sha=file_sha(path)
        if path.name=='LEVEL.json' and cancel_out in path.parents:cancelled[0]=True
        return sha
    monkeypatch.setattr(c,'file_sha',hashing)
    with pytest.raises(InterruptedError):
        c.continue_export(old/'patches',mesh,cancel_out,identity_sha256=file_sha(old/'patches/IDENTITY.json'),
            overrides={},stop=lambda:cancelled[0])
    assert not (cancel_out/'PATCHES.json').exists()


def test_copy_admission_preserves_disk_reserve(tmp_path,monkeypatch):
    from winding_field import patch_continue as c
    from types import SimpleNamespace
    source=tmp_path/'data';source.write_bytes(b'1234')
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=11))
    assert hasattr(c,'copy_verified_file')
    with pytest.raises(RuntimeError,match='disk reserve'):
        c.copy_verified_file(source,tmp_path/'copy',guard=lambda:None,reserve_bytes=8)
    assert not (tmp_path/'copy').exists()


def test_map_admission_checks_headers_before_loading(tmp_path):
    from winding_field import patch_continue as c
    assert hasattr(c,'map_array_bytes')
    path=tmp_path/'map.npz';a=np.zeros((100,3),'f8');b=np.zeros(100,'i8')
    np.savez_compressed(path,world=a,source_face_ids=b)
    assert c.map_array_bytes(path)==a.nbytes+b.nbytes
