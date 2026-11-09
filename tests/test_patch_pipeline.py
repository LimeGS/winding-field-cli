"""Patch adapter contracts; only the unavailable external executable is replaced."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pipeline():
    assert importlib.util.find_spec('winding_field.patch_pipeline'), 'patch adapter is missing'
    from winding_field import patch_pipeline
    return patch_pipeline


def grid():
    x,y=np.meshgrid(np.arange(9.),np.arange(9.))
    v=np.column_stack([x.ravel()+10,y.ravel()+10,np.full(x.size,3.)])
    f=[]
    for j in range(8):
        for i in range(8):
            a=j*9+i;f.extend([[a,a+1,a+10],[a,a+10,a+9]])
    return v,np.asarray(f)


def planar(v,f,path):
    return v[:,:2].copy(),{'status':'TEST_PLANAR_EXTERNAL_BOUNDARY'}


def test_original_id_seam_crosses_angular_branch():
    p=pipeline()
    v=np.array([[-3,-1,0],[-3,0,0],[-3,1,0],[-3,0,1]],float)
    charts=[dict(v=v[[0,1,3]],f=np.array([[0,1,2]]),vi=np.array([91,92,94])),
            dict(v=v[[1,2,3]],f=np.array([[0,1,2]]),vi=np.array([92,93,94]))]
    links,counts=p.seam_links(charts)
    assert counts['shared_edges']==1 and len(links)==1
    assert np.array_equal(charts[0]['vi'][links[0]['vi']],charts[1]['vi'][links[0]['vj']])
    charts[1]['vi']+=1000
    assert p.seam_links(charts)[0]==[]  # coincident XYZ alone is not provenance


def test_all_components_patch_recovery_and_exact_xyz(tmp_path):
    p=pipeline();v,f=grid()
    v=np.concatenate([v,v+[30,0,0]]);f=np.concatenate([f,f+81])
    result=p.build_map(v,f,tmp_path,flatten=planar,max_patch_faces=32)
    assert result['report']['source_components']==2
    assert result['report']['omitted_uv_faces']==0
    assert np.array_equal(result['world'],v[result['parent']])
    assert np.array_equal(result['parent'][result['faces']],f[result['source_face_rows']])
    assert sorted(result['source_face_rows'])==list(range(len(f)))
    assert result['report']['seams']['shared_edges']>0
    assert len(result['parent'])==len(np.unique(result['parent']))  # real shared connectivity for TIFXYZ
    for k in np.unique(result['parent']):
        assert np.ptp(result['uv'][result['parent']==k],axis=0).max()<1e-8


def test_flatten_failure_accounts_all_faces(tmp_path):
    p=pipeline();v,f=grid()
    def fail(v,f,path):return None,{'status':'FAIL','reasons':['fixture failure']}
    result=p.build_map(v,f,tmp_path,flatten=fail,max_patch_faces=32,max_depth=1)
    assert result['report']['omitted_uv_faces']==len(f)
    assert result['report']['omitted_uv_area_world2']==pytest.approx(64.)
    assert sorted(result['omitted_face_rows'])==list(range(len(f)))


def test_raster_tifxyz_real_readback_scale_holes_and_tiling(tmp_path):
    p=pipeline();v,f=grid()
    # Hole in the physical source, never fill it during patch registration/raster.
    centers=v[f].mean(1);f=f[~((centers[:,0]>13)&(centers[:,0]<15)&(centers[:,1]>13)&(centers[:,1]<15))]
    mapped=p.build_map(v,f,tmp_path/'map',flatten=planar,max_patch_faces=256)
    def sample(xyz):return xyz[:,0]+2*xyz[:,1],xyz[:,0]<17
    frame=dict(id='synthetic',voxel_um=9.6,origin_xyz_um=[0,0,0])
    report=p.render_map(mapped,tmp_path/'render',sample,frame,pitch_um=9.6,tile_size=8,export_tifxyz=True)
    assert len(report['tiles'])>1
    assert report['CT_missing_pixels']>0
    retained=0
    for tile in report['tiles']:
        root=tmp_path/'render'/tile['directory'];meta=json.loads((root/'TILE.json').read_text())
        assert meta['canvas']['pitch']==1
        if not (root/'tifxyz/meta.json').exists():continue
        xyz=np.stack([np.asarray(Image.open(root/'tifxyz'/f'{a}.tif')) for a in 'xyz'],-1)
        valid=np.asarray(Image.open(root/'tifxyz/valid-mask.png'))>0
        mask=np.asarray(Image.open(root/'mask.png'));im=np.asarray(Image.open(root/'image.png'))
        assert valid.any();retained+=valid.sum()
        assert np.allclose(xyz[valid,2],3.)
        assert not ((xyz[valid,0]>13)&(xyz[valid,0]<15)&(xyz[valid,1]>13)&(xyz[valid,1]<15)).any()
        observed=valid&(mask==2)
        assert np.array_equal(im[observed],np.rint(xyz[observed,0]+2*xyz[observed,1]).astype('uint8'))
        assert json.loads((root/'tifxyz/meta.json').read_text())['pitch_um']==9.6
    assert retained>0


def test_dependency_pin_fails_before_output(tmp_path):
    p=pipeline();app=tmp_path/'AppRun';app.write_text('wrong')
    with pytest.raises(ValueError,match='launcher SHA256'):
        p.check_flatboi(app)


def test_physical_pitch_validation():
    p=pipeline()
    cfg=dict(output_frame=dict(id='L2',voxel_um=9.6,origin_xyz_um=[0,0,0]),app_run='/missing')
    assert p.settings(cfg)['pitch_um']==19.2
    with pytest.raises(ValueError,match='pitch'):
        p.settings(dict(cfg,pitch=307.2))


def test_cli_patches_help(capsys):
    from winding_field.cli import main
    with pytest.raises(SystemExit) as error:main(['export-sheets','--help'])
    assert error.value.code==0
    assert 'patches' in capsys.readouterr().out


def test_range_input_and_detached_guard(tmp_path):
    p=pipeline()
    from test_range_surface import recipe
    from winding_field.range_surface import extract_range
    root=tmp_path/'range';extract_range(recipe(tmp_path),[-61,-59],root)
    assert hasattr(p,'open_mesh'), 'validated patch mesh adapter missing'
    with p.open_mesh(root) as (reader,identity):
        assert identity['kind']=='continuous_range' and 'w_range' in identity
        v,f,vi,fi,vk,fk=p.read_level(reader,0,memory_gib=.1)
        assert len(f)>0 and len(vk)==len(v) and len(fk)==len(f)
        assert len(set(vk))==len(vk)
    with pytest.raises(ValueError,match='wrapper'):
        with p.open_mesh(root/'mesh'):pass


def test_export_mesh_real_core_and_source_ids(tmp_path,monkeypatch,capsys):
    p=pipeline()
    assert hasattr(p,'export_mesh'), 'patch publication missing'
    from winding_field.surface_mesh import import_mesh
    v,f=grid();root=tmp_path/'mesh'
    metadata=dict(frame='test',voxel_um=9.6,origin_xyz=[0,0,0],spacing_xyz=[32]*3,shape_zyx=[10]*3,sheet_step=1.)
    from winding_field.util import file_sha
    source=tmp_path/'input.npz';np.savez(source,vertices_xyz=v,faces=f)
    import_mesh(dict(path=str(source),sha256=file_sha(source)),root,metadata)
    cfg=p.settings(dict(output_frame=dict(id='test',voxel_um=9.6,origin_xyz_um=[0,0,0]),app_run='/unavailable',pitch_um=9.6,tile_size=32,disk_reserve_gib=.001))
    monkeypatch.setattr(p,'check_flatboi',lambda app,recipe:dict(test_external=True))
    monkeypatch.setattr(p,'flatten_native',lambda v,f,app,path,**kw:planar(v,f,path))
    class CT:
        identity={'synthetic':True}
        chunks=(2,2,2);dtype=np.dtype('uint8')
        def sample(self,xyz,frame):return xyz[:,0]+2*xyz[:,1],np.ones(len(xyz),bool)
    out=tmp_path/'out';receipt=p.export_mesh(root,out,CT(),cfg,export_tifxyz=True)
    assert receipt['status']=='PATCHES_READY_DIAGNOSTIC'
    with np.load(out/'level-0000/map.npz') as m:
        assert np.array_equal(m['field_xyz'],v[m['source_vertex_ids']-1])
        assert np.array_equal(np.sort(m['source_face_ids']),np.arange(1,len(f)+1))
    assert (out/'level-0000/render/tile-000000000-000000000/tifxyz/VERIFIED.json').exists()
    class CLICT(CT):
        def __init__(self,source,output,cfg,**kwargs):
            (output/'ct-chunks.sqlite').write_bytes(b'synthetic transport fixture')
        def close(self):pass
    monkeypatch.setattr(p,'PatchCT',CLICT)
    config=tmp_path/'patch.json';config.write_text(json.dumps(cfg))
    from winding_field.cli import main
    assert main(['export-sheets','--mesh-bundle',str(root),'--source','synthetic',
        '--flattening-method','patches','--flattening-config',str(config),'--output',str(tmp_path/'cli'),
        '--export-tifxyz'])==0
    assert json.loads(capsys.readouterr().out)['status']=='PATCHES_READY_DIAGNOSTIC'
    assert (tmp_path/'cli/patches/level-0000/render/tile-000000000-000000000/tifxyz/VERIFIED.json').is_file()


def test_native_ct_positive_weight_missing_and_hash_change(tmp_path):
    p=pipeline()
    import sqlite3
    from types import SimpleNamespace
    ct=object.__new__(p.PatchCT)
    from collections import OrderedDict
    import threading
    ct._lock=threading.RLock();ct._chunks=OrderedDict();ct._cache_bytes=0;ct._cache_limit=0
    ct._cache_path=tmp_path;ct._disk_reserve=0
    ct.db=sqlite3.connect(tmp_path/'chunks.sqlite',check_same_thread=False);ct.db.execute('CREATE TABLE chunks(key TEXT PRIMARY KEY,sha TEXT)')
    ct.volume=SimpleNamespace(origin_xyz=np.array([20.,30.,40.]),spacing_xyz=np.array([2.,3.,4.]),
        shape=(4,4,4),chunks=(2,2,2),dtype=np.dtype('uint8'),metadata={'order':'C'})
    changed=False
    def load(index):
        z,y,x=np.indices((2,2,2));z+=index[0]*2;y+=index[1]*2;x+=index[2]*2
        return None if tuple(index)==(1,1,1) else (x+2*y+3*z+int(changed)).astype('uint8').tobytes()
    ct._load_chunk=load
    from winding_field.native_ct_sampling import NativeCTSampler
    ct._sampler=NativeCTSampler(ct.volume.shape,ct.volume.chunks,ct.volume.dtype,ct._chunk)
    frame=dict(voxel_um=1.,origin_xyz_um=[0,0,0])
    xyz=np.array([[20,30,40],[21,31.5,42],[22,33,44],[23,34.5,46]])
    values,valid=ct.sample(xyz,frame)
    assert np.array_equal(valid,[True,True,True,False])
    np.testing.assert_allclose(values[:3],[0,3,6]);assert np.isnan(values[3])
    changed=True
    with pytest.raises(ValueError,match='CT chunk changed'):ct.sample(xyz[:1],frame)
    ct._sampler.close();ct.db.close()


def test_native_ct_verified_cache_reuses_and_detects_change_after_eviction(tmp_path,monkeypatch):
    from collections import OrderedDict
    from types import SimpleNamespace
    import sqlite3,threading
    p=pipeline();ct=object.__new__(p.PatchCT)
    ct._lock=threading.RLock();ct._chunks=OrderedDict();ct._cache_bytes=0;ct._cache_limit=520
    ct._cache_path=tmp_path;ct._disk_reserve=0
    ct.db=sqlite3.connect(tmp_path/'chunks.sqlite');ct.db.execute('CREATE TABLE chunks(key TEXT PRIMARY KEY,sha TEXT)')
    ct.volume=SimpleNamespace(chunks=(2,2,2),dtype=np.dtype('u1'),metadata={'order':'C'})
    loads=[];value=[1]
    def load(index):
        loads.append(index)
        return bytes([value[0]])*8
    ct._load_chunk=load
    a=ct._chunk((0,0,0));b=ct._chunk((0,0,0))
    assert a is b and len(loads)==1 and not a.flags.writeable
    ct._chunk((0,0,1));assert ct._cache_bytes<=ct._cache_limit
    value[0]=2
    with pytest.raises(ValueError,match='CT chunk changed'):ct._chunk((0,0,0))
    before=len(loads)
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match='CT cache disk reserve'):ct._chunk((1,0,0))
    assert len(loads)==before
    ct.db.close()


def test_mixed_uv_orientation_records_excluded_face(tmp_path):
    p=pipeline();v=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0]],float);f=np.array([[0,1,2],[0,2,3]])
    def folded(v,f,path):return np.array([[0,0],[2,0],[2,2],[2,1]],float),{}
    result=p.build_map(v,f,tmp_path,flatten=folded)
    assert result['report']['omitted_uv_faces']==1
    assert result['report']['omitted_uv_area_world2']==2
    assert result['source_face_rows'][~result['display']].tolist()==[1]
    assert result['omitted_face_rows'].tolist()==[]  # retained as a placed blocker


def test_frame_scaled_seam_regularization_is_identical():
    p=pipeline()
    assert 'world_unit_um' in __import__('inspect').signature(p.seam_targets).parameters, 'seam regularization still tied to hardcoded L2 units'
    v=np.array([[0.,0,0],[10.,0,0],[0,10.,0]])
    charts=[dict(v=v,uv=v[:,:2],edges=np.array([[0,1],[1,2],[2,0]])),
            dict(v=v,uv=v[:,:2]*1.1,edges=np.array([[0,1],[1,2],[2,0]]))]
    links=[dict(i=0,j=1,vi=np.arange(3),vj=np.arange(3))];poses={0:np.zeros(3),1:np.zeros(3)}
    a=p.seam_targets(charts,links,[0,1],poses,world_unit_um=9.6)
    scaled=[dict(c,v=c['v']*9.6,uv=c['uv']*9.6) for c in charts]
    b=p.seam_targets(scaled,links,[0,1],poses,world_unit_um=1.)
    np.testing.assert_allclose(np.array(list(a[0].values()))*9.6,np.array(list(b[0].values())),atol=1e-12)


def test_memory_budget_includes_raster_and_cache():
    p=pipeline()
    with pytest.raises(ValueError,match='memory'):
        p.settings(dict(output_frame=dict(id='x',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/missing',memory_gib=.01,tile_size=2048,cache_mib=64))


def test_folded_footprint_blocks_ct_and_tifxyz_after_placement(tmp_path):
    p=pipeline()
    v=np.array([[10,10,1],[12,10,1],[12,12,1],[10,12,1]],float)
    f=np.array([[0,1,2],[0,2,3]])
    uv=np.array([[0,0],[2,0],[2,2],[2,1]],float)
    # Force a nonzero final placement with a preceding disconnected component.
    vertices=np.r_[v+[20,0,0],v];faces=np.r_[f,f+4]
    mapped=p.build_map(vertices,faces,tmp_path/'map',flatten=lambda *args:(uv.copy(),{}))
    assert mapped['report']['omitted_uv_faces']==2
    frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0])
    report=p.render_map(mapped,tmp_path/'render',lambda xyz:(np.full(len(xyz),100.),np.ones(len(xyz),bool)),
                        frame,pitch_um=.125,tile_size=64,export_tifxyz=True)
    tile=report['tiles'][0];root=tmp_path/'render'/tile['directory'];canvas=tile['canvas']
    mask=np.asarray(Image.open(root/'mask.png'));image=np.asarray(Image.open(root/'image.png'))
    valid=(np.asarray(Image.open(root/'tifxyz/valid-mask.png'))>0
           if (root/'tifxyz/valid-mask.png').exists() else np.zeros(mask.shape,bool))
    for placement in mapped['report']['registrations']:
        u,w=np.array([1.5,1.])+np.array(placement['translation'])
        row,col=round((canvas['v_top']-w)/canvas['pitch']),round((u-canvas['u0'])/canvas['pitch'])
        assert mask[row,col]==3 and image[row,col]==0 and not valid[row,col]
    assert report['ambiguous_pixels']>0


@pytest.mark.parametrize('pitch',[1.,.3])
def test_tifxyz_halo_matches_single_canvas_and_neighbors(tmp_path,pitch):
    p=pipeline();x,y=np.meshgrid(np.arange(10.),np.arange(10.))
    v=np.column_stack([10+x.ravel(),10+y.ravel(),np.ones(100)])
    f=[]
    for row in range(9):
        for col in range(9):
            if (row,col)==(4,4):continue
            a=row*10+col;f.extend([[a,a+1,a+11],[a,a+11,a+10]])
    f=np.array(f);mapped=dict(world=v,faces=f,uv=v[:,:2]-10,display=np.ones(len(f),bool),source_face_rows=np.arange(len(f)))
    frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0])
    sample=lambda xyz:(np.full(len(xyz),100.),np.ones(len(xyz),bool))
    full=p.render_map(mapped,tmp_path/'full',sample,frame,pitch_um=pitch,tile_size=64,export_tifxyz=True)
    tiled=p.render_map(mapped,tmp_path/'tiles',sample,frame,pitch_um=pitch,tile_size=5,export_tifxyz=True)
    fullroot=tmp_path/'full'/full['tiles'][0]['directory']/'tifxyz'
    reference=np.asarray(Image.open(fullroot/'valid-mask.png'))>0
    reference_xyz=np.stack([np.asarray(Image.open(fullroot/f'{a}.tif')) for a in 'xyz'],-1)
    for tile in tiled['tiles']:
        root=tmp_path/'tiles'/tile['directory']/'tifxyz';canvas=tile['canvas']
        row=round((full['tiles'][0]['canvas']['v_top']-canvas['v_top'])/pitch)
        col=round((canvas['u0']-full['tiles'][0]['canvas']['u0'])/pitch)
        expected=reference[row:row+canvas['height'],col:col+canvas['width']]
        actual=np.asarray(Image.open(root/'valid-mask.png'))>0 if (root/'valid-mask.png').exists() else np.zeros(expected.shape,bool)
        np.testing.assert_array_equal(actual,expected,err_msg=tile['directory'])
        if actual.any():
            xyz=np.stack([np.asarray(Image.open(root/f'{a}.tif')) for a in 'xyz'],-1)
            np.testing.assert_array_equal(xyz[actual],reference_xyz[row:row+canvas['height'],col:col+canvas['width']][actual])
            meta=json.loads((root/'meta.json').read_text());assert meta['canvas']==canvas


def test_ct_chunk_admission_precedes_native_transport(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    p=pipeline();calls=[]
    class Metadata:
        chunks=(512,512,512);dtype=np.dtype('<u2');source='fake';array_path='';metadata={};identity={}
        def __init__(self,*args,**kw):calls.append('metadata')
        def close(self):calls.append('close')
    monkeypatch.setattr('winding_field.io.Volume',Metadata)
    def transport(*args,**kw):
        calls.append('transport');raise AssertionError('native transport must not start')
    monkeypatch.setitem(sys.modules,'vc_reader',SimpleNamespace(_reader=SimpleNamespace(__file__=__file__)))
    monkeypatch.setitem(sys.modules,'vc_reader.reader',SimpleNamespace(open_zarr=transport))
    cfg=p.settings(dict(output_frame=dict(id='x',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/missing',memory_gib=.2,tile_size=8,cache_mib=0))
    with pytest.raises(ValueError,match='CT.*memory'):
        p.PatchCT('fake',tmp_path,cfg)
    assert calls==['metadata','close']


def test_folded_grid_has_valid_witness_but_no_tifxyz_in_fold(tmp_path):
    p=pipeline();v,f=grid()
    v=np.r_[v,[[9.,14.,3.]]];f=np.r_[f,[[0,9,81]]]
    uv=v[:,:2]-10;uv[-1]=[6,4]
    mapped=p.build_map(v,f,tmp_path/'map',flatten=lambda *args:(uv.copy(),{}))
    frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0])
    report=p.render_map(mapped,tmp_path/'render',lambda xyz:(np.full(len(xyz),100.),np.ones(len(xyz),bool)),
                        frame,pitch_um=.25,tile_size=64,export_tifxyz=True)
    root=tmp_path/'render'/report['tiles'][0]['directory']
    mask=np.asarray(Image.open(root/'mask.png'));valid=np.asarray(Image.open(root/'tifxyz/valid-mask.png'))>0
    # Interior of folded triangle (u=3,v=2.25), well away from source boundary rectangles.
    row,col=23,12
    assert mask[row,col]==3 and not valid[row,col]
    assert valid[8,24] and mask[8,24]==2


def test_ct_workspace_is_subtracted_from_geometry_admission(tmp_path,monkeypatch):
    p=pipeline()
    from types import SimpleNamespace
    from contextlib import contextmanager
    cfg=p.settings(dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/missing',memory_gib=1.,tile_size=8,cache_mib=0,disk_reserve_gib=.001))
    ct=SimpleNamespace(chunks=(256,256,256),dtype=np.dtype('u1'),identity={})
    class StopProbe(Exception):pass
    class DB:
        def execute(self,*args):return [(0,0.)]
    @contextmanager
    def reader(path):yield SimpleNamespace(manifest_sha='test',db=DB()),{}
    def admission(reader,level,*,memory_gib):
        assert memory_gib==pytest.approx(1-(p.raster_memory(cfg)+4*256**3)/2**30)
        raise StopProbe
    monkeypatch.setattr(p,'check_flatboi',lambda app,recipe:{})
    monkeypatch.setattr(p,'open_mesh',reader);monkeypatch.setattr(p,'read_level',admission)
    with pytest.raises(StopProbe):p.export_mesh(tmp_path/'mesh',tmp_path/'out',ct,cfg)


def test_explicit_installed_recipe_preserves_historical_pin(tmp_path,monkeypatch):
    p=pipeline();app=tmp_path/'AppRun';app.write_text('launcher');app.chmod(0o700)
    binary=tmp_path/'usr/bin/flatboi';binary.parent.mkdir(parents=True);binary.write_text('binary')
    installed='2f89bfc96512bebf7ef44b25de01c5afd01c7af058e452515b03bc52cf6afd6a'
    monkeypatch.setattr(p,'file_sha',lambda path:p.LAUNCHER_SHA if Path(path)==app else installed)
    with pytest.raises(ValueError,match='binary SHA256'):p.check_flatboi(app)
    cfg=dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run=str(app))
    assert p.settings(cfg).get('native_recipe')=='historical', 'explicit recipe identity missing'
    chosen=p.settings(dict(cfg,native_recipe='installed-2f89bfc9'))
    receipt=p.check_flatboi(app,chosen['native_recipe'])
    assert receipt['recipe']=='installed-2f89bfc9' and receipt['binary_sha256']==installed
    assert receipt['historical_reproduction'] is False
    with pytest.raises(ValueError,match='native_recipe'):p.settings(dict(cfg,native_recipe='unverified'))
    monkeypatch.setattr(p,'file_sha',lambda path:p.LAUNCHER_SHA if Path(path)==app else '0'*64)
    with pytest.raises(ValueError,match='binary SHA256'):p.check_flatboi(app,'installed-2f89bfc9')
