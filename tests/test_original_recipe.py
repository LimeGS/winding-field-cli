import numpy as np
import pytest

def test_original_raster_missing_and_capture():
    from winding_field.original_recipe import raster_sheet
    v=np.array([[0.,0.,0.],[4.,0.,0.],[0.,4.,0.]])
    f=np.array([[0,1,2]]);seen=[]
    image,mask,_=raster_sheet(v,f,v[:,:2],np.array([True]),
        lambda xyz:(np.full(len(xyz),90.),xyz[:,0]!=1),
        dict(width=5,height=5,pitch=1.,u0=0.,v_top=4.),capture=lambda *a:seen.append(a[0].copy()))
    assert image[4,0]==128 and mask[4,0]==2
    assert image[4,1]==0 and mask[4,1]==1
    assert mask[0,4]==0
    assert np.array_equal(seen[0][4,0],[0.,0.,0.])

def test_original_raster_ambiguous_overlap():
    from winding_field.original_recipe import raster_sheet
    v=np.array([[0.,0.,0.],[4.,0.,0.],[0.,4.,0.],[0.,0.,1.],[4.,0.,1.],[0.,4.,1.]])
    image,mask,_=raster_sheet(v,np.array([[0,1,2],[3,4,5]]),v[:,:2],np.array([True,True]),
        lambda xyz:(np.ones(len(xyz)),np.ones(len(xyz),bool)),dict(width=5,height=5,pitch=1.,u0=0.,v_top=4.))
    assert mask[4,0]==3 and image[4,0]==0

def test_original_method_independent_of_resolution():
    from winding_field.config import validate_config
    from winding_field.workflow import stage_path
    for spacing in (17.28,138.24):
        c=dict(flattening_method='original',field_spacing_um=spacing,surface_output='native',export_tifxyz=True)
        validate_config(c)
        assert stage_path(c)[-2:]==(('mesh','MESHES.json'),('original','ORIGINAL.json'))
    with pytest.raises(ValueError,match='flattening'):
        validate_config(dict(flattening_method='typo'))

def test_original_map_preserves_source_geometry():
    pytest.importorskip('igl')
    from winding_field.original_recipe import make_map
    v=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    f=np.array([[0,1,2],[0,2,3]])
    r=make_map(v,f)
    assert np.array_equal(r['world'],v[r['parent']])
    assert np.array_equal(r['parent'][r['faces']],f[r['source_face_rows']])
    assert r['display'].all()

def test_original_frame_and_admission():
    from winding_field.original_pipeline import frame_coordinates,admit,settings
    frame=dict(id='another-scroll',origin_xyz_um=[10.,20.,30.],voxel_um=2.)
    assert np.array_equal(frame_coordinates(np.array([[12.,24.,36.]]),frame),[[1.,2.,3.]])
    with pytest.raises(ValueError,match='frame'):
        settings(dict(output_frame=dict(frame,voxel_um=0)))
    with pytest.raises(RuntimeError,match='memory'):
        admit(100_000_000,100_000_000,4.)

def test_original_tifxyz_readback(tmp_path):
    from winding_field.original_tifxyz import export_xyz
    from PIL import Image
    yy,xx=np.mgrid[:12,:12];xyz=np.stack([xx,11-yy,np.ones_like(xx)*3],axis=-1).astype(float)
    uv=np.array([[0.,0.],[11.,0.],[11.,11.],[0.,11.]])
    out=tmp_path/'xyz';frame=dict(id='other',origin_xyz_um=[10.,20.,30.],voxel_um=2.)
    r=export_xyz(out,xyz,np.ones((12,12),np.uint8),np.array([[0,1,2],[0,2,3]]),uv,np.ones(2,bool),
        dict(width=12,height=12,pitch=1.,u0=0.,v_top=11.),frame)
    assert r['retained_cells']>0 and r['geometry_mask_includes_CT_missing']
    with Image.open(out/'x.tif') as im:
        data=np.asarray(im)
        assert data.dtype==np.float32 and data[5,5]==5 and data[0,0]==-1
    tifffile=pytest.importorskip('tifffile')
    assert np.array_equal(tifffile.imread(out/'x.tif'),data)
    import json
    export_xyz(tmp_path/'other',xyz, np.ones((12,12),np.uint8),np.array([[0,1,2],[0,2,3]]),uv,np.ones(2,bool),
        dict(width=12,height=12,pitch=1.,u0=0.,v_top=11.),frame,binding={'level':2})
    assert json.loads((out/'meta.json').read_bytes())['uuid']!=json.loads((tmp_path/'other/meta.json').read_bytes())['uuid']

def test_original_ct_positive_weight_support():
    from winding_field.original_recipe import sample_points
    data=np.arange(8,dtype=np.uint8).reshape(2,2,2)
    def chunk(index):
        return data if tuple(index)==(0,0,0) else None
    values,valid=sample_points(np.array([[1.,1.,1.],[1.5,1.,1.],[.5,.5,.5]]),(4,4,4),(2,2,2),chunk)
    assert np.array_equal(valid,[True,False,True])
    assert values[0]==7 and np.isnan(values[1]) and values[2]==3.5

def test_original_bundle_resume_and_tamper(tmp_path,monkeypatch):
    pytest.importorskip('igl')
    import json
    from test_field import prepared_run
    from ct_fixture import write_ct
    from winding_field.field import solve
    from winding_field.meshes import mesh
    from winding_field.io import Volume
    from winding_field import original_pipeline as pipeline
    run=prepared_run(tmp_path/'run');solve(run,{'sweeps':1,'field_chunk':8})
    mesh(run,{'levels':[.5],'mesh_tile':4})
    source=write_ct(tmp_path/'ct.zarr',np.full((6,6,24),90,np.uint8),spacing=(2.,3.,4.),origin=(10.,20.,30.))
    volume=Volume(source,cache=tmp_path/'cache')
    request=dict(output_frame=dict(id='synthetic',origin_xyz_um=[10.,20.,30.],voxel_um=2.),memory_gib=4.)
    monkeypatch.setattr(pipeline,'available_memory',lambda:16<<30)
    # Exercise real mapping/publication without allocating the production 4096 canvas in a unit test.
    raster=pipeline.raster_sheet
    def small(v,f,uv,eligible,sample,canvas,**kwargs):
        canvas.update(width=32,height=32,pitch=canvas['pitch']*4095/31)
        return raster(v,f,uv,eligible,sample,canvas,**kwargs)
    monkeypatch.setattr(pipeline,'raster_sheet',small)
    try:
        out=tmp_path/'output'
        result=pipeline.export_mesh(run/'meshes',out,volume,request,export_tifxyz=True)
        assert result['status']=='ORIGINAL_READY_DIAGNOSTIC'
        assert pipeline.export_mesh(run/'meshes',out,volume,request,export_tifxyz=True)==result
        record=json.loads((out/'level-0000/LEVEL.json').read_bytes())
        assert record['tifxyz']['retained_cells']>0
        receipt=out/'level-0000/LEVEL.json';original=receipt.read_bytes()
        record['coverage']['total_source_faces']+=1
        receipt.write_text(json.dumps(record))
        with pytest.raises(ValueError,match='receipt changed'):
            pipeline.export_mesh(run/'meshes',out,volume,request,export_tifxyz=True)
        receipt.write_bytes(original)
        from winding_field.cli import main
        monkeypatch.setenv('HOME',str(tmp_path))
        config=tmp_path/'original.json';config.write_text(json.dumps(request))
        args=['export-sheets','--mesh-bundle',str(run/'meshes'),'--source',str(source),
              '--output',str(tmp_path/'cli-output'),'--flattening-method','original',
              '--flattening-config',str(config),'--export-tifxyz']
        assert main(args)==0
        assert main(args+['--resume'])==0
        (out/'level-0000/mask.png').write_bytes(b'changed')
        with pytest.raises(ValueError,match='product changed'):
            pipeline.export_mesh(run/'meshes',out,volume,request,export_tifxyz=True)
    finally:volume.close()

def test_original_cli_requires_frame_and_rejects_direct_branch(tmp_path,capsys):
    from winding_field.cli import main
    assert main(['run','--source','unused','--output',str(tmp_path/'out'),'--flattening-method','original'])==1
    assert 'output_frame' in capsys.readouterr().err
    from winding_field.config import validate_config
    with pytest.raises(ValueError,match='original flattening'):
        validate_config(dict(flattening_method='original',surface_output='tifxyz'))

def test_external_world_units_and_missing_ct(monkeypatch):
    from types import SimpleNamespace
    from winding_field.original_pipeline import frame_coordinates,sample_ct
    frame=dict(id='scan',origin_xyz_um=[8.64,17.28,25.92],voxel_um=8.64)
    assert np.allclose(frame_coordinates(np.array([[2.,4.,6.]]),frame,world_unit_um=8.64),[[1.,2.,3.]])

def test_missing_local_ct_is_bound():
    from types import SimpleNamespace
    from winding_field.original_pipeline import sample_ct
    def absent(key):raise FileNotFoundError(key)
    volume=SimpleNamespace(metadata={},array_path='',_chunk_bytes=absent,spacing_xyz=[1.,1.,1.],
                           origin_xyz=[0.,0.,0.],shape=(2,2,2),chunks=(2,2,2))
    missing=set();values,valid=sample_ct(volume,np.array([[0.,0.,0.]]),missing=missing)
    assert not valid[0] and np.isnan(values[0]) and missing=={'0.0.0'}

def test_original_resume_requires_complete_inventory(tmp_path):
    from winding_field.original_pipeline import verify_level
    from winding_field.util import atomic_json
    atomic_json(tmp_path/'LEVEL.json',dict(identity={},files={}))
    with pytest.raises(ValueError,match='inventory'):
        verify_level(tmp_path,{})

@pytest.mark.parametrize('frame',[None,[],{'id':'x','origin_xyz_um':[0,0,0],'voxel_um':True}])
def test_original_rejects_invalid_frame(frame):
    from winding_field.original_pipeline import settings
    with pytest.raises(ValueError,match='frame'):
        settings({'output_frame':frame})

def test_integrated_resume_binds_missing_ct(tmp_path):
    import sqlite3
    from types import SimpleNamespace
    from winding_field.workflow import _validate_products
    from winding_field.util import atomic_json,file_sha
    receipt=tmp_path/'ORIGINAL.json'
    atomic_json(receipt,dict(levels=[dict(unavailable_ct_chunks=['0.0.0'])]))
    with sqlite3.connect(tmp_path/'products.sqlite') as db:
        db.execute('CREATE TABLE exports(path TEXT,sha TEXT)')
    state=dict(stages={'original':dict(receipt='ORIGINAL.json',sha256=file_sha(receipt))})
    with pytest.raises(ValueError,match='availability changed'):
        _validate_products(tmp_path,state,volume=SimpleNamespace(_chunk_bytes=lambda key:b'new CT'))
