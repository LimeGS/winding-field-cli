import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ct_fixture import write_ct
from test_uvw import field_array
from winding_field.util import atomic_json,file_sha


def inputs(tmp_path,*,origin=(10.,20.,30.),output_origin=None,output_voxel_um=2.):
    field=field_array(tmp_path,lambda p:p[...,2],origin=origin)
    descriptor=field.source.identity;field.close()
    config=tmp_path/'field.json';atomic_json(config,descriptor)
    source=write_ct(tmp_path/'ct.zarr',np.full((12,12,12),90,np.uint8),spacing=(1.,1.,1.),origin=origin)
    from winding_field.io import Volume
    volume=Volume(source);ct=volume.identity;volume.close()
    root=tmp_path/'reference'/'original';level=root/'level-0000';level.mkdir(parents=True)
    frame=dict(id='test',origin_xyz_um=list(origin if output_origin is None else output_origin),voxel_um=output_voxel_um)
    world=np.array([[1,1,2.5],[4,1,2.5],[4,4,2.5],[1,4,2.5]],float)
    world*=2./output_voxel_um
    world+=(np.asarray(origin)-frame['origin_xyz_um'])/frame['voxel_um']
    uv=np.array([[1,1],[4,1],[4,4],[1,4]],float)*2./output_voxel_um
    np.savez_compressed(level/'map.npz',world=world,uv=uv,faces=np.array([[0,1,2],[0,2,3]]),
        display=np.ones(2,bool),scientific=np.ones(2,bool),source_vertex_ids=np.arange(4),source_face_ids=np.arange(2))
    for name in ('image.png','mask.png'):Image.fromarray(np.ones((7,7),np.uint8)).save(level/name)
    identity=dict(level=float(origin[2])+5.,level_id=0,mesh_sha256='a'*64,settings=dict(output_frame=frame),ct=ct,export_tifxyz=False)
    record=dict(identity=identity,files={p.name:file_sha(p) for p in level.iterdir()},
        rendering=dict(canvas=dict(width=7,height=7,pitch=1./output_voxel_um,u0=2./output_voxel_um,v_top=8./output_voxel_um)),coverage={})
    atomic_json(level/'LEVEL.json',record)
    with sqlite3.connect(root/'publications.sqlite') as db:
        db.execute('CREATE TABLE levels(name TEXT PRIMARY KEY,receipt_sha TEXT,pending TEXT)')
        db.execute('INSERT INTO levels VALUES(?,?,?)',(level.name,file_sha(level/'LEVEL.json'),'.unused'))
    atomic_json(root/'ORIGINAL.json',dict(level_receipt_sha256={level.name:file_sha(level/'LEVEL.json')}))
    return SimpleNamespace(field_config=str(config),reference_sheet=str(level),levels=[float(origin[2])+i for i in (4.,5.,6.)],source=str(source),
        voxel_um=None,level=0,output=str(tmp_path/'output'),resume=False,canvas_samples=7)


def test_reference_receipt_independent_binding_and_physical_units(tmp_path):
    from winding_field.uvw_pipeline import load_reference
    args=inputs(tmp_path);ref,metadata=load_reference(args.reference_sheet,args.levels,7)
    np.testing.assert_allclose(ref.world[0],[12.,22.,35.]);np.testing.assert_allclose(ref.uv[0],[2.,2.])
    assert metadata['canvas']==dict(width=7,height=7,pitch_um=1.,u0_um=2.,v_top_um=8.)
    path=tmp_path/'reference/original/level-0000/LEVEL.json';record=json.loads(path.read_text())
    record['identity']['level']=34.;atomic_json(path,record)
    with pytest.raises(ValueError,match='receipt'):load_reference(args.reference_sheet,args.levels,7)


def test_end_to_end_repeat_resume_units_row_direction(tmp_path):
    import tifffile
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);result=export_existing(args)
    assert result['status']=='UVW_READY_DIAGNOSTIC';assert not result['physical_acceptance']
    out=tmp_path/'output';first=out/'level-0000'
    x=tifffile.imread(first/'tifxyz/x.tif');y=tifffile.imread(first/'tifxyz/y.tif');z=tifffile.imread(first/'tifxyz/z.tif')
    assert all(a.dtype==np.float64 for a in (x,y,z))
    with tifffile.TiffFile(first/'tifxyz/x.tif') as tif:
        assert tif.pages[0].bitspersample==64 and tif.pages[0].sampleformat==3
    assert result['identity']['schema']=='aligned-uvw-v2'
    assert result['identity']['coordinate_dtype']=='float64'
    meta=json.loads((first/'tifxyz/meta.json').read_text())
    assert meta['schema']=='aligned-uvw-tifxyz-v2' and meta['coordinate_dtype']=='float64'
    np.testing.assert_allclose(x[3],[1.,1.5,2.,2.5,3.,3.5,4.])
    np.testing.assert_allclose(y[:,3],[4.,3.5,3.,2.5,2.,1.5,1.]);np.testing.assert_allclose(z,2.)
    assert np.asarray(Image.open(first/'geometry-mask.png')).all()
    assert np.asarray(Image.open(first/'ct-mask.png')).all()
    args.resume=True;assert export_existing(args)==result
    (first/'tifxyz/x.tif').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='product'):export_existing(args)


def test_interrupted_resume_and_changed_settings(tmp_path):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path)
    result=export_existing(args,stop=lambda:True);assert result['status']=='PAUSED'
    args.resume=True;args.canvas_samples=8
    with pytest.raises(ValueError,match='identity'):export_existing(args)
    args.canvas_samples=7;assert export_existing(args)['status']=='UVW_READY_DIAGNOSTIC'


def test_missing_ct_keeps_geometry_and_binds_availability(tmp_path):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path)
    chunks=[p for p in (tmp_path/'ct.zarr').rglob('*') if p.is_file() and p.name[0].isdigit()]
    saved=[(p,p.read_bytes()) for p in chunks]
    for p,_ in saved:p.unlink()
    result=export_existing(args);assert result['quality']['common_geometry_pixels']==49
    assert result['quality']['common_ct_pixels']==0
    for p,data in saved:p.write_bytes(data)
    args.resume=True
    with pytest.raises(ValueError,match='availability'):export_existing(args)


def test_empty_level_does_not_hide_geometry_in_other_levels(tmp_path):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);args.levels=[35.,50.]
    result=export_existing(args)
    assert result['status']=='UVW_READY_DIAGNOSTIC'
    assert result['quality']['common_geometry_pixels']==0
    assert [level['geometry_pixels'] for level in result['quality']['levels']]==[49,0]


def test_invalid_levels_and_descriptor_fail_before_output(tmp_path):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);args.levels=[float('nan')]
    with pytest.raises(ValueError):export_existing(args)
    assert not (tmp_path/'output').exists()
    args.levels=[34.,35.,36.];config=tmp_path/'field.json';d=json.loads(config.read_text());d['field']['sha256']='0'*64;atomic_json(config,d)
    with pytest.raises(ValueError):export_existing(args)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('descriptor',[{}, {'field':None,'geometry':{}}, {'field':{},'geometry':[]}])
def test_malformed_descriptor_fails_cleanly(tmp_path,descriptor):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);atomic_json(tmp_path/'field.json',descriptor)
    with pytest.raises(ValueError,match='descriptor'):export_existing(args)
    assert not (tmp_path/'output').exists()


def test_export_uvw_cli(tmp_path,capsys):
    from winding_field.cli import main
    args=inputs(tmp_path)
    assert main(['export-uvw','--field-config',args.field_config,'--reference-sheet',args.reference_sheet,
        '--levels','34','35','36','--source',args.source,'--output',args.output,'--canvas-samples','7'])==0


def test_midrun_stop_reuses_checkpoint_arrays(tmp_path):
    import tifffile
    from pathlib import Path
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path)
    def stop():
        journal=Path(args.output)/'publications.sqlite'
        if not journal.exists():return False
        with sqlite3.connect(journal) as db:
            return db.execute("SELECT COUNT(*) FROM batches WHERE key='0-0'").fetchone()[0]>0
    assert export_existing(args,stop=stop)['status']=='PAUSED'
    args.resume=True;export_existing(args)
    resumed=tifffile.imread(Path(args.output)/'level-0000/tifxyz/x.tif')
    args.output=str(tmp_path/'fresh');args.resume=False;export_existing(args)
    np.testing.assert_array_equal(resumed,tifffile.imread(Path(args.output)/'level-0000/tifxyz/x.tif'))


def test_missing_checkpoint_ledger_rejected(tmp_path):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);export_existing(args)
    (tmp_path/'output/publications.sqlite').unlink();args.resume=True
    with pytest.raises(ValueError,match='ledger'):export_existing(args)


def test_reference_inventory_tamper(tmp_path):
    from winding_field.uvw_pipeline import load_reference
    args=inputs(tmp_path)
    (tmp_path/'reference/original/level-0000/map.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='product changed'):load_reference(args.reference_sheet,args.levels,7)


def test_quality_is_bound_and_publication_rename_recovers(tmp_path):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);result=export_existing(args);out=tmp_path/'output'
    assert json.loads((out/'quality.json').read_text())==result['quality']
    with sqlite3.connect(out/'publications.sqlite') as db:
        pending=db.execute("SELECT pending FROM levels WHERE name='level-0000'").fetchone()[0]
    (out/'level-0000').rename(out/pending)
    args.resume=True;assert export_existing(args)==result
    atomic_json(out/'quality.json',{})
    with pytest.raises(ValueError,match='quality'):export_existing(args)


def test_float64_output_frame_avoids_float32_precision_loss(tmp_path):
    from pathlib import Path
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);path=Path(args.reference_sheet);record=json.loads((path/'LEVEL.json').read_text())
    old=np.array(record['identity']['settings']['output_frame']['origin_xyz_um'])
    new=np.full(3,1e12);record['identity']['settings']['output_frame']['origin_xyz_um']=new.tolist()
    with np.load(path/'map.npz') as data:mapped={k:data[k] for k in data.files}
    mapped['world']+=(old-new)/2.;np.savez_compressed(path/'map.npz',**mapped)
    record['files']['map.npz']=file_sha(path/'map.npz');atomic_json(path/'LEVEL.json',record)
    sha=file_sha(path/'LEVEL.json')
    with sqlite3.connect(path.parent/'publications.sqlite') as db:db.execute('UPDATE levels SET receipt_sha=?',(sha,))
    atomic_json(path.parent/'ORIGINAL.json',dict(level_receipt_sha256={path.name:sha}))
    result=export_existing(args)
    assert result['quality']['common_geometry_pixels']==49
    assert result['status']=='UVW_READY_DIAGNOSTIC'
    assert not np.asarray(Image.open(tmp_path/'output/level-0000/cause-mask.png')).any()
    args.resume=True
    assert export_existing(args)==result


@pytest.mark.parametrize('unit',[2.,.12345])
def test_global_float64_tiff_readback_and_completed_resume(tmp_path,unit):
    import tifffile
    from winding_field.uvw_pipeline import export_existing,load_reference
    from winding_field.uvw import xyz_to_uvw
    from winding_field.surface_sources import FieldSource
    from winding_field.tifxyz_field import PhysicalField
    origin=np.array([52000.123456,35000.234567,13000.345678])
    output_origin=np.array([-100000.2345,0.,0.])
    args=inputs(tmp_path,origin=origin,output_origin=output_origin,output_voxel_um=unit);result=export_existing(args)
    out=tmp_path/'output';reference,_=load_reference(args.reference_sheet,args.levels,7)
    reference.band=None  # Export checks requested W separately; encoded endpoints can round across the band.
    descriptor=json.loads((tmp_path/'field.json').read_text())
    with PhysicalField(FieldSource(descriptor['field'],descriptor['geometry'])) as field:
        for index,w in enumerate(args.levels):
            level=out/f'level-{index:04d}'
            encoded=np.stack([tifffile.imread(level/'tifxyz'/f'{a}.tif') for a in 'xyz'],axis=-1)
            assert encoded.dtype==np.float64
            expected=np.empty((7,7,3));expected[:,:,0]=origin[0]+np.arange(2.,9.)
            expected[:,:,1]=origin[1]+np.arange(8.,1.,-1.)[:,None];expected[:,:,2]=w
            np.testing.assert_allclose(encoded*unit+output_origin,expected,rtol=0,atol=1e-10)
            with sqlite3.connect(out/'publications.sqlite') as db:
                checkpoint=db.execute('SELECT path FROM batches WHERE key=?',(f'{index}-0',)).fetchone()[0]
            with np.load(out/checkpoint) as saved:
                np.testing.assert_array_equal(encoded,saved['encoded_xyz'].reshape(7,7,3))
                np.testing.assert_array_equal(encoded*unit+output_origin,saved['xyz'].reshape(7,7,3))
            recovered,valid=xyz_to_uvw(reference,field,encoded.reshape(-1,3)*unit+output_origin)
            assert valid.all()
            np.testing.assert_allclose(recovered[:,2],w,rtol=0,atol=1e-9)
            np.testing.assert_allclose(recovered[:,:2],np.column_stack((np.tile(np.arange(2.,9.),7),
                np.repeat(np.arange(8.,1.,-1.),7))),rtol=0,atol=1e-6)
            assert np.asarray(Image.open(level/'geometry-mask.png')).all()
            assert np.asarray(Image.open(level/'ct-mask.png')).all()
            assert (np.asarray(Image.open(level/'image.png'))==128).all()
    hashes={p.relative_to(out):file_sha(p) for p in out.rglob('*') if p.is_file()}
    args.resume=True;assert export_existing(args)==result
    assert hashes=={p.relative_to(out):file_sha(p) for p in out.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change',['v1','float32','missing_dtype'])
def test_resume_rejects_incompatible_coordinate_identity_without_rewriting(tmp_path,change):
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);export_existing(args);out=tmp_path/'output'
    path=out/'IDENTITY.json';identity=json.loads(path.read_text())
    assert identity['coordinate_dtype']=='float64'
    if change=='v1':identity['schema']='aligned-uvw-v1'
    elif change=='float32':identity['coordinate_dtype']='float32'
    else:identity.pop('coordinate_dtype')
    atomic_json(path,identity)
    hashes={p.relative_to(out):file_sha(p) for p in out.rglob('*') if p.is_file()}
    args.resume=True
    with pytest.raises(ValueError,match='resume identity changed'):export_existing(args)
    assert hashes=={p.relative_to(out):file_sha(p) for p in out.rglob('*') if p.is_file()}


def test_curved_projected_reference_uses_physical_inverse_tolerance(tmp_path):
    from pathlib import Path
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path)
    with field_array(tmp_path,lambda p:p[...,2]+.025*(p[...,0]-10)*(p[...,1]-20),origin=(10.,20.,30.)) as field:
        atomic_json(tmp_path/'field.json',field.source.identity)
    path=Path(args.reference_sheet);record=json.loads((path/'LEVEL.json').read_text())
    with np.load(path/'map.npz') as data:mapped={k:data[k] for k in data.files}
    mapped['world'][:,2]=(5.-.1*mapped['world'][:,0]*mapped['world'][:,1])/2
    np.savez_compressed(path/'map.npz',**mapped);record['files']['map.npz']=file_sha(path/'map.npz')
    atomic_json(path/'LEVEL.json',record);sha=file_sha(path/'LEVEL.json')
    with sqlite3.connect(path.parent/'publications.sqlite') as db:db.execute('UPDATE levels SET receipt_sha=?',(sha,))
    atomic_json(path.parent/'ORIGINAL.json',dict(level_receipt_sha256={path.name:sha}))
    export_existing(args)
    geometry=np.asarray(Image.open(tmp_path/'output/level-0001/geometry-mask.png'))
    assert geometry[4,1]==255  # U=3,V=4; projection is small but normal-line inverse UV is approximate.


def test_real_original_mesh_to_uvw(tmp_path,monkeypatch):
    from pathlib import Path
    from winding_field.surface_sources import FieldSource
    from winding_field.surface_mesh import extract_mesh
    from winding_field.io import Volume
    from winding_field import original_pipeline
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path);descriptor=json.loads(Path(args.field_config).read_text());root=tmp_path/'real-original';root.mkdir()
    with FieldSource(descriptor['field'],descriptor['geometry']) as source:extract_mesh(source,root/'meshes',[35.])
    monkeypatch.setattr(original_pipeline,'available_memory',lambda:16<<30)
    raster=original_pipeline.raster_sheet
    def small(v,f,uv,eligible,sample,canvas,**kwargs):
        canvas.update(width=32,height=32,pitch=canvas['pitch']*4095/31)
        return raster(v,f,uv,eligible,sample,canvas,**kwargs)
    monkeypatch.setattr(original_pipeline,'raster_sheet',small)
    from contextlib import closing
    with closing(Volume(args.source)) as volume:
        original_pipeline.export_mesh(root/'meshes',root/'original',volume,
            dict(output_frame=dict(id='test',origin_xyz_um=[10.,20.,30.],voxel_um=2.)))
    args.reference_sheet=str(root/'original/level-0000');args.canvas_samples=16
    result=export_existing(args)
    assert result['quality']['common_geometry_pixels']>100
