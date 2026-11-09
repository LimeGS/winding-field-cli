import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ct_fixture import write_ct
from test_uvw import field_array
from winding_field.util import atomic_json,file_sha


def inputs(tmp_path):
    field=field_array(tmp_path,lambda p:p[...,2],origin=(10.,20.,30.))
    descriptor=field.source.identity;field.close()
    config=tmp_path/'field.json';atomic_json(config,descriptor)
    source=write_ct(tmp_path/'ct.zarr',np.full((12,12,12),90,np.uint8),spacing=(1.,1.,1.),origin=(10.,20.,30.))
    from winding_field.io import Volume
    volume=Volume(source);ct=volume.identity;volume.close()
    root=tmp_path/'reference'/'original';level=root/'level-0000';level.mkdir(parents=True)
    frame=dict(id='test',origin_xyz_um=[10.,20.,30.],voxel_um=2.)
    world=np.array([[1,1,2.5],[4,1,2.5],[4,4,2.5],[1,4,2.5]],float)
    np.savez_compressed(level/'map.npz',world=world,uv=world[:,:2],faces=np.array([[0,1,2],[0,2,3]]),
        display=np.ones(2,bool),scientific=np.ones(2,bool),source_vertex_ids=np.arange(4),source_face_ids=np.arange(2))
    for name in ('image.png','mask.png'):Image.fromarray(np.ones((7,7),np.uint8)).save(level/name)
    identity=dict(level=35.,level_id=0,mesh_sha256='a'*64,settings=dict(output_frame=frame),ct=ct,export_tifxyz=False)
    record=dict(identity=identity,files={p.name:file_sha(p) for p in level.iterdir()},
        rendering=dict(canvas=dict(width=7,height=7,pitch=.5,u0=1.,v_top=4.)),coverage={})
    atomic_json(level/'LEVEL.json',record)
    with sqlite3.connect(root/'publications.sqlite') as db:
        db.execute('CREATE TABLE levels(name TEXT PRIMARY KEY,receipt_sha TEXT,pending TEXT)')
        db.execute('INSERT INTO levels VALUES(?,?,?)',(level.name,file_sha(level/'LEVEL.json'),'.unused'))
    atomic_json(root/'ORIGINAL.json',dict(level_receipt_sha256={level.name:file_sha(level/'LEVEL.json')}))
    return SimpleNamespace(field_config=str(config),reference_sheet=str(level),levels=[34.,35.,36.],source=str(source),
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


def test_loaded_reference_keeps_source_ids_for_xyz_linkage(tmp_path):
    from winding_field.uvw_pipeline import load_reference
    from winding_field.uvw import xyz_to_uvw
    args=inputs(tmp_path)
    ref,metadata=load_reference(args.reference_sheet,args.levels,7)
    np.testing.assert_array_equal(ref.source_vertex_ids,[0,1,2,3])
    np.testing.assert_array_equal(ref.source_face_ids,[0,1])
    with field_array(tmp_path,lambda p:p[...,2],origin=(10.,20.,30.)) as field:
        uvw,valid,details=xyz_to_uvw(ref,field,[[16.,23.,36.],[30.,23.,36.]],with_details=True)
    assert valid.tolist()==[True,False]
    np.testing.assert_allclose(uvw[0],[6.,3.,36.])
    np.testing.assert_array_equal(details['source_face_id'],[0,-1])
    np.testing.assert_array_equal(details['source_vertex_ids'],[[0,1,2],[-1,-1,-1]])
    np.testing.assert_allclose(details['barycentric'][0]@ref.world[ref.faces[0]],[16.,23.,35.])
    assert metadata['identity']['mesh_sha256']=='a'*64


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


def test_tifxyz_precision_loss_is_masked(tmp_path):
    from pathlib import Path
    from winding_field.uvw_pipeline import export_existing,CAUSES
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
    assert result['quality']['common_geometry_pixels']==0
    assert (np.asarray(Image.open(tmp_path/'output/level-0000/cause-mask.png'))==CAUSES['coordinate_precision']).all()


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
