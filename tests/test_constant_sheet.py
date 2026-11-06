"""Installed constant-W stages preserve exact geometry and receipt bindings."""
import importlib.util
import json
import sys

import numpy as np
import pytest

from winding_field.util import atomic_json, file_sha


def preparation():
    assert importlib.util.find_spec('winding_field.sheet_prepare'), 'installed preparation is missing'
    from winding_field import sheet_prepare
    return sheet_prepare


def test_render_rejects_old_ct_identity_before_creating_output(tmp_path):
    from winding_field.sheet_finish import render
    source=tmp_path/'validated';source.mkdir()
    for name in ('map.npz','QUALITY.json','PREPARED.json','disconnected-source.npz'):
        (source/name).write_bytes(b'not-needed-before-identity-check')
    atomic_json(source/'MESH.json',dict(metadata=dict(frame='test',voxel_um=1.)))
    files={p.name:file_sha(p) for p in source.iterdir()}
    atomic_json(source/'VALIDATED.json',dict(status='MAP_VALIDATED_DIAGNOSTIC',overlap={'complete':True},files=files))
    prior=tmp_path/'old.json';atomic_json(prior,dict(ct={'backend':'vc_reader'}))
    config=dict(ct=dict(source='/must-not-open'),prior_identity=str(prior),settings=dict(
        output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/unused'))
    with pytest.raises(ValueError,match='v1.5.0'):
        render(source,tmp_path/'output',config)
    assert not (tmp_path/'output').exists()


def extracted(tmp_path):
    from test_surface_mesh import source
    from winding_field.surface_mesh import extract_mesh
    values=np.broadcast_to(np.arange(7.),(7,7,7)).copy()
    mask=np.ones(values.shape,bool);mask[:,4,:]=False
    with source(tmp_path,values,mask,origin=(0,0,0),spacing=(1,1,1)) as field:
        extract_mesh(field,tmp_path/'mesh',[3.5],tile=3)
    return tmp_path/'mesh'


def test_reference_cut_preserves_every_oriented_source_face():
    p=preparation()
    v=np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    cut,report=p.reference_cut(v,f,[[0,0,-1],[0,0,1]])
    np.testing.assert_array_equal(cut['vertices'],v[cut['parent']])
    np.testing.assert_array_equal(cut['parent'][cut['faces']],f)
    assert report['topology']['cut']['euler_characteristic']==1
    assert 0 in report['reference_path']
    with pytest.raises(ValueError,match='centerline'):
        p.reference_cut(v,f,[[0,0,1],[0,0,-1]])


def test_reference_cut_keeps_an_existing_disk_connected():
    p=preparation()
    v=np.array([[10,-1,0],[10,0,-1],[10,1,0],[10,0,1],[10,0,0]],dtype=float)
    f=np.array([[1,2,4],[2,3,4],[3,0,4],[0,1,4]],dtype=np.int64)
    cut,report=p.reference_cut(v,f,[[0,0,-2],[0,0,2]])
    np.testing.assert_array_equal(cut['vertices'],v[cut['parent']])
    np.testing.assert_array_equal(cut['parent'][cut['faces']],f)
    assert len(cut['vertices'])==5
    assert report['topology']['cut']['euler_characteristic']==1


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_prepare_accounts_disconnected_geometry_and_rejects_tampering(tmp_path):
    p=preparation();mesh=extracted(tmp_path);out=tmp_path/'prepared'
    result=p.prepare(mesh,out,3.5,[[0,0,0],[0,0,6]])
    assert result['principal_faces']>result['disconnected_faces']>0
    assert result['source_faces']==result['principal_faces']+result['disconnected_faces']
    assert result['world_bounds'][0][2]==0 and result['world_bounds'][1][2]==6
    report,mapped=p.load_prepared(out)
    assert report['w']==3.5 and len(mapped['faces'])==result['principal_faces']
    assert report['centerline']==[[0,0,0],[0,0,6]]
    with pytest.raises(FileExistsError):p.prepare(mesh,out,3.5,report['centerline'])
    (out/'inputmesh.obj').write_text('tampered')
    with pytest.raises(ValueError,match='hash'):p.load_prepared(out)


def test_wrong_w_and_invalid_reference_do_not_publish(tmp_path):
    p=preparation();mesh=extracted(tmp_path)
    for w,cp in ((9,[[0,0,0],[0,0,6]]),(3.5,[[0,0,6],[0,0,0]]),(float('nan'),[[0,0,0],[0,0,6]])):
        with pytest.raises(ValueError):p.prepare(mesh,tmp_path/'invalid',w,cp)
    assert not (tmp_path/'invalid'/'PREPARED.json').exists()


def finishing():
    assert importlib.util.find_spec('winding_field.sheet_finish'), 'installed validation/render is missing'
    from winding_field import sheet_finish
    return sheet_finish


def test_isotropic_calibration_z_orientation_and_exact_anchor():
    p=finishing()
    v=np.array([[1.,0,0],[2.,0,0],[2.,0,4],[1.,0,4]])
    f=np.array([[0,1,2],[0,2,3]]);uv=v[:,[0,2]]
    initial=3*(uv@np.array([[0.,1.],[-1.,0.]]))+42
    result,report=p.calibrate(v,f,initial)
    np.testing.assert_allclose(result-result[0],uv-uv[0],atol=1e-12)
    assert report['z_correlation']>.999999
    anchor=p.reference_anchor(v,f,uv,[[0,0,0],[0,0,4]],2.)
    np.testing.assert_array_equal(anchor['xyz'],[1,0,2])
    np.testing.assert_array_equal(anchor['uv'],[1,2])
    with pytest.raises(ValueError):p.reference_anchor(v,f,uv,[[0,0,0],[0,0,4]],10.)


def native_fixture(prepared,tmp_path,recipe='installed-2f89bfc9'):
    from winding_field import flatboi_checkpoint as r
    from winding_field.patch_pipeline import LAUNCHER_SHA,NATIVE_RECIPES
    record,mapped=preparation().load_prepared(prepared)
    root=tmp_path/'native';root.mkdir();batch=root/'batch-fixture';batch.mkdir()
    identity=dict(source_map_sha256=file_sha(prepared/'initial-map.npz'),source_obj_sha256=file_sha(prepared/'inputmesh.obj'),
                  app_sha256=LAUNCHER_SHA,binary_sha256=NATIVE_RECIPES[recipe],runner_sha256=file_sha(r.__file__),
                  source_prepared_sha256=file_sha(prepared/'PREPARED.json'),
                  batch_iterations=10,energy='symmetric_dirichlet',tolerance=1e-6,threads=2)
    # Exact planar native result; the circular initialization itself need not pass visual distortion gates.
    uv=mapped['world'][:,1:].copy();tri=uv[mapped['faces'][0]]
    if np.linalg.det(np.stack([tri[1]-tri[0],tri[2]-tri[0]]))<0:uv[:,0]*=-1
    obj=batch/'native.obj'
    with obj.open('w') as stream:
        np.savetxt(stream,mapped['world'],fmt='v %.17g %.17g %.17g')
        np.savetxt(stream,uv,fmt='vt %.17g %.17g')
        np.savetxt(stream,np.repeat(mapped['faces']+1,2,axis=1),fmt='f %d/%d %d/%d %d/%d')
    checkpoint=r.publish_checkpoint(root,batch,obj,mapped['world'],mapped['faces'],identity,10,8.,True,lambda:None)
    atomic_json(root/'IDENTITY.json',identity)
    atomic_json(root/'STATUS.json',dict(status='NATIVE_FINISHED_PENDING_QUALITY_VALIDATION',iterations=10,converged=True,checkpoint=checkpoint))
    return root


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
@pytest.mark.parametrize('recipe',['installed-2f89bfc9','historical'])
def test_validate_rebinds_source_geometry_and_centerline(tmp_path,recipe):
    p=finishing();mesh=extracted(tmp_path);prepared=tmp_path/'prepared'
    preparation().prepare(mesh,prepared,3.5,[[0,0,0],[0,0,6]])
    native=native_fixture(prepared,tmp_path,recipe)
    with pytest.raises(ValueError,match='centerline'):
        p.validate(prepared,native,tmp_path/'bad',[[1,0,0],[1,0,6]],3.)
    out=tmp_path/'validated';report=p.validate(prepared,native,out,None,3.)
    assert report['status']=='MAP_VALIDATED_DIAGNOSTIC'
    assert report['reference']['xyz'][2]==3 and report['overlap']['complete']
    with np.load(out/'map.npz') as mapped,np.load(prepared/'initial-map.npz') as original:
        for key in ('world','faces','source_vertex_ids','source_face_ids'):np.testing.assert_array_equal(mapped[key],original[key])
    latest=json.loads((native/'LATEST.json').read_text());latest['identity']['source_map_sha256']='0'*64
    atomic_json(native/'LATEST.json',latest)
    with pytest.raises(ValueError):p.validate(prepared,native,tmp_path/'wrong-native',None,3.)
    assert not (tmp_path/'wrong-native'/'VALIDATED.json').exists()


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_installed_stage_dispatch_renders_ct_and_feeds_atlas(tmp_path,capsys):
    finishing()
    from winding_field.cli import main
    from winding_field.sheet_atlas import query_pixel
    mesh=extracted(tmp_path);cp=tmp_path/'centerline.json';atomic_json(cp,[[0,0,0],[0,0,6]])
    prepared=tmp_path/'prepared'
    assert main(['constant-sheet','prepare','--mesh',str(mesh),'--w','3.5','--centerline',str(cp),'--output',str(prepared)])==0
    native=native_fixture(prepared,tmp_path);validated=tmp_path/'validated'
    assert main(['constant-sheet','validate','--prepared',str(prepared),'--native',str(native),'--reference-z','3','--output',str(validated)])==0
    ct=tmp_path/'ct';ct.mkdir()
    atomic_json(ct/'.zarray',dict(zarr_format=2,shape=[8]*3,chunks=[8]*3,dtype='|u1',compressor=None,fill_value=0,filters=None,order='C'))
    z,y,x=np.indices((8,8,8));(ct/'0.0.0').write_bytes((x+2*y+3*z).astype('u1').tobytes())
    cfg=tmp_path/'render.json'
    atomic_json(cfg,dict(ct=dict(source=str(ct),voxel_um=8.64,level=0),settings=dict(
        output_frame=dict(id='synthetic/world',voxel_um=8.64,origin_xyz_um=[0,0,0]),
        app_run='/unused',cache_dir=str(tmp_path/'cache'),cache_mib=0,memory_gib=2.,tile_size=8)))
    sheet=tmp_path/'sheet'
    assert main(['constant-sheet','render','--validated',str(validated),'--config',str(cfg),'--output',str(sheet)])==0
    receipt=json.loads((sheet/'VERIFIED.json').read_text());assert receipt['max_pixel_error']==0 and receipt['scalar_and_xyz_spotchecks']>0
    atlas=tmp_path/'atlas'
    assert main(['sheet-atlas','build','--sheets',str(sheet/'SHEET.json'),'--output',str(atlas)])==0
    assert main(['sheet-atlas','verify','--atlas',str(atlas)])==0
    record=json.loads((sheet/'SHEET.json').read_text())
    assert record['files'].get('ct/ct-chunks.sqlite')==file_sha(sheet/'ct/ct-chunks.sqlite')
    assert (atlas/'sheets/sheet-000/ct/ct-chunks.sqlite').is_file()
    (sheet/'ct/ct-chunks.sqlite').write_bytes(b'changed')
    from winding_field.sheet_atlas import build_atlas
    with pytest.raises(ValueError,match='hash'):build_atlas([sheet/'SHEET.json'],tmp_path/'bad-ledger')
    assert main(['constant-sheet','status','--native',str(native)])==0
    assert 'Guardadas' in capsys.readouterr().out
    from PIL import Image
    with Image.open(atlas/'mask.png') as im:rr,cc=np.where(np.asarray(im)==2)
    assert len(rr)>0
    point=query_pixel(atlas,int(cc[0]),int(rr[0]))
    assert point['w']==3.5 and point['xyz'][0]==pytest.approx(3.5)
    config=json.loads(cfg.read_text());config['settings']['output_frame']['id']='different'
    atomic_json(cfg,config)
    with pytest.raises(ValueError,match='frame'):
        finishing().render(validated,tmp_path/'bad-frame',config)
    assert not (tmp_path/'bad-frame'/'SHEET.json').exists()


@pytest.mark.parametrize('change',['origin','tile-size'])
@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_render_rejects_shifted_origin_and_unreadable_tiles_before_output(tmp_path,change):
    p=finishing();mesh=extracted(tmp_path);prepared=tmp_path/'prepared'
    preparation().prepare(mesh,prepared,3.5,[[0,0,0],[0,0,6]])
    native=native_fixture(prepared,tmp_path);validated=tmp_path/'validated'
    p.validate(prepared,native,validated,None,3.)
    cfg=dict(ct=dict(source='/must-not-open'),settings=dict(output_frame=dict(id='synthetic/world',voxel_um=8.64,origin_xyz_um=[0,0,0]),
             app_run='/unused',cache_dir=str(tmp_path/'cache'),memory_gib=6.,cache_mib=0))
    if change=='origin':cfg['settings']['output_frame']['origin_xyz_um'][0]=8.64
    else:cfg['settings']['tile_size']=2048
    with pytest.raises(ValueError,match='origin|tile'):
        p.render(validated,tmp_path/'rejected',cfg)
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('change',['centerline','runner'])
@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_validation_rejects_replaced_preparation_or_unimported_runner(tmp_path,change):
    p=finishing();mesh=extracted(tmp_path);prepared=tmp_path/'prepared'
    preparation().prepare(mesh,prepared,3.5,[[0,0,0],[0,0,6]])
    native=native_fixture(prepared,tmp_path)
    if change=='centerline':
        path=prepared/'PREPARED.json';value=json.loads(path.read_text());value['centerline']=[[10,0,0],[10,0,6]];atomic_json(path,value)
    else:
        latest=json.loads((native/'LATEST.json').read_text());latest['identity']['runner_sha256']='b'*64
        atomic_json(native/'LATEST.json',latest);atomic_json(native/'IDENTITY.json',latest['identity'])
        atomic_json((native/latest['obj']).parent/'CHECKPOINT.json',latest)
    with pytest.raises(ValueError,match='preparation|runner'):
        p.validate(prepared,native,tmp_path/'rejected',None,3.)
    assert not (tmp_path/'rejected'/'VALIDATED.json').exists()


@pytest.mark.parametrize('stage',['prepare','validate'])
@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_output_symlink_ancestor_rejected_before_writing(tmp_path,stage):
    p=finishing();mesh=extracted(tmp_path);prepared=tmp_path/'prepared'
    target=tmp_path/'target';target.mkdir();alias=tmp_path/'alias';alias.symlink_to(target,target_is_directory=True)
    preparation().prepare(mesh,prepared,3.5,[[0,0,0],[0,0,6]])
    native=native_fixture(prepared,tmp_path)
    with pytest.raises(ValueError,match='symlink'):
        if stage=='prepare':preparation().prepare(mesh,alias/'new',3.5,[[0,0,0],[0,0,6]])
        else:p.validate(prepared,native,alias/'new',None,3.)
    assert not (target/'new').exists()
