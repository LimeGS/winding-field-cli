"""The selected solver/scope must own all faces and survive a verified resume."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from winding_field.util import atomic_json, file_sha


def matrix():
    assert importlib.util.find_spec('winding_field.flattening_matrix'), 'flattening matrix is missing'
    from winding_field import flattening_matrix
    return flattening_matrix


def request(tmp_path):
    return dict(output_frame=dict(id='test',voxel_um=9.6,origin_xyz_um=[0,0,0]),
                tile_size=16,pitch_um=9.6,cache_dir=str(tmp_path/'cache'),memory_gib=2.)


def geometry():
    x,z=np.meshgrid(np.arange(4.,13.),np.arange(4.,13.))
    v=np.column_stack([x.ravel(),np.full(x.size,3.),z.ravel()]);f=[]
    for j in range(8):
        for i in range(8):
            if 3<=i<5 and 3<=j<5:continue
            a=j*9+i;f.extend([[a,a+1,a+10],[a,a+10,a+9]])
    # A separate island must survive whole-sheet as well as patch mapping.
    v=np.concatenate([v,[[20.,3.,4.],[24.,3.,4.],[20.,3.,8.]]])
    f=np.asarray(f+[[81,82,83]],np.int64)
    ids,inv=np.unique(f,return_inverse=True)
    return v[ids],inv.reshape(-1,3)


def test_settings_require_explicit_native_only_for_slim(tmp_path):
    m=matrix();req=request(tmp_path)
    assert m.settings(req,'lscm','sheet')['flattening_solver']=='lscm'
    with pytest.raises(ValueError,match='app_run'):m.settings(req,'slim','sheet')
    for extra in ({'max_patch_faces':32},{'max_depth':2},{'iterations':10}):
        with pytest.raises(ValueError):m.settings(dict(req,**extra),'lscm','sheet')
    assert m.settings(dict(req,max_patch_faces=32),'lscm','patches')['max_patch_faces']==32
    with pytest.raises(ValueError):m.settings(dict(req,disk_reserve_gib=1),'lscm','patches')


@pytest.mark.parametrize('flags',[
    [],['--flattening-solver','lscm'],['--scope','sheet'],
    ['--flattening-solver','slim','--scope','sheet','--flattening-method','original'],
])
def test_incomplete_or_mixed_cli_selection_fails_before_output(tmp_path,capsys,flags):
    from winding_field.cli import main
    assert main(['export-sheets','--mesh-bundle','missing','--source','missing',
                 '--flattening-config','missing','--output',str(tmp_path/'out'),*flags])==1
    assert 'selection' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('scope',['patches','sheet'])
def test_lscm_map_preserves_hole_island_and_oriented_source_faces(tmp_path,scope):
    m=matrix();v,f=geometry();req=request(tmp_path)
    if scope=='patches':req['max_patch_faces']=32
    cfg=m.settings(req,'lscm',scope)
    out=tmp_path/'map';result=m.build_map(v,f,out,cfg)
    np.testing.assert_array_equal(result['world'],v[result['parent']])
    np.testing.assert_array_equal(result['parent'][result['faces']],f[result['source_face_rows']])
    np.testing.assert_array_equal(np.sort(result['source_face_rows']),np.arange(121))
    assert result['report']['source_components']==2
    domains=result['report']['domains']
    assert len(domains)==2 if scope=='sheet' else len(domains)>2 and max(d['faces'] for d in domains)<=32
    assert result['report']['overlap']['complete']
    assert result['report']['overlapping_faces']==0
    assert result['report']['quality']['inverted_or_invalid_area_fraction']==0
    assert result['report']['welded'] is False
    saved=file_sha(out/'MAP.json')
    resumed=m.build_map(v,f,out,cfg)
    np.testing.assert_array_equal(resumed['uv'],result['uv'])
    assert file_sha(out/'MAP.json')==saved
    with pytest.raises(ValueError,match='identity'):
        m.build_map(v,f,out,dict(cfg,scope='sheet' if scope=='patches' else 'patches'))
    (out/'map.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash'):m.build_map(v,f,out,cfg)


def test_solver_failure_never_falls_back_or_subdivides(tmp_path,monkeypatch):
    m=matrix();v,f=geometry();cfg=m.settings(request(tmp_path),'lscm','sheet')
    def fail(*a,**kw):raise RuntimeError('solver failure fixture')
    monkeypatch.setattr(m,'solve',fail)
    with pytest.raises(RuntimeError,match='solver failure'):m.build_map(v,f,tmp_path/'map',cfg)
    assert not (tmp_path/'map'/'MAP.json').exists()
    assert len(list((tmp_path/'map').glob('domain-*')))==1


def source_bundle(tmp_path):
    from winding_field.surface_mesh import import_mesh
    v,f=geometry();source=tmp_path/'source.npz';np.savez(source,vertices_xyz=v,faces=f)
    metadata=dict(frame='test',voxel_um=9.6,origin_xyz=[0,0,0],spacing_xyz=[1]*3,shape_zyx=[32]*3,sheet_step=1.)
    root=tmp_path/'mesh';import_mesh(dict(path=str(source),sha256=file_sha(source)),root,metadata)
    ct=tmp_path/'ct';ct.mkdir()
    atomic_json(ct/'.zarray',dict(zarr_format=2,shape=[32]*3,chunks=[32]*3,dtype='|u1',compressor=None,
                                fill_value=0,order='C',filters=None))
    z,y,x=np.indices((32,32,32));(ct/'0.0.0').write_bytes((x+2*z).astype('u1').tobytes())
    return root,ct


@pytest.mark.parametrize('solver',['lscm','slim'])
@pytest.mark.parametrize('scope',['patches','sheet'])
@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_cli_matrix_ct_xyz_keys_and_completed_resume(tmp_path,capsys,solver,scope):
    app=os.environ.get('MATRIX_NATIVE_APP')
    if solver=='slim' and not app:pytest.skip('native qualification requires MATRIX_NATIVE_APP and bounded Linux cgroup')
    from winding_field.cli import main
    from winding_field.patch_mapping import PatchMapping
    mesh,ct=source_bundle(tmp_path);req=request(tmp_path)
    if scope=='patches':req['max_patch_faces']=32
    if solver=='slim':
        req['app_run']=app
        req['native_recipe']=os.environ.get('MATRIX_NATIVE_RECIPE','installed-2f89bfc9')
    config=tmp_path/'config.json';atomic_json(config,req);out=tmp_path/'out'
    args=['export-sheets','--mesh-bundle',str(mesh),'--source',str(ct),'--voxel-um','9.6',
          '--flattening-solver',solver,'--scope',scope,'--flattening-config',str(config),'--output',str(out)]
    assert main(args)==0
    receipt=json.loads((out/'MATRIX.json').read_text())
    assert receipt['status']=='MATRIX_READY_DIAGNOSTIC'
    level=out/'level-0000';record=json.loads((level/'LEVEL.json').read_text())
    assert record['mapping']['flattening_solver']==solver and record['mapping']['scope']==scope
    with PatchMapping(level) as mapped:
        a=mapped.arrays
        assert len(list(mapped.iter_keys('faces')))==121
        assert len(set(a['source_face_ids']))==121
        np.testing.assert_array_equal(a['field_xyz'][:,1],3.)
        domains=record['mapping']['domains']
        assert len(domains)==2 if scope=='sheet' else len(domains)>2 and max(d['faces'] for d in domains)<=32
        if solver=='slim':
            m=matrix();v,f=geometry();other=dict(req);other.pop('app_run');other.pop('native_recipe')
            comparison=m.build_map(v,f,tmp_path/'lscm-comparison',m.settings(other,'lscm',scope))
            assert record['mapping']['partition_sha256']==comparison['report']['partition_sha256']
            np.testing.assert_array_equal(a['domain_ids'],comparison['domain_ids'])
            np.testing.assert_array_equal(a['faces'],comparison['faces'])
    checked=0
    for tile in record['rendering']['tiles']:
        assert tile['rendering']['disconnected_components_translated'] is True
        assert tile['rendering']['patch_repacking'] is (scope=='patches')
        root=level/'render'/tile['directory']
        with np.load(root/'mapping.npz') as data:
            im=np.asarray(Image.open(root/'image.png'));mask=np.asarray(Image.open(root/'mask.png'))
            xyz=data['xyz'][mask==2];checked+=len(xyz)
            np.testing.assert_array_equal(im[mask==2],np.rint(xyz[:,0]+2*xyz[:,2]).astype('u1'))
            assert not ((xyz[:,0]>7)&(xyz[:,0]<9)&(xyz[:,2]>7)&(xyz[:,2]<9)).any()
            assert not im[mask!=2].any()
    assert checked>20
    before=file_sha(out/'MATRIX.json');level_before=file_sha(level/'LEVEL.json')
    assert main(args+['--resume'])==0
    assert file_sha(out/'MATRIX.json')==before and file_sha(level/'LEVEL.json')==level_before
    changed=args.copy();changed[changed.index('--scope')+1]='sheet' if scope=='patches' else 'patches'
    assert main(changed+['--resume'])==1
    assert file_sha(out/'MATRIX.json')==before
    (ct/'0.0.0').write_bytes(bytes(32**3))
    assert main(args+['--resume'])==1
    assert 'CT chunk changed' in capsys.readouterr().err


def cli_fixture(tmp_path):
    mesh,ct=source_bundle(tmp_path);config=tmp_path/'config.json';atomic_json(config,request(tmp_path));out=tmp_path/'out'
    return out,ct,['export-sheets','--mesh-bundle',str(mesh),'--source',str(ct),'--voxel-um','9.6',
                   '--flattening-solver','lscm','--scope','sheet','--flattening-config',str(config),'--output',str(out)]


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_interrupted_render_retains_map_and_evidence_then_resumes(tmp_path,monkeypatch):
    from winding_field.cli import main
    from winding_field import patch_pipeline as p
    out,ct,args=cli_fixture(tmp_path);real=p.render_map
    def interrupt(mapped,output,*a,**kw):
        output.mkdir();(output/'partial.txt').write_text('retained interrupted render')
        raise InterruptedError('render stop fixture')
    monkeypatch.setattr(p,'render_map',interrupt)
    assert main(args)==1
    initial=file_sha(out/'level-0000/flattening/MAP.json')
    monkeypatch.setattr(p,'render_map',real)
    assert main(args+['--resume'])==0
    assert file_sha(out/'level-0000/flattening/MAP.json')==initial
    assert len(list((out/'level-0000').glob('render-interrupted-*/partial.txt')))==1


@pytest.mark.parametrize('tamper',['missing-ledger','removed-entry','symlink'])
@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_incomplete_publication_rejects_lost_ct_provenance_and_symlinks(tmp_path,tamper):
    import sqlite3
    from winding_field.cli import main
    out,ct,args=cli_fixture(tmp_path)
    assert main(args)==0
    # Simulate interruption after committing a level but before the root receipt.
    (out/'MATRIX.json').unlink()
    if tamper=='missing-ledger':(out/'ct/ct-chunks.sqlite').unlink()
    elif tamper=='removed-entry':
        with sqlite3.connect(out/'ct/ct-chunks.sqlite') as db:db.execute('DELETE FROM chunks')
    else:
        external=tmp_path/'external';external.mkdir()
        (out/'level-0000/flattening/foreign').symlink_to(external,target_is_directory=True)
    assert main(args+['--resume'])==1
    assert not (out/'MATRIX.json').exists()


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree resource guard')
def test_native_checkpoint_interrupt_resumes_without_rewriting_prior_batch(tmp_path,monkeypatch):
    from winding_field import flatboi_checkpoint as r
    from test_flatboi_checkpoint import fake_native
    m=matrix();app=fake_native(tmp_path,monkeypatch,r)
    cfg=m.settings(dict(request(tmp_path),app_run=str(app)),'slim','sheet')
    v=np.array([[1.,1,1],[3.,1,1],[1.,1,3]]);f=np.array([[0,1,2]],np.int64)
    publish=r.publish_checkpoint
    def interrupt(*a,**kw):
        result=publish(*a,**kw)
        raise InterruptedError('after durable native checkpoint')
    monkeypatch.setattr(r,'publish_checkpoint',interrupt)
    out=tmp_path/'map'
    with pytest.raises(InterruptedError):m.build_map(v,f,out,cfg)
    native=out/'domain-000000/native';checkpoint=json.loads((native/'LATEST.json').read_text())
    prior=native/checkpoint['obj'];sha=file_sha(prior)
    assert checkpoint['iterations']==10 and not (out/'MAP.json').exists()
    monkeypatch.setattr(r,'publish_checkpoint',publish)
    mapped=m.build_map(v,f,out,cfg)
    assert file_sha(prior)==sha
    assert mapped['report']['domains'][0]['solver']['checkpoint']['iterations']==200
