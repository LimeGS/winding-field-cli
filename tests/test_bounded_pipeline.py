import numpy as np
import pytest


def test_bounded_prepare_retains_coarse_only_and_releases_model(tmp_path,monkeypatch):
    from winding_field import bounded_pipeline as pipeline
    class Volume:
        shape=(16,16,16);spacing_xyz=np.array([8.64]*3);origin_xyz=np.zeros(3)
        identity=dict(shape=list(shape),spacing_xyz=spacing_xyz.tolist(),origin_xyz=origin_xyz.tolist(),source='fixture')
        def sample(self,axes):return np.full(tuple(map(len,axes)),128,'u1')
    calls=[]
    class Model:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):calls.append('released')
        def predict_raw(self,ct):return np.full((8,64,64,64),.1,'f4')
        def finalize(self,raw):
            result=np.zeros((4,*raw.shape[1:]),'u1')
            result[0]=20;result[1]=255;result[2]=128;result[3]=128
            return result
    monkeypatch.setattr(pipeline,'LasagnaModel',Model)
    monkeypatch.setattr(pipeline,'runtime_identity',lambda p:{'executable':'fixture'})
    monkeypatch.setattr(pipeline,'checkpoint',lambda *a,**kw:'fixture.pt')
    config=dict(prediction='qual-v2',field_spacing_um=34.56,field_chunk=4,device='cpu',
                bounded_bytes=1<<24,reserve_bytes=0,invalid_value=0)
    result=pipeline.prepare(Volume(),tmp_path,config)
    assert result['status']=='PREPARED' and result['bounded_preparation'] is True
    assert result['native_geometry']['shape_zyx']==[16]*3
    assert result['shape_zyx']==[5]*3
    assert calls==['released']
    assert not (tmp_path/'fibres/probabilities.npy').exists()
    assert not (tmp_path/'lasagna-source/input.zarr').exists()


def test_bounded_fibres_publish_global_curves_and_resume_without_duplicate_identity(tmp_path,monkeypatch):
    import json
    import sqlite3
    from contextlib import contextmanager
    from winding_field import bounded_pipeline as pipeline
    from winding_field.bounded_coarse import NativeMaps
    from winding_field.util import file_sha
    class Volume:
        shape=(64,32,32);spacing_xyz=np.array([8.64]*3);origin_xyz=np.zeros(3)
        identity=dict(shape=list(shape),spacing_xyz=spacing_xyz.tolist(),origin_xyz=origin_xyz.tolist(),source='fixture')
        def sample(self,axes):
            z,y,x=np.meshgrid(*axes,indexing='ij')
            return np.rint(255*np.exp(-((x-16.)**2+(y-16.)**2)/18)).astype('u1')
    class Model:
        def __init__(self,*a,**k):pass
        def predict(self,ct):return None,None,np.stack([ct.astype('f4')/255,np.zeros(ct.shape,'f4')],axis=-1)
    @contextmanager
    def surfaces(volume,root,config,lo,hi):
        def read(a,b):
            shape=tuple(b-a)
            return dict(normals=np.broadcast_to(np.array([1.,0.,0.],'f4'),(*shape,3)),
                        density=np.ones(shape,'f4'),support=np.ones(shape,bool))
        yield NativeMaps(tuple(map(int,hi-lo)),read),{}
    monkeypatch.setattr(pipeline,'QualV2',Model)
    monkeypatch.setattr(pipeline,'surface_maps',surfaces)
    volume=Volume()
    meta=dict(source=volume.identity,producer_identity={'fixture':True},roi_lo_zyx=[0]*3,
              roi_hi_zyx=list(volume.shape),tiles_sha256='fixture',bounded_preparation=True)
    (tmp_path/'PREPARED.json').write_text(json.dumps(meta))
    config=dict(device='cpu',bounded_bytes=128<<20,reserve_bytes=0)
    result=pipeline.fibres(volume,tmp_path,config)
    assert result['status']=='FIBRES_READY' and result['curves']>0
    with sqlite3.connect(tmp_path/'fibres/fibres.sqlite') as db:
        assert db.execute('SELECT count(DISTINCT source_id) FROM curves').fetchone()[0]>0
        assert db.execute("SELECT count(*) FROM curves WHERE family!='V'").fetchone()[0]==0
    sha=file_sha(tmp_path/'fibres/fibres.sqlite')
    assert pipeline.fibres(volume,tmp_path,config)==result
    assert file_sha(tmp_path/'fibres/fibres.sqlite')==sha


def test_bounded_field_uses_retained_coarse_prior_and_existing_block_backend(tmp_path):
    import json
    import sqlite3
    from winding_field.artifacts import Tiles
    from winding_field.field import solve
    from winding_field.util import file_sha
    grid=dict(shape_zyx=[5]*3,spacing_xyz=[1.]*3,origin_xyz=[0.]*3,frame='test')
    maps=Tiles(tmp_path/'coarse/maps',[5]*3,4,{'fixture':'coarse'})
    for key in maps.keys():
        lo,hi=maps.box(key);shape=tuple(hi-lo)
        maps.write(key,dict(target=np.broadcast_to(np.array([1.,0.,0.],'f4'),(*shape,3)),
                            weights=np.ones(shape,'f4'),support=np.ones(shape,bool),surface_phase=np.ones(shape,'f4')))
    sha=maps.identity();maps.close()
    cal=dict(mode='planar',axis_xyz=[1.,0.,0.],centre_xyz=[0.]*3,mean_density_per_um=1.)
    (tmp_path/'PREPARED.json').write_text(json.dumps(dict(grid,bounded_preparation=True,
        tiles_sha256=sha,calibration=cal,prediction='qual-v2',surface_method='lasagna')))
    (tmp_path/'fibres').mkdir()
    with sqlite3.connect(tmp_path/'fibres/fibres.sqlite') as db:
        db.execute('CREATE TABLE curves(id INTEGER PRIMARY KEY,points BLOB)')
    (tmp_path/'FIBRES.json').write_text(json.dumps(dict(sha256=file_sha(tmp_path/'fibres/fibres.sqlite'))))
    result=solve(tmp_path,dict(field_chunk=4,sweeps=2,device='cpu',rtol=1e-6,bounded_bytes=16<<20))
    assert result['status']=='FIELD_READY' and result['global_converged'] is True
    assert result['solver_backend']['device']=='cpu'
    assert result['solver_backend']['field_chunk']==4 and result['solver_backend']['dtype']=='float64'
    assert result['verification']['relative_residual']==0
    from winding_field.field import open_field
    from winding_field.meshes import select_levels
    # Exercise automatic phase selection against the retained coarse store.
    with open_field(tmp_path) as source:
        levels,_=select_levels(tmp_path,{},source)
    np.testing.assert_allclose(levels,[1.,2.,3.],rtol=0,atol=1e-14)


def test_public_bounded_plan_uses_existing_cuda_chunk_and_separates_native_retention(tmp_path):
    import json
    import subprocess
    import sys
    from ct_fixture import write_ct
    source=write_ct(tmp_path/'ct.zarr',np.full((64,32,32),128,'u1'),spacing=(8.64,)*3)
    command=[sys.executable,'-m','winding_field','plan','--source',str(source),'--output',str(tmp_path/'run'),
             '--cache',str(tmp_path/'cache'),'--prediction','qual-v2','--bounded-preparation','--bounded-gib','.25',
             '--field-spacing-um','34.56','--field-chunk','64','--device','cuda:0','--reserve-gib','0']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert plan['estimated_products_bytes']==1<<28
    assert plan['estimated_spline_dense_bytes']==0
    assert plan['estimated_lasagna_staging_bytes']==0
    assert plan['field']['shape_zyx']==[17,9,9]
    assert plan['solver_device']=='cuda:0' and plan['field_chunk']==64
    request=tmp_path/'export.json'
    request.write_text(json.dumps(dict(levels=[1.],reserve_disk_gib=0,max_pixels=4096)))
    result=subprocess.run(command+['--surface-output','tifxyz','--tifxyz-config',str(request)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout);stages=plan['estimated_products_by_stage']
    assert plan['estimated_products_bytes']==(1<<28)+stages['tifxyz']
    assert plan['solver_device']=='cuda:0' and plan['field_chunk']==64


def test_public_bounded_field_verifies_quantized_gauge_and_rejects_changed_runtime(tmp_path,monkeypatch):
    import json
    from winding_field import bounded_pipeline as pipeline,lasagna
    from winding_field.cli import main
    from winding_field.artifacts import Tiles
    from ct_fixture import write_ct
    z,y,x=np.indices((64,32,32))
    ct=np.rint(255*np.exp(-((x-16.)**2+(y-16.)**2)/18)).astype('u1')
    source=write_ct(tmp_path/'ct.zarr',ct,spacing=(8.64,)*3)
    runtime={'executable':'fixture'}
    monkeypatch.setattr(lasagna,'runtime_identity',lambda p:dict(runtime))
    monkeypatch.setattr(pipeline,'runtime_identity',lambda p:dict(runtime))
    monkeypatch.setattr(lasagna,'checkpoint',lambda *a,**k:'fixture.pt')
    monkeypatch.setattr(pipeline,'checkpoint',lambda *a,**k:'fixture.pt')
    class Model:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def predict_raw(self,ct):return np.full((8,64,64,64),.1,'f4')
        def finalize(self,raw):
            result=np.zeros((4,*raw.shape[1:]),'u1')
            result[0]=20;result[1]=255;result[2]=128;result[3]=128
            return result
    monkeypatch.setattr(pipeline,'LasagnaModel',Model)
    class Qual:
        def __init__(self,*a,**k):pass
        def predict(self,ct):return None,None,np.stack([ct.astype('f4')/255,np.zeros(ct.shape,'f4')],axis=-1)
    monkeypatch.setattr(pipeline,'QualV2',Qual)
    root=tmp_path/'run'
    args=['run','--source',str(source),'--output',str(root),'--cache',str(tmp_path/'cache'),
          '--prediction','qual-v2','--bounded-preparation','--bounded-gib','.25',
          '--field-spacing-um','34.56','--sweeps','1','--reserve-gib','0','--surface-output','atlas','--until','prepare']
    assert main(args)==0
    assert json.loads((root/'PREPARED.json').read_bytes())['bounded_preparation']
    with_phase=Tiles(root/'coarse/maps')
    try:assert np.isfinite(with_phase.read('surface_phase',[0]*3,[5]*3)).all()
    finally:with_phase.close()
    assert not (root/'maps').exists()
    assert main(['prepare',str(root)])==0
    from winding_field import global_residual
    verify=global_residual.verify
    def not_converged(*a,**k):
        return dict(verify(*a,**k),status='NONCONVERGED',global_converged=False)
    monkeypatch.setattr(global_residual,'verify',not_converged)
    assert main(['field',str(root)])==0
    before_config=json.loads((root/'RUN.json').read_bytes())['config']
    assert not (root/'FIELD.json').exists()
    assert main(['resume',str(root),'--field-sweeps','2'])==0
    extended=json.loads((root/'field/CHECKPOINT.json').read_bytes())
    assert len(extended['progress']['completed_sweeps'])==2
    assert extended['sweep_extensions'][0]['from']==1
    assert not (root/'FIELD.json').exists()
    assert json.loads((root/'RUN.json').read_bytes())['config']==before_config
    assert main(['resume',str(root),'--field-sweeps','1'])==1
    monkeypatch.setattr(global_residual,'verify',verify)
    assert main(['field',str(root)])==0
    state=json.loads((root/'RUN.json').read_bytes())
    assert 'field' in state['stages']
    assert json.loads((root/'FIBRES.json').read_bytes())['curves']>0
    field=json.loads((root/'GLOBAL_RESIDUAL.json').read_bytes())
    assert field['global_converged'] and field['solver_backend']['field_chunk']==64
    assert field['fixed_storage_max_error']==0 and field['gauge_codec_offset'] is not None
    assert field['fixed_max_error']>0 and field['relative_residual']<=field['rtol']
    assert (root/'FIELD.json').exists() and not (root/'flat').exists()
    assert (root/'fibres/fibres.afv').is_file()
    before=(root/'field/CHECKPOINT.json').read_bytes()
    assert main(['prepare',str(root)])==0
    assert (root/'field/CHECKPOINT.json').read_bytes()==before
    runtime['executable']='changed'
    assert main(['prepare',str(root)])==1


@pytest.mark.parametrize('change',[
    'DELETE FROM curve_bounds','UPDATE curve_bounds SET min_x=10,max_x=11',
    'UPDATE curves SET source_id=8','DELETE FROM curves','DELETE FROM done WHERE id=2'])
def test_fitted_resume_rejects_changed_inventory(tmp_path,change):
    import sqlite3
    from winding_field.bounded_pipeline import _curve_db,_store_curves
    path=tmp_path/'curves.sqlite';db=_curve_db(path)
    _store_curves(db,tmp_path,1,[np.array([[0.,0.,0.],[1.,1.,1.]])],'V',1,1<<20,0)
    _store_curves(db,tmp_path,2,[],'V',2,1<<20,0)
    db.close()
    with sqlite3.connect(path) as db:db.execute(change)
    with pytest.raises(ValueError,match='inventory'):_curve_db(path)


def test_empty_fitted_commit_obeys_quota(tmp_path):
    from winding_field.bounded_pipeline import _curve_db,_store_curves
    db=_curve_db(tmp_path/'curves.sqlite')
    try:
        with pytest.raises(OSError,match='quota'):_store_curves(db,tmp_path,1,[],'V',1,1,0)
        assert db.execute('SELECT count(*) FROM done').fetchone()[0]==0
    finally:db.close()
