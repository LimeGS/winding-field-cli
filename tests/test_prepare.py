import numpy as np
import pytest


def test_layers_have_physical_density_and_distinct_tangent():
    from winding_field.predict import observations
    z,y,x = np.indices((32,32,64))
    ct = (30+180*(np.cos(2*np.pi*x/8)>.4)+20*np.sin(2*np.pi*y/12)).astype(np.uint8)
    data = observations(ct, [2.,3.,4.])
    core = (slice(8,-8),slice(8,-8),slice(16,-16))
    assert np.mean(data['support'][core]) > .9
    assert np.median(data['density'][core]) == pytest.approx(1/16,rel=.15)
    assert np.mean(abs(data['normals'][core][...,0])) > .98
    assert np.max(abs(np.sum(data['normals']*data['tangents'],axis=-1))) < 1e-4


def test_empty_ct_is_unsupported():
    from winding_field.predict import observations
    data = observations(np.zeros((16,16,16),np.uint8),[1.,1.,1.])
    assert not data['support'].any()
    assert not data['density'].any()


def test_tile_resume_rejects_changed_identity_and_corruption(tmp_path):
    from winding_field.artifacts import Tiles
    tiles = Tiles(tmp_path/'maps',shape=(4,4,8),chunk=4,identity={'source':'a'})
    tiles.write((0,0,0),{'value':np.ones((4,4,4),np.float32)})
    tiles.write((0,0,1),{'value':np.full((4,4,4),2,np.float32)})
    crop=tiles.read('value',[0,0,2],[4,4,6])
    np.testing.assert_array_equal(crop[0,0],[1,1,2,2])
    with pytest.raises(ValueError,match='identity'):
        Tiles(tmp_path/'maps',shape=(4,4,8),chunk=4,identity={'source':'b'})
    next((tmp_path/'maps').glob('*.npz')).write_bytes(b'broken')
    with pytest.raises(ValueError,match='corrupt'):
        tiles.read('value',[0,0,0],[4,4,8])


def test_streamlines_stop_at_unsupported_gap_and_use_world_coordinates():
    from winding_field.fibres import streamlines
    shape=(16,8,8)
    presence=np.ones(shape,np.float32)
    tangent=np.zeros(shape+(3,),np.float32); tangent[...,2]=1
    support=np.ones(shape,bool); support[8:]=False
    curves=streamlines(presence,tangent,support,seeds=[[4,4,4]],
                       origin_xyz=[10.,20.,30.],spacing_xyz=[2.,3.,4.])
    assert len(curves)==1 and len(curves[0])>3
    assert np.allclose(curves[0][:,:2],[18.,32.])
    assert curves[0][:,2].min()>=30 and curves[0][:,2].max()<=58


def test_model_rejects_wrong_checkpoint_before_loading(tmp_path):
    from winding_field.models import verify_checkpoint
    p=tmp_path/'model.pt'; p.write_bytes(b'not a model')
    with pytest.raises(ValueError,match='SHA'):
        verify_checkpoint(p)


def test_prediction_core_agrees_across_tile_boundary():
    from winding_field.predict import observations
    z,y,x=np.indices((24,24,112))
    ct=(30+190*(np.cos(2*np.pi*x/8)>.5)).astype(np.uint8)
    whole=observations(ct,[1.,1.,1.])
    left=observations(ct[:,:,:96],[1.,1.,1.])
    np.testing.assert_allclose(whole['density'][:,:,40:56],left['density'][:,:,40:56],atol=1e-6)


def test_upstream_model_can_infer_and_decode_direction():
    from winding_field._vendor.vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
    from winding_field._vendor.vesuvius.neural_tracing.fiber_trace_3d.direction import decode_lasagna_direction_3x2_analytic
    import torch
    model=build_fiber_trace_3d_model({'model_3d':{'features_per_stage':[4,8]}}).eval()
    with torch.no_grad():
        output=model(torch.zeros((1,1,16,16,16)))
        tangent=decode_lasagna_direction_3x2_analytic(output[:,:6].permute(0,2,3,4,1))
    assert tuple(output.shape)==(1,7,16,16,16)
    assert torch.isfinite(tangent).all()


def test_prepare_resume_pins_ct_and_roi_geometry(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.predict import prepare
    from winding_field.artifacts import Tiles
    x=np.indices((12,12,32))[2]
    source=write_ct(tmp_path/'ct.zarr',(30+180*(np.cos(2*np.pi*x/8)>.5)).astype(np.uint8),
                    spacing=(2.,3.,4.),origin=(10.,20.,30.))
    volume=Volume(source,cache=tmp_path/'cache')
    config=dict(roi_lo=[2,2,4],roi_hi=[10,10,28],stride=2,chunk=16)
    result=prepare(volume,tmp_path/'run',config)
    assert result['origin_xyz']==[18.,26.,38.]
    assert result['shape_zyx']==[4,4,12]
    tiles=Tiles(tmp_path/'run/maps')
    before=tiles.identity(); tiles.close()
    assert prepare(volume,tmp_path/'run',config)['tiles_sha256']==before
    (source/'0/0.0.0').write_bytes(bytes([0])*32**3)
    with pytest.raises(ValueError,match='changed'):
        prepare(volume,tmp_path/'run',config)


def test_strided_preparation_reads_native_chunks_not_dense_envelope(tmp_path,monkeypatch):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.predict import prepare
    from winding_field.artifacts import Tiles
    data=np.broadcast_to(np.arange(80,dtype=np.uint8),(16,16,80)).copy()
    source=write_ct(tmp_path/'ct.zarr',data)
    volume=Volume(source,cache=tmp_path/'cache')
    read=volume.read
    def bounded(lo,hi):
        assert np.all(np.asarray(hi)-lo<=volume.chunks),'dense native-resolution envelope'
        return read(lo,hi)
    monkeypatch.setattr(volume,'read',bounded)
    meta=prepare(volume,tmp_path/'run',dict(roi_lo=[0,0,1],roi_hi=[16,16,74],stride=8,chunk=64))
    assert meta['origin_xyz']==[1.,0.,0.]
    maps=Tiles(tmp_path/'run/maps')
    np.testing.assert_array_equal(maps.read('ct',[0,0,0],[2,2,10]),data[::8,::8,1:74:8])
    maps.close();volume.close()


def test_classical_threshold_is_shared_across_tiles_and_persisted(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.predict import prepare
    from winding_field.artifacts import Tiles
    data=np.broadcast_to(np.where(np.arange(192)%8<4,20,80),(8,8,192)).astype(np.uint8).copy()
    data[:,:,144:]=220  # Only the right tile's distant halo sees this material.
    volume=Volume(write_ct(tmp_path/'ct.zarr',data),cache=tmp_path/'cache')
    config=dict(roi_hi=[8,8,128],prediction='classical')
    first=prepare(volume,tmp_path/'small',dict(config,chunk=64))
    second=prepare(volume,tmp_path/'large',dict(config,chunk=128))
    a=Tiles(tmp_path/'small/maps'); b=Tiles(tmp_path/'large/maps')
    np.testing.assert_allclose(a.read('density',[2,2,40],[6,6,88]),
                               b.read('density',[2,2,40],[6,6,88]),atol=1e-6)
    assert a.read('support',[2,2,40],[6,6,88]).mean()>.9
    assert first['identity']['threshold']==second['identity']['threshold']==20
    assert (tmp_path/'small/THRESHOLD.json').exists()
    assert prepare(volume,tmp_path/'small',dict(config,chunk=64))['tiles_sha256']==first['tiles_sha256']
    a.close();b.close();volume.close()


@pytest.mark.parametrize('method',['sd2','qual-v2'])
def test_pretrained_storage_chunks_and_overlapping_rois_share_predictions(tmp_path,monkeypatch,method):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.artifacts import Tiles
    import winding_field.predict as prediction
    calls=[]
    class WindowModel:
        def __init__(self,*a,**k): pass
        def predict(self,ct):
            calls.append(ct.shape)
            presence=np.clip((ct.astype(float)-ct.mean())/max(ct.std(),1)*.2+.5,0,1).astype(np.float32)
            tangent=np.zeros(ct.shape+(3,),np.float32); tangent[...,1]=1
            return (presence,tangent,np.stack([presence*.4,presence*.6],axis=-1)) if method=='qual-v2' else (presence,tangent)
    monkeypatch.setattr(prediction,'QualV2' if method=='qual-v2' else 'SD2',WindowModel,raising=False)
    z,y,x=np.indices((72,)*3)
    ct=(20+150*(x%8<3)+x+y//3+z//4).astype(np.uint8)
    v=Volume(write_ct(tmp_path/'ct.zarr',ct,spacing=(9.,)*3),cache=tmp_path/'cache')
    for name,config in [('a',dict(chunk=16)),('b',dict(chunk=32)),
                        ('c',dict(chunk=16,roi_lo=[4,4,4]))]:
        prediction.prepare(v,tmp_path/name,dict(prediction=method,roi_hi=[32]*3,**config))
    maps=[Tiles(tmp_path/name/'maps') for name in 'abc']
    for channel in ('presence','tangents','density','normals','support','ct'):
        a=maps[0].read(channel,[4]*3,[32]*3)
        np.testing.assert_array_equal(a,maps[1].read(channel,[4]*3,[32]*3))
        np.testing.assert_array_equal(a,maps[2].read(channel,[0]*3,[28]*3))
    if method=='qual-v2':
        np.testing.assert_array_equal(maps[0].read('probabilities',[4]*3,[32]*3),
                                      maps[2].read('probabilities',[0]*3,[28]*3))
    assert set(calls)=={(128 if method=='qual-v2' else 112,)*3}
    for m in maps: m.close()
    v.close()


def test_validity_excludes_full_filter_footprint_without_rejecting_valid_zeros():
    from winding_field.predict import observations
    x=np.indices((8,8,128))[2]
    ct=np.where(x%8<3,200,0).astype(np.uint8)
    validity=np.ones(ct.shape,bool); validity[:,:,:16]=False
    masked=observations(ct,[1.]*3,threshold=100,validity=validity)
    assert not masked['support'][:,:,:56].any()
    assert not masked['density'][:,:,:56].any()
    assert not masked['normals'][:,:,:56].any()
    assert masked['support'][:,:,64:112].all()
    assert masked['support'][ct==0].any()  # Zero intensity is valid without a fill declaration.
    assert not masked['presence'][~validity].any()


def test_acquisition_fill_policy_is_explicit_and_bound_to_resume(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.predict import prepare
    from winding_field.artifacts import Tiles
    ct=np.zeros((8,8,64),np.uint8); ct[:,:,32:]=160
    v=Volume(write_ct(tmp_path/'ct.zarr',ct),cache=tmp_path/'cache')
    config=dict(chunk=64,threshold=100,invalid_value=0)
    prepare(v,tmp_path/'run',config)
    maps=Tiles(tmp_path/'run/maps')
    assert not maps.read('support',[0,0,0],[8,8,64]).any()
    maps.close()
    with pytest.raises(ValueError,match='identity'):
        prepare(v,tmp_path/'run',dict(config,invalid_value=None))
    v.close()
