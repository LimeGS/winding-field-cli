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
