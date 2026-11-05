import numpy as np
import pytest


def test_endpoint_grid_covers_exact_native_domain_without_padding_or_crop():
    from winding_field.bounded_coarse import endpoint_grid
    meta=dict(shape_zyx=[15898,8849,8849],spacing_xyz=[8.64]*3,origin_xyz=[0.]*3,frame='483A')
    grid=endpoint_grid(meta,276.48)
    assert grid['shape_zyx']==[498,278,278]
    np.testing.assert_allclose((np.array(grid['shape_zyx'])[::-1]-1)*grid['spacing_xyz'],
                               (np.array(meta['shape_zyx'])[::-1]-1)*meta['spacing_xyz'],rtol=0,atol=1e-10)
    with pytest.raises(ValueError):endpoint_grid(meta,float('nan'))


def test_coarse_producer_matches_dense_orientation_before_interpolation_and_resumes(tmp_path):
    from winding_field.bounded_coarse import NativeMaps,produce
    from winding_field.artifacts import Tiles
    from winding_field.field import calibration,sample_prior
    shape=(8,8,8);rng=np.random.default_rng(41)
    normals=np.zeros((*shape,3),'f4');normals[...,0]=rng.choice([-1.,1.],shape)
    density=np.full(shape,.002,'f4');support=np.ones(shape,bool);support[3,3,3]=False
    data=dict(normals=normals,density=density,support=support,surface_phase=rng.random(shape).astype('f4'))
    calls=[]
    def read(lo,hi):
        calls.append(tuple(lo));return {k:v[tuple(slice(a,b) for a,b in zip(lo,hi))] for k,v in data.items()}
    maps=NativeMaps(shape,read,chunk=4)
    meta=dict(shape_zyx=list(shape),spacing_xyz=[8.64]*3,origin_xyz=[0.]*3,frame='test')
    grid=dict(meta,shape_zyx=[4]*3,spacing_xyz=[20.16]*3)
    cal=calibration(maps,meta)
    target,weights,valid=sample_prior(maps,meta,grid,cal,np.zeros(3,int),np.array([4]*3))
    count=[0]
    def stop():count[0]+=1;return count[0]==3
    assert produce(tmp_path/'coarse',maps,meta,grid,{'source':'fixture'},chunk=2,stop=stop)=='PAUSED'
    assert produce(tmp_path/'coarse',maps,meta,grid,{'source':'fixture'},chunk=2)=='COMPLETE'
    out=Tiles(tmp_path/'coarse/maps')
    try:
        actual=out.read_many(['target','weights','support'],[0]*3,[4]*3)
        np.testing.assert_allclose(actual['target'],target,atol=1e-10)
        np.testing.assert_array_equal(actual['support'],valid)
        np.testing.assert_allclose(actual['weights'],weights)
        phase=out.read('surface_phase',[0]*3,[4]*3)
        from winding_field.field import map_axes
        expected=maps.sample(('surface_phase',),map_axes(meta,grid,np.zeros(3,int),np.array([4]*3)))
        np.testing.assert_allclose(phase,expected['surface_phase'])
    finally:out.close()


def test_calibration_resume_rejects_changed_native_enumeration(tmp_path):
    from winding_field.bounded_coarse import NativeMaps,produce
    calls=[]
    def read(lo,hi):
        calls.append(tuple(lo));shape=tuple(hi-lo)
        return dict(normals=np.broadcast_to(np.array([1.,0.,0.],'f4'),(*shape,3)),
                    density=np.ones(shape,'f4'),support=np.ones(shape,bool))
    meta=dict(shape_zyx=[8]*3,spacing_xyz=[1.]*3,origin_xyz=[0.]*3,frame='test')
    assert produce(tmp_path,NativeMaps((8,8,8),read,chunk=4),meta,meta,{},stop=lambda:len(calls)==1)=='PAUSED'
    with pytest.raises(ValueError,match='changed'):
        produce(tmp_path,NativeMaps((8,8,8),read,chunk=2),meta,meta,{})
    assert len(calls)==1
