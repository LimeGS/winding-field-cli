import json

import numpy as np
import pytest


def curved_strip():
    theta,z=np.meshgrid(np.linspace(-1.4,1.4,13),np.arange(5.))
    vertices=np.stack((10*np.cos(theta),10*np.sin(theta),z),axis=-1).reshape(-1,3)
    faces=[]
    for row in range(4):
        for col in range(12):
            a=row*13+col
            faces.extend([(a,a+1,a+13),(a+1,a+14,a+13)])
    return vertices,np.array(faces)


def test_curved_developable_surface_flattens_with_correspondence():
    from winding_field.flatten import parameterize
    vertices,faces=curved_strip()
    uv,quality=parameterize(vertices,faces)
    assert uv.shape==(65,2)
    assert quality['flipped_faces']==0
    assert quality['overlap_pairs']==0
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    actual=np.linalg.norm(uv[edges[:,0]]-uv[edges[:,1]],axis=1)
    expected=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    np.testing.assert_allclose(actual,expected,rtol=.01,atol=.01)


def test_degenerate_mesh_is_rejected():
    from winding_field.flatten import parameterize
    with pytest.raises(ValueError,match='degenerate'):
        parameterize(np.zeros((3,3)),np.array([[0,1,2]]))


def test_tiled_mesh_and_flatten_preserve_all_faces_and_roi(tmp_path):
    from test_field import prepared_run
    from winding_field.field import solve
    from winding_field.meshes import mesh
    from winding_field.flatten import flatten
    run=prepared_run(tmp_path/'run')
    solve(run,{'sweeps':1,'field_chunk':8})
    receipt=mesh(run,{'levels':[.5,1.5,2.5],'mesh_tile':4})
    assert sum(x['components'] for x in receipt['levels'])==3
    assert all(x['nonmanifold_edges']==0 for x in receipt['levels'])
    flat=flatten(run,{'chart_faces':128})
    assert flat['covered_faces']==sum(x['faces'] for x in receipt['levels'])
    assert flat['coverage_fraction']==1.
    ids=[]
    for file in (run/'flat').glob('chart-*.npz'):
        with np.load(file) as chart:
            ids.extend(chart['source_face_ids'].tolist())
            assert np.min(chart['vertices'],axis=0).tolist()>=[10.,20.,30.]
            assert np.all(np.max(chart['vertices'],axis=0)<=[56.,35.,50.])
            assert np.isfinite(chart['uv']).all()
    assert len(ids)==len(set(ids))==flat['covered_faces']
    assert flatten(run,{'chart_faces':128})['covered_faces']==flat['covered_faces']


def test_partial_atlas_resume_restores_missing_and_corrupt_companions(tmp_path):
    from test_field import prepared_run
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.field import solve
    from winding_field.meshes import mesh
    from winding_field.flatten import flatten
    run=prepared_run(tmp_path/'run')
    solve(run,{'sweeps':1,'field_chunk':8})
    mesh(run,{'levels':[.5,1.5,2.5],'mesh_tile':4})
    source=write_ct(tmp_path/'ct.zarr',np.full((6,6,24),128,np.uint8),spacing=(2.,3.,4.),origin=(10.,20.,30.))
    volume=Volume(source,cache=tmp_path/'cache')
    config={'chart_faces':128}
    assert flatten(run,config,volume,stop=lambda:True)['status']=='PAUSED'
    obj=next((run/'flat').glob('*-flat.obj')); png=next((run/'flat').glob('*.png'))
    expected_obj=obj.read_bytes();expected_png=png.read_bytes()
    obj.unlink();png.write_bytes(b'corrupted image')
    assert flatten(run,config,volume)['status']=='FLAT_READY'
    assert obj.read_bytes()==expected_obj
    assert png.read_bytes()==expected_png
    volume.close()
