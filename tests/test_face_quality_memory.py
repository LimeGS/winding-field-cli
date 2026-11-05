"""Exact frozen quality semantics and bounded workspace, including block boundaries."""
import numpy as np
import pytest


def frozen_quality(vertices,faces,uv):
    # Pre-memory2 implementation: keep the global reduction order as the oracle.
    v,f,uv=np.asarray(vertices,float),np.asarray(faces,np.int64),np.asarray(uv,float)
    edges=np.stack([v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]],axis=-1)
    t=uv[f];matrix=np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=-1)
    det=np.linalg.det(matrix);valid=np.isfinite(matrix).all((1,2))&(det>0)
    singular=np.full((len(f),2),np.nan)
    singular[valid]=np.linalg.svd(edges[valid]@np.linalg.inv(matrix[valid]),compute_uv=False)
    ratio=np.full(len(f),np.inf);positive=valid&(singular[:,1]>0)
    ratio[positive]=singular[positive,0]/singular[positive,1]
    display=positive&(ratio<=1.25)&(singular[:,0]<=2.)&(singular[:,1]>=.5)
    error=np.full(len(f),np.inf);error[positive]=np.max(abs(singular[positive]-1),axis=1)
    scientific=positive&(error<=.02)
    area=np.linalg.norm(np.cross(edges[:,:,0],edges[:,:,1]),axis=1)/2
    finite=error[np.isfinite(error)]
    report=dict(scientific_limit=.02,scientific_acceptance=False,maximum_visual_scale_factor=2.,
        maximum_visual_anisotropy=1.25,display_area_fraction=float(area[display].sum()/area.sum()),
        scientific_area_fraction=float(area[scientific].sum()/area.sum()),
        inverted_or_invalid_area_fraction=float(area[~positive].sum()/area.sum()),
        p95_valid_face_metric_distortion=float(np.percentile(finite,95)) if len(finite) else None)
    return display,scientific,report


def quality_fixture(n):
    rng=np.random.default_rng(417)
    v=np.zeros((n,3,3));v[:,1,0]=rng.uniform(.1,3,n);v[:,2,1]=rng.uniform(.1,3,n)
    v[:,1,1:]=rng.uniform(-.1,.1,(n,2));v[:,2,::2]=rng.uniform(-.1,.1,(n,2))
    uv=v[:,:,:2].copy();uv[:,1,0]*=rng.uniform(.4,2,n);uv[:,2,1]*=rng.uniform(.4,2,n)
    # Every block contains inverted/zero/nonfinite UV, collapsed XYZ, and metric thresholds.
    for k in range(0,n-8,97):
        v[k:k+9]=[[0,0,0],[1,0,0],[0,1,0]]
        uv[k:k+9]=[[0,0],[1,0],[0,1]]
        uv[k+1,1,0]=-1;uv[k+2,1]=0;uv[k+3,1,0]=np.nan
        v[k+4,2]=v[k+4,1]
        uv[k+5: k+9,1,0]=[1/1.02,1/1.25,.5,2.]
    return v.reshape(-1,3),np.arange(n*3).reshape(-1,3),uv.reshape(-1,2)


@pytest.mark.parametrize('n',[0,1,4096,8201])
def test_quality_exact_frozen_masks_reductions_and_percentile(n):
    from winding_field.original_recipe import face_quality
    v,f,uv=quality_fixture(n)
    with np.errstate(invalid='ignore',divide='ignore'):
        expected=frozen_quality(v,f,uv);actual=face_quality(v,f,uv)
    for a,b in zip(actual[:2],expected[:2]):np.testing.assert_array_equal(a,b)
    assert actual[2].keys()==expected[2].keys()
    for key,value in expected[2].items():
        if isinstance(value,float) and np.isnan(value):assert np.isnan(actual[2][key])
        else:assert actual[2][key]==value,key


def test_quality_degenerate_only_retains_nan_area_and_no_percentile():
    from winding_field.original_recipe import face_quality
    with np.errstate(invalid='ignore',divide='ignore'):
        display,scientific,report=face_quality(np.zeros((3,3)),[[0,1,2]],np.zeros((3,2)))
    assert display.tolist()==scientific.tolist()==[False]
    assert report['p95_valid_face_metric_distortion'] is None
    assert all(np.isnan(report[k]) for k in ('display_area_fraction','scientific_area_fraction','inverted_or_invalid_area_fraction'))


def test_quality_bounds_linalg_workspace_across_multiple_blocks(monkeypatch):
    from winding_field.original_recipe import face_quality
    for name in ('det','inv','svd'):
        original=getattr(np.linalg,name)
        def bounded(a,*args,_fn=original,**kwargs):
            assert len(a)<=4096,'unbounded quality matrix workspace'
            return _fn(a,*args,**kwargs)
        monkeypatch.setattr(np.linalg,name,bounded)
    with np.errstate(invalid='ignore'):
        result=face_quality(*quality_fixture(8201))
    assert len(result[0])==8201 and result[0].any() and (~result[0]).any()


def test_export_releases_source_arrays_before_real_raster(tmp_path,monkeypatch):
    import weakref
    from contextlib import contextmanager
    from test_patch_mapping import source
    from winding_field import patch_pipeline as p
    from winding_field.patch_mapping import PatchMapping
    reader,ids=source();reader.metadata={'voxel_um':1.};reader.validate=lambda:None
    reader.db.executescript('CREATE TABLE levels(id INTEGER,value REAL); INSERT INTO levels VALUES(0,0);')
    @contextmanager
    def opened(path):yield reader,dict(kind='test_mesh')
    monkeypatch.setattr(p,'open_mesh',opened)
    monkeypatch.setattr(p,'check_flatboi',lambda *args:{})
    monkeypatch.setattr(p,'flatten_native',lambda v,f,*args,**kwargs:(v[:,:2].copy(),{}))
    refs=[];read=p.read_level;frame=p.frame_coordinates;render=p.render_map
    def tracked_read(*args,**kwargs):
        result=read(*args,**kwargs);refs.extend(weakref.ref(a) for a in result);return result
    def tracked_frame(*args,**kwargs):
        result=frame(*args,**kwargs);refs.append(weakref.ref(result));return result
    def checked_render(*args,**kwargs):
        assert all(ref() is None for ref in refs),'source arrays still live during raster'
        return render(*args,**kwargs)
    monkeypatch.setattr(p,'read_level',tracked_read);monkeypatch.setattr(p,'frame_coordinates',tracked_frame)
    monkeypatch.setattr(p,'render_map',checked_render)
    class CT:
        chunks=(2,2,2);dtype=np.dtype('uint8');identity={'synthetic':True}
        def sample(self,xyz,frame):return np.full(len(xyz),100.),np.ones(len(xyz),bool)
    cfg=dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/test',pitch_um=.25,tile_size=8,memory_gib=.5)
    out=tmp_path/'export';receipt=p.export_mesh(tmp_path/'source',out,CT(),cfg)
    assert receipt['status']=='PATCHES_READY_DIAGNOSTIC'
    with PatchMapping(out/'level-0000') as mapping:
        assert mapping.arrays['source_vertex_ids'].tolist()==ids
        assert [i for i,key in mapping.iter_keys('vertices')]==ids
        np.testing.assert_array_equal(mapping.arrays['field_xyz'],[[0,0,1],[1,0,1],[0,1,1]])
    reader.db.close()
