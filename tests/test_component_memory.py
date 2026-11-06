"""Frozen adjacency/order oracle and allocation/lifetime regressions."""
import gc
import tracemalloc
import weakref
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def frozen_components(faces,ids):
    # Exact pre-memory3 code, including tie order and nonmanifold adjacency.
    f=faces[ids]
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    owners=np.tile(np.arange(len(f)),3)
    order=np.lexsort((edges[:,1],edges[:,0]))
    equal=(edges[order][1:]==edges[order][:-1]).all(1)
    a,b=owners[order][:-1][equal],owners[order][1:][equal]
    graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(f),len(f)))
    count,labels=connected_components(graph,directed=False)
    return sorted([ids[labels==k] for k in range(count)],key=lambda x:int(x.min()))


@pytest.mark.parametrize('ids',[[],[0],[5,4,3,2,1,0],[3,0,5,1,4,2],[4,4,2,0]])
def test_component_ids_order_disconnected_nonmanifold_degenerate(ids):
    from winding_field.surface_atlas import face_components
    faces=np.array([[0,1,2],[1,0,3],[0,1,4],[10,11,12],[10,10,11],[20,20,20]])
    ids=np.array(ids,np.int64)
    old=frozen_components(faces,ids);new=face_components(faces,ids)
    assert len(new)==len(old)
    for a,b in zip(new,old):np.testing.assert_array_equal(a,b)
    if ids.tolist()==[3,0,5,1,4,2]:assert [x.tolist() for x in new]==[[0,1,2],[3,4],[5]]


def test_component_workspace_reduced_on_multiblock_strip():
    from winding_field.surface_atlas import face_components
    n=32768;rows=np.arange(n);faces=np.column_stack((rows,rows+1,rows+2));ids=rows[::-1]
    def measured(fn):
        gc.collect();tracemalloc.start()
        try:
            result=fn(faces,ids);peak=tracemalloc.get_traced_memory()[1]
        finally:tracemalloc.stop()
        return result,peak
    old,before=measured(frozen_components);new,after=measured(face_components)
    assert len(old)==len(new)==1
    np.testing.assert_array_equal(new[0],old[0])
    assert after<before*.85,(before,after)


def test_seam_targets_released_before_quality_and_geometry_preserved(tmp_path,monkeypatch):
    from winding_field import patch_pipeline as p
    side=9;y,x=np.mgrid[:side,:side];v=np.column_stack((x.ravel(),y.ravel(),np.ones(side*side)))
    a=np.array([j*side+i for j in range(side-1) for i in range(side-1)])
    f=np.stack((np.column_stack((a,a+1,a+side+1)),np.column_stack((a,a+side+1,a+side))),axis=1).reshape(-1,3)
    refs=[];targets=p.seam_targets;quality=p.face_quality
    def tracked(*args,**kwargs):
        result=targets(*args,**kwargs)
        refs.extend(weakref.ref(value) for chart in result.values() for value in chart.values())
        return result
    def checked(*args,**kwargs):
        assert refs and all(ref() is None for ref in refs),'seam targets retained after weld'
        return quality(*args,**kwargs)
    monkeypatch.setattr(p,'seam_targets',tracked);monkeypatch.setattr(p,'face_quality',checked)
    result=p.build_map(v,f,tmp_path,flatten=lambda v,f,path:(v[:,:2].copy(),{}),max_patch_faces=16)
    np.testing.assert_array_equal(result['world'],v[result['parent']])
    np.testing.assert_array_equal(result['parent'][result['faces']],f[result['source_face_rows']])
    assert sorted(result['source_face_rows'])==list(range(len(f)))
    assert result['report']['attempts'] and result['report']['registrations']
