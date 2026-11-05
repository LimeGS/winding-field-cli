import numpy as np
import pytest
from winding_field import original_recipe as recipe


@pytest.mark.parametrize('offset',[0,7])
def test_filtered_raster_preserves_geometry_masks_and_original_owners(monkeypatch,offset):
    uv=np.array([[0,0],[10,0],[0,10],[30,30],[40,30],[30,40],
                 [2,2],[4,2],[2,4],[np.nan,0],[1,0],[0,1]],float)
    faces=np.array([[3,4,5],[0,1,2],[6,8,7],[9,10,11],[0,0,1],[0,1,2]])
    vertices=np.column_stack((uv,np.zeros(len(uv))))
    eligible=np.array([True,True,False,True,True,True])
    canvas=dict(width=8,height=8,pitch=1.,u0=float(offset),v_top=10.,
                global_u0=0.,global_v_top=10.,column_offset=offset,row_offset=0)
    def render():
        captured=[]
        image,mask,_=recipe.raster_sheet(vertices,faces,uv,eligible,
            lambda xyz:(xyz[:,0]+xyz[:,1],np.ones(len(xyz),bool)),canvas,
            capture=lambda *values:captured.extend(v.copy() for v in values))
        return [image,mask,*captured]
    selected=list(recipe.raster_candidates(faces,uv,canvas))
    assert 0 not in selected and 1 in selected and 3 not in selected
    filtered=render()
    monkeypatch.setattr(recipe,'raster_candidates',lambda f,uv,canvas,guard=lambda:None:range(len(f)))
    for a,b in zip(filtered,render()):np.testing.assert_array_equal(a,b)


def test_candidates_keep_exact_boundary_and_call_guard():
    uv=np.array([[0.,0.],[1.,0.],[0.,1.],[1e6,1e6]])
    faces=np.array([[0,1,2],[3,3,3]])
    called=[]
    result=list(recipe.raster_candidates(faces,uv,dict(width=2,height=2,pitch=1.,u0=0.,v_top=1.),lambda:called.append(True)))
    assert result==[0] and called
