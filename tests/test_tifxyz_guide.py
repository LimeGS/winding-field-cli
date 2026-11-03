from types import SimpleNamespace
import numpy as np
import pytest


class ArrayField:
    def __init__(self,values,valid=None,spacing=(1.,1.,1.)):
        self.values=np.asarray(values,float);self.support=np.ones(self.values.shape,bool) if valid is None else valid
        self.shape=self.values.shape;self.spacing_um=np.array(spacing);self.origin_um=np.zeros(3)
        self.identity={'fixture':'array','shape':list(self.shape)};self.reads=[]
    def read(self,lo,hi):
        self.reads.append((tuple(lo),tuple(hi)))
        assert max(b-a for a,b in zip(lo,hi))<=65
        sl=tuple(slice(a,b) for a,b in zip(lo,hi));return self.values[sl].copy(),self.support[sl].copy()
    def validate(self,hashes=False):pass


def test_stride_schedule_is_refinable_and_never_native():
    from winding_field.tifxyz_guide import stride_schedule
    from winding_field.tifxyz_contract import Blocked
    assert stride_schedule((8193,8193,8193))==[256,128,64]
    assert all(s>=2 for s in stride_schedule((65,65,65)))
    with pytest.raises(Blocked):stride_schedule((2,2,2))


def test_coarse_support_checks_intervening_fine_nodes(tmp_path):
    from winding_field.tifxyz_field import build_coarse,load_coarse
    z,y,x=np.indices((9,9,9));support=np.ones(x.shape,bool);support[1,1,1]=False
    coarse=build_coarse(ArrayField(x,support),4,tmp_path/'coarse',stop=None)
    values,cells,axes=load_coarse(coarse)
    assert values.shape==(3,3,3) and cells.shape==(2,2,2)
    assert not cells[0,0,0] and cells[1,1,1]
    assert values[2,2,2]==8 and axes[0].tolist()==[0,4,8]


def test_guide_sampling_includes_roi_end_node(tmp_path):
    from winding_field.tifxyz_field import build_coarse,load_coarse
    x=np.indices((8,10,12))[2]
    coarse=build_coarse(ArrayField(x),4,tmp_path/'coarse',stop=None)
    values,cells,axes=load_coarse(coarse)
    assert [a[-1] for a in axes]==[7,9,11]
    assert values[-1,-1,-1]==11 and cells.all()


def test_coarse_stop_and_corruption_do_not_become_guide(tmp_path):
    from winding_field.tifxyz_field import build_coarse,load_coarse
    from winding_field.tifxyz_contract import Paused
    field=ArrayField(np.indices((9,9,9))[2])
    with pytest.raises(Paused):build_coarse(field,4,tmp_path/'stopped',stop=lambda:True)
    assert not (tmp_path/'stopped/COARSE.json').exists()
    coarse=build_coarse(field,4,tmp_path/'good',stop=None)
    (tmp_path/'good/values.npy').write_bytes(b'broken')
    with pytest.raises(ValueError):load_coarse(coarse)


def test_plane_guide_keeps_geometry_and_is_one_global_disk(tmp_path):
    pytest.importorskip('igl')
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_guide import build_guide,validate_map
    field=ArrayField(np.indices((17,17,17))[2])
    coarse=build_coarse(field,4,tmp_path/'coarse',stop=None)
    guide=build_guide(coarse,8.5,None,None)
    assert len(guide['faces'])==32 and guide['stride']==4
    np.testing.assert_allclose(guide['world'][:,0],8.5)
    assert np.min(guide['normals'][:,0])>.99
    validate_map(guide)


def test_two_components_not_silently_reduced_to_largest(tmp_path):
    pytest.importorskip('igl')
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_guide import build_guide
    from winding_field.tifxyz_contract import Blocked
    x=np.indices((17,17,17))[2];field=ArrayField((x-4.5)*(x-12.5))
    coarse=build_coarse(field,2,tmp_path/'coarse',stop=None)
    with pytest.raises(Blocked,match='COMPONENT'):build_guide(coarse,0.,None,None)


def test_uv_checks_shared_edge_orientation_and_star_boundary():
    pytest.importorskip('igl')
    from winding_field.tifxyz_guide import validate_map
    uv=np.array([[0.,0.],[4.,0.],[4.,4.],[0.,4.],[1.,1.],[3.,1.],[2.,.5]])
    f=np.array([[0,1,5],[0,5,4],[1,2,5],[2,6,5],[2,3,6],[3,4,6],[3,0,4],[4,6,5]])
    with pytest.raises(ValueError,match='orientation'):validate_map(dict(world=np.column_stack([uv,np.zeros(7)]),faces=f,uv=uv))


@pytest.mark.parametrize('kind',['sphere','torus','cylinder'])
def test_curved_connected_guides_are_single_global_disks(tmp_path,kind):
    pytest.importorskip('igl')
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_guide import build_guide,validate_map
    z,y,x=np.indices((33,33,33),dtype=float);x-=16;y-=16;z-=16
    values=x*x+y*y+z*z-64 if kind=='sphere' else (np.sqrt(x*x+y*y)-10)**2+z*z-9 if kind=='torus' else x*x+y*y-64
    guide=build_guide(build_coarse(ArrayField(values),2,tmp_path/'coarse'),0.)
    validate_map(guide)
    assert len(guide['faces'])>100 and np.isfinite(guide['normals']).all()


def test_nonmanifold_and_star_rejected():
    pytest.importorskip('igl')
    from winding_field.tifxyz_guide import parameterize,validate_map
    from winding_field.tifxyz_contract import Blocked
    world=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]])
    with pytest.raises(Blocked,match='NONMANIFOLD'):parameterize(world,np.array([[0,1,2],[1,0,3],[0,1,4]]))
    angles=np.arange(5)*2*np.pi/5
    world=np.vstack([[0,0,0],np.column_stack([np.cos(angles),np.sin(angles),np.zeros(5)])])
    faces=np.array([[0,i+1,(i+1)%5+1] for i in range(5)])
    uv=np.vstack([[0,0],np.column_stack([np.cos(2*angles),np.sin(2*angles)])])
    with pytest.raises(ValueError,match='boundary'):validate_map(dict(world=world,faces=faces,uv=uv))


def test_voxel_limit_precedes_native_field_read(tmp_path):
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_contract import Blocked
    field=SimpleNamespace(shape=(8193,8193,8193))
    with pytest.raises(Blocked,match='VOXEL_LIMIT'):build_coarse(field,2,tmp_path/'coarse')
    assert not (tmp_path/'coarse').exists()


def test_voxel_limit_precedes_coarse_axis_allocation(tmp_path):
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_contract import Blocked
    with pytest.raises(Blocked,match='VOXEL_LIMIT'):
        build_coarse(SimpleNamespace(shape=(2,2,2**62)),2,tmp_path/'huge')
