import itertools

import numpy as np
import pytest

from winding_field.tifxyz_field import PhysicalField
from winding_field.surface_sources import FieldSource
from winding_field.util import file_sha


def field_array(tmp_path, function, *, shape=(12, 12, 12), spacing=(1., 1., 1.), origin=(0., 0., 0.), mask=None):
    z,y,x=np.indices(shape)
    xyz=np.stack((x,y,z),axis=-1)*spacing+origin
    path=tmp_path/'field.npy';np.save(path,function(xyz))
    descriptor=dict(kind='npy',path=str(path),sha256=file_sha(path))
    if mask is not None:
        p=tmp_path/'mask.npy';np.save(p,mask)
        descriptor['mask']=dict(path=str(p),sha256=file_sha(p))
    geometry=dict(frame='test',voxel_um=1.,origin_xyz=list(origin),spacing_xyz=list(spacing),shape_zyx=list(shape),sheet_step=1.)
    return PhysicalField(FieldSource(descriptor,geometry))


def reference(z=5., duplicate=False):
    from winding_field.uvw import Reference
    world=np.array([[2,2,z],[8,2,z],[8,8,z],[2,8,z]],float)
    uv=world[:,:2].copy();faces=np.array([[0,1,2],[0,2,3]])
    if duplicate:
        world=np.concatenate((world,world));uv=np.concatenate((uv,uv+20));faces=np.concatenate((faces,faces+4))
    return Reference(world,faces,uv,5.)


def test_physical_trilinear_gradient_anisotropic_origin(tmp_path):
    from winding_field.uvw import sample_w_gradient
    with field_array(tmp_path,lambda p: 2*p[...,0]-3*p[...,1]+.5*p[...,2],spacing=(2.,3.,4.),origin=(10.,-20.,30.)) as field:
        points=np.array([[13.1,-13.,36.],[20.,0.,50.]])
        values,gradient,valid=sample_w_gradient(field,points)
        assert valid.all()
        np.testing.assert_allclose(values,points@np.array([2,-3,.5]))
        np.testing.assert_allclose(gradient,np.tile([2,-3,.5],(2,1)))


def test_fractional_outward_and_back_plane(tmp_path):
    from winding_field.uvw import transport
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        points=np.array([[3.,4.,5.],[6.,7.,5.]])
        moved,valid,info=transport(field,points,[3.25,7.75])
        assert valid.all();assert max(info['normal_residual_um'])<1e-8
        np.testing.assert_allclose(moved,[[3,4,3.25],[6,7,7.75]],atol=1e-8)
        back,valid,_=transport(field,moved,5.)
        assert valid.all();np.testing.assert_allclose(back,points,atol=1e-8)


def test_support_hole_crossed_between_valid_endpoints(tmp_path):
    from winding_field.uvw import transport
    mask=np.ones((12,12,12),bool);mask[5,:,:]=False
    with field_array(tmp_path,lambda p:p[...,2],mask=mask) as field:
        _,valid,info=transport(field,[[3.,3.,2.]],8.)
        assert not valid[0];assert info['cause'][0]=='unsupported_path'


def test_zero_gradient_and_outside(tmp_path):
    from winding_field.uvw import transport
    with field_array(tmp_path,lambda p:np.ones(p.shape[:-1])) as field:
        _,valid,info=transport(field,[[3.,3.,2.],[30.,3.,2.]],2.)
        assert not valid.any();assert info['cause'][0]=='zero_gradient'


def test_cylinder_transport(tmp_path):
    from winding_field.uvw import transport
    with field_array(tmp_path,lambda p:np.hypot(p[...,0],p[...,1]),shape=(8,65,65),spacing=(.25,.25,.25),origin=(-8.,-8.,0.)) as field:
        points=np.array([[3.,4.,.8],[4.,-3.,1.]])
        moved,valid,info=transport(field,points,6.5)
        assert valid.all();np.testing.assert_allclose(np.linalg.norm(moved[:,:2],axis=1),6.5,atol=.01)
        back,valid,_=transport(field,moved,5.)
        assert valid.all();np.testing.assert_allclose(back,points,atol=.025)


def test_forward_inverse_reference_projection_and_domain(tmp_path):
    from winding_field.uvw import uvw_to_xyz,xyz_to_uvw
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        ref=reference(5.02)
        uvw=np.array([[3.,4.,3.5],[7.,6.,7.25],[20.,20.,5.]])
        xyz,valid=uvw_to_xyz(ref,field,uvw)
        assert valid.tolist()==[True,True,False]
        np.testing.assert_allclose(xyz[:2],uvw[:2],atol=1e-7)
        back,valid=xyz_to_uvw(ref,field,xyz[:2])
        assert valid.all();np.testing.assert_allclose(back,uvw[:2],atol=1e-7)
        _,valid=xyz_to_uvw(ref,field,[[10.,10.,5.]])
        assert not valid[0]


def test_inverse_rejects_ambiguous_reference(tmp_path):
    from winding_field.uvw import xyz_to_uvw
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        _,valid=xyz_to_uvw(reference(duplicate=True),field,[[3.,4.,6.]])
        assert not valid[0]


def test_transport_across_read_block_seam(tmp_path):
    from winding_field.uvw import transport
    with field_array(tmp_path,lambda p:p[...,0],shape=(4,4,70)) as field:
        xyz,valid,_=transport(field,[[62.5,1.,1.]],65.5)
        assert valid.all();np.testing.assert_allclose(xyz,[[65.5,1.,1.]])


def test_exact_trilinear_mixed_gradient(tmp_path):
    from winding_field.uvw import sample_w_gradient
    with field_array(tmp_path,lambda p:p[...,0]*p[...,1]*p[...,2]) as field:
        p=np.array([[2.3,3.7,4.1]])
        w,g,valid=sample_w_gradient(field,p)
        assert valid.all();np.testing.assert_allclose(w,np.prod(p,axis=1))
        np.testing.assert_allclose(g,np.prod(p,axis=1)[:,None]/p)


def test_projection_limit_and_band_mask(tmp_path):
    from winding_field.uvw import uvw_to_xyz,xyz_to_uvw
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        _,valid=uvw_to_xyz(reference(5.5),field,[[3.,4.,6.]])
        assert not valid[0]
        ref=reference();ref.band=(4.,6.)
        _,valid=uvw_to_xyz(ref,field,[[3.,4.,6.5]])
        assert not valid[0]
        _,valid=xyz_to_uvw(ref,field,[[3.,4.,6.5]])
        assert not valid[0]


@pytest.mark.parametrize('corner',[[1.,4.,5.],[2.5+1e-13,2.5-1e-13,5.]])
def test_sampled_fold_is_excluded(tmp_path,corner):
    from winding_field.uvw_pipeline import _sampled_cells
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        base=np.array([[[2.,3.,5.],[3.,3.,5.]],[[2.,2.,5.],[3.,2.,5.]]])
        folded=base.copy();folded[1,1]=corner
        valid,cells,bad,stats=_sampled_cells(folded,base,np.ones((2,2),bool),field)
        assert not valid.any();assert not cells.any();assert bad.all();assert stats['rejected_cells']==1


def test_excessive_tolerances_rejected():
    from winding_field.uvw import TransportSettings
    for options in ({'residual_fraction':.1},{'roundtrip_fraction':.5},{'projection_fraction':.5}):
        with pytest.raises(ValueError):TransportSettings(**options)


def test_forward_rejects_reference_tangent_to_flow(tmp_path):
    from winding_field.uvw import Reference,uvw_to_xyz
    world=np.array([[3,2,4.9],[3,8,4.9],[3,8,5.1],[3,2,5.1]])
    ref=Reference(world,np.array([[0,1,2],[0,2,3]]),np.array([[2,2],[8,2],[8,8],[2,8]]),5.)
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        _,valid=uvw_to_xyz(ref,field,[[5.,5.,6.]])
        assert not valid[0]


def test_overflowing_gradient_magnitude_is_invalid(tmp_path):
    from winding_field.uvw import transport
    with field_array(tmp_path,lambda p:p[...,2]*1e200) as field:
        with np.errstate(over='ignore',invalid='ignore'):
            _,valid,_=transport(field,[[3.,3.,2.]],8e200)
        assert not valid[0]


def test_sampled_jacobian_uses_physical_uv_pitch(tmp_path):
    from winding_field.uvw_pipeline import _sampled_cells
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        xyz=np.array([[[2.,4.,5.],[4.,4.,5.]],[[2.,2.,5.],[4.,2.,5.]]])
        _,_,_,stats=_sampled_cells(xyz,xyz,np.ones((2,2),bool),field,pitch_um=2.)
        assert stats['min_abs_jacobian_um_per_w']==pytest.approx(1.)


def test_inverse_rejects_curved_projection_branch_missed_by_normal_line(tmp_path):
    from winding_field.uvw import Reference,transport,reference_points,xyz_to_uvw
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        start=np.array([3.,4.,2.]);other,valid,_=transport(field,start[None],6.7)
        assert valid.all()
        triangles=[]
        for center in (start,other[0]):
            gradient=np.array([center[2],0.,center[0]])
            tangent=np.cross(gradient,[0.,1.,0.]);tangent/=np.linalg.norm(tangent)
            a=np.array([0.,1e-4,0.]);b=tangent*1e-4
            triangles.append(np.array([center+a,center+b,center-a-b]))
        uv=np.array([[0.,0.],[1.,0.],[0.,1.]])
        ref=Reference(np.concatenate(triangles),np.array([[0,1,2],[3,4,5]]),np.concatenate((uv,uv+10)),6.)
        base,admitted=reference_points(ref,field,[[1/3,1/3],[10+1/3,10+1/3]])
        assert admitted.all();assert np.linalg.norm(base[0]-base[1])<1e-3
        _,valid=xyz_to_uvw(ref,field,start[None])
        assert not valid[0]


def test_forward_rejects_duplicate_reference_charts(tmp_path):
    from winding_field.uvw import uvw_to_xyz
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        xyz,valid=uvw_to_xyz(reference(duplicate=True),field,[[3.,4.,6.],[23.,24.,6.]])
        assert not valid.any();assert np.isnan(xyz).all()


def test_sampled_orientation_follows_rotating_reference_gradient(tmp_path):
    from winding_field.uvw import transport
    from winding_field.uvw_pipeline import _sampled_cells
    with field_array(tmp_path,lambda p:np.arctan2(p[...,1],p[...,0]),shape=(16,65,65),spacing=(.25,.25,.25),origin=(-8.,-8.,0.)) as field:
        base=np.array([[[4.,0.,2.],[5.,0.,2.]],[[4.,0.,1.],[5.,0.,1.]]])
        xyz,valid,_=transport(field,base.reshape(-1,3),2.)
        assert valid.all()
        back,good,_=transport(field,xyz,0.)
        assert good.all();assert np.linalg.norm(back-base.reshape(-1,3),axis=1).max()<.0625
        accepted,cells,_,stats=_sampled_cells(xyz.reshape(2,2,3),base,valid.reshape(2,2),field)
        assert accepted.all() and cells.all();assert stats['rejected_cells']==0
