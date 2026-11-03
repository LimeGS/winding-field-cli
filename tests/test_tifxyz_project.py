from types import SimpleNamespace
import numpy as np
import pytest


def field(values,spacing=(1,1,1),support=None,origin=(0,0,0)):
    from winding_field.tifxyz_field import PhysicalField
    data=np.asarray(values,float);known=np.ones(data.shape,bool) if support is None else support
    reads=[]
    def read(lo,hi):
        reads.append((lo,hi));sl=tuple(slice(a,b) for a,b in zip(lo,hi));return data[sl].copy(),known[sl].copy()
    source=SimpleNamespace(shape=data.shape,origin=np.array(origin,float),spacing=np.array(spacing,float),voxel_um=1.,
        identity={'fixture':'root'},read=read,validate=lambda **kw:None,close=lambda:None)
    result=PhysicalField(source);result.read_log=reads;return result


def test_three_roots_inside_one_trilinear_cell_are_all_found():
    from winding_field.tifxyz_project import ray_roots
    z,y,x=np.indices((2,2,2),dtype=float);f=field((x-.25)*(y-.5)*(z-.75))
    result=ray_roots(f,[0,0,0],[1,1,1],(0,2),1e-6,level=0.)
    assert result['status']=='AMBIGUOUS' and result['reason']=='MULTIPLE_ROOTS'
    assert len(result['roots'])==3
    np.testing.assert_allclose([r['xyz_um'] for r in result['roots']],[[.25]*3,[.5]*3,[.75]*3],atol=1e-6)


@pytest.mark.parametrize('reversed_ray',[False,True])
def test_simple_root_on_shared_cell_face_counted_once(reversed_ray):
    from winding_field.tifxyz_project import ray_roots
    f=field(np.indices((2,2,3))[2]-1.)
    origin=[2,.5,.5] if reversed_ray else [0,.5,.5];direction=[-1,0,0] if reversed_ray else [1,0,0]
    result=ray_roots(f,origin,direction,(0,2),1e-6,level=0.)
    assert result['status']=='UNIQUE' and len(result['roots'])==1
    np.testing.assert_allclose(result['roots'][0]['xyz_um'],[1,.5,.5])


def test_cusp_is_not_a_simple_root_and_plateau_is_ambiguous():
    from winding_field.tifxyz_project import ray_roots
    f=field(abs(np.indices((2,2,3))[2]-1.))
    assert ray_roots(f,[0,.5,.5],[1,0,0],(0,2),1e-6,level=0)['status']=='AMBIGUOUS'
    f=field(np.zeros((2,2,2)))
    assert ray_roots(f,[0,.5,.5],[1,0,0],(0,1),1e-6,level=0)['status']=='AMBIGUOUS'


def test_double_root_with_no_endpoint_sign_change_is_detected():
    from winding_field.tifxyz_project import ray_roots
    z,y,x=np.indices((2,2,2),dtype=float);f=field((x-.5)*(y-.5)*(z+.25))
    assert ray_roots(f,[0,0,0],[1,1,1],(0,2),1e-6,level=0.)['status']=='AMBIGUOUS'


def test_close_distinct_roots_are_not_distance_deduplicated():
    from winding_field.tifxyz_project import ray_roots
    z,y,x=np.indices((2,2,2),dtype=float);f=field((x-.25)*(y-.2501)*(z+.25))
    result=ray_roots(f,[0,0,0],[1,1,1],(0,2),.001,level=0.)
    assert result['status']=='AMBIGUOUS' and result['reason']=='MULTIPLE_ROOTS'


def test_roi_endpoint_has_explicit_one_sided_classification():
    from winding_field.tifxyz_project import ray_roots
    f=field(np.indices((2,2,3))[2]-2.)
    result=ray_roots(f,[0,.5,.5],[1,0,0],(0,3),1e-6,level=0.)
    assert result['status']=='UNIQUE' and result['roots'][0]['one_sided']
    np.testing.assert_allclose(result['roots'][0]['xyz_um'],[2,.5,.5])


def test_curved_sheet_uses_fine_values_not_guide_interpolation():
    from winding_field.tifxyz_project import ray_roots
    z,y,x=np.indices((5,5,5),dtype=float);f=field(z-.1*x*y)
    result=ray_roots(f,[2,3,1],[0,0,1],(-1,3),1e-6,level=0.)
    assert result['status']=='UNIQUE'
    root=result['roots'][0]
    np.testing.assert_allclose(root['xyz_um'],[2,3,.6],atol=1e-6)
    assert root['error_um']<=1.1e-6


def test_missing_support_does_not_become_no_root():
    from winding_field.tifxyz_project import ray_roots
    x=np.indices((2,2,3))[2];known=np.ones(x.shape,bool);known[0,0,0]=False
    result=ray_roots(field(x-1.5,support=known),[0,.5,.5],[1,0,0],(0,2),1e-6,level=0.)
    assert result['status']=='UNSUPPORTED'


def test_ray_work_limit_checked_before_field_reads():
    from winding_field.tifxyz_project import ray_roots
    from winding_field.tifxyz_contract import Blocked
    f=field(np.zeros((2,2,5000)))
    with pytest.raises(Blocked,match='RAY_WORK_LIMIT'):ray_roots(f,[0,.5,.5],[1,0,0],(0,4999),.001,level=0.)
    assert not f.read_log


def test_anisotropic_plane_and_zero_direction_component():
    from winding_field.tifxyz_project import ray_roots
    f=field(np.indices((3,3,2001))[2]-1000.25,spacing=(1,1,100))
    r=ray_roots(f,[1000,1,100],[1,0,0],(-400,400),1e-5,level=0.)
    assert r['status']=='UNIQUE'
    np.testing.assert_allclose(r['roots'][0]['xyz_um'],[1000.25,1,100],atol=1e-5)


def test_interval_evaluation_covers_affine_coordinate_roundoff():
    from fractions import Fraction
    from winding_field.tifxyz_project import segment_value_interval
    values=np.indices((2,2,2),dtype=float)[2]
    a=np.array([.5,.5,.9]);b=np.array([.5,.5,1e-20]);t=float(np.nextafter(1.,0.))
    low,high=segment_value_interval(values,a,b,t,0.)
    exact=Fraction(float(a[2]))+Fraction(t)*(Fraction(float(b[2]))-Fraction(float(a[2])))
    assert Fraction(low)<=exact<=Fraction(high)


def test_projected_grid_is_independent_of_tile_size():
    pytest.importorskip('igl')
    from winding_field.tifxyz_project import make_grid,project_tile
    xyz=np.array([[5.,0.,0.],[5.,140.,0.],[5.,140.,140.],[5.,0.,140.]])
    guide=dict(world=xyz,faces=np.array([[0,1,2],[0,2,3]]),uv=np.array([[0.,0.],[140.,0.],[140.,140.],[0.,140.]]),
               normals=np.tile([1.,0.,0.],(4,1)),stride=2)
    f=field(np.indices((141,141,9))[2]-4.5)
    frame=dict(id='target',origin_xyz_um=[0,0,0],voxel_um=1.,shape_zyx=[200]*3,source_identity_sha256='a'*64)
    settings=SimpleNamespace(output_frame=frame)
    grid=make_grid(guide,1.,30000,level=0.)
    a=project_tile(f,guide,grid,(0,0),settings,None)
    assert len(f.read_log)<100,'projection must fill bounded cell batches, not reread metadata per128rays'
    small=dict(grid,tile_pixels=128)
    b=project_tile(f,guide,small,(0,0),settings,None)
    np.testing.assert_array_equal(a['xyz'][:129,:129],b['xyz'])
    assert a['mask'].all()
    np.testing.assert_allclose(a['xyz'][...,0],4.5)


def test_projection_absence_blocks_instead_of_masking():
    pytest.importorskip('igl')
    from winding_field.tifxyz_project import make_grid,project_tile
    from winding_field.tifxyz_contract import Blocked
    guide=dict(world=np.array([[1.,0.,0.],[1.,0.,4.],[1.,4.,4.],[1.,4.,0.]]),
        faces=np.array([[0,1,2],[0,2,3]]),uv=np.array([[0.,0.],[4.,0.],[4.,4.],[0.,4.]]),
        normals=np.tile([1.,0.,0.],(4,1)),stride=2)
    f=field(np.ones((5,5,5)));grid=make_grid(guide,1.,100,level=0.)
    with pytest.raises(Blocked,match='PROJECTION_ABSENT'):project_tile(f,guide,grid,(0,0),None,None)


def test_uv_orphans_are_removed_before_querying_roots():
    pytest.importorskip('igl')
    from winding_field.tifxyz_project import make_grid,project_tile
    guide=dict(world=np.array([[1.,0.,0.],[1.,1.,0.],[1.,0.,1.]]),faces=np.array([[0,1,2]]),
               uv=np.array([[0.,0.],[1.,0.],[0.,1.]]),normals=np.tile([1.,0.,0.],(3,1)),stride=2)
    f=field(np.ones((3,3,3)));grid=make_grid(guide,1.,100,level=0.)
    tile=project_tile(f,guide,grid,(0,0),None,None)
    assert not tile['mask'].any() and not f.read_log


def test_near_tangent_physical_ray_is_not_lost_by_grid_rounding():
    from fractions import Fraction
    from winding_field.tifxyz_project import ray_roots
    z,y,x=np.indices((2,2,2),dtype=float);f=field(x-y,spacing=(1.3,)*3)
    origin=np.array([0.6499999999999999,.65,.65]);direction=np.array([.9999999999999967,1.,0.])
    unit=direction/np.linalg.norm(direction)
    expected=-(Fraction(float(origin[0]))-Fraction(float(origin[1])))/(Fraction(float(unit[0]))-Fraction(float(unit[1])))
    r=ray_roots(f,origin,direction,(-.060619047619047614,-.03461904761904762),.0013,level=0.)
    assert r['status']=='UNIQUE'
    lo,hi=r['roots'][0]['t_bounds']
    assert Fraction(lo)<=expected<=Fraction(hi)


def test_grid_rounding_cannot_create_a_physical_ray_root():
    from winding_field.tifxyz_project import ray_roots
    z,y,x=np.indices((2,2,2),dtype=float);f=field(x-y,spacing=(1.3,)*3)
    r=ray_roots(f,[.6500000000000005,.65,.65],[1.000000000000001,1.,0.],(-1.3/3,1.3/3),.0013,level=0.)
    assert r['status']=='ABSENT'


@pytest.mark.parametrize('reverse',[False,True])
def test_internal_face_cusp_at_search_endpoint_is_not_simple(reverse):
    from winding_field.tifxyz_project import ray_roots
    f=field(abs(np.indices((2,2,3))[2]-1.))
    r=ray_roots(f,[2 if reverse else 0,.5,.5],[-1 if reverse else 1,0,0],(0,1),1e-6,level=0.)
    assert r['status']=='AMBIGUOUS'


def test_float32_encoding_stays_inside_exact_physical_roi():
    pytest.importorskip('igl')
    from fractions import Fraction as Q
    from winding_field.tifxyz_project import make_grid,project_tile
    origin=[.30000000000000004]*3;f=field(np.indices((5,5,5))[2]-1.5,spacing=(.1,)*3,origin=origin)
    lo=f.origin_um;hi=lo+4*f.spacing_um;x=lo[0]+1.5*.1
    guide=dict(world=np.array([[x,lo[1],lo[2]],[x,hi[1],lo[2]],[x,hi[1],hi[2]],[x,lo[1],hi[2]]]),
        faces=np.array([[0,1,2],[0,2,3]]),uv=np.array([[0.,0.],[.4,0.],[.4,.4],[0.,.4]]),
        normals=np.tile([1.,0.,0.],(4,1)),stride=2)
    frame=dict(id='decimal-frame',origin_xyz_um=[0,0,0],voxel_um=.1,shape_zyx=[100]*3,source_identity_sha256='a'*64)
    tile=project_tile(f,guide,make_grid(guide,.1,100,level=0.),(0,0),SimpleNamespace(output_frame=frame))
    assert tile['mask'].any()
    for p in tile['xyz'][tile['mask']]:
        for axis,value in enumerate(p):
            exact=Q(float(value))*Q(.1)
            assert Q(float(lo[axis]))<=exact<=Q(float(lo[axis]))+4*Q(.1)


@pytest.fixture
def planar_precheck(monkeypatch):
    from winding_field import tifxyz_project as code
    guide=dict(world=np.array([[0.,0.,.5],[9.,0.,.5],[9.,9.,.5],[0.,9.,.5]]),
        faces=np.array([[0,1,2],[0,2,3]]),uv=np.array([[0.,0.],[9.,0.],[9.,9.],[0.,9.]]),
        normals=np.tile([0.,0.,1.],(4,1)),stride=2)
    def samples(guide,grid,low,high):
        rows,cols=np.mgrid[low[0]:high[0],low[1]:high[1]]
        r=rows.astype(np.longdouble)/9;c=cols.astype(np.longdouble)/9
        ids=(rows>cols).astype(int)
        weights=np.where(ids[...,None]==0,np.stack([1-c,c-r,r],-1),np.stack([1-r,c,r-c],-1))
        return rows,cols,ids,weights,np.ones(rows.shape,bool)
    monkeypatch.setattr(code,'_uv_samples',samples)
    return guide,code.make_grid(guide,1.,100,level=0.)


def test_boundary_precheck_samples_at_most64_real_mandatory_rays(planar_precheck,monkeypatch):
    from winding_field import tifxyz_project as code
    from winding_field.tifxyz_project import precheck_projection,project_tile
    guide,grid=planar_precheck;f=field(np.indices((3,10,10))[0]-.5)
    result=precheck_projection(f,guide,grid)
    assert result['status']=='NO_REJECTION_WITNESS'
    assert result['visible_nodes']==100 and result['selected_nodes']==result['evaluated_nodes']==64
    assert len({tuple(s['pixel_yx']) for s in result['samples']})==64
    frame=dict(id='target',origin_xyz_um=[0,0,0],voxel_um=1.,shape_zyx=[20]*3,source_identity_sha256='a'*64)
    rays=[];original=code.ray_batch
    def batch(field,origins,directions,*args,**kwargs):
        rays.extend((o.copy(),d.copy()) for o,d in zip(origins,directions))
        return original(field,origins,directions,*args,**kwargs)
    monkeypatch.setattr(code,'ray_batch',batch)
    tile=project_tile(f,guide,grid,(0,0),SimpleNamespace(output_frame=frame))
    for sample in result['samples']:
        y,x=sample['pixel_yx']
        assert tile['mask'][y,x] and sample['status']=='UNIQUE'
        np.testing.assert_array_equal(sample['origin_um'],rays[y*10+x][0])
        np.testing.assert_array_equal(sample['direction'],rays[y*10+x][1])
        np.testing.assert_array_equal(tile['xyz'][y,x],[float(x),float(y),.5])


def test_boundary_precheck_rejection_is_confirmed_by_the_full_projector(planar_precheck):
    import json
    from winding_field.tifxyz_project import precheck_projection,project_tile
    from winding_field.tifxyz_contract import Blocked
    guide,grid=planar_precheck;f=field(np.ones((3,10,10)))
    with pytest.raises(Blocked,match='PROJECTION_ABSENT') as error:precheck_projection(f,guide,grid)
    evidence=json.loads(error.value.detail)
    assert evidence['evaluated_nodes']<=64 and evidence['samples'][0]['status']=='ABSENT'
    with pytest.raises(Blocked,match='PROJECTION_ABSENT'):project_tile(f,guide,grid,(0,0),None)


def test_boundary_precheck_preserves_resource_and_stop_failures(planar_precheck):
    from winding_field.tifxyz_project import precheck_projection
    from winding_field.tifxyz_contract import Blocked,Paused
    guide,grid=planar_precheck;f=field(np.ones((5000,10,10)));guide['stride']=4096
    with pytest.raises(Paused):precheck_projection(f,guide,grid,lambda:True)
    with pytest.raises(Blocked,match='RAY_WORK_LIMIT'):precheck_projection(f,guide,grid)
    assert not f.read_log


def test_boundary_precheck_never_tests_nonparticipating_nodes(planar_precheck,monkeypatch):
    from winding_field import tifxyz_project as code
    guide,grid=planar_precheck;original=code._uv_samples;f=field(np.ones((3,10,10)))
    def no_cells(*args):
        rows,cols,ids,weights,used=original(*args)
        return rows,cols,ids,weights,np.zeros_like(used)
    monkeypatch.setattr(code,'_uv_samples',no_cells)
    result=code.precheck_projection(f,guide,grid)
    assert result['selected_nodes']==result['evaluated_nodes']==0 and not f.read_log
