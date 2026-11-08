import numpy as np
import pytest

from test_uvw import field_array
from winding_field.uvw import Reference,reference_points,uvw_to_xyz,xyz_to_uvw


def curved_patch():
    # W=x*z is exactly trilinear; a dense connected mesh approximates W=6.
    x,y=np.meshgrid(np.linspace(2.8,3.6,9),np.linspace(3.5,4.5,9))
    world=np.column_stack((x.ravel(),y.ravel(),6/x.ravel()))
    faces=[]
    for j in range(8):
        for i in range(8):
            k=j*9+i;faces.extend(((k,k+1,k+10),(k,k+10,k+9)))
    return Reference(world,np.array(faces),world[:,:2],6.)


def test_dense_curved_patch_and_shared_edges_are_admitted(tmp_path):
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        ref=curved_patch()
        # Interior, shared diagonal, shared vertex, and points on either side.
        uv=np.array([[3.03,4.04],[3.05,4.0625],[3.,4.],[3.05,4.06249],[3.05,4.06251]])
        xyz,valid=uvw_to_xyz(ref,field,np.column_stack((uv,np.full(len(uv),6.2))))
        assert valid.all()
        back,valid=xyz_to_uvw(ref,field,xyz)
        assert valid.all();np.testing.assert_allclose(back[:,:2],uv,atol=1e-5)


def test_reference_provenance_reconstructs_raw_mesh_point(tmp_path):
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        ref=curved_patch();uv=np.array([[3.03,4.04],[3.,4.]])
        base,valid,provenance=reference_points(ref,field,uv,with_provenance=True)
        assert valid.all()
        faces=provenance['face'];weights=provenance['barycentric']
        np.testing.assert_allclose(np.einsum('ni,nij->nj',weights,ref.uvtri[faces]),uv,atol=1e-12)
        assert np.isfinite(base).all()
        vertex=ref.faces[faces[0],0]
        neighbors=ref.vertex_faces[ref.vertex_face_offsets[vertex]:ref.vertex_face_offsets[vertex+1]]
        assert len(neighbors)>1 and faces[0] in neighbors


@pytest.mark.parametrize('offset',[0.,.002])
def test_tangent_or_two_roots_cannot_be_hidden_by_known_branch(tmp_path,offset):
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        ref=curved_patch();center=np.array([3.,4.,2.]);normal=np.array([3.,0.,-2.])
        center+=offset*normal/(normal@normal)
        along=np.array([2.,0.,3.]);along*=.3/np.linalg.norm(along)
        side=np.array([0.,.5,0.])
        extra=np.array([center-along-side,center+along-side,center+side])
        ref=Reference(np.concatenate((ref.world,extra)),np.concatenate((ref.faces,[[len(ref.world)+i for i in range(3)]])),
            np.concatenate((ref.uv,[[20.,20.],[21.,20.],[20.,21.]])),6.)
        _,valid=xyz_to_uvw(ref,field,[[3.,4.,2.]])
        assert not valid[0]


def test_curved_uv_cut_keeps_distinct_edge_hits_ambiguous(tmp_path):
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        ref=curved_patch();ref=Reference(np.concatenate((ref.world,ref.world)),
            np.concatenate((ref.faces,ref.faces+len(ref.world))),np.concatenate((ref.uv,ref.uv+20)),6.)
        _,valid=uvw_to_xyz(ref,field,[[3.,4.,6.]])
        assert not valid[0]


def test_curved_search_support_gap_remains_invalid(tmp_path):
    mask=np.ones((12,12,12),bool);mask[2,4,3]=False
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2],mask=mask) as field:
        _,valid=uvw_to_xyz(curved_patch(),field,[[3.03,4.04,6.2]])
        assert not valid[0]


@pytest.mark.parametrize('target',[5.3,6.7])
def test_analytic_curved_duplicate_is_found_in_each_direction(tmp_path,target):
    # x²-z²=5 is invariant along grad(x*z); roots are derived analytically.
    x=np.sqrt((5+np.sqrt(25+4*target*target))/2);other=np.array([x,4.,target/x])
    triangles=[]
    for center in (np.array([3.,4.,2.]),other):
        tangent=np.array([center[0],0.,-center[2]]);tangent*=1e-4/np.linalg.norm(tangent)
        side=np.array([0.,1e-4,0.]);triangles.append([center+side,center+tangent,center-side-tangent])
    ref=Reference(np.reshape(triangles,(-1,3)),np.array([[0,1,2],[3,4,5]]),
        np.array([[0.,0.],[1.,0.],[0.,1.],[10.,10.],[11.,10.],[10.,11.]]),6.)
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        _,valid,details=xyz_to_uvw(ref,field,[[3.,4.,2.]],with_details=True)
        assert not valid[0] and details['cause'][0]=='reference_ambiguous'


def test_two_roots_of_one_triangle_are_enumerated(tmp_path):
    from winding_field.uvw_intersections import trajectory_hits
    normal=np.array([3.,0.,-2.]);center=np.array([3.,4.,2.])+.002*normal/(normal@normal)
    along=np.array([2.,0.,3.]);along*=.3/np.linalg.norm(along);side=np.array([0.,.5,0.])
    ref=Reference(np.array([center-along-side,center+along-side,center+side]),np.array([[0,1,2]]),
        np.array([[0.,0.],[1.,0.],[0.,1.]]),6.)
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        hits,resolved=trajectory_hits(ref,field,np.array([3.,4.,2.]))
        assert resolved
        z=np.array([point[2] for _,_,point in hits])
        assert z.min()<2. and z.max()>2.
        _,valid,details=xyz_to_uvw(ref,field,[[3.,4.,2.]],with_details=True)
        assert not valid[0] and details['cause'][0]=='reference_ambiguous'


def test_curved_patch_across_field_read_seam(tmp_path):
    ref=curved_patch();world=ref.world.copy();world[:,0]+=60;world[:,2]=126/world[:,0]
    ref=Reference(world,ref.faces,ref.uv,126.)
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2],shape=(8,8,70)) as field:
        xyz,valid=uvw_to_xyz(ref,field,[[3.03,4.04,126.2]])
        assert valid.all()
        back,valid=xyz_to_uvw(ref,field,xyz)
        assert valid.all();np.testing.assert_allclose(back[0,:2],[3.03,4.04],atol=1e-5)


def test_search_budget_exhaustion_is_explicitly_unresolved(tmp_path):
    from winding_field.uvw import TransportSettings
    ref=curved_patch();ref.settings=TransportSettings(max_steps=1)
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        _,valid,details=xyz_to_uvw(ref,field,[[3.,4.,2.]],with_details=True)
        assert not valid[0] and details['cause'][0]=='reference_unresolved'


def test_curved_workers_preserve_geometry_and_provenance(tmp_path):
    import json
    from pathlib import Path
    import sqlite3
    from test_uvw_pipeline import inputs
    from winding_field.util import atomic_json,file_sha
    from winding_field.uvw_workers import GeometryWorkers
    args=inputs(tmp_path);ref=curved_patch()
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:descriptor=field.source.identity
    level=Path(args.reference_sheet)
    np.savez_compressed(level/'map.npz',world=ref.world,uv=ref.uv,faces=ref.faces,
        display=np.ones(len(ref.faces),bool),scientific=np.ones(len(ref.faces),bool),
        source_vertex_ids=np.arange(len(ref.world)),source_face_ids=np.arange(len(ref.faces)))
    record=json.loads((level/'LEVEL.json').read_text());record['identity']['level']=6.
    record['identity']['settings']['output_frame']=dict(id='test',origin_xyz_um=[0.,0.,0.],voxel_um=1.)
    record['files']['map.npz']=file_sha(level/'map.npz');atomic_json(level/'LEVEL.json',record)
    sha=file_sha(level/'LEVEL.json')
    with sqlite3.connect(level.parent/'publications.sqlite') as db:db.execute('UPDATE levels SET receipt_sha=?',(sha,))
    atomic_json(level.parent/'ORIGINAL.json',dict(level_receipt_sha256={level.name:sha}))
    uv=np.array([[3.03,4.04],[3.,4.],[3.05,4.0625]])
    jobs=[dict(uv=uv,w=w,reference_data=None) for w in (5.8,6.,6.2)]
    with GeometryWorkers(descriptor,level,[5.8,6.,6.2],7,workers=1,region_cache_bytes=1<<20) as pool:
        serial=list(pool.map(jobs))
    with GeometryWorkers(descriptor,level,[5.8,6.,6.2],7,workers=2,region_cache_bytes=1<<20) as pool:
        parallel=list(pool.map(jobs))
    for (a,_),(b,_) in zip(serial,parallel):
        assert a['geometry'].all() and (a['reference_face']>=0).all()
        np.testing.assert_allclose(a['reference_barycentric'].sum(1),1.)
        for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_curved_inverse_uses_physical_units_at_large_origin(tmp_path):
    origin=np.array([1e6,-2e6,3e6]);spacing=np.array([2.,3.,4.]);ref=curved_patch()
    ref=Reference(ref.world*spacing+origin,ref.faces,ref.uv*spacing[:2],6.)
    with field_array(tmp_path,lambda p:((p-origin)/spacing)[...,0]*((p-origin)/spacing)[...,2],
        origin=origin,spacing=spacing) as field:
        uv=np.array([[3.03*2,4.04*3]])
        xyz,valid=uvw_to_xyz(ref,field,np.column_stack((uv,[6.2])))
        assert valid.all()
        back,valid=xyz_to_uvw(ref,field,xyz)
        assert valid.all();np.testing.assert_allclose(back[:,:2],uv,atol=1e-5)


def test_curved_worker_preserves_geometry_at_large_origin(tmp_path):
    from winding_field.uvw_workers import compute_geometry
    origin=np.array([1e6,-2e6,3e6]);spacing=np.array([2.,3.,4.]);ref=curved_patch()
    ref=Reference(ref.world*spacing+origin,ref.faces,ref.uv*spacing[:2],6.,band=(5.8,6.2))
    with field_array(tmp_path,lambda p:((p-origin)/spacing)[...,0]*((p-origin)/spacing)[...,2],
        origin=origin,spacing=spacing) as field:
        data,_=compute_geometry(ref,field,dict(origin_xyz_um=origin,voxel_um=1.),[[6.06,12.12]],6.2)
        assert data['geometry'].all(),data['cause']
        assert (data['reference_face']>=0).all()


def test_source_change_during_curved_search_is_detected(tmp_path,monkeypatch):
    from winding_field import uvw_intersections
    integrate=uvw_intersections.solve_ivp
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        def changed(*args,**kwargs):
            result=integrate(*args,**kwargs)
            path=tmp_path/'field.npy';raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            return result
        monkeypatch.setattr(uvw_intersections,'solve_ivp',changed)
        with pytest.raises(ValueError):xyz_to_uvw(curved_patch(),field,[[3.,4.,2.]])


def skinny_competing_patch():
    ref=curved_patch();w=6.7;x=np.sqrt((5+np.sqrt(25+4*w*w))/2);center=np.array([x,4.,w/x])
    tangent=np.array([x,0.,-w/x]);tangent*=1e-5/np.linalg.norm(tangent);side=np.array([0.,1e-5,0.])
    return Reference(np.concatenate((ref.world,[center+side,center+tangent,center-side-tangent])),
        np.concatenate((ref.faces,[[len(ref.world),len(ref.world)+1,len(ref.world)+2]])),
        np.concatenate((ref.uv,[[20.,20.],[21.,20.],[20.,21.]])),6.)


def test_transport_error_cannot_hide_skinny_competing_branch(tmp_path):
    from winding_field.uvw import _forward
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        ref=skinny_competing_patch();uvw=np.array([[3.,4.,7.],[20+1/3,20+1/3,7.]])
        trial,transported=_forward(ref,field,uvw)
        assert transported.all() and np.linalg.norm(trial[0]-trial[1])<3e-5
        _,valid=xyz_to_uvw(ref,field,trial)
        assert not valid.any()
        _,valid=uvw_to_xyz(ref,field,uvw)
        assert not valid.any()


@pytest.mark.parametrize('origin,valid',[(0.,True),(1e16,False)])
def test_encoded_coordinate_must_recover_requested_uv(tmp_path,origin,valid):
    from winding_field.uvw_workers import compute_geometry,CAUSES
    world=np.array([[3.00000003,2.,5.],[9.,2.,5.],[9.,8.,5.],[3.00000003,8.,5.]])
    ref=Reference(world,np.array([[0,1,2],[0,2,3]]),world[:,:2],5.)
    with field_array(tmp_path,lambda p:p[...,2]) as field:
        data,_=compute_geometry(ref,field,dict(origin_xyz_um=[origin,0.,0.],voxel_um=1.),np.array([[3.00000003,4.]]),6.)
        assert data['geometry'][0]==valid
        assert data['cause'][0]==(CAUSES['valid'] if valid else CAUSES['coordinate_precision'])
        if valid:np.testing.assert_array_equal(data['xyz'],[[3.00000003,4.,6.]])


@pytest.mark.parametrize('w',[5.8,6.2])
def test_encoded_band_endpoints_preserve_curved_correspondence(tmp_path,w):
    from winding_field.uvw_workers import compute_geometry
    ref=curved_patch();ref.band=(5.8,6.2)
    uv=np.array([[3.03,4.04],[3.,4.],[3.05,4.0625]])
    with field_array(tmp_path,lambda p:p[...,0]*p[...,2]) as field:
        data,_=compute_geometry(ref,field,dict(origin_xyz_um=[0.,0.,0.],voxel_um=1.),uv,w)
        assert data['geometry'].all(),data['cause']
        assert (data['reference_face']>=0).all()
        outside,_=compute_geometry(ref,field,dict(origin_xyz_um=[0.,0.,0.],voxel_um=1.),uv,w+(-.01 if w==5.8 else .01))
        assert not outside['geometry'].any()
        # The public inverse still rejects arbitrary XYZ outside the requested band.
        _,valid=xyz_to_uvw(ref,field,[[3.,4.,5.79/3]])
        assert not valid.any()


def test_publish_exact_encoding_without_second_affine_rounding(tmp_path):
    from winding_field.uvw_workers import compute_geometry
    x=2286189799.342009;origin=np.array([-1996792493.6633632,0.,0.]);unit=.12345
    world=np.array([[x-3,2.,5.],[x+3,2.,5.],[x+3,8.,5.],[x-3,8.,5.]])
    world=((world-origin)/unit)*unit+origin
    ref=Reference(world,np.array([[0,1,2],[0,2,3]]),np.array([[0.,2.],[6.,2.],[6.,8.],[0.,8.]]),5.)
    uv=np.array([[3.,4.]])
    with field_array(tmp_path,lambda p:p[...,2],origin=(x-5,0.,0.)) as field:
        data,_=compute_geometry(ref,field,dict(origin_xyz_um=origin.tolist(),voxel_um=unit),uv,6.)
        assert data['geometry'].all()
        twice=((data['xyz']-origin)/unit)*unit+origin
        recovered,_=xyz_to_uvw(ref,field,twice)
        assert np.linalg.norm(recovered[:,:2]-uv)>1e-6
        stored=data['encoded_xyz']*unit+origin
        np.testing.assert_array_equal(stored,data['xyz'])
        recovered,valid=xyz_to_uvw(ref,field,stored)
        assert valid.all() and np.linalg.norm(recovered[:,:2]-uv)<=1e-6
