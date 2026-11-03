"""Bounded global recipe: source identity, solver rejection and partial accounting."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import eye

from winding_field import native_global as global_uv
from winding_field.flatten import overlap_pairs,iter_overlap_pairs


def disk():
    v=np.array([[0.,0.,0.],[4.,0.,0.],[4.,3.,0.],[0.,3.,0.],[.07,1.15,0.]])
    f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],np.int32)
    return dict(vertices=v,faces=f,parent=np.arange(5),boundary=np.arange(4))


def test_overlap_iterator_preserves_contacts_slivers_and_count():
    uv=np.array([[0.,0.],[1000.,0.],[0.,.001],[500.,-.001],[501.,.002],[499.,.002],[0.,0.],[0.,.001],[-1.,0.]])
    f=np.arange(9).reshape(3,3)
    assert list(iter_overlap_pairs(uv,f))==[(0,1)]
    assert overlap_pairs(uv,f)==1
    assert overlap_pairs(uv,np.empty((0,3),int))==0


def test_overlap_candidate_admission_precedes_neighbor_materialization(monkeypatch):
    import winding_field.flatten as flatten
    real=flatten.cKDTree; requests=[]
    class ObservedTree:
        def __init__(self,*args):self.tree=real(*args)
        def query_ball_point(self,*args,**kwargs):
            requests.append(kwargs.get('return_length',False))
            return self.tree.query_ball_point(*args,**kwargs)
    monkeypatch.setattr(flatten,'cKDTree',ObservedTree)
    uv=np.array([[0.,0.],[1,0],[0,1]])
    with pytest.raises(RuntimeError,match='candidate'):
        list(iter_overlap_pairs(uv,np.tile([0,1,2],(5,1)),max_candidates=4))
    assert requests==[True]


def test_overlap_budget_and_cancellation_never_report_complete():
    uv=np.array([[0.,0.],[1,0],[0,1]])
    with pytest.raises(RuntimeError,match='narrow'):
        list(iter_overlap_pairs(uv,np.tile([0,1,2],(3,1)),max_tests=0))
    def stop():raise InterruptedError('stop')
    with pytest.raises(InterruptedError):list(iter_overlap_pairs(uv,np.array([[0,1,2]]),check=stop))


def test_overlap_adaptive_batches_and_unchanged_area_threshold():
    uv=np.array([[0.,0.],[1,0],[0,1]])
    f=np.tile([0,1,2],(12,1));report={}
    assert len(list(iter_overlap_pairs(uv,f,max_candidates=24,report=report)))==overlap_pairs(uv,f)==66
    assert report['complete'] and report['pairs']==66
    for shift,expected in ((1-1e-3,1),(1-1e-4,0),(1.,0)):
        points=np.concatenate((uv,uv+[shift,0]))
        assert overlap_pairs(points,np.arange(6).reshape(2,3))==expected


def test_qualify_removes_both_overlap_participants_and_keeps_point_contacts_separate():
    v=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[2,0,0],[2,1,0],[3,0,0],[3,1,0]])
    f=np.array([[0,1,2],[1,3,4],[3,5,6]],np.int32)
    cut=dict(vertices=v,faces=f)
    uv=v[:,:2].copy()
    arrays,report=global_uv.qualify_map(cut,uv)
    assert arrays['valid_faces'].tolist()==[True,True,True]
    assert report['components']==3 and report['largest_component_area_fraction']==pytest.approx(1/3)
    assert report['source_area_um2']==report['qualified_area_um2']==1.5
    uv[5:]=uv[[1,2]]
    uv[3]=uv[0]
    arrays,report=global_uv.qualify_map(cut,uv)
    assert not arrays['valid_faces'][0] and not arrays['valid_faces'][2]
    assert arrays['reasons'][0]&8 and arrays['reasons'][2]&8
    assert report['overlap_audit']['complete'] and report['overlap_audit']['pairs']>=1
    json.dumps(report,allow_nan=False)


def test_qualify_exact_native_thresholds_and_all_invalid_report():
    cut=disk();cut['vertices']=cut['vertices'][:4];cut['faces']=np.array([[0,1,2],[0,2,3]],np.int32)
    for scale,valid in ((.25,True),(4.,True),(.249,False),(4.001,False),(-1.,True)):
        arrays,report=global_uv.qualify_map(cut,cut['vertices'][:,:2]*scale)
        assert arrays['valid_faces'].all()==valid
        assert report['qualified_faces']==(2 if valid else 0)
        json.dumps(report,allow_nan=False)
    uv=cut['vertices'][:,:2].copy();uv[:,0]*=-1
    arrays,report=global_uv.qualify_map(cut,uv)
    assert (arrays['reasons']&2).all() and report['largest_component_area_fraction']==0.
    uv[0]=np.nan
    arrays,report=global_uv.qualify_map(cut,uv)
    assert (arrays['reasons']&1).all();json.dumps(report,allow_nan=False)


def test_qualification_budget_failure_is_not_an_accepted_subset(monkeypatch):
    cut=disk();monkeypatch.setattr(global_uv,'MAX_NARROW_TESTS',0)
    with pytest.raises(RuntimeError,match='narrow'):global_uv.qualify_map(cut,cut['vertices'][:,:2])


def test_input_limits_reject_before_allocations_or_optional_library(monkeypatch):
    cut=disk();monkeypatch.setattr(global_uv,'MAX_FACES',3)
    with pytest.raises(ValueError,match='bound'):global_uv.cut_component(cut['vertices'],cut['faces'])
    monkeypatch.setattr(global_uv,'MAX_FACES',1250000)
    for faces in (np.array([[0,1,99]]),np.array([[0,1,-1]]),np.array([[0.,1.,2.]]),np.array([[0,0,1]])):
        with pytest.raises(ValueError):global_uv.cut_component(cut['vertices'],faces)
    v=cut['vertices'].copy();v[0]=np.inf
    with pytest.raises(ValueError):global_uv.cut_component(v,cut['faces'])


def test_mvc_linear_precision_including_obtuse_angles():
    cut=disk();expected=np.array([[10.,-7.],[14.8,-5.8],[14.2,-3.1],[9.4,-4.3],[9.854,-5.944]])
    w=global_uv._weights(cut['vertices'],cut['faces'],None)
    uv,report=global_uv._dirichlet(w,cut['boundary'],expected[:4],None)
    np.testing.assert_allclose(uv,expected,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(uv[:4],expected[:4])
    assert report['true_residual_pass'] and max(report['relative_residuals'])<=1e-10


@pytest.mark.parametrize('info,answer',[(1,np.array([2.])),(0,np.array([0.]))])
def test_solver_rejects_info_or_false_convergence(monkeypatch,info,answer):
    import scipy.sparse.linalg as sla
    monkeypatch.setattr(sla,'gmres',lambda *a,**k:(answer,info))
    with pytest.raises(ValueError,match='converge|residual'):
        global_uv._solve(eye(1,format='csc'),np.array([[2.,2.]]),None)


def test_solver_inner_iteration_budget_and_callback(monkeypatch):
    import scipy.sparse.linalg as sla
    def never_converges(*args,**kwargs):
        for _ in range(301):kwargs['callback'](1.)
        return np.ones(1),0
    monkeypatch.setattr(sla,'gmres',never_converges)
    with pytest.raises(RuntimeError,match='iteration budget'):
        global_uv._solve(eye(1,format='csc'),np.ones((1,2)),None)
    calls=[]
    def stop():
        calls.append(None)
        if len(calls)==5:raise InterruptedError('inner stop')
    with pytest.raises(InterruptedError,match='inner stop'):
        global_uv._solve(eye(1,format='csc'),np.ones((1,2)),stop)


def test_callbacks_raise_without_arguments_before_numerical_work():
    cut=disk()
    def stop():raise InterruptedError('stop')
    for function,args in ((global_uv.cut_component,(cut['vertices'],cut['faces'])),
        (global_uv.mean_value_map,(cut,)),(global_uv.arap_map,(cut,cut['vertices'][:,:2])),
        (global_uv.qualify_map,(cut,cut['vertices'][:,:2]))):
        with pytest.raises(InterruptedError):function(*args,check=stop)


def sphere():
    v=np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]],np.int32)
    return v,f


def torus():
    n,m=9,7;u,v=np.meshgrid(np.arange(n)*2*np.pi/n,np.arange(m)*2*np.pi/m,indexing='ij')
    xyz=np.stack(((3+np.cos(v))*np.cos(u),(3+np.cos(v))*np.sin(u),np.sin(v)),axis=-1).reshape(-1,3)
    faces=[]
    for i in range(n):
        for j in range(m):
            a=i*m+j;b=((i+1)%n)*m+j;c=i*m+(j+1)%m;d=((i+1)%n)*m+(j+1)%m
            faces.extend(((a,b,c),(b,d,c)))
    return xyz,np.array(faces,np.int32)


@pytest.mark.parametrize('kind',['disk','sphere','torus'])
def test_real_cut_mvc_arap_preserves_source_and_exact_pins(kind):
    pytest.importorskip('igl')
    v,f=(disk()['vertices'],disk()['faces']) if kind=='disk' else globals()[kind]()
    before_v=v.copy();before_f=f.copy();cut,topology=global_uv.cut_component(v,f)
    np.testing.assert_array_equal(cut['vertices'],v[cut['parent']])
    np.testing.assert_array_equal(cut['parent'][cut['faces']],f)
    assert cut['faces'].dtype==np.int32 and topology['cut']['euler_characteristic']==1
    uv,report=global_uv.mean_value_map(cut)
    assert report['positive_disk_validated'] and report['mvc']['true_residual_pass']
    b=cut['boundary'];first=int(b[np.lexsort((b,uv[b,1],uv[b,0]))[0]])
    distances=np.sum((uv[b]-uv[first])**2,axis=1);second=int(b[distances==distances.max()].min())
    actual,arap=global_uv.arap_map(cut,uv)
    assert arap['pin_indices']==[first,second] and arap['iterations']==20
    np.testing.assert_array_equal(actual[[first,second]],uv[[first,second]])
    np.testing.assert_array_equal(v,before_v);np.testing.assert_array_equal(f,before_f)
    for data in (topology,report,arap):json.dumps(data,allow_nan=False)


def test_reject_nonmanifold_duplicate_inconsistent_and_disconnected_input():
    pytest.importorskip('igl');v,f=sphere()
    for bad in (np.concatenate((f,f[:1])),np.concatenate((f,[[0,2,5]])),f.copy()):
        if len(bad)==len(f):bad[0]=bad[0,::-1]
        with pytest.raises(ValueError):global_uv.cut_component(v,bad)
    with pytest.raises(ValueError):global_uv.cut_component(np.concatenate((v,v+4)),np.concatenate((f,f+len(v))))
    # Two closed vertex stars touching at one vertex are edge-manifold but not vertex-manifold.
    combined=np.concatenate((v,v[1:]+[0,0,4]));other=f+len(v)-1;other[f==0]=0
    with pytest.raises(ValueError):global_uv.cut_component(combined,np.concatenate((f,other)))


def test_cut_budget_and_corrupt_boundary_rejected(monkeypatch):
    pytest.importorskip('igl');v,f=sphere();monkeypatch.setattr(global_uv,'MAX_CUT_VERTICES',len(v))
    with pytest.raises(ValueError,match='cut.*bound'):global_uv.cut_component(v,f)
    cut=disk();cut['boundary']=np.array([0,1,2,4])
    with pytest.raises(ValueError,match='boundary'):global_uv.mean_value_map(cut)


def test_existing_disk_needs_no_extra_cut_vertex_budget(monkeypatch):
    pytest.importorskip('igl');original=disk()
    monkeypatch.setattr(global_uv,'MAX_CUT_VERTICES',len(original['vertices']))
    cut,report=global_uv.cut_component(original['vertices'],original['faces'])
    np.testing.assert_array_equal(cut['parent'],np.arange(len(original['vertices'])))
    assert report['seam_duplicates']==0


def test_direct_arap_folded_disk_known_isometry():
    igl=pytest.importorskip('igl')
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.75,.75,np.sqrt(.375)]])
    f=np.array([[0,1,2],[1,3,2]],np.int32);uv=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    cut=dict(vertices=v,faces=f,parent=np.arange(4),boundary=igl.boundary_loop(f).astype(np.int64))
    actual,report=global_uv.arap_map(cut,uv)
    np.testing.assert_allclose(actual,uv,rtol=0,atol=1e-10)
    assert report['pin_indices']==[0,3]


@pytest.mark.skipif(not os.environ.get('WF_GLOBAL_CONTROL'),reason='opt-in immutable W50 control parity')
def test_saved_real_control_qualification_parity():
    root=Path(os.environ['WF_GLOBAL_CONTROL']);load=lambda name:np.load(root/name,allow_pickle=False)
    cut=dict(vertices=load('cut-vertices-xyz-um.npy'),faces=load('cut-faces.npy'))
    arrays,report=global_uv.qualify_map(cut,load('direct-arap-01/iteration-020-uv.npy'))
    np.testing.assert_array_equal(arrays['reasons'],load('partial-delivery-01/root_face_reasons.npy'))
    assert report['qualified_faces']==584001 and report['components']==149
    assert report['qualified_area_um2']==pytest.approx(57984133.31188509,abs=1e-6)
    assert report['overlap_audit']['pairs']==18636 and report['overlap_audit']['complete']


@pytest.mark.skipif(not os.environ.get('WF_GLOBAL_CONTROL'),reason='opt-in immutable W50 control parity')
def test_real_control_new_seed_validation_not_bit_identity():
    pytest.importorskip('igl');root=Path(os.environ['WF_GLOBAL_CONTROL'])
    cut={key:np.load(root/name,allow_pickle=False) for key,name in dict(vertices='cut-vertices-xyz-um.npy',faces='cut-faces.npy',
        parent='cut-parent.npy',boundary='cut-boundary.npy').items()}
    uv,report=global_uv.mean_value_map(cut)
    old=np.load(root/'mean-value-mmd-01/ilu-fresh-01/mean-value-uv.npy',allow_pickle=False)
    # Compare numerical validation and physical scale, not bytes from different solvers.
    assert report['positive_disk_validated'] and report['uniform']['true_residual_pass'] and report['mvc']['true_residual_pass']
    np.testing.assert_allclose(uv[cut['boundary']],old[cut['boundary']],rtol=1e-5,atol=1e-7)
    print(json.dumps(dict(uniform=report['uniform'],mvc=report['mvc'],maximum_uv_difference=float(np.abs(uv-old).max())),allow_nan=False))
