import numpy as np
import pytest
import torch


def test_quantized_gauge_requires_exact_codec_value_and_global_residual():
    from winding_field.global_residual import verify
    shape=(5,5,9)
    original=np.broadcast_to(np.arange(9.)*.1,shape).copy()
    offset=float(original.mean())
    phi=(original-offset).astype('f4').astype('f8')+offset
    def read(lo,hi):return phi[tuple(slice(a,b) for a,b in zip(lo,hi))]
    def obs(lo,hi):
        target=np.zeros((*tuple(hi-lo),3),'f4');target[...,0]=.1
        return target,np.ones(tuple(hi-lo),'f4'),np.ones(tuple(hi-lo),bool),{}
    assert not verify(shape,3,[1.]*3,read,obs)['global_converged']
    result=verify(shape,3,[1.]*3,read,obs,gauge_offset=offset)
    assert result['global_converged'] and result['fixed_max_error']>0
    assert result['fixed_storage_max_error']==0
    phi[0,0,0]+=1e-10
    assert not verify(shape,3,[1.]*3,read,obs,gauge_offset=offset)['global_converged']
    phi[0,0,0]-=1e-10;phi[:,:,3]+=.01
    assert not verify(shape,3,[1.]*3,read,obs,gauge_offset=offset)['global_converged']


def test_streamed_residual_matches_full_operator_with_cross_block_group_and_fixed_values():
    from winding_field.global_residual import verify
    from field_solver import _FieldOperator
    from constraint_templates import StencilConstraints
    shape=(5,6,7);rng=np.random.default_rng(42)
    phi=rng.normal(size=shape);target=rng.normal(size=(*shape,3)).astype('f4')
    weights=rng.uniform(.1,2,shape).astype('f4');fixed=np.full(shape,np.nan)
    fixed[0,0,0]=phi[0,0,0];fixed[-1,-1,-1]=phi[-1,-1,-1]
    rows=StencilConstraints([[0,209],[14,140],[55,72]],[[1.,-1.],[1.,-1.],[1.,-1.]],shape,
                           weights=[2.,3.,4.],targets=[.1,.2,.3],groups=[0,0,-1])
    def read(array):return lambda lo,hi:array[tuple(slice(a,b) for a,b in zip(lo,hi))]
    op=_FieldOperator(torch.tensor(weights,dtype=torch.float64),None,0)
    rhs=op.rhs(torch.tensor(target,dtype=torch.float64),[2.,3.,4.]).numpy()
    q=rows.restrict([0,0,0],shape,lambda nodes:phi.ravel()[nodes])
    gradient=op.matvec(torch.tensor(phi)).numpy()-rhs
    gradient+=(q.matrix.T@(q.weights*q.residual(phi))).reshape(shape)
    anchor=np.nan_to_num(fixed)
    base=op.matvec(torch.tensor(anchor)).numpy()-rhs
    base+=(q.matrix.T@(q.weights*q.residual(anchor))).reshape(shape)
    free=np.isnan(fixed)
    expected=np.linalg.norm(gradient[free])/np.linalg.norm(base[free])
    result=verify(shape,2,[2.,3.,4.],read(phi),
        lambda lo,hi:(read(target)(lo,hi),read(weights)(lo,hi),np.ones(tuple(hi-lo),bool),{}),
        fixed=read(fixed),constraints=rows.restrict,rtol=1e-6)
    assert result['relative_residual']==pytest.approx(expected,rel=1e-13)
    assert result['global_converged'] is False
    assert result['free_nodes']==int(free.sum())


def test_exact_linear_field_passes_but_one_seam_error_fails():
    from winding_field.global_residual import verify
    shape=(5,5,9);phi=np.broadcast_to(np.arange(9.),shape).copy()
    def read(lo,hi):return phi[tuple(slice(a,b) for a,b in zip(lo,hi))]
    def obs(lo,hi):
        target=np.zeros((*tuple(hi-lo),3),'f4');target[...,0]=1
        return target,np.ones(tuple(hi-lo),'f4'),np.ones(tuple(hi-lo),bool),{}
    assert verify(shape,3,[1.,1.,1.],read,obs)['global_converged'] is True
    phi[:,:,3]+=.1
    assert verify(shape,3,[1.,1.,1.],read,obs)['global_converged'] is False
    phi[1,1,1]=np.nan
    with pytest.raises(ValueError,match='finite'):verify(shape,3,[1.,1.,1.],read,obs)
