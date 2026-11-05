"""Global free-node normal-equation residual, streamed over owned grid cores."""

import itertools
import math

import numpy as np
import torch

from block_solver import _observations
from field_solver import _FieldOperator


def verify(shape,chunk,spacing,read,observations,*,fixed=None,constraints=None,rtol=1e-6,stop=None,gauge_offset=None):
    """Verify the stored field, including full exterior constraint groups.

    ``constraints(lo,hi,read_nodes)`` must restrict the SAME global rows used by
    the solver, including exterior targets. Norms count each free node once.
    The denominator is ||b_free-A_free,fixed*x_fixed||, not local solver RHSs.
    gauge_offset is the hash-verified origin chunk's float64 codec offset;
    only its exact float32-delta encoding of zero is accepted as the gauge.
    """
    shape=tuple(shape);spacing=np.asarray(spacing,float)
    if (len(shape)!=3 or any(type(n) is not int or n<2 for n in shape)
            or type(chunk) is not int or chunk<1 or spacing.shape!=(3,)
            or not np.isfinite(spacing).all() or np.any(spacing<=0)
            or not math.isfinite(rtol) or rtol<=0):
        raise ValueError('invalid global residual geometry or tolerance')
    encoded_gauge=0.
    if gauge_offset is not None:
        if fixed is not None or not math.isfinite(gauge_offset) or abs(gauge_offset)>np.finfo('f4').max:
            raise ValueError('invalid quantized default gauge offset')
        encoded_gauge=float(np.float32(-gauge_offset))+gauge_offset

    def anchors(lo,hi):
        if fixed is not None:
            value=np.asarray(fixed(lo,hi),float)
            if value.shape!=tuple(hi-lo) or np.isinf(value).any():raise ValueError('invalid fixed field')
            return value
        value=np.full(tuple(hi-lo),np.nan)
        if np.all(lo==0):value[0,0,0]=0
        return value

    def read_nodes(nodes,anchor=False):
        nodes=np.asarray(nodes)
        if nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or np.any(nodes<0) or np.any(nodes>=math.prod(shape)):
            raise ValueError('constraint node outside global field')
        ij=np.column_stack(np.unravel_index(nodes,shape));result=np.empty(len(nodes))
        keys,inverse=np.unique(ij//chunk,axis=0,return_inverse=True)
        for i,key in enumerate(keys):
            lo=key*chunk;hi=np.minimum(lo+chunk,shape);selected=np.flatnonzero(inverse==i)
            values=np.nan_to_num(anchors(lo,hi)) if anchor else np.asarray(read(lo,hi))
            if values.shape!=tuple(hi-lo) or not np.isfinite(values).all():raise ValueError('finite field required')
            result[selected]=values[tuple((ij[selected]-lo).T)]
        return result

    residual_norm=rhs_norm=fixed_error=fixed_storage_error=0.;free_nodes=observed_nodes=0
    for key in itertools.product(*(range((n+chunk-1)//chunk) for n in shape)):
        if stop and stop():return dict(status='PAUSED',global_converged=False,physical_acceptance=False)
        core=np.asarray(key)*chunk;end=np.minimum(core+chunk,shape)
        lo=np.maximum(core-1,0);hi=np.minimum(end+1,shape)
        inner=tuple(slice(a,b) for a,b in zip(core-lo,end-lo))
        phi=np.asarray(read(lo,hi),dtype=np.float64)
        if phi.shape!=tuple(hi-lo) or not np.isfinite(phi).all():raise ValueError('finite field required')
        target,weights,observed,_=_observations(observations(lo,hi),tuple(hi-lo))
        op=_FieldOperator(torch.tensor(weights,dtype=torch.float64),None,0)
        rhs=op.rhs(torch.tensor(target,dtype=torch.float64),spacing).numpy()
        pinned=anchors(lo,hi);anchor=np.nan_to_num(pinned)
        vectors=[]
        for value,is_anchor in ((phi,False),(anchor,True)):
            residual=op.matvec(torch.tensor(value)).numpy()-rhs
            if constraints is not None:
                rows=constraints(lo,hi,lambda nodes:read_nodes(nodes,is_anchor))
                if rows is not None:
                    residual+=(rows.matrix.T@(rows.weights*rows.residual(value))).reshape(value.shape)
            vectors.append(residual[inner])
        free=np.isnan(pinned[inner]);count=int(free.sum());free_nodes+=count
        observed_nodes+=int(np.asarray(observed)[inner].sum())
        residual_norm=math.hypot(residual_norm,float(np.linalg.norm(vectors[0][free])))
        rhs_norm=math.hypot(rhs_norm,float(np.linalg.norm(vectors[1][free])))
        fixed_error=max(fixed_error,float(np.max(abs(phi[inner][~free]-pinned[inner][~free]),initial=0)))
        stored_pinned=pinned[inner].copy()
        if gauge_offset is not None and np.all(core==0):stored_pinned[0,0,0]=encoded_gauge
        fixed_storage_error=max(fixed_storage_error,float(np.max(abs(phi[inner][~free]-stored_pinned[~free]),initial=0)))
    relative=residual_norm/rhs_norm if rhs_norm else (0. if residual_norm==0 else None)
    passed=fixed_storage_error==0 and (residual_norm<=rtol*rhs_norm if rhs_norm else residual_norm==0)
    return dict(status='VERIFIED' if passed else 'NONCONVERGED',global_converged=passed,
                physical_acceptance=False,residual_l2=residual_norm,rhs_l2=rhs_norm,
                relative_residual=relative,rtol=rtol,fixed_max_error=fixed_error,
                fixed_storage_max_error=fixed_storage_error,gauge_codec_offset=gauge_offset,
                free_nodes=free_nodes,observed_nodes=observed_nodes,total_nodes=math.prod(shape),
                norm='global free-node normal equations; fixed-node RHS elimination')
