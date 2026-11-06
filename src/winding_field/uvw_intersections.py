"""Bounded numerical gradient-trajectory/triangle intersections.

All candidate faces in the projection ball are searched, including disconnected
charts. Interval gradient bounds establish transverse/monotone plane crossings;
intervals that cannot be excluded or resolved are rejected, never skipped.
ODE error control is numerical, not a formally rounded interval ODE proof.
"""
import itertools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class _Unresolved(Exception):
    pass


class _Patch:
    """One bounded supported trilinear crop, with exact polynomial extrema bounds."""
    def __init__(self,field,center):
        h=float(min(field.spacing_um));grid=(center-field.origin_um)/field.spacing_um
        lo=np.maximum(0,np.floor(grid-2*h/field.spacing_um).astype(int))
        hi=np.minimum(np.array(field.shape[::-1]),np.ceil(grid+2*h/field.spacing_um).astype(int)+1)
        self.origin=field.origin_um+lo*field.spacing_um;self.spacing=field.spacing_um
        self.shape=(hi-lo-1)[::-1]
        cells=np.indices(self.shape).reshape(3,-1).T+lo[::-1]
        values,valid=field.read_cells(cells)
        nodes=[values[(slice(None),*bits)].reshape(self.shape) for bits in itertools.product((0,1),repeat=3)]
        a,b,c,d,e,f,g,k=nodes
        self.coefficients=np.stack((a,b-a,c-a,e-a,d-c-b+a,f-e-b+a,g-e-c+a,k-g-f-d+e+c+b-a),axis=-1)
        self.support=valid.reshape(self.shape).copy()
        self.support&=np.isfinite(self.coefficients).all(-1)

    def _evaluate(self,cells,local):
        if not self.support[tuple(cells.T)].all():raise _Unresolved('unsupported trajectory cell')
        a,b,c,d,e,f,g,k=self.coefficients[tuple(cells.T)].T
        z,y,x=local.T
        value=a+b*x+c*y+d*z+e*x*y+f*x*z+g*y*z+k*x*y*z
        gradient=np.column_stack((b+e*y+f*z+k*y*z,c+e*x+g*z+k*x*z,d+f*x+g*y+k*x*y))/self.spacing
        return value,gradient

    def sample(self,point):
        grid=((np.atleast_2d(point)-self.origin)/self.spacing)[:,::-1]
        if not np.isfinite(grid).all() or (grid<0).any() or (grid>self.shape).any():raise _Unresolved('trajectory left bounded crop')
        cells=np.minimum(np.floor(grid).astype(int),self.shape-1)
        return self._evaluate(cells,grid-cells)

    def bounds(self,low,high):
        low=((low-self.origin)/self.spacing)[::-1];high=((high-self.origin)/self.spacing)[::-1]
        if (low<0).any() or (high>self.shape).any():raise _Unresolved('interval left bounded crop')
        first=np.maximum(0,np.floor(np.nextafter(low,-np.inf)).astype(int))
        last=np.minimum(self.shape-1,np.floor(high).astype(int))
        cells=np.array(list(itertools.product(*(range(a,b+1) for a,b in zip(first,last)))),int)
        low=np.maximum(low-cells,0);high=np.minimum(high-cells,1)
        local=np.concatenate([np.where(bits,high,low) for bits in itertools.product((0,1),repeat=3)])
        values,gradient=self._evaluate(np.tile(cells,(8,1)),local)
        return values.min(),values.max(),gradient.min(0),gradient.max(0)


def trajectory_hits(reference,field,base,*,stop=None,uncertainty_um=0.):
    """Return (face, barycentrics, XYZ) hits and whether the bounded search resolved.

    Search W extrema over the entire spatial projection box, not just vertex
    neighbors or a seed. Every hit within the projection ball must lie in that
    W interval. A capped/unsupported search returns unresolved.
    """
    from .uvw import _triangle_distance
    h=float(min(field.spacing_um));radius=reference.settings.projection_fraction*h
    if not np.isfinite(uncertainty_um) or uncertainty_um<0 or uncertainty_um>=radius:return [],False
    epsilon=max(1e-9*h,64*np.finfo(float).eps*max(1.,np.max(abs(base))))
    ids=np.sort(reference.candidates(base,True,radius+epsilon+uncertainty_um))
    ids=ids[_triangle_distance(base,reference.tri[ids])<=radius+epsilon+uncertainty_um]
    if not len(ids):return [],True
    if len(ids)>4096:return [],False
    triangles=reference.tri[ids];normals=reference.normal[ids]
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    hits=[];evaluations=0;intervals=0;uncertain_faces=set()
    try:
        patch=_Patch(field,base)
        extent=radius+uncertainty_um
        wmin,wmax,global_lower,global_upper=patch.bounds(base-extent,base+extent)
        global_low=np.where(normals>=0,normals*global_lower,normals*global_upper).sum(1)
        global_high=np.where(normals>=0,normals*global_upper,normals*global_lower).sum(1)
        global_monotone=(global_low>1e-12)|(global_high<-1e-12)
        w0,_=patch.sample(base);w0=float(w0[0])
        def velocity(w,displacement):
            nonlocal evaluations
            evaluations+=1
            if stop and stop():raise InterruptedError('UVW intersection search paused')
            if evaluations>reference.settings.max_steps*32:raise _Unresolved('trajectory work bound')
            _,g=patch.sample(base+displacement);square=float(g[0]@g[0])
            if not np.isfinite(square) or square<=(1e-12/h)**2:raise _Unresolved('zero gradient')
            return g[0]/square

        def interval(curve,a,b,selected,depth=0):
            nonlocal intervals
            intervals+=1
            if intervals>reference.settings.max_steps*4:raise _Unresolved('intersection work bound')
            if stop and stop():raise InterruptedError('UVW intersection search paused')
            p=curve(a);q=curve(b);dt=abs(b-a)
            # Picard displacement enclosure: |dX/dW| <= 1/min|gradient|.
            _,g=patch.sample(p);r=2*dt/np.linalg.norm(g[0])+epsilon*4
            enclosure=False
            for _ in range(6):
                if r>h*.25:break
                _,_,lower,upper=patch.bounds(p-r-uncertainty_um,p+r+uncertainty_um)
                minimum=np.where(lower>0,lower,np.where(upper<0,-upper,0))
                norm=np.linalg.norm(minimum)
                if norm<=1e-12/h:break
                required=dt/norm+epsilon*4
                if required<=r:enclosure=True;break
                r=required*1.1
            if not enclosure:
                if depth>=24:raise _Unresolved('unresolved trajectory enclosure')
                middle=(a+b)*.5;interval(curve,a,middle,selected,depth+1);interval(curve,middle,b,selected,depth+1);return
            selected=selected[_triangle_distance(p,triangles[selected])<=r+epsilon+uncertainty_um]
            if not len(selected):return
            n=normals[selected];tri=triangles[selected]
            left=np.einsum('ij,ij->i',p-tri[:,0],n);right=np.einsum('ij,ij->i',q-tri[:,0],n)
            dlow=np.where(n>=0,n*lower,n*upper).sum(1)
            dhigh=np.where(n>=0,n*upper,n*lower).sum(1)
            monotone=(dlow>1e-8*np.linalg.norm(upper-lower)+1e-12)|(dhigh<-1e-8*np.linalg.norm(upper-lower)-1e-12)
            uncertain=[]
            for j,index in enumerate(selected):
                if not monotone[j]:
                    # The entire enclosing ball lies on one side of this plane.
                    if abs(left[j])>r+epsilon+uncertainty_um:continue
                    uncertain.append(index);continue
                if min(left[j],right[j])>epsilon or max(left[j],right[j])<-epsilon:
                    if min(abs(left[j]),abs(right[j]))<=epsilon+uncertainty_um:uncertain_faces.add(index)
                    continue
                if left[j]*right[j]<0:
                    root=brentq(lambda w:float((curve(w)-tri[j,0])@n[j]),min(a,b),max(a,b),xtol=1e-13,rtol=1e-14)
                    point=curve(root)
                else:point=p if abs(left[j])<=abs(right[j]) else q
                if np.linalg.norm(point-base)>radius+epsilon:
                    if np.linalg.norm(point-base)<=radius+epsilon+uncertainty_um:uncertain_faces.add(index)
                    continue
                edges=np.stack((tri[j,1]-tri[j,0],tri[j,2]-tri[j,0]),axis=1)
                inverse=np.linalg.pinv(edges);ab=inverse@(point-tri[j,0]);weights=np.r_[1-ab.sum(),ab]
                uncertainty=epsilon*max(1.,np.linalg.norm(inverse,axis=1).max())*4
                if weights.min() < -uncertainty:
                    # A near miss inside the incoming transport-error envelope is unresolved.
                    slope=min(abs(dlow[j]),abs(dhigh[j]))
                    projection_error=uncertainty_um*(1+np.linalg.norm(np.maximum(abs(lower),abs(upper)))/max(slope,1e-30))
                    if _triangle_distance(point,tri[j:j+1])[0]<=projection_error+epsilon:uncertain_faces.add(index)
                    continue
                # An edge hit is retained for every incident chart; UV decides equivalence.
                weights=np.maximum(weights,0);weights/=weights.sum()
                hits.append((int(ids[index]),weights,point.copy()))
            if uncertain:
                if depth>=24 or r<=epsilon*8:raise _Unresolved('tangent or unresolved plane interval')
                middle=(a+b)*.5;selected=np.array(uncertain,int)
                interval(curve,a,middle,selected,depth+1);interval(curve,middle,b,selected,depth+1)

        for target in (wmin,wmax):
            if abs(target-w0)<=1e-14:continue
            # Relative error scales with local displacement, not a large scan-frame origin.
            solution=solve_ivp(velocity,(w0,target),np.zeros(3),method='DOP853',rtol=1e-11,atol=1e-11*h,
                max_step=max(abs(target-w0)/4,1e-14),dense_output=True)
            if not solution.success:raise _Unresolved('trajectory integration failed')
            curve=lambda w:base+solution.sol(w)
            for a,b in zip(solution.t[:-1],solution.t[1:]):interval(curve,a,b,np.arange(len(ids)))
    except _Unresolved:
        return [],False
    finally:
        field.validate()
    hit_faces={face for face,_,_ in hits}
    # A globally monotone face has at most one plane root: an endpoint uncertainty
    # is discharged only when that root was actually found elsewhere in the search.
    if any(not global_monotone[i] or int(ids[i]) not in hit_faces for i in uncertain_faces):return [],False
    return hits,True
