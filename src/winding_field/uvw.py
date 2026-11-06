"""Local, masked UVW coordinates; XYZ and UV are physical micrometres.

The ODE is dX/dW = grad(W)/|grad(W)|². Checks are local and sampled:
neither an accepted point nor a raster certifies global invertibility.
"""
from dataclasses import dataclass,replace
import itertools

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TransportSettings:
    step_fraction: float = .25
    error_fraction: float = 1e-4
    residual_fraction: float = .05
    roundtrip_fraction: float = .25
    projection_fraction: float = .25
    max_steps: int = 4096

    def __post_init__(self):
        for name in ('step_fraction','error_fraction','residual_fraction','roundtrip_fraction','projection_fraction'):
            value=getattr(self,name)
            if not np.isfinite(value) or value<=0:raise ValueError('invalid transport '+name)
        if self.step_fraction>.25:raise ValueError('spatial steps must be <= .25 minimum spacing')
        if self.residual_fraction>.05 or self.roundtrip_fraction>.25 or self.projection_fraction>.25:
            raise ValueError('transport tolerances exceed admitted physical bounds')
        if type(self.max_steps) is not int or not 1<=self.max_steps<=100000:raise ValueError('invalid max_steps')


def _points(points,columns=3):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=columns or len(points)>16384:
        raise ValueError('bounded point batch required (<=16384)')
    return points


def sample_w_gradient(field,xyz_um):
    """Trilinear value and analytic XYZ gradient (W/µm), requiring all 8 nodes."""
    xyz=_points(xyz_um);grid=((xyz-field.origin_um)/field.spacing_um)[:,::-1]
    inside=np.isfinite(grid).all(1)&((grid>=0)&(grid<=np.array(field.shape)-1)).all(1)
    value=np.full(len(xyz),np.nan);gradient=np.full((len(xyz),3),np.nan);valid=np.zeros(len(xyz),bool)
    ids=np.flatnonzero(inside)
    if len(ids):
        cells=np.minimum(np.floor(grid[ids]).astype(np.int64),np.array(field.shape)-2)
        samples,support=field.read_cells(cells);local=grid[ids]-cells
        v=np.zeros(len(ids));g=np.zeros((len(ids),3))
        for bits in itertools.product((0,1),repeat=3):
            factors=np.where(bits,local,1-local);s=samples[(slice(None),*bits)]
            v+=np.prod(factors,axis=1)*s
            for axis in range(3):
                other=[i for i in range(3) if i!=axis]
                g[:,axis]+=(1 if bits[axis] else -1)*np.prod(factors[:,other],axis=1)*s
        value[ids]=v;gradient[ids]=g[:,::-1]/field.spacing_um
        valid[ids]=support&np.isfinite(samples).all((1,2,3))&np.isfinite(g).all(1)&np.isfinite(v)
    return value,gradient,valid


def _swept_support(field,a,b):
    # A swept box is conservative: every touched cell, including diagonal crossings.
    ga=(a-field.origin_um)/field.spacing_um;gb=(b-field.origin_um)/field.spacing_um
    shape=np.array(field.shape[::-1]);lo=np.floor(np.minimum(ga,gb)).astype(np.int64)
    hi=np.floor(np.maximum(ga,gb)).astype(np.int64)
    inside=((np.minimum(ga,gb)>=0)&(np.maximum(ga,gb)<=shape-1)).all(1)
    lo=np.clip(lo,0,shape-2);hi=np.clip(hi,0,shape-2)
    result=inside.copy()
    if np.any(hi-lo>1):raise ValueError('unbounded integration step')
    corners=np.concatenate([np.where(bits,hi,lo)[:,::-1] for bits in itertools.product((0,1),repeat=3)])
    cells,inverse=np.unique(corners,axis=0,return_inverse=True);supported=np.zeros(len(cells),bool)
    for start in range(0,len(cells),16384):
        values,support=field.read_cells(cells[start:start+16384])
        supported[start:start+16384]=support&np.isfinite(values).all((1,2,3))
    result&=supported[inverse].reshape(8,-1).all(0)
    return result


def transport(field,xyz_um,target_w,settings=None,*,stop=None):
    """Adaptive midpoint step doubling. Returns XYZ, validity and per-point causes.

    Batches are bounded to 16384; stop raises InterruptedError without publishing.
    Invalid XYZ are NaN, never plausible zero coordinates.
    """
    cfg=settings or TransportSettings();x=_points(xyz_um).copy();n=len(x)
    target=np.broadcast_to(np.asarray(target_w,float),(n,))
    h=float(np.min(field.spacing_um));limit=cfg.step_fraction*h;atol=cfg.error_fraction*h
    value,g,valid=sample_w_gradient(field,x);norm=np.linalg.norm(g,axis=1)
    valid&=np.isfinite(norm)
    cause=np.full(n,'',dtype='<U24');cause[~valid]='invalid_field'
    zero=valid&(norm<=1e-12/h);cause[zero]='zero_gradient';valid&=~zero
    bad=~np.isfinite(target);cause[bad]='invalid_target';valid&=~bad
    steps=np.zeros(n,np.int32);scale=np.ones(n);residual=np.full(n,np.nan);integration_error=np.zeros(n)
    def velocity(p):
        w,grad,ok=sample_w_gradient(field,p);square=np.sum(grad*grad,axis=1)
        ok&=np.isfinite(square)&(square>(1e-12/h)**2)
        return w,grad/np.where(ok,square,1)[:,None],ok
    for iteration in range(cfg.max_steps):
        if stop and stop():raise InterruptedError('UVW transport paused')
        residual[valid]=abs(target[valid]-value[valid])/norm[valid]
        active=valid&(residual>min(atol,.001*h))
        if not active.any():break
        ids=np.flatnonzero(active);p=x[ids];delta=target[ids]-value[ids]
        dw=np.sign(delta)*np.minimum(abs(delta),limit*norm[ids]*scale[ids])
        _,v0,ok=velocity(p)
        mid=p+.5*dw[:,None]*v0
        _,vm,good=velocity(mid);ok&=good
        full=p+dw[:,None]*vm
        quarter=p+.25*dw[:,None]*v0
        _,vq,good=velocity(quarter);ok&=good
        half=p+.5*dw[:,None]*vq
        _,vh,good=velocity(half);ok&=good
        three=half+.25*dw[:,None]*vh
        _,vt,good=velocity(three);ok&=good
        fine=half+.5*dw[:,None]*vt
        wnew,gnew,good=sample_w_gradient(field,fine);ok&=good
        displacement=np.maximum.reduce([np.linalg.norm(q-p,axis=1) for q in (mid,full,quarter,half,three,fine)])
        error=np.linalg.norm(fine-full,axis=1)/3
        accepted=ok&(error<=atol)&(displacement<=limit*(1+1e-10))
        valid[ids[~ok]]=False;cause[ids[~ok]]='unsupported_path'
        # Failed trials shrink; no invalid trial is allowed to become an accepted endpoint.
        rejected=~accepted;scale[ids[rejected]]*=.5
        exhausted=scale[ids]<1e-8
        cause[ids[exhausted]]='nonconvergence';valid[ids[exhausted]]=False
        selected=np.flatnonzero(accepted)
        if len(selected):
            stages=np.stack((p,mid,quarter,half,three,fine,full))[:,selected]
            good=_swept_support(field,stages.min(0),stages.max(0))
            failed=ids[selected[~good]];valid[failed]=False;cause[failed]='unsupported_path'
            selected=selected[good];take=ids[selected]
            x[take]=fine[selected];value[take]=wnew[selected];norm[take]=np.linalg.norm(gnew[selected],axis=1)
            integration_error[take]+=4*error[selected]
            steps[take]+=1;scale[take]=np.minimum(1.,scale[take]*np.where(error[selected]<atol/8,2.,1.))
            nonfinite=~np.isfinite(norm[take]);valid[take[nonfinite]]=False;cause[take[nonfinite]]='invalid_field'
            zero=norm[take]<=1e-12/h;valid[take[zero]]=False;cause[take[zero]]='zero_gradient'
    residual[valid]=abs(target[valid]-value[valid])/norm[valid]
    failed=valid&(residual>cfg.residual_fraction*h);valid[failed]=False;cause[failed]='nonconvergence'
    x[~valid]=np.nan
    return x,valid,dict(cause=cause,normal_residual_um=residual,steps=steps,
        position_error_um=integration_error+residual)


class Reference:
    """One original UV map, in µm. Optional band bounds constrain both APIs."""
    def __init__(self,world,faces,uv,w,*,display=None,band=None,settings=None):
        self.world=np.asarray(world,float);self.faces=np.asarray(faces);self.uv=np.asarray(uv,float);self.w=float(w)
        self.settings=settings or TransportSettings();self.band=band
        if (self.world.ndim!=2 or self.world.shape[1]!=3 or self.uv.shape!=(len(self.world),2)
            or self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.faces.dtype.kind not in 'iu'
            or not len(self.faces) or not np.isfinite(self.world).all() or not np.isfinite(self.uv).all()
            or not np.isfinite(self.w) or self.faces.min()<0 or self.faces.max()>=len(self.world)):
            raise ValueError('invalid reference map')
        if band is not None and (len(band)!=2 or not np.isfinite(band).all() or not band[0]<=self.w<=band[1]):
            raise ValueError('invalid reference band')
        self.tri=self.world[self.faces];self.uvtri=self.uv[self.faces]
        a=self.uvtri[:,1]-self.uvtri[:,0];b=self.uvtri[:,2]-self.uvtri[:,0]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        normal=np.cross(self.tri[:,1]-self.tri[:,0],self.tri[:,2]-self.tri[:,0])
        self.face_valid=(abs(det)>1e-12*np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-30))&(np.linalg.norm(normal,axis=1)>1e-20)
        if display is not None:
            display=np.asarray(display)
            if display.shape!=(len(self.faces),):raise ValueError('invalid reference display mask')
            self.face_valid&=display.astype(bool)
        self.normal=normal
        # Compact vertex-to-face connectivity; no Python object per mesh edge/vertex.
        order=np.argsort(self.faces.ravel(),kind='stable')
        self.vertex_faces=order//3
        self.vertex_face_offsets=np.r_[0,np.cumsum(np.bincount(self.faces.ravel(),minlength=len(self.world)))]
        self._indices=[]
        for triangles in (self.uvtri,self.tri):
            centers=triangles.mean(1);radii=np.linalg.norm(triangles-centers[:,None],axis=2).max(1)
            self._indices.append((cKDTree(centers),radii,float(radii.max())))

    def candidates(self,point,spatial=False,tolerance=0.):
        tree,radii,radius=self._indices[int(spatial)]
        ids=np.asarray(tree.query_ball_point(point,radius+tolerance),np.int64)
        if len(ids):ids=ids[np.linalg.norm(tree.data[ids]-point,axis=1)<=radii[ids]+tolerance]
        return ids[self.face_valid[ids]]

    def locate_uv(self,uv,*,with_faces=False,with_provenance=False):
        uv=_points(uv,2);xyz=np.full((len(uv),3),np.nan);valid=np.zeros(len(uv),bool)
        owner=np.zeros(len(uv),np.int64)
        barycentric=np.full((len(uv),3),np.nan)
        for i,p in enumerate(uv):
            if not np.isfinite(p).all():continue
            ids=np.sort(self.candidates(p,tolerance=1e-9))
            if not len(ids):continue
            tri=self.uvtri[ids];matrix=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)
            ab=np.linalg.solve(matrix,(p-tri[:,0])[...,None])[...,0]
            weights=np.column_stack((1-ab.sum(1),ab));inside=(weights>=-1e-9).all(1)
            candidates=np.einsum('ni,nij->nj',weights[inside],self.tri[ids[inside]])
            if len(candidates) and np.max(np.linalg.norm(candidates-candidates[0],axis=1))<1e-7:
                xyz[i]=candidates[0];valid[i]=True;owner[i]=ids[inside][0]
                barycentric[i]=weights[inside][0]
        if with_provenance:return xyz,valid,dict(face=np.where(valid,owner,-1),barycentric=barycentric)
        return (xyz,valid,owner) if with_faces else (xyz,valid)


def correspondence_transport(field,xyz,target,settings,*,stop=None):
    # Resolve mesh-edge correspondence more tightly than the physical acceptance bounds.
    return transport(field,xyz,target,replace(settings,error_fraction=min(settings.error_fraction,1e-10)),stop=stop)


def reference_points(reference,field,uv,*,stop=None,with_provenance=False):
    """Admit small original extraction residuals, then project along the same flow."""
    raw,valid,provenance=reference.locate_uv(uv,with_provenance=True)
    owner=np.maximum(provenance['face'],0);h=float(min(field.spacing_um));cfg=reference.settings
    values,g,ok=sample_w_gradient(field,raw);norm=np.linalg.norm(g,axis=1)
    valid&=ok&(norm>1e-12/h)&(abs(values-reference.w)<=cfg.projection_fraction*h*norm)
    normals=reference.normal[owner]
    valid&=abs(np.sum(normals*g,axis=1))>1e-8*np.linalg.norm(normals,axis=1)*norm
    xyz=np.full_like(raw,np.nan)
    ids=np.flatnonzero(valid)
    if len(ids):
        moved,ok,_=correspondence_transport(field,raw[ids],reference.w,cfg,stop=stop)
        ok&=np.linalg.norm(moved-raw[ids],axis=1)<=cfg.projection_fraction*h
        xyz[ids]=moved;valid[ids]&=ok
    xyz[~valid]=np.nan
    return (xyz,valid,provenance) if with_provenance else (xyz,valid)


def _forward(reference,field,uvw,*,stop=None):
    uvw=_points(uvw);xyz,valid=reference_points(reference,field,uvw[:,:2],stop=stop)
    if reference.band is not None:valid&=(uvw[:,2]>=reference.band[0])&(uvw[:,2]<=reference.band[1])
    ids=np.flatnonzero(valid)
    if len(ids):
        moved,ok,_=correspondence_transport(field,xyz[ids],uvw[ids,2],reference.settings,stop=stop)
        back,back_ok,_=correspondence_transport(field,moved,reference.w,reference.settings,stop=stop)
        ok&=back_ok&(np.linalg.norm(back-xyz[ids],axis=1)<=reference.settings.roundtrip_fraction*min(field.spacing_um))
        xyz[ids]=moved;valid[ids]&=ok
    xyz[~valid]=np.nan
    return xyz,valid


def _triangle_distance(point,triangles):
    """Distance to closed triangles, including edges; used only for exclusion."""
    a=triangles[:,1]-triangles[:,0];b=triangles[:,2]-triangles[:,0];d=point-triangles[:,0]
    normal=np.cross(a,b);square=np.sum(normal*normal,axis=1)
    s=np.sum(np.cross(d,b)*normal,axis=1)/square
    t=np.sum(np.cross(a,d)*normal,axis=1)/square
    inside=(s>=0)&(t>=0)&(s+t<=1)
    distance=np.full(len(triangles),np.inf)
    distance[inside]=abs(np.sum(d[inside]*normal[inside],axis=1))/np.sqrt(square[inside])
    for first,last in ((0,1),(1,2),(2,0)):
        edge=triangles[:,last]-triangles[:,first];delta=point-triangles[:,first]
        fraction=np.clip(np.sum(delta*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
        distance=np.minimum(distance,np.linalg.norm(delta-fraction[:,None]*edge,axis=1))
    return distance


def _inverse(reference,field,xyz_um,*,stop=None,with_details=False):
    """Shared curved-flow branch admission, without recursive public API calls."""
    from .uvw_intersections import trajectory_hits
    xyz=_points(xyz_um);w,g,valid=sample_w_gradient(field,xyz)
    if reference.band is not None:valid&=(w>=reference.band[0])&(w<=reference.band[1])
    base,ok,back_info=correspondence_transport(field,xyz,reference.w,reference.settings,stop=stop);valid&=ok
    result=np.full_like(xyz,np.nan);h=float(min(field.spacing_um))
    found=np.zeros(len(xyz),bool);ambiguous=found.copy();unresolved=found.copy();pending=[];owners=[]
    cause=np.full(len(xyz),'invalid_field',dtype='<U24');cause[valid]='reference_domain'
    def check_candidates():
        if not pending:return
        candidates=np.concatenate(pending);indices=np.concatenate(owners)
        for start in range(0,len(candidates),16384):
            trial=candidates[start:start+16384];owner=indices[start:start+16384]
            forward,good=_forward(reference,field,trial,stop=stop)
            good&=np.linalg.norm(forward-xyz[owner],axis=1)<=reference.settings.roundtrip_fraction*h
            unresolved[owner[~good]]=True
            for candidate,index in zip(trial[good],owner[good]):
                if found[index] and np.linalg.norm(result[index,:2]-candidate[:2])>1e-6:ambiguous[index]=True
                result[index]=candidate;found[index]=True
        pending.clear();owners.clear()
    for i in np.flatnonzero(valid):
        if stop and stop():raise InterruptedError('UVW inverse paused')
        hits,resolved=trajectory_hits(reference,field,base[i],stop=stop,uncertainty_um=back_info['position_error_um'][i])
        if not resolved:valid[i]=False;cause[i]='reference_unresolved';continue
        uv=[]
        for face,weights,point in hits:
            candidate=weights@reference.uvtri[face]
            if not any(np.linalg.norm(candidate-previous)<=1e-6 for previous in uv):uv.append(candidate)
        candidates=np.column_stack((uv,np.full(len(uv),w[i])))
        if len(candidates):
            pending.append(candidates);owners.append(np.full(len(candidates),i,np.int64))
        if sum(len(p) for p in pending)>=4096:check_candidates()
    check_candidates();valid&=found&~ambiguous&~unresolved
    cause[unresolved]='reference_unresolved';cause[ambiguous]='reference_ambiguous';cause[valid]='valid'
    result[~valid]=np.nan
    if with_details:
        _,_,provenance=reference.locate_uv(result[:,:2],with_provenance=True)
        return result,valid,dict(provenance,cause=cause)
    return result,valid


def uvw_to_xyz(reference,field,uvw,*,stop=None):
    """Forward transport with the same ambiguity admission as the inverse/exporter."""
    xyz,valid=_forward(reference,field,uvw,stop=stop);ids=np.flatnonzero(valid)
    if len(ids):
        recovered,unique=_inverse(reference,field,xyz[ids],stop=stop)
        valid[ids]&=unique&(np.linalg.norm(recovered[:,:2]-np.asarray(uvw)[ids,:2],axis=1)<=1e-6)
    xyz[~valid]=np.nan
    return xyz,valid


def xyz_to_uvw(reference,field,xyz_um,*,stop=None,with_details=False):
    """Local inverse; distinct branches and unresolved curved intersections are invalid."""
    return _inverse(reference,field,xyz_um,stop=stop,with_details=with_details)
