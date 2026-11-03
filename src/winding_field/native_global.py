"""Experimental bounded global UV recipe; callers own source IDs, resources and rendering."""
import numpy as np
from scipy.sparse import coo_matrix,diags

from .flatten import cross2,iter_overlap_pairs
from .native_mesh import _topology
from .tifxyz_guide import _cut_flags,_sphere_seam,validate_map

MAX_FACES=1_250_000
MAX_VERTICES=750_000
MAX_CUT_VERTICES=750_000
MAX_NARROW_TESTS=20_000_000


def _check(check):
    if check:check()


def _input(vertices,faces,check):
    _check(check)
    if (not isinstance(vertices,np.ndarray) or not isinstance(faces,np.ndarray)
        or vertices.ndim!=2 or vertices.shape[1:]!=(3,) or vertices.dtype!=np.float64
        or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
        or not 0<len(vertices)<=MAX_VERTICES or not 0<len(faces)<=MAX_FACES):
        raise ValueError('bounded float64 XYZ and integer triangles required')
    if not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('invalid mesh coordinates or indices')
    areas=np.empty(len(faces))
    for first in range(0,len(faces),8192):
        _check(check);v=vertices[faces[first:first+8192]]
        area=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1)/2
        if not np.isfinite(area).all() or (area<=0).any():raise ValueError('degenerate source triangle')
        areas[first:first+len(v)]=area
    return areas


def _topology_report(v,f,areas,check):
    import igl
    _check(check)
    _,(edges,_,counts,signs),groups=_topology(np.arange(len(f)),f,areas)
    if (counts>2).any() or ((counts==2)&(signs!=0)).any():raise ValueError('nonmanifold or inconsistent edges')
    if len(np.unique(np.sort(f,axis=1),axis=0))!=len(f):raise ValueError('duplicate source faces')
    if len(groups[0])!=1 or len(np.unique(f))!=len(v):raise ValueError('one edge component without unused vertices required')
    _check(check)
    if not igl.is_vertex_manifold(f).all():raise ValueError('nonmanifold vertex')
    _check(check)
    return dict(vertices=len(v),faces=len(f),edges=len(edges),components=1,boundary_edges=int((counts==1).sum()),
                euler_characteristic=len(v)-len(edges)+len(f),edge_manifold=True,vertex_manifold=True)


def cut_component(vertices,faces,check=None):
    areas=_input(vertices,faces,check)
    import igl
    f=faces.astype(np.int64)
    original=_topology_report(vertices,f,areas,check)
    _check(check)
    if original['euler_characteristic']==1:
        paths=[]
    elif original['boundary_edges']==0 and original['euler_characteristic']==2:
        paths=[_sphere_seam(vertices,f)]
    else:
        paths=igl.cut_to_disk(f)
    _check(check);flags=_cut_flags(f,paths,len(vertices))
    # Splitting an edge can add at most its two endpoint copies.
    upper=len(vertices)+2*int(flags.sum())
    if upper>MAX_CUT_VERTICES or max(upper*3*8,len(f)*3*4,upper*8)>64<<20:
        raise ValueError('cut array bound exceeded before cut_mesh')
    _check(check);v,cf,parent=igl.cut_mesh(vertices,f,flags);_check(check)
    if len(v)>MAX_CUT_VERTICES or len(cf)!=len(f) or not np.array_equal(v,vertices[parent]) or not np.array_equal(parent[cf],f):
        raise ValueError('cut changed source correspondence or exceeded bounds')
    cf=cf.astype(np.int32);parent=parent.astype(np.int64)
    topology=_topology_report(v,cf,areas,check)
    if topology['euler_characteristic']!=1:raise ValueError('cut is not a disk')
    boundary=igl.boundary_loop(cf).astype(np.int64);_check(check)
    if len(boundary)!=topology['boundary_edges'] or len(boundary)<3:raise ValueError('cut boundary is not one disk loop')
    return dict(vertices=v,faces=cf,parent=parent,boundary=boundary),dict(original=original,cut=topology,
        cut_vertex_upper_bound=upper,source_xyz_exact=True,source_faces_order_exact=True,seam_duplicates=len(v)-len(vertices))


def _disk(cut,check):
    v,f=cut['vertices'],cut['faces'];areas=_input(v,f,check)
    import igl
    topology=_topology_report(v,f,areas,check)
    b=np.asarray(cut['boundary'])
    if (len(v)>MAX_CUT_VERTICES or b.ndim!=1 or b.dtype.kind not in 'iu' or len(b)<3
        or topology['euler_characteristic']!=1 or len(b)!=topology['boundary_edges']
        or not np.array_equal(b,igl.boundary_loop(f))):raise ValueError('invalid disk boundary')
    _check(check)
    return v,f,b


def _solve(a,rhs,check):
    from scipy.sparse.linalg import spilu,gmres,LinearOperator
    _check(check)
    if not np.isfinite(a.data).all() or not np.isfinite(rhs).all():raise ValueError('nonfinite linear system')
    ilu=spilu(a,fill_factor=8,drop_tol=1e-3,drop_rule='basic,area',permc_spec='MMD_AT_PLUS_A',
              diag_pivot_thresh=0.,options={'SymmetricMode':True})
    _check(check)
    if ilu.nnz>8*a.nnz:raise MemoryError('ILU factor exceeds fill budget')
    preconditioner=LinearOperator(a.shape,matvec=ilu.solve,dtype=np.float64)
    result=np.empty_like(rhs);iterations=[];residuals=[]
    for j in range(rhs.shape[1]):
        _check(check);count=0
        def progress(value):
            nonlocal count
            count+=1;_check(check)
            if count>300:raise RuntimeError('GMRES iteration budget exceeded')
        result[:,j],info=gmres(a,rhs[:,j],M=preconditioner,restart=30,maxiter=10,rtol=1e-11,atol=0.,
                              callback=progress,callback_type='pr_norm')
        _check(check)
        norm=float(np.linalg.norm(rhs[:,j]));error=float(np.linalg.norm(a@result[:,j]-rhs[:,j]));relative=error/norm if norm else error
        if info!=0 or not np.isfinite(result[:,j]).all() or not np.isfinite(relative) or relative>1e-10:
            raise ValueError('GMRES did not converge with a verified true residual')
        iterations.append(count);residuals.append(relative)
    return result,dict(method='spilu8-GMRES',fill_factor=8,drop_tol=1e-3,factor_nnz=int(ilu.nnz),matrix_nnz=int(a.nnz),
        unknowns=int(a.shape[0]),restart=30,max_cycles=10,maximum_iterations_per_rhs=300,iterations=iterations,
        relative_residuals=residuals,true_residual_tolerance=1e-10,true_residual_pass=True)


def _dirichlet(weights,boundary,target,check):
    _check(check);n=weights.shape[0];free=np.ones(n,bool);free[boundary]=False
    uv=np.empty((n,2));uv[boundary]=target
    if not free.any():return uv,dict(method='boundary-only',true_residual_pass=True,relative_residuals=[0.,0.])
    lap=diags(np.asarray(weights.sum(1)).ravel())-weights
    rows=lap.tocsr()[free];a=rows[:,free].tocsc();rhs=-(rows[:,boundary]@target)
    del rows,lap
    uv[free],report=_solve(a,rhs,check)
    return uv,report


def _weights(v,f,check):
    n=len(f);rows=np.empty(6*n,np.int32);cols=np.empty(6*n,np.int32);values=np.empty(6*n)
    for first in range(0,n,8192):
        _check(check);faces=f[first:first+8192];tri=v[faces]
        for i,j,k in ((0,1,2),(1,2,0),(2,0,1)):
            a,b=tri[:,j]-tri[:,i],tri[:,k]-tri[:,i];la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1)
            cross=np.linalg.norm(np.cross(a,b),axis=1);dot=np.sum(a*b,axis=1);product=la*lb
            if not np.isfinite([la,lb,cross,dot,product]).all() or (cross<=0).any() or (la<=0).any() or (lb<=0).any():
                raise ValueError('invalid mean-value geometry')
            positive=dot>=0;half=np.empty(len(faces))
            half[positive]=cross[positive]/(product[positive]+dot[positive])
            half[~positive]=(product[~positive]-dot[~positive])/cross[~positive]
            for offset,neighbor,length in ((0,j,la),(1,k,lb)):
                value=half/length
                if not np.isfinite(value).all() or (value<=0).any():raise ValueError('nonpositive mean-value weights')
                sl=slice(6*first+2*i+offset,6*(first+len(faces)),6)
                rows[sl]=faces[:,i];cols[sl]=faces[:,neighbor];values[sl]=value
    _check(check);w=coo_matrix((values,(rows,cols)),shape=(len(v),len(v))).tocsr()
    w.sum_duplicates();w.sort_indices();sums=np.asarray(w.sum(1)).ravel()
    if not np.isfinite(sums).all() or (sums<=0).any():raise ValueError('invalid mean-value row sums')
    w.data/=np.repeat(sums,np.diff(w.indptr))
    if not np.isfinite(w.data).all() or (w.data<=0).any():raise ValueError('row normalization lost positive weights')
    return w


def mean_value_map(cut,check=None):
    v,f,b=_disk(cut,check)
    import igl
    target=igl.map_vertices_to_circle(v,b.astype(np.int32));_check(check)
    adjacency=igl.adjacency_matrix(f).astype(float);_check(check)
    uv,uniform=_dirichlet(adjacency,b,target,check);del adjacency
    t=uv[f];signed=cross2(t[:,1]-t[:,0],t[:,2]-t[:,0])
    if (signed<0).all():uv[:,0]*=-1
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    native=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1);mapped=np.linalg.norm(uv[edges[:,1]]-uv[edges[:,0]],axis=1)
    if not np.isfinite(mapped).all() or (mapped<=0).any():raise ValueError('collapsed uniform initialization')
    scale=float(native.sum()/mapped.sum())
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid physical UV scale')
    uv-=uv.min(0);uv*=scale
    _check(check);validate_map(dict(world=v,faces=f,uv=uv));_check(check)
    fixed=uv[b].copy();weights=_weights(v,f,check)
    # MVC rows were normalized in the control: diagonal is exactly one.
    weights.setdiag(0);weights.eliminate_zeros()
    free=np.ones(len(v),bool);free[b]=False;result=np.empty_like(uv);result[b]=fixed
    if free.any():
        lap=diags(np.ones(len(v)))-weights;rows=lap.tocsr()[free]
        result[free],mvc=_solve(rows[:,free].tocsc(),-(rows[:,b]@fixed),check)
    else:mvc=dict(method='boundary-only',true_residual_pass=True,relative_residuals=[0.,0.])
    _check(check);validate_map(dict(world=v,faces=f,uv=result));_check(check)
    if not np.array_equal(result[b],fixed):raise ValueError('MVC boundary changed')
    return result,dict(uniform=uniform,mvc=mvc,physical_scale=scale,positive_disk_validated=True,boundary_exact=True,
        initialization='same uniform equations and physical scale; ILU-GMRES replaces direct solve, not bit-identity claimed')


def arap_map(cut,uv,check=None):
    v,f,b=_disk(cut,check)
    import igl
    if uv.shape!=(len(v),2) or not np.isfinite(uv).all():raise ValueError('invalid ARAP initialization')
    _check(check);validate_map(dict(world=v,faces=f,uv=uv));_check(check)
    a=int(b[np.lexsort((b,uv[b,1],uv[b,0]))[0]])
    distance=np.sum((uv[b]-uv[a])**2,axis=1);other=int(np.min(b[distance==distance.max()]))
    if a==other or not np.isfinite(distance).all():raise ValueError('invalid automatic pins')
    pins=np.array([a,other],np.int32);fixed=uv[pins].copy();current=uv.copy()
    data=igl.ARAPData();data.energy=igl.ARAP_ENERGY_TYPE_ELEMENTS;data.with_dynamics=False;data.max_iter=1
    _check(check);igl.arap_precomputation(v,f.astype(np.int64),2,pins,data);_check(check);data.max_iter=1
    for _ in range(20):
        _check(check);current=igl.arap_solve(fixed,data,current);_check(check)
        if current.shape!=uv.shape or not np.isfinite(current).all() or not np.array_equal(current[pins],fixed):
            raise ValueError('ARAP nonfinite output or hard pins changed')
    return current,dict(method='direct libigl ARAP ELEMENTS',iterations=20,maximum_iterations_per_call=1,
        pin_indices=pins.tolist(),pin_uv=fixed.tolist(),pins_exact=True,qualified=False,
        pin_rule='lexicographically leftmost boundary UV, then farthest; cut-index tie break')


def qualify_map(cut,uv,check=None):
    v,f=cut['vertices'],cut['faces'];area=_input(v,f,check)
    if uv.shape!=(len(v),2):raise ValueError('invalid qualification UV shape')
    reasons=np.zeros(len(f),np.uint8)
    for first in range(0,len(f),8192):
        _check(check);part=f[first:first+8192];t=uv[part];w=v[part];bits=np.zeros(len(part),np.uint8)
        finite=np.isfinite(t).all((1,2));bits[~finite]|=1
        with np.errstate(invalid='ignore',over='ignore'):
            signed=cross2(t[:,1]-t[:,0],t[:,2]-t[:,0]);bits[(signed<=1e-12)|~np.isfinite(signed)]|=2
            for i,j in ((0,1),(1,2),(2,0)):
                ratio=np.linalg.norm(t[:,i]-t[:,j],axis=1)/np.linalg.norm(w[:,i]-w[:,j],axis=1)
                bits[(ratio<.25)|(ratio>4)|~np.isfinite(ratio)]|=4
        reasons[first:first+len(part)]=bits
    selected=np.flatnonzero(reasons==0);audit={}
    for i,j in iter_overlap_pairs(uv,f[selected],check,max_candidates=262144,max_tests=MAX_NARROW_TESTS,report=audit):
        reasons[selected[[i,j]]]|=8
    valid=reasons==0;qualified=float(area[valid].sum());total=float(area.sum())
    labels=np.full(len(f),-1,np.int64);largest_area=0.;largest_faces=0;components=0
    if valid.any():
        _check(check);roots,_,groups=_topology(np.flatnonzero(valid),f[valid],area[valid]);_check(check)
        labels[valid]=roots;components=len(groups[0]);largest=int(np.argmax(groups[2]))
        largest_area=float(groups[2][largest]);largest_faces=int(groups[1][largest])
    report=dict(source_faces=len(f),source_area_um2=total,qualified_faces=int(valid.sum()),qualified_area_um2=qualified,
        qualified_area_fraction=qualified/total,excluded_faces=int((~valid).sum()),excluded_area_um2=float(area[~valid].sum()),
        components=components,largest_component_faces=largest_faces,largest_component_area_um2=largest_area,
        largest_component_area_fraction=largest_area/qualified if qualified else 0.,overlap_audit=audit,
        reason_bits=dict(nonfinite=1,area=2,stretch=4,overlap=8),
        exclusions={name:dict(faces=int(((reasons&bit)!=0).sum()),area_um2=float(area[(reasons&bit)!=0].sum()))
                    for name,bit in (('nonfinite',1),('area',2),('stretch',4),('overlap',8))})
    _check(check)
    return dict(valid_faces=valid,reasons=reasons,face_area_um2=area,component_labels=labels),report
