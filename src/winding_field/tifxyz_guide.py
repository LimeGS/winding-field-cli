"""Bounded global guide. UV helpers adapted from mesh_to_tifxyz/geometry.py (MIT)."""
import math
import numpy as np

from .tifxyz_contract import Blocked,Paused
from .tifxyz_field import load_coarse


def stride_schedule(shape):
    if len(shape)!=3 or min(shape)<2 or max(shape)<3:raise Blocked('COARSE_GRID_UNREPRESENTABLE')
    s=2
    while math.prod((n-2)//s+2 for n in shape)>4_000_000:s*=2
    return [4*s,2*s,s]


def validate_input(v,f):
    if (v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all()
        or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v)):
        raise ValueError('finite triangular geometry required')
    for first in range(0,len(f),32768):
        t=v[f[first:first+32768]];area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
        if not np.isfinite(area).all() or (area<=0).any():raise ValueError('degenerate guide triangle')


def validate_orientation(f):
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    directions=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    if ((counts==2)&(directions!=0)).any():raise ValueError('inconsistent shared-edge orientation')


def validate_map(m):
    import igl
    v,f,uv=m['world'],m['faces'],m['uv'];validate_input(v,f);validate_orientation(f)
    if uv.shape!=(len(v),2) or not np.isfinite(uv).all():raise ValueError('invalid UV')
    if (int(igl.euler_characteristic(f))!=1 or int(igl.facet_components(f)[0])!=1
        or not igl.is_edge_manifold(f)[0] or not igl.is_vertex_manifold(f).all()):raise ValueError('UV domain must be a manifold disk')
    for first in range(0,len(f),32768):
        t=uv[f[first:first+32768]].astype(np.longdouble);a,b=t[:,1]-t[:,0],t[:,2]-t[:,0]
        if not ((a[:,0]*b[:,1]-a[:,1]*b[:,0])>0).all():raise ValueError('inverted UV triangle')
    boundary=igl.boundary_loop(f).astype(np.int64);p=uv[boundary].astype(np.longdouble)
    a=np.roll(p,-1,axis=0)-p;b=np.roll(a,-1,axis=0);turns=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if not ((turns>=0).all() or (turns<=0).all()):raise ValueError('nonconvex boundary')
    winding=float(np.arctan2(turns,np.sum(a*b,axis=1)).sum())
    if len(np.unique(p,axis=0))!=len(p) or not np.isclose(abs(winding),2*np.pi,rtol=1e-10,atol=1e-10):
        raise ValueError('boundary is not simple convex')


def _cut_flags(f,paths,nv):
    pairs=[np.column_stack([p[:-1],p[1:]]) for p in paths if len(p)>1];flags=np.zeros(f.shape,bool)
    if not pairs:return flags
    pairs=np.sort(np.concatenate(pairs).astype(np.int64),axis=1);keys=np.unique(pairs[:,0]*nv+pairs[:,1])
    for edge in range(3):
        a,b=f[:,edge],f[:,(edge+1)%3];q=np.minimum(a,b)*nv+np.maximum(a,b)
        index=np.searchsorted(keys,q);valid=index<len(keys);flags[valid,edge]=keys[index[valid]]==q[valid]
    return flags


def _sphere_seam(v,f):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    graph=coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    a=int(np.argmax(dijkstra(graph,directed=False,indices=0)));distance,parent=dijkstra(graph,directed=False,indices=a,return_predecessors=True)
    current=int(np.argmax(distance));path=[current]
    while current!=a:
        current=int(parent[current])
        if current<0 or len(path)>len(v):raise ValueError('invalid sphere seam')
        path.append(current)
    if len(path)==2:
        neighbors=graph.indices[graph.indptr[path[-1]]:graph.indptr[path[-1]+1]]
        path.append(next(int(x) for x in neighbors if x not in path))
    return np.array(path,np.int64)


def parameterize(v,f,stop=None):
    import igl
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    validate_input(v,f);validate_orientation(f)
    if int(igl.facet_components(f)[0])!=1:raise Blocked('GUIDE_MULTIPLE_COMPONENTS')
    if not igl.is_edge_manifold(f)[0] or not igl.is_vertex_manifold(f).all():raise Blocked('GUIDE_NONMANIFOLD')
    original_normals=igl.per_vertex_normals(v,f)
    paths=[_sphere_seam(v,f)] if not len(igl.boundary_loop(f)) and int(igl.euler_characteristic(f))==2 else igl.cut_to_disk(f)
    world,faces,parent=igl.cut_mesh(v,f,_cut_flags(f,paths,len(v)))
    if not np.array_equal(world,v[parent]) or not np.array_equal(parent[faces],f):raise ValueError('guide cuts changed geometry')
    if int(igl.euler_characteristic(faces))!=1:raise Blocked('GUIDE_NOT_DISK')
    boundary=igl.boundary_loop(faces).astype(np.int32);uv=np.zeros((len(world),2));uv[boundary]=igl.map_vertices_to_circle(world,boundary)
    free=np.setdiff1d(np.arange(len(world)),boundary);adj=igl.adjacency_matrix(faces).astype(float)
    lap=diags(np.asarray(adj.sum(1)).ravel())-adj
    if stop and stop():raise Paused()
    if len(free):uv[free]=spsolve(lap.tocsc()[free][:,free],-lap.tocsr()[free][:,boundary]@uv[boundary])
    t=uv[faces];a,b=t[:,1]-t[:,0],t[:,2]-t[:,0]
    if ((a[:,0]*b[:,1]-a[:,1]*b[:,0])<0).all():uv[:,0]*=-1
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    native=np.linalg.norm(world[edges[:,1]]-world[edges[:,0]],axis=1);mapped=np.linalg.norm(uv[edges[:,1]]-uv[edges[:,0]],axis=1)
    if (mapped<=0).any() or not np.isfinite(mapped).all():raise Blocked('GUIDE_COLLAPSED_UV')
    uv-=uv.min(0);uv*=float(native.sum()/mapped.sum())
    result=dict(world=world,faces=faces,uv=uv,normals=original_normals[parent],source_vertices=parent,boundary=boundary)
    validate_map(result)
    if len(free):
        fixed=uv[boundary].copy()
        data=igl.slim_precompute(world,faces.astype(np.int32),uv,igl.MappingEnergyType.SYMMETRIC_DIRICHLET,boundary,fixed,1e12)
        for _ in range(50):
            if stop and stop():raise Paused()
            candidate=igl.slim_solve(data,1)
        result['uv']=candidate.copy();result['uv'][boundary]=fixed;validate_map(result)
    return result


def build_guide(coarse,level,settings=None,stop=None):
    from skimage.measure import marching_cubes
    values,cells,axes=load_coarse(coarse)
    if not np.isfinite(level):raise ValueError('finite guide level required')
    if not cells.any():raise Blocked('GUIDE_NO_SUPPORT')
    if stop and stop():raise Paused()
    mask=np.zeros(values.shape,bool);mask[1:,1:,1:]=cells
    work=np.where(np.isfinite(values),values-level,0).astype(np.float32)
    if not np.isfinite(work).all():raise Blocked('GUIDE_FLOAT_RANGE')
    try:v,f,hints,_=marching_cubes(work,0.,mask=mask,allow_degenerate=False)
    except (RuntimeError,ValueError) as error:raise Blocked('GUIDE_NO_SURFACE',str(error)) from error
    if len(f)>250000:raise Blocked('GUIDE_FACE_LIMIT',str(len(f)))
    spacing=np.asarray(coarse['spacing_um']);origin=np.asarray(coarse['origin_um'])
    nodes=np.column_stack([np.interp(v[:,i],np.arange(len(a)),a) for i,a in enumerate(axes)])
    world=origin+nodes[:,::-1]*spacing;faces=f[:,::-1].astype(np.int32)
    lengths=np.column_stack([np.diff(a)[np.minimum(v[:,i].astype(int),len(a)-2)] for i,a in enumerate(axes)])
    hint=-hints[:,::-1]/(lengths[:,::-1]*spacing)
    tri=world[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if np.sum(cross*hint[faces].mean(1))<0:faces=faces[:,::-1].copy()
    guide=parameterize(world,faces,stop)
    guide.update(stride=coarse['stride'],level=float(level),coarse=coarse,geometry_audit='PENDING_FULL_AUDIT')
    return guide
