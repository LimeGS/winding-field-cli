"""Historical Flatboi patches on immutable meshes; original-edge seams, no angle cut.

Registration/weld equations adapted from six-surfaces-opt-20261016/global_sheet.py;
positive-face recovery from face_recovery.py. No historical wrapper is executed.
"""
from collections import defaultdict, OrderedDict, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, wait
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import threading

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .original_pipeline import frame_coordinates
from .original_recipe import raster_sheet, face_quality
from .surface_atlas import face_components, local_chart, split_faces
from .surface_export import flatten_native
from .util import atomic_json, file_sha

LAUNCHER_SHA='c1e0ca2ea1616967f6da5907c60ea3fe114e840834154ccde758856f85510517'
BINARY_SHA='d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38'
NATIVE_RECIPES={'historical':BINARY_SHA,
    'installed-2f89bfc9':'2f89bfc96512bebf7ef44b25de01c5afd01c7af058e452515b03bc52cf6afd6a'}


def settings(request,*,native_required=True):
    from .original_pipeline import settings as frame_settings
    if not isinstance(request,dict):raise ValueError('patch settings must be an object')
    allowed={'output_frame','pitch_um','app_run','max_patch_faces','max_depth','max_charts','tile_size',
             'memory_gib','timeout_seconds','cache_dir','cache_mib','cache_gib','native_recipe','ct_sampler',
             'workers','raster_workers','ct_workers','native_cpus','reuse_export','reuse_identity_sha256','process_tree_gib','disk_reserve_gib'}
    if set(request)-allowed:raise ValueError('unknown patch settings: '+str(sorted(set(request)-allowed)))
    frame_settings({'output_frame':request.get('output_frame')})
    cfg=dict(pitch_um=19.2,max_patch_faces=4096,max_depth=3,max_charts=4096,tile_size=1024,
             memory_gib=2.,timeout_seconds=120.,cache_mib=64,cache_gib=2.,
               cache_dir='~/.cache/winding-field/patch-ct',native_recipe='historical',ct_sampler='native-float64',workers=1,raster_workers=1,ct_workers=1,
             native_cpus=None,reuse_export=None,reuse_identity_sha256=None,process_tree_gib=11.,disk_reserve_gib=8.)
    cfg.update(request)
    if cfg['ct_sampler'] not in ('scalar','native-float64'):raise ValueError('invalid ct_sampler')
    if not isinstance(cfg['native_recipe'],str) or cfg['native_recipe'] not in NATIVE_RECIPES:raise ValueError('unknown native_recipe')
    if native_required or cfg.get('app_run') is not None:
        if not isinstance(cfg.get('app_run'),str) or not cfg['app_run']:raise ValueError('patch app_run required')
    for key in ('pitch_um','memory_gib','timeout_seconds','cache_gib','process_tree_gib','disk_reserve_gib'):
        if type(cfg[key]) not in (int,float) or not math.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError('invalid patch '+key)
    for key,lo,hi in (('workers',1,4),('raster_workers',1,4),('ct_workers',1,4),('max_patch_faces',1,100000),('max_depth',0,16),('max_charts',1,100000),('tile_size',2,2048),('cache_mib',0,8192)):
        if type(cfg[key]) is not int or not lo<=cfg[key]<=hi:raise ValueError('invalid patch '+key)
    if raster_memory(cfg)>=cfg['memory_gib']*2**30:raise ValueError('patch memory budget cannot hold raster, CT cache and runtime reserve')
    cpus=cfg['native_cpus']
    if cpus is not None and (not isinstance(cpus,list) or not cpus or any(type(c) is not int or c<0 for c in cpus) or len(set(cpus))!=len(cpus)):
        raise ValueError('invalid native CPU set')
    if cfg['workers']>1 and (cpus is None or len(cpus)<cfg['workers']):raise ValueError('parallel workers require an explicit CPU set with at least one CPU per worker')
    if (cfg['reuse_export'] is None)!=(cfg['reuse_identity_sha256'] is None):raise ValueError('reuse requires export path and identity SHA256')
    if cfg['reuse_export'] is not None and (not isinstance(cfg['reuse_export'],str) or not cfg['reuse_export'] or not isinstance(cfg['reuse_identity_sha256'],str) or len(cfg['reuse_identity_sha256'])!=64):raise ValueError('invalid reuse export binding')
    return cfg


def raster_memory(cfg):
    pixels=(cfg['tile_size']+4)**2
    # Eight-corner jobs, concatenation, stable sort and reordered arrays coexist.
    sampling=min(pixels,100000)*1024
    return (pixels*160+sampling)*cfg.get('raster_workers',1)+cfg['cache_mib']*2**20+(128<<20)


def ct_memory(chunks,dtype,cfg):
    decoded=math.prod(chunks)*np.dtype(dtype).itemsize
    # Native decode + codec workspace + new Python bytes + previous sampler chunk.
    reserved=4*decoded*(cfg.get('raster_workers',1)+(cfg.get('ct_workers',1) if cfg.get('ct_workers',1)>1 else 0))
    if raster_memory(cfg)+reserved>=cfg['memory_gib']*2**30:
        raise ValueError('CT chunk memory exceeds whole patch memory budget')
    return reserved


def check_flatboi(app_run,recipe='historical'):
    if recipe not in NATIVE_RECIPES:raise ValueError('unknown native_recipe')
    expected=NATIVE_RECIPES[recipe]
    app=Path(app_run).expanduser().resolve();binary=app.parent/'usr/bin/flatboi'
    if not app.is_file() or file_sha(app)!=LAUNCHER_SHA:raise ValueError('historical Flatboi launcher SHA256 mismatch or unavailable: '+str(app))
    if not binary.is_file() or file_sha(binary)!=expected:raise ValueError(recipe+' Flatboi binary SHA256 mismatch or unavailable: '+str(binary))
    if not os.access(app,os.X_OK):raise ValueError('historical Flatboi launcher is not executable')
    return dict(app_run=str(app),launcher_sha256=LAUNCHER_SHA,binary_sha256=expected,recipe=recipe,historical_reproduction=recipe=='historical',
                iterations=200,energy='symmetric_dirichlet',relative_energy_tolerance=1e-6)


def boundary_edges(faces):
    directed=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,inv,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    if (counts>2).any():raise ValueError('nonmanifold patch edges')
    return directed[counts[inv]==1]


def seam_links(charts):
    owners=defaultdict(list);pairs={}
    counts=dict(shared_edges=0,unmatched_edges=0,ambiguous_edges=0,angular_cut_edges=0)
    for i,c in enumerate(charts):
        c['edges']=boundary_edges(c['f'])
        for a,b in c['edges']:
            ka,kb=int(c['vi'][a]),int(c['vi'][b])
            owners[tuple(sorted((ka,kb)))].append((i,int(a),int(b),ka<kb))
    for entries in owners.values():
        if len(entries)!=2:
            counts['unmatched_edges' if len(entries)==1 else 'ambiguous_edges']+=1;continue
        (i,a,b,direction),(j,c,d,other)=entries
        if i==j or direction==other:
            counts['ambiguous_edges']+=1;continue
        if not np.array_equal(charts[i]['v'][[a,b]],charts[j]['v'][[d,c]]):raise ValueError('original seam XYZ differs')
        pair=pairs.setdefault((i,j),dict(i=i,j=j,vertices={},length=0.))
        for x,y in ((a,d),(b,c)):
            if pair['vertices'].get(x,y)!=y:raise ValueError('ambiguous original seam ID')
            pair['vertices'][x]=y
        pair['length']+=float(np.linalg.norm(charts[i]['v'][a]-charts[i]['v'][b]));counts['shared_edges']+=1
    links=[]
    for pair in pairs.values():
        vi=sorted(pair.pop('vertices').items());links.append(dict(pair,vi=np.array([a for a,b in vi]),vj=np.array([b for a,b in vi])))
    return links,counts


def rotation(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s],[s,c]])


def place(uv,pose):return uv@rotation(pose[0]).T+pose[1:]


def rigid_fit(a,b):
    ac,bc=a.mean(0),b.mean(0)
    if np.sum((a-ac)**2)<=1e-20:raise ValueError('zero-length seam')
    u,_,vt=np.linalg.svd((a-ac).T@(b-bc))
    if np.linalg.det(vt.T@u.T)<0:vt[-1]*=-1
    r=vt.T@u.T
    return r,bc-r@ac


def fit_poses(charts,links,component):
    """Historical sparse Gauss-Newton rigid registration, including line search."""
    root=min(component);selected=set(component);edges=[e for e in links if e['i'] in selected and e['j'] in selected]
    neighbours=defaultdict(list)
    for e in edges:neighbours[e['i']].append(e);neighbours[e['j']].append(e)
    poses={root:np.zeros(3)};pending=[];serial=0
    def add(node):
        nonlocal serial
        for e in neighbours[node]:
            other=e['j'] if e['i']==node else e['i']
            if other not in poses:
                heapq.heappush(pending,(-e['length'],serial,node,other,e));serial+=1
    add(root)
    while pending:
        _,_,known,new,e=heapq.heappop(pending)
        if new in poses:continue
        vi,vj=(e['vi'],e['vj']) if known==e['i'] else (e['vj'],e['vi'])
        r,t=rigid_fit(charts[new]['uv'][vj],charts[known]['uv'][vi]);parent=poses[known]
        poses[new]=np.r_[parent[0]+math.atan2(r[1,0],r[0,0]),rotation(parent[0])@t+parent[1:]];add(new)
    if set(poses)!=selected:raise ValueError('disconnected seam graph')
    free=sorted(selected-{root});columns={i:k*3 for k,i in enumerate(free)}
    x=np.concatenate([poses[i] for i in free]) if free else np.empty(0)
    ii=[];jj=[];aa=[];bb=[];ww=[]
    for e in edges:
        pick=np.unique(np.linspace(0,len(e['vi'])-1,min(48,len(e['vi']))).astype(int))
        ii.extend([e['i']]*len(pick));jj.extend([e['j']]*len(pick))
        aa.extend(charts[e['i']]['uv'][e['vi'][pick]]);bb.extend(charts[e['j']]['uv'][e['vj'][pick]])
        ww.extend([math.sqrt(e['length']/len(pick))]*len(pick))
    ii,jj=np.array(ii,int),np.array(jj,int);aa,bb=np.asarray(aa).reshape(-1,2),np.asarray(bb).reshape(-1,2);weight=np.asarray(ww)
    def values(x):
        p=np.zeros((len(charts),3))
        for i in free:p[i]=x[columns[i]:columns[i]+3]
        return p
    def residual(x):
        p=values(x)
        def transform(ids,points):
            c,s=np.cos(p[ids,0]),np.sin(p[ids,0])
            return np.column_stack([c*points[:,0]-s*points[:,1],s*points[:,0]+c*points[:,1]])+p[ids,1:]
        return ((transform(ii,aa)-transform(jj,bb))*weight[:,None]).ravel()
    def jacobian(x):
        p=values(x);rows=[];cols=[];data=[];k=np.arange(len(ii))
        for ids,points,sign in ((ii,aa,1.),(jj,bb,-1.)):
            use=ids!=root;base=np.array([columns.get(int(i),0) for i in ids]);c,s=np.cos(p[ids,0]),np.sin(p[ids,0])
            for row,offset,derivative in ((2*k,0,-s*points[:,0]-c*points[:,1]),(2*k+1,0,c*points[:,0]-s*points[:,1]),(2*k,1,np.ones(len(k))),(2*k+1,2,np.ones(len(k)))):
                rows.extend(row[use]);cols.extend((base+offset)[use]);data.extend((sign*weight*derivative)[use])
        return coo_matrix((data,(rows,cols)),shape=(2*len(ii),len(x))).tocsr()
    r=residual(x);initial=cost=float(r@r);converged=cost<1e-20;iterations=0;scale=np.tile([.01,1.,1.],len(free))
    for iteration in range(200):
        if converged or not len(x):break
        r=residual(x);j=jacobian(x).multiply(scale).tocsr();h=(j.T@j).tocsc()
        step=spsolve(h+diags(np.maximum(h.diagonal(),1)*1e-12),-(j.T@r))*scale
        if not np.isfinite(step).all():raise ValueError('nonfinite registration step')
        fraction=1.
        for _ in range(24):
            proposed=x+fraction*step;trial=residual(proposed);next_cost=float(trial@trial)
            if next_cost<=cost+1e-14:break
            fraction*=.5
        else:raise ValueError('registration did not descend')
        converged=abs(cost-next_cost)<=1e-10*max(1.,cost) or np.max(abs(fraction*step))<1e-8
        x,cost,iterations=proposed,next_cost,iteration+1
    p=values(x)
    return {i:p[i] for i in component},dict(initial_cost=initial,cost=cost,iterations=iterations,converged=bool(converged))


def seam_targets(charts,links,component,poses,*,world_unit_um=9.6):
    selected=set(component);occurrence={};parents=[]
    def add(key):
        if key not in occurrence:occurrence[key]=len(parents);parents.append(len(parents))
        return occurrence[key]
    def find(k):
        while parents[k]!=k:parents[k]=parents[parents[k]];k=parents[k]
        return k
    for e in links:
        if e['i'] in selected and e['j'] in selected:
            for vi,vj in zip(e['vi'],e['vj']):
                a,b=add((e['i'],int(vi))),add((e['j'],int(vj)));parents[find(b)]=find(a)
    if not occurrence:return {}
    sums=defaultdict(lambda:np.zeros(2));counts=defaultdict(int)
    for (i,vi),k in occurrence.items():
        root=find(k);sums[root]+=place(charts[i]['uv'][vi:vi+1],poses[i])[0];counts[root]+=1
    roots=sorted(sums);compact={r:i for i,r in enumerate(roots)};base=np.array([sums[r]/counts[r] for r in roots])
    rows=[];cols=[];weights=[];rhs=np.zeros_like(base)
    for i in component:
        c=charts[i]
        for a,b in c['edges']:
            ka,kb=(i,int(a)),(i,int(b))
            if ka not in occurrence or kb not in occurrence:continue
            ga,gb=compact[find(occurrence[ka])],compact[find(occurrence[kb])]
            if ga==gb:raise ValueError('seam collapses original edge')
            # Historical 256 L2^2 regularizer and 1e-4 L2 floor, converted to declared units.
            unit_scale=9.6/world_unit_um
            length=float(np.linalg.norm(c['v'][a]-c['v'][b]));weight=256*unit_scale**2/max(length,1e-4*unit_scale)**2
            error=rotation(poses[i][0])@(c['uv'][b]-c['uv'][a])-(base[gb]-base[ga])
            rows.extend([ga,gb,ga,gb]);cols.extend([ga,gb,gb,ga]);weights.extend([weight,weight,-weight,-weight])
            rhs[ga]-=weight*error;rhs[gb]+=weight*error
    system=coo_matrix((weights,(rows,cols)),shape=(len(roots),len(roots))).tocsc()+diags([counts[r] for r in roots],dtype=float)
    corrected=base+spsolve(system,rhs)
    if not np.isfinite(corrected).all():raise ValueError('nonfinite seam targets')
    targets=defaultdict(dict)
    for (i,vi),k in occurrence.items():targets[i][vi]=corrected[compact[find(k)]]
    return targets


def weld_chart(uv,faces,fixed,target):
    if not len(fixed):return uv.copy()
    t=uv[faces];a,b=t[:,1]-t[:,0],t[:,2]-t[:,0]
    faces=faces[(a[:,0]*b[:,1]-a[:,1]*b[:,0])!=0]
    if not len(faces):
        result=uv.copy();result[fixed]=target;return result
    rows=[];cols=[];data=[];tri=uv[faces]
    for a,b,opposite in ((0,1,2),(1,2,0),(2,0,1)):
        x,y=tri[:,a]-tri[:,opposite],tri[:,b]-tri[:,opposite];den=abs(x[:,0]*y[:,1]-x[:,1]*y[:,0])
        if (den<=0).any():raise ValueError('degenerate native UV triangle')
        weight=.5*np.sum(x*y,axis=1)/den;i,j=faces[:,a],faces[:,b]
        rows.extend([i,j,i,j]);cols.extend([i,j,j,i]);data.extend([weight,weight,-weight,-weight])
    matrix=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(len(uv),len(uv))).tocsr()
    free=np.setdiff1d(np.unique(faces),fixed);delta=np.zeros_like(uv);delta[fixed]=target-uv[fixed]
    if len(free):delta[free]=spsolve(matrix[free][:,free].tocsc(),-(matrix[free][:,fixed]@delta[fixed]))
    result=uv+delta;result[fixed]=target
    if not np.isfinite(result).all():raise ValueError('nonfinite seam extension')
    return result


class NativeQueue:
    """Bounded lookahead; only DFS consumption assigns chart IDs/budget."""
    def __init__(self,v,f,output,flatten,workers,max_faces,guard):
        self.v,self.f,self.output,self.flatten=v,f,output,flatten
        self.workers,self.max_faces,self.guard=workers,max_faces,guard
        self.pool=ThreadPoolExecutor(max_workers=workers);self.jobs={};self.serial=0

    def submit(self,ids):
        key=id(ids)
        if key not in self.jobs:
            cv,cf,vi=local_chart(self.v,self.f,ids)
            path=self.output/f'_pending-{self.serial:06d}';self.serial+=1
            self.jobs[key]=(self.pool.submit(self.flatten,cv,cf,path),cv,cf,vi,path)

    def get(self,ids,pending,remaining,directory):
        self.submit(ids)
        # Reserve one slot for the next DFS child even when siblings finish first.
        index=len(pending)-1;lookahead=0
        while index>=0 and len(self.jobs)<self.workers and lookahead<remaining-1:
            other,depth=pending[index]
            if len(other)>self.max_faces:
                parts=[(part,depth) for part in reversed(split_faces(self.v,self.f,other))]
                pending[index:index+1]=parts;index+=len(parts)-1;continue
            self.submit(other);index-=1;lookahead+=1
        future,cv,cf,vi,path=self.jobs[id(ids)]
        while True:
            self.guard()
            try:uv,receipt=future.result(timeout=.1);break
            except FutureTimeout:
                if future.done():raise
        del self.jobs[id(ids)]
        if path.exists():path.rename(directory)
        return cv,cf,vi,uv,receipt

    def close(self):
        for future,*_ in self.jobs.values():future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)


def build_map(vertices,faces,output,*,flatten,max_patch_faces=4096,max_depth=3,max_charts=4096,world_unit_um=9.6,guard=lambda:None,workers=1,cancel=lambda:None):
    """All face ownership survives failures; no largest-component selection."""
    v,f=np.asarray(vertices,float),np.asarray(faces,np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or (len(f) and (f.min()<0 or f.max()>=len(v))):raise ValueError('finite indexed mesh required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    components=face_components(f,np.arange(len(f))) if len(f) else []
    tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    del tri
    charts=[];omitted=[];attempts=[];pending=[(ids,0) for ids in reversed(components)]
    queue=NativeQueue(v,f,output,flatten,workers,max_patch_faces,guard) if workers>1 else None
    try:
        while pending:
            guard();ids,depth=pending.pop()
            if len(ids)>max_patch_faces:
                pending.extend((part,depth) for part in reversed(split_faces(v,f,ids)));continue
            if len(attempts)>=max_charts:
                omitted.extend(ids.tolist());attempts.append(dict(status='OMITTED',reason='chart budget',face_rows=ids.tolist()));continue
            directory=output/f'patch-{len(attempts):06d}'
            if queue:cv,cf,vi,uv,receipt=queue.get(ids,pending,max_charts-len(attempts),directory)
            else:
                cv,cf,vi=local_chart(v,f,ids);uv,receipt=flatten(cv,cf,directory)
            attempts.append(dict(native=receipt,face_rows=ids.tolist()))
            if uv is not None:
                uv=np.asarray(uv,float).copy()
                if uv.shape!=(len(cv),2) or not np.isfinite(uv).all():uv=None
            if uv is not None:
                t=uv[cf];a,b=t[:,1]-t[:,0],t[:,2]-t[:,0];signed=a[:,0]*b[:,1]-a[:,1]*b[:,0]
                if areas[ids][signed<0].sum()>areas[ids][signed>0].sum():uv[:,0]*=-1;signed*=-1
                # Invalid faces retain the SAME chart deformation/placement as ambiguity blockers.
                charts.append(dict(v=cv,f=cf,vi=vi,fi=ids,uv=uv,eligible=signed>0))
            elif depth<max_depth and len(ids)>1:
                pending.extend((part,depth+1) for part in reversed(split_faces(v,f,ids)))
            else:omitted.extend(ids.tolist())
    finally:
        if queue:
            cancel();queue.close()
    cv=cf=vi=uv=t=a=b=signed=None
    links,seams=seam_links(charts)
    from scipy.sparse.csgraph import connected_components
    graph=coo_matrix((np.ones(len(links)),([e['i'] for e in links],[e['j'] for e in links])),shape=(len(charts),len(charts))).tocsr()
    count,labels=connected_components(graph,directed=False) if charts else (0,np.empty(0,int))
    del graph
    registrations=[];offset=0.
    for label in range(count):
        guard();ids=np.flatnonzero(labels==label).tolist()
        poses,report=fit_poses(charts,links,ids);targets=seam_targets(charts,links,ids,poses,world_unit_um=world_unit_um)
        for i in ids:
            c=charts[i];fixed=np.array(sorted(targets.get(i,{})),int)
            c['uv']=weld_chart(place(c['uv'],poses[i]),c['f'],fixed,np.array([targets[i][k] for k in fixed]))
        del poses,targets,fixed
        lo=np.min([charts[i]['uv'].min(0) for i in ids],axis=0);hi=np.max([charts[i]['uv'].max(0) for i in ids],axis=0)
        shift=np.array([offset,0])-lo
        for i in ids:charts[i]['uv']+=shift
        offset+=hi[0]-lo[0]+max(1.,float(np.max(hi-lo))*.05)
        registrations.append(dict(report,charts=ids,translation=shift.tolist(),disconnected_placement=True))
    del labels
    vertex_count=sum(len(c['vi']) for c in charts);face_count=sum(len(c['fi']) for c in charts)
    offsets=np.r_[0,np.cumsum([len(c['vi']) for c in charts])]
    eligible=np.concatenate([c['eligible'] for c in charts]) if charts else np.zeros(0,bool)
    parent=np.empty(vertex_count,np.int64);outfaces=np.empty((face_count,3),np.int64)
    uvs=np.empty((vertex_count,2));rows=np.empty(face_count,np.int64);start=face_start=0
    for chart_id,c in enumerate(charts):
        end=start+len(c['vi']);face_end=face_start+len(c['fi'])
        parent[start:end]=c['vi'];outfaces[face_start:face_end]=c['f']+start;uvs[start:end]=c['uv'];rows[face_start:face_end]=c['fi']
        start,face_start=end,face_end
        charts[chart_id]=None
    charts.clear();c=None
    # Weld topology as well as coordinates: duplicate chart vertices would invent TIFXYZ boundaries.
    roots=np.arange(len(parent))
    def find(k):
        while roots[k]!=k:roots[k]=roots[roots[k]];k=roots[k]
        return k
    for e in links:
        for a,b in zip(offsets[e['i']]+e['vi'],offsets[e['j']]+e['vj']):
            if parent[a]!=parent[b] or not np.array_equal(uvs[a],uvs[b]):raise ValueError('welded seam differs')
            roots[find(b)]=find(a)
    del links,offsets
    e=None
    representatives=np.array([find(k) for k in range(len(parent))],int)
    used,inverse=np.unique(representatives,return_inverse=True)
    parent,uvs,outfaces=parent[used],uvs[used],inverse[outfaces]
    del find,roots,representatives,used,inverse
    if not np.array_equal(np.sort(np.r_[rows,omitted].astype(int)),np.arange(len(f))):raise ValueError('patch face ownership is not an exact partition')
    quality={}
    if len(rows):
        _,scientific,quality=face_quality(v[parent],outfaces,uvs)
        t=uvs[outfaces];eligible&=((t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0]))>0
        scientific&=eligible
        del t,_
    else:scientific=np.zeros(0,bool)
    omitted_uv=np.r_[np.asarray(omitted,int),rows[~eligible]]
    report=dict(source_faces=len(f),source_components=len(components),source_area_world2=float(areas.sum()),
                components=[dict(faces=len(ids),area_world2=float(areas[ids].sum()),omitted_uv_faces=int(np.isin(ids,omitted_uv).sum())) for ids in components],
                omitted_uv_faces=len(omitted_uv),omitted_uv_area_world2=float(areas[omitted_uv].sum()),
                seams=seams,registrations=registrations,attempts=attempts,quality=quality,
                original_XYZ_preserved=True,physical_acceptance=False,metric_failures_retained_as_diagnostic=True)
    return dict(world=v[parent],faces=outfaces,uv=uvs,parent=parent,source_face_rows=rows,
                omitted_face_rows=np.asarray(omitted,int),display=eligible,scientific=scientific,report=report)


def render_map(mapped,output,sample,frame,*,pitch_um=19.2,tile_size=1024,export_tifxyz=False,guard=lambda:None,binding=None,raster_workers=1,reuse=None):
    from .original_tifxyz import export_xyz
    from PIL import Image
    if type(raster_workers) is not int or not 1<=raster_workers<=4:raise ValueError('invalid raster_workers')
    if reuse:
        if export_tifxyz:raise ValueError('tile continuation supports PNG only')
        reuse.validate()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pitch=pitch_um/frame['voxel_um'];uv=mapped['uv'];faces=mapped['faces']
    totals={k:0 for k in ('CT_observed_pixels','CT_missing_pixels','ambiguous_pixels','black_unmapped_pixels')};tiles=[]
    if not len(faces):return dict(tiles=tiles,**totals,status='EMPTY_UV')
    low,high=uv.min(0),uv.max(0);wh=np.ceil((high-low)/pitch).astype(np.int64)+1
    if (wh<=0).any() or (wh>2**31).any():raise ValueError('patch canvas exceeds integer bounds')
    stopped=threading.Event()
    def check():
        if stopped.is_set():raise InterruptedError('raster cancelled')
        guard()
    def render_tile(position):
        row,col=position
        check();directory=f'tile-{row:09d}-{col:09d}';root=output/directory
        canvas=dict(width=min(tile_size,int(wh[0])-col),height=min(tile_size,int(wh[1])-row),pitch=pitch,u0=float(low[0]+col*pitch),v_top=float(high[1]-row*pitch))
        canvas.update(global_u0=float(low[0]),global_v_top=float(high[1]),column_offset=col,row_offset=row)
        if reuse:
            saved=reuse.get(directory,canvas,root)
            if saved is not None:return saved
        root.mkdir()
        # Two-node halo: orphan membership -> adjacent pruned nodes -> incident unsafe cells.
        r0,c0=max(0,row-2),max(0,col-2)
        r1,c1=min(int(wh[1]),row+canvas['height']+2),min(int(wh[0]),col+canvas['width']+2)
        expanded=dict(canvas,width=c1-c0,height=r1-r0,u0=float(low[0]+c0*pitch),v_top=float(high[1]-r0*pitch),column_offset=c0,row_offset=r0)
        crop=(slice(row-r0,row-r0+canvas['height']),slice(col-c0,col-c0+canvas['width']))
        xyz_report={}
        def capture(xyz,mask,owner):
            own=owner[crop]
            np.savez_compressed(root/'mapping.npz',xyz=xyz[crop],mask=mask[crop],source_face_rows=np.where(own>=0,mapped['source_face_rows'][np.maximum(own,0)],-1))
            if export_tifxyz:
                try:xyz_report.update(export_xyz(root/'tifxyz',xyz,mask,faces,uv,mapped['display'],expanded,frame,check,binding=binding,publish=(crop,canvas)))
                except ValueError as error:
                    if str(error)!='TIFXYZ contains no retained surface cells':raise
                    xyz_report.update(status='EMPTY_TIFXYZ',reason=str(error))
        image,mask,report=raster_sheet(mapped['world'],faces,uv,mapped['display'],sample,expanded,
            window=(0,255),guard=check,capture=capture,xyz_tolerance=.048/frame['voxel_um'])
        image,mask=image[crop],mask[crop];report['canvas']=canvas
        for value,key in enumerate(('black_unmapped_pixels','CT_missing_pixels','CT_observed_pixels','ambiguous_pixels')):report[key]=int((mask==value).sum())
        placement=mapped.get('report',{})
        report['disconnected_components_translated']=placement.get('disconnected_components_translated',len(placement.get('registrations',[]))>1)
        report['patch_repacking']=placement.get('patch_repacking',report['disconnected_components_translated'])
        Image.fromarray(image).save(root/'image.png');Image.fromarray(mask).save(root/'mask.png')
        return dict(directory=directory,canvas=canvas,rendering=report,tifxyz=xyz_report,
                    files={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    def publish(record):
        check();atomic_json(output/record['directory']/'TILE.json',record);tiles.append(record)
        for k in totals:totals[k]+=record['rendering'][k]
    positions=iter((row,col) for row in range(0,int(wh[1]),tile_size-1) for col in range(0,int(wh[0]),tile_size-1))
    if raster_workers==1:
        for position in positions:publish(render_tile(position))
    else:
        with ThreadPoolExecutor(max_workers=raster_workers) as pool:
            pending=deque()
            try:
                for _ in range(raster_workers):
                    position=next(positions,None)
                    if position is not None:pending.append(pool.submit(render_tile,position))
                while pending:
                    publish(pending.popleft().result())
                    position=next(positions,None)
                    if position is not None:pending.append(pool.submit(render_tile,position))
            except BaseException:
                stopped.set()
                for future in pending:future.cancel()
                raise
    if reuse:
        reuse.validate()
        if not set(reuse.receipts)<={t['directory'] for t in tiles}:raise ValueError('unexpected reused tile outside canvas')
    return dict(tiles=tiles,**totals,shape_yx=wh[::-1].tolist(),pitch_um=pitch_um,
                counts_include_tile_overlap=True,row_direction='decreasing UV v',window=[0,255],sampling='central trilinear',physical_acceptance=False)


@contextmanager
def open_mesh(path):
    from .range_surface import RangeReader
    from .surface_mesh import MeshReader
    path=Path(path)
    if (path/'RANGE.json').is_file():
        with RangeReader(path) as reader:
            receipt=reader.receipt;n=receipt['recipe']['samples_per_w']
            yield reader.mesh,dict(kind='continuous_range',range_receipt_sha256=reader.receipt_sha,
                recipe_sha256=receipt['recipe_sha256'],ticks=receipt['ticks'],samples_per_w=n,
                w_range=[tick/n for tick in receipt['ticks']],storage_slot_semantics='slot 0 is not W0')
            if file_sha(path/'RANGE.json')!=reader.receipt_sha:raise ValueError('range wrapper changed during export')
    else:
        with MeshReader(path) as reader:
            kind=reader.manifest['identity']['kind']
            if kind=='continuous_range':raise ValueError('patch continuous range requires its RANGE.json wrapper')
            yield reader,dict(kind=kind,storage_slot_semantics='each saved mesh level is exported independently')


def read_level(reader,level,*,memory_gib):
    from .patch_mapping import SourceKeys
    db=reader.db;nv=db.execute('SELECT COUNT(*) FROM vertices WHERE level=?',(level,)).fetchone()[0]
    nf=db.execute('SELECT COUNT(*) FROM faces WHERE level=?',(level,)).fetchone()[0]
    # shortcut: conservative in-memory graph admission, use a disk graph for larger meshes.
    estimate=(nv+nf)*2048
    if estimate>memory_gib*2**30:raise ValueError(f'patch geometry admission exceeds {memory_gib:g} GiB ({nv} vertices, {nf} faces)')
    vi=np.empty(nv,np.int64);v=np.empty((nv,3));fi=np.empty(nf,np.int64);f=np.empty((nf,3),np.int64)
    cursor=db.execute('SELECT id,x,y,z FROM vertices WHERE level=? ORDER BY id',(level,));offset=0
    while batch:=cursor.fetchmany(4096):
        count=len(batch)
        block=np.array(batch,dtype=[('id','i8'),('x','f8'),('y','f8'),('z','f8')])
        vi[offset:offset+count]=block['id']
        for axis,key in enumerate(('x','y','z')):v[offset:offset+count,axis]=block[key]
        offset+=count
    if offset!=nv:raise ValueError('source vertex count changed')
    cursor=db.execute('SELECT id,a,b,c FROM faces WHERE level=? ORDER BY id',(level,));offset=0
    while batch:=cursor.fetchmany(4096):
        block=np.asarray(batch,dtype=np.int64);count=len(batch);ids=block[:,1:];indices=np.searchsorted(vi,ids)
        if (indices>=nv).any() or not np.array_equal(vi[indices],ids):raise ValueError('unknown original mesh vertex')
        fi[offset:offset+count]=block[:,0];f[offset:offset+count]=indices;offset+=count
    if offset!=nf:raise ValueError('source face count changed')
    return v,f,vi,fi,SourceKeys(db,'vertices',vi),SourceKeys(db,'faces',fi)


def export_mesh(mesh_path,out,volume,request,*,export_tifxyz=False,stop=lambda:False):
    cfg=settings(request);native=check_flatboi(cfg['app_run'],cfg['native_recipe']);frame=cfg['output_frame'];out=Path(out)
    ct_reserve=ct_memory(volume.chunks,volume.dtype,cfg)
    source=Path(mesh_path).resolve();destination=out.resolve()
    if destination==source or destination in source.parents or source in destination.parents:raise ValueError('patch output must be separate from source mesh')
    if cfg['reuse_export']:
        previous=Path(cfg['reuse_export']).expanduser().resolve()
        if destination==previous or destination in previous.parents or previous in destination.parents:raise ValueError('patch output must be separate from immutable reuse export')
    with open_mesh(mesh_path) as (reader,source_identity):
        out.mkdir(exist_ok=False)
        identity=dict(source=source_identity,mesh_sha256=reader.manifest_sha,settings=cfg,ct=volume.identity,
                      native=native,implementation_sha256=file_sha(__file__),export_tifxyz=export_tifxyz)
        from importlib.metadata import version
        identity['kernel_sha256']={name:file_sha(Path(__file__).with_name(name+'.py')) for name in
            ('patch_pipeline','native_ct_sampling','native_patches','patch_mapping','surface_export','surface_atlas','original_recipe','original_pipeline','original_tifxyz','surface_mesh','range_surface','io')}
        identity['runtime']={name:version(name) for name in ('numpy','scipy','Pillow')}
        identity['memory_admission']=dict(raster_cache_runtime_bytes=raster_memory(cfg),ct_chunk_workspace_bytes=ct_reserve,
            geometry_budget_bytes=cfg['memory_gib']*2**30-raster_memory(cfg)-ct_reserve)
        from .native_patches import NativeReuse,ResourceGuard
        reuse=NativeReuse(cfg['reuse_export'],cfg['reuse_identity_sha256'],identity) if cfg['reuse_export'] else None
        if reuse:identity['native_reuse']=dict(export=str(reuse.root),identity_sha256=reuse.identity_sha256,inventory=reuse.inventory)
        resources=ResourceGuard(out,cfg['process_tree_gib'],cfg['disk_reserve_gib']) if max(cfg['workers'],cfg['raster_workers'])>1 else None
        cancelled=threading.Event()
        atomic_json(out/'IDENTITY.json',identity)
        def guard():
            if stop():raise InterruptedError('patch export paused; use a fresh output for retry')
            if resources:resources()
            required=max(64<<20,cfg['tile_size']**2*128*cfg['raster_workers'])+int(cfg['disk_reserve_gib']*2**30)
            if shutil.disk_usage(out).free<required:raise RuntimeError('insufficient disk for next patch tile')
        records=[]
        try:
            for level,value in reader.db.execute('SELECT id,value FROM levels ORDER BY id'):
                guard();v,f,vi,fi,vk,fk=read_level(reader,level,memory_gib=cfg['memory_gib']-(raster_memory(cfg)+ct_reserve)/2**30)
                target=out/f'level-{level:04d}';target.mkdir()
                world=frame_coordinates(v,frame,reader.metadata['voxel_um'])
                cancelled.clear()
                def native_guard():
                    if cancelled.is_set():raise InterruptedError('native lookahead cancelled')
                    guard()
                def flatten(v,f,path):
                    native_guard()
                    check_flatboi(cfg['app_run'],cfg['native_recipe'])
                    saved=reuse.get(level,v,f,path) if reuse else None
                    if saved is not None:uv,record=saved
                    else:
                        uv,record=flatten_native(v,f,cfg['app_run'],path,iters=200,timeout=cfg['timeout_seconds'],expected_sha256=LAUNCHER_SHA,
                            guard=native_guard,cpu_affinity=cfg['native_cpus'],threads=1 if cfg['workers']>1 else None)
                    check_flatboi(cfg['app_run'],cfg['native_recipe'])
                    return uv,record
                mapped=build_map(world,f,target/'patches',flatten=flatten,max_patch_faces=cfg['max_patch_faces'],
                                 max_depth=cfg['max_depth'],max_charts=cfg['max_charts'],world_unit_um=frame['voxel_um'],guard=guard,
                                 workers=cfg['workers'],cancel=cancelled.set)
                parent=mapped['parent'];rows=mapped['source_face_rows']
                from .patch_mapping import write_key_mapping
                embedded={}
                if vk.unicode_bytes(count=len(parent))+fk.unicode_bytes(count=len(rows))<=1<<20:
                    embedded=dict(source_vertex_keys=vk[parent],source_face_keys=fk[rows])
                np.savez_compressed(target/'map.npz',**{k:mapped[k] for k in ('world','faces','uv','display','scientific','omitted_face_rows')},
                    field_xyz=v[parent],source_vertex_ids=vi[parent],source_face_ids=fi[rows],**embedded,
                    omitted_source_face_ids=fi[mapped['omitted_face_rows']],
                    omitted_uv_source_face_ids=np.r_[fi[mapped['omitted_face_rows']],fi[rows[~mapped['display']]]])
                write_key_mapping(target,reader,vk,fk,guard)
                binding=dict(identity=identity,map_sha256=file_sha(target/'map.npz'),slot=level)
                del v,f,vi,fi,vk,fk,world,embedded,parent,rows
                rendered=render_map(mapped,target/'render',lambda xyz:volume.sample(xyz,frame),frame,
                    pitch_um=cfg['pitch_um'],tile_size=cfg['tile_size'],export_tifxyz=export_tifxyz,guard=guard,binding=binding,raster_workers=cfg['raster_workers'])
                record=dict(slot=level,semantics=source_identity,mapping=mapped['report'],rendering=rendered,
                            source_geometry=reader.metadata,output_frame=frame,map_sha256=binding['map_sha256'])
                if source_identity['kind']!='continuous_range':record['saved_level_value']=value
                atomic_json(target/'LEVEL.json',record);records.append(record)
            reader.validate();guard()
            if source_identity['kind']=='continuous_range' and file_sha(source/'RANGE.json')!=source_identity['range_receipt_sha256']:
                raise ValueError('range wrapper changed during export')
            result=dict(status='PATCHES_READY_DIAGNOSTIC' if any(r['rendering']['CT_observed_pixels'] for r in records) else 'PATCHES_EMPTY_DIAGNOSTIC',
                        identity=identity,levels=records,physical_acceptance=False,
                         files={str(p.relative_to(out)):file_sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
            result['native_execution']=dict(workers=cfg['workers'],reused_copied=reuse.count if reuse else 0,
                reused_committed=sum('reused_from' in a.get('native',{}) for r in records for a in r['mapping']['attempts']),
                peak_process_tree_rss_bytes=resources.peak if resources else None)
            atomic_json(out/'PATCHES.json',result)
            return result
        except Exception as error:
            atomic_json(out/'FAILED.json',dict(error=str(error),identity=identity));raise


class PatchCT:
    """Installed vc_reader transport and verified native or scalar interpolation.

    Native shared disk cache is lossless and bounded; chunk provenance lives on disk.
    """
    def __init__(self,source,output,cfg,*,voxel_um=None,level=0,resume=False):
        from .io import Volume
        from vc_reader.reader import open_zarr
        from vc_reader import _reader
        import sqlite3
        self.volume=Volume(source,voxel_um=voxel_um,level=level,cache=cfg['cache_dir'],cache_bytes=int(cfg['cache_gib']*2**30),
                           reserve_bytes=int(cfg['disk_reserve_gib']*2**30))
        self.chunks,self.dtype=self.volume.chunks,self.volume.dtype
        self._lock=threading.RLock();self._chunks=OrderedDict();self._cache_bytes=0
        self._cache_limit=cfg['cache_mib']*2**20
        self._cache_path=Path(cfg['cache_dir']).expanduser()
        self._disk_reserve=int(cfg['disk_reserve_gib']*2**30)
        try:self.chunk_workspace_bytes=ct_memory(self.chunks,self.dtype,cfg)
        except Exception:
            self.volume.close();raise
        self.db=None;self.native=None;self._ct_pool=None;self._sampler=None
        try:
            self.db=sqlite3.connect(Path(output)/'ct-chunks.sqlite',check_same_thread=False)
            self.db.execute('CREATE TABLE '+('IF NOT EXISTS ' if resume else '')+'chunks (key TEXT PRIMARY KEY,sha TEXT)')
            url=self.volume.source.rstrip('/')+('/'+self.volume.array_path if self.volume.array_path else '')
            self._local_source=None if self.volume.remote else Path(url)
            self._chunk_layers=[]
            if self.volume.remote:
                self.native=open_zarr(url,threads=1,connections=cfg['ct_workers'],auth=None,memory_cache=0,
                    cache_dir=str(Path(cfg['cache_dir']).expanduser()/'native'),cache_max_bytes=int(cfg['cache_gib']*2**30),
                    compress_cache=False,allow_lossy_cache=False)
                if self.native.meta.raw!=self.volume.metadata:raise ValueError('vc_reader metadata differs')
                self._chunk_layers=[layer for layer in self.native.cache_layers if layer.kind=='decoded' and not layer.allow_lossy]
                if not self._chunk_layers:raise ValueError('verified CT requires a lossless decoded cache')
            self._ct_pool=ThreadPoolExecutor(max_workers=cfg['ct_workers']) if cfg['ct_workers']>1 else None
            if cfg.get('ct_sampler','native-float64')=='native-float64':
                from .native_ct_sampling import NativeCTSampler
                self._sampler=NativeCTSampler(self.volume.shape,self.chunks,self.dtype,self._chunk)
            self.identity=dict(self.volume.identity,backend='vc_reader',extension_sha256=file_sha(_reader.__file__),
                               threads=1,memory_cache_bytes=self._cache_limit,memory_cache_owner='verified-python-chunks',
                                disk_cache_bytes=int(cfg['cache_gib']*2**30),chunk_workspace_bytes=self.chunk_workspace_bytes)
            self.identity['sampler']='vc_reader-float64-support-v1' if self._sampler else 'scalar'
            if self._sampler:
                self.identity['sampler_sha256']=file_sha(Path(__file__).with_name('native_ct_sampling.py'))
            libraries=Path(_reader.__file__).parent.parent/'lib'
            self.identity['reader_libraries_sha256']={p.name:file_sha(p) for p in sorted(libraries.glob('libvc_reader_native.*')) if p.is_file()}
        except Exception:
            self.close();raise

    def _load_chunk(self,index):
        """Read whole stored chunks; a fill-valued sample cannot prove presence."""
        name=self.volume.metadata.get('dimension_separator','.').join(map(str,index))
        def cached():
            for layer in self._chunk_layers:
                path=Path(layer.root)/name
                if path.is_file():return path,layer.stored.raw
                if Path(str(path)+'.missing').is_file():return None,layer.stored.raw
            return False,None
        if self._local_source is not None:
            path,metadata=self._local_source/name,self.volume.metadata
            if not path.exists():return None
        else:
            path,metadata=cached()
            if path is False:
                # A one-voxel read loads the complete chunk; flush makes its missing marker durable too.
                start=np.asarray(index)*self.volume.chunks
                self.native[tuple(slice(int(n),int(n)+1) for n in start)]
                self.native._array.flush()
                path,metadata=cached()
                if path is False:raise RuntimeError('CT prefetch publication failed')
            if path is None:return None
        with path.open('rb') as handle:raw=handle.read((256<<20)+1)
        if len(raw)>256<<20:raise ValueError('encoded CT chunk exceeds 256 MiB read budget')
        return self._decode_chunk(raw,metadata,cache=self._local_source is None)

    def _decode_chunk(self,raw,metadata,*,cache=False):
        if metadata.get('compressor'):
            from vc_reader.reader.writer import decompress_bytes
            if cache and metadata['compressor'].get('id')=='vc-delta3d':
                # shortcut: validate D3D1 v1/v2 headers; require native validation for newer wire formats.
                if not (len(raw)>=20 and raw[:4]==b'D3D1' and raw[4] in (1,2) and raw[6]<=1):
                    raise ValueError('CT cache payload is not verified lossless Delta3D')
            raw=decompress_bytes(raw,metadata['compressor'],math.prod(self.volume.chunks)*self.volume.dtype.itemsize)
        if metadata.get('filters'):
            from numcodecs import get_codec
            for spec in reversed(metadata['filters']):raw=get_codec(spec).decode(raw)
        if metadata['order']!=self.volume.metadata['order']:
            raw=np.frombuffer(raw,dtype=self.volume.dtype).reshape(self.volume.chunks,order=metadata['order']).tobytes(order=self.volume.metadata['order'])
        return raw

    def _chunk(self,index):
        import hashlib
        key=','.join(map(str,index));v=self.volume
        with self._lock:
            if key in self._chunks:
                value=self._chunks.pop(key);self._chunks[key]=value
                return value
            needed=math.prod(v.chunks)*v.dtype.itemsize
            if shutil.disk_usage(self._cache_path).free<self._disk_reserve+2*needed:
                raise RuntimeError('insufficient CT cache disk reserve')
            while self._chunks and self._cache_bytes+needed+512>self._cache_limit:
                _,old=self._chunks.popitem(last=False);self._cache_bytes-=512+(0 if old is None else old.nbytes)
            raw=self._load_chunk(index)
            if raw is not None and len(raw)!=needed:raise ValueError('invalid native CT chunk size')
            sha=None if raw is None else hashlib.sha256(raw).hexdigest()
            saved=self.db.execute('SELECT sha FROM chunks WHERE key=?',(key,)).fetchone()
            if saved is not None and saved[0]!=sha:raise ValueError('CT chunk changed during export: '+key)
            self.db.execute('INSERT OR IGNORE INTO chunks VALUES (?,?)',(key,sha))
            value=None if raw is None else np.frombuffer(raw,dtype=v.dtype).reshape(v.chunks,order=v.metadata['order'])
            if value is not None:value.flags.writeable=False
            size=512+(0 if value is None else value.nbytes)
            if size<=self._cache_limit:
                self._chunks[key]=value;self._cache_bytes+=size
            return value

    def _prefetch(self,indices):
        # Publish bounded batches to SSD; the scalar demand path still verifies every hash.
        with self._lock:
            missing=[index for index in indices if ','.join(map(str,index)) not in self._chunks]
            needed=math.prod(self.volume.chunks)*self.volume.dtype.itemsize
            if missing and shutil.disk_usage(self._cache_path).free<self._disk_reserve+2*needed*len(missing):
                raise RuntimeError('insufficient CT cache disk reserve')
            futures=[self._ct_pool.submit(self._load_chunk,index) for index in missing]
            try:
                for future in futures:future.result()
            finally:
                for future in futures:future.cancel()
                # Do not release shared cache access while a failed batch still writes.
                wait(futures)

    def sample(self,xyz,frame):
        from .original_recipe import sample_points
        v=self.volume
        grid=(xyz*frame['voxel_um']+np.asarray(frame['origin_xyz_um'])-v.origin_xyz)/v.spacing_xyz
        prefetch=self._prefetch if getattr(self,'_ct_pool',None) else None
        sampler=getattr(self,'_sampler',None)
        result=(sampler.sample(grid,prefetch=prefetch) if sampler else
                sample_points(grid,v.shape,v.chunks,self._chunk,prefetch=prefetch))
        with self._lock:self.db.commit()
        return result

    def validate_used(self,guard=lambda:None):
        """Revalidate authoritative support before accepting cached decoded chunks."""
        import hashlib
        self.volume.validate_used()
        self._chunks.clear();self._cache_bytes=0
        cursor=self.db.execute('SELECT key,sha FROM chunks ORDER BY key')
        while rows:=cursor.fetchmany(256):
            for key,expected in rows:
                guard();index=tuple(map(int,key.split(',')))
                if self.volume.remote:
                    name=self.volume.metadata.get('dimension_separator','.').join(map(str,index))
                    if self.volume.array_path:name=self.volume.array_path+'/'+name
                    raw=self.volume._chunk_bytes(name)
                    if raw is not None:raw=self._decode_chunk(raw,self.volume.metadata)
                    actual=None if raw is None else hashlib.sha256(raw).hexdigest()
                    if actual!=expected:raise ValueError('CT chunk changed on resume: '+key)
                self._chunk(index)
        self.db.commit()

    def close(self):
        if getattr(self,'_sampler',None):self._sampler.close();self._sampler=None
        if getattr(self,'_ct_pool',None):self._ct_pool.shutdown(wait=True,cancel_futures=True)
        self._chunks.clear();self._cache_bytes=0
        if self.db:self.db.close();self.db=None
        self.volume.close()
        if self.native:self.native.close();self.native=None


def export_existing(args):
    import signal
    if args.field_config:raise ValueError('patches requires --mesh-bundle; extract a mesh or join ranges first')
    if args.levels:raise ValueError('patch mesh identity already defines its levels; omit --levels')
    if args.resume:raise ValueError('patch export does not resume; use a fresh output (source meshes remain immutable)')
    cfg=settings(json.loads(Path(args.flattening_config).read_bytes()));check_flatboi(cfg['app_run'],cfg['native_recipe'])
    # Validate the wrapper before creating CT caches or output.
    with open_mesh(args.mesh_bundle):pass
    out=Path(args.output).expanduser().resolve();source=Path(args.mesh_bundle).expanduser().resolve()
    if out==source or out in source.parents or source in out.parents:raise ValueError('patch output must be separate from source mesh')
    if cfg['reuse_export']:
        previous=Path(cfg['reuse_export']).expanduser().resolve()
        if out==previous or out in previous.parents or previous in out.parents:raise ValueError('patch output must be separate from immutable reuse export')
    out.mkdir(exist_ok=False)
    volume=PatchCT(args.source,out,cfg,voxel_um=args.voxel_um,level=args.level)
    interrupted=threading.Event()
    previous_handler=signal.signal(signal.SIGTERM,lambda *_:interrupted.set())
    try:
        result=export_mesh(args.mesh_bundle,out/'patches',volume,cfg,export_tifxyz=args.export_tifxyz,stop=lambda:interrupted.is_set() or (out/'STOP').exists())
    finally:
        signal.signal(signal.SIGTERM,previous_handler);volume.close()
    result=dict(result,ct_ledger_sha256=file_sha(out/'ct-chunks.sqlite'))
    atomic_json(out/'PATCHES.json',result)
    return result
