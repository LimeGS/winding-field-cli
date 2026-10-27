"""Bounded conformal charts, with every mesh face and 3D correspondence retained."""

import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from ._core.surface_mesh import MeshReader
from .util import atomic_bytes,atomic_json,digest,file_sha


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def overlap_pairs(uv,faces):
    triangles=uv[faces]; lows=triangles.min(1); highs=triangles.max(1)
    count=0
    for i,triangle in enumerate(triangles):
        candidates=np.flatnonzero(np.all(np.minimum(highs[i],highs)-np.maximum(lows[i],lows)>1e-9,axis=1))
        for j in candidates[candidates>i]:
            polygon=list(triangles[j])
            for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
                clipped=[]
                for p,q in zip(polygon,polygon[1:]+polygon[:1]):
                    dp=float(cross2(b-a,p-a)); dq=float(cross2(b-a,q-a))
                    if dp>=0:
                        clipped.append(p)
                    if (dp>=0)!=(dq>=0):
                        clipped.append(p+(q-p)*dp/(dp-dq))
                polygon=clipped
                if not polygon:
                    break
            if len(polygon)>=3:
                polygon=np.asarray(polygon)
                area=abs(np.sum(cross2(polygon,np.roll(polygon,-1,axis=0))))*.5
                scale=abs(float(cross2(triangle[1]-triangle[0],triangle[2]-triangle[0])))
                if area>max(scale*1e-7,1e-10):
                    count+=1
    return count


def parameterize(vertices,faces):
    vertices=np.asarray(vertices,dtype=float); faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3
            or faces.dtype.kind not in 'iu' or not len(faces) or not np.isfinite(vertices).all()
            or np.any(faces<0) or np.any(faces>=len(vertices))):
        raise ValueError('invalid mesh arrays')
    triangles=vertices[faces]
    e1=triangles[:,1]-triangles[:,0]; e2=triangles[:,2]-triangles[:,0]
    length=np.linalg.norm(e1,axis=1)
    area2=np.linalg.norm(np.cross(e1,e2),axis=1)
    if np.any(area2<=1e-12) or np.any(length<=1e-12):
        raise ValueError('degenerate mesh triangle')
    x=(e1*e2).sum(1)/length; h=area2/length
    gx=np.stack((-h,h,np.zeros_like(h)),axis=1)/area2[:,None]
    gy=np.stack((x-length,-x,length),axis=1)/area2[:,None]
    factor=np.sqrt(area2/2)[:,None]; gx*=factor; gy*=factor
    n=len(vertices); m=len(faces); rows=np.repeat(np.arange(m),3)
    A=coo_matrix((np.concatenate((gx.ravel(),-gy.ravel(),gy.ravel(),gx.ravel())),
             (np.concatenate((rows,rows,rows+m,rows+m)),
              np.concatenate((faces.ravel(),faces.ravel()+n,faces.ravel(),faces.ravel()+n)))),shape=(2*m,2*n)).tocsr()
    first=int(np.argmax(np.linalg.norm(vertices-vertices.mean(0),axis=1)))
    second=int(np.argmax(np.linalg.norm(vertices-vertices[first],axis=1)))
    fixed=np.array([first,second,first+n,second+n])
    values=np.array([0.,np.linalg.norm(vertices[second]-vertices[first]),0.,0.])
    free=np.ones(2*n,bool); free[fixed]=False
    solution=lsqr(A[:,free],-A[:,fixed]@values,atol=1e-10,btol=1e-10,iter_lim=5000)
    if solution[1] not in (1,2):
        raise ValueError('chart parametrization did not converge')
    packed=np.empty(2*n); packed[fixed]=values; packed[free]=solution[0]
    uv=np.stack((packed[:n],packed[n:]),axis=-1)
    signed=cross2(uv[faces[:,1]]-uv[faces[:,0]],uv[faces[:,2]]-uv[faces[:,0]])
    if np.sum(signed)<0:
        uv[:,1]*=-1; signed=-signed
    if np.any(signed<=1e-12):
        raise ValueError('flipped or collapsed chart triangles')
    scale=np.sqrt(area2.sum()/signed.sum()); uv*=scale
    uv-=uv.min(0)
    edge=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    ratio=np.linalg.norm(uv[edge[:,0]]-uv[edge[:,1]],axis=1)/np.linalg.norm(vertices[edge[:,0]]-vertices[edge[:,1]],axis=1)
    overlap=overlap_pairs(uv,faces)
    if overlap or ratio.min()<.25 or ratio.max()>4:
        raise ValueError('overlapping or excessively distorted chart')
    return uv,dict(method='LSCM',flipped_faces=0,overlap_pairs=overlap,
                   edge_stretch_min=float(ratio.min()),edge_stretch_max=float(ratio.max()))


def components(faces):
    count=int(faces.max())+1
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(count,count))
    _,labels=connected_components(graph,directed=False)
    for label in np.unique(labels[faces[:,0]]):
        yield np.flatnonzero(labels[faces[:,0]]==label)


def chart_pieces(vertices,faces,face_ids,vertex_ids):
    for group in components(faces):
        f=faces[group]; used,inverse=np.unique(f,return_inverse=True)
        v=vertices[used]; f=inverse.reshape(-1,3)
        try:
            uv,quality=parameterize(v,f)
        except ValueError:
            if len(group)==1:
                raise
            mid=len(group)//2
            for half in (group[:mid],group[mid:]):
                yield from chart_pieces(vertices,faces[half],face_ids[half],vertex_ids)
        else:
            yield v,f,uv,face_ids[group],vertex_ids[used],quality


def write_obj(path,vertices,faces,uv,flat=False):
    lines=['# XYZ and UV in micrometres; linked source IDs in matching NPZ']
    if flat:
        vertices=np.column_stack((uv,np.zeros(len(uv))))
    lines.extend('v '+' '.join(f'{x:.12g}' for x in p) for p in vertices)
    lines.extend('vt '+' '.join(f'{x:.12g}' for x in p) for p in uv)
    lines.extend('f '+' '.join(f'{i+1}/{i+1}' for i in f) for f in faces)
    atomic_bytes(path,('\n'.join(lines)+'\n').encode())


def chart_exports(root,name,vertices,faces,uv,volume):
    names=[name+'.obj',name+'-flat.obj']
    write_obj(root/names[0],vertices,faces,uv)
    write_obj(root/names[1],vertices,faces,uv,flat=True)
    if volume is not None:
        from .preview import preview
        names.append(name+'.png')
        preview(volume,vertices,faces,uv,root/names[-1])
    return {path:file_sha(root/path) for path in names}


def flatten(run,config,volume=None,stop=None):
    run=Path(run); root=run/'flat'; root.mkdir(exist_ok=True)
    limit=config.get('chart_faces',1024)
    if type(limit) is not int or not 1<=limit<=8192:
        raise ValueError('chart_faces must be 1..8192')
    identity=dict(mesh=file_sha(run/'meshes/MESH.json'),chart_faces=limit,algorithm='lscm-linked-charts-v2',
                  ct_previews=volume is not None)
    if (root/'IDENTITY.json').exists() and json.loads((root/'IDENTITY.json').read_bytes())!=identity:
        raise ValueError('flatten identity changed')
    atomic_json(root/'IDENTITY.json',identity)
    reader=MeshReader(run/'meshes')
    db=sqlite3.connect(root/'atlas.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS charts (name TEXT PRIMARY KEY, sha TEXT, faces INTEGER, quality TEXT, exports TEXT);'
                    'CREATE TABLE IF NOT EXISTS covered (face INTEGER PRIMARY KEY, chart TEXT, local INTEGER);')
    try:
        for name,sha,exports in db.execute('SELECT name,sha,exports FROM charts'):
            if file_sha(root/(name+'.npz'))!=sha:
                raise ValueError('flat chart changed')
            expected=json.loads(exports)
            if any(not (root/path).is_file() or file_sha(root/path)!=sha for path,sha in expected.items()):
                with np.load(root/(name+'.npz'),allow_pickle=False) as chart:
                    actual=chart_exports(root,name,chart['vertices'],chart['faces'],chart['uv'],volume)
                if actual!=expected:
                    raise ValueError('regenerated chart exports differ from committed payload')
            if stop and stop():
                return dict(status='PAUSED')
        cursor=db.execute('SELECT coalesce(max(face),0) FROM covered').fetchone()[0]
        while True:
            rows=reader.db.execute('SELECT id,a,b,c FROM faces WHERE id>? ORDER BY id LIMIT ?',(cursor,limit)).fetchall()
            if not rows:
                break
            rows=np.asarray(rows,dtype=np.int64); vertex_ids,faces=np.unique(rows[:,1:],return_inverse=True)
            vertices=np.array([reader.db.execute('SELECT x,y,z FROM vertices WHERE id=?',(int(i),)).fetchone() for i in vertex_ids])
            records=[]
            for v,f,uv,face_ids,source_ids,quality in chart_pieces(vertices,faces.reshape(-1,3),rows[:,0],vertex_ids):
                name=f'chart-{int(face_ids.min()):09d}'
                buffer=io.BytesIO(); np.savez_compressed(buffer,vertices=v,faces=f,uv=uv,
                                                       source_face_ids=face_ids,source_vertex_ids=source_ids)
                atomic_bytes(root/(name+'.npz'),buffer.getvalue())
                exports=chart_exports(root,name,v,f,uv,volume)
                records.append((name,file_sha(root/(name+'.npz')),face_ids,quality,exports))
            with db:
                for name,sha,ids,quality,exports in records:
                    db.execute('INSERT INTO charts VALUES (?,?,?,?,?)',(name,sha,len(ids),json.dumps(quality),json.dumps(exports)))
                    db.executemany('INSERT INTO covered VALUES (?,?,?)',[(int(fid),name,i) for i,fid in enumerate(ids)])
            cursor=int(rows[-1,0])
            if stop and stop():
                return dict(status='PAUSED')
        count=db.execute('SELECT count(*) FROM covered').fetchone()[0]
        total=reader.db.execute('SELECT count(*) FROM faces').fetchone()[0]
        if not total or count!=total:
            raise ValueError('incomplete flattened face coverage')
        receipt=dict(status='FLAT_READY',charts=db.execute('SELECT count(*) FROM charts').fetchone()[0],
                     covered_faces=count,total_faces=total,coverage_fraction=count/total,identity=identity,
                     atlas_sha256=file_sha(root/'atlas.sqlite'),physical_acceptance=False,
                     continuity='linked atlas; cuts retain common source vertex IDs')
        atomic_json(run/'FLAT.json',receipt)
        return receipt
    finally:
        db.close(); reader.close()
