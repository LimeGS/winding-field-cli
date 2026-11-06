"""Exact constant-W preparation: reference seam, disk cut and positive mean-value UV."""
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .native_global import cut_component, mean_value_map, _input, _topology_report
from .native_patches import ResourceGuard
from .patch_pipeline import read_level
from .sheet_atlas import _arrays, _json, _path, _verified
from .surface_mesh import MeshReader
from .tifxyz_guide import _cut_flags
from .util import atomic_json, file_sha


def centerline_points(centerline):
    cp=np.asarray(centerline,float)
    if cp.ndim!=2 or cp.shape[1]!=3 or len(cp)<2 or not np.isfinite(cp).all() or not (np.diff(cp[:,2])>0).all():
        raise ValueError('invalid centerline; require finite XYZ with strictly increasing Z')
    return cp


def reference_cut(v,f,centerline,guard=lambda:None):
    import igl
    cp=centerline_points(centerline);guard()
    original=_topology_report(v,f,_input(v,f,guard),guard)
    if original['euler_characteristic']==1:
        # A boundary-to-boundary seam can disconnect an already open disk.
        cut,topology=cut_component(v,f,guard)
        return cut,dict(reference_angle=0.,reference_path=[],angle_penalty=100.,
                        reference_seam='unnecessary: source is already a disk',topology=topology)
    centers=np.column_stack([np.interp(v[:,2],cp[:,2],cp[:,i]) for i in range(2)])
    delta=v[:,:2]-centers;angle=np.arctan2(delta[:,1],delta[:,0])
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    # Prefer +X at every height; seams use existing edges and never move XYZ.
    costs=length*(1+100*np.mean(np.abs(angle[edges]),axis=1)**2)
    graph=coo_matrix((np.r_[costs,costs],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    start=int(np.argmin(v[:,2]));end=int(np.argmax(v[:,2]));guard()
    distance,previous=dijkstra(graph,indices=start,return_predecessors=True)
    if start==end or not np.isfinite(distance[end]):raise ValueError('disconnected or zero-height reference path')
    path=[end]
    while path[-1]!=start:
        node=int(previous[path[-1]])
        if node<0 or len(path)>len(v):raise ValueError('invalid reference path')
        path.append(node)
    path=np.asarray(path,np.int64)
    first_v,first_f,first_parent=igl.cut_mesh(v,f.astype(np.int64),_cut_flags(f,[path],len(v)))
    np.testing.assert_array_equal(first_v,v[first_parent])
    np.testing.assert_array_equal(first_parent[first_f],f)
    cut,topology=cut_component(first_v,first_f,guard)
    cut['parent']=first_parent[cut['parent']]
    np.testing.assert_array_equal(cut['vertices'],v[cut['parent']])
    np.testing.assert_array_equal(cut['parent'][cut['faces']],f)
    return cut,dict(reference_angle=0.,reference_path=path.tolist(),angle_penalty=100.,
                    centerline_endpoints_clamped_for_caps=True,topology=topology)


def load_prepared(root):
    root=Path(root).absolute();record=_json(_path(root,'PREPARED.json'))
    required={'initial-map.npz','inputmesh.obj','MESH.json','disconnected-source.npz'}
    if record.get('status')!='CONSTANT_W_INITIALIZED' or not required<=record['files'].keys():
        raise ValueError('complete prepared sheet required')
    for name,sha in record['files'].items():_verified(root,name,sha)
    mesh=_json(root/'MESH.json');step=mesh['metadata']['sheet_step']
    level=next((x for x in mesh['levels'] if x['id']==record['level_index']),None)
    if not level or not math.isfinite(record['w']) or not math.isfinite(step) or step<=0 or not math.isclose(level['level']/step,record['w'],rel_tol=0,abs_tol=1e-10):
        raise ValueError('prepared normalized W differs from source')
    if mesh['database_sha256']!=record['source_database_sha256'] or mesh['metadata']!=record['source_geometry']:
        raise ValueError('prepared mesh identity differs')
    if 'centerline' in record:centerline_points(record['centerline'])
    return record,_arrays(root/'initial-map.npz')


def prepare(source,output,w,centerline):
    import igl
    cp=centerline_points(centerline)
    if not math.isfinite(w):raise ValueError('finite normalized W required')
    source=Path(source).resolve();out=Path(output).absolute();start=time.monotonic()
    if any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('output symlink refused')
    with MeshReader(source) as reader:
        step=reader.metadata['sheet_step']
        levels=[x for x in reader.manifest['levels'] if math.isclose(x['level']/step,w,rel_tol=0,abs_tol=1e-10)]
        if len(levels)!=1:raise ValueError('exactly one matching normalized W level required')
        level=levels[0]['id']
        out.mkdir(parents=True,exist_ok=False);resources=ResourceGuard(out,6.,8.)
        def guard():
            resources()
            if (out/'STOP').exists():raise InterruptedError('preparation stopped')
            if time.monotonic()-start>1800:raise TimeoutError('preparation exceeded 30 minutes')
        guard();v,f,vi,fi,*_=read_level(reader,level,memory_gib=6.)
        if not len(f):raise ValueError('empty constant-W mesh')
        _,labels=igl.facet_components(f)
        area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
        totals=np.bincount(labels,weights=area);selected=labels==int(totals.argmax())
        rows=np.flatnonzero(selected);other=np.flatnonzero(~selected)
        ids,inverse=np.unique(f[rows],return_inverse=True);world,faces=v[ids],inverse.reshape(-1,3)
        oi,of=np.unique(f[other],return_inverse=True)
        np.savez_compressed(out/'disconnected-source.npz',world=v[oi],faces=of.reshape(-1,3),source_vertex_ids=vi[oi],source_face_ids=fi[other])
        print('CUT_START',w,len(world),len(faces),flush=True)
        cut,cut_report=reference_cut(world,faces,cp,guard)
        print('INIT_START',w,len(cut['vertices']),flush=True)
        uv,initialization=mean_value_map(cut,guard)
        mapped=dict(world=cut['vertices'],faces=cut['faces'],uv=uv,source_vertex_ids=vi[ids[cut['parent']]],
                    source_face_ids=fi[rows],source_face_rows=rows,source_vertex_rows=ids[cut['parent']])
        np.testing.assert_array_equal(mapped['world'],v[mapped['source_vertex_rows']])
        np.testing.assert_array_equal(mapped['source_vertex_rows'][mapped['faces']],f[rows])
        np.savez_compressed(out/'initial-map.npz',**mapped)
        with (out/'inputmesh.obj').open('x') as stream:
            for values,fmt in ((mapped['world'],'v %.17g %.17g %.17g'),(uv,'vt %.17g %.17g')):
                for first in range(0,len(values),100000):guard();np.savetxt(stream,values[first:first+100000],fmt=fmt)
            for first in range(0,len(faces),100000):
                guard();np.savetxt(stream,np.repeat(mapped['faces'][first:first+100000]+1,2,axis=1),fmt='f %d/%d %d/%d %d/%d')
        reader.validate();atomic_json(out/'MESH.json',reader.manifest)
        report=dict(status='CONSTANT_W_INITIALIZED',w=w,raw_level=levels[0]['level'],level_index=level,
                    source=str(source),source_manifest_sha256=reader.manifest_sha,source_database_sha256=reader.manifest['database_sha256'],
                    source_geometry=reader.metadata,centerline=cp.tolist(),cut=cut_report,initialization=initialization,
                    source_faces=len(f),principal_faces=len(rows),disconnected_faces=len(other),
                    principal_area_fraction=float(area[selected].sum()/area.sum()),world_bounds=[world.min(0).tolist(),world.max(0).tolist()],
                    files={p.name:file_sha(p) for p in out.iterdir() if p.is_file()},seconds=time.monotonic()-start,
                    peak_tree_rss_bytes=resources.peak,physical_acceptance=False)
        guard();atomic_json(out/'PREPARED.json',report)
        return report
