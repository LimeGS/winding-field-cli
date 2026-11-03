"""Source-bound original global mapping, CT raster and optional XYZ publication."""
import gc
from contextlib import closing
import json
import math
import os
from pathlib import Path
import tempfile
import sys
import sqlite3

import numpy as np
from PIL import Image

from ._core.surface_mesh import MeshReader
from .original_recipe import make_map,raster_sheet,sample_points
from .util import atomic_json,digest,file_sha


def settings(request):
    if not isinstance(request,dict):
        raise ValueError('--original-config requires an output_frame object')
    unknown=set(request)-{'output_frame','pitch','memory_gib'}
    if unknown:raise ValueError('unknown original settings: '+str(sorted(unknown)))
    frame=request.get('output_frame',{})
    if not isinstance(frame,dict):raise ValueError('original output_frame must be an object')
    origin=np.asarray(frame.get('origin_xyz_um'),float)
    unit=frame.get('voxel_um')
    if (not isinstance(frame.get('id'),str) or not frame['id'] or origin.shape!=(3,)
            or not np.isfinite(origin).all() or type(unit) not in (int,float)
            or not math.isfinite(unit) or unit<=0):
        raise ValueError('original output frame requires id, finite origin_xyz_um and positive voxel_um')
    result=dict(output_frame=frame,pitch=request.get('pitch'),memory_gib=request.get('memory_gib',4.))
    for key in ('pitch','memory_gib'):
        value=result[key]
        if value is None and key=='pitch':continue
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('invalid original '+key)
    return result


def frame_coordinates(xyz,frame,world_unit_um=1.):
    return xyz*(world_unit_um/frame['voxel_um'])-np.asarray(frame['origin_xyz_um'])/frame['voxel_um']


def sample_ct(volume,xyz,frame=None,missing=None):
    from numcodecs import get_codec
    def chunk(index):
        name=volume.metadata.get('dimension_separator','.').join(map(str,index))
        key=volume.array_path+'/'+name if volume.array_path else name
        try:raw=volume._chunk_bytes(key)
        except FileNotFoundError:
            if missing is None:raise
            missing.add(key);return None
        if raw is None:return None
        if volume.metadata.get('compressor'):raw=get_codec(volume.metadata['compressor']).decode(raw)
        for spec in reversed(volume.metadata.get('filters') or []):raw=get_codec(spec).decode(raw)
        array=np.frombuffer(raw,dtype=volume.dtype)
        if array.size!=math.prod(volume.chunks):raise ValueError('corrupt CT chunk: '+key)
        return array.reshape(volume.chunks,order=volume.metadata['order'])
    grid=(xyz-volume.origin_xyz)/volume.spacing_xyz if frame is None else (
        xyz*(frame['voxel_um']/np.asarray(volume.spacing_xyz))+(np.asarray(frame['origin_xyz_um'])-volume.origin_xyz)/volume.spacing_xyz)
    return sample_points(grid,volume.shape,volume.chunks,chunk)


def admit(vertices,faces,memory_gib):
    # shortcut: conservative solver estimate, replace with measured sparse-fill admission for very large domains.
    required=(2<<30)+vertices*2048+faces*1024
    if required>memory_gib*2**30:
        raise RuntimeError(f'original memory admission: estimated {required/2**30:.2f} GiB exceeds {memory_gib:g} GiB')
    return required


def available_memory():
    if sys.platform.startswith('linux'):
        from .tifxyz_runtime import memory_report
        report=memory_report()
        return report['effective_headroom_bytes'] if report['known'] else None
    if sys.platform=='darwin':
        import re,subprocess
        text=subprocess.check_output(['vm_stat'],text=True)
        page=int(re.search(r'page size of (\d+) bytes',text)[1])
        return page*sum(int(re.search(r'^'+name+r':\s+(\d+)',text,re.M)[1]) for name in ('Pages free','Pages inactive'))
    return None


def component(reader,level,budget):
    db=reader.db
    counts=db.execute('SELECT COUNT(*) FROM vertices WHERE level=?',(level,)).fetchone()[0]
    groups=db.execute('SELECT root,faces,area FROM components WHERE level=? ORDER BY area DESC,root',(level,)).fetchall()
    if not groups:raise ValueError('level has no supported surface')
    root,nfaces,area=groups[0]
    required=admit(counts,sum(g[1] for g in groups),budget)
    available=available_memory()
    if available is None or available<required+(512<<20):raise RuntimeError('original memory headroom unavailable')
    vertices=np.asarray(db.execute('SELECT id,x,y,z FROM vertices WHERE level=? ORDER BY id',(level,)).fetchall())
    rows=[]
    for identity,a,b,c,parent in db.execute('SELECT id,a,b,c,parent FROM faces WHERE level=? ORDER BY id',(level,)):
        # Finished bundles compress parents; resolve chains too, without mutating the source.
        while parent!=root:
            next_parent=db.execute('SELECT parent FROM faces WHERE id=?',(parent,)).fetchone()[0]
            if next_parent==parent:break
            if next_parent>=parent:raise ValueError('invalid mesh component parent')
            parent=next_parent
        if parent==root:rows.append((identity,a,b,c))
    rows=np.asarray(rows,np.int64)
    if len(rows)!=nfaces:raise ValueError('mesh component inventory differs')
    used,inverse=np.unique(rows[:,1:],return_inverse=True)
    indices=np.searchsorted(vertices[:,0].astype(np.int64),used)
    if not np.array_equal(vertices[indices,0],used):raise ValueError('unknown mesh vertex')
    return vertices[indices,1:],inverse.reshape(-1,3),used,rows[:,0],dict(
        main_component=root,total_source_faces=sum(g[1] for g in groups),total_source_components=len(groups),
        total_source_area_um2=sum(g[2] for g in groups),source_main_component_area_um2=area,
        unplaced_other_component_area_um2=sum(g[2] for g in groups)-area)


def verify_level(path,identity):
    record=json.loads((path/'LEVEL.json').read_bytes())
    if record['identity']!=identity:raise ValueError('original resume identity changed')
    required={'map.npz','image.png','mask.png'}
    if identity.get('export_tifxyz'):
        required|={'tifxyz/'+name for name in ('x.tif','y.tif','z.tif','meta.json','valid-mask.png','VERIFIED.json')}
    if set(record['files'])!=required:raise ValueError('original product inventory differs')
    for name,sha in record['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid product path')
        if file_sha(path/name)!=sha:raise ValueError('original product changed: '+name)
    return record


def verify_missing_ct(volume,record):
    for key in record.get('unavailable_ct_chunks',[]):
        try:raw=volume._chunk_bytes(key)
        except FileNotFoundError:continue
        if raw is not None:raise ValueError('CT availability changed; use a new output')


def export_mesh(mesh_path,out,volume,request,*,export_tifxyz=False,stop=lambda:False,reserve_bytes=0):
    from ._core.field_runtime import check_resources
    from .original_tifxyz import export_xyz
    from .workflow import implementation_sha
    cfg=settings(request);frame=cfg['output_frame'];out=Path(out)
    try:
        import igl
    except ImportError as error:
        raise ValueError("original flattening requires pip install 'winding-field-cli[tifxyz]'") from error
    out.mkdir(exist_ok=True)
    def guard():
        if stop():raise InterruptedError('original export paused')
        check_resources(out,reserve_bytes=reserve_bytes+(256<<20))
    records=[]
    with closing(sqlite3.connect(out/'publications.sqlite')) as journal,MeshReader(mesh_path) as reader:
        from importlib.metadata import version
        journal.execute('CREATE TABLE IF NOT EXISTS levels(name TEXT PRIMARY KEY,receipt_sha TEXT,pending TEXT)')
        identity=dict(mesh_sha256=reader.manifest_sha,settings=cfg,ct=volume.identity,
                      export_tifxyz=export_tifxyz,implementation_sha256=implementation_sha(),
                      runtime={name:version(name) for name in ('numpy','scipy','libigl','Pillow')})
        identity_path=out/'IDENTITY.json'
        if identity_path.exists() and json.loads(identity_path.read_bytes())!=identity:
            raise ValueError('original resume identity changed')
        atomic_json(identity_path,identity)
        aggregate=out/'ORIGINAL.json'
        if aggregate.exists():
            saved=json.loads(aggregate.read_bytes())
            if saved.get('level_receipt_sha256')!=dict(journal.execute('SELECT name,receipt_sha FROM levels')):
                raise ValueError('original publication ledger changed')
        for level,value in reader.db.execute('SELECT id,value FROM levels ORDER BY id'):
            target=out/f'level-{level:04d}'
            binding=dict(identity,level=value,level_id=level)
            committed=journal.execute('SELECT receipt_sha,pending FROM levels WHERE name=?',(target.name,)).fetchone()
            if committed:
                if not target.exists():
                    pending=out/committed[1]
                    if pending.parent!=out:raise ValueError('invalid pending path')
                    if file_sha(pending/'LEVEL.json')!=committed[0]:raise ValueError('original receipt changed')
                    verify_level(pending,binding);pending.rename(target)
                if file_sha(target/'LEVEL.json')!=committed[0]:raise ValueError('original receipt changed')
                record=verify_level(target,binding)
                verify_missing_ct(volume,record)
                records.append(record);continue
            if target.exists():raise ValueError('unbound original level publication')
            guard()
            try:
                v,f,vertex_ids,face_ids,coverage=component(reader,level,cfg['memory_gib'])
                world_unit_um=reader.metadata['voxel_um']
                v=frame_coordinates(v,frame,world_unit_um)
                for key in ('total_source_area_um2','source_main_component_area_um2','unplaced_other_component_area_um2'):
                    coverage[key]*=world_unit_um**2
                mapped=make_map(v,f);guard()
            except Exception as error:
                failed=Path(tempfile.mkdtemp(prefix=f'.failed-level-{level:04d}-',dir=out))
                atomic_json(failed/'FAILED.json',dict(identity=binding,error=str(error)))
                raise
            bounds=np.asarray(mapped['report']['uv_bounds']);center=bounds.mean(0)
            pitch=max(cfg['pitch'] or 0.,float(np.max(bounds[1]-bounds[0]))/4095)
            canvas=dict(width=4096,height=4096,pitch=pitch,u0=float(center[0]-4095*pitch/2),v_top=float(center[1]+4095*pitch/2))
            pending=Path(tempfile.mkdtemp(prefix=f'.level-{level:04d}-',dir=out))
            try:
                np.savez_compressed(pending/'map.npz',**{k:mapped[k] for k in ('world','faces','uv','display','scientific')},
                    source_vertex_ids=vertex_ids[mapped['parent']],source_face_ids=face_ids[mapped['source_face_rows']])
                exported={};missing=set()
                map_binding=dict(level=value,map_sha256=file_sha(pending/'map.npz'),mesh_sha256=reader.manifest_sha)
                def capture(xyz,mask,owner):
                    exported.update(export_xyz(pending/'tifxyz',xyz,mask,mapped['faces'],mapped['uv'],mapped['display'],canvas,frame,guard,binding=map_binding))
                def sample(xyz):
                    return sample_ct(volume,xyz,frame,missing)
                image,mask,rendering=raster_sheet(mapped['world'],mapped['faces'],mapped['uv'],mapped['display'],sample,canvas,
                    guard=guard,capture=capture if export_tifxyz else None)
                Image.fromarray(image).save(pending/'image.png');Image.fromarray(mask).save(pending/'mask.png')
                reader.validate();guard()
                files={str(p.relative_to(pending)):file_sha(p) for p in sorted(pending.rglob('*')) if p.is_file()}
                record=dict(identity=binding,files=files,mapping=mapped['report'],coverage=coverage,rendering=rendering,
                            tifxyz=exported,unavailable_ct_chunks=sorted(missing),physical_acceptance=False)
                atomic_json(pending/'LEVEL.json',record)
                for path in pending.rglob('*'):
                    if path.is_file():
                        with path.open('rb') as stream:os.fsync(stream.fileno())
                with journal:
                    journal.execute('INSERT INTO levels VALUES(?,?,?)',(target.name,file_sha(pending/'LEVEL.json'),pending.name))
                pending.rename(target);records.append(record)
            except BaseException as error:
                atomic_json(pending/'FAILED.json',dict(error=str(error)));raise
            del v,f,mapped,image,mask;gc.collect()
        receipt_hashes=dict(journal.execute('SELECT name,receipt_sha FROM levels'))
    result=dict(status='ORIGINAL_READY_DIAGNOSTIC',levels=records,level_receipt_sha256=receipt_hashes,physical_acceptance=False)
    atomic_json(out/'ORIGINAL.json',result)
    return result


def run_original(run,config,volume,stop=None):
    run=Path(run)
    try:
        result=export_mesh(run/'meshes',run/'original',volume,config.get('original_request'),
            export_tifxyz=config.get('export_tifxyz',False),stop=stop or (lambda:False),reserve_bytes=config.get('reserve_bytes',0))
    except InterruptedError:
        return dict(status='PAUSED')
    atomic_json(run/'ORIGINAL.json',result)
    return result


def export_existing(args):
    import fcntl
    from ._core.surface_sources import FieldSource
    from ._core.surface_mesh import extract_mesh
    from .io import Volume
    from .workflow import implementation_sha
    out=Path(args.output).expanduser().resolve()
    request=settings(json.loads(Path(args.original_config).read_bytes()))
    descriptor=json.loads(Path(args.field_config).read_bytes()) if args.field_config else None
    if descriptor is not None and not args.levels:raise ValueError('--levels required with --field-config')
    if args.mesh_bundle and args.levels:raise ValueError('mesh bundle already binds its levels; omit --levels')
    mesh=Path(args.mesh_bundle).expanduser().resolve() if args.mesh_bundle else out/'meshes'
    if args.mesh_bundle and (out==mesh or mesh in out.parents or out in mesh.parents):
        raise ValueError('output must be separate from source mesh bundle')
    identity=dict(settings=request,field=descriptor,mesh=str(mesh),levels=args.levels,
                  source=args.source,level=args.level,voxel_um=args.voxel_um,export_tifxyz=args.export_tifxyz,
                  implementation_sha256=implementation_sha())
    out.mkdir(exist_ok=args.resume)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if json.loads((out/'EXPORT.json').read_bytes())!=identity:raise ValueError('original resume identity changed')
        else:atomic_json(out/'EXPORT.json',identity)
        if descriptor is not None:
            with FieldSource(descriptor['field'],descriptor['geometry']) as source:
                result=extract_mesh(source,mesh,args.levels,resume=mesh.exists())
                if result['status']!='MESH_READY':return result
        volume=Volume(args.source,voxel_um=args.voxel_um,level=args.level)
        try:
            volume.bind_ledger(out/'source.sqlite')
            if args.resume:volume.validate_used()
            return export_mesh(mesh,out/'original',volume,request,export_tifxyz=args.export_tifxyz,
                               stop=lambda:(out/'STOP').exists())
        finally:volume.close()
