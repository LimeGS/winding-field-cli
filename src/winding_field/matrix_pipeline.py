"""Resumable matrix exports with exact source keys and verified tiled CT."""
import fcntl
from importlib.metadata import version
from pathlib import Path
import signal
import shutil
import sqlite3
import uuid

import numpy as np

from . import patch_pipeline as p
from .flattening_matrix import build_map, settings
from .native_patches import ResourceGuard
from .original_pipeline import frame_coordinates
from .patch_mapping import write_key_mapping
from .sheet_atlas import _arrays, _json, _path, _verified
from .util import atomic_json, file_sha


def verify_level(root,identity,ct):
    record=_json(_path(root,'LEVEL.json'))
    if record['identity']!=identity:raise ValueError('matrix level identity changed')
    for name,sha in record['files'].items():_verified(root,name,sha)
    with sqlite3.connect((root/'ct-used.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as previous:
        for key,sha in previous.execute('SELECT key,sha FROM chunks'):
            if ct.db.execute('SELECT sha FROM chunks WHERE key=?',(key,)).fetchone()!=(sha,):
                raise ValueError('matrix CT ledger lost committed provenance: '+key)
    return record


def export_existing(args):
    from .workflow import implementation_sha
    cfg=settings(_json(Path(args.flattening_config)),args.flattening_solver,args.scope)
    if args.mesh_bundle and args.levels:raise ValueError('mesh bundle already binds levels; omit --levels')
    if args.field_config and not args.levels:raise ValueError('--levels required with --field-config')
    out=Path(args.output).expanduser().absolute()
    if any(q.is_symlink() for q in (out,*out.parents)) or (out/'.lock').is_symlink():raise ValueError('matrix output symlink refused')
    out=out.resolve()
    mesh=Path(args.mesh_bundle).expanduser().resolve() if args.mesh_bundle else out/'meshes'
    if args.mesh_bundle and (out==mesh or out in mesh.parents or mesh in out.parents):raise ValueError('matrix output must be separate from source mesh')
    descriptor=_json(Path(args.field_config)) if args.field_config else None
    native=None
    if cfg['flattening_solver']=='slim':
        from .flatboi_checkpoint import require_cgroup
        native=p.check_flatboi(cfg['app_run'],cfg['native_recipe']);require_cgroup()
    # Validate mesh provenance before creating output or caches.
    mesh_sha=None
    if args.mesh_bundle:
        with p.open_mesh(mesh) as (reader,_):mesh_sha=reader.manifest_sha
    identity=dict(schema='flattening-matrix-v1',settings=cfg,mesh=str(mesh),mesh_sha256=mesh_sha,field=descriptor,
                  levels=args.levels,ct_source=args.source,ct_level=args.level,ct_voxel_um=args.voxel_um,
                  export_tifxyz=args.export_tifxyz,native=native,implementation_sha256=implementation_sha(),
                  runtime={name:version(name) for name in ('numpy','scipy','libigl','Pillow')})
    out.mkdir(exist_ok=args.resume)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(q.is_symlink() for q in out.rglob('*')):raise ValueError('matrix output symlink refused')
        if args.resume:
            if _json(_path(out,'EXPORT.json'))!=identity:raise ValueError('matrix resume identity changed')
        else:atomic_json(out/'EXPORT.json',identity)
        resources=ResourceGuard(out,cfg['process_tree_gib'],cfg['disk_reserve_gib']);stopped=False
        def stop(*_):
            nonlocal stopped
            stopped=True
        def guard():
            resources()
            if stopped or (out/'STOP').exists():raise InterruptedError('matrix export stopped; resume the same output')
        handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        ct=None
        try:
            guard()
            if descriptor is not None:
                from .surface_sources import FieldSource
                from .surface_mesh import extract_mesh
                with FieldSource(descriptor['field'],descriptor['geometry']) as source:
                    result=extract_mesh(source,mesh,args.levels,resume=mesh.exists(),stop=guard,
                                        reserve_bytes=int(cfg['disk_reserve_gib']*2**30))
                    if result['status']!='MESH_READY':return result
            saved=_json(_path(out,'MATRIX.json')) if (out/'MATRIX.json').exists() else None
            if saved:
                if saved['identity']!=identity:raise ValueError('matrix publication identity changed')
                _verified(out,'ct/ct-chunks.sqlite',saved['ct_ledger_sha256'])
                for name,sha in saved['levels'].items():_verified(out,name+'/LEVEL.json',sha)
            ctroot=out/'ct'
            if ctroot.is_symlink():raise ValueError('matrix CT output symlink refused')
            ctroot.mkdir(exist_ok=True)
            for name in ('ct-chunks.sqlite','ct-chunks.sqlite-journal'):
                if (ctroot/name).is_symlink():raise ValueError('matrix CT ledger symlink refused')
            if (out/'CT-IDENTITY.json').exists() and not (ctroot/'ct-chunks.sqlite').is_file():
                raise ValueError('matrix CT ledger missing on resume')
            ct=p.PatchCT(args.source,ctroot,cfg,voxel_um=args.voxel_um,level=args.level,resume=args.resume)
            if (out/'CT-IDENTITY.json').exists():
                if _json(_path(out,'CT-IDENTITY.json'))!=ct.identity:raise ValueError('matrix CT identity changed')
            else:atomic_json(out/'CT-IDENTITY.json',ct.identity)
            if args.resume:ct.validate_used(guard)
            records={};observed=0
            with p.open_mesh(mesh) as (reader,source_identity):
                if mesh_sha is not None and reader.manifest_sha!=mesh_sha:raise ValueError('matrix source mesh changed')
                frame=cfg['output_frame']
                geometry_gib=cfg['memory_gib']-(p.raster_memory(cfg)+p.ct_memory(ct.chunks,ct.dtype,cfg))/2**30
                for level,value in reader.db.execute('SELECT id,value FROM levels ORDER BY id'):
                    guard();target=out/f'level-{level:04d}'
                    if target.is_symlink():raise ValueError('matrix level symlink refused')
                    binding=dict(export=identity,mesh_sha256=reader.manifest_sha,source=source_identity,slot=level,saved_level_value=value,ct=ct.identity)
                    if (target/'LEVEL.json').exists():
                        record=verify_level(target,binding,ct)
                    else:
                        target.mkdir(exist_ok=True)
                        v,f,vi,fi,vk,fk=p.read_level(reader,level,memory_gib=geometry_gib)
                        world=frame_coordinates(v,frame,reader.metadata['voxel_um'])
                        mapped=build_map(world,f,target/'flattening',cfg,guard=guard)
                        parent=mapped['parent'];rows=mapped['source_face_rows']
                        if (target/'MAPPING.json').exists():
                            provenance=_json(_path(target,'MAPPING.json'))
                            _verified(target,'map.npz',provenance['map_sha256']);_verified(target,'keys.sqlite',provenance['keys_sha256'])
                        else:
                            # Incomplete key projection is retained, not mistaken for committed provenance.
                            if (target/'keys.sqlite').exists():(target/'keys.sqlite').rename(target/('keys-interrupted-'+uuid.uuid4().hex+'.sqlite'))
                            np.savez_compressed(target/'map.npz',**{k:mapped[k] for k in ('world','faces','uv','display','scientific','source_face_rows','domain_ids')},
                                                field_xyz=v[parent],source_vertex_ids=vi[parent],source_face_ids=fi[rows])
                            write_key_mapping(target,reader,vk,fk,guard)
                        del v,f,vi,fi,vk,fk,world,parent,rows
                        render=target/'render'
                        if render.exists():render.rename(target/('render-interrupted-'+uuid.uuid4().hex))
                        rendered=p.render_map(mapped,render,lambda xyz:ct.sample(xyz,frame),frame,
                                              pitch_um=cfg['pitch_um'],tile_size=cfg['tile_size'],export_tifxyz=args.export_tifxyz,
                                              guard=guard,binding=binding,raster_workers=cfg['raster_workers'])
                        reader.validate();guard()
                        ct.db.commit();shutil.copyfile(ctroot/'ct-chunks.sqlite',target/'ct-used.sqlite')
                        files={str(q.relative_to(target)):file_sha(q) for q in sorted(target.rglob('*')) if q.is_file() and q.name!='.lock'}
                        record=dict(identity=binding,mapping=mapped['report'],rendering=rendered,files=files,
                                    source_geometry=reader.metadata,output_frame=frame,physical_acceptance=False)
                        atomic_json(target/'LEVEL.json',record)
                        del mapped
                    records[target.name]=file_sha(target/'LEVEL.json');observed+=record['rendering']['CT_observed_pixels']
                reader.validate()
            guard();ct.close();ct=None
            result=dict(status='MATRIX_READY_DIAGNOSTIC' if observed else 'MATRIX_EMPTY_DIAGNOSTIC',identity=identity,
                        levels=records,ct_ledger_sha256=file_sha(ctroot/'ct-chunks.sqlite'),physical_acceptance=False)
            if saved and saved!=result:raise ValueError('matrix publication changed on resume')
            if not saved:atomic_json(out/'MATRIX.json',result)
            return result
        finally:
            if ct is not None:ct.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)
