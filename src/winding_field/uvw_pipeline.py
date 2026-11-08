"""Receipt-bound aligned UVW raster publication; original mapping stays read-only."""
from contextlib import closing
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image

from .original_pipeline import settings as original_settings,verify_level,sample_ct,verify_missing_ct
from .tifxyz_field import PhysicalField,read_metadata
from .surface_sources import FieldSource
from .io import Volume
from .util import atomic_json,file_sha,digest
from .uvw import Reference,sample_w_gradient
from .uvw_workers import CAUSES,GeometryWorkers
BATCH=1024


def load_reference(path,levels,canvas_samples=512):
    """Verify independent original ledger and complete inventory, then convert units."""
    path=Path(path).expanduser()
    if path.is_symlink():raise ValueError('symlink reference')
    path=path.resolve(strict=True);root=path.parent
    record,receipt=read_metadata(path/'LEVEL.json')
    with closing(sqlite3.connect((root/'publications.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        ledger=dict(db.execute('SELECT name,receipt_sha FROM levels'))
    original,_=read_metadata(root/'ORIGINAL.json')
    if ledger.get(path.name)!=receipt['sha256'] or original.get('level_receipt_sha256')!=ledger:
        raise ValueError('reference receipt/ledger binding differs')
    for name in record['files']:
        if any(p.is_symlink() for p in (path/name,*list((path/name).parents)[:len(Path(name).parts)])):
            raise ValueError('symlink reference product')
    verify_level(path,record['identity'])
    frame=original_settings(record['identity']['settings'])['output_frame'];unit=frame['voxel_um']
    if type(canvas_samples) is not int or not 2<=canvas_samples<=1024:raise ValueError('canvas-samples must be 2..1024')
    canvas=record['rendering']['canvas']
    if (any(type(canvas[k]) is not int or not 2<=canvas[k]<=4096 for k in ('width','height'))
        or not all(math.isfinite(canvas[k]) for k in ('pitch','u0','v_top')) or canvas['pitch']<=0):
        raise ValueError('invalid reference canvas')
    span=(max(canvas['width'],canvas['height'])-1)*canvas['pitch']*unit
    pitch=span/(canvas_samples-1)
    grid=dict(width=canvas_samples,height=canvas_samples,pitch_um=pitch,
        u0_um=canvas['u0']*unit,v_top_um=canvas['v_top']*unit)
    # Bound decompression before numpy allocates the original map.
    with zipfile.ZipFile(path/'map.npz') as archive:
        if sum(p.file_size for p in archive.infolist())>512<<20:raise ValueError('reference map exceeds 512 MiB admission')
    with np.load(path/'map.npz',allow_pickle=False) as mapped:
        required={'world','faces','uv','display','scientific','source_vertex_ids','source_face_ids'}
        if set(mapped.files)!=required or len(mapped.files)!=len(required):raise ValueError('reference map inventory differs')
        ref=Reference(mapped['world']*unit+np.asarray(frame['origin_xyz_um']),mapped['faces'],mapped['uv']*unit,
            record['identity']['level'],display=mapped['display'],band=(min(levels),max(levels)))
        if mapped['scientific'].shape!=(len(ref.faces),) or mapped['source_vertex_ids'].shape!=(len(ref.world),) or mapped['source_face_ids'].shape!=(len(ref.faces),):
            raise ValueError('reference map source inventory differs')
    metadata=dict(path=str(path),level_receipt_sha256=receipt['sha256'],map_sha256=record['files']['map.npz'],
        original_receipt_sha256=file_sha(root/'ORIGINAL.json'),identity=record['identity'],
        canvas=grid,original_canvas=canvas,output_frame=frame,reference_w=ref.w,
        coordinates='XYZ physical micrometres; UV physical micrometres; W field units',
        reference_faces=len(ref.faces),excluded_reference_faces=int((~ref.face_valid).sum()))
    return ref,metadata


def _verify_files(root,record):
    expected={'image.png','geometry-mask.png','ct-mask.png','cause-mask.png','cell-mask.png','quality.json',
        'tifxyz/x.tif','tifxyz/y.tif','tifxyz/z.tif','tifxyz/meta.json','tifxyz/valid-mask.png'}
    if set(record['files'])!=expected:raise ValueError('UVW product inventory differs')
    for name,sha in record['files'].items():
        path=root/name
        if path.is_symlink() or path.parent.is_symlink() or file_sha(path)!=sha:raise ValueError('UVW product changed: '+name)


def _checkpoint(out,db,key,compute,volume):
    row=db.execute('SELECT path,sha,missing FROM batches WHERE key=?',(key,)).fetchone()
    if row:
        path=out/row[0]
        if path.parent!=out/'.checkpoints' or path.is_symlink() or file_sha(path)!=row[1]:raise ValueError('UVW checkpoint product changed')
        verify_missing_ct(volume,dict(unavailable_ct_chunks=json.loads(row[2])))
        with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files},set(json.loads(row[2]))
    if compute is None:return None,set()
    data,missing=compute();directory=out/'.checkpoints';directory.mkdir(exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='batch-',suffix='.npz',dir=directory)
    with os.fdopen(fd,'wb') as stream:
        np.savez_compressed(stream,**data);stream.flush();os.fsync(stream.fileno())
    path=Path(name)
    with db:db.execute('INSERT INTO batches VALUES(?,?,?,?)',(key,str(path.relative_to(out)),file_sha(path),json.dumps(sorted(missing))))
    return data,missing


def _sampled_cells(xyz,base,valid,field,*,pitch_um=1.):
    """Two raster triangles and ODE W derivative per cell, not a global proof."""
    shape=valid.shape;cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    determinant=np.full(cells.shape,np.nan);ratio=np.full(cells.shape,np.nan)
    velocities=[]
    for surface in (base,xyz):
        points=surface[:-1,:-1].reshape(-1,3);grad=np.full_like(points,np.nan)
        for start in range(0,len(points),BATCH):
            _,g,_=sample_w_gradient(field,points[start:start+BATCH]);grad[start:start+BATCH]=g
        grad=grad.reshape((*cells.shape,3));norm2=np.sum(grad*grad,axis=-1)
        velocities.append(grad/np.where(norm2>0,norm2,1)[...,None])
    reference_velocity,velocity=velocities
    area=np.cross(xyz[:-1,1:]-xyz[:-1,:-1],xyz[:-1,:-1]-xyz[1:,:-1])
    other=np.cross(xyz[1:,1:]-xyz[1:,:-1],xyz[:-1,1:]-xyz[1:,1:])
    original=np.cross(base[:-1,1:]-base[:-1,:-1],base[:-1,:-1]-base[1:,:-1])
    original_other=np.cross(base[1:,1:]-base[1:,:-1],base[:-1,1:]-base[1:,1:])
    determinant[:]=np.sum(area*velocity,axis=-1)
    expected=np.sum(original*reference_velocity,axis=-1);second=np.sum(other*velocity,axis=-1)
    expected_second=np.sum(original_other*reference_velocity,axis=-1)
    ratio[:]=np.linalg.norm(area,axis=-1)/np.maximum(np.linalg.norm(original,axis=-1),1e-30)
    ratio_second=np.linalg.norm(other,axis=-1)/np.maximum(np.linalg.norm(original_other,axis=-1),1e-30)
    passed=(np.isfinite(determinant)&np.isfinite(second)&(abs(determinant)>1e-8*abs(expected))
        &(abs(second)>1e-8*abs(expected_second))&(determinant*expected>0)&(second*expected_second>0))
    failed=cells&~passed
    bad=np.zeros(shape,bool)
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):bad[dy:dy+shape[0]-1,dx:dx+shape[1]-1]|=failed
    retained=cells&passed
    stats=dict(sampled_cells=int(cells.sum()),rejected_cells=int(failed.sum()),
        min_abs_jacobian_um_per_w=float(min(abs(determinant[retained]).min(),abs(second[retained]).min())/pitch_um**2) if retained.any() else None,
        area_ratio_min=float(min(ratio[retained].min(),ratio_second[retained].min())) if retained.any() else None,
        area_ratio_max=float(max(ratio[retained].max(),ratio_second[retained].max())) if retained.any() else None,
        scope='two triangles per raster cell and sampled ODE W derivative; not global injectivity')
    valid=valid&~bad
    retained&=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    return valid,retained,bad,stats


def _render_level(out,db,index,w,ref,metadata,field,volume,identity,guard,workers):
    name=f'level-{index:04d}';target=out/name
    row=db.execute('SELECT sha,pending FROM levels WHERE name=?',(name,)).fetchone()
    if row:
        pending=out/row[1]
        if pending.parent!=out:raise ValueError('invalid pending publication')
        path=target if target.exists() else pending
        if path.is_symlink() or file_sha(path/'LEVEL.json')!=row[0]:raise ValueError('UVW level receipt changed')
        record,_=read_metadata(path/'LEVEL.json')
        if record['identity_sha256']!=digest(identity) or record['w']!=w:raise ValueError('UVW level identity changed')
        _verify_files(path,record);verify_missing_ct(volume,record)
        if path!=target:path.rename(target)
        return record
    if target.exists():raise ValueError('unbound UVW publication')
    canvas=metadata['canvas'];n=canvas['width'];total=n*n;start_time=time.monotonic()
    arrays=dict(xyz=np.full((total,3),np.nan),encoded_xyz=np.full((total,3),np.nan),base=np.full((total,3),np.nan),reference_valid=np.zeros(total,bool),
        geometry=np.zeros(total,bool),ct=np.zeros(total,bool),value=np.full(total,np.nan),cause=np.ones(total,np.uint8),
        residual=np.full(total,np.nan),roundtrip=np.full(total,np.nan),coordinate_rounding=np.full(total,np.nan))
    missing=set();metrics=[]
    timings=dict(geometry_wall_seconds=0.,ct_seconds=0.,checkpoint_seconds=0.)
    # One bounded geometry window feeds the existing chunk-sorted positive-weight sampler.
    for window in range(0,total,BATCH*workers.workers):
        guard();ready={};todo=[];jobs=[];tick=time.perf_counter()
        for start in range(window,min(window+BATCH*workers.workers,total),BATCH):
            data,absent=_checkpoint(out,db,f'{index}-{start}',None,volume)
            if data is not None:ready[start]=data;missing|=absent;continue
            reference_data,_=_checkpoint(out,db,f'reference-{start}',None,volume)
            linear=np.arange(start,min(start+BATCH,total))
            uv=np.column_stack((canvas['u0_um']+(linear%n)*canvas['pitch_um'],canvas['v_top_um']-(linear//n)*canvas['pitch_um']))
            todo.append(start);jobs.append(dict(uv=uv,w=w,reference_data=reference_data))
        timings['checkpoint_seconds']+=time.perf_counter()-tick;tick=time.perf_counter()
        generated=list(workers.map(jobs))
        timings['geometry_wall_seconds']+=time.perf_counter()-tick
        if generated:
            guard();tick=time.perf_counter()
            for start,(data,measurement) in zip(todo,generated):
                reference_data=dict(base=data['base'],reference_valid=data['reference_valid'],cause=data['reference_cause'],
                    reference_face=data['reference_face'],reference_barycentric=data['reference_barycentric'])
                _checkpoint(out,db,f'reference-{start}',lambda:(reference_data,set()),volume)
            timings['checkpoint_seconds']+=time.perf_counter()-tick
            selected=[np.flatnonzero(data['geometry']) for data,_ in generated]
            points=np.concatenate([data['xyz'][ids] for (data,_),ids in zip(generated,selected)])
            absent=set();tick=time.perf_counter()
            values,observed=sample_ct(volume,points,missing=absent,stop=guard.stop) if len(points) else (np.empty(0),np.empty(0,bool))
            timings['ct_seconds']+=time.perf_counter()-tick;cursor=0;tick=time.perf_counter()
            for start,(data,measurement),ids in zip(todo,generated,selected):
                count=len(ids);data['value']=np.full(len(data['base']),np.nan);data['ct']=np.zeros(len(data['base']),bool)
                data['value'][ids]=values[cursor:cursor+count];data['ct'][ids]=observed[cursor:cursor+count];cursor+=count
                data['_metrics']=np.array(json.dumps(measurement,sort_keys=True))
                data,_=_checkpoint(out,db,f'{index}-{start}',lambda:(data,absent),volume)
                ready[start]=data
            missing|=absent;timings['checkpoint_seconds']+=time.perf_counter()-tick
        for start,data in ready.items():
            end=min(start+BATCH,total)
            for key in arrays:arrays[key][start:end]=data[key]
            metrics.append(json.loads(str(data['_metrics'])))
    for key in arrays:arrays[key]=arrays[key].reshape((n,n,3) if key in ('xyz','encoded_xyz','base') else (n,n))
    tick=time.perf_counter()
    geometry,cells,bad,jacobian=_sampled_cells(arrays['xyz'],arrays['base'],arrays['geometry'],field,pitch_um=canvas['pitch_um'])
    timings['jacobian_seconds']=time.perf_counter()-tick
    timings['recorded_geometry_worker_seconds']=sum(m['geometry_seconds'] for m in metrics)
    timings['recorded_reference_worker_seconds']=sum(m['reference_seconds'] for m in metrics)
    worker_stats={}
    for measurement in metrics:
        pid=str(measurement['pid']);entry=worker_stats.setdefault(pid,dict(peak_rss_bytes=0,initialization_seconds=0.,field={}))
        entry['peak_rss_bytes']=max(entry['peak_rss_bytes'],measurement['peak_rss_bytes'])
        entry['initialization_seconds']=max(entry['initialization_seconds'],measurement.get('initialization_seconds',0.))
        for key,value in measurement['field'].items():entry['field'][key]=entry['field'].get(key,0)+value
    arrays['cause'][bad]=CAUSES['sampled_jacobian'];ct=arrays['ct']&geometry
    xyz=arrays['xyz'];xyz[~geometry]=np.nan
    quality=dict(w=w,canvas_pixels=total,reference_valid_pixels=int(arrays['reference_valid'].sum()),
        reference_excluded_pixels=int((~arrays['reference_valid']).sum()),geometry_pixels=int(geometry.sum()),ct_pixels=int(ct.sum()),
        geometry_fraction_canvas=float(geometry.mean()),ct_fraction_canvas=float(ct.mean()),
        geometry_fraction_reference=float(geometry.sum()/arrays['reference_valid'].sum()) if arrays['reference_valid'].any() else 0.,
        causes={key:int((arrays['cause']==code).sum()) for key,code in CAUSES.items()},sampled_jacobian=jacobian,
        normal_residual_max_um=float(arrays['residual'][geometry].max()) if geometry.any() else None,
        roundtrip_max_um=float(arrays['roundtrip'][geometry].max()) if geometry.any() else None,
        coordinate_rounding_max_um=float(arrays['coordinate_rounding'][geometry].max()) if geometry.any() else None,
        elapsed_seconds=time.monotonic()-start_time,timings=timings,geometry_workers=worker_stats,
        physical_acceptance=False,global_invertibility_proven=False)
    publication_started=time.perf_counter()
    pending=Path(tempfile.mkdtemp(prefix='.'+name+'-',dir=out));(pending/'tifxyz').mkdir()
    image=np.zeros((n,n),np.uint8);image[ct]=np.clip(np.rint(arrays['value'][ct]*255/180),0,255).astype(np.uint8)
    for filename,data in [('image.png',image),('geometry-mask.png',geometry.astype(np.uint8)*255),
        ('ct-mask.png',ct.astype(np.uint8)*255),('cause-mask.png',arrays['cause']),('cell-mask.png',cells.astype(np.uint8)*255),
        ('tifxyz/valid-mask.png',geometry.astype(np.uint8)*255)]:Image.fromarray(data).save(pending/filename)
    import tifffile
    # Publish the exact encoding validated by workers; another affine conversion can add rounding.
    frame=metadata['output_frame'];coordinates=arrays['encoded_xyz'];coordinates[~geometry]=np.nan
    for axis,label in enumerate('xyz'):
        path=pending/'tifxyz'/f'{label}.tif';encoded=coordinates[:,:,axis].astype(np.float64)
        tifffile.imwrite(path,encoded,metadata=None)
        readback=tifffile.imread(path)
        if readback.dtype!=np.float64 or not np.array_equal(readback,encoded,equal_nan=True):raise ValueError('TIFXYZ readback differs')
    atomic_json(pending/'tifxyz/meta.json',dict(schema='aligned-uvw-tifxyz-v2',coordinate_dtype='float64',w=w,coordinate_units='output-frame voxels',
        output_frame=frame,canvas=canvas,row_direction='decreasing V',invalid_coordinates='NaN',
        validity_mask='valid-mask.png',mask_values={'invalid':0,'valid':255},ct_validity='../ct-mask.png',
        cell_validity='../cell-mask.png',causes=CAUSES,reference=metadata['map_sha256'],identity_sha256=digest(identity),
        geometry_includes_ct_missing=True,physical_acceptance=False,global_invertibility_proven=False))
    timings['publication_prepare_seconds']=time.perf_counter()-publication_started;tick=time.perf_counter()
    guard();field.validate();volume.validate_used(fresh=False)
    timings['validation_seconds']=time.perf_counter()-tick;quality['ct_cumulative']=volume.stats()
    atomic_json(pending/'quality.json',quality)
    files={str(p.relative_to(pending)):file_sha(p) for p in pending.rglob('*') if p.is_file()}
    record=dict(w=w,identity_sha256=digest(identity),files=files,quality=quality,unavailable_ct_chunks=sorted(missing))
    atomic_json(pending/'LEVEL.json',record)
    for path in pending.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    guard();field.validate()
    with db:db.execute('INSERT INTO levels VALUES(?,?,?)',(name,file_sha(pending/'LEVEL.json'),pending.name))
    pending.rename(target)
    return record


def export_existing(args,*,stop=None):
    """CLI/programmatic entry; repeat identical arguments with resume=True."""
    from .workflow import implementation_sha
    from importlib.metadata import version
    import tifffile
    started=time.perf_counter();levels=args.levels
    worker_count=getattr(args,'workers',1);cache_mib=getattr(args,'cache_mib',256)
    if type(worker_count) is not int or not 1<=worker_count<=8:raise ValueError('workers must be 1..8')
    if type(cache_mib) is not int or not 0<=cache_mib<=8192:raise ValueError('cache-mib must be 0..8192')
    ct_budget=cache_mib*(1<<20)//2;region_budget=ct_budget//(1 if worker_count==1 else worker_count+1)
    performance=dict(workers=worker_count,cache_mib=cache_mib,ct_decoded_cache_bytes=ct_budget,field_region_cache_bytes_per_reader=region_budget)
    if (not isinstance(levels,(list,tuple)) or not 1<=len(levels)<=16 or not all(type(w) in (int,float) and math.isfinite(w) for w in levels)
        or len(set(levels))!=len(levels)):raise ValueError('1..16 distinct finite W levels required')
    if type(args.canvas_samples) is not int or not 2<=args.canvas_samples<=1024:raise ValueError('canvas-samples must be 2..1024')
    descriptor,descriptor_record=read_metadata(Path(args.field_config))
    if not all(isinstance(descriptor.get(key),dict) for key in ('field','geometry')):
        raise ValueError('descriptor requires field and geometry objects')
    out=Path(args.output).expanduser().resolve();reference_root=Path(args.reference_sheet).expanduser().resolve().parent
    if out==reference_root or reference_root in out.parents or out in reference_root.parents:raise ValueError('output must be separate from reference')
    if not str(args.source).startswith(('http://','https://','s3://')):
        source_path=Path(args.source).expanduser().resolve()
        if out==source_path or out in source_path.parents or source_path in out.parents:raise ValueError('output overlaps CT source')
    with PhysicalField(FieldSource(descriptor['field'],descriptor['geometry']),region_cache_bytes=region_budget) as field:
        field.validate()
        for item in (descriptor['field'],descriptor['field'].get('mask',{})):
            if item.get('path') and (out==Path(item['path']) or out in Path(item['path']).parents):raise ValueError('output overlaps field source')
        reference,metadata=load_reference(args.reference_sheet,levels,args.canvas_samples)
        with closing(Volume(args.source,voxel_um=args.voxel_um,level=args.level,decoded_cache_bytes=ct_budget)) as volume:
            if metadata['identity']['ct']!=volume.identity:raise ValueError('CT identity differs from reference physical frame/source')
            identity=dict(schema='aligned-uvw-v2',coordinate_dtype='float64',field=field.identity,descriptor_sha256=descriptor_record['sha256'],
                reference=metadata,levels=list(levels),ct=volume.identity,transport=asdict(reference.settings),
                implementation_sha256=implementation_sha(),batch_size=BATCH,performance=performance,
                runtime={name:version(name) for name in ('numpy','scipy','Pillow','tifffile')})
            if out.exists() and not args.resume:raise ValueError('output exists; use --resume')
            if args.resume:
                saved,_=read_metadata(out/'IDENTITY.json')
                if saved!=identity:raise ValueError('UVW resume identity changed')
                if any(not (out/name).is_file() or (out/name).is_symlink() for name in ('publications.sqlite','source.sqlite')):
                    raise ValueError('UVW resume ledger missing or unsafe')
            out.mkdir(parents=True,exist_ok=args.resume)
            with (out/'.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if not args.resume:
                    atomic_json(out/'IDENTITY.json',identity);atomic_json(out/'reference.json',metadata)
                elif read_metadata(out/'reference.json')[0]!=metadata:raise ValueError('reference product changed')
                volume.bind_ledger(out/'source.sqlite');volume.validate_used()
                with closing(sqlite3.connect(out/'publications.sqlite')) as db:
                    db.execute('CREATE TABLE IF NOT EXISTS batches(key TEXT PRIMARY KEY,path TEXT,sha TEXT,missing TEXT)')
                    db.execute('CREATE TABLE IF NOT EXISTS levels(name TEXT PRIMARY KEY,sha TEXT,pending TEXT)')
                    db.execute('CREATE TABLE IF NOT EXISTS aggregate(sha TEXT,pending TEXT)');db.commit()
                    # Verify every completed checkpoint, including reference batches, before continuing.
                    for key, in db.execute('SELECT key FROM batches').fetchall():_checkpoint(out,db,key,None,volume)
                    def guard():
                        if guard.stop():raise InterruptedError('UVW export paused')
                        if read_metadata(Path(args.field_config))[1]!=descriptor_record:raise ValueError('field descriptor changed')
                        field.validate()
                    guard.stop=lambda:(stop and stop()) or (out/'STOP').exists()
                    records=[]
                    try:
                        with GeometryWorkers(descriptor,args.reference_sheet,levels,args.canvas_samples,workers=worker_count,
                            region_cache_bytes=region_budget,stop=guard.stop,inline=(reference,field,metadata['output_frame']),
                            reference_identity=digest(metadata)) as workers:
                            for i,w in enumerate(levels):records.append(_render_level(out,db,i,w,reference,metadata,field,volume,identity,guard,workers))
                        # Revalidate the reference's independent receipt and inventory before final publication.
                        _,current=load_reference(args.reference_sheet,levels,args.canvas_samples)
                        if current!=metadata:raise ValueError('reference changed during export')
                        field.validate();volume.validate_used(fresh=False)
                    except InterruptedError:return dict(status='PAUSED',completed_levels=len(records),physical_acceptance=False)
                    hashes=dict(db.execute('SELECT name,sha FROM levels'))
                    existing=db.execute('SELECT sha,pending FROM aggregate').fetchone()
                    if existing:
                        pending=out/existing[1]
                        if pending.parent!=out or pending.is_symlink():raise ValueError('invalid aggregate pending path')
                        receipt=out/'UVW.json' if (out/'UVW.json').exists() else pending/'UVW.json'
                        if receipt.is_symlink() or file_sha(receipt)!=existing[0]:raise ValueError('UVW aggregate receipt changed')
                        result,_=read_metadata(receipt)
                        if result['level_receipt_sha256']!=hashes:raise ValueError('UVW publication ledger changed')
                        quality=out/'quality.json' if (out/'quality.json').exists() else pending/'quality.json'
                        if quality.is_symlink() or file_sha(quality)!=result['quality_sha256']:raise ValueError('UVW quality product changed')
                        if quality.parent!=out:quality.rename(out/'quality.json')
                        if receipt.parent!=out:receipt.rename(out/'UVW.json')
                        return result
                    if (out/'UVW.json').exists() or (out/'quality.json').exists():raise ValueError('unbound UVW aggregate product')
                    common_geometry=np.ones((args.canvas_samples,args.canvas_samples),bool);common_ct=common_geometry.copy()
                    for i in range(len(levels)):
                        with Image.open(out/f'level-{i:04d}/geometry-mask.png') as im:common_geometry&=np.asarray(im)>0
                        with Image.open(out/f'level-{i:04d}/ct-mask.png') as im:common_ct&=np.asarray(im)>0
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                    quality=dict(common_geometry_pixels=int(common_geometry.sum()),common_ct_pixels=int(common_ct.sum()),
                        canvas_pixels=common_geometry.size,reference_excluded_pixels=records[0]['quality']['reference_excluded_pixels'],
                        process_peak_rss_bytes=rss,levels=[r['quality'] for r in records],performance=performance,
                        invocation_elapsed_seconds=time.perf_counter()-started,ct_cumulative=volume.stats(),coordinator_field=dict(field.stats))
                    status='UVW_READY_DIAGNOSTIC' if any(r['quality']['geometry_pixels'] for r in records) else 'UVW_EMPTY_DIAGNOSTIC'
                    result=dict(status=status,identity=identity,level_receipt_sha256=hashes,quality=quality,
                        physical_acceptance=False,global_invertibility_proven=False)
                    pending=Path(tempfile.mkdtemp(prefix='.aggregate-',dir=out))
                    atomic_json(pending/'quality.json',quality);result['quality_sha256']=file_sha(pending/'quality.json')
                    atomic_json(pending/'UVW.json',result)
                    with db:db.execute('INSERT INTO aggregate VALUES(?,?)',(file_sha(pending/'UVW.json'),pending.name))
                    (pending/'quality.json').rename(out/'quality.json');(pending/'UVW.json').rename(out/'UVW.json')
                    return result
