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
from .uvw import Reference,TransportSettings,reference_points,transport,xyz_to_uvw,sample_w_gradient

CAUSES={'valid':0,'reference_domain':1,'reference_ambiguous':2,'invalid_field':3,
        'zero_gradient':4,'unsupported_path':5,'nonconvergence':6,'roundtrip':7,
        'sampled_jacobian':8,'invalid_target':9,'coordinate_precision':10}
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


def _render_level(out,db,index,w,ref,metadata,field,volume,identity,guard):
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
    arrays=dict(xyz=np.full((total,3),np.nan),base=np.full((total,3),np.nan),reference_valid=np.zeros(total,bool),
        geometry=np.zeros(total,bool),ct=np.zeros(total,bool),value=np.full(total,np.nan),cause=np.ones(total,np.uint8),
        residual=np.full(total,np.nan),roundtrip=np.full(total,np.nan),coordinate_rounding=np.full(total,np.nan))
    missing=set()
    for start in range(0,total,BATCH):
        guard();end=min(start+BATCH,total);linear=np.arange(start,end)
        uv=np.column_stack((canvas['u0_um']+(linear%n)*canvas['pitch_um'],canvas['v_top_um']-(linear//n)*canvas['pitch_um']))
        def reference_batch():
            base,valid=reference_points(ref,field,uv,stop=guard.stop)
            cause=np.full(len(uv),CAUSES['reference_domain'],np.uint8)
            ids=np.flatnonzero(valid)
            if len(ids):
                # The inverse already checks the declared physical roundtrip tolerance.
                _,good=xyz_to_uvw(ref,field,base[ids],stop=guard.stop)
                cause[ids[~good]]=CAUSES['reference_ambiguous'];valid[ids]&=good
            cause[valid]=0
            return dict(base=base,reference_valid=valid,cause=cause),set()
        reference,_=_checkpoint(out,db,f'reference-{start}',reference_batch,volume)
        def batch():
            base=reference['base'];valid=reference['reference_valid'].copy();cause=reference['cause'].copy()
            xyz=np.full_like(base,np.nan);residual=np.full(len(base),np.nan);roundtrip=np.full(len(base),np.nan)
            rounding=np.full(len(base),np.nan)
            ids=np.flatnonzero(valid)
            if len(ids):
                moved,ok,info=transport(field,base[ids],w,ref.settings,stop=guard.stop)
                for key,code in CAUSES.items():cause[ids[info['cause']==key]]=code
                frame=metadata['output_frame'];origin=np.asarray(frame['origin_xyz_um']);unit=frame['voxel_um']
                with np.errstate(over='ignore',invalid='ignore'):
                    encoded=((moved-origin)/unit).astype(np.float32)
                    stored=encoded.astype(float)*unit+origin
                rounding[ids]=np.linalg.norm(stored-moved,axis=1)
                sampled,g,supported=sample_w_gradient(field,stored);norm=np.linalg.norm(g,axis=1)
                distance=abs(sampled-w)/np.maximum(norm,1e-30)
                precise=supported&(rounding[ids]<=.01*min(field.spacing_um))&(distance<=ref.settings.residual_fraction*min(field.spacing_um))
                cause[ids[ok&~precise]]=CAUSES['coordinate_precision'];ok&=precise
                moved=stored;moved[~ok]=np.nan
                back,good,_=transport(field,moved,ref.w,ref.settings,stop=guard.stop)
                errors=np.linalg.norm(back-base[ids],axis=1)
                failed=ok&(~good|(errors>ref.settings.roundtrip_fraction*min(field.spacing_um)))
                cause[ids[failed]]=CAUSES['roundtrip'];ok&=~failed
                valid[ids]&=ok;xyz[ids]=moved;residual[ids]=distance;roundtrip[ids]=errors
            values=np.full(len(base),np.nan);ct=np.zeros(len(base),bool);missing_batch=set();ids=np.flatnonzero(valid)
            if len(ids):values[ids],ct[ids]=sample_ct(volume,xyz[ids],missing=missing_batch)
            return dict(xyz=xyz,base=base,reference_valid=reference['reference_valid'],geometry=valid,
                ct=ct,value=values,cause=cause,residual=residual,roundtrip=roundtrip,coordinate_rounding=rounding),missing_batch
        data,absent=_checkpoint(out,db,f'{index}-{start}',batch,volume);missing|=absent
        for key in arrays:arrays[key][start:end]=data[key]
    for key in arrays:arrays[key]=arrays[key].reshape((n,n,3) if key in ('xyz','base') else (n,n))
    geometry,cells,bad,jacobian=_sampled_cells(arrays['xyz'],arrays['base'],arrays['geometry'],field,pitch_um=canvas['pitch_um'])
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
        elapsed_seconds=time.monotonic()-start_time,physical_acceptance=False,global_invertibility_proven=False)
    pending=Path(tempfile.mkdtemp(prefix='.'+name+'-',dir=out));(pending/'tifxyz').mkdir()
    image=np.zeros((n,n),np.uint8);image[ct]=np.clip(np.rint(arrays['value'][ct]*255/180),0,255).astype(np.uint8)
    for filename,data in [('image.png',image),('geometry-mask.png',geometry.astype(np.uint8)*255),
        ('ct-mask.png',ct.astype(np.uint8)*255),('cause-mask.png',arrays['cause']),('cell-mask.png',cells.astype(np.uint8)*255),
        ('tifxyz/valid-mask.png',geometry.astype(np.uint8)*255)]:Image.fromarray(data).save(pending/filename)
    import tifffile
    frame=metadata['output_frame'];coordinates=(xyz-np.asarray(frame['origin_xyz_um']))/frame['voxel_um']
    for axis,label in enumerate('xyz'):
        path=pending/'tifxyz'/f'{label}.tif';encoded=coordinates[:,:,axis].astype(np.float32)
        tifffile.imwrite(path,encoded,metadata=None)
        if not np.array_equal(tifffile.imread(path),encoded,equal_nan=True):raise ValueError('TIFXYZ readback differs')
    atomic_json(pending/'quality.json',quality)
    atomic_json(pending/'tifxyz/meta.json',dict(schema='aligned-uvw-tifxyz-v1',w=w,coordinate_units='output-frame voxels',
        output_frame=frame,canvas=canvas,row_direction='decreasing V',invalid_coordinates='NaN',
        validity_mask='valid-mask.png',mask_values={'invalid':0,'valid':255},ct_validity='../ct-mask.png',
        cell_validity='../cell-mask.png',causes=CAUSES,reference=metadata['map_sha256'],identity_sha256=digest(identity),
        geometry_includes_ct_missing=True,physical_acceptance=False,global_invertibility_proven=False))
    files={str(p.relative_to(pending)):file_sha(p) for p in pending.rglob('*') if p.is_file()}
    record=dict(w=w,identity_sha256=digest(identity),files=files,quality=quality,unavailable_ct_chunks=sorted(missing))
    atomic_json(pending/'LEVEL.json',record)
    for path in pending.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    guard();field.validate();volume.validate_used()
    with db:db.execute('INSERT INTO levels VALUES(?,?,?)',(name,file_sha(pending/'LEVEL.json'),pending.name))
    pending.rename(target)
    return record


def export_existing(args,*,stop=None):
    """CLI/programmatic entry; repeat identical arguments with resume=True."""
    from .workflow import implementation_sha
    from importlib.metadata import version
    import tifffile
    levels=args.levels
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
    with PhysicalField(FieldSource(descriptor['field'],descriptor['geometry'],cache_bytes=64<<20)) as field:
        field.validate()
        for item in (descriptor['field'],descriptor['field'].get('mask',{})):
            if item.get('path') and (out==Path(item['path']) or out in Path(item['path']).parents):raise ValueError('output overlaps field source')
        reference,metadata=load_reference(args.reference_sheet,levels,args.canvas_samples)
        with closing(Volume(args.source,voxel_um=args.voxel_um,level=args.level)) as volume:
            if metadata['identity']['ct']!=volume.identity:raise ValueError('CT identity differs from reference physical frame/source')
            identity=dict(schema='aligned-uvw-v1',field=field.identity,descriptor_sha256=descriptor_record['sha256'],
                reference=metadata,levels=list(levels),ct=volume.identity,transport=asdict(reference.settings),
                implementation_sha256=implementation_sha(),batch_size=BATCH,
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
                        for i,w in enumerate(levels):records.append(_render_level(out,db,i,w,reference,metadata,field,volume,identity,guard))
                        # Revalidate the reference's independent receipt and inventory before final publication.
                        _,current=load_reference(args.reference_sheet,levels,args.canvas_samples)
                        if current!=metadata:raise ValueError('reference changed during export')
                        field.validate();volume.validate_used()
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
                        process_peak_rss_bytes=rss,levels=[r['quality'] for r in records])
                    result=dict(status='UVW_READY_DIAGNOSTIC',identity=identity,level_receipt_sha256=hashes,quality=quality,
                        physical_acceptance=False,global_invertibility_proven=False)
                    pending=Path(tempfile.mkdtemp(prefix='.aggregate-',dir=out))
                    atomic_json(pending/'quality.json',quality);result['quality_sha256']=file_sha(pending/'quality.json')
                    atomic_json(pending/'UVW.json',result)
                    with db:db.execute('INSERT INTO aggregate VALUES(?,?)',(file_sha(pending/'UVW.json'),pending.name))
                    (pending/'quality.json').rename(out/'quality.json');(pending/'UVW.json').rename(out/'UVW.json')
                    return result
