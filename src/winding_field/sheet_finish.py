"""Source-rebound constant-W validation and verified CT rendering."""
import math
from pathlib import Path
import shutil
import signal
import sqlite3
import time

import numpy as np

from . import patch_pipeline as p
from . import flatboi_checkpoint
from .flatboi_checkpoint import load_checkpoint, positive_uv
from .flatten import iter_overlap_pairs
from .native_patches import ResourceGuard
from .original_recipe import _points, face_quality, sample_points, scale_uv_to_world
from .sheet_atlas import _arrays, _edge, _halo, _json, _path, _tile, _verified, MAX_PIXELS
from .sheet_prepare import centerline_points, load_prepared
from .surface_mesh import MeshReader
from .util import atomic_json, file_sha


def calibrate(v,f,uv):
    scaled,report=scale_uv_to_world(v,f,uv)
    direction=np.linalg.lstsq(scaled-scaled.mean(0),v[:,2]-v[:,2].mean(),rcond=None)[0]
    norm=np.linalg.norm(direction)
    if not np.isfinite(norm) or norm<=1e-12:raise ValueError('no finite Z orientation')
    direction/=norm;rotation=np.column_stack([[direction[1],-direction[0]],direction]);result=scaled@rotation
    report.update(rotation=rotation.tolist(),z_correlation=float(np.corrcoef(result[:,1],v[:,2])[0,1]))
    return result,report


def reference_anchor(v,f,uv,centerline,z):
    centerline=centerline_points(centerline)
    if not math.isfinite(z):raise ValueError('finite reference height required')
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    a,b=v[edges[:,0]],v[edges[:,1]];crossing=(np.minimum(a[:,2],b[:,2])<=z)&(np.maximum(a[:,2],b[:,2])>=z)&(a[:,2]!=b[:,2])
    edges=edges[crossing];a,b=a[crossing],b[crossing]
    if not len(edges):raise ValueError('reference height outside mesh')
    weight=(z-a[:,2])/(b[:,2]-a[:,2]);xyz=a+weight[:,None]*(b-a)
    q=uv[edges[:,0]]+weight[:,None]*(uv[edges[:,1]]-uv[edges[:,0]])
    center=np.array([np.interp(z,centerline[:,2],centerline[:,i]) for i in range(2)])
    delta=xyz[:,:2]-center;angles=np.arctan2(delta[:,1],delta[:,0])
    # Exact seam copies share XYZ; choose the leftmost UV copy consistently.
    index=np.lexsort((q[:,0],np.round(np.abs(angles),12)))[0]
    return dict(uv=q[index].tolist(),xyz=xyz[index].tolist(),angle=float(angles[index]),
                edge=edges[index].tolist(),weight=float(weight[index]),reference_z=z,
                rule='closest +X angular crossing of exact reference Z; leftmost UV tie-break')


def validate(prepared,native,output,centerline,reference_z,*,mesh_source=None):
    source=Path(prepared).absolute();native=Path(native).absolute();out=Path(output).absolute()
    if any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('output symlink refused')
    receipt,mapped=load_prepared(source)
    cp=centerline_points(receipt.get('centerline') if centerline is None else centerline)
    if 'centerline' in receipt and not np.array_equal(cp,receipt['centerline']):raise ValueError('centerline differs from preparation')
    identity=_json(_path(native,'IDENTITY.json'))
    if identity.get('source_prepared_sha256')!=file_sha(source/'PREPARED.json'):raise ValueError('native preparation receipt differs; import legacy native first')
    if identity.get('runner_sha256')!=file_sha(flatboi_checkpoint.__file__):raise ValueError('native runner differs; use import-native with original runner SHA256')
    if (identity['source_map_sha256']!=file_sha(source/'initial-map.npz') or identity['source_obj_sha256']!=file_sha(source/'inputmesh.obj')
        or identity['app_sha256']!=p.LAUNCHER_SHA or identity['binary_sha256'] not in [p.NATIVE_RECIPES[r] for r in p.CHECKPOINT_RECIPES]
        or identity['energy']!='symmetric_dirichlet' or identity['tolerance']!=1e-6 or identity['threads']!=2):
        raise ValueError('native source/recipe identity differs')
    latest,checkpoint=load_checkpoint(native,identity)
    if latest is None:raise ValueError('native checkpoint required')
    state=_json(_path(native,'STATUS.json'))
    if state.get('status') not in ('NATIVE_FINISHED_PENDING_QUALITY_VALIDATION','IMPORTED_CHECKPOINT') or state.get('iterations')!=latest['iterations']:
        raise ValueError('finished native checkpoint required')
    if state['status']=='IMPORTED_CHECKPOINT' and not latest['converged'] and latest['iterations']<200:
        raise ValueError('imported native checkpoint must finish before validation')
    if not math.isfinite(reference_z):raise ValueError('finite reference height required')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();resources=ResourceGuard(out,6.,8.)
    def guard():
        resources()
        if (out/'STOP').exists():raise InterruptedError('validation stopped')
        if time.monotonic()-started>6*3600:raise TimeoutError('six-hour validation bound')
    guard()
    with MeshReader(mesh_source or receipt['source']) as reader:
        if reader.manifest_sha!=receipt['source_manifest_sha256']:raise ValueError('source mesh manifest differs')
        if reader.manifest!=_json(source/'MESH.json'):raise ValueError('copied source mesh differs')
        v,f,vi,fi,*_=p.read_level(reader,receipt['level_index'],memory_gib=6.)
        rows=mapped['source_vertex_rows'];face_rows=mapped['source_face_rows']
        np.testing.assert_array_equal(mapped['world'],v[rows])
        np.testing.assert_array_equal(rows[mapped['faces']],f[face_rows])
        np.testing.assert_array_equal(mapped['source_vertex_ids'],vi[rows])
        np.testing.assert_array_equal(mapped['source_face_ids'],fi[face_rows])
        disconnected=_arrays(source/'disconnected-source.npz')
        np.testing.assert_array_equal(np.sort(np.r_[mapped['source_face_ids'],disconnected['source_face_ids']]),fi)
        dr=np.searchsorted(vi,disconnected['source_vertex_ids']);df=np.searchsorted(fi,disconnected['source_face_ids'])
        np.testing.assert_array_equal(vi[dr],disconnected['source_vertex_ids'])
        np.testing.assert_array_equal(disconnected['world'],v[dr])
        np.testing.assert_array_equal(dr[disconnected['faces']],f[df])
        reader.validate()
    del v,f,vi,fi,rows,face_rows,disconnected
    v,f=mapped['world'],mapped['faces'];uv,binding=positive_uv(checkpoint,v,f,guard)
    uv,calibration=calibrate(v,f,uv);display,scientific,quality=face_quality(v,f,uv)
    if quality['inverted_or_invalid_area_fraction']!=0 or not display.any():raise ValueError('invalid calibrated UV geometry')
    anchor=reference_anchor(v,f,uv,cp,reference_z)
    report=dict(status='MAP_VALIDATED_DIAGNOSTIC',w=receipt['w'],level_index=receipt['level_index'],
                source_prepared_sha256=file_sha(source/'PREPARED.json'),checkpoint=latest,binding=binding,
                calibration=calibration,quality=quality,reference=anchor,centerline=cp.tolist(),
                world_bounds=[v.min(0).tolist(),v.max(0).tolist()],uv_bounds=[uv.min(0).tolist(),uv.max(0).tolist()],
                overlap=dict(complete=False),source_faces=receipt['source_faces'],principal_faces=len(f),
                disconnected_faces=receipt['disconnected_faces'],code_sha256=file_sha(__file__),physical_acceptance=False)
    involved=np.zeros(len(f),bool)
    for a,b in iter_overlap_pairs(uv,f,guard,max_candidates=262144,max_tests=200_000_000,report=report['overlap']):involved[[a,b]]=True
    if not report['overlap']['complete']:raise ValueError('incomplete overlap audit')
    report['overlapping_faces']=int(involved.sum());np.save(out/'overlap-faces.npy',np.flatnonzero(involved))
    mapped.update(uv=uv,display=display,scientific=scientific&~involved);np.savez_compressed(out/'map.npz',**mapped)
    report['map_sha256']=file_sha(out/'map.npz');atomic_json(out/'QUALITY.json',report)
    for name in ('MESH.json','PREPARED.json','disconnected-source.npz'):shutil.copyfile(source/name,out/name)
    if 'import_sha256' in identity:shutil.copyfile(native/'IMPORTED.json',out/'IMPORTED.json')
    report=dict(report,files={p.name:file_sha(p) for p in out.iterdir() if p.is_file()},seconds=time.monotonic()-started)
    guard();atomic_json(out/'VALIDATED.json',report);return report


def check_samples(mapped,data,canvas,scalar):
    if data['image'][data['mask']!=2].any():raise ValueError('nonblack invalid pixels')
    rr,cc=np.where(data['mask']==2)
    if not len(rr):return 0
    pick=np.unique(np.linspace(0,len(rr)-1,min(8,len(rr))).astype(int));rr,cc=rr[pick],cc[pick]
    order=np.argsort(mapped['source_face_rows']);rows=mapped['source_face_rows'][order]
    selected=order[np.searchsorted(rows,data['source_face_rows'][rr,cc])]
    np.testing.assert_array_equal(mapped['source_face_rows'][selected],data['source_face_rows'][rr,cc])
    query=np.column_stack([canvas['global_u0']+(cc+canvas['column_offset'])*canvas['pitch'],
                           canvas['global_v_top']-(rr+canvas['row_offset'])*canvas['pitch']])
    expected,known=_points(mapped['world'],mapped['faces'],mapped['uv'],selected,query,.005)
    if not known.all():raise ValueError('saved pixel not in its source triangle')
    np.testing.assert_allclose(data['xyz'][rr,cc],expected,rtol=0,atol=1e-10)
    values,valid=scalar(data['xyz'][rr,cc])
    if not valid.all():raise ValueError('saved observed pixel missing scalar CT')
    np.testing.assert_array_equal(np.rint(np.clip(values,0,255)).astype('u1'),data['image'][rr,cc])
    return len(rr)


def render(validated,output,config):
    source=Path(validated).absolute();out=Path(output).absolute();report=_json(_path(source,'VALIDATED.json'))
    if any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('output symlink refused')
    if report.get('status')!='MAP_VALIDATED_DIAGNOSTIC' or not report['overlap']['complete']:raise ValueError('validated map required')
    required={'map.npz','QUALITY.json','MESH.json','PREPARED.json','disconnected-source.npz'}
    if not required<=report['files'].keys():raise ValueError('unbound validated payloads')
    for name,sha in report['files'].items():_verified(source,name,sha)
    if set(config)-{'ct','settings','prior_identity','prior_ledger','prior_ledger_sha256'}:raise ValueError('unknown CT render configuration')
    ctconfig=config['ct']
    if set(ctconfig)-{'source','voxel_um','level'} or not isinstance(ctconfig.get('source'),str) or not ctconfig['source']:
        raise ValueError('explicit CT source required')
    if type(ctconfig.get('level',0)) is not int or ctconfig.get('level',0)<0:raise ValueError('invalid CT level')
    request=dict(memory_gib=6.,process_tree_gib=6.,raster_workers=2,ct_workers=4)
    request.update(config['settings']);cfg=p.settings(request)
    if cfg['tile_size']**2*33>128<<20:raise ValueError('tile mapping exceeds128MiB verification bound')
    if cfg['pitch_um']!=19.2 or cfg['ct_sampler']!='native-float64' or cfg['disk_reserve_gib']<8 or cfg['process_tree_gib']>6 or cfg['memory_gib']>6:
        raise ValueError('constant-sheet requires19.2um/native-float64, <=6GiB memory and >=8GiB disk reserve')
    if cfg['reuse_export'] is not None:raise ValueError('patch reuse is not a constant-sheet input')
    frame=cfg['output_frame'];mesh=_json(source/'MESH.json')
    if frame['origin_xyz_um']!=[0,0,0]:raise ValueError('constant-sheet global XYZ requires zero frame origin')
    if frame['id']!=mesh['metadata']['frame'] or frame['voxel_um']!=mesh['metadata']['voxel_um']:raise ValueError('CT output frame differs from exact mesh frame')
    prior=config.get('prior_identity');ledger=config.get('prior_ledger')
    if (ledger is None)!=(config.get('prior_ledger_sha256') is None) or (ledger and not prior):raise ValueError('prior CT ledger requires bound identity and SHA256')
    frozen=_json(Path(prior)) if prior else None
    if ledger and file_sha(ledger)!=config['prior_ledger_sha256']:raise ValueError('prior CT ledger hash differs')
    mapped=_arrays(source/'map.npz');uv=mapped['uv']
    wh=np.ceil(np.ptp(uv,axis=0)/(19.2/frame['voxel_um'])).astype(np.int64)+1;pixels=math.prod(map(int,wh))
    if pixels<=0 or pixels>MAX_PIXELS:raise ValueError('sheet exceeds atlas pixel bound')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();resources=ResourceGuard(out,6.,8.);stopped=False;ct=None
    def stop(*_):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def guard():
        resources()
        if stopped or (out/'STOP').exists():raise InterruptedError('sheet stopped; completed tiles retained')
        if time.monotonic()-started>6*3600:raise TimeoutError('six-hour sheet bound')
    try:
        guard()
        if shutil.disk_usage(out).free<8*2**30+pixels*40+sum((source/n).stat().st_size for n in report['files']):raise InterruptedError('sheet disk admission failed')
        for name,sha in report['files'].items():
            guard();shutil.copyfile(source/name,out/name);_verified(out,name,sha)
        shutil.copyfile(source/'VALIDATED.json',out/'VALIDATED.json')
        ctroot=out/'ct';ctroot.mkdir();ct=p.PatchCT(ctconfig['source'],ctroot,cfg,voxel_um=ctconfig.get('voxel_um'),level=ctconfig.get('level',0))
        if frozen:
            for key in ('source','metadata_sha256','origin_xyz','spacing_xyz','shape','level'):
                if ct.identity[key]!=frozen['ct'][key]:raise ValueError('prior CT identity differs: '+key)
        if ledger:
            with sqlite3.connect(Path(ledger).absolute().as_uri()+'?mode=ro',uri=True) as previous:
                cursor=previous.execute('SELECT key,sha FROM chunks')
                while batch:=cursor.fetchmany(4096):guard();ct.db.executemany('INSERT INTO chunks VALUES (?,?)',batch)
            ct.db.commit()
        atomic_json(out/'CT-IDENTITY.json',dict(ct=ct.identity,settings=cfg,
                    prior_identity_sha256=file_sha(prior) if prior else None,prior_ct_ledger_sha256=config.get('prior_ledger_sha256'),
                    code={q.name:file_sha(q) for q in Path(p.__file__).parent.glob('*.py')}))
        rendered=p.render_map(mapped,out/'render',lambda xyz:ct.sample(xyz,frame),frame,
                              pitch_um=19.2,tile_size=cfg['tile_size'],raster_workers=cfg['raster_workers'],guard=guard)
        atomic_json(out/'RENDER.json',rendered)
        def scalar(xyz):
            grid=(xyz*frame['voxel_um']+np.asarray(frame['origin_xyz_um'])-ct.volume.origin_xyz)/ct.volume.spacing_xyz
            return sample_points(grid,ct.volume.shape,ct.volume.chunks,ct._chunk)
        checks=0;tops={};left=None;last_row=None
        for tile in rendered['tiles']:
            guard();data=_tile(out,tile);canvas=tile['canvas'];row,col=canvas['row_offset'],canvas['column_offset']
            if row!=last_row:left=None;last_row=row
            if left is not None:_halo(left,_edge(data,(slice(None),0)))
            if row:_halo(tops[col],_edge(data,(0,slice(None))))
            left=_edge(data,(slice(None),-1));tops[col]=_edge(data,(-1,slice(None)))
            checks+=check_samples(mapped,data,canvas,scalar)
        if not checks:raise ValueError('no observed CT pixels verified')
        ctidentity=ct.identity;ct.close();ct=None
        atomic_json(out/'VERIFIED.json',dict(status='CT_SHEET_VERIFIED_DIAGNOSTIC',w=report['w'],scalar_and_xyz_spotchecks=checks,max_pixel_error=0,
                    all_tile_hashes_and_halos_verified=True,ct_ledger_sha256=file_sha(ctroot/'ct-chunks.sqlite'),
                    seconds=time.monotonic()-started,peak_tree_rss_bytes=resources.peak,physical_acceptance=False))
        record=dict(schema='winding-atlas-sheet-v1',kind='constant_w',w=report['w'],level_index=report['level_index'],frame=frame,
                    ct=ctidentity,reference_z=report['reference']['reference_z'],anchor_uv=report['reference']['uv'],z_increases_up=True,
                    files={q.name:file_sha(q) for q in sorted(out.iterdir()) if q.is_file()})
        record['files']['ct/ct-chunks.sqlite']=file_sha(ctroot/'ct-chunks.sqlite')
        guard();atomic_json(out/'SHEET.json',record);return record
    except BaseException as error:
        atomic_json(out/'FAILED.json',dict(error=repr(error)));raise
    finally:
        if ct is not None:ct.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
