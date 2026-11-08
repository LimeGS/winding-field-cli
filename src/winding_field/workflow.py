"""Single-writer, resumable end-to-end orchestration."""

import fcntl
import itertools
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time

import numpy as np

from . import __version__
from .config import validate_config,validate_observations,field_geometry,tifxyz_settings
from .io import Volume,bounds,resolve_source
from .util import atomic_json,digest,file_sha

STAGES=['prepare','fibres','field','mesh','flatten']
RECEIPTS=['PREPARED.json','FIBRES.json','FIELD.json','MESHES.json','FLAT.json']


def stage_path(config):
    legacy=tuple(zip(STAGES,RECEIPTS))
    if config.get('flattening_method')=='original':
        return legacy[:4]+(('original','ORIGINAL.json'),)
    if config.get('surface_output')=='native':
        return legacy[:3]+(('surfaces','SURFACES.json'),)
    return legacy[:3]+(('tifxyz','tifxyz/TIFXYZ.json'),) if config.get('surface_output','atlas')=='tifxyz' else legacy


def implementation_sha():
    package=Path(__file__).parent
    files={str(p.relative_to(package)):file_sha(p) for p in sorted(package.rglob('*'))
           if p.suffix in ('.py','.json')}
    dependencies={name:importlib.metadata.version(name) for name in
                  ('torch','numpy','scipy','numba','dynamic-network-architectures','avf-spline-generator',
                    'winding-field')}
    core=importlib.metadata.distribution('winding-field')
    files['winding-field']={name:file_sha(importlib.util.find_spec(name).origin)
                                      for name in sorted(core.read_text('top_level.txt').splitlines())}
    for name in ('avf_spline_generator','dynamic_network_architectures'):
        root=Path(next(iter(importlib.util.find_spec(name).submodule_search_locations)))
        files[name]={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*.py'))}
    return digest(dict(files=files,dependencies=dependencies))


def snapshot_exports(root,stage):
    patterns={'prepare':['lasagna/**/*','LASAGNA.json','lasagna-source/IDENTITY.json','lasagna-source/WORK.json',
                         'lasagna-source/input.sqlite','coarse/IDENTITY.json','coarse/COARSE.json',
                         'coarse/CALIBRATION.json'],'fibres':['fibres/*.afv'],
              'field':[],'mesh':['meshes/*.obj'],
              'flatten':['flat/*.obj','flat/*.png','flat/atlas.sqlite'],
              'surfaces':['surfaces/NATIVE.json','surfaces/LEVELS.json','surfaces/level-*/LEVEL.json',
                          'surfaces/level-*/IDENTITY.json','surfaces/level-*/work.sqlite'],
              'tifxyz':['tifxyz/**/*'],'original':['original/IDENTITY.json','original/ORIGINAL.json','original/publications.sqlite','original/level-*/**/*']}
    with sqlite3.connect(root/'products.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS exports (path TEXT PRIMARY KEY,stage TEXT,sha TEXT)')
        for pattern in patterns[stage]:
            for path in root.glob(pattern):
                if stage=='tifxyz' and path.name in ('.lock','STOP'):
                    continue
                if path.is_file():
                    db.execute('INSERT OR REPLACE INTO exports VALUES (?,?,?)',
                               (str(path.relative_to(root)),stage,file_sha(path)))


def open_volume(config):
    local=config.get('local')
    if not local and str(config['source']).startswith(('http','s3://')):
        name=str(config['source']).rstrip('/').rsplit('/',1)[-1]
        for base in os.environ.get('WINDING_CT_ROOTS','./data:~/data:/data:/workspace/ct').split(os.pathsep):
            path=Path(base).expanduser()/name
            if path.is_dir():
                local=str(path); break
    return Volume(config['source'],local=local,cache=config.get('cache'),cache_bytes=config.get('cache_bytes',2<<30),
                  offline=config.get('offline',False),voxel_um=config.get('voxel_um'),level=config.get('level',0),
                  reserve_bytes=config.get('reserve_bytes',1<<30))


def plan(config,volume=None,stages=()):
    validate_config(config)
    own=volume is None
    volume=open_volume(config) if own else volume
    try:
        lo,hi=bounds(config.get('roi_lo',[0,0,0]),config.get('roi_hi',list(volume.shape)),volume.shape)
        stride=config.get('stride',1)
        shape=(hi-lo-1)//stride+1
        if min(shape)<2:
            raise ValueError('ROI needs at least two observation samples per axis')
        validate_observations(config,volume,shape)
        method='external-afv' if config.get('fibres') else config.get('prediction','classical')
        halo=0 if method=='external-afv' and config.get('lasagna_input') else 40*stride
        input_lo=np.maximum(0,lo-halo); input_hi=np.minimum(volume.shape,hi+halo)
        if method in ('sd2','qual-v2'):
            core=48 if method=='qual-v2' else 32
            start=(lo//stride)//core*core
            end=((lo//stride+shape-1)//core+1)*core
            input_lo=np.maximum(0,lo%stride+start*stride-halo)
            input_hi=np.minimum(volume.shape,lo%stride+(end+40-1)*stride+1)
        chunk_count=np.prod((input_hi-1)//volume.chunks-input_lo//volume.chunks+1,dtype=np.int64)
        ct_bytes=int(chunk_count)*math.prod(volume.chunks)*volume.dtype.itemsize
        observations=dict(shape_zyx=shape.tolist(),spacing_xyz=(np.asarray(volume.spacing_xyz)*stride).tolist(),
                          origin_xyz=(np.asarray(volume.origin_xyz)+lo[::-1]*volume.spacing_xyz).tolist(),
                          frame=digest(volume.identity))
        lasagna_staging=lasagna_workspace=lasagna_import=0; lasagna_input=None
        if config.get('lasagna_input'):
            from .lasagna_input import source_identity
            lasagna_input=source_identity(config['lasagna_input'],config.get('lasagna_geometry'),
                                          volume.identity,observations,volume.cache)
            lasagna_import=math.prod(shape.tolist())*(21+volume.dtype.itemsize)
        elif method!='classical':
            from .lasagna import input_geometry,CHUNK,BORDER
            staged=input_geometry(volume,dict(observations,stride=stride,roi_lo_zyx=lo.tolist()))
            stage_shape=np.asarray(staged['shape_zyx'])
            lasagna_staging=math.prod(((stage_shape+CHUNK-1)//CHUNK*CHUNK).tolist())*volume.dtype.itemsize
            lasagna_workspace=math.prod((stage_shape+2*BORDER).tolist())*32+lasagna_staging*8
            stage_lo=np.asarray(staged['source_start_zyx']); stage_hi=stage_lo+(stage_shape-1)*stride+1
            input_lo=np.minimum(input_lo,stage_lo); input_hi=np.maximum(input_hi,stage_hi)
            chunk_count=np.prod((input_hi-1)//volume.chunks-input_lo//volume.chunks+1,dtype=np.int64)
            ct_bytes=int(chunk_count)*math.prod(volume.chunks)*volume.dtype.itemsize
        field=field_geometry(observations,config.get('field_spacing_um'))
        if config.get('bounded_preparation'):
            from .bounded_coarse import endpoint_grid
            field=endpoint_grid(observations,config['field_spacing_um'])
        observation_nodes=math.prod(shape.tolist()); field_nodes=math.prod(field['shape_zyx'])
        prepare_bytes=observation_nodes*48+lasagna_staging+lasagna_workspace+lasagna_import
        fibre_bytes=observation_nodes*12
        if config.get('fibres'): fibre_bytes+=Path(config['fibres']).stat().st_size*4
        spline_bytes=0
        if method=='qual-v2':
            spline_bytes=observation_nodes*8+math.prod((shape//2).tolist())*18
            fibre_bytes+=observation_nodes*8+spline_bytes
        output_bytes=prepare_bytes+fibre_bytes+field_nodes*40
        path=stage_path(config); terminal=path[-1][0]; extra={}
        if config.get('bounded_preparation'):
            output_bytes=config.get('bounded_bytes',12<<30)
            prepare_bytes=output_bytes//4;fibre_bytes=output_bytes//2
            spline_bytes=lasagna_staging=lasagna_workspace=0
            extra.update(bounded_preparation=True,solver_device=config.get('device','cpu'),
                         field_chunk=config.get('field_chunk',64),field_halo=2,solver_dtype='float64',
                         canonical_windows=dict(qual=128,qual_core=48,lasagna=64,lasagna_core=32),
                         output_budget_is_cap=True,geometry_bytes_are_data_dependent=True)
        folders=('maps','lasagna','lasagna-source','fibres','field','meshes','flat','surfaces','original')
        if config.get('bounded_preparation'):folders=(*folders,'coarse','pairs')
        product_reserve=config.get('reserve_bytes',1<<30)
        if terminal=='tifxyz':
            request=config['tifxyz_request']; scratch=int(request['scratch_gib']*(1<<30))
            reserve=int(request['reserve_disk_gib']*(1<<30))
            # Bounded retained work + TIFF allowance, independent of native face count.
            direct_bytes=min(scratch,36_000_000+(256<<20)+256*request['max_pixels']*len(request['levels']))
            products=dict(prepare=prepare_bytes,fibres=fibre_bytes,
                          field=field_nodes*40,tifxyz=direct_bytes)
            if config.get('bounded_preparation'):
                products['field']=output_bytes-prepare_bytes-fibre_bytes
            output_bytes=sum(products.values()); folders=('maps','lasagna','lasagna-source','fibres','field','tifxyz')
            product_reserve=max(product_reserve,reserve)
            extra.update(stages=[name for name,_ in path],estimated_products_by_stage=products,
                       tifxyz_scratch_bytes=scratch,tifxyz_reserve_disk_bytes=reserve,reserve_ram_bytes=2<<30,
                       experimental=True,comparison_status='UNMEASURED',comparison_reason='reference OOM')
        elif terminal=='surfaces':
            extra.update(stages=[name for name,_ in path],surface_time_limit_s=config.get('surface_time_limit_s',1800.),
                       coverage_target=config.get('coverage_target',1.),performance_target='UNMEASURED',
                        optional_mesh_derived_tifxyz=config.get('export_tifxyz',False))
        if config.get('bounded_preparation') and terminal=='tifxyz':folders=(*folders,'coarse','pairs')
        output=Path(config['output']); parent=output if output.exists() else output.parent
        written=sum(p.stat().st_size for name in folders
                    for p in (output/name).rglob('*') if p.is_file())
        remaining=0 if terminal in stages else max(0,output_bytes-written)
        while not parent.exists():
            parent=parent.parent
        free=shutil.disk_usage(parent).free; reserve=config.get('reserve_bytes',1<<30)
        cache_free=shutil.disk_usage(volume.cache).free
        shared=os.stat(parent).st_dev==os.stat(volume.cache).st_dev
        fits=(ct_bytes<=volume.limit and ct_bytes+reserve<cache_free
              and remaining+product_reserve+(ct_bytes if shared else 0)<free)
        mode=config.get('ct_mode','auto')
        if mode=='auto':
            mode='download' if volume.remote and fits else 'stream' if volume.remote else 'local'
        if mode=='download' and not fits and volume.remote and not stages:
            raise ValueError('requested CT download exceeds cache or available-space budget; increase budget or use stream')
        return dict(source=volume.identity,prediction=method,roi_lo_zyx=lo.tolist(),roi_hi_zyx=hi.tolist(),shape_zyx=shape.tolist(),
                     surface_method='external-lasagna' if lasagna_input else 'classical' if method=='classical' else 'lasagna',
                     lasagna_input=lasagna_input,estimated_lasagna_import_bytes=lasagna_import,
                     observations=observations,field=field,
                    input_lo_zyx=input_lo.tolist(),input_hi_zyx=input_hi.tolist(),ct_policy=mode,
                     estimated_ct_bytes=ct_bytes,estimated_products_bytes=output_bytes,
                      estimated_spline_dense_bytes=spline_bytes,
                      estimated_lasagna_staging_bytes=lasagna_staging,estimated_lasagna_workspace_bytes=lasagna_workspace,
                    existing_products_bytes=written,estimated_remaining_bytes=remaining,free_bytes=free,
                     reserve_bytes=reserve,cache_free_bytes=cache_free,shared_filesystem=shared,
                     physical_accuracy='UNVALIDATED',**extra)
    finally:
        if own:
            volume.close()


def _validate_products(root,state,stop=None,volume=None):
    from .artifacts import Tiles
    from .lasagna import validate_retained as validate_lasagna
    from .afv import validate_retained as validate_afv
    validate_lasagna(root,state.get('config',{}))
    bounded=state.get('config',{}).get('bounded_preparation',False)
    if bounded and (root/'PREPARED.json').exists():
        from .bounded_pipeline import runtime_identity,checkpoint
        meta=json.loads((root/'PREPARED.json').read_bytes())
        if meta['producer_identity']['runtime']!=runtime_identity(state['config'].get('lasagna_python')):
            raise ValueError('bounded Lasagna runtime changed')
        checkpoint(state['config'].get('lasagna_model_path'),state['config'].get('offline',False),prediction='lasagna')
    if state.get('config',{}).get('fibres'): validate_afv(root,state['config'])
    if state['stages']:
        with sqlite3.connect((root/'products.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            for name,sha in db.execute('SELECT path,sha FROM exports'):
                path=root/name
                if not path.is_file() or file_sha(path)!=sha:
                    raise ValueError('export product changed or missing: '+name)
    for stage,record in state['stages'].items():
        if file_sha(root/record['receipt'])!=record['sha256']:
            raise ValueError('stage receipt changed: '+stage)
    if 'prepare' in state['stages']:
        maps=Tiles(root/('coarse/maps' if bounded else 'maps'))
        expected=json.loads((root/'PREPARED.json').read_bytes())['tiles_sha256']
        actual=maps.identity(); maps.close()
        if expected!=actual:
            raise ValueError('prepared maps changed')
    if 'fibres' in state['stages']:
        expected=json.loads((root/'FIBRES.json').read_bytes())
        if file_sha(root/'fibres/fibres.sqlite')!=expected['sha256']:
            raise ValueError('fibres changed')
        if bounded:
            for name in ('raw','candidates'):
                if file_sha(root/'fibres'/f'{name}.sqlite')!=expected[f'{name}_sha256']:
                    raise ValueError('bounded fibre inventory changed: '+name)
        if ('probabilities_sha256' in expected
                and file_sha(root/'fibres/probabilities.npy')!=expected['probabilities_sha256']):
            raise ValueError('spline probabilities changed')
    if 'field' in state['stages']:
        from .field import open_field
        with open_field(root) as source:
            source.validate(hashes=True)
    if 'mesh' in state['stages']:
        from .surface_mesh import MeshReader
        reader=MeshReader(root/'meshes'); reader.close()
    if 'original' in state['stages'] and volume is not None:
        from .original_pipeline import verify_missing_ct
        for record in json.loads((root/'ORIGINAL.json').read_bytes())['levels']:
            verify_missing_ct(volume,record)
    if 'flatten' in state['stages']:
        # Immutable chart payload hashes are checked by the normal stage resume.
        db=sqlite3.connect((root/'flat/atlas.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for name,sha in db.execute('SELECT name,sha FROM charts'):
                if file_sha(root/'flat'/(name+'.npz'))!=sha:
                    raise ValueError('flat chart changed')
        finally:
            db.close()
    if 'surfaces' in state['stages']:
        from .native_pipeline import run_native
        native=json.loads((root/'surfaces/NATIVE.json').read_bytes())
        receipt=json.loads((root/'SURFACES.json').read_bytes())
        if digest(native)!=receipt['identity_sha256'] or native['field_sha256']!=file_sha(root/'FIELD.json'):
            raise ValueError('native publication identity changed')
        return run_native(root,state['config'],stop=stop)
    if 'tifxyz' in state['stages']:
        from .tifxyz import verify_export
        result=verify_export(root/'tifxyz',stop=stop)
        if result['status'] not in ('TIFXYZ_VERIFIED','TIFXYZ_PARTIAL_VERIFIED','PAUSED','BLOCKED','FAILED'):
            raise ValueError('unexpected tifxyz verification result')
        if result['status'] in ('TIFXYZ_VERIFIED','TIFXYZ_PARTIAL_VERIFIED'):
            config=state['config'];record=state['stages']['tifxyz']
            marker_sha=_tifxyz_ready(root/'tifxyz',result,config['tifxyz_request_sha256'],
                                     config['tifxyz_request'].get('coverage_policy','strict'))
            status='READY' if result['status']=='TIFXYZ_VERIFIED' else 'PARTIAL'
            if (record.get(status.lower()+'_sha256')!=marker_sha or record.get('status')!=status
                    or record.get('coverage_complete') is not (status=='READY')):
                raise ValueError('tifxyz stage publication differs')
        return result


def _tifxyz_ready(out,result,request_sha,coverage_policy='strict'):
    """Bind child publication and quality before registering or resuming its stage."""
    from .tifxyz import FLAGS,READER,_audit_status
    from .tifxyz_field import read_metadata
    markers=[out/name for name in ('READY.json','PARTIAL.json')
             if (out/name).exists() or (out/name).is_symlink()]
    if len(markers)!=1:
        raise ValueError('tifxyz requires exactly one READY/PARTIAL publication marker')
    marker,marker_record=read_metadata(markers[0]);status=markers[0].stem;complete=status=='READY'
    receipt,receipt_record=read_metadata(out/'TIFXYZ.json')
    identity,_=read_metadata(out/'EXPORT.json')
    for value in (result,marker,receipt,identity):
        if any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in FLAGS.items()):
            raise ValueError('tifxyz acceptance flags differ')
    expected_verify='TIFXYZ_VERIFIED' if complete else 'TIFXYZ_PARTIAL_VERIFIED'
    if (result.get('status') not in (status,expected_verify)
            or any(value.get('coverage_complete') is not complete for value in (result,marker,receipt))
            or marker.get('status')!=status or receipt.get('status')!=status
            or marker.get('tifxyz_sha256')!=receipt_record['sha256']
            or marker.get('identity_sha256')!=digest(identity)
            or receipt.get('identity_sha256')!=digest(identity)
            or receipt.get('reader')!=READER or identity.get('reader')!=READER
            or identity.get('effective_settings',{}).get('request_sha256')!=request_sha):
        raise ValueError('tifxyz publication binding/quality differs')
    settings=identity['effective_settings'];policy=settings.get('coverage_policy','strict')
    if (policy not in ('strict','allow_partial') or policy!=coverage_policy
            or (not complete and policy!='allow_partial')):
        raise ValueError('tifxyz partial publication requires explicit allow_partial policy')
    levels=receipt.get('levels');requested=settings.get('levels')
    if not isinstance(levels,list) or not levels or not isinstance(requested,list) or len(levels)!=len(requested):
        raise ValueError('tifxyz level count differs')
    all_complete=True
    for index,item in enumerate(levels):
        if (not isinstance(item,dict) or item.get('status') not in ('READY','PARTIAL')
                or item.get('coverage_complete') is not (item['status']=='READY')
                or item.get('segment')!=f'level-{index:04d}' or item.get('level')!=requested[index]):
            raise ValueError('tifxyz level quality/selection differs')
        verified,verified_record=read_metadata(out/item['segment']/'VERIFIED.json')
        if (any(verified.get(k)!=item[k] or type(verified.get(k)) is not type(item[k])
                for k in ('status','coverage_complete'))
                or any(type(verified.get(k)) is not type(v) or verified.get(k)!=v for k,v in FLAGS.items())
                or item.get('manifest',{}).get('files',{}).get('VERIFIED.json',{}).get('sha256')!=verified_record['sha256']):
            raise ValueError('tifxyz VERIFIED quality/binding differs')
        if _audit_status(verified.get('audit',{}),policy)!=item['status']:
            raise ValueError('tifxyz audit/publication quality differs')
        all_complete=all_complete and item['coverage_complete']
    if complete!=all_complete:
        raise ValueError('tifxyz aggregate coverage differs')
    return marker_record['sha256']


def run(config,resume=False,until=None,extra_surface_seconds=0,field_sweeps=None):
    validate_config(config)
    if field_sweeps is not None and (type(field_sweeps) is not int or field_sweeps<1
                                    or not resume or not config.get('bounded_preparation')):
        raise ValueError('field sweep target requires a bounded resume and positive integer')
    if config.get('flattening_method')=='original':
        from .original_pipeline import settings
        settings(config.get('original_request'))
    elif config.get('surface_output')=='native':
        from .native_pipeline import _settings
        _settings(config)
    path=stage_path(config); terminal=path[-1][0]; until=until or terminal
    native_until='render'
    if terminal=='surfaces' and until in ('mesh','flatten','assemble','render'):
        native_until=until; until=terminal
    if (type(extra_surface_seconds) not in (int,float) or not math.isfinite(extra_surface_seconds)
            or extra_surface_seconds<0 or extra_surface_seconds and (not resume or terminal!='surfaces')):
        raise ValueError('extra surface seconds require a native resume and a finite nonnegative allowance')
    if until not in dict(path):
        raise ValueError('invalid final stage for selected surface output')
    import torch
    from .predict import prepare
    from .fibres import trace
    from .field import solve
    from .meshes import mesh
    from .flatten import flatten
    from .exports import export_lasagna,export_afv
    config=dict(config); root=Path(config['output']).expanduser().resolve(); config['output']=str(root)
    torch.set_num_threads(config.get('threads',min(4,os.cpu_count() or 1)))
    if root.exists() and not resume:
        raise FileExistsError('output exists; use resume or a new directory')
    if not resume:
        probe=open_volume(config)
        try:
            plan(config,probe)
            if config.get('fibres'):
                from .afv import source_identity
                source_identity(config['fibres'],config.get('fibres_geometry'),probe.identity)
        finally: probe.close()
        if not config.get('lasagna_input') and (config.get('fibres') or config.get('prediction','classical')!='classical'):
            from . import lasagna
            lasagna.runtime_identity(config.get('lasagna_python'))
            lasagna.checkpoint(config.get('lasagna_model_path'),config.get('offline',False),prediction='lasagna')
    root.mkdir(parents=True,exist_ok=resume)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            state=json.loads((root/'RUN.json').read_bytes())
            if state['config']!=config:
                raise ValueError('run configuration changed; use a new output')
            if state.get('implementation_sha256')!=implementation_sha():
                raise ValueError('implementation changed; resume with the original package (v1.5.0 for pre-cutover runs) or use a new output')
            from .ct_reader import runtime_identity
            if 'source' in state and state['source'].get('native_runtime')!=runtime_identity():
                raise ValueError('CT native runtime changed; resume with the original runtime')
            if field_sweeps is not None:
                from .field_budget import effective_budget
                if 'field' in state['stages'] or (root/'FIELD.json').exists():
                    raise ValueError('cannot extend a published field')
                if not (root/'field/CHECKPOINT.json').is_file():
                    raise ValueError('field sweep extension requires an existing checkpoint')
                if field_sweeps<effective_budget(root/'field',config.get('sweeps',4)):
                    raise ValueError('field sweep target cannot decrease')
            (root/'STOP').unlink(missing_ok=True)
        else:
            state=dict(schema='winding-field-run-v1',version=__version__,config=config,stages={},
                       status='STARTING',physical_acceptance=False,created=time.time(),implementation_sha256=implementation_sha())
            if terminal=='tifxyz':
                state.update(experimental=True,comparison_status='UNMEASURED',comparison_reason='reference OOM',
                             native_face_identity=False,native_topology_equivalence='UNMEASURED')
            atomic_json(root/'RUN.json',state)
        interrupted=False
        def handle(signum,frame):
            nonlocal interrupted
            interrupted=True
        previous={sig:signal.signal(sig,handle) for sig in (signal.SIGINT,signal.SIGTERM)}
        stop=lambda:interrupted or (root/'STOP').exists()
        def unfinished(result):
            if result.get('status') not in ('PAUSED','BLOCKED','FAILED'):
                raise ValueError('unexpected tifxyz worker result')
            state.update(status=result['status'],tifxyz_result=result,updated=time.time())
            atomic_json(root/'RUN.json',state)
            return state
        volume=None
        try:
            volume=open_volume(config)
            if 'source' in state and state['source']!=volume.identity:
                raise ValueError('source metadata changed')
            state['source']=volume.identity
            if config.get('fibres'):
                from .afv import source_identity
                fibres_input=source_identity(config['fibres'],config.get('fibres_geometry'),volume.identity)
                if 'fibres_input' in state and state['fibres_input']!=fibres_input:
                    raise ValueError('external AFV input identity changed')
                state['fibres_input']=fibres_input
            volume.bind_ledger(root/'source.sqlite')
            if resume:
                volume.validate_used(); result=_validate_products(root,state,stop,volume)
                if result is not None:
                    if terminal=='surfaces':
                        state['native_result']=result
                        if result['status'] not in ('COMPLETE','PARTIAL'):
                            state.update(status=result['status'],updated=time.time())
                            atomic_json(root/'RUN.json',state)
                            return state
                    else:
                        if result['status'] not in ('TIFXYZ_VERIFIED','TIFXYZ_PARTIAL_VERIFIED'):
                            return unfinished(result)
                        state['tifxyz_result']=result
            estimate=plan(config,volume,state['stages']); state['plan']=estimate
            if estimate['lasagna_input']:
                if 'lasagna_input' in state and state['lasagna_input']!=estimate['lasagna_input']:
                    raise ValueError('external Lasagna input identity changed')
                state['lasagna_input']=estimate['lasagna_input']
            reserve=max(estimate['reserve_bytes'],estimate.get('tifxyz_reserve_disk_bytes',0))
            if terminal not in state['stages'] and estimate['free_bytes']<estimate['estimated_remaining_bytes']+reserve:
                if terminal=='tifxyz':
                    return unfinished(dict(status='BLOCKED',reasons=[dict(code='DISK_HEADROOM',detail='planned products and reserve')]))
                raise OSError('insufficient disk space for products and reserve; choose explicit stride/ROI or another disk')
            if estimate['ct_policy']=='download' and not state['stages']:
                low=np.array(estimate['input_lo_zyx']); high=np.array(estimate['input_hi_zyx']); chunks=np.array(volume.chunks)
                for key in itertools.product(*(range(a//c,(b-1)//c+1) for a,b,c in zip(low,high,chunks))):
                    a=np.array(key)*chunks; b=np.minimum(a+chunks,volume.shape)
                    volume.read(a,b)
                    if stop():
                        state['status']='PAUSED'; atomic_json(root/'RUN.json',state); return state
            for stage,receipt in path:
                if stage not in state['stages']:
                    state.update(status='RUNNING',active_stage=stage,updated=time.time())
                    atomic_json(root/'RUN.json',state)
                    print(f'[{stage}] started',file=sys.stderr,flush=True)
                    started=time.monotonic()
                    if stage=='prepare':
                        if config.get('bounded_preparation'):
                            from .bounded_pipeline import prepare as bounded_prepare
                            result=bounded_prepare(volume,root,config,stop)
                        else:result=prepare(volume,root,config,stop)
                        if result['status']=='PREPARED' and not config.get('bounded_preparation'):
                            if export_lasagna(root,stop) is None:
                                result=dict(status='PAUSED')
                    elif stage=='fibres':
                        if config.get('bounded_preparation'):
                            from .bounded_pipeline import fibres as bounded_fibres
                            result=bounded_fibres(volume,root,config,stop)
                        else:result=trace(root,config,stop)
                        if result['status']=='FIBRES_READY':
                            sha=export_afv(root,stop)
                            if sha is None:
                                result=dict(status='PAUSED')
                            else:
                                result['afv_sha256']=sha; atomic_json(root/receipt,result)
                    elif stage=='field':
                        result=solve(root,config,stop,**({'sweep_target':field_sweeps} if field_sweeps is not None else {}))
                    elif stage=='mesh':
                        result=mesh(root,config,stop)
                    elif stage=='original':
                        from .original_pipeline import run_original
                        result=run_original(root,config,volume,stop)
                    elif stage=='tifxyz':
                        from .tifxyz import export_field
                        out=root/'tifxyz'; continuing=out.exists()
                        settings=tifxyz_settings(out,config['tifxyz_request'],resume=continuing)
                        result=export_field(root,out,settings,resume=continuing,stop=stop)
                        if result.get('status') not in ('READY','PARTIAL'):
                            return unfinished(result)
                        marker_sha=_tifxyz_ready(out,result,config['tifxyz_request_sha256'],
                                                config['tifxyz_request'].get('coverage_policy','strict'))
                        state['tifxyz_result']=result
                    elif stage=='surfaces':
                        from .native_pipeline import run_native
                        result=run_native(root,config,stop,native_until,extra_seconds=extra_surface_seconds)
                        state['native_result']=result
                        if result.get('status') not in ('COMPLETE','PARTIAL'):
                            state.update(status=result.get('status','FAILED'),updated=time.time())
                            atomic_json(root/'RUN.json',state)
                            return state
                    else:
                        result=flatten(root,config,volume if config.get('previews',True) else None,stop)
                    if result['status']=='PAUSED':
                        state['status']='PAUSED'; atomic_json(root/'RUN.json',state); return state
                    snapshot_exports(root,stage)
                    state['stages'][stage]=dict(receipt=receipt,sha256=file_sha(root/receipt),seconds=time.monotonic()-started)
                    if stage=='tifxyz':
                        state['stages'][stage].update(status=result['status'],coverage_complete=result['coverage_complete'],
                                                     **{result['status'].lower()+'_sha256':marker_sha})
                    state['io']=volume.stats()
                    atomic_json(root/'RUN.json',state)
                    print(f'[{stage}] ready ({state["stages"][stage]["seconds"]:.2f}s)',file=sys.stderr,flush=True)
                if stage==until:
                    break
            state.update(status='COMPLETE' if terminal in state['stages'] else 'PAUSED',updated=time.time(),io=volume.stats())
            if terminal=='tifxyz' and terminal in state['stages']:
                state['coverage_complete']=state['stages'][terminal]['coverage_complete']
                if not state['coverage_complete']:
                    state['status']='PARTIAL'
            if terminal=='surfaces' and terminal in state['stages']:
                state['status']=json.loads((root/'SURFACES.json').read_bytes())['status']
            atomic_json(root/'RUN.json',state)
            if state['status']=='COMPLETE' and terminal=='flatten':
                from .demo import gallery
                gallery(root)
            return state
        except Exception as error:
            state.update(status='FAILED',error=str(error),updated=time.time())
            atomic_json(root/'RUN.json',state)
            raise
        finally:
            if volume is not None:
                volume.close()
            for sig,handler in previous.items():
                signal.signal(sig,handler)
