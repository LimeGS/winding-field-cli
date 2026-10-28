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
from .config import validate_config,validate_observations,field_geometry
from .io import Volume,bounds,resolve_source
from .util import atomic_json,digest,file_sha

STAGES=['prepare','fibres','field','mesh','flatten']
RECEIPTS=['PREPARED.json','FIBRES.json','FIELD.json','MESHES.json','FLAT.json']


def implementation_sha():
    package=Path(__file__).parent
    files={str(p.relative_to(package)):file_sha(p) for p in sorted(package.rglob('*'))
           if p.suffix in ('.py','.json')}
    dependencies={name:importlib.metadata.version(name) for name in
                  ('torch','numpy','scipy','numba','dynamic-network-architectures','avf-spline-generator',
                   'winding-field-experiments')}
    core=importlib.metadata.distribution('winding-field-experiments')
    files['winding-field-experiments']={name:file_sha(importlib.util.find_spec(name).origin)
                                      for name in sorted(core.read_text('top_level.txt').splitlines())}
    for name in ('avf_spline_generator','dynamic_network_architectures'):
        root=Path(next(iter(importlib.util.find_spec(name).submodule_search_locations)))
        files[name]={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*.py'))}
    return digest(dict(files=files,dependencies=dependencies))


def snapshot_exports(root,stage):
    patterns={'prepare':['lasagna/**/*','LASAGNA.json','lasagna-source/IDENTITY.json','lasagna-source/WORK.json',
                         'lasagna-source/input.sqlite'],'fibres':['fibres/*.afv'],
              'field':[],'mesh':['meshes/*.obj'],
              'flatten':['flat/*.obj','flat/*.png','flat/atlas.sqlite']}
    with sqlite3.connect(root/'products.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS exports (path TEXT PRIMARY KEY,stage TEXT,sha TEXT)')
        for pattern in patterns[stage]:
            for path in root.glob(pattern):
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
        halo=40*stride
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
        lasagna_staging=lasagna_workspace=0
        if method!='classical':
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
        output_bytes=math.prod(shape.tolist())*60+math.prod(field['shape_zyx'])*40+lasagna_staging+lasagna_workspace
        if config.get('fibres'): output_bytes+=Path(config['fibres']).stat().st_size*4
        spline_bytes=0
        if method=='qual-v2':
            spline_bytes=math.prod(shape.tolist())*8+math.prod((shape//2).tolist())*18
            output_bytes+=math.prod(shape.tolist())*8+spline_bytes
        output=Path(config['output']); parent=output if output.exists() else output.parent
        written=sum(p.stat().st_size for name in ('maps','lasagna','lasagna-source','fibres','field','meshes','flat')
                    for p in (output/name).rglob('*') if p.is_file())
        remaining=0 if 'flatten' in stages else max(0,output_bytes-written)
        while not parent.exists():
            parent=parent.parent
        free=shutil.disk_usage(parent).free; reserve=config.get('reserve_bytes',1<<30)
        cache_free=shutil.disk_usage(volume.cache).free
        shared=os.stat(parent).st_dev==os.stat(volume.cache).st_dev
        fits=(ct_bytes<=volume.limit and ct_bytes+reserve<cache_free
              and remaining+reserve+(ct_bytes if shared else 0)<free)
        mode=config.get('ct_mode','auto')
        if mode=='auto':
            mode='download' if volume.remote and fits else 'stream' if volume.remote else 'local'
        if mode=='download' and not fits and volume.remote and not stages:
            raise ValueError('requested CT download exceeds cache or available-space budget; increase budget or use stream')
        return dict(source=volume.identity,prediction=method,roi_lo_zyx=lo.tolist(),roi_hi_zyx=hi.tolist(),shape_zyx=shape.tolist(),
                     observations=observations,field=field,
                    input_lo_zyx=input_lo.tolist(),input_hi_zyx=input_hi.tolist(),ct_policy=mode,
                     estimated_ct_bytes=ct_bytes,estimated_products_bytes=output_bytes,
                      estimated_spline_dense_bytes=spline_bytes,
                      estimated_lasagna_staging_bytes=lasagna_staging,estimated_lasagna_workspace_bytes=lasagna_workspace,
                    existing_products_bytes=written,estimated_remaining_bytes=remaining,free_bytes=free,
                     reserve_bytes=reserve,cache_free_bytes=cache_free,shared_filesystem=shared,
                     physical_accuracy='UNVALIDATED')
    finally:
        if own:
            volume.close()


def _validate_products(root,state):
    from .artifacts import Tiles
    from .lasagna import validate_retained as validate_lasagna
    from .afv import validate_retained as validate_afv
    validate_lasagna(root,state.get('config',{}))
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
        maps=Tiles(root/'maps')
        expected=json.loads((root/'PREPARED.json').read_bytes())['tiles_sha256']
        actual=maps.identity(); maps.close()
        if expected!=actual:
            raise ValueError('prepared maps changed')
    if 'fibres' in state['stages']:
        expected=json.loads((root/'FIBRES.json').read_bytes())
        if file_sha(root/'fibres/fibres.sqlite')!=expected['sha256']:
            raise ValueError('fibres changed')
        if ('probabilities_sha256' in expected
                and file_sha(root/'fibres/probabilities.npy')!=expected['probabilities_sha256']):
            raise ValueError('spline probabilities changed')
    if 'field' in state['stages']:
        from .field import open_field
        with open_field(root) as source:
            source.validate(hashes=True)
    if 'mesh' in state['stages']:
        from surface_mesh import MeshReader
        reader=MeshReader(root/'meshes'); reader.close()
    if 'flatten' in state['stages']:
        # Immutable chart payload hashes are checked by the normal stage resume.
        db=sqlite3.connect((root/'flat/atlas.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for name,sha in db.execute('SELECT name,sha FROM charts'):
                if file_sha(root/'flat'/(name+'.npz'))!=sha:
                    raise ValueError('flat chart changed')
        finally:
            db.close()


def run(config,resume=False,until='flatten'):
    validate_config(config)
    import torch
    from .predict import prepare
    from .fibres import trace
    from .field import solve
    from .meshes import mesh
    from .flatten import flatten
    from .exports import export_lasagna,export_afv
    if until not in STAGES:
        raise ValueError('invalid final stage')
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
        if config.get('fibres') or config.get('prediction','classical')!='classical':
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
                raise ValueError('implementation changed; resume with the original package or use a new output')
            (root/'STOP').unlink(missing_ok=True)
        else:
            state=dict(schema='winding-field-run-v1',version=__version__,config=config,stages={},
                       status='STARTING',physical_acceptance=False,created=time.time(),implementation_sha256=implementation_sha())
            atomic_json(root/'RUN.json',state)
        interrupted=False
        def handle(signum,frame):
            nonlocal interrupted
            interrupted=True
        previous={sig:signal.signal(sig,handle) for sig in (signal.SIGINT,signal.SIGTERM)}
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
                volume.validate_used(); _validate_products(root,state)
            estimate=plan(config,volume,state['stages']); state['plan']=estimate
            if 'flatten' not in state['stages'] and estimate['free_bytes']<estimate['estimated_remaining_bytes']+estimate['reserve_bytes']:
                raise OSError('insufficient disk space for products and reserve; choose explicit stride/ROI or another disk')
            stop=lambda:interrupted or (root/'STOP').exists()
            if estimate['ct_policy']=='download' and not state['stages']:
                low=np.array(estimate['input_lo_zyx']); high=np.array(estimate['input_hi_zyx']); chunks=np.array(volume.chunks)
                for key in itertools.product(*(range(a//c,(b-1)//c+1) for a,b,c in zip(low,high,chunks))):
                    a=np.array(key)*chunks; b=np.minimum(a+chunks,volume.shape)
                    volume.read(a,b)
                    if stop():
                        state['status']='PAUSED'; atomic_json(root/'RUN.json',state); return state
            for stage,receipt in zip(STAGES,RECEIPTS):
                if stage not in state['stages']:
                    state.update(status='RUNNING',active_stage=stage,updated=time.time())
                    atomic_json(root/'RUN.json',state)
                    print(f'[{stage}] started',file=sys.stderr,flush=True)
                    started=time.monotonic()
                    if stage=='prepare':
                        result=prepare(volume,root,config,stop)
                        if result['status']=='PREPARED':
                            if export_lasagna(root,stop) is None:
                                result=dict(status='PAUSED')
                    elif stage=='fibres':
                        result=trace(root,config,stop)
                        if result['status']=='FIBRES_READY':
                            sha=export_afv(root,stop)
                            if sha is None:
                                result=dict(status='PAUSED')
                            else:
                                result['afv_sha256']=sha; atomic_json(root/receipt,result)
                    elif stage=='field':
                        result=solve(root,config,stop)
                    elif stage=='mesh':
                        result=mesh(root,config,stop)
                    else:
                        result=flatten(root,config,volume if config.get('previews',True) else None,stop)
                    if result['status']=='PAUSED':
                        state['status']='PAUSED'; atomic_json(root/'RUN.json',state); return state
                    snapshot_exports(root,stage)
                    state['stages'][stage]=dict(receipt=receipt,sha256=file_sha(root/receipt),seconds=time.monotonic()-started)
                    state['io']=volume.stats()
                    atomic_json(root/'RUN.json',state)
                    print(f'[{stage}] ready ({state["stages"][stage]["seconds"]:.2f}s)',file=sys.stderr,flush=True)
                if stage==until:
                    break
            state.update(status='COMPLETE' if 'flatten' in state['stages'] else 'PAUSED',updated=time.time(),io=volume.stats())
            atomic_json(root/'RUN.json',state)
            if state['status']=='COMPLETE':
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
