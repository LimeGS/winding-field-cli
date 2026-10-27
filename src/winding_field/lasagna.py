"""Official Lasagna subprocess and physically scaled surface observations."""

import json
import hashlib
import itertools
from pathlib import Path
import shutil
import sqlite3
import subprocess

import numpy as np

from common import decode_lasagna_normals
from .io import Volume,bounds
from .models import checkpoint,LASAGNA_MODEL_SHA
from .splines import bind,run_child
from .util import atomic_bytes,atomic_json,file_sha

VILLA_REV='b29f310957c7b29e7d332f9876e0e487819cfaa3'
DEFAULT_PYTHON='~/.cache/winding-field/lasagna-venv/bin/python'
TILE=64
BORDER=16
CHUNK=32

_PROBE='''
import hashlib, importlib.metadata as m, json, sys
from pathlib import Path
if sys.version_info[:2] != (3,14):
    raise RuntimeError('official Lasagna runtime requires Python 3.14')
dist=m.distribution('vesuvius-lasagna')
direct=json.loads(dist.read_text('direct_url.json') or '{}')
expected='https://github.com/ScrollPrize/villa/archive/'+sys.argv[1]+'.tar.gz'
if direct.get('url') != expected or direct.get('subdirectory') != 'lasagna':
    raise RuntimeError('Lasagna must be installed from the pinned official Villa archive')
sources={}
for entry in sorted(dist.files or []):
    if entry.suffix in ('.py','.json','.so','.dylib'):
        path=Path(dist.locate_file(entry))
        with path.open('rb') as stream:
            sources[str(entry)]=hashlib.file_digest(stream,'sha256').hexdigest()
if not sources:
    raise RuntimeError('Lasagna runtime has no verifiable source inventory')
versions={d.metadata['Name']:d.version for d in m.distributions()}
print(json.dumps(dict(python=sys.version.split()[0],versions=versions,sources=sources,
                     revision=sys.argv[1],installation=direct)))
'''


def runtime_identity(python=None):
    python=Path(python or DEFAULT_PYTHON).expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f'Lasagna Python runtime missing: {python}; see docs/LASAGNA.md')
    try:
        result=subprocess.run([str(python),'-I','-c',_PROBE,VILLA_REV],
                              capture_output=True,text=True,timeout=120,check=True)
    except (OSError,subprocess.SubprocessError) as error:
        detail=getattr(error,'stderr',None) or str(error)
        raise ValueError('Lasagna runtime preflight failed: '+detail.strip()) from error
    try:
        identity=json.loads(result.stdout)
    except (ValueError,TypeError) as error:
        raise ValueError('invalid Lasagna runtime identity') from error
    identity['executable']=str(python)
    return identity


def decode_observations(grad_mag,nx,ny,cos,*,input_spacing_xyz,encode_scale,grad_mag_factor,validity):
    arrays=[np.asarray(a) for a in (grad_mag,nx,ny,cos)]
    shape=arrays[0].shape
    if len(shape)!=3 or any(a.shape!=shape for a in arrays) or validity.shape!=shape or validity.dtype!=np.bool_:
        raise ValueError('Lasagna products require matching ZYX arrays and boolean validity')
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Lasagna products must be finite')
    if any(np.any((a<0)|(a>255)) for a in arrays):
        raise ValueError('Lasagna encoded products must be in 0..255')
    spacing=np.asarray(input_spacing_xyz,float)
    if (spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any()
            or not np.isfinite([encode_scale,grad_mag_factor]).all() or min(encode_scale,grad_mag_factor)<=0):
        raise ValueError('Lasagna input spacing and density scales must be positive finite values')
    normals=decode_lasagna_normals(nx,ny)/spacing
    magnitude=np.linalg.norm(normals,axis=-1)
    density=np.asarray(grad_mag,dtype=np.float32)/encode_scale*grad_mag_factor*magnitude
    support=validity & (density>0) & (magnitude>0)
    normals/=np.maximum(magnitude[...,None],1e-12)
    normals[~support]=0; density[~support]=0
    phase=np.where(support,np.asarray(cos,dtype=np.float32)/255,0)
    return dict(density=density.astype(np.float32),normals=normals.astype(np.float32),
                support=support,surface_phase=phase.astype(np.float32))


def input_geometry(volume,geometry):
    stride=geometry['stride']; lo=np.asarray(geometry['roi_lo_zyx'])
    start=np.maximum(0,lo//stride-BORDER)
    end=np.minimum((np.asarray(volume.shape)-1-lo%stride)//stride+1,
                   lo//stride+geometry['shape_zyx']+BORDER)
    return dict(shape_zyx=(end-start).tolist(),source_start_zyx=(lo%stride+start*stride).tolist(),
                roi_offset_zyx=(lo//stride-start).tolist(),spacing_xyz=geometry['spacing_xyz'],stride=stride)


def _hashes(root):
    return {str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def _validate_stage(root):
    if not (root/'input.sqlite').exists(): return
    with sqlite3.connect((root/'input.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        for name,sha in db.execute('SELECT name,sha FROM chunks'):
            path=root/'input.zarr/0'/name
            if not path.is_file() or file_sha(path)!=sha:
                raise ValueError('Lasagna staged input changed or corrupt: '+name)


def _validate_work(root):
    seal=root/'WORK.json'; current=_hashes(root/'output')
    if seal.exists():
        if json.loads(seal.read_bytes())!=current:
            raise ValueError('Lasagna output changed or unsealed; use a new output')
    elif current:
        raise ValueError('Lasagna output is unsealed; use a new output')


def validate_retained(run,config):
    root=Path(run)/'lasagna-source'
    if not root.exists(): return
    identity=json.loads((root/'IDENTITY.json').read_bytes())
    if (identity['runtime']!=runtime_identity(config.get('lasagna_python'))
            or identity['worker_sha256']!=file_sha(Path(__file__).with_name('lasagna_worker.py'))):
        raise ValueError('Lasagna runtime changed')
    checkpoint(config.get('lasagna_model_path'),config.get('offline',False),prediction='lasagna')
    _validate_stage(root); _validate_work(root)
    receipt=Path(run)/'LASAGNA.json'
    if receipt.exists():
        saved=json.loads(receipt.read_bytes())
        if (saved['identity']!=identity or saved['products_sha256']!=file_sha(root/'WORK.json')
                or saved['input_sha256']!=file_sha(root/'input.sqlite')
                or saved['input_metadata_sha256']!={name:file_sha(root/name) for name in
                    ('input.zarr/.zgroup','input.zarr/0/.zarray')}):
            raise ValueError('Lasagna receipt or input inventory changed')


def _products(root,shape):
    manifest=json.loads((root/'output/surface.lasagna.json').read_bytes())
    if (manifest.get('version')!=2 or manifest.get('base_shape_zyx')!=shape
            or manifest.get('source_to_base')!=1 or manifest.get('grad_mag_encode_scale')!=1000.
            or manifest.get('grad_mag_factor')!=1.):
        raise ValueError('Lasagna output geometry or density encoding mismatch')
    paths={}
    for name in ('grad_mag','nx','ny','cos'):
        group=manifest['groups'][name]
        relative=f'surface_{name}.ome.zarr/0'
        if group!=dict(zarr=relative,scaledown=0,channels=[name]):
            raise ValueError('Lasagna output group mismatch: '+name)
        path=root/'output'/relative
        metadata=json.loads((path/'.zarray').read_bytes())
        if metadata['shape']!=shape or metadata['dtype']!='|u1':
            raise ValueError('Lasagna output array geometry mismatch')
        paths[name]=path
    return manifest,paths


def prepare(volume,run,config,geometry,stop=None):
    runtime=runtime_identity(config.get('lasagna_python'))
    weights=checkpoint(config.get('lasagna_model_path'),config.get('offline',False),prediction='lasagna')
    staged=input_geometry(volume,geometry)
    worker=Path(__file__).with_name('lasagna_worker.py')
    identity=dict(source=volume.identity,geometry=geometry,input=staged,runtime=runtime,
        checkpoint_sha256=LASAGNA_MODEL_SHA,worker_sha256=file_sha(worker),
        invalid_value=config.get('invalid_value'),tile=TILE,overlap=32,border=BORDER,
        device=config.get('device','cpu'),precision='float32-cpu/bfloat16-cuda',output_scaledown=1)
    run=Path(run); root=run/'lasagna-source'; root.mkdir(parents=True,exist_ok=True)
    bind(root/'IDENTITY.json',identity)
    validate_retained(run,config)
    if (run/'LASAGNA.json').exists(): return json.loads((run/'LASAGNA.json').read_bytes())
    shape=staged['shape_zyx']; array=root/'input.zarr/0'
    bind(array/'.zarray',dict(zarr_format=2,shape=shape,chunks=[CHUNK]*3,dtype=volume.dtype.str,
         compressor=None,fill_value=0,filters=None,order='C',dimension_separator='.'))
    bind(root/'input.zarr/.zgroup',dict(zarr_format=2))
    with sqlite3.connect(root/'input.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS chunks (name TEXT PRIMARY KEY,sha TEXT)')
        for key in itertools.product(*(range((n-1)//CHUNK+1) for n in shape)):
            if stop and stop(): return dict(status='PAUSED')
            start=np.asarray(key)*CHUNK; end=np.minimum(start+CHUNK,shape)
            axes=[staged['source_start_zyx'][i]+np.arange(a,b)*staged['stride'] for i,(a,b) in enumerate(zip(start,end))]
            ct=volume.sample(axes)
            if not np.isfinite(ct).all(): raise ValueError('Lasagna requires finite CT')
            chunk=np.zeros((CHUNK,)*3,volume.dtype)
            chunk[tuple(slice(0,n) for n in end-start)]=ct
            data=chunk.tobytes(); sha=hashlib.sha256(data).hexdigest(); name='.'.join(map(str,key))
            row=db.execute('SELECT sha FROM chunks WHERE name=?',(name,)).fetchone()
            if row:
                if row!=(sha,): raise ValueError('Lasagna source CT changed')
                continue
            if shutil.disk_usage(root).free<len(data)+config.get('reserve_bytes',0):
                raise OSError('insufficient disk space for Lasagna input and reserve')
            atomic_bytes(array/name,data)
            db.execute('INSERT INTO chunks VALUES (?,?)',(name,sha)); db.commit()
    if stop and stop(): return dict(status='PAUSED')
    output=root/'output'; output.mkdir(exist_ok=True)
    command=[runtime['executable'],'-I',str(worker.resolve()),'predict3d',
        '--input',str(array.resolve()),'--output',str((output/'surface.lasagna.json').resolve()),
        '--unet-checkpoint',str(Path(weights).resolve()),'--tile-size',str(TILE),'--overlap','32',
        '--border',str(BORDER),'--cos-scaledown','1','--scaledown','1','--device',config.get('device','cpu'),
        '--input-reader','python-zarr','--prefetch-workers','1','--prefetch-tiles-per-gpu','1',
        '--flush-workers','0','--accumulator-workers','0','--input-cache-gib','0',
        '--no-download','--levels','2','--ome-chunk',str(CHUNK),'--ome-compressor','none']
    atomic_json(root/'WORK.json',_hashes(output))
    try:
        complete=run_child(command,root/'inference.log',stop,config.get('reserve_bytes',0),config.get('threads',4),label='Lasagna')
    finally:
        # shortcut: a hard-killed parent leaves unverifiable work; use a new run in that case.
        atomic_json(root/'WORK.json',_hashes(output))
    if not complete: return dict(status='PAUSED')
    manifest,paths=_products(root,shape)
    # Read every full-resolution product chunk before sealing the ready receipt.
    for path in paths.values():
        product=Volume(path,voxel_um=1.,cache=root/'reader-cache',offline=True)
        try:
            for key in itertools.product(*(range((n-1)//CHUNK+1) for n in shape)):
                lo=np.asarray(key)*CHUNK; product.read(lo,np.minimum(lo+CHUNK,shape))
        finally: product.close()
    result=dict(status='LASAGNA_READY',identity=identity,input_sha256=file_sha(root/'input.sqlite'),
        input_metadata_sha256={name:file_sha(root/name) for name in ('input.zarr/.zgroup','input.zarr/0/.zarray')},
        products_sha256=file_sha(root/'WORK.json'),encoding=manifest,
        density_units='cycles/micrometer',surface_phase='Lasagna cos encoded 0..255, scaled to 0..1')
    atomic_json(run/'LASAGNA.json',result)
    return result


def read_observations(run,geometry,lo,hi):
    root=Path(run)/'lasagna-source'; receipt=json.loads((Path(run)/'LASAGNA.json').read_bytes())
    identity=receipt['identity']
    if geometry!=identity['geometry']: raise ValueError('Lasagna observation geometry changed')
    lo,hi=bounds(lo,hi,geometry['shape_zyx']); staged=identity['input']
    lo=lo+staged['roi_offset_zyx']; hi=hi+staged['roi_offset_zyx']
    manifest,paths=_products(root,staged['shape_zyx']); values={}
    for name,path in dict(paths,ct=root/'input.zarr/0').items():
        product=Volume(path,voxel_um=1.,cache=root/'reader-cache',offline=True)
        try: values[name]=product.read(lo,hi)
        finally: product.close()
    validity=np.ones(values['ct'].shape,bool)
    if identity['invalid_value'] is not None: validity &= values['ct']!=identity['invalid_value']
    result=decode_observations(*(values[n] for n in ('grad_mag','nx','ny','cos')),
        input_spacing_xyz=staged['spacing_xyz'],encode_scale=manifest['grad_mag_encode_scale'],
        grad_mag_factor=manifest['grad_mag_factor'],validity=validity)
    result['ct']=values['ct']
    return result
