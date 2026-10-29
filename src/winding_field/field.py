"""Automatic observation orientation, bounded fibre constraints and core solves."""

import io
import itertools
import json
import math
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np

from block_solver import solve_blocks
from field_constraints import pair_constraints
from field_store import BlockStore
from .surface_sources import FieldSource
from .artifacts import Tiles
from .config import field_geometry
from .util import atomic_json, digest, file_sha


def calibration(maps,meta):
    matrix=np.zeros((3,3)); centre=np.zeros(3); count=0; density_sum=0.
    for key in maps.keys():
        data=maps.load(key); valid=data['support']; normal=data['normals'][valid].astype(float)
        matrix+=normal.T@normal
        lo,_=maps.box(key)
        coords=np.argwhere(valid)+lo
        centre+=(coords[:,::-1]*meta['spacing_xyz']+meta['origin_xyz']).sum(0)
        count+=len(coords); density_sum+=float(data['density'][valid].sum(dtype=np.float64))
    if count==0 or density_sum<=0:
        raise ValueError('no supported sheets in the requested CT domain')
    eigenvalues,vectors=np.linalg.eigh(matrix)
    axis=vectors[:,-1]; axis*=1 if axis[np.argmax(abs(axis))]>=0 else -1
    concentration=float(eigenvalues[-1]/max(eigenvalues.sum(),1e-12))
    return dict(mode='planar' if concentration>.65 else 'radial-z',axis_xyz=axis.tolist(),
                centre_xyz=(centre/count).tolist(),mean_density_per_um=density_sum/count,
                orientation_concentration=concentration,supported_voxels=count,
                physical_accuracy='UNVALIDATED',topology='local chart; global sheet continuity unverified')


def coordinates(lo,hi,meta):
    z,y,x=np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)),indexing='ij')
    return np.stack((x,y,z),axis=-1)*meta['spacing_xyz']+meta['origin_xyz']


def oriented(data,xyz,cal):
    direction=np.asarray(cal['axis_xyz']) if cal['mode']=='planar' else xyz-cal['centre_xyz']
    if cal['mode']!='planar':
        direction[...,2]=0
        direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-12)
    dot=np.sum(data['normals']*direction,axis=-1)
    valid=data['support'] & (abs(dot)>.15)
    target=(data['normals']*np.where(dot>=0,1.,-1.)[...,None]*data['density'][...,None]).astype(np.float32)
    target[~valid]=0
    weights=np.where(valid,1.,1e-3).astype(np.float32)
    return target,weights,valid


def initial_field(xyz,meta,cal):
    if cal['mode']=='planar':
        return np.sum((xyz-meta['origin_xyz'])*cal['axis_xyz'],axis=-1)*cal['mean_density_per_um']
    radius=np.linalg.norm((xyz-cal['centre_xyz'])[...,:2],axis=-1)
    first=np.linalg.norm((np.array(meta['origin_xyz'])-cal['centre_xyz'])[:2])
    return (radius-first)*cal['mean_density_per_um']


def map_axes(meta,geometry,lo,hi):
    return [np.clip((geometry['origin_xyz'][2-i]-meta['origin_xyz'][2-i]
                     +np.arange(a,b)*geometry['spacing_xyz'][2-i])/meta['spacing_xyz'][2-i],
                    0,meta['shape_zyx'][i]-1) for i,(a,b) in enumerate(zip(lo,hi))]


def sample_prior(maps,meta,geometry,cal,lo,hi):
    names=('normals','density','support')
    if all(geometry[k]==meta[k] for k in ('origin_xyz','spacing_xyz','shape_zyx')):
        return oriented(maps.read_many(names,lo,hi),coordinates(lo,hi,meta),cal)
    def transform(data,start):
        target,weights,valid=oriented(data,coordinates(start,start+data['support'].shape,meta),cal)
        return dict(target=target,weights=weights,support=valid)
    data=maps.sample(names,map_axes(meta,geometry,lo,hi),transform)
    valid=data['support']; data['target'][~valid]=0; data['weights'][~valid]=1e-3
    return data['target'],data['weights'],valid


def open_field(run):
    run=Path(run); meta=json.loads((run/'FIELD.json').read_bytes())
    return FieldSource(dict(kind='chunks',path=str((run/'field/CHECKPOINT.json').resolve()),
                            sha256=meta['checkpoint_sha256'],support_policy='observed'),
                       meta['geometry'],cache_bytes=64<<20)


def spline_samples(points,low,high,step):
    """Clip each supported edge, then sample inside the block, including free nodes."""
    start=points[:-1]; delta=np.diff(points,axis=0)
    near=np.divide(low-start,delta,out=np.full_like(start,-np.inf),where=delta!=0)
    far=np.divide(high-start,delta,out=np.full_like(start,np.inf),where=delta!=0)
    enter=np.maximum(0,np.minimum(near,far).max(1))
    leave=np.minimum(1,np.maximum(near,far).min(1))
    valid=(leave>enter)&np.all((delta!=0)|((start>=low)&(start<=high)),axis=1)
    for i in np.flatnonzero(valid):
        a=np.clip(start[i]+enter[i]*delta[i],low,high)
        b=np.clip(start[i]+leave[i]*delta[i],low,high)
        length=np.linalg.norm(b-a)
        if length:
            yield np.linspace(a,b,max(1,int(np.ceil(length/step)))+1)


def solve(run,config,stop=None):
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes())
    fibres=json.loads((run/'FIBRES.json').read_bytes())
    maps=Tiles(run/'maps')
    if maps.identity()!=meta['tiles_sha256'] or file_sha(run/'fibres/fibres.sqlite')!=fibres['sha256']:
        raise ValueError('prepared maps or fibres changed')
    cal=calibration(maps,meta)
    atomic_json(run/'CALIBRATION.json',cal)
    geometry=field_geometry(meta,config.get('field_spacing_um'))
    geometry.update(voxel_um=1.,sheet_step=1.)
    shape=tuple(geometry['shape_zyx']); chunk=config.get('field_chunk',64)
    if type(chunk) is not int or not 4<=chunk<=256:
        raise ValueError('field chunk must be 4..256')
    identity=dict(maps=meta['tiles_sha256'],fibres=fibres['sha256'],geometry=geometry,calibration=cal,
                   fibre_weight=config.get('fibre_weight',1.),algorithm='physical-trilinear-prior-v2')
    root=run/'field'
    options=dict(origin_xyz=geometry['origin_xyz'],spacing_xyz=geometry['spacing_xyz'],
                 order=list(itertools.product(*(range((n-1)//chunk+1) for n in shape))),
                 halo=2,sweeps=config.get('sweeps',4),rtol=config.get('rtol',1e-6),
                 maxiter=config.get('maxiter',2000),dtype='float64',device=config.get('device','cpu'),
                 cache_bytes=64<<20,reserve_bytes=config.get('reserve_bytes',0))
    db=sqlite3.connect((run/'fibres/fibres.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    row_count=0

    def load(lo,hi):
        target,weights,valid=sample_prior(maps,meta,geometry,cal,lo,hi)
        return target,weights,valid,{'maps':meta['tiles_sha256']}

    def constraints(lo,hi,read_nodes):
        nonlocal row_count
        a=[]; b=[]
        if meta['prediction']=='qual-v2' or fibres.get('spatial_index'):
            low=np.asarray(geometry['origin_xyz'])+lo[::-1]*geometry['spacing_xyz']
            high=np.asarray(geometry['origin_xyz'])+(hi-1)[::-1]*geometry['spacing_xyz']
            rows=db.execute('SELECT c.points FROM curve_bounds b JOIN curves c ON c.id=b.id '
                            'WHERE b.max_x>=? AND b.min_x<=? AND b.max_y>=? AND b.min_y<=? '
                            'AND b.max_z>=? AND b.min_z<=? ORDER BY c.id',tuple(v for pair in zip(low,high) for v in pair))
        else:
            axes=map_axes(meta,geometry,lo,hi)
            keys=itertools.product(*(range(int(axis[0])//maps.chunk,int(axis[-1])//maps.chunk+1) for axis in axes))
            rows=(row for key in keys for row in db.execute('SELECT points FROM curves WHERE tile=?',('_'.join(map(str,key)),)))
        for (raw,) in rows:
            points=np.load(io.BytesIO(raw),allow_pickle=False)
            if meta['prediction']=='qual-v2' or fibres.get('spatial_index'):
                for sampled in spline_samples(points,low,high,min(geometry['spacing_xyz'])):
                    a.extend(sampled[:-1]); b.extend(sampled[1:])
                continue
            points=points[::3]
            if len(points)<2:
                continue
            grid=(points-geometry['origin_xyz'])/geometry['spacing_xyz']
            inside=((grid>=lo[::-1]) & (grid<=(hi-1)[::-1])).all(1)
            valid=inside[:-1]&inside[1:]
            a.extend(points[:-1][valid]); b.extend(points[1:][valid])
        if not a:
            return None
        row_count+=len(a)
        return pair_constraints(np.array(a),np.array(b),
                np.array(geometry['origin_xyz'])+lo[::-1]*geometry['spacing_xyz'],geometry['spacing_xyz'],tuple(hi-lo),
                weight=identity['fibre_weight'])

    options['constraints']=constraints
    try:
        checkpoint=root/'CHECKPOINT.json'
        if not checkpoint.exists():
            # Publish only a complete initial checkpoint; failed admission is retryable.
            with TemporaryDirectory(prefix='.field-init-',dir=run) as temporary:
                initial=Path(temporary)/'field'
                solve_blocks(initial,shape,chunk,identity,load,stop=lambda:True,**options)
                initial.rename(root)
        saved=json.loads(checkpoint.read_bytes())
        if not saved['progress'].get('initialization_done') and not saved['progress'].get('solver'):
            with BlockStore(root,shape,chunk,saved['identity'],resume=True,cache_bytes=64<<20,
                            reserve_bytes=options['reserve_bytes']) as store:
                progress=store.manifest['progress']
                for key in options['order']:
                    if stop and stop():
                        return dict(status='PAUSED')
                    if '-'.join(map(str,key)) in store.manifest['chunks']:
                        continue
                    lo=np.array(key)*chunk; hi=np.minimum(lo+chunk,shape)
                    _,_,valid,_=load(lo,hi)
                    store.commit(key,initial_field(coordinates(lo,hi,geometry),geometry,cal),progress,valid)
                progress['initialization_done']=True
                store.save_progress(progress)
        result=solve_blocks(root,shape,chunk,identity,load,resume=True,stop=stop,**options)
        if result['status']!='SWEEP_BUDGET_REACHED':
            return dict(status='PAUSED',solver=result)
        receipt=dict(status='FIELD_READY',geometry=geometry,checkpoint_sha256=file_sha(checkpoint),
                     calibration=cal,solver=result,global_converged=False,physical_acceptance=False,
                     fibre_constraint_rows=row_count,numerical_status='local solves converged; sweep budget exhausted')
        atomic_json(run/'FIELD.json',receipt)
        return receipt
    finally:
        maps.close(); db.close()
