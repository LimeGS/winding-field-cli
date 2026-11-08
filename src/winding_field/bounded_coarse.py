"""Recomputable native observations, durable calibration and oriented coarse tiles."""

import json
from pathlib import Path
import shutil

import numpy as np

from .artifacts import Tiles
from .field import finish_calibration,oriented,coordinates,map_axes
from .splines import bind
from .util import atomic_json


def endpoint_grid(meta,pitch):
    if not np.isfinite(pitch) or pitch<=0:raise ValueError('positive finite coarse pitch required')
    extent=(np.asarray(meta['shape_zyx'])[::-1]-1)*meta['spacing_xyz']
    if not np.isfinite(extent).all() or np.any(extent<=0):raise ValueError('invalid native physical extent')
    intervals=np.maximum(1,np.ceil(extent/pitch)).astype(np.int64)
    return dict(meta,shape_zyx=(intervals+1)[::-1].tolist(),spacing_xyz=(extent/intervals).tolist())


class NativeMaps(Tiles):
    """Use the existing tile sampler without retaining native prediction maps."""
    def __init__(self,shape,read,*,chunk=32):
        self.shape=tuple(shape);self.chunk=chunk;self.reader=read
        if len(self.shape)!=3 or any(type(n) is not int or n<1 for n in self.shape):
            raise ValueError('invalid native map shape')
        if type(chunk) is not int or not 2<=chunk<=64:raise ValueError('native map chunks must be 2..64')

    def load(self,key,names=None):
        lo,hi=self.box(key);data=self.reader(lo,hi)
        for name,tail in [('normals',(3,)),('density',()),('support',())]:
            value=data[name]
            if (value.shape!=(*tuple(hi-lo),*tail) or not np.isfinite(value).all()
                    or value.dtype!=np.dtype(bool if name=='support' else 'f4')):
                raise ValueError('invalid native surface observation: '+name)
        if np.any(data['density']<0):raise ValueError('negative sheet density')
        return data if names is None else {name:data[name] for name in names}

    def close(self):pass


def produce(root,native,meta,grid,identity,*,chunk=32,stop=None,reserve_bytes=0,quota_bytes=3<<30,calibration=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    calibration_maps,calibration_meta=calibration or (native,meta)
    bind(root/'IDENTITY.json',dict(source=identity,native=meta,grid=grid,chunk=chunk,
                                 native_shape=list(native.shape),native_chunk=native.chunk,
                                  native_order='zyx-lexicographic-v1',
                                  calibration_geometry=calibration_meta,calibration_chunk=calibration_maps.chunk,
                                 algorithm='native-statistics-orient-before-coarse-phase-v2'))
    progress=root/'calibration-progress.json';cal_path=root/'CALIBRATION.json'
    if not cal_path.exists():
        saved=json.loads(progress.read_bytes()) if progress.exists() else dict(
            next_tile=0,matrix=np.zeros((3,3)).tolist(),centre=[0.,0.,0.],count=0,density_sum=0.)
        matrix=np.asarray(saved['matrix']);centre=np.asarray(saved['centre'])
        count=saved['count'];density_sum=saved['density_sum']
        for index,key in enumerate(calibration_maps.keys()):
            if stop and stop():return 'PAUSED'
            if index<saved['next_tile']:continue
            data=calibration_maps.load(key);valid=data['support'];normal=data['normals'][valid].astype(float)
            matrix+=normal.T@normal;lo,_=calibration_maps.box(key)
            coords=np.argwhere(valid)+lo
            centre+=(coords[:,::-1]*calibration_meta['spacing_xyz']+calibration_meta['origin_xyz']).sum(0)
            count+=len(coords);density_sum+=float(data['density'][valid].sum(dtype=np.float64))
            atomic_json(progress,dict(next_tile=index+1,matrix=matrix.tolist(),centre=centre.tolist(),
                                      count=count,density_sum=density_sum))
        atomic_json(cal_path,finish_calibration(matrix,centre,count,density_sum))
    cal=json.loads(cal_path.read_bytes())
    maps=Tiles(root/'maps',grid['shape_zyx'],chunk,dict(source=identity,grid=grid,calibration=cal))
    try:
        retained=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        for key in maps.keys():
            if stop and stop():return 'PAUSED'
            if maps.has(key):continue
            lo,hi=maps.box(key)
            needed=int(np.prod(hi-lo))*21+65536
            if retained+2*needed>quota_bytes or shutil.disk_usage(root).free<reserve_bytes+2*needed:
                raise OSError('coarse observation quota/reserve exhausted')
            def transform(data,start):
                target,weights,valid=oriented(data,coordinates(start,start+data['support'].shape,meta),cal)
                return dict(target=target,weights=weights,support=valid,surface_phase=data['surface_phase'])
            data=native.sample(('normals','density','support','surface_phase'),map_axes(meta,grid,lo,hi),transform)
            valid=data['support'];data['target'][~valid]=0;data['weights'][~valid]=1e-3
            maps.write(key,data)
            retained+=maps.path(key).stat().st_size
        atomic_json(root/'COARSE.json',dict(geometry=grid,calibration=cal,tiles_sha256=maps.identity(),
                                          status='COMPLETE',physical_acceptance=False))
        return 'COMPLETE'
    finally:maps.close()
