"""Tiled CT preparation with explicit physical units and model provenance."""

import hashlib
from functools import lru_cache
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from skimage.filters import threshold_otsu

from field_observations import density_normals
from .artifacts import Tiles
from .config import validate_config,validate_observations
from .io import bounds
from .models import SD2,QualV2,model_sha
from .util import atomic_json, digest,file_sha


def observations(ct,spacing_xyz,*,threshold=None,presence=None,tangents=None,validity=None,device='cpu'):
    if ct.ndim!=3 or not np.isfinite(ct).all():
        raise ValueError('finite 3D CT required')
    spacing=np.asarray(spacing_xyz,float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('positive XYZ spacing required')
    if validity is not None and (validity.shape!=ct.shape or validity.dtype!=np.bool_):
        raise ValueError('validity must be a boolean CT-shaped mask')
    if presence is None:
        threshold=float(threshold_otsu(ct)) if threshold is None else threshold
        presence=(ct>threshold).astype(np.float32)
    if validity is not None:
        presence=np.where(validity,presence,0)
    density,normal,support=density_normals(np.rint(presence*255).astype(np.uint8),halo=0,device=device)
    normal=normal.cpu().numpy()/spacing
    magnitude=np.linalg.norm(normal,axis=-1)
    normal=normal/np.maximum(magnitude[...,None],1e-12)
    density=density.cpu().numpy()*magnitude
    support=support.cpu().numpy() & (magnitude>1e-6) & (density>1e-7)
    if validity is not None:
        # The core's derivative/tensor/crossing/density recipe fits inside halo 40.
        support &= minimum_filter(validity,size=81,mode='nearest')
    normal[~support]=0; density[~support]=0
    if tangents is None:
        gradients=np.stack([gaussian_filter(ct.astype(np.float32),1,order=tuple(int(j==i) for j in range(3)))
                            /spacing[2-i] for i in range(3)],axis=-1)
        tensor=np.empty((*ct.shape,3,3),np.float32)
        for i in range(3):
            for j in range(i,3):
                tensor[...,i,j]=tensor[...,j,i]=gaussian_filter(gradients[...,i]*gradients[...,j],2)
        eigenvalues,eigenvectors=np.linalg.eigh(tensor)
        tangents=eigenvectors[..., :,0][...,::-1].copy()
        reliable=(eigenvalues[...,1]-eigenvalues[...,0])>np.maximum(eigenvalues[...,2]*.005,1e-6)
        tangents[~reliable]=0
    else:
        tangents=np.asarray(tangents,dtype=np.float32)*spacing
    tangents=tangents-(tangents*normal).sum(-1)[...,None]*normal
    tangents/=np.maximum(np.linalg.norm(tangents,axis=-1,keepdims=True),1e-12)
    tangents[~support]=0
    return dict(density=density.astype(np.float32),normals=normal.astype(np.float32),
                support=support,presence=presence.astype(np.float32),tangents=tangents.astype(np.float32))


def prepare(volume,run,config,stop=None):
    validate_config(config)
    run=Path(run); run.mkdir(parents=True,exist_ok=True)
    lo,hi=bounds(config.get('roi_lo',[0,0,0]),config.get('roi_hi',list(volume.shape)),volume.shape)
    stride=config.get('stride',1); chunk=config.get('chunk',32)
    if type(stride) is not int or not 1<=stride<=8 or type(chunk) is not int or not 4<=chunk<=128:
        raise ValueError('stride must be 1..8 and chunk 4..128')
    method='external-afv' if config.get('fibres') else config.get('prediction','classical')
    surface=method!='classical'
    halo=40
    core_size=48 if method=='qual-v2' else 32
    shape=tuple(((hi-lo-1)//stride+1).tolist())
    if min(shape)<2:
        raise ValueError('ROI needs at least two output samples per axis')
    validate_observations(config,volume,shape)
    spacing=(np.array(volume.spacing_xyz)*stride).tolist()
    origin=(np.array(volume.origin_xyz)+lo[::-1]*volume.spacing_xyz).tolist()
    invalid=config.get('invalid_value')
    geometry=dict(shape_zyx=list(shape),spacing_xyz=spacing,origin_xyz=origin,units='micrometer',
                  frame=digest(volume.identity),roi_lo_zyx=lo.tolist(),roi_hi_zyx=hi.tolist(),
                  stride=stride,input_halo_voxels=halo*stride,
                  validity=dict(invalid_value=invalid,filter_radius=40))
    if surface:
        from . import lasagna
        geometry['validity']['filter_radius']=0
        geometry['input_halo_voxels']=(lasagna.BORDER if method=='external-afv' else halo)*stride
        ready=lasagna.prepare(volume,run,config,geometry,stop)
        if ready['status']=='PAUSED': return ready
    threshold=config.get('threshold')
    if method=='classical' and threshold is None:
        threshold_path=run/'THRESHOLD.json'
        binding=dict(source=volume.identity,geometry=geometry,recipe='otsu-27-patches-16-v1')
        if threshold_path.exists():
            saved=json.loads(threshold_path.read_bytes())
            if saved['identity']!=binding:
                raise ValueError('threshold calibration identity changed')
            threshold=saved['threshold']
        else:
            # shortcut: fixed ROI patches may miss rare intensities; use a calibrated --threshold when needed.
            samples=[]
            starts=[np.unique(np.linspace(0,max(n-16,0),3,dtype=int)) for n in shape]
            for start in itertools.product(*starts):
                indices=[lo[i]+np.arange(a,min(a+16,shape[i]))*stride for i,a in enumerate(start)]
                samples.append(volume.sample(indices).ravel())
                if stop and stop():
                    return dict(status='PAUSED')
            values=np.concatenate(samples)
            if not np.isfinite(values).all():
                raise ValueError('finite CT required for threshold calibration')
            if invalid is not None:
                values=values[values!=invalid]
            if not len(values):
                raise ValueError('no valid CT samples for threshold calibration')
            threshold=float(threshold_otsu(values))
            atomic_json(threshold_path,dict(identity=binding,threshold=threshold,samples=int(values.size)))
    identity=dict(source=volume.identity,geometry=geometry,method=method,model_sha256=None if method=='external-afv' else model_sha(method),
                   threshold=threshold,recipe='fractional-crossings-validity-v2',
                    inference=dict(core=core_size,halo=40,anchor='selected-source sampling lattice',
                                   phase_zyx=(lo%stride).tolist(),normalization='tile-sample-zscore-floor10' if method=='qual-v2' else 'tile-zscore',
                                    padding='edge') if method in ('qual-v2','sd2') else None)
    if surface:
        identity.update(surface_method='lasagna',lasagna_sha256=file_sha(run/'LASAGNA.json'),recipe='independent-lasagna-surface-v1')
    maps=Tiles(run/'maps',shape=shape,chunk=chunk,identity=identity)
    model_type={'sd2':SD2,'qual-v2':QualV2}.get(method)
    model=model_type(config.get('model_path'),config.get('device','cpu'),config.get('offline',False)) if model_type else None
    def infer(ct):
        result=model.predict(ct) if model else (None,None)
        if surface:
            valid=np.ones(ct.shape,bool) if invalid is None else ct!=invalid
            tangent=np.asarray(result[1],np.float32)*spacing
            tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
            arrays=dict(presence=np.where(valid,result[0],0).astype(np.float32),
                        tangents=np.where(valid[...,None],tangent,0).astype(np.float32),
                        fibre_support=valid & (result[0]>0))
            if method=='qual-v2': arrays['probabilities']=result[2]
            return arrays
        arrays=observations(ct,spacing,threshold=threshold,presence=result[0],tangents=result[1],
                             validity=ct!=invalid if invalid is not None else None,device=config.get('device','cpu'))
        if method=='qual-v2':
            arrays['probabilities']=result[2]
        return arrays

    @lru_cache(maxsize=8)
    def canonical(key,compute):
        start=np.asarray(key)*core_size
        indices=[np.clip(lo[i]%stride+np.arange(start[i]-halo,start[i]+core_size+halo)*stride,
                         0,volume.shape[i]-1) for i in range(3)]
        ct=volume.sample(indices)
        sha=hashlib.sha256(ct.tobytes()).hexdigest()
        arrays={}
        if compute:
            core=(slice(halo,halo+core_size),)*3
            arrays={name:value[core].copy() for name,value in infer(ct).items()}
            arrays['ct']=ct[core].copy()
        return arrays,sha

    completed=0
    for key in maps.keys():
        if stop and stop():
            maps.close(); return dict(status='PAUSED',new_tiles=completed)
        start,end=maps.box(key)
        existing=maps.has(key)
        if method=='external-afv':
            sha=digest([identity['lasagna_sha256'],key])
            arrays={}
        elif model:
            a,b=lo//stride+start,lo//stride+end
            arrays={}; hashes=[]
            for tile in itertools.product(*(range(x//core_size,(y-1)//core_size+1) for x,y in zip(a,b))):
                values,ct_sha=canonical(tile,not existing); hashes.append([tile,ct_sha])
                if not existing:
                    low=np.maximum(a,np.array(tile)*core_size); high=np.minimum(b,np.array(tile)*core_size+core_size)
                    src=tuple(slice(int(x),int(y)) for x,y in zip(low-np.array(tile)*core_size,high-np.array(tile)*core_size))
                    dst=tuple(slice(int(x),int(y)) for x,y in zip(low-a,high-a))
                    for name,value in values.items():
                        if name not in arrays:
                            arrays[name]=np.empty((*tuple(end-start),*value.shape[3:]),value.dtype)
                        arrays[name][dst]=value[src]
                if stop and stop():
                    maps.close(); return dict(status='PAUSED',new_tiles=completed)
            sha=digest(hashes)
        else:
            indices=[np.clip(lo[i]+np.arange(start[i]-halo,end[i]+halo)*stride,0,volume.shape[i]-1) for i in range(3)]
            ct=volume.sample(indices)
            sha=hashlib.sha256(ct.tobytes()).hexdigest()
            if not existing:
                core=tuple(slice(halo,halo+int(n)) for n in end-start)
                arrays={name:value[core].copy() for name,value in infer(ct).items()}
                arrays['ct']=ct[core].copy()
        if existing:
            if maps.info(key)['ct_sha256']!=sha:
                raise ValueError('source CT changed since preparation')
            continue
        if surface:
            arrays.update(lasagna.read_observations(run,geometry,start,end))
        maps.write(key,arrays,dict(ct_sha256=sha))
        completed+=1
        if stop and stop():
            maps.close(); return dict(status='PAUSED',new_tiles=completed)
    binding=maps.identity()
    meta=dict(geometry,identity=identity,tiles_sha256=binding,prediction=method,
              surface_method='lasagna' if surface else 'classical',fibre_method=method,
              source=volume.identity,model_sha256=identity['model_sha256'],
              status='PREPARED',physical_accuracy='UNVALIDATED',
              method_note=('official Lasagna surfaces with '+method+' fibres') if surface else 'classical CT threshold/structure-tensor baseline')
    atomic_json(run/'PREPARED.json',meta)
    maps.close()
    return meta
