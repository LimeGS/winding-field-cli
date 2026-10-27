"""Tiled CT preparation with explicit physical units and model provenance."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from ._core.field_observations import density_normals
from .artifacts import Tiles
from .io import bounds
from .models import SD2, MODEL_SHA
from .util import atomic_json, digest


def observations(ct,spacing_xyz,*,threshold=None,presence=None,tangents=None,device='cpu'):
    if ct.ndim!=3 or not np.isfinite(ct).all():
        raise ValueError('finite 3D CT required')
    spacing=np.asarray(spacing_xyz,float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('positive XYZ spacing required')
    if presence is None:
        threshold=float(threshold_otsu(ct)) if threshold is None else threshold
        presence=(ct>threshold).astype(np.float32)
    density,normal,support=density_normals(np.rint(presence*255).astype(np.uint8),halo=0,device=device)
    normal=normal.cpu().numpy()/spacing
    magnitude=np.linalg.norm(normal,axis=-1)
    normal=normal/np.maximum(magnitude[...,None],1e-12)
    density=density.cpu().numpy()*magnitude
    support=support.cpu().numpy() & (magnitude>1e-6) & (density>1e-7)
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
    run=Path(run); run.mkdir(parents=True,exist_ok=True)
    lo,hi=bounds(config.get('roi_lo',[0,0,0]),config.get('roi_hi',list(volume.shape)),volume.shape)
    stride=config.get('stride',1); chunk=config.get('chunk',32)
    if type(stride) is not int or not 1<=stride<=8 or type(chunk) is not int or not 4<=chunk<=128:
        raise ValueError('stride must be 1..8 and chunk 4..128')
    method=config.get('prediction','classical')
    if method not in ('classical','sd2'):
        raise ValueError('prediction must be classical or sd2')
    halo=40
    shape=tuple(((hi-lo-1)//stride+1).tolist())
    if min(shape)<2:
        raise ValueError('ROI needs at least two output samples per axis')
    spacing=(np.array(volume.spacing_xyz)*stride).tolist()
    origin=(np.array(volume.origin_xyz)+lo[::-1]*volume.spacing_xyz).tolist()
    geometry=dict(shape_zyx=list(shape),spacing_xyz=spacing,origin_xyz=origin,units='micrometer',
                  frame=digest(volume.identity),roi_lo_zyx=lo.tolist(),roi_hi_zyx=hi.tolist(),
                  stride=stride,input_halo_voxels=halo*stride)
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
            threshold=float(threshold_otsu(values))
            atomic_json(threshold_path,dict(identity=binding,threshold=threshold,samples=int(values.size)))
    identity=dict(source=volume.identity,geometry=geometry,method=method,model_sha256=MODEL_SHA if method=='sd2' else None,
                  threshold=threshold,recipe='fractional-crossings-v1')
    maps=Tiles(run/'maps',shape=shape,chunk=chunk,identity=identity)
    model=SD2(config.get('model_path'),config.get('device','cpu'),config.get('offline',False)) if method=='sd2' else None
    completed=0
    for key in maps.keys():
        start,end=maps.box(key)
        indices=[np.clip(lo[i]+np.arange(start[i]-halo,end[i]+halo)*stride,0,volume.shape[i]-1) for i in range(3)]
        ct=volume.sample(indices)
        sha=hashlib.sha256(ct.tobytes()).hexdigest()
        if maps.has(key):
            if maps.info(key)['ct_sha256']!=sha:
                raise ValueError('source CT changed since preparation')
            continue
        presence,tangent=model.predict(ct) if model else (None,None)
        arrays=observations(ct,spacing,threshold=threshold,presence=presence,tangents=tangent,
                            device=config.get('device','cpu'))
        core=tuple(slice(halo,halo+int(n)) for n in end-start)
        arrays={name:value[core].copy() for name,value in arrays.items()}
        arrays['ct']=ct[core].copy()
        maps.write(key,arrays,dict(ct_sha256=sha))
        completed+=1
        if stop and stop():
            maps.close(); return dict(status='PAUSED',new_tiles=completed)
    binding=maps.identity()
    meta=dict(geometry,identity=identity,tiles_sha256=binding,prediction=method,
              source=volume.identity,model_sha256=identity['model_sha256'],
              status='PREPARED',physical_accuracy='UNVALIDATED',
              method_note='SD2 pretrained fibre prediction' if model else 'classical CT threshold/structure-tensor baseline')
    atomic_json(run/'PREPARED.json',meta)
    maps.close()
    return meta
