"""Small flattened previews sampled from the original CT, not a predicted map."""

import io
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .util import atomic_bytes


def sample_ct(volume,xyz):
    grid=(xyz-volume.origin_xyz)/volume.spacing_xyz
    zyx=grid[:,::-1]
    valid=((zyx>=0)&(zyx<=np.array(volume.shape)-1)).all(1)
    values=np.zeros(len(xyz),np.float32)
    keys=np.floor(zyx/volume.chunks).astype(np.int64)
    unique,groups=np.unique(keys,axis=0,return_inverse=True)
    for group in range(len(unique)):
        selected=np.flatnonzero((groups==group)&valid)
        if not len(selected):
            continue
        points=zyx[selected]
        lo=np.floor(points.min(0)).astype(int); hi=np.minimum(np.ceil(points.max(0)).astype(int)+1,volume.shape)
        array=volume.read(lo,hi)
        values[selected]=map_coordinates(array.astype(np.float32),(points-lo).T,order=1,mode='nearest')
    return values,valid


def preview(volume,vertices,faces,uv,path,size=256):
    scale=(size-3)/max(float(np.ptp(uv,axis=0).max()),1e-12)
    pixels=(uv-uv.min(0))*scale+1
    shape=np.ceil(pixels.max(0)).astype(int)+2
    xyz=np.zeros((shape[1],shape[0],3)); mask=np.zeros(xyz.shape[:2],bool)
    for face in faces:
        tri=pixels[face]; low=np.floor(tri.min(0)).astype(int); high=np.ceil(tri.max(0)).astype(int)
        y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        query=np.stack((x,y),axis=-1)
        matrix=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]))
        bary=(query-tri[0])@np.linalg.inv(matrix).T
        inside=(bary>=-1e-8).all(-1)&(bary.sum(-1)<=1+1e-8)
        points=vertices[face[0]]+bary[...,0,None]*(vertices[face[1]]-vertices[face[0]])+bary[...,1,None]*(vertices[face[2]]-vertices[face[0]])
        xyz[y[inside],x[inside]]=points[inside]; mask[y[inside],x[inside]]=True
    values,valid=sample_ct(volume,xyz[mask])
    image=np.zeros((*mask.shape,4),np.uint8)
    if len(values):
        low,high=np.percentile(values[valid],[1,99]) if valid.any() else (0.,1.)
        grey=np.clip((values-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8)
        rgba=np.column_stack((grey,grey,grey,valid.astype(np.uint8)*255))
        image[mask]=rgba
    buffer=io.BytesIO(); Image.fromarray(image).save(buffer,format='PNG')
    atomic_bytes(path,buffer.getvalue())
