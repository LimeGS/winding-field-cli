"""Small flattened previews sampled from the original CT, not a predicted map."""

import io
import numpy as np
from PIL import Image
from contextlib import closing

from .util import atomic_bytes


def sample_ct(volume,xyz):
    from .native_ct_sampling import NativeCTSampler
    grid=(xyz-volume.origin_xyz)/volume.spacing_xyz
    values=np.zeros(len(xyz),np.float32)
    valid=np.zeros(len(xyz),bool)
    def chunk(index):
        data=volume._chunk(index)
        # Previews retain Volume.read's authoritative sparse-fill semantics.
        return np.full(volume.chunks,volume.metadata.get('fill_value',0),volume.dtype) if data is None else data
    with closing(NativeCTSampler(volume.shape,volume.chunks,volume.dtype,chunk)) as sampler:
        for start in range(0,len(grid),100000):
            sampled,observed=sampler.sample(grid[start:start+100000])
            values[start:start+100000]=np.where(observed,sampled,0)
            valid[start:start+100000]=observed
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
