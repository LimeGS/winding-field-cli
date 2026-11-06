"""Canonical native inference windows independent of output/cache boundaries."""

from collections import OrderedDict
import itertools
import math

import numpy as np


def region(shape,lo,hi):
    lo=np.asarray(lo); hi=np.asarray(hi)
    if (lo.shape!=(3,) or hi.shape!=(3,) or lo.dtype.kind not in 'iu' or hi.dtype.kind not in 'iu'
            or np.any(lo<0) or np.any(hi<=lo) or np.any(hi>shape)):
        raise ValueError('invalid native prediction region')
    if math.prod((hi-lo).tolist())>256**3:
        raise ValueError('bounded prediction reads require at most 256 cubed samples')
    return lo.astype(np.int64),hi.astype(np.int64)


class QualWindows:
    def __init__(self,volume,model,*,cache_bytes=64<<20,invalid_value=None):
        if type(cache_bytes) is not int or cache_bytes<48**3*8:
            raise ValueError('Qual cache must hold one owned 48-cube V/H window')
        self.volume=volume;self.model=model;self.shape=tuple(volume.shape)
        self.limit=cache_bytes;self.invalid_value=invalid_value
        self.cache=OrderedDict();self.resident_bytes=0

    def window(self,key):
        value=self.cache.pop(key,None)
        if value is None:
            while self.resident_bytes+48**3*8>self.limit:
                _,old=self.cache.popitem(last=False);self.resident_bytes-=old.nbytes
            start=np.asarray(key)*48
            axes=[np.clip(np.arange(s-40,s+88),0,n-1) for s,n in zip(start,self.shape)]
            ct=self.volume.sample(axes)
            if ct.dtype!=np.uint8 or ct.shape!=(128,)*3:
                raise ValueError('Qual requires canonical uint8 128-cube CT')
            probabilities=self.model.predict(ct)[2]
            if (probabilities.shape!=(128,128,128,2) or not np.isfinite(probabilities).all()
                    or np.any(probabilities<0) or np.any(probabilities>1)):
                raise ValueError('invalid Qual V/H predictions')
            value=probabilities[40:88,40:88,40:88].transpose(3,0,1,2).astype('f4',copy=True)
            if self.invalid_value is not None:
                value[:,ct[40:88,40:88,40:88]==self.invalid_value]=0
            self.resident_bytes+=value.nbytes
        self.cache[key]=value
        return value

    def read(self,lo,hi):
        lo,hi=region(self.shape,lo,hi)
        result=np.empty((2,*tuple(hi-lo)),'f4')
        for key in itertools.product(*(range(a//48,(b-1)//48+1) for a,b in zip(lo,hi))):
            start=np.asarray(key)*48; a=np.maximum(lo,start); b=np.minimum(hi,start+48)
            src=tuple(slice(x,y) for x,y in zip(a-start,b-start))
            dst=tuple(slice(x,y) for x,y in zip(a-lo,b-lo))
            result[(slice(None),)+dst]=self.window(key)[(slice(None),)+src]
        return result


class LasagnaWindows:
    """Pinned 64/32/16 full-resolution raw blending, without native-volume output.

    The official ramp is zero in the 16-voxel borders and one in the 32-cube
    core for overlap32. Only the shifted final tiles overlap owned cores.
    """

    def __init__(self,volume,model,*,cache_bytes=64<<20):
        if type(cache_bytes) is not int or cache_bytes<32**3*8*4:
            raise ValueError('Lasagna cache must hold one raw eight-channel core')
        self.volume=volume;self.model=model;self.shape=tuple(volume.shape)
        if any(n<2 for n in self.shape):raise ValueError('Lasagna reflection needs two samples per axis')
        self.positions=[]
        for n in self.shape:
            last=max(0,n-32);values=list(range(0,last+1,32))
            if values[-1]!=last:values.append(last)
            self.positions.append(values)
        self.limit=cache_bytes;self.cache=OrderedDict();self.resident_bytes=0

    def window(self,key):
        value=self.cache.pop(key,None)
        if value is None:
            while self.resident_bytes+32**3*8*4>self.limit:
                _,old=self.cache.popitem(last=False);self.resident_bytes-=old.nbytes
            axes=[]
            for start,n in zip(key,self.shape):
                index=np.arange(start-16,start+48)%(2*(n-1))
                axes.append(np.minimum(index,2*(n-1)-index))
            ct=self.volume.sample(axes)
            if ct.dtype!=np.uint8 or ct.shape!=(64,)*3:
                raise ValueError('bounded Lasagna requires canonical uint8 64-cube CT')
            raw=self.model.predict_raw(ct)
            if raw.shape!=(8,64,64,64) or not np.isfinite(raw).all():
                raise ValueError('invalid Lasagna eight-channel prediction')
            value=np.array(raw[:,16:48,16:48,16:48],dtype='f4',copy=True)
            self.resident_bytes+=value.nbytes
        self.cache[key]=value
        return value

    def read_raw(self,lo,hi):
        lo,hi=region(self.shape,lo,hi)
        values=np.zeros((8,*tuple(hi-lo)),'f4');weight=np.zeros(tuple(hi-lo),'f4')
        positions=[[p for p in axis if p<b and p+32>a] for axis,a,b in zip(self.positions,lo,hi)]
        for key in itertools.product(*positions):
            start=np.asarray(key);a=np.maximum(start,lo);b=np.minimum(start+32,hi)
            src=tuple(slice(x,y) for x,y in zip(a-start,b-start))
            dst=tuple(slice(x,y) for x,y in zip(a-lo,b-lo))
            values[(slice(None),)+dst]+=self.window(key)[(slice(None),)+src]
            weight[dst]+=1
        if not np.all(weight>0):raise ValueError('uncovered canonical Lasagna output')
        values/=weight[None]
        return values
