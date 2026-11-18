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


class LasagnaProducts:
    """Globally anchored raw Gaussian grids, encoded only after reduction."""
    chunk=16

    def __init__(self,windows,factors):
        self.windows=windows;self.factors=factors
        self.cache=OrderedDict();self.resident_bytes=0

    def raw(self,factor,lo,hi):
        if factor==1:return self.windows.read_raw(lo,hi)
        from scipy.ndimage import convolve1d
        parent=(np.asarray(self.windows.shape)+factor//2-1)//(factor//2)
        axes=[]
        for a,b,n in zip(lo,hi,parent):
            q=np.arange(2*a-2,2*(b-1)+3)
            if n==1:q=np.zeros_like(q)
            else:
                q%=2*(n-1);q=np.minimum(q,2*(n-1)-q)
            axes.append(q)
        start=np.array([a.min() for a in axes]);end=np.array([a.max()+1 for a in axes])
        values=self.raw(factor//2,start,end)
        values=values[np.ix_(np.arange(8),*(a-s for a,s in zip(axes,start)))]
        for axis in (1,2,3):
            values=convolve1d(values,np.array([1,4,6,4,1],dtype='f4')/16,axis=axis,mode='mirror')
        return values[:,2:-2:2,2:-2:2,2:-2:2].copy()

    def tile(self,factor,key):
        cache_key=(factor,*key)
        value=self.cache.pop(cache_key,None)
        if value is None:
            lo=np.array(key)*self.chunk
            shape=(np.asarray(self.windows.shape)+factor-1)//factor
            raw=self.raw(factor,lo,np.minimum(lo+self.chunk,shape))
            value=self.windows.model.finalize(raw)
            while self.cache and self.resident_bytes+value.nbytes>32<<20:
                _,old=self.cache.popitem(last=False);self.resident_bytes-=old.nbytes
            self.resident_bytes+=value.nbytes
        self.cache[cache_key]=value
        return value

    def channel(self,name):
        owner=self;factor=self.factors[name];index=('grad_mag','nx','ny','cos').index(name)
        class Channel:
            shape=tuple((n+factor-1)//factor for n in owner.windows.shape)
            def sample(self,axes):
                result=np.empty(tuple(map(len,axes)),'u1')
                for key in itertools.product(*(np.unique(a//owner.chunk) for a in axes)):
                    indices=[np.flatnonzero(a//owner.chunk==k) for a,k in zip(axes,key)]
                    local=[a[i]-k*owner.chunk for a,i,k in zip(axes,indices,key)]
                    result[np.ix_(*indices)]=owner.tile(factor,key)[index][np.ix_(*local)]
                return result
        return Channel()
