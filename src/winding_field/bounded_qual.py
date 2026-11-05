"""Globally addressed Qual kernels with bounded, recomputable backing fields."""

from collections import OrderedDict
import itertools
import math
from types import FunctionType,SimpleNamespace

import numpy as np


class PagedField:
    """Read-only scalar ZYX field; cache boundaries never change its geometry."""

    def __init__(self,shape,dtype,load,*,chunk=32,cache_bytes=8<<20):
        self.shape=tuple(shape); self.dtype=np.dtype(dtype)
        if len(self.shape)!=3 or any(type(n) is not int or n<1 for n in self.shape):
            raise ValueError('field requires positive integer ZYX shape')
        if type(chunk) is not int or chunk<1 or self.dtype.hasobject:
            raise ValueError('invalid field chunk or dtype')
        largest=math.prod(min(n,chunk) for n in self.shape)*self.dtype.itemsize
        if type(cache_bytes) is not int or cache_bytes<largest:
            raise ValueError('field cache must hold one complete chunk')
        self.chunk=chunk; self.limit=cache_bytes; self.load=load
        self.cache=OrderedDict(); self._resident_bytes=0; self._compiled=None

    @property
    def resident_bytes(self):
        return self._resident_bytes if self._compiled is None else self._compiled.resident_bytes

    def __getitem__(self,index):
        if not isinstance(index,tuple) or len(index)!=3:
            raise IndexError('global scalar ZYX index required')
        if any(not isinstance(v,(int,np.integer)) or v<0 or v>=n for v,n in zip(index,self.shape)):
            raise IndexError('global scalar index outside field')
        if self._compiled is not None:
            return self.dtype.type(self._compiled[tuple(int(v) for v in index)])
        key=tuple(int(v)//self.chunk for v in index)
        value=self.cache.pop(key,None)
        if value is None:
            lo=np.asarray(key)*self.chunk; hi=np.minimum(lo+self.chunk,self.shape)
            # Evict before loading so even an incompressible tile fits the cache budget.
            needed=math.prod((hi-lo).tolist())*self.dtype.itemsize
            while self.resident_bytes+needed>self.limit:
                _,old=self.cache.popitem(last=False); self._resident_bytes-=old.nbytes
            value=self.load(lo,hi)
            if not isinstance(value,np.ndarray) or value.shape!=tuple(hi-lo) or value.dtype!=self.dtype:
                raise ValueError('field loader returned incorrect shape or dtype')
            # Own the allocation: a sliced view must not retain an entire source volume.
            value=np.array(value,copy=True,order='C')
            value.flags.writeable=False
            self._resident_bytes+=value.nbytes
        self.cache[key]=value
        return value[tuple(int(v)%self.chunk for v in index)]


def python_kernels():
    """Run the pinned scalar math against paged fields, without patching upstream.

    Numba signatures require dense arrays. Rebinding the original Python functions
    keeps the same tracing/history rules while allowing bounded global readers.
    """
    from avf_spline_generator import core
    namespace=dict(vars(core))
    for name,value in vars(core).items():
        function=getattr(value,'py_func',None)
        if function is not None and function.__module__==core.__name__:
            namespace[name]=FunctionType(function.__code__,namespace,name,
                                         function.__defaults__,function.__closure__)
    return namespace


class QualFields:
    """Recompute the pinned half-resolution stencils from native V/H windows.

    ``read_native(lo,hi)`` returns float32 CZYX probabilities on the ORIGINAL
    acquisition lattice, with acquisition validity already applied.
    """

    def __init__(self,native_shape,read_native,*,chunk=32,cache_bytes=8<<20):
        if len(native_shape)!=3 or any(type(n) is not int or n<4 for n in native_shape):
            raise ValueError('native Qual shape must have at least four samples per axis')
        if type(chunk) is not int or chunk<2 or chunk%2:
            raise ValueError('even Qual chunk preserves global seed parity')
        self.shape=tuple(n//2 for n in native_shape)
        self.read_native=read_native; self.chunk=chunk; self.cache_bytes=cache_bytes

    def pooled(self,lo,hi):
        lo=np.asarray(lo,dtype=np.int64); hi=np.asarray(hi,dtype=np.int64)
        if lo.shape!=(3,) or hi.shape!=(3,) or np.any(lo<0) or np.any(hi<=lo) or np.any(hi>self.shape):
            raise ValueError('pooled bounds outside global Qual lattice')
        native=self.read_native(lo*2,hi*2)
        if (not isinstance(native,np.ndarray) or native.shape!=(2,*tuple((hi-lo)*2))
                or native.dtype!=np.dtype('f4') or not np.isfinite(native).all()
                or np.any(native<0) or np.any(native>1)):
            raise ValueError('native Qual reader must supply finite float32 V/H probabilities')
        pooled=np.zeros((2,*tuple(hi-lo)),'f4')
        for z,y,x in itertools.product((0,1),repeat=3):
            pooled+=native[:,z::2,y::2,x::2]
        return np.rint(pooled/8*255).clip(0,255).astype('u1')

    def axes(self,lo,hi):
        from avf_spline_generator.prepare import tensor_axes
        start=np.maximum(0,lo-16); end=np.minimum(self.shape,hi+16)
        inner=tuple(slice(a,b) for a,b in zip(lo-start,hi-start))
        return tensor_axes(self.pooled(start,end),inner)

    def presence(self,lo,hi,family):
        pooled=self.pooled(lo,hi); own,other=pooled[family-1],pooled[2-family]
        presence=np.minimum(own.astype('u2')+other,255).astype('u1')
        presence[own<other]=0
        return presence

    def centrality(self,lo,hi,family):
        from avf_spline_generator import core
        start=np.maximum(0,lo-8); end=np.minimum(self.shape,hi+8)
        if np.any(start%2):
            raise ValueError('centrality region must retain global checkerboard phase')
        values=core.stage1_centrality_and_ridge(self.presence(start,end,family),*self.axes(start,end),2)
        inner=tuple(slice(a,b) for a,b in zip(lo-start,hi-start))
        return tuple(v[inner].copy() for v in values)

    def family(self,family):
        if family not in (1,2):raise ValueError('Qual family must be 1 (V) or 2 (H)')
        loaders=dict(presence=(np.uint8,lambda a,b:self.presence(a,b,family)),
                     nx=(np.uint8,lambda a,b:self.axes(a,b)[0]),
                     ny=(np.uint8,lambda a,b:self.axes(a,b)[1]),
                     centrality=(np.uint16,lambda a,b:self.centrality(a,b,family)[0]),
                     seeds=(np.uint8,lambda a,b:self.centrality(a,b,family)[1]))
        # Each reader has a separately accounted bounded cache; five readers/family.
        return {name:PagedField(self.shape,dtype,load,chunk=self.chunk,cache_bytes=self.cache_bytes)
                for name,(dtype,load) in loaders.items()}


class CoverageReplay:
    """Coverage is a disposable derivative of committed, ordered raw curves.

    ``curves(lo,hi)`` yields this family's raw points/tangents in commit order,
    including every curve whose points can influence the requested chunk.
    Recreate ``fields()`` after a commit adds curves to invalidate its cache.
    """
    def __init__(self,shape,fields,curves,*,chunk=32,cache_bytes=8<<20):
        self.shape=shape;self.geometry=fields;self.curves=curves
        self.chunk=chunk;self.cache_bytes=cache_bytes

    def replay(self,lo,hi):
        from avf_spline_generator import core
        from .bounded_numba import as_numba,CoverageWindow
        outputs=[CoverageWindow(self.shape,tuple(lo),tuple(hi)) for _ in range(3)]
        for points,tangents in self.curves(np.maximum(0,lo-4),np.minimum(self.shape,hi+4)):
            if (points.ndim!=2 or points.shape[1:]!=(3,) or points.shape!=tangents.shape
                    or not np.isfinite(points).all() or not np.isfinite(tangents).all()):
                raise ValueError('invalid committed coverage curve')
            # mark_curve_coverage only writes within ±3 of each rounded point.
            rounded=np.trunc(points[:,::-1]+.5).astype(np.int64)
            keep=np.all((rounded+3>=lo)&(rounded-3<hi),axis=1)
            p=points[keep];t=tangents[keep]
            core.mark_curve_coverage(*outputs,
                *(as_numba(self.geometry[n]) for n in ('centrality','presence','nx','ny')),p,t,len(p),2.05)
        return tuple(v.array for v in outputs)

    def fields(self):
        return tuple(PagedField(self.shape,np.uint8,lambda lo,hi,i=i:self.replay(lo,hi)[i],
                               chunk=self.chunk,cache_bytes=self.cache_bytes) for i in range(3))


def trace_one(seed_zyx,priority,fields,coverage):
    """Trace one global seed and return raw curves for an atomic journal commit."""
    seed=np.asarray(seed_zyx)
    if seed.shape!=(3,) or seed.dtype.kind not in 'iu' or priority not in (1,2):
        raise ValueError('integer global seed and ridge priority required')
    if np.any(seed<0) or np.any(seed>=fields['presence'].shape):
        raise ValueError('seed outside global tracing lattice')
    from .bounded_numba import as_numba,trace_kernel
    points=np.empty((131075,3),'f4');tangents=np.empty_like(points)
    offsets=np.zeros(65538,'i8')
    _,count=trace_kernel()(*(as_numba(fields[n]) for n in ('centrality','presence','nx','ny')),
        *(np.asarray([v],dtype='i4') for v in seed),*(as_numba(f) for f in coverage),points,tangents,offsets,0,0,0,1,
        min_raw_len=7.5,step_size=.75,max_low_steps=3,core_r=2.05,n_pass1=1 if priority==2 else 0)
    return [(points[a:b].copy(),tangents[a:b].copy()) for a,b in zip(offsets[:count],offsets[1:count+1])]


def trace_seeds(seeds,family,fields,journal,*,chunk=32,cache_bytes=8<<20,stop=None):
    """Commit whole global seeds; resume reconstructs coverage from raw history."""
    next_seed=journal.next_seed(family)
    replay=CoverageReplay(fields['presence'].shape,fields,
        lambda lo,hi:journal.curves(family,lo,hi),chunk=chunk,cache_bytes=cache_bytes)
    from .bounded_seeds import SeedPreparationPaused
    iterator=iter(seeds)
    coverage=replay.fields()
    try:
        while True:
            if stop and stop():return 'PAUSED'
            try:sequence,seed,priority=next(iterator)
            except StopIteration:return 'COMPLETE'
            except SeedPreparationPaused:return 'PAUSED'
            if sequence<next_seed:continue
            if sequence!=next_seed:raise ValueError('global seed sequence has a gap')
            curves=trace_one(seed,priority,fields,coverage)
            journal.commit(family,sequence,curves)
            if curves:coverage=replay.fields()
            next_seed+=1
    finally:
        close=getattr(iterator,'close',None)
        if close is not None:close()


def sample_field(field,xyz):
    """Nearest-boundary trilinear samples, without materializing a global field."""
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1:]!=(3,) or not np.isfinite(xyz).all():
        raise ValueError('finite XYZ sampling points required')
    q=np.clip(xyz[:,::-1],0,np.asarray(field.shape)-1)
    low=np.floor(q).astype(np.int64);fraction=q-low
    result=np.zeros(len(q),np.float64)
    for corner in itertools.product((0,1),repeat=3):
        c=np.asarray(corner); ij=np.minimum(low+c,np.asarray(field.shape)-1)
        weight=np.prod(np.where(c,fraction,1-fraction),axis=1)
        for i in np.flatnonzero(weight):result[i]+=float(field[tuple(ij[i])])*weight[i]
    return result.astype('f4')


def finish_curves(curves,tracing,presence,family,native):
    """Pinned fitting and family validation with paged half-resolution reads."""
    from avf_spline_generator import tracing as upstream
    from avf_spline_generator import core
    from .bounded_numba import as_numba,fit_kernel
    if family not in (1,2):raise ValueError('invalid Qual family')
    def valid_segments(points,tracing,family,native):
        delta=np.diff(points,axis=0)
        counts=np.maximum(1,np.ceil(np.linalg.norm(delta,axis=1)/.25).astype(int))
        valid=np.ones(len(delta),bool)
        for n in np.unique(counts):
            all_ids=np.flatnonzero(counts==n)
            step=max(1,32768//(int(n)+1))
            for start in range(0,len(all_ids),step):
                ids=all_ids[start:start+step];t=np.linspace(0,1,int(n)+1,dtype='f4')
                q=(points[ids,None]+delta[ids,None]*t[None,:,None]).reshape(-1,3)
                values=[sample_field(field,q) for field in tracing]
                good=values[family-1]+1e-4>=values[2-family]
                v=native.sample(q*2+.5)
                good &= v[:,family-1]+1e-4>=v[:,2-family]
                valid[ids]=good.reshape(len(ids),-1).all(axis=1)
        return valid
    kernels=SimpleNamespace(**dict(vars(core),fit_and_resample_splines=fit_kernel()))
    namespace=dict(vars(upstream),valid_segments=valid_segments,core=kernels)
    finish=FunctionType(upstream.finish.__code__,namespace,'finish',upstream.finish.__defaults__)
    points,offsets=upstream.pack(curves)
    return finish(points,offsets,tracing,as_numba(presence),family,native)
