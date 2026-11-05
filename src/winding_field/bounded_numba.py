"""Compiled global scalar reads; only a tile miss enters the Python loader."""
from functools import lru_cache
import itertools
import weakref
from types import FunctionType

import numpy as np
from numba import from_dtype, njit, objmode, types
from numba.experimental import jitclass
from numba.typed import Dict

_fields=weakref.WeakValueDictionary()
_tokens=itertools.count()
_key_type=types.UniTuple(types.int64,3)


def _load_tile(token,key):
    field=_fields.get(token)
    if field is None:raise RuntimeError('compiled field loader was released')
    lo=np.asarray(key)*field.chunk;hi=np.minimum(lo+field.chunk,field.shape)
    value=field.load(lo,hi)
    if not isinstance(value,np.ndarray) or value.shape!=tuple(hi-lo) or value.dtype!=field.dtype:
        raise ValueError('field loader returned incorrect shape or dtype')
    return np.array(value,copy=True,order='C')


@lru_cache(maxsize=2)
def _reader_type(dtype):
    array_type=types.Array(from_dtype(dtype),3,'C')
    itemsize=dtype.itemsize
    spec=[('shape',_key_type),('chunk',types.int64),('limit',types.int64),
          ('token',types.int64),('resident_bytes',types.int64),('clock',types.int64),
          ('tiles',types.DictType(_key_type,array_type)),
          ('ages',types.DictType(_key_type,types.int64))]

    @jitclass(spec)
    class Reader:
        def __init__(self,shape,chunk,limit,token):
            self.shape=shape;self.chunk=chunk;self.limit=limit;self.token=token
            self.resident_bytes=0;self.clock=0
            self.tiles=Dict.empty(_key_type,array_type)
            self.ages=Dict.empty(_key_type,types.int64)

        def __getitem__(self,index):
            for i in range(3):
                if index[i]<0 or index[i]>=self.shape[i]:
                    raise IndexError('global scalar index outside field')
            key=(index[0]//self.chunk,index[1]//self.chunk,index[2]//self.chunk)
            if key not in self.tiles:
                needed=itemsize
                for i in range(3):needed*=min(self.chunk,self.shape[i]-key[i]*self.chunk)
                while self.resident_bytes+needed>self.limit:
                    oldest=self.clock+1;victim=(0,0,0)
                    for k,age in self.ages.items():
                        if age<oldest:oldest=age;victim=k
                    self.resident_bytes-=self.tiles[victim].nbytes
                    del self.tiles[victim];del self.ages[victim]
                token=self.token
                with objmode(value=array_type):
                    value=_load_tile(token,key)
                self.tiles[key]=value;self.resident_bytes+=value.nbytes
            self.clock+=1;self.ages[key]=self.clock
            value=self.tiles[key]
            return value[index[0]%self.chunk,index[1]%self.chunk,index[2]%self.chunk]

    return Reader


def as_numba(field):
    if isinstance(field,np.ndarray):return field
    if field._compiled is None:
        if field.dtype not in (np.dtype('u1'),np.dtype('u2')):
            raise ValueError('compiled Qual fields require uint8 or uint16')
        token=next(_tokens);_fields[token]=field
        reader=_reader_type(field.dtype)(field.shape,field.chunk,field.limit,token)
        for key,value in field.cache.items():
            value.flags.writeable=True
            reader.tiles[key]=value;reader.clock+=1;reader.ages[key]=reader.clock
            reader.resident_bytes+=value.nbytes
        field.cache.clear();field._resident_bytes=0;field._compiled=reader
    return field._compiled


@njit
def _defer_coverage(*args):
    pass


@lru_cache(maxsize=1)
def trace_kernel():
    from avf_spline_generator import core
    function=core.trace_seed_range.py_func
    # One seed commits atomically; coverage becomes visible only after its journal commit.
    namespace=dict(vars(core),mark_curve_coverage=_defer_coverage)
    return njit(FunctionType(function.__code__,namespace,function.__name__,function.__defaults__))


@lru_cache(maxsize=1)
def fit_kernel():
    from avf_spline_generator import core
    # Paged readers mutate their LRU; do not share them between parallel curve workers.
    return njit(core.fit_and_resample_splines.py_func,parallel=False)


@jitclass([('shape',_key_type),('lo',_key_type),('hi',_key_type),
           ('array',types.uint8[:,:,::1])])
class CoverageWindow:
    """Global coverage semantics with writes retained only in one output tile."""
    def __init__(self,shape,lo,hi):
        self.shape=shape;self.lo=lo;self.hi=hi
        self.array=np.zeros((hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]),np.uint8)

    def __getitem__(self,index):
        for i in range(3):
            if index[i]<self.lo[i] or index[i]>=self.hi[i]:return np.uint8(0)
        return self.array[index[0]-self.lo[0],index[1]-self.lo[1],index[2]-self.lo[2]]

    def __setitem__(self,index,value):
        for i in range(3):
            if index[i]<self.lo[i] or index[i]>=self.hi[i]:return
        self.array[index[0]-self.lo[0],index[1]-self.lo[1],index[2]-self.lo[2]]=value
