"""Bounded filters only: -1 means the caller must evaluate its exact predicate.

Native code is loaded solely from an operator-selected environment manifest, never
from a field/export receipt. Importing this module loads neither NumPy nor CUDA.
"""
import ctypes as C
import os
from pathlib import Path
import platform
import re

from .tifxyz_contract import Blocked, parse_json
from .util import file_sha

MAX_ROWS = 65536
BATCH_ROWS = 4096
GPU_RESERVE = 256 << 20


def cuda_device():
    value = os.environ.get('WINDING_TIFXYZ_CUDA_DEVICE', '')
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        raise Blocked('ACCELERATOR_UNAVAILABLE', 'set WINDING_TIFXYZ_CUDA_DEVICE to an explicit GPU UUID')
    return value


def _native(kind):
    from .tifxyz import _path, _record
    chosen = os.environ.get('WINDING_TIFXYZ_ACCELERATOR')
    if not chosen: raise Blocked('ACCELERATOR_UNAVAILABLE', 'set WINDING_TIFXYZ_ACCELERATOR to your built manifest')
    path = _path(chosen)
    if path.stat().st_size > 65536: raise ValueError('accelerator manifest envelope exceeded')
    record = _record(path); data = parse_json(path.read_bytes())
    keys = {'schema','abi','backend','max_rows','library','library_sha256','library_bytes','sources','platform','compiler'}
    if (not isinstance(data,dict) or set(data)!=keys or data['schema']!='winding-field-tifxyz-accelerator-v1'
        or type(data['abi']) is not int or data['abi']!=1 or type(data['max_rows']) is not int or data['max_rows']!=MAX_ROWS
        or data['backend'] not in ('cpu','cuda') or data['platform']!=dict(system=platform.system(),machine=platform.machine())):
        raise ValueError('accelerator ABI/platform/schema differs')
    name = data['library']
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.(so|dylib)',name):
        raise ValueError('accelerator requires a local library basename')
    if type(data['library_bytes']) is not int or not 0<data['library_bytes']<=32<<20:
        raise ValueError('accelerator library envelope exceeded')
    sources = Path(__file__).parent/'native'
    if data['sources']!={n:file_sha(sources/n) for n in ('predicates.h','predicates.cpp','predicates.cu')}:
        raise ValueError('accelerator sources differ from this implementation')
    library = _path(path.parent/name)
    if library.stat().st_size!=data['library_bytes'] or _record(library)!=dict(bytes=data['library_bytes'],sha256=data['library_sha256']):
        raise ValueError('accelerator library hash/size differs')
    if _record(path)!=record: raise ValueError('accelerator manifest changed')
    if kind=='cuda' and data['backend']!='cuda': raise Blocked('ACCELERATOR_UNAVAILABLE','CUDA artifact required')
    binding = dict(backend=kind,abi=1,manifest_sha256=record['sha256'],library_sha256=data['library_sha256'])
    return path,library,data,binding


def identity(kind):
    if kind=='numpy': return dict(backend='numpy')
    if kind not in ('cpu','cuda'): raise ValueError('invalid audit backend')
    return _native(kind)[3]


def _signature(path):
    info=path.stat()
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def _numpy(mode,rows):
    import numpy as np
    if mode==2: return np.full(len(rows),-1,dtype=np.int8)
    def point(x): return x,x
    def add(a,b): return np.nextafter(a[0]+b[0],-np.inf),np.nextafter(a[1]+b[1],np.inf)
    def sub(a,b): return np.nextafter(a[0]-b[1],-np.inf),np.nextafter(a[1]-b[0],np.inf)
    def mul(a,b):
        p=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1])
        return np.nextafter(np.minimum.reduce(p),-np.inf),np.nextafter(np.maximum.reduce(p),np.inf)
    def div(a,b): return np.nextafter(a[0]/b,-np.inf),np.nextafter(a[1]/b,np.inf)
    count=16 if mode==0 else 21
    valid=np.isfinite(rows[:,:count]).all(1)&(np.abs(rows[:,:count])<=1e20).all(1)
    scalars=rows[:,[6,15] if mode==0 else [9,10,11,12]]
    valid&=((scalars>=1e-20)&(scalars<=1e20)).all(1)
    with np.errstate(under='ignore',over='ignore',invalid='ignore',divide='ignore'):
        x=rows.T
        if mode==0:
            valid&=(rows[:,7:10]<=rows[:,10:13]).all(1)&(x[13]>=0)&(x[13]<=x[14])
            d2=point(np.zeros(len(rows)))
            for k in range(3):
                p=add(mul(point(x[k]),point(x[6])),point(x[3+k]))
                delta=sub((x[7+k],x[10+k]),p);d2=add(d2,mul(delta,delta))
            d2=(np.maximum(0,d2[0]),d2[1]); h2=(x[13],x[14])
            r=sub(sub(mul(point(x[15]),point(x[15])),d2),h2)
            test=sub(mul(r,r),mul(point(np.full(len(rows),4.)),mul(d2,h2)))
            valid&=np.isfinite((*r,*test)).all(0)
            return np.where(valid&(r[0]>=0)&(test[0]>=0),1,np.where(valid&((r[1]<0)|(test[1]<0)),0,-1)).astype(np.int8)
        a=[mul(sub(point(x[3+k]),point(x[k])),point(x[9])) for k in range(3)]
        b=[mul(sub(point(x[6+k]),point(x[k])),point(x[9])) for k in range(3)]
        normal=[sub(mul(a[(k+1)%3],b[(k+2)%3]),mul(a[(k+2)%3],b[(k+1)%3])) for k in range(3)]
        lower=point(np.zeros(len(rows)))
        for axis in range(3):
            bit=1<<axis
            terms=[mul(div(sub(point(x[13+j+bit]),point(x[13+j])),x[10+axis]),normal[axis])
                   for j in range(8) if not j&bit]
            lower=add(lower,(np.minimum.reduce([t[0] for t in terms]),np.minimum.reduce([t[1] for t in terms])))
        valid&=np.isfinite(lower).all(0)
        return np.where(valid&(lower[0]>0),1,np.where(valid&(lower[1]<=0),0,-1)).astype(np.int8)


class Backend:
    def __init__(self,kind='numpy',*,expected=None):
        self.kind=kind;self.library=None;self.paths=();self._gpu=False;self._closed=False
        if kind not in ('numpy','cpu','cuda'):raise ValueError('invalid audit backend')
        if kind=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES')!=cuda_device():
            raise Blocked('ACCELERATOR_UNAVAILABLE','CUDA visibility must match the explicitly selected UUID')
        native=None if kind=='numpy' else _native(kind)
        self.binding=dict(backend='numpy') if native is None else native[3]
        if expected is not None and self.binding!=expected:raise ValueError('accelerator differs from frozen export identity')
        if native is not None:
            path,library,data,_=native
            self.paths=(path,library);self.signatures=tuple(map(_signature,self.paths))
            self.library=C.CDLL(str(library))
            for name in ('wf_accel_abi','wf_accel_kind'):
                function=getattr(self.library,name);function.argtypes=[];function.restype=C.c_int
            if self.library.wf_accel_abi()!=1 or self.library.wf_accel_kind()!=int(data['backend']=='cuda'):
                raise ValueError('loaded accelerator ABI/backend differs')
            self.library.wf_cpu_filter.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_int8)]
            self.library.wf_cpu_filter.restype=C.c_int
        if kind=='cuda':
            self.library.wf_cuda_init.argtypes=[C.c_int,C.c_uint64]
            self.library.wf_cuda_filter.argtypes=self.library.wf_cpu_filter.argtypes
            self.library.wf_cuda_memory.argtypes=[C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
            self.library.wf_cuda_close.argtypes=[]

    @staticmethod
    def _check(code):
        if code:
            reason='NUMERICAL_UNCERTAINTY' if code==-3 else 'GPU_MEMORY_BUDGET' if code in (-2,2) else 'ACCELERATOR_UNAVAILABLE'
            raise Blocked(reason,'native accelerator status '+str(code))

    def _unchanged(self):
        if self.paths and tuple(map(_signature,self.paths))!=self.signatures:
            raise ValueError('accelerator changed during audit')

    def classify(self,mode,rows):
        import numpy as np
        if (self._closed or type(mode) is not int or mode not in (0,1,2) or not isinstance(rows,np.ndarray)
            or rows.dtype!=np.float64 or rows.ndim!=2 or rows.shape[1]!=24 or not 1<=len(rows)<=MAX_ROWS):
            raise ValueError('invalid bounded predicate batch')
        self._unchanged()
        if self.kind=='numpy': return _numpy(mode,rows)
        rows=np.ascontiguousarray(rows);out=np.empty(len(rows),dtype=np.int8)
        function=self.library.wf_cpu_filter
        if self._gpu and mode!=2 and len(rows)>=256:
            free=C.c_uint64();total=C.c_uint64()
            self._check(self.library.wf_cuda_memory(C.byref(free),C.byref(total)))
            if free.value<GPU_RESERVE:raise Blocked('GPU_MEMORY_BUDGET','free VRAM reserve consumed')
            function=self.library.wf_cuda_filter
        self._check(function(mode,len(rows),rows.ctypes.data_as(C.POINTER(C.c_double)),out.ctypes.data_as(C.POINTER(C.c_int8))))
        if not np.isin(out,[-1,0,1]).all():raise ValueError('invalid native predicate result')
        return out

    def __enter__(self):
        import numpy as np
        with np.errstate(under='ignore'):
            for dtype,ulp in ((np.float64,2**-53),(np.float32,2**-24)):
                tiny=dtype(np.finfo(dtype).tiny)
                if (tiny/dtype(2))*dtype(2)!=tiny or dtype(1)+dtype(ulp)!=1:
                    raise Blocked('NUMERICAL_UNCERTAINTY','unsupported floating-point environment')
        self._unchanged()
        if self.paths and identity(self.kind)!=self.binding:raise ValueError('accelerator changed before audit')
        if self.kind=='cuda':
            self._check(self.library.wf_cuda_init(MAX_ROWS,GPU_RESERVE));self._gpu=True
        return self

    def __exit__(self,kind,value,trace):
        self._closed=True
        if self._gpu:
            code=self.library.wf_cuda_close();self._gpu=False
            if kind is None:self._check(code)
        if kind is None:
            self._unchanged()
            if self.paths and identity(self.kind)!=self.binding:raise ValueError('accelerator changed during audit')
