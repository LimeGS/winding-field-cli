"""Spawn-safe, read-only geometry workers. This module imports no native numerical libraries at import time."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor,TimeoutError
import multiprocessing
import os
import resource
import sys
import time

CAUSES={'valid':0,'reference_domain':1,'reference_ambiguous':2,'invalid_field':3,
        'zero_gradient':4,'unsupported_path':5,'nonconvergence':6,'roundtrip':7,
        'sampled_jacobian':8,'invalid_target':9,'coordinate_precision':10,'reference_unresolved':11}
THREAD_VARIABLES=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                  'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
_STATE=None


def compute_geometry(reference,field,frame,uv,w,*,reference_data=None,stop=None):
    """One <=16384-point batch, no CT or writes. Returns numeric arrays and timing/cache metrics."""
    import numpy as np
    from copy import copy
    from dataclasses import replace
    from .uvw import _points,reference_points,xyz_to_uvw,correspondence_transport,sample_w_gradient
    target_in_band=reference.band is None or reference.band[0]<=w<=reference.band[1]
    reference=copy(reference)
    # Admit the requested W; float32 XYZ can round across its endpoint during inverse verification.
    reference.band=None
    uv=_points(uv,2)
    if reference_data is not None:
        if (not isinstance(reference_data,dict) or set(reference_data)!={'base','reference_valid','cause','reference_face','reference_barycentric'}
            or np.shape(reference_data['base'])!=(len(uv),3) or np.shape(reference_data['reference_valid'])!=(len(uv),)
            or np.asarray(reference_data['reference_valid']).dtype!=bool or np.shape(reference_data['cause'])!=(len(uv),)
            or np.asarray(reference_data['cause']).dtype!=np.uint8
            or np.shape(reference_data['reference_face'])!=(len(uv),)
            or np.asarray(reference_data['reference_face']).dtype.kind not in 'iu'
            or np.shape(reference_data['reference_barycentric'])!=(len(uv),3)):raise ValueError('invalid cached reference batch')
    started=time.perf_counter();before=dict(field.stats);reference_seconds=0.
    if reference_data is None:
        begin=time.perf_counter();base,valid,provenance=reference_points(reference,field,uv,stop=stop,with_provenance=True)
        cause=np.full(len(uv),CAUSES['reference_domain'],np.uint8);ids=np.flatnonzero(valid)
        if len(ids):
            recovered,good,details=xyz_to_uvw(reference,field,base[ids],stop=stop,with_details=True)
            good&=np.linalg.norm(recovered[:,:2]-uv[ids],axis=1)<=1e-6
            cause[ids[~good]]=CAUSES['reference_ambiguous'];valid[ids]&=good
            for key in ('reference_unresolved','reference_domain','invalid_field'):
                cause[ids[details['cause']==key]]=CAUSES[key]
        cause[valid]=0;reference_data=dict(base=base,reference_valid=valid,cause=cause,
            reference_face=provenance['face'],reference_barycentric=provenance['barycentric'])
        reference_seconds=time.perf_counter()-begin
    base=reference_data['base'];valid=reference_data['reference_valid'].copy();cause=reference_data['cause'].copy()
    # Tighten correspondence only as far as this batch's float64 resolution permits.
    precision=np.spacing(np.abs(base[valid]).max(initial=0.))/min(field.spacing_um)
    reference.settings=replace(reference.settings,error_fraction=min(reference.settings.error_fraction,1e-10,max(1e-12,precision)))
    if not target_in_band:
        cause[valid]=CAUSES['invalid_target'];valid[:]=False
    xyz=np.full_like(base,np.nan);residual=np.full(len(base),np.nan);roundtrip=np.full(len(base),np.nan)
    rounding=np.full(len(base),np.nan);ids=np.flatnonzero(valid)
    if len(ids):
        moved,ok,info=correspondence_transport(field,base[ids],w,reference.settings,stop=stop)
        for key,code in CAUSES.items():cause[ids[info['cause']==key]]=code
        origin=np.asarray(frame['origin_xyz_um']);unit=frame['voxel_um']
        with np.errstate(over='ignore',invalid='ignore'):
            encoded=((moved-origin)/unit).astype(np.float32);stored=encoded.astype(float)*unit+origin
        rounding[ids]=np.linalg.norm(stored-moved,axis=1)
        sampled,g,supported=sample_w_gradient(field,stored);norm=np.linalg.norm(g,axis=1)
        distance=abs(sampled-w)/np.maximum(norm,1e-30)
        precise=supported&(rounding[ids]<=.01*min(field.spacing_um))&(distance<=reference.settings.residual_fraction*min(field.spacing_um))
        recovered,inverse_valid=xyz_to_uvw(reference,field,stored,stop=stop)
        precise&=inverse_valid&(np.linalg.norm(recovered[:,:2]-uv[ids],axis=1)<=1e-6)
        cause[ids[ok&~precise]]=CAUSES['coordinate_precision'];ok&=precise
        moved=stored;moved[~ok]=np.nan
        back,good,_=correspondence_transport(field,moved,reference.w,reference.settings,stop=stop)
        errors=np.linalg.norm(back-base[ids],axis=1)
        failed=ok&(~good|(errors>reference.settings.roundtrip_fraction*min(field.spacing_um)))
        cause[ids[failed]]=CAUSES['roundtrip'];ok&=~failed
        valid[ids]&=ok;xyz[ids]=moved;residual[ids]=distance;roundtrip[ids]=errors
    field.validate()
    data=dict(xyz=xyz,base=base,reference_valid=reference_data['reference_valid'],reference_cause=reference_data['cause'],
        reference_face=reference_data['reference_face'],reference_barycentric=reference_data['reference_barycentric'],
        geometry=valid,cause=cause,residual=residual,roundtrip=roundtrip,coordinate_rounding=rounding)
    metrics=dict(pid=os.getpid(),geometry_seconds=time.perf_counter()-started,reference_seconds=reference_seconds,
        field={key:value-before.get(key,0) for key,value in field.stats.items() if key!='region_cache_bytes'},
        region_cache_bytes=field.stats.get('region_cache_bytes',0),native_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    return data,metrics


def _open_geometry(descriptor,path,levels,samples,region_cache_bytes,reference_identity=None):
    from .surface_sources import FieldSource
    from .tifxyz_field import PhysicalField
    from .uvw_pipeline import load_reference
    from .util import digest
    reference,metadata=load_reference(path,levels,samples)
    if reference_identity is not None and digest(metadata)!=reference_identity:raise ValueError('worker reference identity changed')
    field=PhysicalField(FieldSource(descriptor['field'],descriptor['geometry']),region_cache_bytes=region_cache_bytes)
    return reference,field,metadata['output_frame']


def _initialize(descriptor,path,levels,samples,region_cache_bytes,reference_identity,event):
    global _STATE
    from multiprocessing.util import Finalize
    started=time.perf_counter();reference,field,frame=_open_geometry(descriptor,path,levels,samples,region_cache_bytes,reference_identity)
    Finalize(field,field.close,exitpriority=10)
    _STATE=(reference,field,frame,event,time.perf_counter()-started)


def _geometry_job(job):
    reference,field,frame,event,elapsed=_STATE
    if event.is_set():raise InterruptedError('UVW geometry paused')
    data,metrics=compute_geometry(reference,field,frame,**job,stop=event.is_set)
    metrics['initialization_seconds']=elapsed
    return data,metrics


class GeometryWorkers:
    """Ordered bounded jobs, spawn only, no SQLite connection or CT reader passed to children.

    Benchmark API: with GeometryWorkers(...) as pool: list(pool.map(jobs)), where
    each job is {uv: Nx2 physical UV, w: target, reference_data: None or a saved reference batch}.
    """
    def __init__(self,descriptor,reference_sheet,levels,canvas_samples=512,*,workers=1,region_cache_bytes=64<<20,
                 stop=None,inline=None,reference_identity=None):
        if type(workers) is not int or not 1<=workers<=8:raise ValueError('workers must be 1..8')
        if type(region_cache_bytes) is not int or region_cache_bytes<0:raise ValueError('invalid region cache budget')
        self.arguments=(descriptor,str(reference_sheet),levels,canvas_samples,region_cache_bytes)
        self.workers=workers;self.stop=stop or (lambda:False);self.inline=inline;self.owned=inline is None
        self.reference_identity=reference_identity
        self.pool=None;self.event=None;self.environment={}

    def __enter__(self):
        if self.workers==1:
            if self.inline is None:self.inline=_open_geometry(*self.arguments)
        else:
            if self.reference_identity is None:
                from .uvw_pipeline import load_reference
                from .util import digest
                _,metadata=load_reference(self.arguments[1],self.arguments[2],self.arguments[3])
                self.reference_identity=digest(metadata)
            # Spawn imports __main__ before the initializer; set limits in the inherited environment first.
            self.environment={key:os.environ.get(key) for key in THREAD_VARIABLES}
            for key in THREAD_VARIABLES:os.environ[key]='1'
            try:
                context=multiprocessing.get_context('spawn');self.event=context.Event()
                self.pool=ProcessPoolExecutor(self.workers,mp_context=context,initializer=_initialize,
                    initargs=(*self.arguments,self.reference_identity,self.event))
            except BaseException:
                self._restore_environment();raise
        return self

    def map(self,jobs):
        if self.pool is None:
            reference,field,frame=self.inline
            for job in jobs:
                if self.stop():raise InterruptedError('UVW geometry paused')
                yield compute_geometry(reference,field,frame,**job,stop=self.stop)
            return
        pending=deque();iterator=iter(jobs);exhausted=False
        try:
            while pending or not exhausted:
                if self.stop():raise InterruptedError('UVW geometry paused')
                while not exhausted and len(pending)<2*self.workers:
                    try:job=next(iterator)
                    except StopIteration:exhausted=True;break
                    pending.append(self.pool.submit(_geometry_job,job))
                if not pending:break
                try:result=pending[0].result(timeout=.1)
                except TimeoutError:
                    if pending[0].done():raise
                    continue
                pending.popleft();yield result
        except BaseException:
            self.event.set()
            for future in pending:future.cancel()
            raise

    def _restore_environment(self):
        for key,value in self.environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value

    def __exit__(self,*unused):
        try:
            if self.pool is not None:
                self.event.set();self.pool.shutdown(wait=True,cancel_futures=True)
            elif self.owned and self.inline is not None:self.inline[1].close()
        finally:self._restore_environment()
