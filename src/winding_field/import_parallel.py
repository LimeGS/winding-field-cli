"""Resource-admitted, bounded ordered computation; only the parent publishes."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

from .surface_parallel import cpu_limit
from .tifxyz_runtime import memory_report

_STOP=None
RESULT_BYTES=16<<20


def local_inputs(config):
    return not any(str(config.get(k) or '').startswith(('http://','https://','s3://'))
                   for k in ('source','lasagna_input'))


def limits(requested,*,output,reserve_bytes,worker_bytes):
    if (type(requested) is not int or requested<0 or type(worker_bytes) is not int or worker_bytes<=0
            or type(reserve_bytes) is not int or reserve_bytes<0):
        raise ValueError('invalid import worker admission')
    cpus=max(1,cpu_limit());memory=memory_report()
    slots=max(1,cpus-1);reason='cpu'
    if not memory['known']:slots=1;reason='unknown-memory'
    else:
        # Available headroom already excludes live parent RSS; charge all new
        # worker scratch, caches and parent-side in-flight results in worker_bytes.
        head=min(memory['host_available_bytes']-(2<<30),memory['effective_headroom_bytes']-(128<<20))
        memory_slots=max(0,head//worker_bytes)
        if memory_slots<slots:slots=memory_slots;reason='memory'
    output=Path(output)
    while not output.exists():output=output.parent
    disk_slots=max(0,(shutil.disk_usage(output).free-reserve_bytes)//(16<<20))
    if disk_slots<slots:slots=disk_slots;reason='disk'
    if slots<1:raise OSError('insufficient space or memory for import workers and reserve')
    workers=min(requested or slots,slots)
    return dict(workers=workers,max_inflight=2*workers,worker_bytes=worker_bytes,
                limiting_resource=reason,requested=requested)


def worker_stopped():
    return _STOP is not None and _STOP.is_set()


def _initialize(event,initializer,args):
    global _STOP
    _STOP=event
    if initializer is not None:initializer(*args)


@contextmanager
def _spawn_environment():
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','CUDA_VISIBLE_DEVICES')
    previous={name:os.environ.get(name) for name in names}
    try:
        os.environ.update({name:('' if name=='CUDA_VISIBLE_DEVICES' else '1') for name in names})
        yield
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def ordered_map(fn,jobs,*,workers,max_inflight,initializer=None,initargs=(),stop=None):
    if type(workers) is not int or workers<1 or type(max_inflight) is not int or max_inflight<1:
        raise ValueError('positive worker/in-flight limits required')
    if workers==1:
        if initializer is not None:initializer(*initargs)
        for job in jobs:
            if stop and stop():raise InterruptedError('import paused')
            yield fn(job)
        return
    context=mp.get_context('spawn');event=context.Event()
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=_initialize,
                             initargs=(event,initializer,initargs))
    pending=deque();iterator=iter(jobs);exhausted=False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending)<min(max_inflight,2*workers):
                if stop and stop():raise InterruptedError('import paused')
                try:job=next(iterator)
                except StopIteration:exhausted=True;break
                with _spawn_environment():pending.append(pool.submit(fn,job))
            if not pending:break
            while True:
                if stop and stop():raise InterruptedError('import paused')
                try:result=pending[0].result(timeout=.05);break
                except TimeoutError:continue
            pending.popleft()
            yield result
    finally:
        event.set()
        for future in pending:future.cancel()
        # Python3.11/3.12 expose no public terminate-workers API.
        processes=list((pool._processes or {}).values())
        manager=pool._executor_manager_thread
        pool.shutdown(wait=False,cancel_futures=True)
        deadline=time.monotonic()+5
        for process in processes:process.join(max(0.,deadline-time.monotonic()))
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:
            process.join(.5)
            if process.is_alive():process.kill();process.join()
        if manager is not None:manager.join(1.)
