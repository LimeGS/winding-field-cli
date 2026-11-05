"""Bounded request/reply transport to the pinned private Lasagna runtime."""

import io
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np

MAX_FRAME=64<<20


def _read(stream,count,deadline=None):
    result=bytearray()
    while len(result)<count:
        if deadline is not None:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([stream],[],[],remaining)[0]:
                raise TimeoutError('Lasagna worker response timed out')
        part=stream.read(count-len(result))
        if not part:raise EOFError('Lasagna worker ended during response')
        result.extend(part)
    return bytes(result)


def read_frame(stream,deadline=None):
    size=int.from_bytes(_read(stream,8,deadline),'little')
    if not 0<size<=MAX_FRAME:raise ValueError('bounded worker payload exceeds limit')
    value=np.load(io.BytesIO(_read(stream,size,deadline)),allow_pickle=False)
    if not isinstance(value,np.ndarray) or value.dtype.hasobject or not np.isfinite(value).all():
        raise ValueError('finite numeric worker payload required')
    return value


def _write(stream,part,deadline=None):
    remaining=memoryview(part)
    while remaining:
        if deadline is not None:
            seconds=deadline-time.monotonic()
            if seconds<=0 or not select.select([],[stream],[],seconds)[1]:
                raise TimeoutError('Lasagna worker request timed out')
            try:count=os.write(stream.fileno(),remaining)
            except BlockingIOError:continue
        else:count=stream.write(remaining)
        if count is None or count<=0:raise EOFError('Lasagna worker pipe closed')
        remaining=remaining[count:]


def write_frame(stream,value,deadline=None):
    value=np.asarray(value)
    if value.dtype.hasobject or value.nbytes>MAX_FRAME-4096 or not np.isfinite(value).all():
        raise ValueError('bounded finite numeric worker payload required')
    buffer=io.BytesIO();np.save(buffer,value,allow_pickle=False);payload=buffer.getvalue()
    for part in (len(payload).to_bytes(8,'little'),payload):
        _write(stream,part,deadline)
    stream.flush()


def finalize(adapter,raw):
    if raw.ndim!=4 or raw.shape[0]!=8 or raw.dtype!=np.float32:
        raise ValueError('eight raw float32 Lasagna channels required')
    cos=adapter.finalize_product_slab(adapter.cos_product,raw[:1])
    normal=adapter.finalize_product_slab(adapter.normal_product,raw[1:])
    return np.stack([normal['grad_mag'],normal['nx'],normal['ny'],cos['cos']])


class LasagnaModel:
    """One model child; context exit always releases its CPU/GPU allocations."""
    def __init__(self,python,checkpoint,device,log,*,threads=1,timeout=600):
        self.timeout=timeout
        self.log=Path(log).open('ab')
        command=[str(python),'-s',str(Path(__file__).with_name('lasagna_worker.py')),
                 '--bounded',str(checkpoint),str(device)]
        env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
        env.pop('PYTHONPATH',None)
        try:
            self.child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                         stderr=self.log,env=env,bufsize=0)
            os.set_blocking(self.child.stdin.fileno(),False)
        except BaseException:
            self.log.close();raise

    def __enter__(self):return self

    def __exit__(self,*args):
        if self.log.closed:return
        self.child.stdin.close()
        try:self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            try:self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
        self.child.stdout.close();self.log.close()

    def _request(self,operation,value):
        deadline=time.monotonic()+self.timeout
        try:
            _write(self.child.stdin,operation,deadline)
            write_frame(self.child.stdin,value,deadline)
            return read_frame(self.child.stdout,deadline)
        except BaseException:
            self.child.kill();self.child.wait()
            self.__exit__()
            raise

    def predict_raw(self,ct):
        if ct.shape!=(64,)*3 or ct.dtype!=np.uint8:raise ValueError('canonical uint8 Lasagna tile required')
        value=self._request(b'P',ct)
        if value.shape!=(8,64,64,64) or value.dtype!=np.float32:raise ValueError('invalid raw Lasagna response')
        return value

    def finalize(self,raw):
        value=self._request(b'F',raw)
        if value.shape!=(4,*raw.shape[1:]) or value.dtype!=np.uint8:raise ValueError('invalid encoded Lasagna response')
        return value
