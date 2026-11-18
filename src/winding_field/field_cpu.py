"""Bounded static CPU lookahead; the caller retains every solve and write."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from contextlib import closing
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from field_constraints import QuadraticConstraints
from field_runtime import ResourceUnavailable
from .artifacts import Tiles
from .import_parallel import _spawn_environment
from .surface_parallel import cpu_limit
from .tifxyz_runtime import memory_report
from .util import digest,file_sha

_STATE=None
WORKER_BASE_BYTES=512<<20


def headroom_bytes():
    if sys.platform=='darwin':
        text=subprocess.check_output(['vm_stat'],text=True)
        page=int(text.split('page size of ')[1].split(' bytes')[0])
        values={line.split(':')[0]:int(line.split(':')[1].strip().rstrip('.'))
                for line in text.splitlines()[1:] if ':' in line}
        # Only immediately free/reclaimable file pages, not compressed or anonymous pages.
        return max(0,page*(values['Pages free']+values['File-backed pages'])-(2<<30))
    report=memory_report()
    if not report['known']:return 0
    return max(0,min(report['effective_headroom_bytes']-(128<<20),
                     report['host_available_bytes']-(2<<30)))


def admit_workers(requested,max_bytes,cache_bytes,*,parent_threads,source_bytes=0):
    slots=max(0,cpu_limit()-parent_threads)
    # Headroom already excludes the parent's current RSS; reserve future cache,
    # IPC copies and conservative numeric workspaces in addition to each runtime.
    available=headroom_bytes()-cache_bytes-max_bytes
    per_worker=WORKER_BASE_BYTES+source_bytes+8*max_bytes
    slots=min(slots,max(0,available//per_worker))
    if slots<1:raise ResourceUnavailable('CPU preparation has no admitted CPU/RAM slot')
    return min(requested,slots)


def retained_bytes(observations,rows):
    return (4096+sum(a.nbytes for a in observations[:3])+
            (0 if rows is None else rows.targets.nbytes+sum(a.nbytes for a in rows._structural_arrays())))


@dataclass
class PreparedBlock:
    lo: tuple
    hi: tuple
    identity: str
    observations: tuple
    rows: QuadraticConstraints | None
    retained_bytes: int
    worker_info: dict


def _initialize(spec):
    global _STATE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU worker has an unsafe CUDA environment')
    run=Path(spec['run'])
    maps=Tiles(run/'maps',readonly=True)
    if maps.identity()!=spec['meta']['tiles_sha256'] or file_sha(run/'fibres/fibres.sqlite')!=spec['fibres']['sha256']:
        maps.close()
        raise ValueError('CPU worker source identity changed')
    db=sqlite3.connect((run/'fibres/fibres.sqlite').as_uri()+'?mode=ro',uri=True)
    db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,spec['curve_bytes'])
    _STATE=(spec,maps,db)


def prepare_static_block(spec,lo,hi,build_rows=True):
    from .field import sample_prior,_constraint_rows
    if _STATE is None or _STATE[0]!=spec:raise ValueError('CPU worker was not initialized for this source')
    _,maps,db=_STATE
    low=np.asarray(lo);high=np.asarray(hi)
    values=sample_prior(maps,spec['meta'],spec['geometry'],spec['calibration'],low,high)
    observations=(*values,{'maps':spec['meta']['tiles_sha256']})
    try:
        rows=(_constraint_rows(db,spec['meta'],spec['fibres'],spec['geometry'],maps,low,high,spec['weight'],
                               max_pairs=spec['max_pairs']) if build_rows else None)
    except sqlite3.DataError as error:
        if error.sqlite_errorcode!=sqlite3.SQLITE_TOOBIG:raise
        raise ResourceUnavailable('CPU curve exceeds the admitted SQLite payload limit') from error
    if rows is not None:
        rows=rows.with_targets(rows.targets)
        rows.targets.setflags(write=False)
    size=retained_bytes(observations,rows)
    if size>spec['payload_bytes']:raise ResourceUnavailable('CPU prepared payload exceeds its buffer slot')
    info=dict(pid=os.getpid(),threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized(),
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
              built_constraints=build_rows)
    return PreparedBlock(tuple(lo),tuple(hi),digest(spec),observations,rows,size,info)


def validate_block(block,spec,lo,hi,*,build_rows):
    if not isinstance(block,PreparedBlock) or block.lo!=tuple(lo) or block.hi!=tuple(hi) or block.identity!=digest(spec):
        raise ValueError('CPU payload bounds or source identity mismatch')
    values=block.observations;shape=tuple(b-a for a,b in zip(lo,hi))
    if (not isinstance(values,tuple) or len(values)!=4 or values[3]!={'maps':spec['meta']['tiles_sha256']}):
        raise ValueError('CPU payload has incomplete observations')
    for array,want,dtype in zip(values[:3],(shape+(3,),shape,shape),(np.float32,np.float32,np.bool_)):
        if (not isinstance(array,np.ndarray) or array.shape!=want or array.dtype!=dtype
                or not np.isfinite(array).all()):
            raise ValueError('CPU payload observation dtype/shape/value mismatch')
    if (values[1]<0).any():raise ValueError('CPU payload has negative weights')
    rows=block.rows
    if rows is not None:
        if not build_rows or not isinstance(rows,QuadraticConstraints) or rows.matrix.shape[1]!=int(np.prod(shape)):
            raise ValueError('CPU payload has invalid constraints')
        rows.matrix.check_format(full_check=True)
        if (rows.targets.shape!=(rows.matrix.shape[0],) or
                any(a.dtype.hasobject or not np.isfinite(a).all() for a in (*rows._structural_arrays(),rows.targets))
                or rows.targets.dtype!=np.float64 or (rows.weights<0).any()):
            raise ValueError('CPU payload has corrupt constraint arrays')
        canonical=QuadraticConstraints(rows.matrix,rows.weights,rows.targets,rows.groups)
        if any(not np.array_equal(a,b) for a,b in zip(rows._structural_arrays(),canonical._structural_arrays())):
            raise ValueError('CPU payload has inconsistent derived constraints')
        for a in (*rows._structural_arrays(),rows.targets):a.setflags(write=False)
    size=retained_bytes(values,rows)
    if type(block.retained_bytes) is not int or block.retained_bytes!=size or size>spec['payload_bytes']:
        raise ValueError('CPU payload byte accounting mismatch')
    info=block.worker_info
    if (info.get('cuda_initialized') is not False or info.get('threads')!=1
            or info.get('built_constraints') is not build_rows):
        raise ValueError('CPU worker startup contract failed')
    return block


class CPUFieldInputs:
    timeout_seconds=300.

    def __init__(self,spec,*,workers,max_bytes,cache=None,observations=None,stop=None):
        self.cache={} if cache is None else cache
        # Boxes whose observations and rows the parent retains are never resubmitted.
        self.observations={} if observations is None else observations
        self.stop=stop;self.pending=deque();self.active=None;self.pool=None;self.closed=False
        self.boxes=[(tuple(map(int,lo)),tuple(map(int,hi))) for lo,hi in spec['boxes']]
        self.next_index=0
        run=Path(spec['run'])
        # Bound source decompression separately from retained/IPC payloads.
        with closing(Tiles(run/'maps',readonly=True)) as maps:
            source=0
            for key in maps.keys():
                with zipfile.ZipFile(maps.path(key)) as archive:
                    source=max(source,sum(i.file_size for i in archive.infolist()))
        requested=workers
        nodes=max(int(np.prod(np.subtract(hi,lo))) for lo,hi in self.boxes)
        workers=admit_workers(workers,max_bytes,spec.get('cache_bytes',0),
                              parent_threads=torch.get_num_threads(),source_bytes=16*source+256*nodes)
        self.slots=min(2*workers,len(self.boxes))
        self.slot_bytes=max_bytes//(self.slots+1)
        spec=dict(spec,payload_bytes=self.slot_bytes//4,curve_bytes=max(4096,self.slot_bytes//8),
                  max_pairs=self.slot_bytes//2048)
        if spec['max_pairs']<1 or 4096+17*nodes>spec['payload_bytes']:
            raise ResourceUnavailable('CPU buffer too small for block observations/preparation slot')
        self.spec=spec
        self.stats=dict(workers=workers,requested_workers=requested,completed=0,submitted=0,
                        skipped=0,constraint_builds=0,peak_buffer_bytes=0,buffer_bytes=max_bytes,
                        wait_seconds=0.,worker_info={})
        context=mp.get_context('spawn')
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=_initialize,initargs=(spec,))

    def _fill(self):
        while len(self.pending)+(self.active is not None)<self.slots and self.next_index<len(self.boxes):
            lo,hi=self.boxes[self.next_index];key=(*lo,*hi);build=key not in self.cache
            if not build and key in self.observations:
                self.next_index+=1;self.stats['skipped']+=1
                continue
            with _spawn_environment():
                future=self.pool.submit(prepare_static_block,self.spec,lo,hi,build)
            self.pending.append((lo,hi,build,future));self.next_index+=1
            self.stats['submitted']+=1;self.stats['constraint_builds']+=int(build)
            self.stats['peak_buffer_bytes']=max(self.stats['peak_buffer_bytes'],
                (len(self.pending)+1)*self.slot_bytes)

    def load(self,lo,hi):
        if self.closed:raise RuntimeError('CPU pool is closed')
        self.active=None
        bounds=(tuple(map(int,lo)),tuple(map(int,hi)))
        if not self.pending:
            self.next_index=self.boxes.index(bounds)
        elif self.pending[0][:2]!=bounds:
            raise ValueError('CPU observations requested out of solver order')
        self._fill()
        low,high,build,future=self.pending[0];started=time.monotonic()
        while True:
            if self.stop and self.stop():raise InterruptedError('CPU preparation stopped')
            remaining=self.timeout_seconds-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('CPU preparation timed out')
            try:
                block=future.result(timeout=min(.05,remaining));break
            except TimeoutError:
                if future.done():raise
        self.stats['wait_seconds']+=time.monotonic()-started
        self.active=validate_block(block,self.spec,low,high,build_rows=build)
        self.pending.popleft()
        self.stats['completed']+=1
        self.stats['worker_info'][str(block.worker_info['pid'])]=block.worker_info
        return block.observations

    def constraints(self,lo,hi,read_nodes):
        if self.active is None or (self.active.lo,self.active.hi)!=(tuple(lo),tuple(hi)):
            raise ValueError('CPU constraints requested without matching observations')
        return self.active.rows

    def close(self):
        if self.closed:return
        self.closed=True
        if self.pool is not None:
            for *_,future in self.pending:future.cancel()
            processes=list((self.pool._processes or {}).values())
            manager=self.pool._executor_manager_thread;queue=self.pool._result_queue
            self.pool.shutdown(wait=False,cancel_futures=True)
            deadline=time.monotonic()+2.
            for process in processes:process.join(max(0.,deadline-time.monotonic()))
            for process in processes:
                if process.is_alive():process.terminate()
            for process in processes:
                process.join(.5)
                if process.is_alive():process.kill();process.join()
            if manager is not None:manager.join(1.)
            if queue is not None:queue.close()
            self.pool._initargs=()
        self.pending.clear();self.active=None;self.pool=None
