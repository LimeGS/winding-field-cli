"""Purely local Lasagna tile workers with parent-owned source pins and publication."""
from contextlib import ExitStack, closing
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .import_parallel import RESULT_BYTES, limits, ordered_map, worker_stopped
from .io import PinBudgetExceeded
from .util import atomic_json

_READER=None
_VOLUME=None
_GEOMETRY=None


def _initialize(config,work,geometry,physical,identity):
    from .workflow import open_volume
    from .lasagna_input import CalibratedLasagnaInput
    from .lasagna_stream import LasagnaInput
    global _READER,_VOLUME,_GEOMETRY
    root=Path(work)/str(os.getpid());root.mkdir()
    config=dict(config,cache=str(root/'cache'),_import_worker=True)
    _VOLUME=open_volume(config);_VOLUME.bind_ledger(root/'source.sqlite')
    _READER=(CalibratedLasagnaInput(_VOLUME,root,config,geometry) if physical else LasagnaInput(_VOLUME,root,config))
    if _READER.identity!=identity:raise ValueError('worker Lasagna input identity changed')
    _GEOMETRY=geometry


def _compute(job):
    if worker_stopped():raise InterruptedError('local import paused')
    key,lo,hi=job
    products=dict(_READER.products,ct=_VOLUME)
    before={name:sum(p._stats[name] for p in products.values()) for name in ('decodes','hashed_bytes','chunk_commits')}
    budget=dict(remaining=RESULT_BYTES-4096);pins={}
    try:
        with ExitStack() as stack:
            for name,product in products.items():
                pins[name]=stack.enter_context(product.capture_pins(budget))
                stack.enter_context(product.verified_chunks(pins=pins[name]))
            arrays=_READER.read(_GEOMETRY,lo,hi)
    except PinBudgetExceeded:
        arrays=pins=None
    if arrays is not None and sum(a.nbytes for a in arrays.values())>budget['remaining']:
        arrays=pins=None
    stats={name:sum(p._stats[name] for p in products.values())-before[name] for name in before}
    return key,arrays,pins,dict(stats,pid=os.getpid())


def observations(reader,volume,geometry,maps,config,stack,stop):
    """Return ordered missing-tile results, or retain serial/remote reading."""
    from .lasagna_input import CalibratedLasagnaInput
    local=not volume.remote and not any(p.remote for p in reader.products.values())
    requested=config.get('import_workers',1)
    largest=max(math.prod(p.chunks)*p.dtype.itemsize for p in (*reader.products.values(),volume))
    # Charge a parent replay as well as worker interpolation/pin scratch.
    scratch=(1280 if isinstance(reader,CalibratedLasagnaInput) else 256)*maps.chunk**3
    execution=(limits(requested,output=maps.root,reserve_bytes=config.get('reserve_bytes',0),
                      worker_bytes=(768<<20)+8*largest+scratch+2*RESULT_BYTES)
               if local and config.get('fibres') else dict(workers=1,max_inflight=2,requested=requested,
                   limiting_resource='remote-input' if not local else 'parent-inference'))
    execution.update(worker_pids=[],decodes=0,hashed_bytes=0,chunk_commits=0,streaming_replays=0,complete=False)
    if volume._ledger is None and execution['workers']>1:
        execution.update(workers=1,max_inflight=2,limiting_resource='unbound-ct-ledger')
    stack.callback(lambda:atomic_json(reader.root/'PERFORMANCE.json',execution))
    if execution['workers']==1:return None,execution
    budget=(256<<20)//(execution['workers']+1)
    for product in (*reader.products.values(),volume):
        product._decoded.clear();product._decoded_bytes=0
        product._decoded_limit=min(product._decoded_limit,budget//6)
    worker_config=dict(config,source=volume.source,level=volume.identity['level'],
                       local=str(volume.local) if volume.local else None,_import_cache_bytes=budget)
    work=stack.enter_context(TemporaryDirectory(prefix='.import-workers-',dir=maps.root.parent))
    jobs=((key,*maps.box(key)) for key in maps.keys() if not maps.has(key))
    results=stack.enter_context(closing(ordered_map(_compute,jobs,
        workers=execution['workers'],max_inflight=execution['max_inflight'],initializer=_initialize,
        initargs=(worker_config,work,geometry,isinstance(reader,CalibratedLasagnaInput),reader.identity),stop=stop)))
    def consume():
        for key,arrays,pins,stats in results:
            if stats['pid'] not in execution['worker_pids']:execution['worker_pids'].append(stats['pid'])
            for name in ('decodes','hashed_bytes','chunk_commits'):execution[name]+=stats[name]
            if arrays is None:
                execution['streaming_replays']+=1
                arrays=reader.read(geometry,*maps.box(key))
            else:
                if set(pins)!=set(reader.products)|{'ct'}:raise ValueError('worker source pin inventory changed')
                for name,values in pins.items():
                    (volume if name=='ct' else reader.products[name]).merge_pins(values)
            yield key,arrays
    return stack.enter_context(closing(consume())),execution
