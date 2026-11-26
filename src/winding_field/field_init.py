"""Initial field values of every chunk, computed in worker processes.

The first checkpoint of an ordinary field holds, per chunk, the radial (or prior) start and
the observation support. Each chunk is independent, so workers compute them and the caller
commits them in solver order: the bytes are those of the one-process loop.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np

_STATE = None
# A field smaller than this samples its observations in process: spawning workers (each
# imports torch and opens the maps and the prior) costs more than they save, and many small
# solves at once (a test suite) would oversubscribe the CPUs.
POOL_MIN_NODES = 1 << 24


def _open(spec):
    import torch

    from .artifacts import Tiles

    global _STATE
    torch.set_num_threads(1)
    maps = Tiles(Path(spec["run"]) / "maps", readonly=True)
    if maps.identity() != spec["meta"]["tiles_sha256"]:
        maps.close()
        raise ValueError("initial-field worker: maps changed")
    prior = None
    if spec["prior"] is not None:
        from .field_prior import FieldPrior

        prior = FieldPrior(spec["prior"]["path"], spec["geometry"], spec["meta"].get("source"))
        if prior.identity["field_json_sha256"] != spec["prior"]["field_json_sha256"]:
            prior.close()
            maps.close()
            raise ValueError("initial-field worker: field prior changed")
    _STATE = (spec, maps, prior)


def initial_chunk(spec, maps, prior, lo, hi):
    """(start, valid) of the chunk ``lo..hi``, exactly as the solve's first checkpoint."""
    from .field import coordinates, initial_field, sample_prior

    orientation = prior if spec["orientation"] == "prior-gradient" else None
    _, _, valid = sample_prior(
        maps,
        spec["meta"],
        spec["geometry"],
        spec["calibration"],
        lo,
        hi,
        orientation,
        None,
        spec["sheet"],
    )
    xyz = coordinates(lo, hi, spec["geometry"])
    start = initial_field(xyz, spec["geometry"], spec["calibration"])
    if prior is not None:
        start = prior.initial(xyz, start)
    return start, valid


def _worker(key, chunk, shape):
    spec, maps, prior = _STATE
    lo = np.array(key) * chunk
    hi = np.minimum(lo + chunk, shape)
    return initial_chunk(spec, maps, prior, lo, hi)


def initial_chunks(spec, keys, chunk, shape, workers):
    """Yield ``(key, start, valid)`` for ``keys`` in order, ``workers`` processes ahead."""
    from .import_parallel import _spawn_environment

    keys = [tuple(map(int, key)) for key in keys]
    shape = tuple(map(int, shape))
    context = multiprocessing.get_context("spawn")
    with _spawn_environment():
        pool = ProcessPoolExecutor(workers, mp_context=context, initializer=_open, initargs=(spec,))
    pending = deque()
    position = 0
    try:
        while position < len(keys) or pending:
            while position < len(keys) and len(pending) < 2 * workers:
                with _spawn_environment():
                    pending.append(
                        (keys[position], pool.submit(_worker, keys[position], chunk, shape))
                    )
                position += 1
            key, future = pending.popleft()
            start, valid = future.result()
            yield key, start, valid
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def _observe(lo, hi):
    """Observations of ``lo..hi`` and the orientation counts of that box alone."""
    from .field import sample_prior

    spec, maps, prior = _STATE
    orientation = prior if spec["orientation"] == "prior-gradient" else None
    stats = {} if orientation is not None else None
    values = sample_prior(
        maps,
        spec["meta"],
        spec["geometry"],
        spec["calibration"],
        np.asarray(lo),
        np.asarray(hi),
        orientation,
        stats,
        spec["sheet"],
    )
    return values, stats


class ObservationPool:
    """Observations of the solve's boxes, computed ``workers`` processes ahead.

    A sweep visits every block twice (coarse residual pass, then local solves), each time
    in ``boxes`` order, so the pool keeps the next ``2 * workers`` boxes in flight, wrapping
    around. ``retained(key)`` skips boxes the caller already holds. One caller thread.
    """

    def __init__(self, spec, boxes, workers):
        from .import_parallel import _spawn_environment

        self.boxes = [tuple(map(int, (*lo, *hi))) for lo, hi in boxes]
        self.index = {key: i for i, key in enumerate(self.boxes)}
        self.workers = workers
        self.futures = {}
        self.stats = dict(prefetched=0, inline=0)
        self._spawn = _spawn_environment
        with _spawn_environment():
            self.pool = ProcessPoolExecutor(
                workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_open,
                initargs=(spec,),
            )

    def get(self, lo, hi, retained=lambda key: False):
        """``(values, stats)`` of ``lo..hi``, or None for a box outside the solve order."""
        key = tuple(map(int, (*lo, *hi)))
        if key not in self.index:
            self.stats["inline"] += 1
            return None
        start = self.index[key]
        for step in range(2 * self.workers):
            ahead = self.boxes[(start + step) % len(self.boxes)]
            if ahead in self.futures or (step and retained(ahead)):
                continue
            with self._spawn():
                self.futures[ahead] = self.pool.submit(_observe, ahead[:3], ahead[3:])
        self.stats["prefetched"] += 1
        return self.futures.pop(key).result()

    def close(self):
        if self.pool is not None:
            for future in self.futures.values():
                future.cancel()
            self.futures.clear()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
