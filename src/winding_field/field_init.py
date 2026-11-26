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
