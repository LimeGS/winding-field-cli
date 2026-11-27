"""Global fibre pairs on disk, bucketed by node position, for multi-block field solves.

The in-memory global pair set holds every pair row of the grid (about 280 bytes per pair
with its stencils), which does not fit for a whole scroll. This catalog stores the same
pairs, in the same order, in shards of consecutive curves; each shard is sorted by the
bucket of its pairs' first endpoint. A block reads only the buckets near its crop, so a
restriction returns exactly the rows the in-memory set would return for that crop and
the field does not change.

Shards are built in worker processes and are resumable: a finished shard is published
atomically and skipped on the next build. Shard boundaries depend only on the curves, not
on the number of workers.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from .util import atomic_json

ALGORITHM = "global-grid-pairs-buckets-v1"
BUCKET = 32  # nodes per bucket edge
SHARD_BYTES = 64 << 20  # curve point bytes per shard


def _grid(geometry, shape):
    origin = np.asarray(geometry["origin_xyz"], float)
    spacing = np.asarray(geometry["spacing_xyz"], float)
    shape = np.asarray(shape, np.int64)
    return origin, spacing, shape, origin + (shape[::-1] - 1) * spacing


def _buckets(shape, bucket):
    return (np.asarray(shape, np.int64) + bucket - 1) // bucket


def _identity(fibres_sha, geometry, shape):
    return dict(
        algorithm=ALGORITHM,
        fibres_sha256=fibres_sha,
        origin_xyz=[float(v) for v in geometry["origin_xyz"]],
        spacing_xyz=[float(v) for v in geometry["spacing_xyz"]],
        shape_zyx=[int(n) for n in shape],
        bucket=BUCKET,
        shard_bytes=SHARD_BYTES,
    )


def _shards(db):
    """Consecutive curve-id ranges of at most SHARD_BYTES of point data (at least one curve)."""
    ranges, first, size, last = [], None, 0, None
    for curve, length in db.execute("SELECT id, length(points) FROM curves ORDER BY id"):
        if first is not None and size + length > SHARD_BYTES:
            ranges.append((first, last))
            first, size = None, 0
        if first is None:
            first = curve
        size += length
        last = curve
    if first is not None:
        ranges.append((first, last))
    return ranges


def _build_shard(task):
    """Pairs of the curves ``first..last`` in curve order, sorted by bucket into ``out``."""
    from .field import spline_pairs

    source, geometry, shape, bucket_size, first, last, out = task
    origin, spacing, shape, upper = _grid(geometry, shape)
    db = sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)
    left, right = [], []
    try:
        rows = db.execute(
            "SELECT points FROM curves WHERE id BETWEEN ? AND ? ORDER BY id", (first, last)
        )
        for (raw,) in rows:
            points = np.load(io.BytesIO(raw), allow_pickle=False)
            a, b = spline_pairs(points, origin, upper, min(spacing))
            if len(a):
                left.append(a)
                right.append(b)
    finally:
        db.close()
    pairs = (
        np.stack([np.concatenate(left), np.concatenate(right)], 1) if left else np.empty((0, 2, 3))
    )
    node = np.floor((pairs[:, 0] - origin) / spacing).astype(np.int64)[:, ::-1]  # ZYX
    node = np.clip(node, 0, shape - 1)
    grid = _buckets(shape, bucket_size)
    bucket = np.ravel_multi_index(tuple((node // bucket_size).T), grid)
    order = np.argsort(bucket, kind="stable")
    bucket = bucket[order]
    keys, starts = np.unique(bucket, return_index=True)
    temporary = Path(str(out) + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    arrays = dict(
        pairs=np.ascontiguousarray(pairs[order]),
        local=order.astype(np.int64),
        keys=keys.astype(np.int64),
        starts=np.r_[starts, len(bucket)].astype(np.int64),
    )
    digest = hashlib.sha256()
    for name, array in arrays.items():
        np.save(temporary / f"{name}.npy", array)
        digest.update((temporary / f"{name}.npy").read_bytes())
    os.replace(temporary, out)
    return len(pairs), digest.hexdigest()


def build(root, source, fibres_sha, geometry, shape, *, workers=None, stop=None):
    """Build or finish the catalog at ``root``; returns "COMPLETE" or "PAUSED"."""
    root = Path(root)
    identity = _identity(fibres_sha, geometry, shape)
    marker = root / "PAIRS.json"
    if marker.exists():
        if json.loads(marker.read_bytes())["identity"] != identity:
            raise ValueError(
                f"global pair catalog belongs to other fibres or another grid; remove {root}"
            )
        return "COMPLETE"
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        ranges = _shards(db)
    finally:
        db.close()
    tasks = [
        (
            str(source),
            geometry,
            list(map(int, shape)),
            identity["bucket"],
            a,
            b,
            str(root / f"shard-{i:05d}"),
        )
        for i, (a, b) in enumerate(ranges)
    ]
    done = {}
    pending = []
    for i, task in enumerate(tasks):
        if Path(task[-1]).is_dir():
            done[i] = None
        else:
            pending.append((i, task))
    workers = max(1, min(workers or os.cpu_count() or 1, len(pending) or 1))
    if pending:
        if workers == 1:
            for i, task in pending:
                if stop and stop():
                    return "PAUSED"
                done[i] = _build_shard(task)
        else:
            from .import_parallel import _spawn_environment

            context = multiprocessing.get_context("spawn")
            with _spawn_environment(), ProcessPoolExecutor(workers, mp_context=context) as pool:
                futures = {i: pool.submit(_build_shard, task) for i, task in pending}
                for i, future in futures.items():
                    if stop and stop():
                        for other in futures.values():
                            other.cancel()
                        return "PAUSED"
                    done[i] = future.result()
    shards = []
    for i, task in enumerate(tasks):
        out = Path(task[-1])
        if done[i] is None:
            digest = hashlib.sha256()
            for name in ("pairs", "local", "keys", "starts"):
                digest.update((out / f"{name}.npy").read_bytes())
            count = int(np.load(out / "local.npy", mmap_mode="r").shape[0])
            done[i] = (count, digest.hexdigest())
        shards.append(dict(name=out.name, count=int(done[i][0]), sha256=done[i][1]))
    atomic_json(
        marker, dict(identity=identity, shards=shards, count=sum(s["count"] for s in shards))
    )
    return "COMPLETE"


class GlobalPairs:
    """Restriction provider over a finished catalog (``__call__`` and ``exterior``).

    ``cache_bytes`` bounds retained templates (default none: a two-level sweep visits every
    block twice in two separate passes, so only a cache holding all blocks would hit).
    """

    def __init__(self, root, geometry, shape, weight, *, cache_bytes=0):
        root = Path(root)
        manifest = json.loads((root / "PAIRS.json").read_bytes())
        self.identity = manifest["identity"]
        if self.identity["shape_zyx"] != [int(n) for n in shape]:
            raise ValueError("global pair catalog belongs to another grid")
        self.count = int(manifest["count"])
        self.digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.origin, self.spacing, self.shape, _ = _grid(geometry, shape)
        self.weight = weight
        self.cache_bytes = cache_bytes
        self.stats = dict(bytes=0, hits=0, misses=0)
        self._templates: OrderedDict = OrderedDict()
        self._recent: OrderedDict = OrderedDict()
        self._pool = None
        self._futures: dict = {}
        self.root, self.geometry = root, geometry
        self._shards = []
        for shard in manifest["shards"]:
            path = root / shard["name"]
            arrays = {
                name: np.load(path / f"{name}.npy", mmap_mode="r")
                for name in ("pairs", "local", "keys", "starts")
            }
            if len(arrays["local"]) != shard["count"]:
                raise ValueError(f"global pair shard {shard['name']} is incomplete")
            self._shards.append(arrays)

    def __len__(self):
        return self.count

    def prepared_key(self, lo, hi):
        return self.digest

    def candidates(self, lo, hi):
        """Pairs that can touch nodes ``lo..hi``, in catalog (curve) order, as (N, 2, 3) XYZ."""
        size = self.identity["bucket"]
        grid = _buckets(self.shape, size)
        # A pair's nodes lie within [floor(a) - 1, floor(a) + 2] per axis (its length is at
        # most one grid step), so only first endpoints at nodes lo - 2 .. hi can touch the box.
        first = np.maximum(np.asarray(lo) - 2, 0) // size
        last = np.minimum(np.asarray(hi), self.shape - 1) // size
        z, y = np.meshgrid(
            np.arange(first[0], last[0] + 1), np.arange(first[1], last[1] + 1), indexing="ij"
        )
        line = (z.ravel() * grid[1] + y.ravel()) * grid[2]
        parts = []
        for arrays in self._shards:
            keys, starts = arrays["keys"], arrays["starts"]
            a = np.searchsorted(keys, line + first[2], side="left")
            b = np.searchsorted(keys, line + last[2], side="right")
            counts = starts[b] - starts[a]
            if not counts.sum():
                continue
            begin = np.repeat(starts[a] - np.cumsum(counts) + counts, counts)
            rows = np.sort(begin + np.arange(int(counts.sum())))
            local = np.asarray(arrays["local"][rows])
            parts.append(np.asarray(arrays["pairs"][rows])[np.argsort(local, kind="stable")])
        return np.concatenate(parts) if parts else np.empty((0, 2, 3))

    def _build(self, lo, hi):
        from constraint_templates import StencilConstraints

        pairs = self.candidates(lo, hi)
        if not len(pairs):
            return None
        rows = StencilConstraints.pairs(
            pairs[:, 0],
            pairs[:, 1],
            self.origin,
            self.spacing,
            tuple(map(int, self.shape)),
            weight=self.weight,
        )
        return rows.prepare(lo, hi)

    def prefetch(self, boxes, workers):
        """Build the templates of ``boxes`` (the solve's block crops, in visiting order) ahead
        of the solve in ``workers`` processes. A sweep visits every block twice (coarse
        residual pass, then local solves), each time in this order, so the pool keeps the
        next ``2 * workers`` boxes in flight, wrapping around."""
        if workers < 2 or len(boxes) < 2:
            return
        self._boxes = [tuple(map(int, (*lo, *hi))) for lo, hi in boxes]
        self._index = {key: i for i, key in enumerate(self._boxes)}
        from .import_parallel import _spawn_environment

        self._spawn = _spawn_environment
        context = multiprocessing.get_context("spawn")
        self._pool = ProcessPoolExecutor(
            workers,
            mp_context=context,
            initializer=_open_worker,
            initargs=(str(self.root), self.geometry, list(map(int, self.shape)), self.weight),
        )

    def prepare(self, lo, hi, read_nodes=None):
        key = tuple(map(int, (*lo, *hi)))
        if key in self._recent:
            self._recent.move_to_end(key)
            self.stats["hits"] += 1
            return self._recent[key]
        if key in self._templates:
            self.stats["hits"] += 1
            self._templates.move_to_end(key)
            return self._templates[key]
        self.stats["misses"] += 1
        if self._pool is not None and key in self._index:
            start = self._index[key]
            for step in range(2 * self._pool._max_workers):
                ahead = self._boxes[(start + step) % len(self._boxes)]
                if ahead not in self._futures:
                    with self._spawn():
                        self._futures[ahead] = self._pool.submit(_build_worker, ahead)
            template = self._futures.pop(key).result()
            self.stats["prefetched"] = self.stats.get("prefetched", 0) + 1
        else:
            template = self._build(np.asarray(lo), np.asarray(hi))
        # The coarse pass asks for a block's rows and then its exterior terms.
        self._recent[key] = template
        while len(self._recent) > 2:
            self._recent.popitem(last=False)
        size = 4096 + (0 if template is None else template.nbytes)
        if size <= self.cache_bytes:
            while self.stats["bytes"] + size > self.cache_bytes:
                _, retired = self._templates.popitem(last=False)
                self.stats["bytes"] -= 4096 + (0 if retired is None else retired.nbytes)
            self._templates[key] = template
            self.stats["bytes"] += size
        return template

    def close(self):
        if self._pool is not None:
            for future in self._futures.values():
                future.cancel()
            self._futures.clear()
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __call__(self, lo, hi, read_nodes):
        template = self.prepare(np.asarray(lo), np.asarray(hi), read_nodes)
        return None if template is None else template.restrict(read_nodes)

    def exterior(self, lo, hi):
        template = self.prepare(np.asarray(lo), np.asarray(hi))
        return None if template is None else template.exterior()


_WORKER = None


def _open_worker(root, geometry, shape, weight):
    global _WORKER
    _WORKER = GlobalPairs(root, geometry, shape, weight)


def _build_worker(key):
    lo, hi = np.array(key[:3]), np.array(key[3:])
    template = _WORKER._build(lo, hi)
    if template is not None:
        template.live = None  # a weak reference does not pickle
    return template


def default_workers():
    """Three quarters of the CPUs this process may use: its affinity and its cgroup quota
    (a container on a 128-thread host may get 15 CPUs; os.cpu_count() reports 128)."""
    from .surface_parallel import cpu_limit

    count = max(1, int(cpu_limit()))
    return max(1, min(count, int(math.ceil(count * 0.75))))
