"""Bounded, invocation-owned support snapshots for exact AFV clipping."""

from collections import OrderedDict
import itertools
import math

import numpy as np

MAX_INTERVALS = 65536


class SupportTiles:
    def __init__(self, maps, *, cache_bytes):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("invalid support cache budget")
        self.maps = maps
        self.shape = maps.shape
        self.chunk = maps.chunk
        self.limit = cache_bytes
        self.bytes = 0
        self.cache = OrderedDict()
        self.loads = 0

    def load(self, key, names=("support",)):
        if names != ("support",):
            raise ValueError("support cache only reads support")
        key = tuple(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        data = self.maps.load(key, names)
        self.loads += 1
        for array in data.values():
            array.flags.writeable = False
        charge = 512 + sum(a.nbytes for a in data.values())
        if charge <= self.limit:
            while self.bytes + charge > self.limit:
                _, old = self.cache.popitem(last=False)
                self.bytes -= 512 + sum(a.nbytes for a in old.values())
            self.cache[key] = data
            self.bytes += charge
        return data

    def close(self):
        self.cache.clear()
        self.bytes = 0


def _supported(points, maps, stop):
    q = points[:, ::-1]
    low = np.floor(q).astype(np.int64)
    fraction = q - low
    valid = np.ones(len(q), bool)
    dense = getattr(maps, "dense", None)
    for corner in itertools.product((0, 1), repeat=3):
        if stop and stop():
            raise InterruptedError("AFV import paused")
        active = np.flatnonzero(
            valid & np.all(np.where(corner, fraction, 1 - fraction) > 0, axis=1)
        )
        if not len(active):
            continue
        ij = np.minimum(low[active] + corner, np.asarray(maps.shape) - 1)
        if dense is not None:
            # Shared snapshot: the same contributing-corner lookup without tiles.
            valid[active] &= dense[tuple(ij.T)]
            continue
        keys, inverse = np.unique(ij // maps.chunk, axis=0, return_inverse=True)
        for index, key in enumerate(keys):
            if stop and stop():
                raise InterruptedError("AFV import paused")
            selected = np.flatnonzero(inverse == index)
            tile = maps.load(tuple(key), ("support",))["support"]
            valid[active[selected]] &= tile[tuple((ij[selected] % maps.chunk).T)]
    return valid


def pieces(points, maps, stop=None):
    """Exact scalar arithmetic, batched interval/support evaluation, ordered fragments."""
    from .afv import _scalar_pieces

    if stop and stop():
        raise InterruptedError("AFV import paused")
    high = np.asarray(maps.shape)[::-1] - 1
    edges = []
    count = 0
    for a, b in zip(points[:-1], points[1:]):
        delta = b - a
        near = np.divide(-a, delta, out=np.full(3, -np.inf), where=delta != 0)
        far = np.divide(high - a, delta, out=np.full(3, np.inf), where=delta != 0)
        enter = max(0.0, np.minimum(near, far).max())
        leave = min(1.0, np.maximum(near, far).min())
        if leave <= enter or np.any((delta == 0) & ((a < 0) | (a > high))) or not np.any(delta):
            edges.append(None)
            continue
        start = np.clip(a + enter * delta, 0, high)
        end = np.clip(a + leave * delta, 0, high)
        count += 1 + sum(
            max(0, math.ceil(max(x, y)) - math.floor(min(x, y)) - 1) for x, y in zip(start, end)
        )
        if count > MAX_INTERVALS:
            yield from _scalar_pieces(points, maps, stop)
            return
        edges.append((start, end))
    batches = []
    for edge in edges:
        if stop and stop():
            raise InterruptedError("AFV import paused")
        if edge is None:
            batches.append(None)
            continue
        start, end = edge
        delta = end - start
        axes = []
        for axis in range(3):
            if not delta[axis]:
                continue
            values = np.arange(
                math.floor(min(start[axis], end[axis])) + 1, math.ceil(max(start[axis], end[axis]))
            )
            if delta[axis] < 0:
                values = values[::-1]
            axes.append((values - start[axis]) / delta[axis])
        times = np.unique(np.concatenate([*axes, np.array([1.0])]))
        times = times[times > 0]
        # Exceptional floating-point orderings retain the streaming reference.
        if np.any(times > 1):
            yield from _scalar_pieces(points, maps, stop)
            return
        batches.append((start, delta, times))
    # One support query for the whole block: the per-segment arithmetic is the
    # same element-wise IEEE operations as per edge, only batched.
    segments = []
    for batch in batches:
        if batch is None:
            segments.append(None)
            continue
        start, delta, times = batch
        previous = np.r_[0.0, times[:-1]]
        left = np.clip(start + previous[:, None] * delta, 0, high)
        right = np.clip(start + times[:, None] * delta, 0, high)
        segments.append((left, right))
    midpoints = [(left + right) / 2 for left, right in filter(None, segments)]
    flags = _supported(np.concatenate(midpoints), maps, stop) if midpoints else None
    offset = 0
    piece = []
    for segment in segments:
        if stop and stop():
            raise InterruptedError("AFV import paused")
        if segment is None:
            if len(piece) > 1:
                yield np.asarray(piece)
            piece = []
            continue
        left, right = segment
        supported = flags[offset : offset + len(left)]
        offset += len(left)
        boundaries = np.diff(np.r_[False, supported, False].astype(np.int8))
        for first, last in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            if piece and (first > 0 or not np.allclose(piece[-1], left[first], rtol=0, atol=1e-9)):
                if len(piece) > 1:
                    yield np.asarray(piece)
                piece = []
            if not piece:
                piece = [left[first], right[last - 1]]
            else:
                piece.append(right[last - 1])
            if last < len(supported):
                yield np.asarray(piece)
                piece = []
        if not supported.any() and piece:
            if len(piece) > 1:
                yield np.asarray(piece)
            piece = []
    if len(piece) > 1:
        yield np.asarray(piece)
