"""A completed (usually coarser) field as the starting point and gauge of a new solve.

``run --field-prior PRIOR_RUN`` interpolates the prior's W onto the new grid as the
initial field and pins one node to the prior's value, so the new field shares the
prior's origin and scale: a fine tile and its coarse parent report the same W for the
same sheet, and neighbouring tiles agree. The prior only seeds and gauges the solve;
the observations and fibres still determine the result.
"""

import itertools
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .util import file_sha


_READER_KEYS = ("backend", "native_runtime")


def _physical(source):
    return {k: v for k, v in source.items() if k not in _READER_KEYS}


def ct_differences(mine, theirs):
    """Keys in which two CT source identities differ, leaving out the reader that read them.

    A frame digests the whole CT identity, including the reader backend and the native
    reader binary, which each host builds for itself and which runs from earlier releases
    did not record. Two runs on the same CT source, level, metadata, shape, spacing and
    origin share their physical coordinates whatever reader produced them.
    """
    mine, theirs = _physical(mine), _physical(theirs)
    return sorted(
        k
        for k in set(mine) | set(theirs)
        if (k in mine) != (k in theirs) or mine.get(k) != theirs.get(k)
    )


def frame_match(**details):
    """Receipt for frames accepted under the ``ct_differences`` rule."""
    return dict(rule="ct-identity-without-reader-runtime", ignored=list(_READER_KEYS), **details)


class FieldPrior:
    def __init__(self, path, geometry, source=None):
        from .field import open_field

        self.run = Path(path).expanduser().resolve()
        meta_path = self.run / "FIELD.json"
        if not meta_path.is_file():
            raise ValueError(f"field prior has no FIELD.json: {self.run}")
        meta = json.loads(meta_path.read_bytes())
        prior = meta["geometry"]
        frame_match = None
        if prior.get("frame") != geometry.get("frame"):
            frame_match = self._same_ct(source, geometry, prior)
        if float(prior.get("sheet_step", 1.0)) != float(geometry.get("sheet_step", 1.0)):
            raise ValueError("field prior sheet_step differs")
        self.geometry = prior
        self.origin = np.asarray(prior["origin_xyz"], float)
        self.spacing = np.asarray(prior["spacing_xyz"], float)
        self.shape = np.asarray(prior["shape_zyx"], int)
        self.source = open_field(self.run)
        # Every stored node, observed or not: read only by gradient(), on first use.
        self._stored = None
        self.identity = dict(
            run=str(self.run),
            field_json_sha256=file_sha(meta_path),
            checkpoint_sha256=meta["checkpoint_sha256"],
            interpolation="trilinear-all-corners-supported",
        )
        if frame_match:
            self.identity["frame_match"] = frame_match

    def _same_ct(self, source, geometry, prior):
        """Accept a frame digest that differs only in CT reader metadata (``ct_differences``)."""
        prepared = self.run / "PREPARED.json"
        theirs = json.loads(prepared.read_bytes()).get("source") if prepared.is_file() else None
        message = f"field prior frame {prior.get('frame')!r} differs from {geometry.get('frame')!r}"
        if not isinstance(source, dict) or not isinstance(theirs, dict):
            raise ValueError(message + " and a CT source identity is unavailable to compare")
        keys = ct_differences(source, theirs)
        if keys:
            raise ValueError(message + "; CT identity differs in " + ", ".join(keys))
        return frame_match()

    def close(self):
        self.source.close()
        if self._stored is not None:
            self._stored.close()

    def _stored_source(self):
        if self._stored is None:
            from .field import open_field

            self._stored = open_field(self.run, support_policy="stored")
        return self._stored

    def sample(self, xyz):
        """Trilinear W at world points ``xyz`` (..., 3); valid only if all 8 corners are observed."""
        return self._interpolate(xyz, self.source, finite=False)

    def _interpolate(self, xyz, source, finite):
        """Trilinear W from ``source``: valid where all 8 corners are in its support (and,
        with ``finite``, hold finite values)."""
        points = np.asarray(xyz, float)
        flat = points.reshape(-1, 3)
        index = (flat - self.origin) / self.spacing  # XYZ grid coordinates
        zyx = index[:, ::-1]
        values = np.full(len(flat), np.nan)
        valid = np.zeros(len(flat), bool)
        inside = np.all((zyx >= 0) & (zyx <= self.shape - 1), axis=1)
        if inside.any():
            base = np.minimum(np.floor(zyx[inside]).astype(int), np.maximum(self.shape - 2, 0))
            lo = base.min(0)
            hi = np.minimum(base.max(0) + 2, self.shape)
            crop, support = source.read(lo, hi)
            local = base - lo
            frac = zyx[inside] - base
            acc = np.zeros(len(local))
            ok = np.ones(len(local), bool)
            for corner in np.ndindex(2, 2, 2):
                offset = np.asarray(corner)
                idx = np.minimum(local + offset, np.asarray(crop.shape) - 1)
                weight = np.prod(np.where(offset, frac, 1 - frac), axis=1)
                at = tuple(idx.T)
                acc += weight * crop[at]
                ok &= support[at]
                if finite:
                    ok &= np.isfinite(crop[at])
            values[inside] = np.where(ok, acc, np.nan)
            valid[inside] = ok
        return values.reshape(points.shape[:-1]), valid.reshape(points.shape[:-1])

    def gradient(self, xyz, max_points=1 << 22):
        """Central-difference gradient of the trilinear prior at world points ``xyz`` (..., 3).

        Axis i takes (W(x + h_i e_i) - W(x - h_i e_i)) / (2 h_i), with h_i the prior's own
        spacing. The result is continuous in ``xyz``, so a node that two tiles share gets
        the same gradient in both, whichever coarse cell face it lies on. The samples read
        the prior's STORED values, observed or not: a coarse solved over the whole scroll
        holds its best estimate between observations too, and only the sign is used. Valid
        where all six samples have 8 stored, finite corners; XYZ components in W per µm,
        zero where invalid. Points are taken ``max_points`` at a time to bound the
        temporary arrays.
        """
        source = self._stored_source()
        points = np.asarray(xyz, float)
        flat = points.reshape(-1, 3)
        result = np.zeros(flat.shape)
        valid = np.zeros(len(flat), bool)
        for start in range(0, len(flat), max_points):
            part = flat[start : start + max_points]
            ok = np.ones(len(part), bool)
            for axis in range(3):
                step = np.zeros(3)
                step[axis] = self.spacing[axis]
                ahead, ahead_ok = self._interpolate(part + step, source, finite=True)
                behind, behind_ok = self._interpolate(part - step, source, finite=True)
                result[start : start + len(part), axis] = (ahead - behind) / (2 * step[axis])
                ok &= ahead_ok & behind_ok
            valid[start : start + len(part)] = ok
        result[~valid] = 0.0
        return result.reshape(points.shape), valid.reshape(points.shape[:-1])

    def initial(self, xyz, fallback):
        """Prior W where supported; elsewhere ``fallback`` shifted onto the prior's gauge."""
        values, valid = self.sample(xyz)
        result = np.asarray(fallback, float).copy()
        if valid.any():
            offset = float(np.median(values[valid] - result[valid]))
            result += offset
            result[valid] = values[valid]
        return result

    def anchor(self, order, chunk, shape, coordinates):
        """First supported node in solve order, with the prior's W there."""
        for key in order:
            lo = np.array(key) * chunk
            hi = np.minimum(lo + chunk, shape)
            values, valid = self.sample(coordinates(lo, hi))
            if valid.any():
                local = np.argwhere(valid)[0]
                return tuple(int(v) for v in lo + local), float(values[tuple(local)])
        raise ValueError("field prior has no supported value inside the new grid")


def fixed_provider(anchor_index, anchor_value):
    """Exact fixed values for solve_blocks: only the gauge node, NaN (free) elsewhere."""
    anchor = np.asarray(anchor_index)

    def fixed(lo, hi):
        lo, hi = np.asarray(lo), np.asarray(hi)
        values = np.full(tuple(hi - lo), np.nan)
        if np.all(anchor >= lo) and np.all(anchor < hi):
            values[tuple(anchor - lo)] = anchor_value
        return values

    return fixed


def owned_chunks(order, chunk, shape, supported):
    """Chunks with at least one supported fine node, in solve order."""
    owned = []
    for key in order:
        lo = np.array(key) * chunk
        if supported(lo, np.minimum(lo + chunk, shape)):
            owned.append(tuple(int(k) for k in key))
    return owned


def scoped_fixed(anchor_index, anchor_value, owned, chunk, shape, start, cache_chunks=64):
    """Fixed values for a solve over ``owned`` chunks only (``partial_order``).

    Every node of an unowned chunk is held at ``start(lo, hi)`` evaluated over that whole
    chunk, the value initialization would have given it (the prior where it is supported),
    so an owned block's halo sees the prior instead of the unstored chunk's zeros. The
    gauge node is fixed as in ``fixed_provider``.
    """
    owned = set(owned)
    anchor = np.asarray(anchor_index)
    shape = np.asarray(shape)
    cache = OrderedDict()

    def chunk_values(key):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        lo = np.asarray(key) * chunk
        value = np.asarray(start(lo, np.minimum(lo + chunk, shape)), float)
        cache[key] = value
        if len(cache) > cache_chunks:
            cache.popitem(last=False)
        return value

    def fixed(lo, hi):
        lo, hi = np.asarray(lo), np.asarray(hi)
        values = np.full(tuple(hi - lo), np.nan)
        spans = (range(a // chunk, (b - 1) // chunk + 1) for a, b in zip(lo, hi))
        for key in itertools.product(*spans):
            if key in owned:
                continue
            origin = np.asarray(key) * chunk
            a, b = np.maximum(lo, origin), np.minimum(hi, origin + chunk)
            target = tuple(slice(x - low, y - low) for x, y, low in zip(a, b, lo))
            source = tuple(slice(x - o, y - o) for x, y, o in zip(a, b, origin))
            values[target] = chunk_values(key)[source]
        if np.all(anchor >= lo) and np.all(anchor < hi):
            values[tuple(anchor - lo)] = anchor_value
        return values

    return fixed
