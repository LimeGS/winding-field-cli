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


def _lattice_axes(points):
    """Per-axis XYZ coordinates if ``points`` (Z, Y, X, 3) is an axis-aligned lattice, else None."""
    if points.ndim != 4 or points.shape[-1] != 3 or 0 in points.shape:
        return None
    x, y, z = points[0, 0, :, 0], points[0, :, 0, 1], points[:, 0, 0, 2]
    if (
        np.array_equal(points[..., 0], np.broadcast_to(x[None, None, :], points.shape[:-1]))
        and np.array_equal(points[..., 1], np.broadcast_to(y[None, :, None], points.shape[:-1]))
        and np.array_equal(points[..., 2], np.broadcast_to(z[:, None, None], points.shape[:-1]))
    ):
        return [x, y, z]
    return None


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
        """Accept a frame digest that differs only in CT reader metadata.

        The frame digests the whole CT identity, including the reader backend and the
        native reader binary, which runs from earlier releases did not record. Two runs
        on the same CT source, level, metadata, shape, spacing and origin share their
        physical coordinates whatever reader produced them.
        """
        prepared = self.run / "PREPARED.json"
        theirs = json.loads(prepared.read_bytes()).get("source") if prepared.is_file() else None
        message = f"field prior frame {prior.get('frame')!r} differs from {geometry.get('frame')!r}"
        if not isinstance(source, dict) or not isinstance(theirs, dict):
            raise ValueError(message + " and a CT source identity is unavailable to compare")
        mine, theirs = _physical(source), _physical(theirs)
        if mine != theirs:
            keys = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
            raise ValueError(message + "; CT identity differs in " + ", ".join(keys))
        return dict(rule="ct-identity-without-reader-runtime", ignored=list(_READER_KEYS))

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
        axes = _lattice_axes(points)
        if axes is not None:
            return self._interpolate_lattice(axes, points.shape[:-1], self.source, finite=False)
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

    def _interpolate_lattice(self, axes, shape, source, finite):
        """``_interpolate`` for an axis-aligned lattice given by its per-axis XYZ coordinates.

        The trilinear weights are products of per-axis weights, so the eight corners are
        eight outer-indexed crops instead of eight gathers of one index triple per point.
        Same operations in the same order as the point path: identical results.
        """
        values = np.full(shape, np.nan)
        valid = np.zeros(shape, bool)
        keep, base, frac = [], [], []
        for d in range(3):  # ZYX
            index = (axes[2 - d] - self.origin[2 - d]) / self.spacing[2 - d]
            inside = (index >= 0) & (index <= self.shape[d] - 1)
            b = np.minimum(np.floor(index[inside]).astype(int), max(self.shape[d] - 2, 0))
            keep.append(np.flatnonzero(inside))
            base.append(b)
            frac.append(index[inside] - b)
        if all(len(k) for k in keep):
            lo = np.array([b.min() for b in base])
            hi = np.minimum(np.array([b.max() for b in base]) + 2, self.shape)
            crop, support = source.read(lo, hi)
            acc = np.zeros(tuple(len(k) for k in keep))
            ok = np.ones(acc.shape, bool)
            for corner in np.ndindex(2, 2, 2):
                at = np.ix_(
                    *(np.minimum(base[d] - lo[d] + corner[d], crop.shape[d] - 1) for d in range(3))
                )
                wz, wy, wx = (frac[d] if corner[d] else 1 - frac[d] for d in range(3))
                weight = (wz[:, None, None] * wy[None, :, None]) * wx[None, None, :]
                corner_values = crop[at]
                acc += weight * corner_values
                ok &= support[at]
                if finite:
                    ok &= np.isfinite(corner_values)
            inside = np.ix_(*keep)
            values[inside] = np.where(ok, acc, np.nan)
            valid[inside] = ok
        return values, valid

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
