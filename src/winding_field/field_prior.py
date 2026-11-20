"""A completed (usually coarser) field as the starting point and gauge of a new solve.

``run --field-prior PRIOR_RUN`` interpolates the prior's W onto the new grid as the
initial field and pins one node to the prior's value, so the new field shares the
prior's origin and scale: a fine tile and its coarse parent report the same W for the
same sheet, and neighbouring tiles agree. The prior only seeds and gauges the solve;
the observations and fibres still determine the result.
"""

import json
from pathlib import Path

import numpy as np

from .util import file_sha


class FieldPrior:
    def __init__(self, path, geometry):
        from .field import open_field

        self.run = Path(path).expanduser().resolve()
        meta_path = self.run / "FIELD.json"
        if not meta_path.is_file():
            raise ValueError(f"field prior has no FIELD.json: {self.run}")
        meta = json.loads(meta_path.read_bytes())
        prior = meta["geometry"]
        if prior.get("frame") != geometry.get("frame"):
            raise ValueError(
                f"field prior frame {prior.get('frame')!r} differs from {geometry.get('frame')!r}"
            )
        if float(prior.get("sheet_step", 1.0)) != float(geometry.get("sheet_step", 1.0)):
            raise ValueError("field prior sheet_step differs")
        self.geometry = prior
        self.origin = np.asarray(prior["origin_xyz"], float)
        self.spacing = np.asarray(prior["spacing_xyz"], float)
        self.shape = np.asarray(prior["shape_zyx"], int)
        self.source = open_field(self.run)
        self.identity = dict(
            run=str(self.run),
            field_json_sha256=file_sha(meta_path),
            checkpoint_sha256=meta["checkpoint_sha256"],
            interpolation="trilinear-all-corners-supported",
        )

    def close(self):
        self.source.close()

    def sample(self, xyz):
        """Trilinear W at world points ``xyz`` (..., 3); valid only if all 8 corners are supported."""
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
            crop, support = self.source.read(lo, hi)
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
            values[inside] = np.where(ok, acc, np.nan)
            valid[inside] = ok
        return values.reshape(points.shape[:-1]), valid.reshape(points.shape[:-1])

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
