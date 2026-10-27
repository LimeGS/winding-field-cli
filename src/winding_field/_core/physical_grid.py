"""Physical XYZ grids over ZYX arrays, with no implicit resize or edge clamping."""

import hashlib
import itertools

import numpy as np
from scipy.ndimage import affine_transform


def point_stencils(xyz, origin_xyz, spacing_xyz, shape_zyx, support=None):
    """Flat ZYX nodes and positive trilinear weights, no outside clamping.

    At the final node the preceding cell gives the exact boundary value. Only
    nodes with nonzero coefficients require support; invalid rows carry zero weights.
    """
    xyz = np.asarray(xyz, float)
    origin = np.asarray(origin_xyz, float)
    h = np.asarray(spacing_xyz, float)
    shape = np.asarray(shape_zyx)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or origin.shape != (3,)
        or h.shape != (3,)
        or shape.shape != (3,)
        or shape.dtype.kind not in "iu"
        or (shape < 2).any()
        or not np.isfinite([origin, h]).all()
        or (h <= 0).any()
    ):
        raise ValueError("invalid physical XYZ grid/stencil input")
    if support is not None and (support.shape != tuple(shape) or support.dtype.kind != "b"):
        raise ValueError("support must be a matching boolean array")
    grid = (xyz - origin) / h
    valid = np.isfinite(grid).all(1) & (grid >= 0).all(1) & (grid <= shape[::-1] - 1).all(1)
    base = np.zeros_like(grid, dtype=np.int64)
    base[valid] = np.minimum(np.floor(grid[valid]).astype(int), shape[::-1] - 2)
    fraction = np.where(valid[:, None], grid - base, 0)
    nodes = []
    weights = []
    for corner in itertools.product((0, 1), repeat=3):
        q = base + corner
        node = np.ravel_multi_index(q[:, ::-1].T, shape)
        w = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        if support is not None:
            valid &= (w == 0) | support.ravel()[node]
        nodes.append(node)
        weights.append(w)
    nodes = np.stack(nodes, axis=1)
    weights = np.stack(weights, axis=1)
    weights[~valid] = 0
    return nodes, weights, valid


def grid_shape(lo_xyz, hi_xyz, spacing_xyz, shape_zyx=None):
    lo, hi, spacing = (np.asarray(a, float) for a in (lo_xyz, hi_xyz, spacing_xyz))
    if any(a.shape != (3,) for a in (lo, hi, spacing)) or not np.isfinite([lo, hi, spacing]).all():
        raise ValueError("expected finite XYZ vectors")
    if (hi <= lo).any() or (spacing <= 0).any():
        raise ValueError("nonpositive extent/spacing")
    if shape_zyx is not None:
        shape = np.asarray(shape_zyx)
        if shape.shape != (3,) or shape.dtype.kind not in "iu" or (shape < 1).any():
            raise ValueError("declared shape must contain positive integer ZYX dimensions")
        expected_hi = lo + shape[::-1] * spacing
        tolerance = 4 * np.spacing(np.maximum(np.abs(hi), np.abs(expected_hi)))
        if not np.isfinite(expected_hi).all() or (np.abs(hi - expected_hi) > tolerance).any():
            raise ValueError("declared shape disagrees with physical bounds")
        return tuple(int(v) for v in shape)
    extent = (hi - lo) / spacing
    nearest = np.rint(extent)
    near_integer = (nearest >= 1) & (np.abs(extent - nearest) <= 4 * np.spacing(np.maximum(np.abs(nearest), 1)))
    extent = np.where(near_integer, nearest, extent)
    return tuple(int(v) for v in np.ceil(extent)[::-1])


def sample_volume(
    volume, origin_xyz, spacing_xyz, shape_zyx, source_spacing_xyz, source_origin_xyz=(0, 0, 0), order=1, receipts=None
):
    origin, spacing, src_spacing, src_origin = (
        np.asarray(a, float) for a in (origin_xyz, spacing_xyz, source_spacing_xyz, source_origin_xyz)
    )
    if any(a.shape != (3,) for a in (origin, spacing, src_spacing, src_origin)):
        raise ValueError("expected XYZ vectors")
    if (
        not np.isfinite([origin, spacing, src_spacing, src_origin]).all()
        or (spacing <= 0).any()
        or (src_spacing <= 0).any()
    ):
        raise ValueError("invalid physical grid")
    shape = np.asarray(shape_zyx, int)
    if shape.shape != (3,) or (shape < 1).any():
        raise ValueError("invalid ZYX shape")
    first = ((origin - src_origin) / src_spacing)[::-1]
    ratio = (spacing / src_spacing)[::-1]
    last = first + (shape - 1) * ratio
    lo = np.floor(first).astype(int)
    hi = np.ceil(last).astype(int) + 1
    if (lo < 0).any() or (hi > volume.shape).any():
        raise ValueError("requested grid extends outside source support")
    crop = np.asarray(volume[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))])
    if receipts is not None:
        receipts.append(
            dict(
                source_shape_zyx=list(volume.shape),
                crop_lo_zyx=lo.tolist(),
                crop_hi_zyx=hi.tolist(),
                source_spacing_xyz=src_spacing.tolist(),
                dtype=str(crop.dtype),
                crop_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
            )
        )
    result = affine_transform(
        crop.astype(np.float32),
        np.diag(ratio),
        offset=first - lo,
        output_shape=tuple(shape),
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite sampled volume")
    return result


def offset_correction(a, b):
    delta = np.asarray(a) - np.asarray(b)
    if not np.isfinite(delta).any():
        raise ValueError("empty overlap")
    return float(np.median(delta[np.isfinite(delta)]))
