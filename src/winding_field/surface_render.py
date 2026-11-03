"""Bounded CPU orthographic rasterization of world-space triangle batches."""

from collections.abc import Callable, Iterator
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _camera(bounds_xyz: ArrayLike, width: int, height: int, azimuth: float, elevation: float):
    bounds = np.asarray(bounds_xyz)
    if bounds.shape != (2, 3) or bounds.dtype.kind not in "fiu" or not np.isfinite(bounds).all():
        raise ValueError("bounds_xyz must be finite real (2, 3) lower/upper XYZ bounds")
    bounds = bounds.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = bounds[1] - bounds[0]
    if not np.isfinite(extent).all() or (extent < 0).any() or not (extent > 0).any():
        raise ValueError("bounds_xyz must be ordered with a finite nonzero extent")
    center = bounds[0] * 0.5 + bounds[1] * 0.5
    unit = float(extent.max())

    az, el = np.deg2rad([azimuth % 360, elevation])
    forward = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    if abs(elevation) == 90:
        forward = np.array([0.0, 0.0, -np.sign(elevation)])
    reference_up = [0.0, 0.0, 1.0]
    if abs(forward[2]) > 1 - 1e-12:
        reference_up = [0.0, 1.0, 0.0]
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.column_stack([right, up, forward])
    # AABB projection extents are the absolute basis weighted by world extents.
    projected_extent = (extent / unit) @ np.abs(basis[:, :2])
    available = np.array([width, height], dtype=float)
    positive = projected_extent > 0
    scale = (
        float(np.min(available[positive] / projected_extent[positive])) if positive.any() else float(min(width, height))
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        world_scale = scale / unit
    if not np.isfinite(world_scale) or world_scale <= 0:
        raise ValueError("bounds/view cannot be represented by a finite positive camera scale")
    return center, unit, basis, scale, world_scale


def _raster_triangle(projected: np.ndarray, rgba: np.ndarray, image: np.ndarray, depth: np.ndarray) -> None:
    height, width = depth.shape
    # Identical faces must perform identical arithmetic even when their windings differ.
    projected = projected[np.lexsort(projected.T[::-1])]
    lower = np.maximum(np.floor(projected[:, :2].min(0)), 0)
    upper = np.minimum(np.floor(projected[:, :2].max(0)) + 1, [width, height])
    if (lower >= upper).any():
        return
    # Barycentrics are scale invariant; normalize to avoid overflow outside the image.
    coordinate_scale = max(float(np.abs(projected[:, :2]).max()), 1.0)
    (x0, y0), (x1, y1), (x2, y2) = projected[:, :2] / coordinate_scale
    z0, z1, z2 = projected[:, 2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denominator == 0:
        return
    left, top = map(int, lower)
    right, bottom = map(int, upper)
    # Patch both axes: scratch memory never follows an unbounded face footprint.
    for y in range(top, bottom, 128):
        for x in range(left, right, 128):
            stop_y, stop_x = min(y + 128, bottom), min(x + 128, right)
            px = (np.arange(x, stop_x, dtype=float)[None, :] + 0.5) / coordinate_scale
            py = (np.arange(y, stop_y, dtype=float)[:, None] + 0.5) / coordinate_scale
            a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
            b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
            inside = (a >= 0) & (b >= 0) & (a + b <= 1)
            fragment_depth = z2 + a * (z0 - z2) + b * (z1 - z2)
            previous_depth = depth[y:stop_y, x:stop_x]
            # Strict comparison preserves the first streamed face on exact ties.
            winner = inside & (fragment_depth < previous_depth)
            previous_depth[winner] = fragment_depth[winner]
            image[y:stop_y, x:stop_x][winner] = rgba


def render_view(
    triangles: Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]],
    bounds_xyz: ArrayLike,
    *,
    width: int = 1024,
    height: int = 1024,
    azimuth: float = 0,
    elevation: float = 0,
) -> tuple[NDArray[np.uint8], dict]:
    """Rasterize all faces once, retaining only the current batch and image buffers.

    ``bounds_xyz`` is the full world AABB, shaped (2, 3), lower then upper.
    Angles are degrees: (0, 0) observes from +X, with +Y right and +Z up;
    elevation is in [-90, 90]. At the poles camera up falls back to +Y.
    Output rows run top to bottom. Camera ``basis_xyz`` columns are right,
    up, forward (towards the scene); ``scale`` is pixels per world unit.

    Each batch contains float64 (N, 3, 3) world XYZ and floating (N, 3) RGB
    in [0, 1]. Arrays are never modified. Pixel centers use barycentric
    coverage, minimum depth wins, and exact ties keep the first face.
    Both windings use geometric-normal lighting with a 0.55 ambient floor.
    Degenerate/offscreen faces are counted but contribute no fragments.
    """
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    width, height = int(width), int(height)
    for name, angle in (("azimuth", azimuth), ("elevation", elevation)):
        if isinstance(angle, (bool, np.bool_)) or not isinstance(angle, Real) or not np.isfinite(float(angle)):
            raise ValueError(f"{name} must be a finite real angle in degrees")
    azimuth, elevation = float(azimuth), float(elevation)
    if not -90 <= elevation <= 90:
        raise ValueError("elevation must be in [-90, 90] degrees")
    if not callable(triangles):
        raise ValueError("triangles must be a callable returning triangle/color batches")
    center, unit, basis, scale, world_scale = _camera(bounds_xyz, width, height, azimuth, elevation)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    light = np.array([0.3, -0.4, np.sqrt(0.75)])
    processed = 0
    for batch in triangles():
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("expected a pair of triangle and RGB arrays")
        world, colors = batch
        if (
            not isinstance(world, np.ndarray)
            or world.ndim != 3
            or world.shape[1:] != (3, 3)
            or world.dtype != np.float64
            or not isinstance(colors, np.ndarray)
            or colors.shape != (len(world), 3)
            or colors.dtype.kind != "f"
        ):
            raise ValueError("expected float64 (N, 3, 3) XYZ and floating (N, 3) RGB arrays")
        if not np.isfinite(world).all() or not np.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
            raise ValueError("triangle/RGB data must be finite and colors must be in [0, 1]")
        processed += len(world)
        if not len(world):
            continue
        # Normalize by the global extent, so tiny/huge world units do not affect raster arithmetic.
        with np.errstate(over="ignore", invalid="ignore"):
            centered = (world - center) / unit
            projected = centered @ basis
            projected[..., 0] = projected[..., 0] * scale + width * 0.5
            projected[..., 1] = -projected[..., 1] * scale + height * 0.5
        if not np.isfinite(projected).all():
            raise ValueError("triangle coordinates cannot be represented in the camera frame")
        edges = centered[:, 1:] - centered[:, :1]
        edge_scale = np.abs(edges).max(axis=(1, 2))
        edges = np.divide(
            edges, edge_scale[:, None, None], out=np.zeros_like(edges), where=edge_scale[:, None, None] > 0
        )
        normals = np.cross(edges[:, 0], edges[:, 1])
        lengths = np.linalg.norm(normals, axis=1)
        diffuse = np.divide(np.abs(normals @ light), lengths, out=np.zeros_like(lengths), where=lengths > 0)
        shade = 0.55 + 0.45 * np.clip(diffuse, 0, 1)
        rgba = np.empty((len(world), 4), dtype=np.uint8)
        rgba[:, :3] = np.rint(colors * shade[:, None] * 255).astype(np.uint8)
        rgba[:, 3] = 255
        for face, color, length in zip(projected, rgba, lengths, strict=True):
            if length > 0:
                _raster_triangle(face, color, image, depth)
    nonempty = int(np.count_nonzero(image[..., 3]))
    return image, {
        "width": width,
        "height": height,
        "camera": {
            "projection": "orthographic",
            "center_xyz": center.tolist(),
            "basis_xyz": basis.tolist(),
            "scale": world_scale,
            "azimuth": azimuth,
            "elevation": elevation,
            "angle_unit": "degrees",
        },
        "nonempty_pixels": nonempty,
        "transparent_pixels": width * height - nonempty,
        "triangles_processed": processed,
    }
