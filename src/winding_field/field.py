"""Automatic observation orientation, bounded fibre constraints and core solves."""

import io
import itertools
import json
import math
import sys
import time
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
import torch

from block_solver import solve_blocks
from field_constraints import pair_constraints
from field_store import BlockStore
from .surface_sources import FieldSource, chunk_cache_floor
from .artifacts import Tiles
from .config import field_geometry, validate_field_cpu
from .util import atomic_json, file_sha


def calibration(maps, meta):
    matrix = np.zeros((3, 3))
    centre = np.zeros(3)
    count = 0
    density_sum = 0.0
    for key in maps.keys():
        data = maps.load(key)
        valid = data["support"]
        normal = data["normals"][valid].astype(float)
        matrix += normal.T @ normal
        lo, _ = maps.box(key)
        coords = np.argwhere(valid) + lo
        centre += (coords[:, ::-1] * meta["spacing_xyz"] + meta["origin_xyz"]).sum(0)
        count += len(coords)
        density_sum += float(data["density"][valid].sum(dtype=np.float64))
    return finish_calibration(matrix, centre, count, density_sum)


def finish_calibration(matrix, centre, count, density_sum):
    if count == 0 or density_sum <= 0:
        raise ValueError("no supported sheets in the requested CT domain")
    eigenvalues, vectors = np.linalg.eigh(matrix)
    axis = vectors[:, -1]
    axis *= 1 if axis[np.argmax(abs(axis))] >= 0 else -1
    concentration = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    return dict(
        mode="planar" if concentration > 0.65 else "radial-z",
        axis_xyz=axis.tolist(),
        centre_xyz=(centre / count).tolist(),
        mean_density_per_um=density_sum / count,
        orientation_concentration=concentration,
        supported_voxels=count,
        physical_accuracy="UNVALIDATED",
        topology="local chart; global sheet continuity unverified",
    )


def coordinates(lo, hi, meta):
    z, y, x = np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)), indexing="ij")
    return np.stack((x, y, z), axis=-1) * meta["spacing_xyz"] + meta["origin_xyz"]


# Normals closer than this (|cos|) to perpendicular to the orientation direction get no sign.
ORIENTATION_MIN_DOT = 0.15
# --field-orientation prior-gradient: a prior gradient weaker than this fraction of the local
# sheet density (the bottom of a valley, a flat prior) gives no sign.
PRIOR_GRADIENT_MIN_DENSITY = 0.3
FIELD_ORIENTATIONS = ("radial", "prior-gradient")
# The prior-gradient sheet check (see sheet_inconsistent): probe radii in nodes of the maps'
# lattice, directions per radius in the node's tangent plane, and the check's defaults.
SHEET_CHECK_RADII = (3, 6)
SHEET_CHECK_DIRECTIONS = 8
SHEET_CHECK_MAX_DISAGREEMENT = 0.3
SHEET_CHECK_MIN_PROBES = 0


def _sheet_margin(spacing_xyz):
    """Probe radii in µm and the ZYX lattice margin that holds every rounded probe."""
    spacing = np.asarray(spacing_xyz, float)
    radii = np.asarray(SHEET_CHECK_RADII, float) * spacing.mean()
    # rint(r * d / s) <= ceil(r_max / s) for |d| <= 1, so no probe leaves the margin.
    return radii, np.ceil(radii.max() / spacing - 1e-9).astype(int)[::-1]


def sheet_inconsistent(normals, density, along, candidates, gradient, defined, spacing_xyz, sheet):
    """Prior-signed nodes whose sign the prior contradicts along their own sheet.

    The prior's floor can cross a sheet where the scroll was crushed: the gradient sign then
    flips along the sheet and signs part of it inward. For each ``candidates`` node the prior
    gradient is read at SHEET_CHECK_DIRECTIONS directions in the node's tangent plane (normal
    ``normals``) at each of SHEET_CHECK_RADII, rounded to lattice nodes. ``gradient`` and
    ``defined`` cover the node lattice grown by _sheet_margin. A probe counts where the gradient
    is defined, at least PRIOR_GRADIENT_MIN_DENSITY x the node's density and farther than
    ORIENTATION_MIN_DOT from perpendicular to the normal. A node with at least
    ``sheet["min_probes"]`` counted probes is flagged when more than
    ``sheet["max_disagreement"]`` of them give the sign opposite to its own (``along``).

    The probes depend only on the node's normal and density and on the global prior, so tiles
    that share a node flag it alike, wherever their faces cut its neighbourhood.
    """
    shape = candidates.shape
    radii, margin = _sheet_margin(spacing_xyz)
    spacing = np.asarray(spacing_xyz, float)
    angles = np.arange(SHEET_CHECK_DIRECTIONS) * (2 * np.pi / SHEET_CHECK_DIRECTIONS)
    grown = defined.shape
    gradient = gradient.reshape(-1, 3)
    defined = defined.reshape(-1)
    normals = normals.reshape(-1, 3)
    nodes = np.flatnonzero(candidates)
    flagged = np.zeros(candidates.size, bool)
    batch = 1 << 20  # bounds the per-probe temporaries
    for start in range(0, len(nodes), batch):
        part = nodes[start : start + batch]
        at_node = np.stack(np.unravel_index(part, shape), axis=1) + margin
        normal = normals[part].astype(float)
        unit = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        u = np.cross(unit, np.eye(3)[np.argmin(abs(unit), axis=1)])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v = np.cross(unit, u)
        positive = along.reshape(-1)[part] > 0
        floor = PRIOR_GRADIENT_MIN_DENSITY * density.reshape(-1)[part]
        counted = np.zeros(len(part), np.int64)
        against = np.zeros(len(part), np.int64)
        for radius in radii:
            for angle in angles:
                step = np.rint(radius * (np.cos(angle) * u + np.sin(angle) * v) / spacing)
                probe = np.ravel_multi_index((at_node + step[:, ::-1].astype(int)).T, grown)
                g = gradient[probe]
                magnitude = np.linalg.norm(g, axis=1)
                sign = np.sum(normal * g, axis=1) / np.maximum(magnitude, 1e-300)
                usable = defined[probe] & (magnitude >= floor) & (abs(sign) > ORIENTATION_MIN_DOT)
                counted += usable
                against += usable & ((sign > 0) != positive)
        flagged[part] = (counted >= sheet["min_probes"]) & (
            against > sheet["max_disagreement"] * counted
        )
    return flagged.reshape(shape)


def oriented(data, xyz, cal, prior=None, stats=None, sheet=None, lattice=None):
    """Sheet-normal targets signed outward, with weights and the nodes that keep them.

    By default (radial) a normal is signed against the direction from the calibration
    centre (or the planar axis). In a flattened scroll that fails near the plane the
    sheets were pressed into: normals there are nearly perpendicular to the radial
    direction, so a wedge of nodes is dropped and a tilted sheet takes its tilt's sign.

    With ``prior`` (a FieldPrior; --field-orientation prior-gradient) a normal is signed
    against the prior's own gradient (FieldPrior.gradient: stored values, observed or
    not) wherever it is defined, and dropped where that gradient is nearly perpendicular
    to the normal or weaker than PRIOR_GRADIENT_MIN_DENSITY x the sheet density. With
    ``sheet`` (max_disagreement, min_probes) a node so signed is also dropped where
    sheet_inconsistent flags it; ``lattice`` (lo_zyx, meta) then places ``xyz`` on the map
    lattice. The radial rule remains where the gradient is undefined. ``stats``, a dict,
    accumulates the supported nodes and their split into prior_signed, prior_dropped,
    prior_inconsistent, radial_signed and radial_dropped.
    """
    direction = np.asarray(cal["axis_xyz"]) if cal["mode"] == "planar" else xyz - cal["centre_xyz"]
    if cal["mode"] != "planar":
        direction[..., 2] = 0
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(data["normals"] * direction, axis=-1)
    valid = data["support"] & (abs(dot) > ORIENTATION_MIN_DOT)
    if prior is not None:
        if sheet is None:
            gradient, defined = prior.gradient(xyz)
        else:
            # One gradient over the lattice grown by the probe margin: the nodes' own values
            # are its interior, the same as prior.gradient(xyz) node by node.
            lo, meta = lattice
            shape = data["support"].shape
            margin = _sheet_margin(meta["spacing_xyz"])[1]
            low = np.asarray(lo) - margin
            grown = prior.gradient(coordinates(low, low + shape + 2 * margin, meta))
            inner = tuple(slice(m, m + n) for m, n in zip(margin, shape))
            gradient, defined = grown[0][inner], grown[1][inner]
        magnitude = np.linalg.norm(gradient, axis=-1)
        along = np.sum(data["normals"] * gradient, axis=-1) / np.maximum(magnitude, 1e-300)
        kept = (
            data["support"]
            & (abs(along) > ORIENTATION_MIN_DOT)
            & (magnitude >= PRIOR_GRADIENT_MIN_DENSITY * data["density"])
        )
        inconsistent = np.zeros_like(kept)
        if sheet is not None:
            inconsistent = sheet_inconsistent(
                data["normals"],
                data["density"],
                along,
                kept & defined,
                *grown,
                meta["spacing_xyz"],
                sheet,
            )
        dot = np.where(defined, along, dot)
        valid = np.where(defined, kept & ~inconsistent, valid)
        if stats is not None:
            support = data["support"]
            for name, count in (
                ("supported", support),
                ("prior_signed", support & defined & valid),
                ("prior_dropped", support & defined & ~kept),
                ("prior_inconsistent", defined & inconsistent),
                ("radial_signed", support & ~defined & valid),
                ("radial_dropped", support & ~defined & ~valid),
            ):
                stats[name] = stats.get(name, 0) + int(np.count_nonzero(count))
    target = (
        data["normals"] * np.where(dot >= 0, 1.0, -1.0)[..., None] * data["density"][..., None]
    ).astype(np.float32)
    target[~valid] = 0
    weights = np.where(valid, 1.0, 1e-3).astype(np.float32)
    return target, weights, valid


# A prior-gradient run whose radial fallback exceeds this share is reported as a warning.
RADIAL_FALLBACK_WARNING = 0.05
ORIENTATION_CLASSES = (
    "prior_signed",
    "prior_dropped",
    "prior_inconsistent",
    "radial_signed",
    "radial_dropped",
)


def _orientation_receipt(rule, *counters):
    """The orientation rule with its node counts and shares, also logged on stderr.

    The ORIENTATION_CLASSES partition the supported nodes (prior_inconsistent: dropped by
    the sheet check); radial_fallback is the sum of the two radial ones.
    """
    total = dict.fromkeys(("supported", *ORIENTATION_CLASSES), 0)
    for counter in counters:
        for name, count in counter.items():
            total[name] += int(count)
    total["radial_fallback"] = total["radial_signed"] + total["radial_dropped"]
    supported = total["supported"]
    shares = {
        name + "_share": total[name] / supported if supported else None
        for name in (*ORIENTATION_CLASSES, "radial_fallback")
    }
    receipt = dict(
        rule, counted="each solved observation box once, halos included", **total, **shares
    )
    if supported:
        share = {name: f"{value:.1%}" for name, value in shares.items()}
        warning = "WARNING: " if shares["radial_fallback_share"] > RADIAL_FALLBACK_WARNING else ""
        print(
            f"[field] {warning}orientation prior-gradient over {supported} supported nodes: "
            f"prior gradient signed {share['prior_signed_share']}, dropped "
            f"{share['prior_dropped_share']}, dropped by the sheet check "
            f"{share['prior_inconsistent_share']}; radial fallback {share['radial_fallback_share']} "
            f"(signed {share['radial_signed_share']}, dropped {share['radial_dropped_share']})",
            file=sys.stderr,
            flush=True,
        )
    return receipt


def initial_field(xyz, meta, cal):
    if cal["mode"] == "planar":
        return (
            np.sum((xyz - meta["origin_xyz"]) * cal["axis_xyz"], axis=-1)
            * cal["mean_density_per_um"]
        )
    radius = np.linalg.norm((xyz - cal["centre_xyz"])[..., :2], axis=-1)
    first = np.linalg.norm((np.array(meta["origin_xyz"]) - cal["centre_xyz"])[:2])
    return (radius - first) * cal["mean_density_per_um"]


def map_axes(meta, geometry, lo, hi):
    return [
        np.clip(
            (
                geometry["origin_xyz"][2 - i]
                - meta["origin_xyz"][2 - i]
                + np.arange(a, b) * geometry["spacing_xyz"][2 - i]
            )
            / meta["spacing_xyz"][2 - i],
            0,
            meta["shape_zyx"][i] - 1,
        )
        for i, (a, b) in enumerate(zip(lo, hi))
    ]


def _owned_chunks(run, order, chunk, shape, identity, maps, meta, cal, prior=None, sheet=None):
    """Chunks with supported observations, cached in the run for resumes."""
    from .field_prior import owned_chunks

    geometry = identity["geometry"]
    key = dict(maps=identity["maps"], geometry=geometry, chunk=chunk)
    if "orientation" in identity:
        key["orientation"] = identity["orientation"]
    path = run / "FIELD_SCOPE.json"
    if path.exists():
        saved = json.loads(path.read_bytes())
        if saved["key"] == key:
            return [tuple(k) for k in saved["owned"]]

    def supported(lo, hi):
        return bool(sample_prior(maps, meta, geometry, cal, lo, hi, prior, sheet=sheet)[2].any())

    owned = owned_chunks(order, chunk, shape, supported)
    if not owned:
        raise ValueError("no field chunk has supported observations")
    atomic_json(path, dict(key=key, owned=[list(k) for k in owned]))
    return owned


def sample_prior(maps, meta, geometry, cal, lo, hi, prior=None, stats=None, sheet=None):
    """Oriented observations on field nodes ``lo..hi``; ``prior``, ``stats`` and ``sheet``
    as in ``oriented`` (stats count map nodes when the maps and field grids differ)."""
    names = ("normals", "density", "support")
    if all(geometry[k] == meta[k] for k in ("origin_xyz", "spacing_xyz", "shape_zyx")):
        data = maps.read_many(names, lo, hi)
        xyz = coordinates(lo, hi, meta)
        return oriented(data, xyz, cal, prior, stats, sheet, (lo, meta))

    def transform(data, start):
        hi = start + data["support"].shape
        target, weights, valid = oriented(
            data, coordinates(start, hi, meta), cal, prior, stats, sheet, (start, meta)
        )
        return dict(target=target, weights=weights, support=valid)

    data = maps.sample(names, map_axes(meta, geometry, lo, hi), transform)
    valid = data["support"]
    data["target"][~valid] = 0
    data["weights"][~valid] = 1e-3
    return data["target"], data["weights"], valid


def open_field(run, support_policy="observed"):
    """A completed field run's checkpoint; ``support_policy`` "observed" (default) admits
    observed nodes, "stored" every stored node."""
    run = Path(run)
    meta = json.loads((run / "FIELD.json").read_bytes())
    descriptor = dict(
        kind="chunks",
        path=str((run / "field/CHECKPOINT.json").resolve()),
        sha256=meta["checkpoint_sha256"],
        support_policy=support_policy,
    )
    # Retain at least one decoded chunk so a field stored as a few large blocks is not
    # decoded again for every read; affects memory only.
    floor = chunk_cache_floor(descriptor, meta["geometry"])
    return FieldSource(descriptor, meta["geometry"], cache_bytes=max(64 << 20, floor))


def spline_samples(points, low, high, step):
    """Clip each supported edge, then sample inside the block, including free nodes."""
    start = points[:-1]
    delta = np.diff(points, axis=0)
    near = np.divide(low - start, delta, out=np.full_like(start, -np.inf), where=delta != 0)
    far = np.divide(high - start, delta, out=np.full_like(start, np.inf), where=delta != 0)
    enter = np.maximum(0, np.minimum(near, far).max(1))
    leave = np.minimum(1, np.maximum(near, far).min(1))
    valid = (leave > enter) & np.all((delta != 0) | ((start >= low) & (start <= high)), axis=1)
    for i in np.flatnonzero(valid):
        a = np.clip(start[i] + enter[i] * delta[i], low, high)
        b = np.clip(start[i] + leave[i] * delta[i], low, high)
        length = np.linalg.norm(b - a)
        if length:
            yield np.linspace(a, b, max(1, int(np.ceil(length / step))) + 1)


def spline_pairs(points, low, high, step, *, max_pairs=None):
    """Batch short edges; preserve scalar norm/linspace for sensitive lengths."""
    start = points[:-1]
    delta = np.diff(points, axis=0)
    near = np.divide(low - start, delta, out=np.full_like(start, -np.inf), where=delta != 0)
    far = np.divide(high - start, delta, out=np.full_like(start, np.inf), where=delta != 0)
    enter = np.maximum(0, np.minimum(near, far).max(1))
    leave = np.minimum(1, np.maximum(near, far).min(1))
    valid = (leave > enter) & np.all((delta != 0) | ((start >= low) & (start <= high)), axis=1)
    indices = np.flatnonzero(valid)
    a = np.clip(start[indices] + enter[indices, None] * delta[indices], low, high)
    b = np.clip(start[indices] + leave[indices, None] * delta[indices], low, high)
    difference = abs(b - a)
    # A generous margin keeps near-step and underflow rounding on the old path.
    short = (difference.sum(1) <= step / 2) & (
        difference.max(1, initial=0) > np.sqrt(np.finfo(difference.dtype).tiny)
    )
    counts = short.astype(np.int64)
    long = {}
    total = int(short.sum())
    if max_pairs is not None and total > max_pairs:
        from field_runtime import ResourceUnavailable

        raise ResourceUnavailable("CPU pair preparation exceeds its slot")
    for i in np.flatnonzero(~short):
        length = np.linalg.norm(b[i] - a[i])
        if length:
            count = max(1, int(np.ceil(length / step)))
            total += count
            if max_pairs is not None and total > max_pairs:
                from field_runtime import ResourceUnavailable

                raise ResourceUnavailable("CPU pair preparation exceeds its slot")
            counts[i] = count
            long[i] = count
    pairs = np.linspace(a[short], b[short], 2, axis=1)
    offsets = np.r_[0, np.cumsum(counts)]
    left = np.empty((offsets[-1], 3), dtype=pairs.dtype)
    right = np.empty_like(left)
    left[offsets[:-1][short]] = pairs[:, 0]
    right[offsets[:-1][short]] = pairs[:, 1]
    for i, count in long.items():
        samples = np.linspace(a[i], b[i], count + 1)
        left[offsets[i] : offsets[i + 1]] = samples[:-1]
        right[offsets[i] : offsets[i + 1]] = samples[1:]
    return left, right


def _constraint_rows(
    db, meta, fibres, geometry, maps, lo, hi, weight, *, max_pairs=None, stop=None
):
    """Immutable ordinary-provider rows; no live field or exterior targets."""
    pairs = _pair_points(db, meta, fibres, geometry, maps, lo, hi, max_pairs=max_pairs, stop=stop)
    if pairs is None:
        return None
    block_origin = np.asarray(geometry["origin_xyz"]) + lo[::-1] * geometry["spacing_xyz"]
    return pair_constraints(
        *pairs, block_origin, geometry["spacing_xyz"], tuple(hi - lo), weight=weight
    )


def global_pair_constraints(db, meta, fibres, geometry, maps, shape, weight, cache_bytes=0):
    """One box-independent pair set over the whole grid, for multi-block solves.

    The ordinary provider samples and clips fibre pairs to each block crop, so the
    union of block problems is not the global problem and cut pairs never reach the
    coarse level: a two-level sweep then stalls. These rows are restricted per crop by
    the core, which moves exterior terms into targets and reports them to the coarse
    space. On a single block they equal the ordinary rows.
    """
    from constraint_templates import StencilConstraints

    lo, hi = np.zeros(3, np.int64), np.asarray(shape, np.int64)
    pairs = _pair_points(db, meta, fibres, geometry, maps, lo, hi)
    if pairs is None:
        return None
    return StencilConstraints.pairs(
        *pairs,
        geometry["origin_xyz"],
        geometry["spacing_xyz"],
        tuple(int(n) for n in shape),
        weight=weight,
        cache_bytes=cache_bytes,
    )


def _pair_points(db, meta, fibres, geometry, maps, lo, hi, *, max_pairs=None, stop=None):
    """Fibre pair endpoints (XYZ) sampled inside the box ``lo..hi``, or None."""
    block_origin = np.asarray(geometry["origin_xyz"]) + lo[::-1] * geometry["spacing_xyz"]
    block_shape = hi - lo
    block_upper = block_origin + (block_shape[::-1] - 1) * geometry["spacing_xyz"]
    a = []
    b = []
    count = 0
    if meta["prediction"] == "qual-v2" or fibres.get("spatial_index"):
        rows = db.execute(
            "SELECT c.points FROM curve_bounds b JOIN curves c ON c.id=b.id "
            "WHERE b.max_x>=? AND b.min_x<=? AND b.max_y>=? AND b.min_y<=? "
            "AND b.max_z>=? AND b.min_z<=? ORDER BY c.id",
            tuple(v for pair in zip(block_origin, block_upper) for v in pair),
        )
    else:
        axes = map_axes(meta, geometry, lo, hi)
        keys = itertools.product(
            *(range(int(axis[0]) // maps.chunk, int(axis[-1]) // maps.chunk + 1) for axis in axes)
        )
        rows = (
            row
            for key in keys
            for row in db.execute(
                "SELECT points FROM curves WHERE tile=?", ("_".join(map(str, key)),)
            )
        )
    for (raw,) in rows:
        if stop and stop():
            raise InterruptedError("CPU preparation stopped")
        if max_pairs is not None and len(raw) > (max_pairs + 1) * 24 + 4096:
            from field_runtime import ResourceUnavailable

            raise ResourceUnavailable("CPU curve exceeds its preparation slot")
        points = np.load(io.BytesIO(raw), allow_pickle=False)
        if max_pairs is not None and (
            not isinstance(points, np.ndarray)
            or points.ndim != 2
            or points.shape[1] != 3
            or points.dtype not in (np.float32, np.float64)
            or not np.isfinite(points).all()
        ):
            raise ValueError("CPU curve must contain finite float32/float64 XYZ points")
        if meta["prediction"] == "qual-v2" or fibres.get("spatial_index"):
            left, right = spline_pairs(
                points,
                block_origin,
                block_upper,
                min(geometry["spacing_xyz"]),
                max_pairs=None if max_pairs is None else max_pairs - count,
            )
        else:
            points = points[::3]
            if len(points) < 2:
                continue
            inside = ((points >= block_origin) & (points <= block_upper)).all(1)
            valid = inside[:-1] & inside[1:]
            left, right = points[:-1][valid], points[1:][valid]
        count += len(left)
        if max_pairs is not None and count > max_pairs:
            from field_runtime import ResourceUnavailable

            raise ResourceUnavailable("CPU constraint rows exceed their preparation slot")
        if len(left):
            a.append(left)
            b.append(right)
    if not a:
        return None
    return np.concatenate(a), np.concatenate(b)


def memory_plan(config, shape, chunk, halo, *, convergence=False, bounded=False, available=None):
    """Host-memory tier for this solve (``field_memory``: auto, disk, field or all).

    Configured cache budgets count as workspace and explicit cache sizes win. Sweep
    checkpoints need the whole field cached, and are not used with field advice or
    automatic convergence (they read the checkpoint at block boundaries) nor with a
    bounded field (its disk cap admits one block at a time); the caches still apply.
    """
    from field_memory import plan_memory

    tier = config.get("field_memory", "auto")
    if tier not in MEMORY_TIERS:
        raise ValueError("field_memory must be auto, disk, field or all")
    workspace = sum(
        config.get(name, 0) or 0
        for name in ("prepared_cache_bytes", "constraint_cache_bytes", "field_cpu_buffer_bytes")
    )
    plan = plan_memory(
        shape,
        chunk,
        halo,
        tier=tier,
        workspace_bytes=workspace,
        **({} if available is None else {"available": available}),
    )
    explicit = config.get("field_cache_bytes")
    reason = (
        "automatic convergence reads the checkpoint at every block boundary"
        if convergence
        else "field advice reads the checkpoint at every block boundary"
        if config.get("field_advice")
        else "the bounded disk cap admits one block at a time"
        if bounded
        else "field_cache_bytes is smaller than the field"
        if explicit is not None and explicit < plan["field_bytes"]
        else None
    )
    if plan["checkpoint"] == "sweep" and reason:
        plan.update(checkpoint="block", note="block checkpoints: " + reason)
    return plan


MEMORY_TIERS = ("auto", "disk", "field", "all")


def solve(run, config, stop=None, *, sweep_target=None, convergence=None):
    validate_field_cpu(config, convergence=convergence is not None)
    from .field_advice import observer_options, require_observer, record_verification, finish

    if type(config.get("field_advice", False)) is not bool:
        raise ValueError("field_advice must be boolean")
    if config.get("field_advice"):
        require_observer()
    budget = config.get("prepared_cache_bytes", 0)
    if type(budget) is not int or budget < 0:
        raise ValueError("prepared_cache_bytes must be a nonnegative integer")
    constraint_budget = config.get("constraint_cache_bytes", 0)
    if type(constraint_budget) is not int or constraint_budget < 0:
        raise ValueError("constraint_cache_bytes must be a nonnegative integer")
    observation_budget = config.get("observation_cache_bytes", 0)
    if type(observation_budget) is not int or observation_budget < 0:
        raise ValueError("observation_cache_bytes must be a nonnegative integer")
    field_cache_bytes = config.get("field_cache_bytes", 64 << 20)
    if type(field_cache_bytes) is not int or field_cache_bytes < 0:
        raise ValueError("field_cache_bytes must be a nonnegative integer")
    preconditioner = config.get("preconditioner", "jacobi")
    if preconditioner not in ("jacobi", "amg"):
        raise ValueError("block workflows require preconditioner='jacobi' or 'amg'")
    status_policy = config.get("field_status_policy", "all")
    if status_policy not in ("all", "checkpoints"):
        raise ValueError("field_status_policy must be all or checkpoints")
    run = Path(run)
    meta = json.loads((run / "PREPARED.json").read_bytes())
    fibres = json.loads((run / "FIBRES.json").read_bytes())
    bounded = meta.get("bounded_preparation", False)
    if config.get("field_cpu_workers", 0) and bounded:
        raise ValueError("CPU preparation pool supports only ordinary static providers")
    maps = Tiles(run / ("coarse/maps" if bounded else "maps"), readonly=convergence is not None)
    if (
        maps.identity() != meta["tiles_sha256"]
        or file_sha(run / "fibres/fibres.sqlite") != fibres["sha256"]
    ):
        raise ValueError("prepared maps or fibres changed")
    cal = meta["calibration"] if bounded else calibration(maps, meta)
    inherited = None
    if config.get("field_prior_calibration") == "inherit":
        # A tile's own ROI centroid is not the scroll centre: radial orientation from it
        # flips normals across the tile and against the coarse. Use the prior's calibration.
        source = Path(config["field_prior"]).expanduser() / "CALIBRATION.json"
        if not source.is_file():
            raise ValueError("field prior has no CALIBRATION.json to inherit")
        cal = json.loads(source.read_bytes())
        inherited = file_sha(source)
    atomic_json(run / "CALIBRATION.json", cal)
    geometry = field_geometry(meta, None if bounded else config.get("field_spacing_um"))
    geometry.update(voxel_um=1.0, sheet_step=1.0)
    shape = tuple(geometry["shape_zyx"])
    chunk = config.get("field_chunk", 64)
    if type(chunk) is not int or chunk < 4:
        raise ValueError("field chunk must be at least 4")
    identity = dict(
        maps=meta["tiles_sha256"],
        fibres=fibres["sha256"],
        geometry=geometry,
        calibration=cal,
        fibre_weight=config.get("fibre_weight", 1.0),
        algorithm="physical-trilinear-prior-v2",
    )
    if inherited is not None:
        identity["calibration_source"] = dict(kind="field_prior", sha256=inherited)
    orientation = config.get("field_orientation", "radial")
    orientation_sheet = None
    if orientation not in FIELD_ORIENTATIONS:
        raise ValueError("field_orientation must be " + " or ".join(FIELD_ORIENTATIONS))
    if orientation == "prior-gradient":
        if bounded or not config.get("field_prior"):
            raise ValueError(
                "field_orientation prior-gradient requires field_prior and ordinary preparation"
            )
        # Only recorded when not radial, so radial runs keep their identity and resume.
        # v2: the gradient reads the prior's stored values (v1 required observed corners).
        identity["orientation"] = dict(
            rule="prior-gradient-central-v2",
            prior_values="stored-finite",
            min_dot=ORIENTATION_MIN_DOT,
            min_gradient_per_density=PRIOR_GRADIENT_MIN_DENSITY,
            fallback="radial",
        )
        share = config.get("field_orientation_max_disagreement")
        least = config.get("field_orientation_min_probes")
        orientation_sheet = dict(
            max_disagreement=float(SHEET_CHECK_MAX_DISAGREEMENT if share is None else share),
            min_probes=SHEET_CHECK_MIN_PROBES if least is None else least,
        )
        probes = len(SHEET_CHECK_RADII) * SHEET_CHECK_DIRECTIONS
        if not 0 <= orientation_sheet["max_disagreement"] <= 1:
            raise ValueError("field_orientation_max_disagreement must be from 0 to 1")
        if type(orientation_sheet["min_probes"]) is not int or not (
            0 <= orientation_sheet["min_probes"] <= probes
        ):
            raise ValueError(f"field_orientation_min_probes must be an integer from 0 to {probes}")
        if orientation_sheet["max_disagreement"] < 1:
            # v3: also drop prior-signed nodes the prior contradicts along their own sheet.
            # A share of 1 flags nothing and keeps the v2 identity, so v2 runs resume.
            identity["orientation"].update(
                rule="prior-gradient-central-v3",
                sheet_check=dict(
                    rule="tangent-probes-v1",
                    radii_nodes=list(SHEET_CHECK_RADII),
                    directions=SHEET_CHECK_DIRECTIONS,
                    **orientation_sheet,
                ),
            )
        else:
            orientation_sheet = None
    elif any(
        config.get(k) is not None
        for k in ("field_orientation_max_disagreement", "field_orientation_min_probes")
    ):
        raise ValueError("the orientation sheet check requires field_orientation prior-gradient")
    pair_provider = None
    if bounded or convergence is not None:
        from .bounded_constraints import build, PairCatalog

        status = build(
            run / "pairs",
            run / "fibres/fibres.sqlite",
            fibres["sha256"],
            geometry,
            weight=identity["fibre_weight"],
            quota_bytes=config.get("bounded_bytes", 12 << 30) // 8,
            reserve_bytes=config.get("reserve_bytes", 0),
            stop=stop,
        )
        if status == "PAUSED":
            maps.close()
            return dict(status="PAUSED")
        pair_provider = PairCatalog(
            run / "pairs", geometry, cache_bytes=config.get("constraint_cache_bytes", 0)
        )
        identity.update(
            algorithm="global-arclength-pairs-coarse-prior-v1",
            pairs=file_sha(run / "pairs/PAIRS.json"),
        )
        if convergence is not None:
            identity.update(
                algorithm="global-arclength-pairs-coarse-prior-v1"
                if bounded
                else "global-arclength-pairs-full-prior-v1",
                mode="automatic-convergence-v1",
            )
    root = run / "field"

    def preparation_stopped():
        return (root / "STOP").exists() or bool(stop and stop())

    from .field_budget import effective_budget, extend

    # The audited budget equals the configured one until a field is extended
    # (field_budget.extend), for ordinary and bounded fields alike.
    sweeps = effective_budget(root, config.get("sweeps", 4))
    if sweep_target is not None and (type(sweep_target) is not int or sweep_target < sweeps):
        raise ValueError("invalid field sweep target")
    options = dict(
        origin_xyz=geometry["origin_xyz"],
        spacing_xyz=geometry["spacing_xyz"],
        order=list(itertools.product(*(range((n - 1) // chunk + 1) for n in shape))),
        # The core records the overlap in the checkpoint's problem identity.
        halo=config.get("field_halo", 2),
        sweeps=sweeps,
        rtol=config.get("rtol", 1e-6),
        maxiter=config.get("maxiter", 2000),
        dtype="float64",
        device=config.get("device", "cpu"),
        nonconvergence_policy=config.get("nonconvergence_policy", "fail"),
        cache_bytes=field_cache_bytes,
        reserve_bytes=config.get("reserve_bytes", 0),
        prepared_cache_bytes=config.get("prepared_cache_bytes", 0),
    )
    if preconditioner != "jacobi":
        # Only a non-default policy is passed: Jacobi run identities stay unchanged.
        options["preconditioner"] = preconditioner
    coarse_factor = config.get("field_coarse_factor", 0)
    if type(coarse_factor) is not int or coarse_factor < 0 or coarse_factor == 1:
        raise ValueError("field_coarse_factor must be 0 (off) or at least 2")
    if coarse_factor:
        # Only when enabled: the core adds the coarse space to the problem identity.
        options["coarse_factor"] = coarse_factor
    if status_policy != "all":
        # Execution choice, not checkpoint identity: STATUS fsyncs only at checkpoints.
        options["status_policy"] = status_policy
    memory = memory_plan(
        config,
        shape,
        chunk,
        options["halo"],
        convergence=convergence is not None,
        bounded=bool(bounded),
    )
    if "field_cache_bytes" not in config:
        field_cache_bytes = max(field_cache_bytes, memory["field_cache_bytes"])
        options["cache_bytes"] = field_cache_bytes
    if "observation_cache_bytes" not in config:
        observation_budget = memory["observation_cache_bytes"]
    memory.update(field_cache_bytes=field_cache_bytes, observation_cache_bytes=observation_budget)
    if memory["checkpoint"] != "block":
        # Execution choice, not checkpoint identity: may differ on another host or resume.
        options["checkpoint"] = memory["checkpoint"]
    options["memory_report"] = memory
    if memory["warning"]:
        print(f"[field] WARNING: {memory['warning']}", file=sys.stderr, flush=True)
    if bounded:
        options["max_bytes"] = config.get("bounded_bytes", 12 << 30) // 8
    prior = None
    orientation_prior = None
    scope = config.get("field_scope", "all")
    if scope not in ("all", "supported"):
        raise ValueError("field_scope must be all or supported")
    if scope == "supported" and (not config.get("field_prior") or convergence is not None):
        raise ValueError("field_scope supported requires field_prior and a fixed sweep budget")
    if config.get("field_prior"):
        from .field_prior import FieldPrior, fixed_provider, scoped_fixed

        if bounded:
            raise ValueError("field_prior supports ordinary (unbounded) preparation only")
        prior = FieldPrior(config["field_prior"], geometry, meta.get("source"))
        # The prior seeds and gauges the solve, so it is part of the numerical problem.
        identity["field_prior"] = prior.identity
        if orientation == "prior-gradient":
            orientation_prior = prior
        if scope == "supported":
            total = len(options["order"])
            owned = _owned_chunks(
                run,
                options["order"],
                chunk,
                shape,
                identity,
                maps,
                meta,
                cal,
                orientation_prior,
                orientation_sheet,
            )
            options.update(order=owned, partial_order=True)
            identity["field_scope"] = dict(
                mode="supported-chunks-v1", owned_chunks=len(owned), total_chunks=total
            )
        anchor_index, anchor_value = prior.anchor(
            options["order"], chunk, shape, lambda lo, hi: coordinates(lo, hi, geometry)
        )
        identity["field_prior"]["anchor"] = dict(index_zyx=list(anchor_index), value=anchor_value)
        if scope == "supported":

            def start(lo, hi):
                xyz = coordinates(lo, hi, geometry)
                return prior.initial(xyz, initial_field(xyz, geometry, cal))

            options["fixed"] = scoped_fixed(
                anchor_index, anchor_value, options["order"], chunk, shape, start
            )
        else:
            options["fixed"] = fixed_provider(anchor_index, anchor_value)
    db = sqlite3.connect((run / "fibres/fibres.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    row_count = 0
    static_rows = {}
    static_bytes = 0
    cpu_inputs = None
    cache_stats = dict(
        budget=constraint_budget,
        hits=0,
        misses=0,
        builds=0,
        entries=0,
        retained_bytes=0,
        peak_bytes=0,
        peak_entries=0,
        denied_entries=0,
    )
    # Static per-box observations (target, weights, support); never field values.
    observations = {}
    observation_state = dict(retain=bool(observation_budget), bytes=0)
    solve_boxes = set()
    for key in options["order"]:
        low = np.array(key) * chunk
        high = np.minimum(low + chunk, shape)
        solve_boxes.add(
            tuple(
                map(
                    int,
                    (
                        *np.maximum(low - options["halo"], 0),
                        *np.minimum(high + options["halo"], shape),
                    ),
                )
            )
        )
    observation_stats = dict(
        budget=observation_budget,
        hits=0,
        misses=0,
        entries=0,
        retained_bytes=0,
        peak_bytes=0,
        peak_entries=0,
        denied_entries=0,
    )

    def source_receipt():
        return {"maps": meta["tiles_sha256"]}

    # prior-gradient orientation: node counts, each solved observation box once (halos
    # included); the CPU pool keeps its own (CPUFieldInputs.stats["orientation"]).
    orientation_stats = {}
    orientation_boxes = set()

    def compute_observations(lo, hi, stats=None):
        if bounded:
            values = maps.read_many(("target", "weights", "support"), lo, hi)
            target, weights, valid = (values[k] for k in ("target", "weights", "support"))
        else:
            target, weights, valid = sample_prior(
                maps, meta, geometry, cal, lo, hi, orientation_prior, stats, orientation_sheet
            )
        return target, weights, valid, source_receipt()

    def release_observations():
        observations.clear()
        observation_state.update(retain=False, bytes=0)
        observation_stats.update(entries=0, retained_bytes=0)

    def load(lo, hi):
        key = tuple(map(int, (*lo, *hi)))
        cached = observations.get(key)
        # The CPU pool skips a box only when both static inputs are retained.
        if cached is not None and (cpu_inputs is None or key in static_rows):
            observation_stats["hits"] += 1
            return (*cached, source_receipt())
        observation_stats["misses"] += 1
        if cpu_inputs is not None:
            values = cpu_inputs.load(lo, hi)
        else:
            counted = orientation_prior is not None and key not in orientation_boxes
            orientation_boxes.add(key)
            values = compute_observations(lo, hi, orientation_stats if counted else None)
        if observation_state["retain"] and cached is None and key in solve_boxes:
            arrays = tuple(values[:3])
            charge = 4096 + sum(a.nbytes for a in arrays)
            if charge <= observation_budget - observation_state["bytes"]:
                for array in arrays:
                    array.setflags(write=False)
                observations[key] = arrays
                observation_state["bytes"] += charge
                observation_stats.update(
                    entries=len(observations),
                    retained_bytes=observation_state["bytes"],
                    peak_bytes=observation_state["bytes"],
                    peak_entries=len(observations),
                )
            else:
                observation_stats["denied_entries"] += 1
        return values

    def constraints(lo, hi, read_nodes):
        nonlocal row_count, static_bytes
        key = tuple(map(int, (*lo, *hi)))
        if key in static_rows:
            cache_stats["hits"] += 1
            rows = static_rows[key]
            row_count += 0 if rows is None else rows.matrix.shape[0]
            return rows
        prefetched = cpu_inputs.constraints(lo, hi, read_nodes) if cpu_inputs is not None else None
        cache_stats["misses"] += 1
        cache_stats["builds"] += 1
        result = (
            prefetched
            if cpu_inputs is not None
            else _constraint_rows(
                db, meta, fibres, geometry, maps, lo, hi, identity["fibre_weight"]
            )
        )
        if result is None:
            if static_bytes + 4096 <= constraint_budget:
                static_rows[key] = None
                static_bytes += 4096
                cache_stats.update(
                    entries=len(static_rows),
                    retained_bytes=static_bytes,
                    peak_bytes=static_bytes,
                    peak_entries=len(static_rows),
                )
            else:
                cache_stats["denied_entries"] += 1
            return None
        row_count += result.matrix.shape[0]
        charge = 4096 + result.targets.nbytes + sum(a.nbytes for a in result._structural_arrays())
        if charge <= constraint_budget - static_bytes:
            static_rows[key] = result.with_targets(result.targets)
            static_bytes += charge
            cache_stats.update(
                entries=len(static_rows),
                retained_bytes=static_bytes,
                peak_bytes=static_bytes,
                peak_entries=len(static_rows),
            )
        else:
            cache_stats["denied_entries"] += 1
        return result

    constraints.prepared_key = lambda lo, hi: fibres["sha256"]
    options["constraints"] = pair_provider if pair_provider is not None else constraints
    # Two-level Schwarz needs the global pair set: with block-sampled pairs the union of
    # block problems is not the global one and the sweeps plateau (175A plan-192, fibre
    # weight 1000: coarse residual ~1.0 with block pairs, x0.41 per sweep with global).
    pair_scope = config.get("field_pair_scope", "global" if coarse_factor else "block")
    if pair_scope not in ("block", "global"):
        raise ValueError("field_pair_scope must be block or global")
    if pair_scope == "block" and coarse_factor and len(options["order"]) > 1:
        print(
            "[field] WARNING: --field-coarse-factor with --field-pair-scope block stalls; "
            "use global pairs",
            file=sys.stderr,
            flush=True,
        )
    if pair_scope == "global" and pair_provider is None and len(options["order"]) > 1:
        # Several blocks: one global pair set, restricted per crop with exterior terms.
        rows = global_pair_constraints(
            db, meta, fibres, geometry, maps, shape, identity["fibre_weight"], constraint_budget
        )
        options["constraints"] = rows
        identity["pair_scope"] = "global-grid-v1"
        row_count = 0 if rows is None else len(rows._weights)
    try:
        checkpoint = root / "CHECKPOINT.json"
        if not checkpoint.exists():
            # Publish only a complete initial checkpoint; failed admission is retryable.
            with TemporaryDirectory(prefix=".field-init-", dir=run) as temporary:
                initial = Path(temporary) / "field"
                solve_blocks(initial, shape, chunk, identity, load, stop=lambda: True, **options)
                initial.rename(root)
        saved = json.loads(checkpoint.read_bytes())
        if not saved["progress"].get("initialization_done") and not saved["progress"].get("solver"):
            with BlockStore(
                root,
                shape,
                chunk,
                saved["identity"],
                resume=True,
                cache_bytes=field_cache_bytes,
                reserve_bytes=options["reserve_bytes"],
                max_bytes=options.get("max_bytes"),
            ) as store:
                progress = store.manifest["progress"]
                for key in options["order"]:
                    if stop and stop():
                        return dict(status="PAUSED")
                    if "-".join(map(str, key)) in store.manifest["chunks"]:
                        continue
                    lo = np.array(key) * chunk
                    hi = np.minimum(lo + chunk, shape)
                    _, _, valid, _ = compute_observations(lo, hi)
                    xyz = coordinates(lo, hi, geometry)
                    start = initial_field(xyz, geometry, cal)
                    if prior is not None:
                        start = prior.initial(xyz, start)
                    store.commit(key, start, progress, valid)
                progress["initialization_done"] = True
                store.save_progress(progress)
        if config.get("field_cpu_workers", 0):
            from .field_cpu import CPUFieldInputs

            boxes = []
            for key in options["order"]:
                low = np.array(key) * chunk
                high = np.minimum(low + chunk, shape)
                boxes.append(
                    (
                        tuple(map(int, np.maximum(low - options["halo"], 0))),
                        tuple(map(int, np.minimum(high + options["halo"], shape))),
                    )
                )
            spec = dict(
                run=str(run.resolve()),
                meta=meta,
                fibres=fibres,
                geometry=geometry,
                calibration=cal,
                weight=identity["fibre_weight"],
                boxes=boxes,
                # Workers open the prior themselves; absent for radial orientation.
                **(
                    dict(
                        orientation_prior=dict(
                            run=str(prior.run),
                            field_json_sha256=prior.identity["field_json_sha256"],
                            sheet=orientation_sheet,
                        )
                    )
                    if orientation_prior is not None
                    else {}
                ),
                cache_bytes=constraint_budget + budget + observation_budget,
            )
            cpu_inputs = CPUFieldInputs(
                spec,
                workers=config["field_cpu_workers"],
                max_bytes=config["field_cpu_buffer_bytes"],
                cache=static_rows,
                observations=observations,
                stop=preparation_stopped,
            )
        if bounded:
            device = torch.device(config.get("device", "cpu"))
            if device.type == "cuda":
                # A fresh resume has not initialized the CUDA allocator through inference.
                torch.empty(0, device=device)
                torch.cuda.reset_peak_memory_stats(device)
        if convergence is not None:
            options.update(
                observer_options(
                    config.get("field_advice", False),
                    root,
                    geometry,
                    reserve_bytes=options["reserve_bytes"],
                )
            )
            return convergence.solve(
                root, shape, chunk, identity, load, geometry, cal, pair_provider, options, stop
            )
        options.update(
            observer_options(
                config.get("field_advice", False),
                root,
                geometry,
                reserve_bytes=options["reserve_bytes"],
            )
        )
        result = solve_blocks(root, shape, chunk, identity, load, resume=True, stop=stop, **options)
        if (
            result["status"] == "SWEEP_BUDGET_REACHED"
            and sweep_target is not None
            and sweep_target > sweeps
        ):
            published = root.parent / "FIELD.json"
            if published.exists() and not bounded:
                # Reopen a finished ordinary field: keep its receipt next to the run;
                # extend() still refuses when meshes or surfaces were built from it.
                archive = root.parent / f"FIELD.sweeps-{sweeps}.json"
                if archive.exists():
                    archive = root.parent / f"FIELD.sweeps-{sweeps}.{time.time_ns()}.json"
                published.replace(archive)
            options["sweeps"] = extend(
                root,
                sweep_target,
                config.get("sweeps", 4),
                reserve_bytes=options["reserve_bytes"],
                max_bytes=options.get("max_bytes"),
            )
            result = solve_blocks(
                root, shape, chunk, identity, load, resume=True, stop=stop, **options
            )
        result["effective_threads"] = torch.get_num_threads()
        finish(options.get("observer"))
        static_rows.clear()
        static_bytes = 0
        cache_stats.update(entries=0, retained_bytes=0)
        release_observations()
        if result["status"] != "SWEEP_BUDGET_REACHED":
            return dict(
                status="PAUSED",
                solver=result,
                constraint_cache_stats=cache_stats,
                observation_cache_stats=observation_stats,
            )
        receipt = dict(
            status="FIELD_READY",
            geometry=geometry,
            checkpoint_sha256=file_sha(checkpoint),
            effective_threads=torch.get_num_threads(),
            calibration=cal,
            solver=result,
            global_converged=False,
            physical_acceptance=False,
            fibre_constraint_rows=row_count,
            constraint_cache_stats=cache_stats,
            observation_cache_stats=observation_stats,
            numerical_status=(
                "local solves converged; sweep budget exhausted"
                if result["local_solves_converged"]
                else "local solves not converged; sweep budget exhausted"
            ),
        )
        if cpu_inputs is not None:
            receipt["field_cpu_stats"] = cpu_inputs.stats
        if bounded:
            from .global_residual import verify

            backend = dict(
                name="canonical-block-solver",
                device=str(torch.empty(0, device=device).device),
                configured_device=str(device),
                dtype="float64",
                field_chunk=chunk,
                halo=options["halo"],
            )
            if device.type == "cuda":
                backend.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    gpu_name=torch.cuda.get_device_name(device),
                )
            descriptor = dict(
                kind="chunks",
                path=str(checkpoint.resolve()),
                sha256=receipt["checkpoint_sha256"],
                support_policy="stored",
            )
            advisor = options.get("observer")
            regions = [] if advisor is not None else None
            verification_started = time.perf_counter() if advisor is not None else None
            with FieldSource(descriptor, geometry, cache_bytes=64 << 20) as source:
                # Decode the hash-bound gauge chunk through the same verified reader.
                gauge_offset = source._load("0-0-0", tuple(min(chunk, n) for n in shape))[1]

                def read(lo, hi):
                    value, present = source.read(lo, hi)
                    if not present.all():
                        raise ValueError("global verification requires complete stored field")
                    return value

                verification = verify(
                    shape,
                    chunk,
                    geometry["spacing_xyz"],
                    read,
                    load,
                    constraints=pair_provider,
                    rtol=options["rtol"],
                    stop=stop,
                    gauge_offset=gauge_offset,
                    **(dict(regions=regions) if advisor is not None else {}),
                )
                source.validate(hashes=True)
            verification.update(
                checkpoint_sha256=receipt["checkpoint_sha256"], solver_backend=backend
            )
            if verification["status"] != "PAUSED" and (
                verification["fixed_storage_max_error"] != 0
                or not all(math.isfinite(verification[k]) for k in ("residual_l2", "rhs_l2"))
            ):
                raise ValueError("global verification found invalid gauge or nonfinite residual")
            record_verification(
                advisor,
                receipt["checkpoint_sha256"],
                verification,
                verification_started,
                sweep=len(result["completed_sweeps"]),
                regions=regions,
            )
            atomic_json(run / "GLOBAL_RESIDUAL.json", verification)
            if verification["status"] == "PAUSED" or (
                not verification["global_converged"] and options["nonconvergence_policy"] == "fail"
            ):
                return dict(
                    status="PAUSED",
                    solver=result,
                    verification=verification,
                    reason="global numerical verification not passed; sweep budget is not convergence",
                )
            receipt.update(
                global_converged=verification["global_converged"],
                verification=verification,
                solver_backend=backend,
                numerical_status=receipt["numerical_status"]
                + "; "
                + (
                    "stored global normal equations and codec-encoded zero gauge verified"
                    if verification["global_converged"]
                    else "stored global normal equations not converged"
                ),
            )
        if prior is not None:
            receipt["field_prior"] = identity["field_prior"]
        if "field_scope" in identity:
            receipt["field_scope"] = identity["field_scope"]
        if orientation_prior is not None:
            pooled = cpu_inputs.stats.get("orientation", {}) if cpu_inputs is not None else {}
            receipt["orientation"] = _orientation_receipt(
                identity["orientation"], orientation_stats, pooled
            )
        atomic_json(run / "FIELD.json", receipt)
        return receipt
    except InterruptedError:
        if cpu_inputs is None or not preparation_stopped():
            raise
        status = json.loads((root / "STATUS.json").read_bytes())
        status.update(status="CHECKPOINTED_STOP")
        status.pop("error", None)
        atomic_json(root / "STATUS.json", status)
        finish(options.get("observer"))
        return dict(
            status="PAUSED",
            solver=status,
            constraint_cache_stats=cache_stats,
            observation_cache_stats=observation_stats,
        )
    finally:
        if cpu_inputs is not None:
            cpu_inputs.close()
        static_rows.clear()
        cache_stats.update(entries=0, retained_bytes=0)
        release_observations()
        maps.close()
        db.close()
        if prior is not None:
            prior.close()
        if pair_provider is not None:
            pair_provider.__exit__()
