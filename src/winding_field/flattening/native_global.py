"""Whole-sheet flattening for the native renderer (``flattening_scope: sheet``).

The chart route flattens ~1,000-face charts and welds each one to a single parent;
every other seam becomes a cut, and drift between charts opens those cuts into gaps or
overlaps in the rendered sheet. This route flattens each edge-connected piece of the
mesh as ONE map instead: it is cut to a topological disk (``cut_component``), seeded
with a global LSCM (mean-value when LSCM folds) and optimized by one Flatboi SLIM
solve (symmetric Dirichlet). The global UV is then stored as ordinary MAPPED raster
charts that already agree on every shared vertex, so nothing is welded and the only
seams are the explicit disk cuts. The existing rasterizer, CT sampling and TIFXYZ
export read those charts unchanged.

Real level meshes are not clean: a pinched vertex, a third face on an edge, a flipped
or a duplicated face would each make a whole sheet non-manifold. Faces therefore stay
joined only across regular edges (two faces, opposite directions) and every vertex is
copied once per fan of faces joined that way (``_manifold_pieces``); geometry and
source ids are untouched, and the receipt counts the copies.

A piece that still cannot be flattened this way (a failed or folded solve) falls back
to the chart route under its own component and is listed in the receipt; nothing is
dropped silently.
"""

import json
import tempfile
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra

from .legacy_charts import cross2
from .native_charts import (
    SLIM_BATCH_FACES,
    SLIM_ITERATIONS,
    SLIM_TOLERANCE,
    _flush_slim,
    _pieces,
    _schema,
)

SCHEMA = "native-sheet-flatten-v1"
# State rows naming one finished piece each, in commit order: ``<prefix><sequence>``.
PIECE_ROW = "sheet_piece/"


# One Flatboi call per group keeps process start-up amortized over small pieces.
GROUP_FACES = 4_000_000
# A whole-sheet seed may fold on this share of a piece's area; those faces leave the piece.
SEED_DROP = 0.05
# Seed faces below this share of the median area scale count as folded.
SEED_FLOOR = 1e-9
# A piece flattened as a disk after its holes failed must stay inside these edge stretches.
DISK_STRETCH = (0.25, 4.0)
# After handle removal, a rim at least this share of the longest one is an end of a wrap,
# not a hole: a seam joins it to the outline, so the wrap unrolls instead of mapping as a ring.
LONG_RIM = 0.25
# An LSCM seed whose edge stretch p99 / p01 is above this has drifted: try to unroll instead.
SEED_SPREAD = 16.0
# Angle step of the cross-section profile a piece is unrolled along.
UNROLL_STEP = np.radians(0.5)


class _NeedsDisk(Exception):
    """A piece flattened with its holes kept came out distorted: cut it to a disk."""


def _manifold_pieces(faces):
    """Split vertices so faces meet only across regular edges; group faces into pieces.

    A regular edge has exactly two faces that run along it in opposite directions. Each
    vertex gets one copy per fan of faces joined across regular edges, so irregular
    contacts (pinched vertices, edges with three or more faces, flipped or duplicated
    faces) come apart instead of making a piece non-manifold. Copies are numbered in
    source-vertex order: a clean mesh comes back unchanged.

    Returns ``(split, parent, groups, repair)``: faces over the copies, the source vertex
    of each copy, the face rows of every edge-connected piece (ordered by first face)
    and the counts of what was separated.
    """
    count = len(faces)
    tail = faces.ravel()  # corner 3*f+i sits on vertex faces[f, i]
    head = np.roll(faces, -1, axis=1).ravel()  # half-edge 3*f+i runs tail -> head
    low, high = np.minimum(tail, head), np.maximum(tail, head)
    order = np.lexsort((high, low))
    low, high = low[order], high[order]
    first = np.flatnonzero(np.r_[True, (low[1:] != low[:-1]) | (high[1:] != high[:-1])])
    size = np.diff(np.r_[first, len(order)])
    pair = first[size == 2]
    a, b = order[pair], order[pair + 1]
    opposite = tail[a] != tail[b]
    a, b = a[opposite], b[opposite]

    def following(corner):
        return corner - corner % 3 + (corner + 1) % 3

    # Across a regular edge the two faces share both end vertices: join those corners.
    joined = coo_matrix(
        (
            np.ones(2 * len(a), np.uint8),
            (np.concatenate([a, following(a)]), np.concatenate([following(b), b])),
        ),
        shape=(3 * count, 3 * count),
    )
    _, fan = connected_components(joined, directed=False)
    _, corner = np.unique(fan, return_index=True)
    rank = np.lexsort((corner, tail[corner]))
    number = np.empty(len(rank), np.int64)
    number[rank] = np.arange(len(rank))
    parent = tail[corner][rank]
    split = number[fan].reshape(count, 3)
    adjacency = coo_matrix((np.ones(len(a), np.uint8), (a // 3, b // 3)), shape=(count, count))
    _, labels = connected_components(adjacency, directed=False)
    by_label = np.argsort(labels, kind="stable")
    groups = np.split(by_label, np.flatnonzero(np.diff(labels[by_label])) + 1)
    repair = dict(
        vertex_copies=int(len(parent) - len(np.unique(tail))),
        irregular_edges=int((size > 2).sum() + (size == 2).sum() - len(a)),
    )
    return split, parent, sorted(groups, key=lambda g: int(g[0])), repair


def _minimal_cut(vertices, faces, check, all_holes=False):
    """Cut a piece only where it cannot lie flat; holes stay holes.

    ``cut_component`` joins every hole to the outer boundary, and each of those cuts can
    open or overlap in the render. A hole in a near-developable sheet needs none: its rim
    turns by about -2*pi (Gauss-Bonnet), while the outer boundary turns by about +2*pi and
    either end of a closed wrap by about 0. Loops that turn by less than -pi are capped
    for the topological cut only, so just wraps and handles are cut. With ``all_holes``
    the turning is not trusted: what is left of a removed tunnel turns like the end of a
    wrap, and so can a small hole in a rough sheet. Every loop but the outline is then a
    hole, except the long ones (``LONG_RIM``): those are the ends of a wrap, and a wrap
    kept closed maps as a ring with its far end squeezed into the middle. Each gets the
    shortest seam to the outline (``_rim_seams``).

    Returns ``(cut, topology)`` like ``cut_component``, or None when there is no hole to
    keep. Raises ValueError when the result is not a planar domain.
    """
    import igl

    from ..tifxyz_guide import _cut_flags
    from .native_sheet import _input, _topology_report

    areas = _input(vertices, faces, check)
    f = faces.astype(np.int64, copy=False)
    original = _topology_report(vertices, f, areas, check)
    loops = [np.asarray(loop, np.int64) for loop in igl.boundary_loop_all(f)]
    if len(loops) < 2:
        return None
    angle = np.zeros(len(vertices))
    corners = vertices[f]
    for i in range(3):
        a = corners[:, (i + 1) % 3] - corners[:, i]
        b = corners[:, (i + 2) % 3] - corners[:, i]
        cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        np.add.at(angle, f[:, i], np.arccos(np.clip(cosine, -1.0, 1.0)))
    turning = np.array([float((np.pi - angle[loop]).sum()) for loop in loops])
    outer = int(np.argmax(turning))
    ends = []
    if all_holes:
        length = np.array(
            [
                np.linalg.norm(vertices[loop] - vertices[np.roll(loop, -1)], axis=1).sum()
                for loop in loops
            ]
        )
        ends = [k for k in np.flatnonzero(length >= LONG_RIM * length.max()).tolist() if k != outer]
    holes = [
        k
        for k in range(len(loops))
        if k != outer and k not in ends and (all_holes or turning[k] < -np.pi)
    ]
    if not holes:
        return None
    # Cap each hole with a fan around one extra vertex; the fan reverses the rim edges.
    loop_of = np.full(len(vertices), -1, np.int64)
    for index, k in enumerate(holes):
        loop_of[loops[k]] = len(vertices) + index
    tail, head = f.ravel(), np.roll(f, -1, axis=1).ravel()
    key = np.minimum(tail, head) * len(vertices) + np.maximum(tail, head)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    rim = (counts[inverse] == 1).reshape(-1, 3)
    capped_rim = rim.ravel() & (loop_of[tail] >= 0)
    caps = np.column_stack([head[capped_rim], tail[capped_rim], loop_of[tail[capped_rim]]])
    capped = np.concatenate([f, caps])
    if original["euler_characteristic"] + len(holes) == 1:
        paths = []
    elif ends and original["euler_characteristic"] + len(holes) == 1 - len(ends):
        # A wrap without handles: one seam per end, nothing else is cut.
        paths = _rim_seams(vertices, f, loops, outer, ends)
    else:
        paths = igl.cut_to_disk(capped)
    if check:
        check()
    # A path may follow a kept rim or cross a cap: only interior edges of real faces are cut.
    flags = _cut_flags(capped, paths, len(vertices) + len(holes))[: len(f)] & ~rim
    if flags.any():
        v, cf, parent = igl.cut_mesh(vertices, f, flags)
        if (
            len(cf) != len(f)
            or not np.array_equal(v, vertices[parent])
            or not np.array_equal(parent[cf], f)
        ):
            raise ValueError("cut changed source correspondence")
    else:
        v, cf, parent = vertices, f, np.arange(len(vertices))
    cf = cf.astype(np.int32 if len(v) <= np.iinfo(np.int32).max else np.int64, copy=False)
    topology = _topology_report(v, cf, areas, check)
    kept = len(igl.boundary_loop_all(cf.astype(np.int64))) - 1
    if kept < 1 or topology["euler_characteristic"] != 1 - kept:
        raise ValueError("minimal cut is not a planar domain")
    boundary = igl.boundary_loop(cf.astype(np.int64)).astype(np.int64)
    return dict(vertices=v, faces=cf, parent=parent.astype(np.int64), boundary=boundary), dict(
        original=original, cut=topology, seam_duplicates=len(v) - len(vertices), holes=kept
    )


def _rim_seams(vertices, faces, loops, outer, ends):
    """Shortest seams that join each rim in ``ends`` to the outline or to a rim already joined.

    A seam runs through interior vertices only, so it touches no other rim and every hole
    stays a hole. Returns one vertex path per seam, as ``igl.cut_to_disk`` does. Raises
    ValueError when a rim cannot be reached.
    """
    count = len(vertices)
    rim_of = np.full(count, -1, np.int64)
    for k, loop in enumerate(loops):
        rim_of[loop] = k
    tail, head = faces.ravel(), np.roll(faces, -1, axis=1).ravel()
    key = np.unique(np.minimum(tail, head) * count + np.maximum(tail, head))
    a, b = key // count, key % count
    # An edge between two rim vertices runs along a rim or straight across a gap.
    inner = (rim_of[a] < 0) | (rim_of[b] < 0)
    a, b = a[inner], b[inner]
    weight = np.linalg.norm(vertices[a] - vertices[b], axis=1) + 1e-9
    joined = [outer]
    seams = []
    for k in sorted(ends, key=lambda end: -len(loops[end])):
        open_to = (rim_of < 0) | (rim_of == k) | np.isin(rim_of, joined)
        keep = open_to[a] & open_to[b]
        # One extra node stands for the whole rim, so a single search starts from all of it.
        start = loops[k]
        graph = coo_matrix(
            (
                np.r_[weight[keep], np.full(len(start), 1e-9)],
                (np.r_[a[keep], np.full(len(start), count)], np.r_[b[keep], start]),
            ),
            shape=(count + 1, count + 1),
        ).tocsr()
        dist, before = dijkstra(graph, directed=False, indices=count, return_predecessors=True)
        targets = np.concatenate([loops[j] for j in joined])
        at = int(targets[np.argmin(dist[targets])])
        if not np.isfinite(dist[at]):
            raise ValueError("no seam joins the end of a wrap to the outline")
        path = [at]
        while before[at] != count:
            at = int(before[at])
            path.append(at)
        seams.append(np.asarray(path, np.int64))
        joined.append(k)
    return seams


def _part(vertices, faces, rows):
    """The largest edge-connected manifold part of the face rows ``rows`` of a piece.

    Returns ``(vertices, faces, parent, rows)``: its own vertices and faces, the piece
    vertex of each of its vertices and the rows it keeps, in order.
    """
    split, parent, groups, _ = _manifold_pieces(faces[rows].astype(np.int64, copy=False))
    group = np.sort(max(groups, key=len))
    used, local = np.unique(split[group], return_inverse=True)
    return vertices[parent[used]], local.reshape(-1, 3), parent[used], rows[group]


def _planar_seed(cut, drop=False):
    """Global LSCM for a planar domain with holes; None when it folds.

    With ``drop``, a map that folds on at most ``SEED_DROP`` of the area is kept for the
    rest: returns ``(uv, initialization, rows)`` with the face rows that are not folded
    (None when all are), and the caller takes the folded faces out of the piece.
    """
    from .matrix import _global_lscm

    uv, pins, _ = _global_lscm(cut)
    faces = cut["faces"].astype(np.int64, copy=False)
    tri = uv[faces]
    signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if signed.sum() < 0:
        uv[:, 0] *= -1
        signed = -signed
    initialization = dict(method="LSCM", backend="libigl", pins=pins.tolist())
    if not np.isfinite(uv).all():
        return None
    if not drop:
        return None if (signed <= 0).any() else (uv, initialization)
    return _without_folds(cut, uv, signed, initialization)


def _without_folds(cut, uv, signed, initialization):
    """The face rows a seed is usable on: ``(uv, initialization, rows)``, or None.

    ``rows`` is None when no face is folded; None is returned when the folded faces hold
    more than ``SEED_DROP`` of the area.
    """
    faces = cut["faces"].astype(np.int64, copy=False)
    world = cut["vertices"]
    area = np.linalg.norm(
        np.cross(world[faces[:, 1]] - world[faces[:, 0]], world[faces[:, 2]] - world[faces[:, 0]]),
        axis=1,
    )
    # A face squeezed to nothing is as unusable a start as a folded one.
    folded = signed <= SEED_FLOOR * np.median(signed / np.maximum(area, 1e-300)) * area
    if not folded.any():
        return uv, initialization, None
    if area[folded].sum() > SEED_DROP * area.sum():
        return None
    initialization["folded_faces"] = int(folded.sum())
    return uv, initialization, np.flatnonzero(~folded)


def _seed_spread(cut, seeded):
    """Edge stretch p99 / p01 of a seed on the faces it keeps: 1 for a map without drift."""
    uv, _, rows = seeded
    faces = cut["faces"].astype(np.int64, copy=False)
    if rows is not None:
        faces = faces[rows]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    world = cut["vertices"]
    length = np.linalg.norm(world[edges[:, 0]] - world[edges[:, 1]], axis=1)
    ratio = (
        np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1)[length > 0] / length[length > 0]
    )
    low, high = np.percentile(ratio, [1, 99])
    return float(high / low) if low > 0 else np.inf


def _unrolled_seed(cut):
    """Unroll a piece that bends around one axis, as a wrap of a scroll does; None otherwise.

    The axis is the direction the face normals avoid, and it stands at the centre of the
    circle that fits the piece's cross-section. Around it every vertex has a radius and an
    angle, followed from vertex to vertex so that it may pass a full turn. The map is the
    arc length of the piece's median cross-section against the position along the axis.
    Unlike LSCM it cannot drift: on a rough sheet the conformal scale wanders by orders of
    magnitude, and SLIM does not undo that in a few hundred steps on millions of faces.

    Returns ``(uv, initialization, rows)`` like ``_planar_seed`` with ``drop``. None for a
    piece that closes around the axis (its angle has no single value), for one with no
    such axis, and when the map folds on more than ``SEED_DROP`` of the area.
    """
    world = cut["vertices"].astype(np.float64, copy=False)
    faces = cut["faces"].astype(np.int64, copy=False)
    normal = np.cross(
        world[faces[:, 1]] - world[faces[:, 0]], world[faces[:, 2]] - world[faces[:, 0]]
    )
    area = np.linalg.norm(normal, axis=1)
    solid = area > 0
    if not solid.any():
        return None
    unit = normal[solid] / area[solid, None]
    _, frame = np.linalg.eigh((unit * area[solid, None]).T @ unit)
    axis, first = frame[:, 0], frame[:, 2]
    second = np.cross(axis, first)
    origin = world.mean(0)
    plane = (world - origin) @ np.column_stack([first, second])
    along = (world - origin) @ axis
    # The axis stands at the centre of the circle that fits the cross-section (Kasa).
    spot = plane[faces[solid]].mean(1)
    share = np.sqrt(area[solid])[:, None]
    centre, _, rank, _ = np.linalg.lstsq(
        np.column_stack([2 * spot, np.ones(len(spot))]) * share,
        (spot**2).sum(1, keepdims=True) * share,
        rcond=1e-12,
    )
    if rank < 3:
        return None
    arm = plane - centre[:2, 0]
    radius = np.linalg.norm(arm, axis=1)
    angle = np.arctan2(arm[:, 1], arm[:, 0])
    if not (np.isfinite(radius).all() and np.isfinite(angle).all()):
        return None

    def turn(value):
        return (value + np.pi) % (2 * np.pi) - np.pi

    # Follow the angle along a spanning tree; pointer jumping sums the steps to the root.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    count = len(world)
    graph = coo_matrix(
        (np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), shape=(count,) * 2
    )
    order, up = breadth_first_order(graph.tocsr(), int(edges[0, 0]), directed=False)
    if len(order) != count:
        return None
    up[order[0]] = order[0]
    theta = turn(angle - angle[up])
    while (up != order[0]).any():
        theta += theta[up]
        up = up[up]
    step = turn(angle[edges[:, 1]] - angle[edges[:, 0]])
    if (np.abs(theta[edges[:, 1]] - theta[edges[:, 0]] - step) > np.pi).any():
        return None
    low, extent = float(theta.min()), float(np.ptp(theta))
    if not extent > 0:
        return None
    # Median radius per angle step, smoothed: the cross-section the piece is unrolled along.
    bins = max(8, int(np.ceil(extent / UNROLL_STEP)))
    which = np.minimum(((theta - low) / extent * bins).astype(np.int64), bins - 1)
    by = np.lexsort((radius, which))
    bounds = np.searchsorted(which[by], np.arange(bins + 1))
    filled = bounds[1:] > bounds[:-1]
    middle = low + (np.arange(bins) + 0.5) * extent / bins
    profile = np.interp(middle, middle[filled], radius[by[(bounds[:-1] + bounds[1:])[filled] // 2]])
    profile = np.convolve(np.pad(profile, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    speed = np.hypot(profile, np.gradient(profile, middle))
    arc = np.concatenate([[0.0], np.cumsum(speed * extent / bins)])
    uv = np.column_stack([np.interp(theta, low + np.arange(bins + 1) * extent / bins, arc), along])
    tri = uv[faces]
    signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if signed.sum() < 0:
        uv[:, 0] = arc[-1] - uv[:, 0]
        signed = -signed
    uv = np.ascontiguousarray(uv - uv.min(0))
    initialization = dict(
        method="unroll",
        axis=[float(t) for t in axis],
        angle_degrees=float(np.degrees(extent)),
        cross_section_length=float(arc[-1]),
    )
    return _without_folds(cut, uv, signed, initialization)


def _capped_seed(cut, check):
    """Mean-value seed of a planar domain with holes, for when LSCM folds too much of it.

    A fan around one extra vertex caps every hole; what is left is a disk, and a disk has a
    valid mean-value or uniform map on a circle. The caps go away again, so the holes stay.
    The outer loop is the longest one: any loop gives a valid map, the longest the least
    distorted. Raises ValueError when the capped domain is not a disk or its map fails.
    """
    import igl

    from .native_sheet import mean_value_map

    world = cut["vertices"]
    faces = cut["faces"].astype(np.int64, copy=False)
    loops = [np.asarray(loop, np.int64) for loop in igl.boundary_loop_all(faces)]
    length = [
        float(np.linalg.norm(world[loop] - world[np.roll(loop, -1)], axis=1).sum())
        for loop in loops
    ]
    outer = int(np.argmax(length))
    centre_of = np.full(len(world), -1, np.int64)
    centres = []
    for k, loop in enumerate(loops):
        if k != outer:
            centre_of[loop] = len(world) + len(centres)
            centres.append(world[loop].mean(0))
    tail, head = faces.ravel(), np.roll(faces, -1, axis=1).ravel()
    key = np.minimum(tail, head) * len(world) + np.maximum(tail, head)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    rim = (counts[inverse] == 1) & (centre_of[tail] >= 0)
    capped = np.concatenate([faces, np.column_stack([head[rim], tail[rim], centre_of[tail[rim]]])])
    capped = capped.astype(cut["faces"].dtype, copy=False)
    disk = dict(
        vertices=np.concatenate([world, np.asarray(centres).reshape(-1, 3)]),
        faces=capped,
        boundary=igl.boundary_loop(capped.astype(np.int64)).astype(np.int64),
    )
    uv, initialization = mean_value_map(disk, check, uniform_fallback=True)
    return np.ascontiguousarray(uv[: len(world)]), dict(initialization, capped_holes=len(centres))


def _stats(vertices, faces, uv):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ratio = np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1) / np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )
    world = vertices[faces]
    area3 = np.linalg.norm(np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0]), axis=1)
    tri = uv[faces]
    area2 = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    scale = area2 / area3

    def spread(values):
        low, mid, high = np.percentile(values, [1, 50, 99])
        return dict(
            min=float(values.min()),
            p01=float(low),
            p50=float(mid),
            p99=float(high),
            max=float(values.max()),
        )

    return dict(edge_stretch=spread(ratio), area_ratio=spread(scale))


def _orient(vertices, uv):
    """Rotate so +v follows world +Z: every level of a scroll is laid out the same way."""
    direction = np.linalg.lstsq(
        uv - uv.mean(0), vertices[:, 2] - vertices[:, 2].mean(), rcond=None
    )[0]
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction = direction / norm
        uv = uv @ np.column_stack([[direction[1], -direction[0]], direction])
    return uv - uv.min(0)


def _raster_charts(cut, uv, face_ids, source_ids, cut_base, chart_faces):
    """Split one flattened piece into compact UV cells of about ``chart_faces`` faces."""
    faces = cut["faces"].astype(np.int64)
    tri = uv[faces]
    area = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2
    cell = max(float(np.sqrt(chart_faces * area.mean())), 1e-9)
    key = np.floor(tri.mean(1) / cell).astype(np.int64)
    order = np.lexsort((face_ids, key[:, 0], key[:, 1]))
    key = key[order]
    bounds = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
    for rows in np.split(order, bounds):
        rows = np.sort(rows)
        used, local = np.unique(faces[rows], return_inverse=True)
        yield dict(
            vertices=cut["vertices"][used],
            faces=local.reshape(-1, 3),
            uv=uv[used],
            face_ids=face_ids[rows],
            vertex_ids=source_ids[used],
            # Cut-vertex identity: two charts share an edge only when their UVs agree.
            cut_ids=cut_base + used,
        )


def _save_mapped(store, component, charts, quality):
    """Persist raster charts as MAPPED with their global UV; returns mapped area."""
    records, edges, total = [], [], 0.0
    for chart in charts:
        store.check()
        v, f, uv = chart["vertices"], chart["faces"], chart["uv"]
        name = f"chart-{int(chart['face_ids'].min()):012d}"
        area = float(
            np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
            / 2
        )
        store.save_arrays(
            "charts/" + name + ".npz",
            vertices=v,
            faces=f,
            uv=uv,
            face_ids=chart["face_ids"],
            vertex_ids=chart["vertex_ids"],
        )
        store.save_arrays("uv/" + name + ".npz", uv=uv)
        low, high = uv.min(0), uv.max(0)
        records.append(
            (
                name,
                int(component),
                area,
                len(f),
                "uv/" + name + ".npz",
                float(low[0]),
                float(high[0]),
                float(low[1]),
                float(high[1]),
                json.dumps({"local": quality, "assembled": quality["edge_stretch_gate"]}),
            )
        )
        local = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        ids = chart["cut_ids"][local]
        orientation = np.where(ids[:, 0] < ids[:, 1], 1, -1)
        order = np.argsort(ids, axis=1)
        local = np.take_along_axis(local, order, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        _, first, counts = np.unique(ids, axis=0, return_index=True, return_counts=True)
        for index in first[counts == 1]:
            edges.append(
                (
                    int(ids[index, 0]),
                    int(ids[index, 1]),
                    name,
                    int(local[index, 0]),
                    int(local[index, 1]),
                    int(orientation[index]),
                )
            )
        total += area
    with store.db:
        store.db.executemany(
            "INSERT OR IGNORE INTO charts(name,component,area,faces,status,uv_path,minx,maxx,miny,maxy,quality) "
            "VALUES (?,?,?,?,'MAPPED',?,?,?,?,?,?)",
            records,
        )
        store.db.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?)", edges)
    return total


def _fallback_charts(store, component, vertices, faces, face_ids, vertex_ids, chart_faces):
    """Seeded local charts of one piece for the chart route (``assemble`` welds them)."""
    items = []
    for first in range(0, len(faces), chart_faces):
        rows = slice(first, first + chart_faces)
        items += [
            (component, piece)
            for piece in _pieces(
                vertices, faces[rows], face_ids[rows], vertex_ids, check=store.check
            )
        ]
    return items


def _fallback(store, component, vertices, faces, face_ids, vertex_ids, chart_faces, app_run):
    """Chart route for one piece: local charts that ``assemble`` welds as usual."""
    for first in range(0, len(faces), chart_faces):
        rows = slice(first, first + chart_faces)
        _flush_slim(
            store,
            _fallback_charts(
                store, component, vertices, faces[rows], face_ids[rows], vertex_ids, chart_faces
            ),
            app_run,
            0,
            -1,
        )


def _unchecked():
    """Stand-in for ``store.check`` in prepare workers; the parent checks between results."""


def _sheet_seed(cut, check, scratch):
    """Global LSCM seed; when it folds, mean-value, and the uniform map if that inverts.

    The matrix route's ``_lscm_seed`` without the last step: a piece with no valid seed
    leaves the whole-sheet route for local charts, so a valid uniform seed keeps it
    whole (SLIM then removes the uniform map's distortion).
    """
    from .matrix import _global_lscm
    from .native_sheet import mean_value_map

    uv, pins, _ = _global_lscm(cut)
    tri = uv[cut["faces"]]
    signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if signed.sum() < 0:
        uv[:, 0] *= -1
        signed = -signed
    folded = int((signed <= 0).sum())
    if np.isfinite(uv).all() and folded == 0:
        return uv, dict(method="LSCM", backend="libigl", pins=pins.tolist())
    uv, initialization = mean_value_map(cut, check, diagnostics=scratch, uniform_fallback=True)
    return uv, dict(initialization, lscm_rejected=dict(folded_faces=folded))


def _prepare(vertices, faces, minimal, check):
    """Cut one piece and seed its map; returns the fields ``commit`` needs, or an error.

    With ``minimal`` the piece keeps its holes. Handles go first: the faces around each
    tunnel leave the piece (``handles.handle_collars``), and so do the few faces on which
    the seed folds. ``rows`` then lists the face rows that stay in the whole-sheet map
    (None for all); the others are flattened as small pieces of their own.
    """
    from .native_sheet import cut_component

    try:
        with tempfile.TemporaryDirectory(prefix="sheet-seed-") as scratch:
            made = seeded = rows = handles = None
            if minimal:
                try:
                    made, seeded, rows, handles = _keep_holes(vertices, faces, check)
                except ValueError:
                    seeded = None
            if seeded:
                (cut, topology), (seed, initialization) = made, seeded
            elif handles and handles["loops"]:
                # A disk cut through the handles is the map that collapses: use charts.
                raise ValueError("no whole-sheet seed after handle removal")
            else:
                rows = None
                cut, topology = cut_component(vertices, faces, check)
                seed, initialization = _sheet_seed(cut, check, scratch)
        tri = seed[cut["faces"]]
        if not np.isfinite(seed).all() or np.any(
            cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) <= 0
        ):
            raise ValueError("no valid whole-sheet seed")
        return dict(
            cut=cut,
            seed=seed,
            topology=topology,
            initialization=initialization,
            minimal=bool(seeded),
            rows=rows,
            handles=handles,
        )
    except (ValueError, RuntimeError, MemoryError) as error:
        return dict(error=str(error) or type(error).__name__)


def _keep_holes(vertices, faces, check):
    """Minimal cut and seed of a piece without its handles: ``(made, seeded, rows, handles)``."""
    import igl

    from .handles import handle_collars
    from .native_sheet import cut_component

    faces = faces.astype(np.int64, copy=False)
    world, f, up, rows = vertices, faces, None, None
    aside, handles = handle_collars(vertices, faces, check)
    if aside.any():
        world, f, up, rows = _part(vertices, faces, np.flatnonzero(~aside))
        lengths = handles.pop("loop_length")
        handles = dict(
            genus=handles["genus"],
            loops=handles["loops"],
            genus_left=handles["genus_left"],
            collar_faces=handles["collar_faces"],
            rounds=len(handles["rounds"]),
            loop_length=dict(
                min=float(np.min(lengths)),
                median=float(np.median(lengths)),
                max=float(np.max(lengths)),
            ),
        )
    else:
        handles = None
    made = _minimal_cut(world, f, check, rows is not None)
    if made is None:
        if rows is None:
            return None, None, None, None
        # The removed faces opened the piece to its outline: no hole is left to keep.
        made = cut_component(world, f, check)
    cut, topology = made
    seeded = _planar_seed(cut, drop=True)
    if seeded is None or _seed_spread(cut, seeded) > SEED_SPREAD:
        unrolled = _unrolled_seed(cut)
        if unrolled is not None and (
            seeded is None or _seed_spread(cut, unrolled) < _seed_spread(cut, seeded)
        ):
            seeded = unrolled
    if seeded is None:
        try:
            seeded = (*_capped_seed(cut, check), None)
        except (ValueError, RuntimeError):
            return made, None, rows, handles
    uv, initialization, keep = seeded
    if keep is not None:
        # The faces that stay are still a planar domain, and the same map is valid on them.
        part_world, part_faces, part_up, keep = _part(cut["vertices"], cut["faces"], keep)
        cut = dict(
            vertices=part_world,
            faces=part_faces.astype(cut["faces"].dtype, copy=False),
            parent=cut["parent"][part_up],
            boundary=igl.boundary_loop(part_faces).astype(np.int64),
        )
        uv = uv[part_up]
        rows = keep if rows is None else rows[keep]
        topology = dict(topology, seed_dropped_faces=int(len(f) - len(keep)))
    if up is not None:
        cut["parent"] = up[cut["parent"]]
    if handles:
        topology = dict(topology, handles=handles)
    return (cut, topology), (uv, initialization), rows, handles


def _prepare_task(task):
    vertices, faces = task
    return _prepare(vertices, faces, True, _unchecked)


def flatten_sheet(reader, store, chart_faces=1024, *, app_run=None, offline=False, workers=1):
    """Flatten every edge-connected piece of a single-level mesh as one map.

    Idempotent: a completed store returns its saved receipt. Interrupted work restarts
    from the pieces that were not committed. ``workers`` > 1 cuts and seeds pieces in
    that many processes; the results, and so the outputs, are the same as with one.
    """
    from .flatboi import resolve_app_run, smoke_test, solve_batch
    from .patches import check_flatboi

    if type(chart_faces) is not int or not 1 <= chart_faces <= 8192:
        raise ValueError("chart_faces must be1..8192")
    if type(workers) is not int or workers < 1:
        raise ValueError("flatten workers must be a positive integer")
    _schema(store)
    if store.get("flatten_complete", False) and store.get("sheet_flatten") is not None:
        return store.get("sheet_flatten")
    app_run = check_flatboi(str(resolve_app_run(app_run, offline=offline)), "historical")["app_run"]
    smoke_test(app_run)
    store.phase = "flatten"
    started = time.monotonic()
    reader.validate()
    rows = np.asarray(
        reader.db.execute("SELECT id,a,b,c FROM faces ORDER BY id").fetchall(), np.int64
    )
    if not len(rows):
        raise ValueError("empty native surface")
    ids = []
    xyz = []
    for row in reader.db.execute("SELECT id,x,y,z FROM vertices ORDER BY id"):
        ids.append(row[0])
        xyz.append(row[1:])
    ids = np.asarray(ids, np.int64)
    xyz = np.asarray(xyz, np.float64)
    reader.validate()
    faces = np.searchsorted(ids, rows[:, 1:4])
    areas = np.linalg.norm(
        np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]]), axis=1
    )
    rows_last = int(rows[-1, 0])
    with store.db:
        store.set("source_faces", int(len(faces)))
        store.set("source_area", float(areas.sum() / 2))
        store.set("last_face", int(rows[-1, 0]))
    split, parent, groups, repair = _manifold_pieces(faces)
    # Stores written before per-piece rows keep their whole receipt in one state value.
    done = set(store.get("sheet_pieces", []))
    receipt = store.get("sheet_receipt", dict(schema=SCHEMA, repair=repair, pieces=[], fallback=[]))
    finished = store.db.execute(
        "SELECT value FROM state WHERE name GLOB ? ORDER BY name", (PIECE_ROW + "*",)
    ).fetchall()
    for (value,) in finished:
        entry = json.loads(value)
        receipt[entry["kind"]].append(entry["record"])
        done.add(entry["component"])
    sequence = [len(finished)]
    cut_base = int(ids.max()) + 1
    prepared = []

    def prepare(piece, minimal):
        piece.pop("error", None)
        piece.update(_prepare(piece["vertices"], piece["faces"], minimal, store.check))

    progress = dict(pieces=len(groups), finished=len(done), prepared=0, prepare_seconds=0.0)
    shown = [0.0]

    def report(force=False):
        # Visible while it runs: work.sqlite state "sheet_progress" and STATUS.json.
        now = time.monotonic()
        if force or now - shown[0] >= 2.0:
            shown[0] = now
            progress["prepare_seconds"] = round(now - started, 1)
            with store.db:
                store.set("sheet_progress", progress)

    for group in groups:
        store.check()
        component = int(rows[group[0], 0])
        group = np.sort(group)
        used, local = np.unique(split[group], return_inverse=True)
        piece = dict(
            component=component,
            vertices=xyz[parent[used]],
            faces=local.reshape(-1, 3),
            face_ids=rows[group, 0],
            vertex_ids=ids[parent[used]],
        )
        piece["cut_base"] = cut_base
        # A cut adds at most two copies per split edge; reserve the worst case per piece.
        cut_base += 8 * len(used) + 16
        if component in done:
            continue
        prepared.append(piece)
    report(True)
    if workers > 1 and len(prepared) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(workers, len(prepared))) as pool:
            tasks = ((piece["vertices"], piece["faces"]) for piece in prepared)
            for piece, made in zip(prepared, pool.map(_prepare_task, tasks, chunksize=4)):
                store.check()
                piece.update(made)
                progress["prepared"] += 1
                report()
    else:
        for piece in prepared:
            prepare(piece, True)
            progress["prepared"] += 1
            report()
    report(True)

    def commit(piece, uv, solver):
        cut = piece["cut"]
        tri = uv[cut["faces"]]
        signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if not np.isfinite(uv).all() or np.any(signed <= 0):
            raise ValueError("whole-sheet solve folded or is not finite")
        uv = _orient(cut["vertices"], uv)
        stats = _stats(cut["vertices"], cut["faces"].astype(np.int64), uv)
        stretch = stats["edge_stretch"]
        if piece["minimal"] and not (stretch["p01"] > 0.5 and stretch["p99"] < 2.0):
            # A kept hole was really a wrap: flatten this piece as a disk instead.
            raise _NeedsDisk(
                f"edge stretch p01 {stretch['p01']:.3f}, p50 {stretch['p50']:.3f}, "
                f"p99 {stretch['p99']:.3f}"
            )
        if piece.get("retried") and not (
            stretch["p01"] > DISK_STRETCH[0] and stretch["p99"] < DISK_STRETCH[1]
        ):
            # Neither with its holes nor as a disk: a collapsed map is worse than charts.
            raise ValueError("whole-sheet map is distorted with holes kept and as a disk")
        quality = dict(
            method="SLIM",
            backend="flatboi",
            scope="sheet",
            iterations=solver["steps"],
            converged=solver["converged"],
            initialization=initialization_name(piece["initialization"]),
            edge_stretch_gate=dict(
                edge_stretch_min=stats["edge_stretch"]["min"],
                edge_stretch_max=stats["edge_stretch"]["max"],
            ),
        )
        source_ids = piece["vertex_ids"][cut["parent"]]
        rows = piece.get("rows")
        face_ids = piece["face_ids"] if rows is None else piece["face_ids"][rows]
        _save_mapped(
            store,
            piece["component"],
            _raster_charts(cut, uv, face_ids, source_ids, piece["cut_base"], chart_faces),
            quality,
        )
        aside = 0 if rows is None else set_aside(piece, rows)
        finish(
            "pieces",
            piece["component"],
            dict(
                component=piece["component"],
                faces=int(len(cut["faces"])),
                aside_faces=int(aside),
                handles=piece.get("handles"),
                seam_vertex_copies=int(piece["topology"]["seam_duplicates"]),
                holes=int(piece["topology"].get("holes", 0)),
                euler_characteristic=piece["topology"]["original"]["euler_characteristic"],
                boundary_edges=piece["topology"]["original"]["boundary_edges"],
                initialization=quality["initialization"],
                iterations=solver["steps"],
                converged=solver["converged"],
                **stats,
            ),
        )

    def set_aside(piece, rows):
        """Flatten the faces a piece gave up as charts of their own components."""
        rest = np.setdiff1d(np.arange(len(piece["faces"])), rows)
        if not len(rest):
            return 0
        faces = piece["faces"][rest].astype(np.int64, copy=False)
        _, _, groups, _ = _manifold_pieces(faces)
        items = []
        for group in groups:
            ids = piece["face_ids"][rest[group]]
            # A chart joins the sheet only inside its component: these stay apart, each
            # under its first face (or past the last face when that names the piece).
            component = int(ids.min())
            if component == piece["component"]:
                component += int(rows_last) + 1
            items += _fallback_charts(
                store,
                component,
                piece["vertices"],
                faces[group],
                ids,
                piece["vertex_ids"],
                chart_faces,
            )
        _flush_slim(store, items, app_run, 0, -1)
        return len(rest)

    def initialization_name(record):
        return record.get("method") or record.get("algorithm") or "mean-value"

    def finish(kind, component, record):
        finish_all(kind, [(component, record)])

    def finish_all(kind, entries):
        # One small row per piece: rewriting the whole receipt per piece grew as the square
        # of the piece count (hours on real levels with tens of thousands of pieces). A
        # group's rows commit together, so a resume regroups the remaining pieces the same.
        progress["finished"] += len(entries)
        with store.db:
            for component, record in entries:
                store.set(
                    f"{PIECE_ROW}{sequence[0]:09d}",
                    dict(kind=kind, component=component, record=record),
                )
                sequence[0] += 1
            store.set("elapsed_s", store.elapsed)
            store.set("sheet_progress", progress)
        for component, record in entries:
            receipt[kind].append(record)
            done.add(component)

    def chart_route(piece, reason):
        _fallback(
            store,
            piece["component"],
            piece["vertices"],
            piece["faces"],
            piece["face_ids"],
            piece["vertex_ids"],
            chart_faces,
            app_run,
        )
        finish(
            "fallback",
            piece["component"],
            dict(component=piece["component"], faces=int(len(piece["faces"])), reason=reason),
        )

    # Pieces with no whole-sheet seed (mostly small ones) share Flatboi calls: one solve
    # per piece cost a process start each, hours on levels with thousands of such pieces.
    failed = [piece for piece in prepared if "error" in piece]
    while failed:
        items, members, faces_in = [], [], 0
        while failed and (not members or faces_in < SLIM_BATCH_FACES):
            piece = failed.pop(0)
            items += _fallback_charts(
                store,
                piece["component"],
                piece["vertices"],
                piece["faces"],
                piece["face_ids"],
                piece["vertex_ids"],
                chart_faces,
            )
            members.append(piece)
            faces_in += len(piece["faces"])
        _flush_slim(store, items, app_run, 0, -1)
        receipt_entries = [
            (
                piece["component"],
                dict(
                    component=piece["component"],
                    faces=int(len(piece["faces"])),
                    reason=piece["error"],
                ),
            )
            for piece in members
        ]
        finish_all("fallback", receipt_entries)
    queue = []
    group, group_faces = [], 0
    for piece in (p for p in prepared if "error" not in p):
        group.append(piece)
        group_faces += len(piece["faces"])
        if group_faces >= GROUP_FACES:
            queue.append(group)
            group, group_faces = [], 0
    if group:
        queue.append(group)
    while queue:
        group = queue.pop(0)
        store.check()
        try:
            uvs, solver = solve_batch(
                [(p["cut"]["vertices"], p["cut"]["faces"], p["seed"]) for p in group],
                app_run,
                iterations=SLIM_ITERATIONS,
                tolerance=SLIM_TOLERANCE,
                check=store.check,
                timeout=max(60.0, store.seconds - store.elapsed),
            )
        except RuntimeError as error:
            # Isolate a native failure (for example PaStiX's negative diagonal) by bisection.
            if len(group) > 1:
                mid = len(group) // 2
                queue[:0] = [group[:mid], group[mid:]]
            else:
                chart_route(group[0], str(error))
            continue
        for piece, uv in zip(group, uvs):
            try:
                commit(piece, uv, solver)
            except _NeedsDisk as stretched:
                if piece.get("rows") is not None:
                    # Without its handles and folds the piece still stretched: the disk
                    # cut would run through them, so it goes to charts whole.
                    chart_route(
                        piece, f"whole-sheet map is distorted after handle removal ({stretched})"
                    )
                    continue
                prepare(piece, False)
                piece["retried"] = True
                if "error" in piece:
                    chart_route(piece, piece["error"])
                else:
                    queue.append([piece])
            except ValueError as error:
                chart_route(piece, str(error))
    receipt["seconds"] = time.monotonic() - started
    with store.db:
        # Mapped charts already agree on shared vertices: nothing to weld or cut.
        store.db.execute(
            "DELETE FROM frontier WHERE child IN (SELECT name FROM charts WHERE status='MAPPED')"
        )
        store.set("sheet_flatten", receipt)
        store.set("flatten_complete", True)
        store.set("elapsed_s", store.elapsed)
    return receipt
