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
from scipy.sparse.csgraph import connected_components

from .legacy_charts import cross2
from .native_charts import (
    SLIM_ITERATIONS,
    SLIM_TOLERANCE,
    _flush_slim,
    _pieces,
    _schema,
)

SCHEMA = "native-sheet-flatten-v1"


# One Flatboi call per group keeps process start-up amortized over small pieces.
GROUP_FACES = 4_000_000


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


def _minimal_cut(vertices, faces, check):
    """Cut a piece only where it cannot lie flat; holes stay holes.

    ``cut_component`` joins every hole to the outer boundary, and each of those cuts can
    open or overlap in the render. A hole in a near-developable sheet needs none: its rim
    turns by about -2*pi (Gauss-Bonnet), while the outer boundary turns by about +2*pi and
    either end of a closed wrap by about 0. Loops that turn by less than -pi are capped
    for the topological cut only, so just wraps and handles are cut.

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
    holes = [k for k in range(len(loops)) if turning[k] < -np.pi and k != outer]
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


def _planar_seed(cut):
    """Global LSCM for a planar domain with holes; None when it folds."""
    from .matrix import _global_lscm

    uv, pins, _ = _global_lscm(cut)
    tri = uv[cut["faces"]]
    signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if signed.sum() < 0:
        uv[:, 0] *= -1
        signed = -signed
    if not np.isfinite(uv).all() or (signed <= 0).any():
        return None
    return uv, dict(method="LSCM", backend="libigl", pins=pins.tolist())


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


def _fallback(store, component, vertices, faces, face_ids, vertex_ids, chart_faces, app_run):
    """Chart route for one piece: local charts that ``assemble`` welds as usual."""
    for first in range(0, len(faces), chart_faces):
        rows = slice(first, first + chart_faces)
        items = [
            (component, piece)
            for piece in _pieces(
                vertices, faces[rows], face_ids[rows], vertex_ids, check=store.check
            )
        ]
        _flush_slim(store, items, app_run, 0, -1)


def flatten_sheet(reader, store, chart_faces=1024, *, app_run=None, offline=False):
    """Flatten every edge-connected piece of a single-level mesh as one map.

    Idempotent: a completed store returns its saved receipt. Interrupted work restarts
    from the pieces that were not committed.
    """
    from .flatboi import resolve_app_run, solve_batch
    from .matrix import _lscm_seed
    from .native_sheet import cut_component
    from .patches import check_flatboi

    if type(chart_faces) is not int or not 1 <= chart_faces <= 8192:
        raise ValueError("chart_faces must be1..8192")
    _schema(store)
    if store.get("flatten_complete", False) and store.get("sheet_flatten") is not None:
        return store.get("sheet_flatten")
    app_run = check_flatboi(str(resolve_app_run(app_run, offline=offline)), "historical")["app_run"]
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
    with store.db:
        store.set("source_faces", int(len(faces)))
        store.set("source_area", float(areas.sum() / 2))
        store.set("last_face", int(rows[-1, 0]))
    done = set(store.get("sheet_pieces", []))
    split, parent, groups, repair = _manifold_pieces(faces)
    receipt = store.get("sheet_receipt", dict(schema=SCHEMA, repair=repair, pieces=[], fallback=[]))
    cut_base = int(ids.max()) + 1
    prepared = []

    def prepare(piece, minimal):
        piece.pop("error", None)
        try:
            with tempfile.TemporaryDirectory(prefix="sheet-seed-") as scratch:
                made = seeded = None
                if minimal:
                    try:
                        made = _minimal_cut(piece["vertices"], piece["faces"], store.check)
                        seeded = made and _planar_seed(made[0])
                    except ValueError:
                        seeded = None
                if seeded:
                    (cut, topology), (seed, initialization) = made, seeded
                else:
                    cut, topology = cut_component(piece["vertices"], piece["faces"], store.check)
                    seed, initialization = _lscm_seed(cut, store.check, scratch)
            tri = seed[cut["faces"]]
            if not np.isfinite(seed).all() or np.any(
                cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) <= 0
            ):
                raise ValueError("no valid whole-sheet seed")
            piece.update(
                cut=cut,
                seed=seed,
                topology=topology,
                initialization=initialization,
                minimal=bool(seeded),
            )
        except (ValueError, RuntimeError, MemoryError) as error:
            piece["error"] = str(error) or type(error).__name__

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
        prepare(piece, True)
        prepared.append(piece)

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
            raise _NeedsDisk
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
        _save_mapped(
            store,
            piece["component"],
            _raster_charts(cut, uv, piece["face_ids"], source_ids, piece["cut_base"], chart_faces),
            quality,
        )
        receipt["pieces"].append(
            dict(
                component=piece["component"],
                faces=int(len(cut["faces"])),
                seam_vertex_copies=int(piece["topology"]["seam_duplicates"]),
                holes=int(piece["topology"].get("holes", 0)),
                euler_characteristic=piece["topology"]["original"]["euler_characteristic"],
                boundary_edges=piece["topology"]["original"]["boundary_edges"],
                initialization=quality["initialization"],
                iterations=solver["steps"],
                converged=solver["converged"],
                **stats,
            )
        )
        finish(piece["component"])

    def initialization_name(record):
        return record.get("method") or record.get("algorithm") or "mean-value"

    def finish(component):
        done.add(component)
        with store.db:
            store.set("sheet_pieces", sorted(done))
            store.set("sheet_receipt", receipt)
            store.set("elapsed_s", store.elapsed)

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
        receipt["fallback"].append(
            dict(component=piece["component"], faces=int(len(piece["faces"])), reason=reason)
        )
        finish(piece["component"])

    for piece in prepared:
        if "error" in piece:
            chart_route(piece, piece["error"])
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
            except _NeedsDisk:
                prepare(piece, False)
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
