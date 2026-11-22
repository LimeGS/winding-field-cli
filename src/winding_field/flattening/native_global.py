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

A piece that cannot be flattened this way (non-manifold input, a failed or folded
solve) falls back to the chart route under its own component and is listed in the
receipt; nothing is dropped silently.
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


def _edge_pieces(faces):
    """Edge-connected face groups (a point contact does not join two pieces)."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, edge_ids, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    face_of = np.tile(np.arange(len(faces)), 3)
    order = np.argsort(edge_ids, kind="stable")
    sorted_ids, sorted_faces = edge_ids[order], face_of[order]
    same = sorted_ids[1:] == sorted_ids[:-1]
    graph = coo_matrix(
        (np.ones(int(same.sum()), np.uint8), (sorted_faces[:-1][same], sorted_faces[1:][same])),
        shape=(len(faces), len(faces)),
    )
    count, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, bounds)
    return sorted(groups, key=lambda g: int(g[0]))


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
    receipt = store.get("sheet_receipt", dict(schema=SCHEMA, pieces=[], fallback=[]))
    cut_base = int(ids.max()) + 1
    prepared = []
    for group in _edge_pieces(faces):
        store.check()
        component = int(rows[group[0], 0])
        group = np.sort(group)
        used, local = np.unique(faces[group], return_inverse=True)
        piece = dict(
            component=component,
            vertices=xyz[used],
            faces=local.reshape(-1, 3),
            face_ids=rows[group, 0],
            vertex_ids=ids[used],
        )
        base = cut_base
        # A cut adds at most two copies per split edge; reserve the worst case per piece.
        cut_base += 8 * len(used) + 16
        if component in done:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="sheet-seed-") as scratch:
                cut, topology = cut_component(piece["vertices"], piece["faces"], store.check)
                seed, initialization = _lscm_seed(cut, store.check, scratch)
            tri = seed[cut["faces"]]
            if not np.isfinite(seed).all() or np.any(
                cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) <= 0
            ):
                raise ValueError("no valid whole-sheet seed")
            piece.update(
                cut=cut, seed=seed, topology=topology, initialization=initialization, cut_base=base
            )
        except (ValueError, RuntimeError) as error:
            piece["error"] = str(error)
        prepared.append(piece)

    def commit(piece, uv, solver):
        cut = piece["cut"]
        tri = uv[cut["faces"]]
        signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if not np.isfinite(uv).all() or np.any(signed <= 0):
            raise ValueError("whole-sheet solve folded or is not finite")
        uv = _orient(cut["vertices"], uv)
        stats = _stats(cut["vertices"], cut["faces"].astype(np.int64), uv)
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
