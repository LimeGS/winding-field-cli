"""Handles of a sheet piece are cut out along short loops and the piece stays whole.

A piece with tunnels through it used to be cut to a disk through every tunnel, and its
whole-sheet map collapsed. These tests build a sheet with holes and bent tubes (one handle
each) and check the loops, the faces set aside and the receipt of the whole-sheet route.
"""

import numpy as np
import pytest

from test_native_sheet_rows import run, seeded_solve  # noqa: F401
from test_native_sheet_scope import bundle


def tunnel_sheet(n=60, holes=6, arches=3, hole=2, seed=0, long=1.0):
    """Grid sheet with square holes and ``arches`` tubes that each join two of its holes."""
    rng = np.random.default_rng(seed)
    nx, ny = int(n * long), n
    idx = np.arange(nx * ny).reshape(nx, ny)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    v = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)]).astype(float)
    alive = np.ones((nx - 1, ny - 1), bool)
    taken = np.zeros((nx, ny), bool)
    taken[:4] = taken[-4:] = True
    taken[:, :4] = taken[:, -4:] = True

    def place(w, h):
        for _ in range(10000):
            i, j = int(rng.integers(4, nx - w - 4)), int(rng.integers(4, ny - h - 4))
            if not taken[i - 3 : i + w + 4, j - 3 : j + h + 4].any():
                taken[i - 3 : i + w + 4, j - 3 : j + h + 4] = True
                return i, j
        raise RuntimeError("no room")

    def rim(i, j, k):
        a = [(i + t, j) for t in range(k)] + [(i + k, j + t) for t in range(k)]
        return a + [(i + k - t, j + k) for t in range(k)] + [(i, j + k - t) for t in range(k)]

    more_v, more_f = [], []
    for _ in range(arches):
        gap, rings = 4, 6
        i, j = place(2 * hole + gap, hole)
        alive[i : i + hole, j : j + hole] = False
        i2 = i + hole + gap
        alive[i2 : i2 + hole, j : j + hole] = False
        ra = rim(i, j, hole)
        rb = [(i2 + hole - (p - i), q) for p, q in ra]  # mirrored, so the tube is orientable
        prev = [idx[p] for p in ra]
        for r in range(1, rings + 1):
            t = r / rings
            if r == rings:
                cur = [idx[p] for p in rb]
            else:
                # Half a torus: each rim point turns about the axis midway between the holes.
                start = np.array(ra, float)
                middle = i + hole + gap / 2
                reach = middle - start[:, 0]
                first = nx * ny + len(more_v)
                cur = list(range(first, first + len(ra)))
                more_v.extend(
                    np.column_stack(
                        [
                            middle - reach * np.cos(np.pi * t),
                            start[:, 1],
                            reach * np.sin(np.pi * t),
                        ]
                    ).tolist()
                )
            for a in range(len(ra)):
                b = (a + 1) % len(ra)
                more_f += [[prev[a], prev[b], cur[a]], [prev[b], cur[b], cur[a]]]
            prev = cur
    for _ in range(holes):
        i, j = place(hole, hole)
        alive[i : i + hole, j : j + hole] = False
    ii, jj = np.nonzero(alive)
    a, b, c, d = idx[ii, jj], idx[ii + 1, jj], idx[ii + 1, jj + 1], idx[ii, jj + 1]
    f = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    if more_f:
        v = np.concatenate([v, np.asarray(more_v)])
        f = np.concatenate([f, np.asarray(more_f, np.int64)])
    used, inverse = np.unique(f, return_inverse=True)
    return v[used], inverse.reshape(-1, 3).astype(np.int64)


@pytest.fixture
def relaxed_solve(monkeypatch):
    """Flatboi's place is taken by libigl's own SLIM, so the stretch gates see a real map."""
    import igl

    from winding_field.flattening import flatboi, patches

    def solve(items, app_run, **kw):
        maps = []
        for vertices, faces, seed in items:
            faces = np.asarray(faces).astype(np.int32)
            data = igl.slim_precompute(
                np.asfortranarray(vertices, dtype=float),
                np.asfortranarray(faces),
                np.asfortranarray(seed, dtype=float),
                igl.SYMMETRIC_DIRICHLET,
                np.zeros(0, np.int32),
                np.zeros((0, 2), order="F"),
                0.0,
            )
            maps.append(np.ascontiguousarray(igl.slim_solve(data, 30)[:, :2]))
        return maps, dict(steps=30, converged=True)

    monkeypatch.setattr(flatboi, "resolve_app_run", lambda app_run, offline=False: "libigl")
    monkeypatch.setattr(patches, "check_flatboi", lambda app_run, mode: {"app_run": app_run})
    monkeypatch.setattr(flatboi, "smoke_test", lambda app_run: None)
    monkeypatch.setattr(flatboi, "solve_batch", solve)


def genus(faces):
    import igl

    rims = len(igl.boundary_loop_all(faces))
    return (2 - int(igl.euler_characteristic(faces)) - rims) // 2, rims


def test_edges_on_a_cycle_are_told_from_bridges():
    from winding_field.flattening.handles import _on_cycle

    # A triangle 0-1-2, a tail 2-3-4, a doubled edge 4-5 and a self loop at 5.
    a = np.array([0, 1, 2, 2, 3, 4, 4, 5])
    b = np.array([1, 2, 0, 3, 4, 5, 5, 5])
    found = _on_cycle(7, a, b, np.random.default_rng(0))
    assert found.tolist() == [True, True, True, False, False, True, True, False]


def test_each_handle_is_cut_along_a_short_loop_and_the_piece_stays_whole():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from winding_field.flattening.handles import cut_handles

    vertices, faces = tunnel_sheet()
    assert genus(faces) == (3, 7)
    v, f, parent, report = cut_handles(vertices, faces)
    assert report["genus"] == 3 and report["loops"] == 3 and report["genus_left"] == 0
    assert np.array_equal(parent[f], faces) and np.array_equal(v, vertices[parent])
    # A loop adds two rims, or one when it runs through a hole.
    left, rims = genus(f)
    assert left == 0 and 7 + 3 <= rims <= 7 + 2 * 3
    graph = coo_matrix((np.ones(f.size), (f.ravel(), np.roll(f, -1, axis=1).ravel())))
    assert connected_components(graph, directed=False)[0] == 1
    # A tube of 2 x 2 cells is 8 edges around; no loop runs through a tube and back.
    assert max(report["loop_length"]) < 12


def test_sites_are_found_when_the_first_vertex_is_shut_in():
    # On PHerc0175A the first vertex of the main piece had links only to the closed outline:
    # the single root reached nothing and no handle was found at any radius.
    from winding_field.flattening.handles import _Surface

    vertices, faces = tunnel_sheet()
    surface = _Surface(vertices, faces, np.inf)
    start = int(surface.node[np.flatnonzero(~surface.blocked[surface.node])[0]])
    assert len(surface.sites()) > 0
    row = slice(surface.graph.indptr[start], surface.graph.indptr[start + 1])
    for x in surface.graph.indices[row].tolist():
        surface.block(x)
    assert len(surface.sites()) > 0


def strap_sheet(n=24):
    """Grid sheet with two holes and a one-face-wide strap that arches from one to the other."""
    idx = np.arange(n * n).reshape(n, n)
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)]).astype(float)
    alive = np.ones((n - 1, n - 1), bool)
    alive[6:8, 10:12] = alive[14:16, 10:12] = False
    ii, jj = np.nonzero(alive)
    a, b, c, d = idx[ii, jj], idx[ii + 1, jj], idx[ii + 1, jj + 1], idx[ii, jj + 1]
    f = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    # From the rim edge (8,10)-(8,11) of one hole over the sheet to (14,10)-(14,11) of the other.
    rungs = [(idx[8, 10], idx[8, 11])]
    for x in np.linspace(8, 14, 8)[1:-1]:
        rungs.append((len(v), len(v) + 1))
        z = 2.0 * np.sin(np.pi * (x - 8) / 6)
        v = np.concatenate([v, [[x, 10.0, z], [x, 11.0, z]]])
    rungs.append((idx[14, 10], idx[14, 11]))
    strap = []
    for (p0, p1), (q0, q1) in zip(rungs[:-1], rungs[1:]):
        strap += [[p0, p1, q1], [p0, q1, q0]]
    used, inverse = np.unique(np.concatenate([f, np.asarray(strap, np.int64)]), return_inverse=True)
    return v[used], inverse.reshape(-1, 3).astype(np.int64)


def test_a_handle_that_runs_from_hole_to_hole_gets_its_collar():
    # A strap between two holes is a handle whose shortest loop has both ends on a rim.
    from winding_field.flattening.handles import cut_handles, handle_collars

    vertices, faces = strap_sheet()
    assert genus(faces) == (1, 2)
    _, _, _, report = cut_handles(vertices, faces)
    assert report["loops"] == 1 and report["genus_left"] == 0
    assert len(report["loop_vertices"][0]) >= 2
    aside, _ = handle_collars(vertices, faces)
    assert 0 < aside.sum() < 12
    left = faces[~aside]
    assert genus(np.unique(left, return_inverse=True)[1].reshape(-1, 3))[0] == 0


def test_a_piece_without_handles_is_left_alone():
    from winding_field.flattening.handles import cut_handles, handle_collars

    vertices, faces = tunnel_sheet(arches=0)
    v, f, parent, report = cut_handles(vertices, faces)
    assert report["genus"] == 0 and report["loops"] == 0 and f is not None
    assert np.array_equal(f, faces) and np.array_equal(parent, np.arange(len(vertices)))
    aside, _ = handle_collars(vertices, faces)
    assert not aside.any()


def test_a_piece_with_no_inner_edge_has_no_handles(tmp_path, relaxed_solve):
    from winding_field.flattening.handles import handle_collars

    # One triangle, and two that share an edge whose ends are both on the rim: a level has
    # thousands of such slivers next to its sheet.
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    for faces in ([[0, 1, 2]], [[0, 1, 2], [1, 3, 2]]):
        aside, report = handle_collars(points[: np.max(faces) + 1], np.asarray(faces))
        assert not aside.any() and report["genus"] == 0 and report["loops"] == 0
    vertices, faces = tunnel_sheet()
    sliver = np.array([[0.0, 0, 50], [1, 0, 50], [0, 1, 50]])
    both = np.concatenate([faces, [[len(vertices), len(vertices) + 1, len(vertices) + 2]]])
    root = bundle(tmp_path, np.concatenate([vertices, sliver]), both)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    assert sum(c[3] for c in charts) == len(both)
    assert max(p["handles"]["loops"] for p in receipt["pieces"] if p["handles"]) == 3


def test_the_faces_around_each_tunnel_leave_a_piece_with_only_holes():
    from winding_field.flattening.handles import handle_collars
    from winding_field.flattening.native_global import _part

    vertices, faces = tunnel_sheet()
    aside, report = handle_collars(vertices, faces)
    assert report["loops"] == 3 and 0 < aside.sum() < 0.1 * len(faces)
    _, kept, _, rows = _part(vertices, faces, np.flatnonzero(~aside))
    assert genus(kept)[0] == 0
    assert len(rows) > 0.9 * len(faces)


def test_a_sheet_with_handles_is_flattened_whole_and_the_tunnels_apart(tmp_path, relaxed_solve):
    vertices, faces = tunnel_sheet()
    root = bundle(tmp_path, vertices, faces)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    assert receipt["fallback"] == []
    (piece,) = receipt["pieces"]
    assert piece["handles"]["loops"] == 3 and piece["handles"]["genus_left"] == 0
    assert piece["aside_faces"] > 0 and piece["faces"] + piece["aside_faces"] == len(faces)
    assert piece["holes"] >= 6
    # Every face is in exactly one chart; the tunnels are components of their own.
    assert sum(c[3] for c in charts) == len(faces)
    components = {c[1] for c in charts}
    assert piece["component"] in components and len(components) > 1
    sheet_faces = sum(c[3] for c in charts if c[1] == piece["component"])
    assert sheet_faces == piece["faces"]


def test_a_folded_first_seed_is_replaced_by_the_capped_mean_value_map(monkeypatch):
    from winding_field.flattening import native_global as sheet

    vertices, faces = tunnel_sheet()
    monkeypatch.setattr(sheet, "_planar_seed", lambda cut, drop=False: None)
    monkeypatch.setattr(sheet, "_unrolled_seed", lambda cut: None)
    made = sheet._prepare(vertices, faces, True, sheet._unchecked)
    assert "error" not in made and made["minimal"]
    assert made["initialization"]["capped_holes"] == made["topology"]["holes"] >= 6
    cut, seed = made["cut"], made["seed"]
    assert len(seed) == len(cut["vertices"])
    corners = seed[cut["faces"]]
    edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    assert (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0] > 0).all()


def rolled(vertices, turn=300.0, spiral=0.1):
    """Bend a flat sheet into ``turn`` degrees of a spiral around the y axis."""
    x = vertices[:, 0] - vertices[:, 0].min()
    angle = np.radians(turn) * x / x.max()
    radius = x.max() / np.radians(turn) * (1 + spiral * angle) + vertices[:, 2]
    return np.column_stack([radius * np.cos(angle), vertices[:, 1], radius * np.sin(angle)])


def closed_wrap(**sheet):
    """A sheet rolled into a full turn, its last column of vertices joined to the first."""
    flat, faces = tunnel_sheet(long=2.0, **sheet)
    last = flat[:, 0].max()
    on_grid = flat[:, 2] == 0
    first = np.flatnonzero(on_grid & (flat[:, 0] == 0))
    final = np.flatnonzero(on_grid & (flat[:, 0] == last))
    first, final = first[np.argsort(flat[first, 1])], final[np.argsort(flat[final, 1])]
    a, b, c, d = final[:-1], first[:-1], first[1:], final[1:]
    joint = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    vertices = rolled(flat, turn=360 * last / (last + 1), spiral=0.0)
    return vertices, np.concatenate([faces, joint])


def flat_domain(vertices, faces):
    import igl

    return dict(
        vertices=vertices,
        faces=faces,
        parent=np.arange(len(vertices)),
        boundary=igl.boundary_loop(faces).astype(np.int64),
    )


def test_a_wrap_is_unrolled_around_its_axis_without_drift():
    from winding_field.flattening import native_global as sheet

    flat, faces = tunnel_sheet(arches=0, long=2.0)
    cut = flat_domain(rolled(flat), faces)
    uv, initialization, rows = sheet._unrolled_seed(cut)
    assert initialization["method"] == "unroll" and rows is None
    assert abs(initialization["angle_degrees"] - 300) < 3
    assert abs(abs(initialization["axis"][1]) - 1) < 1e-6
    stretch = sheet._stats(cut["vertices"], faces, uv)["edge_stretch"]
    assert stretch["p01"] > 0.95 and stretch["p99"] < 1.05


def test_a_closed_wrap_and_a_flat_sheet_are_not_unrolled():
    from winding_field.flattening import native_global as sheet

    flat, faces = tunnel_sheet(arches=0)
    assert sheet._unrolled_seed(flat_domain(flat, faces)) is None
    # A tube: the last column of a grid joined to the first.
    n, m = 40, 12
    index = np.arange(n * m).reshape(n, m)
    a, b = index[:, :-1].ravel(), np.roll(index, -1, axis=0)[:, :-1].ravel()
    tube = np.concatenate([np.column_stack([a, b, b + 1]), np.column_stack([a, b + 1, a + 1])])
    angle = 2 * np.pi * (np.arange(n * m) // m) / n
    points = np.column_stack([5 * np.cos(angle), np.arange(n * m) % m, 5 * np.sin(angle)])
    assert sheet._unrolled_seed(flat_domain(points, tube)) is None


def test_a_folded_first_seed_gives_way_to_the_unrolled_one(monkeypatch):
    from winding_field.flattening import native_global as sheet

    flat, faces = tunnel_sheet(long=2.0)
    vertices = rolled(flat)
    made = sheet._prepare(vertices, faces, True, sheet._unchecked)
    assert made["initialization"]["method"] == "LSCM"
    monkeypatch.setattr(sheet, "_planar_seed", lambda cut, drop=False: None)
    made = sheet._prepare(vertices, faces, True, sheet._unchecked)
    assert "error" not in made and made["minimal"]
    assert made["initialization"]["method"] == "unroll"
    assert made["handles"]["loops"] == 3 and made["topology"]["holes"] >= 6
    # Asked for alone, the unrolled map is all there is: a flat sheet has none.
    again = sheet._prepare(vertices, faces, True, sheet._unchecked, True)
    assert again["initialization"]["method"] == "unroll"
    assert "error" in sheet._prepare(*tunnel_sheet(arches=0), True, sheet._unchecked, True)


def test_a_stretched_solve_is_run_again_from_the_unrolled_map(tmp_path, relaxed_solve, monkeypatch):
    from winding_field.flattening import flatboi

    solve, calls = flatboi.solve_batch, []

    def drifted(items, app_run, **kw):
        calls.append(len(items))
        if len(calls) == 1:
            # What a solve that ran out of steps leaves: a valid map, far too stretched.
            return [seed * [4.0, 0.25] for _, _, seed in items], dict(steps=30, converged=False)
        return solve(items, app_run, **kw)

    monkeypatch.setattr(flatboi, "solve_batch", drifted)
    flat, faces = tunnel_sheet(long=2.0)
    root = bundle(tmp_path, rolled(flat), faces)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    assert len(calls) >= 2 and receipt["fallback"] == []
    (piece,) = receipt["pieces"]
    assert piece["initialization"] == "unroll" and piece["handles"]["loops"] == 3
    assert sum(c[3] for c in charts) == len(faces)


def test_a_rolled_sheet_with_handles_is_flattened_whole(tmp_path, relaxed_solve):
    flat, faces = tunnel_sheet(long=2.0)
    root = bundle(tmp_path, rolled(flat), faces)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    assert receipt["fallback"] == []
    (piece,) = receipt["pieces"]
    assert piece["handles"]["loops"] == 3 and piece["handles"]["genus_left"] == 0
    assert piece["faces"] + piece["aside_faces"] == len(faces)
    assert sum(c[3] for c in charts) == len(faces)


def test_the_ends_of_a_wrap_get_a_seam_and_its_holes_stay(monkeypatch):
    from winding_field.flattening import native_global as sheet

    vertices, faces = closed_wrap(arches=0)
    assert genus(faces) == (0, 8)
    cut, topology = sheet._minimal_cut(vertices, faces, sheet._unchecked, True)
    assert topology["holes"] == 6 and 0 < topology["seam_duplicates"] < 70
    assert sheet._unrolled_seed(dict(cut, vertices=vertices, faces=faces)) is None
    uv, initialization, rows = sheet._unrolled_seed(cut)
    assert rows is None and abs(initialization["angle_degrees"] - 360) < 1
    stretch = sheet._stats(cut["vertices"], cut["faces"].astype(np.int64), uv)["edge_stretch"]
    assert stretch["p01"] > 0.95 and stretch["p99"] < 1.05
    # Kept closed, the wrap is a ring: no seam, and one end counts as a hole.
    monkeypatch.setattr(sheet, "LONG_RIM", 2.0)
    _, ring = sheet._minimal_cut(vertices, faces, sheet._unchecked, True)
    assert ring["holes"] == 7 and ring["seam_duplicates"] == 0


def test_a_closed_wrap_with_handles_is_flattened_whole(tmp_path, relaxed_solve):
    vertices, faces = closed_wrap()
    root = bundle(tmp_path, vertices, faces)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    assert receipt["fallback"] == []
    (piece,) = receipt["pieces"]
    assert piece["handles"]["loops"] == 3 and piece["handles"]["genus_left"] == 0
    assert piece["holes"] >= 6 and piece["seam_vertex_copies"] > 0
    assert piece["faces"] + piece["aside_faces"] == len(faces)
    assert sum(c[3] for c in charts) == len(faces)


def test_a_sheet_without_handles_gives_the_same_charts_as_before(tmp_path, seeded_solve):  # noqa: F811
    vertices, faces = tunnel_sheet(arches=0)
    root = bundle(tmp_path, vertices, faces)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    (piece,) = receipt["pieces"]
    assert piece["aside_faces"] == 0 and piece["handles"] is None and piece["holes"] == 6
    assert {c[1] for c in charts} == {piece["component"]}
