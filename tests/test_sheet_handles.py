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
    made = sheet._prepare(vertices, faces, True, sheet._unchecked)
    assert "error" not in made and made["minimal"]
    assert made["initialization"]["capped_holes"] == made["topology"]["holes"] >= 6
    cut, seed = made["cut"], made["seed"]
    assert len(seed) == len(cut["vertices"])
    corners = seed[cut["faces"]]
    edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    assert (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0] > 0).all()


def test_a_sheet_without_handles_gives_the_same_charts_as_before(tmp_path, seeded_solve):  # noqa: F811
    vertices, faces = tunnel_sheet(arches=0)
    root = bundle(tmp_path, vertices, faces)
    receipt, charts, _, _ = run(tmp_path, root, "out")
    (piece,) = receipt["pieces"]
    assert piece["aside_faces"] == 0 and piece["handles"] is None and piece["holes"] == 6
    assert {c[1] for c in charts} == {piece["component"]}
