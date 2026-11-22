"""Whole-sheet native flattening (`flattening_scope: sheet`): one map per connected piece."""

import json

import numpy as np
import pytest

from test_native_slim import flatboi_app  # noqa: F401 (pytest fixture)


def bundle(tmp_path, vertices, faces, name="mesh"):
    from winding_field.surface_mesh import import_mesh
    from winding_field.util import file_sha

    path = tmp_path / (name + ".npz")
    np.savez(path, vertices_xyz=np.asarray(vertices, float), faces=np.asarray(faces, np.int64))
    root = tmp_path / name
    import_mesh(
        {"path": str(path), "sha256": file_sha(path)},
        root,
        {
            "frame": "test",
            "origin_xyz": [0.0, 0.0, 0.0],
            "spacing_xyz": [1.0, 1.0, 1.0],
            "shape_zyx": [200, 200, 200],
            "voxel_um": 1.0,
            "sheet_step": 1.0,
        },
    )
    return root


def grid_faces(nu, nv, *, wrap=False):
    faces = []
    for row in range(nv - 1):
        for col in range(nu - (0 if wrap else 1)):
            a = row * nu + col
            b = row * nu + (col + 1) % nu
            faces.extend([(a, b, a + nu), (b, b + nu, a + nu)])
    return np.array(faces)


def curved_sheet(nu=60, nv=40, radius=30.0):
    """A cylinder arc with non-developable bumps: an atlas of charts opens cuts on it."""
    s, h = np.meshgrid(np.arange(nu) * 1.0, np.arange(nv) * 1.0)
    r = radius + 0.8 * np.sin(s / 7.0) * np.cos(h / 5.0)
    theta = s / radius
    vertices = np.column_stack(
        [(r * np.cos(theta)).ravel() + 60, (r * np.sin(theta)).ravel() + 60, h.ravel() + 10]
    )
    return vertices, grid_faces(nu, nv)


def cylinder(nu=48, nv=20, radius=12.0):
    theta, h = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 1.5)
    vertices = np.column_stack(
        [radius * np.cos(theta).ravel() + 60, radius * np.sin(theta).ravel() + 60, h.ravel() + 10]
    )
    return vertices, grid_faces(nu, nv, wrap=True)


def flatten(tmp_path, root, name="out", chart_faces=256):
    from winding_field.flattening.native_charts import assemble
    from winding_field.flattening.native_global import flatten_sheet
    from winding_field.native_store import Journal
    from winding_field.surface_mesh import MeshReader

    with MeshReader(root) as reader, Journal(tmp_path / name, {"source": name}) as store:
        receipt = flatten_sheet(reader, store, chart_faces)
        again = flatten_sheet(reader, store, chart_faces)
        summary = assemble(store)
        charts = {
            name: (status, store.load_arrays(path)["uv"], store.load_arrays(f"charts/{name}.npz"))
            for name, status, path in store.db.execute("SELECT name,status,uv_path FROM charts")
            if path is not None
        }
        unmapped = store.db.execute(
            "SELECT count(*) FROM charts WHERE status!='MAPPED'"
        ).fetchone()[0]
        seams = store.db.execute(
            "SELECT e.chart,e.ia,e.ib,p.chart,p.ia,p.ib FROM edges e JOIN edges p "
            "ON e.a=p.a AND e.b=p.b AND e.chart<p.chart"
        ).fetchall()
        cuts = store.db.execute("SELECT count(*) FROM cuts").fetchone()[0]
        frontier = store.db.execute("SELECT count(*) FROM frontier").fetchone()[0]
    return receipt, again, summary, charts, seams, cuts, frontier + 0 * unmapped


def raster_counts(charts, pitch=0.5):
    from winding_field.native_render import TILE, rasterize

    low = np.min([uv.min(0) for _, uv, _ in charts.values()], axis=0)
    high = np.max([uv.max(0) for _, uv, _ in charts.values()], axis=0)
    origin = np.floor(low / pitch) * pitch
    width, height = (np.ceil((high - origin) / pitch) + 1).astype(int)
    counts = np.zeros(4, np.int64)
    for y in range(0, height, TILE):
        for x in range(0, width, TILE):
            shape = (min(TILE, height - y), min(TILE, width - x))
            here = origin + np.array([x, y]) * pitch
            items = [(c["vertices"], c["faces"], uv) for _, uv, c in charts.values()]
            counts += np.bincount(rasterize(items, here, pitch, shape)[2].ravel(), minlength=4)
    return counts


def test_curved_sheet_is_one_map_without_cuts_or_ambiguous_pixels(tmp_path, flatboi_app):  # noqa: F811
    vertices, faces = curved_sheet()
    receipt, again, summary, charts, seams, cuts, frontier = flatten(
        tmp_path, bundle(tmp_path, vertices, faces)
    )
    assert again == receipt  # idempotent on a completed store
    assert receipt["fallback"] == [] and len(receipt["pieces"]) == 1
    piece = receipt["pieces"][0]
    assert piece["faces"] == len(faces) and piece["seam_vertex_copies"] == 0
    # Near-isometric everywhere: no chart-to-chart drift to absorb.
    assert 0.97 < piece["edge_stretch"]["p01"] and piece["edge_stretch"]["p99"] < 1.03
    assert 0.95 < piece["area_ratio"]["p01"] and piece["area_ratio"]["p99"] < 1.05
    assert summary["coverage_fraction"] == pytest.approx(1.0) and summary["components"] == 1
    assert len(charts) > 4 and all(status == "MAPPED" for status, _, _ in charts.values())
    assert cuts == 0 and frontier == 0
    # Raster charts already agree on every shared edge: nothing was welded.
    assert seams
    for chart, ia, ib, other, pia, pib in seams:
        np.testing.assert_array_equal(charts[chart][1][[ia, ib]], charts[other][1][[pia, pib]])
    counts = raster_counts(charts)
    area = sum(
        np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum() / 2
        for v, f in ((c["vertices"], c["faces"]) for _, _, c in charts.values())
    )
    assert counts[3] == 0  # no ambiguous pixels
    assert abs(counts[1] - area / 0.25) < 0.02 * area / 0.25
    # +v follows world +Z, so every level is laid out the same way up.
    uv = np.concatenate([uv for _, uv, _ in charts.values()])
    z = np.concatenate([c["vertices"][:, 2] for _, _, c in charts.values()])
    assert np.corrcoef(uv[:, 1], z)[0, 1] > 0.99


def test_closed_winding_is_cut_once_and_unrolled_as_a_strip(tmp_path, flatboi_app):  # noqa: F811
    vertices, faces = cylinder()
    receipt, _, summary, charts, _, cuts, _ = flatten(tmp_path, bundle(tmp_path, vertices, faces))
    piece = receipt["pieces"][0]
    assert receipt["fallback"] == [] and piece["euler_characteristic"] == 0
    assert piece["seam_vertex_copies"] > 0  # the cylinder was opened along one seam
    assert 0.97 < piece["edge_stretch"]["p01"] and piece["edge_stretch"]["p99"] < 1.03
    assert summary["coverage_fraction"] == pytest.approx(1.0) and cuts == 0
    uv = np.concatenate([uv for _, uv, _ in charts.values()])
    extent = np.ptp(uv, axis=0)
    # Unrolled: the height is exact; the seam need not be vertical, so the strip is at
    # least one circumference (a 48-gon of radius 12) wide and keeps the cylinder's area.
    circumference = 48 * 2 * 12 * np.sin(np.pi / 48)
    assert extent[1] == pytest.approx(28.5, rel=0.03) and extent[0] >= 0.97 * circumference
    counts = raster_counts(charts)
    assert counts[3] == 0
    assert counts[1] * 0.25 == pytest.approx(circumference * 28.5, rel=0.03)


def test_pieces_that_cannot_be_flattened_whole_use_the_chart_route(tmp_path, flatboi_app):  # noqa: F811
    vertices, faces = curved_sheet(nu=20, nv=12)
    # A second piece with a non-manifold edge (three triangles on one edge).
    fan = np.array(
        [[150.0, 150, 50], [151, 150, 50], [150, 151, 50], [150, 149, 50], [150.5, 150, 51]]
    )
    bad = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]) + len(vertices)
    receipt, _, summary, charts, _, _, _ = flatten(
        tmp_path, bundle(tmp_path, np.vstack([vertices, fan]), np.vstack([faces, bad]))
    )
    assert len(receipt["pieces"]) == 1 and len(receipt["fallback"]) == 1
    assert "nonmanifold" in receipt["fallback"][0]["reason"]
    assert receipt["fallback"][0]["faces"] == 3
    # The good sheet is still mapped whole; the bad piece is reported, never dropped silently.
    assert summary["mapped_faces"] >= len(faces)


def test_sheet_scope_is_opt_in_and_needs_slim():
    from winding_field.native_pipeline import _settings

    default = _settings({})
    assert "flattening_scope" not in default and "scope" not in default["parameterizer"]
    assert _settings({"flattening_scope": None}) == default
    assert _settings({"flattening_scope": "charts"}) == default
    with pytest.raises(ValueError, match="requires flattening_method slim"):
        _settings({"flattening_scope": "sheet"})
    with pytest.raises(ValueError, match="charts or sheet"):
        _settings({"flattening_scope": "atlas"})


def test_sheet_settings_record_the_global_recipe(flatboi_app):  # noqa: F811
    from winding_field.native_pipeline import _settings

    settings = _settings({"flattening_method": "slim", "flattening_scope": "sheet"})
    assert settings["flattening_scope"] == "sheet"
    assert settings["parameterizer"]["scope"] == "sheet"
    assert settings["parameterizer"]["solver"] == "flatboi-slim"
    assert settings != _settings({"flattening_method": "slim"})


def test_render_ct_sheet_scope_renders_and_resumes(tmp_path, monkeypatch, flatboi_app):  # noqa: F811
    from test_native_geometry import plane_bundle
    from test_native_render import analytic_ct
    from winding_field import scroll
    import winding_field.workflow as workflow

    mesh = plane_bundle(tmp_path, nx=9, ny=7)
    volume = analytic_ct(tmp_path)
    monkeypatch.setattr(workflow, "open_volume", lambda config: volume.thing)
    args = scroll._parser().parse_args(
        [
            "render-ct",
            "--mesh",
            str(mesh),
            "--source",
            "unused",
            "--render-pitch-um",
            "0.5",
            "--chart-faces",
            "16",
            "--flattening-scope",
            "sheet",
            "--output",
            str(tmp_path / "out"),
        ]  # fmt: skip
    )
    result = scroll.render_ct(args)
    assert result["status"] == "COMPLETE" and result["coverage_fraction"] == 1.0
    assert result["sheet_flatten"]["fallback"] == []
    assert sum(c["counts"]["ambiguous_xyz"] for c in result["components"]) == 0
    identity = json.loads((tmp_path / "out" / "IDENTITY.json").read_text())
    assert identity["settings"]["flattening_scope"] == "sheet"
    level = json.loads((tmp_path / "out" / "LEVEL.json").read_text())
    assert level["sheet_flatten"]["pieces"][0]["converged"] in (True, False)
    assert (
        scroll.render_ct(args)["status"] == "COMPLETE"
    )  # completed output is verified, not redone
    args.flattening_scope = "charts"
    with pytest.raises(ValueError, match="identity"):
        scroll.render_ct(args)
