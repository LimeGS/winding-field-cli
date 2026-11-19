"""Original CT sampling, missingness and physical controls, not model predictions."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from ct_fixture import write_ct
from test_flattening_benchmark import source_mesh


@pytest.fixture
def ct_tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    assert importlib.util.find_spec("flattening_ct"), "CT benchmark not implemented"
    return importlib.import_module("flattening_ct")


def profile(path, unit=1.0, origin=(0.0, 0.0, 0.0), field="175a", shape=(32, 32, 32)):
    return dict(
        field_id=field,
        frame=field,
        source=str(path),
        level=0,
        spacing_xyz_um=[unit] * 3,
        origin_xyz_um=list(origin),
        shape_zyx=list(shape),
        pitch_um=unit,
        overview_pitch_um=unit * 2,
        offsets_um=[-unit, 0.0, unit],
    )


def config(tmp_path, p):
    return dict(
        profile=p,
        cache_dir=str(tmp_path / "cache"),
        cache_bytes=1 << 24,
        decoded_cache_bytes=1 << 20,
        reserve_bytes=0,
    )


@pytest.mark.parametrize("unit,field", [(8.64, "175a"), (2.4, "paris4")])
def test_analytic_ct_with_nonzero_origin_and_different_spacing(ct_tool, tmp_path, unit, field):
    z, y, x = np.indices((32, 32, 32))
    source = write_ct(
        tmp_path / "ct", (x + 2 * y + 3 * z).astype("u1"), spacing=[unit] * 3, origin=[10, 20, 30]
    )
    p = profile(source, unit, [10, 20, 30], field)
    with ct_tool.CheckedCT(config(tmp_path, p), tmp_path / "ledger") as ct:
        values, valid = ct.sample(np.array([[2.25, 3.5, 4.75]]) * unit + [10, 20, 30])
    assert valid.tolist() == [True]
    np.testing.assert_allclose(values, [23.5], atol=1e-4)


def test_cross_field_ct_binding_is_rejected(ct_tool):
    with pytest.raises(ValueError, match="binding|field"):
        ct_tool.validate_profile(
            dict(field_id="175a", frame="175a"), profile("/unused", field="paris4")
        )


def test_missing_positive_weight_corner_is_unsupported(ct_tool, tmp_path):
    source = write_ct(tmp_path / "ct", np.zeros((8, 8, 8), "u1"), chunk=4)
    (source / "0/0.0.1").unlink()
    p = profile(source, shape=(8, 8, 8))
    with ct_tool.CheckedCT(config(tmp_path, p), tmp_path / "ledger") as ct:
        values, valid = ct.sample(np.array([[3.5, 1.0, 1.0], [3.0, 1.0, 1.0]]))
    assert valid.tolist() == [False, True]
    assert np.isnan(values[0]) and values[1] == 0.0


def test_zero_intensity_is_not_missing(ct_tool, tmp_path):
    source = write_ct(tmp_path / "ct", np.zeros((32, 32, 32), "u1"))
    with ct_tool.CheckedCT(config(tmp_path, profile(source)), tmp_path / "ledger") as ct:
        values, valid = ct.sample(np.array([[2.0, 3.0, 4.0]]))
    assert valid.all() and values[0] == 0.0


def test_seam_duplicates_share_source_normals(ct_tool):
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    normals = ct_tool.original_normals(v, f)
    points = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    actual = ct_tool.normals_at(v, f, normals, points, np.array([0, 1]))
    np.testing.assert_allclose(actual, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def test_same_source_point_same_depth_same_ct_across_methods(ct_tool, tmp_path):
    z, y, x = np.indices((32, 32, 32))
    source = write_ct(tmp_path / "ct", (x + 2 * y + 3 * z).astype("u1"))
    point = np.array([[5.25, 6.5, 7.0]])
    with ct_tool.CheckedCT(config(tmp_path, profile(source)), tmp_path / "ledger") as ct:
        a, ok = ct.sample(point + [0.0, 0.0, 1.0])
        b, other = ct.sample(point + [0.0, 0.0, 1.0])
    assert ok.all() and other.all()
    np.testing.assert_allclose(a, [42.25], atol=1e-4)
    np.testing.assert_array_equal(a, b)


def test_tiles_and_roi_fragments_do_not_double_count_area(ct_tool):
    bary = ct_tool.quadrature_barycentric(1)
    assert bary.shape == (4, 3)
    np.testing.assert_allclose(bary.sum(1), 1.0)
    np.testing.assert_allclose(bary.mean(0), [1 / 3] * 3)
    assert ct_tool.quadrature_barycentric(2).shape == (16, 3)


def test_same_depth_and_window_for_all_candidates(ct_tool, tmp_path):
    import bench_flattening as b

    v = np.array([[4.0, 4.0, 3.0], [8.0, 4.0, 3.0], [8.0, 8.0, 3.0], [4.0, 8.0, 3.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    src, _, _ = source_mesh(tmp_path / "src", unit=1.0, geometry=(v, f))
    z, y, x = np.indices((32, 32, 32))
    source = write_ct(tmp_path / "ct", (x + 2 * y + 3 * z).astype("u1"))
    src["ct_profile"] = profile(source)
    case = b.load_case(src, tmp_path / "case")
    mapped = dict(
        world=v,
        faces=f,
        uv=v[:, :2].copy(),
        parent=np.arange(4),
        source_face_rows=np.arange(2),
        domain_ids=np.zeros(2, int),
    )
    rois = ct_tool.prepare_rois(case, tmp_path / "rois")
    report = ct_tool.render_case(
        case, mapped, rois, tmp_path / "render", config(tmp_path, src["ct_profile"]), lambda: None
    )
    assert report["window"] == [0, 255]
    assert report["offsets_um"] == [-1.0, 0.0, 1.0]
    assert report["CT_observed_area_fraction_estimate"] == 1.0
    tile = next(r for r in report["regions"] if r["id"].startswith("roi-"))["tiles"][0]
    with np.load(tmp_path / "render" / tile["raw"], allow_pickle=False) as data:
        valid = data["support"][1]
        xyz = data["xyz_um"][valid]
        np.testing.assert_allclose(
            data["ct"][1][valid], xyz[:, 0] + 2 * xyz[:, 1] + 3 * xyz[:, 2], atol=1e-4
        )
    sha = b.file_sha(tmp_path / "render" / "CT.json")
    ct_tool.render_case(
        case, mapped, rois, tmp_path / "render", config(tmp_path, src["ct_profile"]), lambda: None
    )
    assert b.file_sha(tmp_path / "render" / "CT.json") == sha


def test_ambiguous_surface_is_not_recovered_ct_area(ct_tool, tmp_path):
    import bench_flattening as b

    v = np.array(
        [
            [4.0, 4.0, 3.0],
            [8.0, 4.0, 3.0],
            [8.0, 8.0, 3.0],
            [12.0, 4.0, 3.0],
            [16.0, 4.0, 3.0],
            [16.0, 8.0, 3.0],
        ]
    )
    f = np.array([[0, 1, 2], [3, 4, 5]])
    src, _, _ = source_mesh(tmp_path / "src", unit=1.0, geometry=(v, f))
    source = write_ct(tmp_path / "ct", np.full((32, 32, 32), 90, "u1"))
    src["ct_profile"] = profile(source)
    case = b.load_case(src, tmp_path / "case")
    mapped = dict(
        world=v,
        faces=f,
        uv=np.r_[v[:3, :2], v[:3, :2]],
        parent=np.arange(6),
        source_face_rows=np.arange(2),
        domain_ids=np.arange(2),
    )
    rois = ct_tool.prepare_rois(case, tmp_path / "rois")
    r = ct_tool.render_case(
        case, mapped, rois, tmp_path / "render", config(tmp_path, src["ct_profile"]), lambda: None
    )
    assert r["CT_observed_area_fraction_estimate"] == 0.0


def test_completed_ct_rejects_changed_interpretation_metadata(ct_tool, tmp_path):
    import bench_flattening as b
    import json
    from test_flattening_benchmark import cell_request

    req = cell_request(b, tmp_path)
    b.run_cell(req, tmp_path / "cell")
    path = Path(req["ct_config"]["profile"]["source"]) / "0/.zarray"
    metadata = json.loads(path.read_text())
    metadata["order"] = "F"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata|identity"):
        b.run_cell(req, tmp_path / "cell")


def test_closed_surface_still_gets_six_distinct_source_rois(ct_tool, tmp_path):
    import bench_flattening as b

    vertices = 1000.0 * np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.array(
        [[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0], [5, 1, 0], [5, 2, 1], [5, 3, 2], [5, 0, 3]]
    )
    source, _, _ = source_mesh(tmp_path / "src", unit=1.0, geometry=(vertices, faces))
    case = b.load_case(source, tmp_path / "case")
    rois = ct_tool.prepare_rois(case, tmp_path / "rois")
    assert len({r["center_face_row"] for r in rois["regions"]}) == 6
    assert not rois["insufficient_distinct_regions"]
    assert rois["original_boundary_absent"]


def reference_fixture(ct_tool, tmp_path, unit=8.64, field="175a"):
    import bench_flattening as b

    origin = np.array([10.0, 20.0, 30.0])
    v = (
        np.array([[10.0, 10.0, 16.0], [14.0, 10.0, 16.0], [14.0, 14.0, 16.0], [10.0, 14.0, 16.0]])
        * unit
        + origin
    )
    f = np.array([[0, 1, 2], [0, 2, 3]])
    src, _, _ = source_mesh(tmp_path / "src", field, 1.0, geometry=(v, f))
    z, y, x = np.indices((32, 32, 32))
    source = write_ct(
        tmp_path / "ct", (x + 2 * y + 3 * z).astype("u1"), spacing=[unit] * 3, origin=origin
    )
    src["ct_profile"] = profile(source, unit, origin, field)
    case = b.load_case(src, tmp_path / "case")
    rois = ct_tool.prepare_rois(case, tmp_path / "rois")
    return case, rois, config(tmp_path, src["ct_profile"]), v, f, origin


@pytest.mark.parametrize("unit,field", [(8.64, "175a"), (2.4, "paris4")])
def test_reference_views_keep_native_physical_frame(ct_tool, tmp_path, unit, field):
    import bench_flattening as b

    case, rois, cfg, v, f, origin = reference_fixture(ct_tool, tmp_path, unit, field)
    output = tmp_path / "references"
    report = ct_tool.render_references(case, rois, output, cfg, lambda: None)
    assert report["diagnostic_only"] and not report["physical_acceptance"]
    region = report["regions"][0]
    assert region["plane_names"] == ["tangent", "xy", "xz", "yz"]
    assert region["pitch_um"] == unit and region["offset_um"] == 0.0
    center = v[f[rois["regions"][0]["center_face_row"]]].mean(0)
    with np.load(output / region["raw"], allow_pickle=False) as data:
        xyz, raw, support = data["xyz_um"], data["ct"], data["support"]
        for plane in range(4):
            np.testing.assert_allclose(
                xyz[plane, 127:129, 127:129].mean((0, 1)), center, atol=1e-10
            )
            np.testing.assert_allclose(
                np.linalg.norm(xyz[plane, :, 1:] - xyz[plane, :, :-1], axis=-1), unit
            )
            grid = (xyz[plane][support[plane]] - origin) / unit
            assert len(grid)
            np.testing.assert_allclose(
                raw[plane][support[plane]], grid @ [1.0, 2.0, 3.0], atol=1e-4
            )
        np.testing.assert_allclose(xyz[1, :, :, 2], center[2])
        np.testing.assert_allclose(xyz[2, :, :, 1], center[1])
        np.testing.assert_allclose(xyz[3, :, :, 0], center[0])
    before = b.file_sha(output / "REFERENCES.json")
    ct_tool.render_references(case, rois, output, cfg, lambda: None)
    assert b.file_sha(output / "REFERENCES.json") == before


def test_undefined_normal_has_no_tangent_reference(ct_tool, tmp_path, monkeypatch):
    case, rois, cfg, _, _, _ = reference_fixture(ct_tool, tmp_path)
    monkeypatch.setattr(ct_tool, "normals_at", lambda v, f, n, p, rows: np.full_like(p, np.nan))
    report = ct_tool.render_references(case, rois, tmp_path / "references", cfg, lambda: None)
    region = report["regions"][0]
    assert region["tangent_missing_reason"] == "undefined original normal"
    with np.load(tmp_path / "references" / region["raw"], allow_pickle=False) as data:
        assert not data["support"][0].any()
        assert data["support"][1:].any()


def test_reference_run_refuses_active_sampler(ct_tool, tmp_path, monkeypatch):
    import bench_flattening as b

    run = tmp_path / "run"
    run.mkdir()
    b.atomic_json(run / "QUEUE.json", dict(rows=[dict(status="RUNNING")]))
    monkeypatch.setattr(
        b, "require_benchmark_envelope", lambda: dict(slice="field-flattening-test.slice")
    )
    with pytest.raises(InterruptedError, match="pause|active"):
        ct_tool.render_reference_run(run, tmp_path / "controls")
    assert not (tmp_path / "controls").exists()


def test_reference_run_preserves_source_run_and_geometry(ct_tool, tmp_path, monkeypatch):
    import bench_flattening as b

    sources = []
    for unit, field in [(8.64, "175a"), (2.4, "paris4")]:
        case, _, _, _, _, _ = reference_fixture(ct_tool, tmp_path / field, unit, field)
        source = b.read_json(Path(case["root"]) / "SOURCE.json")["identity"]["source"]
        sources.append(dict(source, stage="pilot"))
    config = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0,
            seconds=60.0,
            scratch_bytes=20 << 30,
            cache_bytes=1 << 24,
            decoded_cache_bytes=1 << 20,
        ),
    )
    run = tmp_path / "run"
    b.inventory(config, run)
    files = [run / name for name in ("RUN.json", "INVENTORY.json", "QUEUE.json")]
    files += list((run / "cases").rglob("*.npy"))
    before = {p: b.file_sha(p) for p in files}
    monkeypatch.setattr(
        b, "require_benchmark_envelope", lambda: dict(slice="field-flattening-test.slice")
    )
    report = ct_tool.render_reference_run(run, tmp_path / "controls")
    assert report["fields"] == ["175a", "paris4"]
    assert len(report["rows"]) == 2 and all(row["status"] == "COMPLETE" for row in report["rows"])
    assert report["diagnostic_only"] and not report["physical_acceptance"]
    assert before == {p: b.file_sha(p) for p in files}
    receipt = b.file_sha(tmp_path / "controls/CONTROL-RUN.json")
    metadata_path = Path(sources[0]["ct_profile"]["source"]) / "0/.zarray"
    metadata = b.read_json(metadata_path)
    metadata["order"] = "F"
    b.atomic_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="metadata|identity"):
        ct_tool.render_reference_run(run, tmp_path / "controls")
    assert b.file_sha(tmp_path / "controls/CONTROL-RUN.json") == receipt
