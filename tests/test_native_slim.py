import json
import os

import numpy as np
import pytest


@pytest.fixture
def flatboi_app():
    """Real pinned Flatboi runtime; CI sets WINDING_REQUIRE_FLATBOI=1 so absence fails."""
    from winding_field.flattening import flatboi

    required = os.environ.get("WINDING_REQUIRE_FLATBOI") == "1"
    if not flatboi.supported_platform():
        if required:
            pytest.fail("WINDING_REQUIRE_FLATBOI=1 on a platform without Flatboi")
        pytest.skip("Flatboi SLIM is available only on Linux x86_64")
    try:
        app = flatboi.resolve_app_run(offline=not required)
        from winding_field.patch_pipeline import check_flatboi

        return check_flatboi(str(app), "historical")["app_run"]
    except ValueError as error:
        if required:
            raise
        pytest.skip(f"Flatboi runtime not provisioned: {error}")


def test_base_distribution_installs_pinned_lscm_without_tiff_extra():
    from importlib.metadata import requires

    requirements = requires("winding-field-cli")
    assert "libigl==2.6.3" in requirements
    assert not any(r.startswith("libigl") and ";" in r for r in requirements)
    assert any(r.startswith("tifffile") and 'extra == "tifxyz"' in r for r in requirements)
    assert not any(r.startswith("tifffile") and ";" not in r for r in requirements)


def test_embedded_appimage_pin_matches_provenance():
    from pathlib import Path

    from winding_field.flattening import flatboi
    from winding_field.patch_pipeline import LAUNCHER_SHA, NATIVE_RECIPES

    provenance = json.loads(
        (Path(__file__).resolve().parents[1] / "provenance/flatboi.json").read_text()
    )
    assert flatboi.APPIMAGE_URL == provenance["url"]
    assert flatboi.APPIMAGE_SHA256 == provenance["sha256"]
    assert LAUNCHER_SHA == provenance["launcher_sha256"]
    assert NATIVE_RECIPES["historical"] == provenance["binary_sha256"]


def test_flatboi_reduces_real_symmetric_dirichlet_energy_without_changing_source(flatboi_app):
    from winding_field.flattening.flatboi import solve_batch
    from winding_field.native_geometry import _validate

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    seed = vertices[:, :2] * [2.0, 0.5]
    before = [a.copy() for a in (vertices, faces, seed)]
    [result], receipt = solve_batch([(vertices, faces, seed)], flatboi_app)
    # Hand-derived inverse XY edge matrices for these two unit-square triangles.
    inverse = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [-1.0, 0.0]]])

    def energy(uv):
        tri = uv[faces]
        jac = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], axis=-1) @ inverse
        singular = np.linalg.svd(jac, compute_uv=False)
        return np.sum(singular**2 + singular**-2)

    assert energy(result) < energy(seed) * 0.8
    _validate(vertices, faces, result - result.min(0))
    assert receipt["steps"] >= 1 and receipt["energy_last"] <= receipt["energy_first"]
    for actual, expected in zip((vertices, faces, seed), before):
        np.testing.assert_array_equal(actual, expected)


def test_batched_charts_match_individual_solves(flatboi_app):
    from winding_field.flattening.flatboi import solve_batch

    from test_meshes import curved_strip

    from winding_field.native_geometry import _lscm

    vertices, faces = curved_strip()
    seed, _ = _lscm(vertices, faces)  # production seeds are validated LSCM/projections
    charts = [(vertices + [3.0 * k, 0, 0], faces, seed) for k in range(3)]
    batched, _ = solve_batch(charts, flatboi_app)
    single, _ = solve_batch(charts[:1], flatboi_app)
    for uv in batched:
        np.testing.assert_allclose(uv - uv.min(0), single[0] - single[0].min(0), atol=1e-6)


def test_slim_native_pipeline_records_flatboi_and_reuses_completed_charts(tmp_path, flatboi_app):
    from test_native_runtime import field_case
    from winding_field.native_pipeline import work
    from winding_field.native_store import Journal

    run, config = field_case(tmp_path, levels=(0.5,))
    config["flattening_method"] = "slim"
    assert work(run, config, until="flatten")["status"] == "PAUSED"
    identity = json.loads((run / "surfaces/NATIVE.json").read_bytes())
    parameterizer = identity["settings"]["parameterizer"]
    assert parameterizer["solver"] == "flatboi-slim" and parameterizer["recipe"] == "historical"
    root = run / "surfaces/level-0000"
    with Journal(root, json.loads((root / "IDENTITY.json").read_bytes())) as store:
        qualities = [
            json.loads(q)["local"] for (q,) in store.db.execute("SELECT quality FROM charts")
        ]
        assert qualities and all(
            q["method"] == "SLIM" and q["backend"] == "flatboi" and 1 <= q["iterations"] <= 200
            for q in qualities
        )
    before = {p: p.read_bytes() for p in (root / "charts").glob("*.npz")}
    assert work(run, config)["status"] == "COMPLETE"
    assert work(run, config)["status"] == "COMPLETE"
    assert all(p.read_bytes() == raw for p, raw in before.items())
    with pytest.raises(ValueError, match="identity changed"):
        work(run, dict(config, flattening_method="default"))


@pytest.mark.parametrize("method", [None, "default"])
def test_default_native_configs_keep_identity_and_never_invoke_slim(tmp_path, monkeypatch, method):
    import winding_field.native_geometry as geometry
    from test_native_runtime import field_case
    from winding_field.native_pipeline import work, _settings

    run, config = field_case(tmp_path, levels=(0.5,))
    if method:
        config["flattening_method"] = method
    before = config.copy()
    settings = _settings(config)
    assert settings == dict(
        coverage_target=1.0,
        surface_time_limit_s=30.0,
        chart_faces=32,
        mesh_tile=4,
        export_tifxyz=False,
        parameterizer=geometry.solver_identity(),
    )
    monkeypatch.setattr(
        geometry, "_slim_batch", lambda *a, **kw: pytest.fail("default chart invoked SLIM")
    )
    assert work(run, config, until="flatten")["status"] == "PAUSED"
    assert work(run, config)["status"] == "COMPLETE"
    assert config == before


def test_invalid_seeds_never_reach_flatboi(monkeypatch):
    import winding_field.native_geometry as geometry

    seen = []
    monkeypatch.setattr(geometry, "_slim_batch", lambda items, *a, **kw: seen.extend(items) or [])
    saved = []
    monkeypatch.setattr(
        geometry, "_save_pieces", lambda store, items: saved.extend(items) or ([], [])
    )
    monkeypatch.setattr(geometry, "_commit", lambda *a: None)
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    good = (v, np.array([[0, 1, 2]]), v[:, :2], np.array([1]), np.arange(3), {})
    bad = (v, np.array([[0, 1, 2]]), np.empty((0, 2)), np.array([2]), np.arange(3), {"error": "x"})

    class Store:
        def check(self):
            pass

    geometry._flush_slim(Store(), [(7, good), (8, bad)], "unused", 2, 2)
    assert [c for c, _ in seen] == [7]
    assert [c for c, _ in saved] == [8]


def test_slim_stop_kills_the_native_process(flatboi_app):
    from winding_field.flattening.flatboi import solve_batch
    from winding_field.native_store import Paused

    from test_meshes import curved_strip

    vertices, faces = curved_strip()
    calls = []

    def check():
        calls.append(True)
        raise Paused("STOP")

    with pytest.raises(Paused, match="STOP"):
        solve_batch([(vertices, faces, vertices[:, :2])], flatboi_app, check=check)
    assert calls == [True]


@pytest.mark.parametrize("shape", ["curved-strip", "closed-octahedron"])
def test_slim_keeps_source_ids_after_lscm_or_recursive_chart_splitting(shape, flatboi_app):
    from winding_field.native_geometry import _pieces, _slim_batch, _validate

    if shape == "curved-strip":
        from test_meshes import curved_strip

        vertices, faces = curved_strip()
    else:
        vertices = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        faces = np.array(
            [[4, 0, 2], [4, 1, 3], [5, 1, 2], [5, 0, 3], [4, 2, 1], [4, 3, 0], [5, 2, 0], [5, 3, 1]]
        )
    ids = np.arange(len(faces)) + 10
    source_ids = np.arange(len(vertices)) + 100
    seeded = [(0, piece) for piece in _pieces(vertices, faces, ids, source_ids, minimum=4)]
    pieces = _slim_batch([item for item in seeded if len(item[1][2])], flatboi_app, minimum=4)
    seen = []
    for component, (v, f, uv, face_ids, vertex_ids, quality) in pieces:
        assert component == 0
        assert len(uv) and quality["method"] == "SLIM" and quality["backend"] == "flatboi"
        if shape == "curved-strip":
            assert quality["initialization"] == "LSCM"
        _validate(v, f, uv)
        seen.extend(face_ids.tolist())
        np.testing.assert_array_equal(v[f], vertices[faces[face_ids - 10]])
        np.testing.assert_array_equal(vertex_ids[f], source_ids[faces[face_ids - 10]])
    assert sorted(seen) == ids.tolist()


def test_slim_rejects_collapsed_optimizer_output_instead_of_reusing_seed(monkeypatch):
    import winding_field.flattening.flatboi as flatboi
    from winding_field.native_geometry import _slim_batch

    monkeypatch.setattr(
        flatboi,
        "solve_batch",
        lambda charts, *a, **kw: (
            [np.zeros((len(v), 2)) for v, _, _ in charts],
            dict(steps=1, converged=True),
        ),
    )
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    piece = (v, np.array([[0, 1, 2]]), v[:, :2], np.array([1]), np.arange(3), {"method": "LSCM"})
    [(component, (vertices, faces, uv, ids, source_ids, quality))] = _slim_batch(
        [(3, piece)], "unused"
    )
    assert component == 3 and uv.shape == (0, 2)
    assert quality["error"]


def test_slim_is_rejected_off_linux_x86_64_before_creating_a_run(tmp_path, monkeypatch):
    import platform

    from winding_field.workflow import run

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="Linux x86_64"):
        run(
            dict(
                source="unused",
                output=str(tmp_path / "out"),
                surface_output="native",
                flattening_method="slim",
            )
        )
    assert not (tmp_path / "out").exists()


def test_explicit_recipe_without_app_run_is_rejected():
    from winding_field.flattening.flatboi import resolve_app_run

    with pytest.raises(ValueError, match="explicit app_run"):
        resolve_app_run(None, "installed-2f89bfc9")


def test_offline_without_cached_runtime_does_not_download(tmp_path, monkeypatch):
    import winding_field.flattening.flatboi as flatboi

    monkeypatch.delenv("WINDING_FLATBOI_APP_RUN", raising=False)
    monkeypatch.setenv("WINDING_FLATBOI_CACHE", str(tmp_path))
    monkeypatch.setattr(flatboi, "CONTAINER_APP_RUN", tmp_path / "absent")
    monkeypatch.setattr(flatboi, "supported_platform", lambda: True)
    monkeypatch.setattr(flatboi, "provision", lambda *a, **kw: pytest.fail("downloaded offline"))
    with pytest.raises(ValueError, match="downloads are disabled"):
        flatboi.resolve_app_run(offline=True)
