"""The selected solver/scope must own all faces and survive a verified resume."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

from winding_field.util import atomic_json, file_sha


def matrix():
    assert importlib.util.find_spec("winding_field.flattening.matrix"), (
        "flattening matrix is missing"
    )
    from winding_field.flattening import matrix as flattening_matrix

    return flattening_matrix


def request(tmp_path):
    return dict(
        output_frame=dict(id="test", voxel_um=9.6, origin_xyz_um=[0, 0, 0]),
        tile_size=16,
        pitch_um=9.6,
        cache_dir=str(tmp_path / "cache"),
        memory_gib=2.0,
    )


def geometry():
    x, z = np.meshgrid(np.arange(4.0, 13.0), np.arange(4.0, 13.0))
    v = np.column_stack([x.ravel(), np.full(x.size, 3.0), z.ravel()])
    f = []
    for j in range(8):
        for i in range(8):
            if 3 <= i < 5 and 3 <= j < 5:
                continue
            a = j * 9 + i
            f.extend([[a, a + 1, a + 10], [a, a + 10, a + 9]])
    # A separate island must survive whole-sheet as well as patch mapping.
    v = np.concatenate([v, [[20.0, 3.0, 4.0], [24.0, 3.0, 4.0], [20.0, 3.0, 8.0]]])
    f = np.asarray(f + [[81, 82, 83]], np.int64)
    ids, inv = np.unique(f, return_inverse=True)
    return v[ids], inv.reshape(-1, 3)


def test_settings_select_native_recipe_only_for_slim(tmp_path):
    m = matrix()
    req = request(tmp_path)
    assert m.settings(req, "lscm", "sheet")["flattening_solver"] == "lscm"
    slim = m.settings(req, "slim", "sheet")
    # SLIM defaults to the pinned historical AppImage, resolved (and verified) when it runs.
    assert slim["native_recipe"] == "historical" and slim.get("app_run") is None
    with pytest.raises(ValueError, match="native_recipe"):
        m.settings(dict(req, native_recipe="unknown"), "slim", "sheet")
    for extra in ({"max_patch_faces": 32}, {"max_depth": 2}, {"iterations": 10}):
        with pytest.raises(ValueError):
            m.settings(dict(req, **extra), "lscm", "sheet")
    assert m.settings(dict(req, max_patch_faces=32), "lscm", "patches")["max_patch_faces"] == 32
    with pytest.raises(ValueError):
        m.settings(dict(req, disk_reserve_gib=0), "lscm", "patches")


@pytest.mark.parametrize(
    "flags",
    [
        ["--flattening-solver", "slim", "--scope", "sheet", "--flattening-method", "original"],
        ["--flattening-solver", "lscm", "--flattening-method", "patches"],
        ["--scope", "patches", "--flattening-method", "original"],
    ],
)
def test_mixed_cli_selection_fails_before_output(tmp_path, capsys, flags):
    """Omitted solver/scope default to lscm-slim/sheet; mixing with a method is an error."""
    from winding_field.cli import main

    assert (
        main(
            [
                "export-sheets",
                "--mesh-bundle",
                "missing",
                "--source",
                "missing",
                "--flattening-config",
                "missing",
                "--output",
                str(tmp_path / "out"),
                *flags,
            ]
        )
        == 1
    )
    assert "cannot combine" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("scope", ["patches", "sheet"])
def test_lscm_map_preserves_hole_island_and_oriented_source_faces(tmp_path, scope):
    m = matrix()
    v, f = geometry()
    req = request(tmp_path)
    if scope == "patches":
        req["max_patch_faces"] = 32
    cfg = m.settings(req, "lscm", scope)
    out = tmp_path / "map"
    result = m.build_map(v, f, out, cfg)
    np.testing.assert_array_equal(result["world"], v[result["parent"]])
    np.testing.assert_array_equal(result["parent"][result["faces"]], f[result["source_face_rows"]])
    np.testing.assert_array_equal(np.sort(result["source_face_rows"]), np.arange(121))
    assert result["report"]["source_components"] == 2
    domains = result["report"]["domains"]
    assert (
        len(domains) == 2
        if scope == "sheet"
        else len(domains) > 2 and max(d["faces"] for d in domains) <= 32
    )
    assert result["report"]["overlap"]["complete"]
    assert result["report"]["overlapping_faces"] == 0
    assert result["report"]["quality"]["inverted_or_invalid_area_fraction"] == 0
    assert result["report"]["welded"] is False
    saved = file_sha(out / "MAP.json")
    resumed = m.build_map(v, f, out, cfg)
    np.testing.assert_array_equal(resumed["uv"], result["uv"])
    assert file_sha(out / "MAP.json") == saved
    with pytest.raises(ValueError, match="identity"):
        m.build_map(v, f, out, dict(cfg, scope="sheet" if scope == "patches" else "patches"))
    (out / "map.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        m.build_map(v, f, out, cfg)


def test_solver_failure_never_falls_back_or_subdivides(tmp_path, monkeypatch):
    m = matrix()
    v, f = geometry()
    cfg = m.settings(request(tmp_path), "lscm", "sheet")

    def fail(*a, **kw):
        raise RuntimeError("solver failure fixture")

    monkeypatch.setattr(m, "solve", fail)
    with pytest.raises(RuntimeError, match="solver failure"):
        m.build_map(v, f, tmp_path / "map", cfg)
    assert not (tmp_path / "map" / "MAP.json").exists()
    assert len(list((tmp_path / "map").glob("domain-*"))) == 1


def test_sheet_component_coverage_is_not_limited_by_patch_chart_count(tmp_path):
    m = matrix()
    v, f = geometry()
    cfg = m.settings(request(tmp_path), "lscm", "sheet")
    cfg["max_charts"] = 1
    result = m.build_map(v, f, tmp_path / "map", cfg)
    assert result["report"]["source_components"] == 2
    assert len(result["report"]["domains"]) == 2
    np.testing.assert_array_equal(np.sort(result["source_face_rows"]), np.arange(len(f)))


def test_saved_domains_finish_audit_without_solver(tmp_path, monkeypatch):
    m = matrix()
    v, f = geometry()
    cfg = m.settings(request(tmp_path), "lscm", "sheet")
    out = tmp_path / "map"
    first = m.build_map(v, f, out, cfg)
    for name in ("MAP.json", "map.npz", "AUDIT.json"):
        (out / name).unlink(missing_ok=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("solver must not run for saved domains")

    monkeypatch.setattr(m, "solve", forbidden)
    again = m.build_map(v, f, out, cfg)
    assert again["report"]["overlapping_faces"] == first["report"]["overlapping_faces"]
    assert again["report"]["overlap"]["status"] == "COMPLETE"
    np.testing.assert_array_equal(np.sort(again["source_face_rows"]), np.arange(len(f)))
    assert len(again["report"]["domains"]) == len(first["report"]["domains"])


def source_bundle(tmp_path):
    from winding_field.surface_mesh import import_mesh

    v, f = geometry()
    source = tmp_path / "source.npz"
    np.savez(source, vertices_xyz=v, faces=f)
    metadata = dict(
        frame="test",
        voxel_um=9.6,
        origin_xyz=[0, 0, 0],
        spacing_xyz=[1] * 3,
        shape_zyx=[32] * 3,
        sheet_step=1.0,
    )
    root = tmp_path / "mesh"
    import_mesh(dict(path=str(source), sha256=file_sha(source)), root, metadata)
    ct = tmp_path / "ct"
    ct.mkdir()
    atomic_json(
        ct / ".zarray",
        dict(
            zarr_format=2,
            shape=[32] * 3,
            chunks=[32] * 3,
            dtype="|u1",
            compressor=None,
            fill_value=0,
            order="C",
            filters=None,
        ),
    )
    z, y, x = np.indices((32, 32, 32))
    (ct / "0.0.0").write_bytes((x + 2 * z).astype("u1").tobytes())
    return root, ct


@pytest.mark.parametrize("kind", ["npy", "chunks"])
@pytest.mark.parametrize("scope", ["patches", "sheet"])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_field_config_extracts_scalar_mesh_and_resumes_verified_export(tmp_path, kind, scope):
    from winding_field.cli import main
    from winding_field.flattening.patch_mapping import PatchMapping
    from field_store import BlockStore

    _, ct = source_bundle(tmp_path)
    values = np.broadcast_to(np.arange(7.0), (7, 7, 7)).copy()
    geometry = dict(
        frame="test",
        voxel_um=9.6,
        origin_xyz=[4.0] * 3,
        spacing_xyz=[1.0] * 3,
        shape_zyx=[7] * 3,
        sheet_step=1.0,
    )
    if kind == "npy":
        path = tmp_path / "field.npy"
        np.save(path, values)
        field = dict(kind=kind, path=str(path), sha256=file_sha(path))
    else:
        with BlockStore(tmp_path / "field", values.shape, 8, {"grid": geometry}) as store:
            store.commit((0, 0, 0), values, {}, np.ones(values.shape, bool))
        path = tmp_path / "field/CHECKPOINT.json"
        field = dict(kind=kind, path=str(path), sha256=file_sha(path), support_policy="observed")
    atomic_json(tmp_path / "field.json", dict(field=field, geometry=geometry))
    atomic_json(tmp_path / "config.json", request(tmp_path))
    out = tmp_path / "out"
    args = [
        "export-sheets",
        "--field-config",
        str(tmp_path / "field.json"),
        "--levels",
        "3.5",
        "--source",
        str(ct),
        "--voxel-um",
        "9.6",
        "--flattening-solver",
        "lscm",
        "--scope",
        scope,
        "--flattening-config",
        str(tmp_path / "config.json"),
        "--output",
        str(out),
    ]
    assert main(args) == 0
    record = json.loads((out / "level-0000/LEVEL.json").read_text())
    assert record["rendering"]["CT_observed_pixels"] > 0
    with PatchMapping(out / "level-0000") as mapped:
        np.testing.assert_array_equal(mapped.arrays["field_xyz"][:, 0], 7.5)
        assert len(mapped.arrays["faces"]) == 72
    before = {
        p: file_sha(p)
        for p in out.rglob("*")
        if p.is_file() and p.suffix in (".json", ".npz", ".png", ".tif", ".obj")
    }
    assert main(args + ["--resume"]) == 0
    assert all(file_sha(p) == sha for p, sha in before.items())


@pytest.mark.parametrize("solver", ["lscm", "slim", "lscm-slim"])
@pytest.mark.parametrize("scope", ["patches", "sheet"])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_cli_matrix_ct_xyz_keys_and_completed_resume(tmp_path, capsys, solver, scope):
    app = os.environ.get("MATRIX_NATIVE_APP")
    if solver != "lscm" and not app:
        pytest.skip("native qualification requires MATRIX_NATIVE_APP and bounded Linux cgroup")
    from winding_field.cli import main
    from winding_field.flattening.patch_mapping import PatchMapping

    mesh, ct = source_bundle(tmp_path)
    req = request(tmp_path)
    if scope == "patches":
        req["max_patch_faces"] = 32
    if solver != "lscm":
        req["app_run"] = app
        req["native_recipe"] = os.environ.get("MATRIX_NATIVE_RECIPE", "installed-2f89bfc9")
    config = tmp_path / "config.json"
    atomic_json(config, req)
    out = tmp_path / "out"
    args = [
        "export-sheets",
        "--mesh-bundle",
        str(mesh),
        "--source",
        str(ct),
        "--voxel-um",
        "9.6",
        "--flattening-solver",
        solver,
        "--scope",
        scope,
        "--flattening-config",
        str(config),
        "--output",
        str(out),
    ]
    assert main(args) == 0
    receipt = json.loads((out / "MATRIX.json").read_text())
    assert receipt["status"] == "MATRIX_READY_DIAGNOSTIC"
    level = out / "level-0000"
    record = json.loads((level / "LEVEL.json").read_text())
    assert record["mapping"]["flattening_solver"] == solver and record["mapping"]["scope"] == scope
    with PatchMapping(level) as mapped:
        a = mapped.arrays
        assert len(list(mapped.iter_keys("faces"))) == 121
        assert len(set(a["source_face_ids"])) == 121
        np.testing.assert_array_equal(a["field_xyz"][:, 1], 3.0)
        domains = record["mapping"]["domains"]
        assert (
            len(domains) == 2
            if scope == "sheet"
            else len(domains) > 2 and max(d["faces"] for d in domains) <= 32
        )
        if solver != "lscm":
            m = matrix()
            v, f = geometry()
            other = dict(req)
            other.pop("app_run")
            other.pop("native_recipe")
            comparison = m.build_map(
                v, f, tmp_path / "lscm-comparison", m.settings(other, "lscm", scope)
            )
            assert record["mapping"]["partition_sha256"] == comparison["report"]["partition_sha256"]
            np.testing.assert_array_equal(a["domain_ids"], comparison["domain_ids"])
            np.testing.assert_array_equal(a["faces"], comparison["faces"])
    checked = 0
    for tile in record["rendering"]["tiles"]:
        assert tile["rendering"]["disconnected_components_translated"] is True
        assert tile["rendering"]["patch_repacking"] is (scope == "patches")
        root = level / "render" / tile["directory"]
        with np.load(root / "mapping.npz") as data:
            im = np.asarray(Image.open(root / "image.png"))
            mask = np.asarray(Image.open(root / "mask.png"))
            xyz = data["xyz"][mask == 2]
            checked += len(xyz)
            np.testing.assert_array_equal(
                im[mask == 2], np.rint(xyz[:, 0] + 2 * xyz[:, 2]).astype("u1")
            )
            assert not ((xyz[:, 0] > 7) & (xyz[:, 0] < 9) & (xyz[:, 2] > 7) & (xyz[:, 2] < 9)).any()
            assert not im[mask != 2].any()
    assert checked > 20
    before = file_sha(out / "MATRIX.json")
    level_before = file_sha(level / "LEVEL.json")
    assert main(args + ["--resume"]) == 0
    assert (
        file_sha(out / "MATRIX.json") == before and file_sha(level / "LEVEL.json") == level_before
    )
    changed = args.copy()
    changed[changed.index("--scope") + 1] = "sheet" if scope == "patches" else "patches"
    assert main(changed + ["--resume"]) == 1
    assert file_sha(out / "MATRIX.json") == before
    (ct / "0.0.0").write_bytes(bytes(32**3))
    assert main(args + ["--resume"]) == 1
    assert "CT chunk changed" in capsys.readouterr().err


def cli_fixture(tmp_path):
    mesh, ct = source_bundle(tmp_path)
    config = tmp_path / "config.json"
    atomic_json(config, request(tmp_path))
    out = tmp_path / "out"
    return (
        out,
        ct,
        [
            "export-sheets",
            "--mesh-bundle",
            str(mesh),
            "--source",
            str(ct),
            "--voxel-um",
            "9.6",
            "--flattening-solver",
            "lscm",
            "--scope",
            "sheet",
            "--flattening-config",
            str(config),
            "--output",
            str(out),
        ],
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_interrupted_render_retains_map_and_evidence_then_resumes(tmp_path, monkeypatch):
    from winding_field.cli import main
    from winding_field.flattening import patches as p

    out, ct, args = cli_fixture(tmp_path)
    real = p.render_map

    def interrupt(mapped, output, *a, **kw):
        output.mkdir()
        (output / "partial.txt").write_text("retained interrupted render")
        raise InterruptedError("render stop fixture")

    monkeypatch.setattr(p, "render_map", interrupt)
    assert main(args) == 1
    initial = file_sha(out / "level-0000/flattening/MAP.json")
    monkeypatch.setattr(p, "render_map", real)
    assert main(args + ["--resume"]) == 0
    assert file_sha(out / "level-0000/flattening/MAP.json") == initial
    assert len(list((out / "level-0000").glob("render-interrupted-*/partial.txt"))) == 1


@pytest.mark.parametrize("tamper", ["missing-ledger", "removed-entry", "symlink"])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_incomplete_publication_rejects_lost_ct_provenance_and_symlinks(tmp_path, tamper):
    import sqlite3
    from winding_field.cli import main

    out, ct, args = cli_fixture(tmp_path)
    assert main(args) == 0
    # Simulate interruption after committing a level but before the root receipt.
    (out / "MATRIX.json").unlink()
    if tamper == "missing-ledger":
        (out / "ct/ct-chunks.sqlite").unlink()
    elif tamper == "removed-entry":
        with sqlite3.connect(out / "ct/ct-chunks.sqlite") as db:
            db.execute("DELETE FROM chunks")
    else:
        external = tmp_path / "external"
        external.mkdir()
        (out / "level-0000/flattening/foreign").symlink_to(external, target_is_directory=True)
    assert main(args + ["--resume"]) == 1
    assert not (out / "MATRIX.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_native_checkpoint_interrupt_resumes_without_rewriting_prior_batch(tmp_path, monkeypatch):
    from winding_field.flattening import flatboi_checkpoint as r
    from test_flatboi_checkpoint import fake_native

    m = matrix()
    app = fake_native(tmp_path, monkeypatch, r)
    cfg = m.settings(dict(request(tmp_path), app_run=str(app)), "slim", "sheet")
    v = np.array([[1.0, 1, 1], [3.0, 1, 1], [1.0, 1, 3]])
    f = np.array([[0, 1, 2]], np.int64)
    publish = r.publish_checkpoint

    def interrupt(*a, **kw):
        publish(*a, **kw)
        raise InterruptedError("after durable native checkpoint")

    monkeypatch.setattr(r, "publish_checkpoint", interrupt)
    out = tmp_path / "map"
    with pytest.raises(InterruptedError):
        m.build_map(v, f, out, cfg)
    native = out / "domain-000000/native"
    checkpoint = json.loads((native / "LATEST.json").read_text())
    prior = native / checkpoint["obj"]
    sha = file_sha(prior)
    assert checkpoint["iterations"] == 10 and not (out / "MAP.json").exists()
    monkeypatch.setattr(r, "publish_checkpoint", publish)
    mapped = m.build_map(v, f, out, cfg)
    assert file_sha(prior) == sha
    assert mapped["report"]["domains"][0]["solver"]["checkpoint"]["iterations"] == 200


def uv_args(tmp_path, mesh, out, stages, ct=None):
    from winding_field.util import atomic_json

    config = tmp_path / f"config-{stages}.json"
    atomic_json(config, request(tmp_path))
    args = [
        "export-sheets",
        "--mesh-bundle",
        str(mesh),
        "--flattening-solver",
        "lscm",
        "--scope",
        "sheet",
        "--flattening-config",
        str(config),
        "--stages",
        stages,
        "--output",
        str(out),
    ]
    if ct is not None:
        args += ["--source", str(ct), "--voxel-um", "9.6"]
    return args


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_uv_stage_publishes_without_opening_ct(tmp_path):
    from winding_field.cli import main
    from winding_field.flattening import matrix_pipeline as pipeline

    mesh, _ = source_bundle(tmp_path)
    out = tmp_path / "uv-only"

    def forbidden(*args, **kwargs):
        raise AssertionError("UV publication must not open the CT")

    pipeline.PatchCT = forbidden if hasattr(pipeline, "PatchCT") else None
    import winding_field.flattening.patches as patch

    original = patch.PatchCT
    patch.PatchCT = forbidden
    try:
        assert main(uv_args(tmp_path, mesh, out, "uv")) == 0
    finally:
        patch.PatchCT = original
    assert (out / "level-0000/LEVEL-UV.json").exists() and (out / "level-0000/map.npz").exists()
    assert not (out / "ct").exists() and not (out / "level-0000/LEVEL.json").exists()
    receipt = json.loads((out / "UV-MATRIX.json").read_text())
    assert receipt["status"] == "UV_READY_DIAGNOSTIC" and receipt["levels"]
    record = json.loads((out / "level-0000/LEVEL-UV.json").read_text())
    assert record["map_sha256"] == file_sha(out / "level-0000/map.npz")
    assert record["identity"]["ct_source"] is None and record["published_by"] == "uv"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_ct_stage_consumes_published_uv_without_solving(tmp_path, monkeypatch):
    from winding_field.cli import main
    from winding_field.flattening import matrix_pipeline as pipeline

    mesh, ct = source_bundle(tmp_path)
    out = tmp_path / "split"
    assert main(uv_args(tmp_path, mesh, out, "uv")) == 0
    before = file_sha(out / "level-0000/map.npz")

    def forbidden(*args, **kwargs):
        raise AssertionError("CT stage must not solve the UV")

    monkeypatch.setattr(pipeline, "build_map", forbidden)
    assert main(uv_args(tmp_path, mesh, out, "ct", ct)) == 0
    assert file_sha(out / "level-0000/map.npz") == before
    record = json.loads((out / "level-0000/LEVEL.json").read_text())
    assert record["rendering"]["CT_observed_pixels"] > 0
    assert record["mapping"]["flattening_solver"] == "lscm"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_ct_stage_requires_published_uv_and_rejects_tampering(tmp_path, capsys):
    from winding_field.cli import main

    mesh, ct = source_bundle(tmp_path)
    assert main(uv_args(tmp_path, mesh, tmp_path / "missing", "ct", ct)) != 0
    assert "UV" in capsys.readouterr().err
    out = tmp_path / "tampered"
    assert main(uv_args(tmp_path, mesh, out, "uv")) == 0
    (out / "level-0000/map.npz").write_bytes((out / "level-0000/map.npz").read_bytes() + b"x")
    assert main(uv_args(tmp_path, mesh, out, "ct", ct)) != 0
    assert "map" in capsys.readouterr().err or "UV" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree resource guard")
def test_uv_stage_rejects_ct_inputs(tmp_path, capsys):
    from winding_field.cli import main

    mesh, ct = source_bundle(tmp_path)
    assert main(uv_args(tmp_path, mesh, tmp_path / "bad", "uv", ct)) != 0
    assert "CT" in capsys.readouterr().err


def test_render_contract_ignores_solver_and_runtime_identity():
    """The CT stage must not need the native binary to agree on its identity."""
    from winding_field.flattening import matrix_pipeline as pipeline
    from winding_field.util import digest

    identity = dict(
        schema="flattening-matrix-v1",
        settings=dict(flattening_solver="slim"),
        mesh="/m/mesh",
        mesh_sha256="a" * 64,
        levels=None,
        ct_source=None,
        ct_level=None,
        ct_voxel_um=None,
        export_tifxyz=False,
        native=dict(binary_sha256="b" * 64, launcher_sha256="c" * 64),
        native_runtime=dict(kind="flatboi"),
        runtime=dict(numpy="2"),
        implementation_sha256="d" * 64,
        field=None,
    )
    contract = pipeline.render_contract(identity)
    assert (
        "native" not in contract and "native_runtime" not in contract and "runtime" not in contract
    )
    published = dict(identity, ct_source=None, ct_level=None, ct_voxel_um=None)
    # A render host without the solver accepts the published UV, and a different
    # settings/mesh contract is still rejected.
    assert all(published.get(k) == v for k, v in contract.items())
    assert not all(dict(published, mesh_sha256="e" * 64).get(k) == v for k, v in contract.items())
    assert digest(contract) != digest(identity)


def test_lscm_slim_seeds_flatboi_with_global_lscm_and_falls_back_when_it_folds(monkeypatch):
    from winding_field.flattening import matrix as m
    from winding_field.flattening.native_sheet import cut_component

    v, f = geometry()
    keep = np.unique(f[:-1])  # the main sheet, without the island
    cut, _ = cut_component(*_compact(v, f[:-1]))
    uv, init = m._lscm_seed(cut, lambda: None, diagnostics=None)
    assert init["method"] == "LSCM" and uv.shape == (len(cut["vertices"]), 2)
    assert len(keep)
    folded = lambda cut: (np.zeros((len(cut["vertices"]), 2)), np.array([0, 1]), None)
    monkeypatch.setattr(m, "_global_lscm", folded)
    uv, init = m._lscm_seed(cut, lambda: None, diagnostics=None)
    assert init["lscm_rejected"]["folded_faces"] == len(cut["faces"])
    assert np.isfinite(uv).all()


def _compact(v, f):
    used, inverse = np.unique(f, return_inverse=True)
    return v[used], inverse.reshape(-1, 3)


def test_export_sheets_defaults_to_lscm_slim_whole_sheets(monkeypatch):
    from winding_field.flattening import flatboi, selection

    monkeypatch.setattr(flatboi, "supported_platform", lambda: True)
    assert (selection.default_solver(), selection.DEFAULT_SCOPE) == ("lscm-slim", "sheet")
    monkeypatch.setattr(flatboi, "supported_platform", lambda: False)
    assert selection.default_solver() == "lscm"
    m = matrix()
    for solver in selection.FLATBOI_SOLVERS:
        assert m.settings(request(Path("/tmp")), solver, "sheet")["native_recipe"] == "historical"
