"""export-sheets defaults to the scroll render-ct sheet route; other lanes stay selectable."""

import json

import numpy as np
import pytest

from test_native_slim import flatboi_app  # noqa: F401 (pytest fixture)


def _ct(tmp_path, monkeypatch):
    from test_native_render import analytic_ct
    import winding_field.workflow as workflow

    volume = analytic_ct(tmp_path)
    monkeypatch.setattr(workflow, "open_volume", lambda config: volume.thing)
    return volume


def _tifxyz(root):
    import tifffile

    folders = sorted(root.rglob("tifxyz"))
    assert folders
    return [
        np.stack([tifffile.imread(f / n) for n in ("x.tif", "y.tif", "z.tif")]) for f in folders
    ]


def test_export_sheets_default_is_the_render_ct_sheet_route(
    tmp_path, monkeypatch, capsys, flatboi_app
):  # noqa: F811
    from test_native_geometry import plane_bundle
    from winding_field import scroll
    from winding_field.cli import main

    mesh = plane_bundle(tmp_path, nx=9, ny=7)
    _ct(tmp_path, monkeypatch)
    out = tmp_path / "export"
    argv = ["export-sheets", "--mesh-bundle", str(mesh), "--source", "unused",
            "--pitch-um", "0.5", "--output", str(out)]  # fmt: skip
    assert main(argv) == 0
    receipt = json.loads((out / "SHEETS.json").read_text())
    assert receipt["status"] == "SHEETS_READY" and receipt["route"] == "native-sheet"
    level = json.loads((out / "level-0000" / "LEVEL.json").read_text())
    assert level["sheet_flatten"]["fallback"] == []
    identity = json.loads((out / "level-0000" / "IDENTITY.json").read_text())
    assert identity["settings"]["flattening_scope"] == "sheet"
    assert identity["settings"]["parameterizer"]["solver"] == "flatboi-slim"

    # The same mesh through scroll render-ct gives the same map, byte for byte.
    direct = tmp_path / "direct"
    args = scroll._parser().parse_args(
        [
            "render-ct",
            "--mesh",
            str(mesh),
            "--source",
            "unused",
            "--render-pitch-um",
            "0.5",
            "--output",
            str(direct),
        ]  # fmt: skip
    )
    assert scroll.render_ct(args)["status"] == "COMPLETE"
    for ours, theirs in zip(_tifxyz(out / "level-0000"), _tifxyz(direct), strict=True):
        np.testing.assert_array_equal(ours, theirs)

    # A finished export resumes to the same receipt; another route on it is refused.
    assert main([*argv, "--resume"]) == 0
    assert json.loads((out / "SHEETS.json").read_text()) == receipt
    other = [("0.25" if a == "0.5" else a) for a in argv]
    assert main([*other, "--resume"]) == 1
    assert "identity changed" in capsys.readouterr().err


def test_multi_level_bundle_renders_every_level(tmp_path, monkeypatch, flatboi_app):  # noqa: F811
    from test_surface_workflow import prepared
    from winding_field.cli import main

    monkeypatch.chdir(tmp_path)
    path, _ = prepared(tmp_path)
    mesh = tmp_path / "mesh"
    assert main(["prepared", "mesh", "--config", str(path), "--output", str(mesh)]) == 0
    _ct(tmp_path, monkeypatch)
    out = tmp_path / "flat"
    argv = ["export-sheets", "--mesh-bundle", str(mesh), "--source", "unused",
            "--output", str(out)]  # fmt: skip
    assert main(argv) == 0
    receipt = json.loads((out / "SHEETS.json").read_text())
    assert sorted(receipt["levels"]) == ["level-0000", "level-0001"]
    assert [receipt["levels"][k]["level_value"] for k in sorted(receipt["levels"])] == [1.5, 3.5]
    assert receipt["pitch_um"] == 2.0  # the finest mesh spacing, as run/plan render
    for name in receipt["levels"]:
        assert (out / name / "LEVEL.json").is_file()
        assert (out / "level-meshes" / name / "mesh" / "MESH.json").is_file()


@pytest.mark.parametrize(
    "extra,lane",
    [
        ((), "native"),
        (("--flattening-solver", "lscm"), "matrix"),
        (("--scope", "patches"), "matrix"),
        (("--flattening-solver", "lscm-slim", "--scope", "sheet"), "matrix"),
        (("--stages", "uv"), "matrix"),
        (("--flattening-method", "patches"), "patches"),
        (("--flattening-method", "original"), "original"),
    ],
)
def test_explicit_flags_keep_the_experimental_lanes(tmp_path, monkeypatch, extra, lane):
    from winding_field.cli import main
    from winding_field.flattening import (
        matrix_pipeline,
        native_export,
        original_pipeline,
        patches,
    )

    seen = []
    for module, name in (
        (native_export, "native"),
        (matrix_pipeline, "matrix"),
        (patches, "patches"),
        (original_pipeline, "original"),
    ):
        monkeypatch.setattr(
            module, "export_existing", lambda args, name=name: seen.append(name) or {}
        )
    monkeypatch.setattr(native_export, "available", lambda: True)
    from test_native_geometry import plane_bundle

    mesh = plane_bundle(tmp_path)
    source = [] if "uv" in extra else ["--source", "unused"]
    main(["export-sheets", "--mesh-bundle", str(mesh), *source, "--output",
          str(tmp_path / "out"), *extra])  # fmt: skip
    assert seen == [lane]


def test_default_falls_back_to_the_lscm_matrix_where_flatboi_cannot_run(tmp_path, monkeypatch):
    from test_native_geometry import plane_bundle
    from winding_field.cli import main
    from winding_field.flattening import matrix_pipeline, native_export

    seen = []
    monkeypatch.setattr(native_export, "available", lambda: False)
    monkeypatch.setattr(
        matrix_pipeline,
        "export_existing",
        lambda args: seen.append((args.flattening_solver, args.scope)) or {},
    )
    monkeypatch.setattr("winding_field.flattening.flatboi.supported_platform", lambda: False)
    mesh = plane_bundle(tmp_path)
    main(["export-sheets", "--mesh-bundle", str(mesh), "--source", "unused",
          "--output", str(tmp_path / "out")])  # fmt: skip
    assert seen == [("lscm", "sheet")]


def test_matrix_only_settings_are_refused_with_the_lane_to_use(tmp_path, monkeypatch, capsys):
    from test_native_geometry import plane_bundle
    from winding_field.cli import main
    from winding_field.flattening import native_export

    monkeypatch.setattr(native_export, "available", lambda: True)
    mesh = plane_bundle(tmp_path)
    argv = ["export-sheets", "--mesh-bundle", str(mesh), "--source", "unused",
            "--output", str(tmp_path / "out")]  # fmt: skip
    assert main([*argv, "--tile-size", "64"]) == 1
    assert "tile_size" in capsys.readouterr().err
    assert main([*argv, "--output-frame-id", "another/world"]) == 1
    assert "differs from the mesh frame" in capsys.readouterr().err
    assert not (tmp_path / "out" / "SHEETS.json").exists()


def test_an_old_matrix_output_is_refused_cleanly_without_its_flags(tmp_path, monkeypatch, capsys):
    from test_native_geometry import plane_bundle
    from winding_field.cli import main
    from winding_field.flattening import native_export

    monkeypatch.setattr(native_export, "available", lambda: True)
    mesh = plane_bundle(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "EXPORT.json").write_text(json.dumps({"schema": "flattening-matrix-v1"}))
    argv = ["export-sheets", "--mesh-bundle", str(mesh), "--source", "unused",
            "--output", str(out), "--resume"]  # fmt: skip
    assert main(argv) == 1
    assert "--flattening-solver/--scope" in capsys.readouterr().err


def test_then_flatten_chains_the_default_route(tmp_path, monkeypatch, flatboi_app):  # noqa: F811
    from ct_fixture import write_ct
    from test_surface_workflow import prepared
    from winding_field.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WINDING_FLATBOI_APP_RUN", flatboi_app)
    path, _ = prepared(tmp_path)
    z, y, x = np.indices((12, 12, 12))
    ct = write_ct(tmp_path / "ct.zarr", (x + 2 * y + 3 * z).astype(np.uint8))
    settings = tmp_path / "flatten.yaml"
    settings.write_text(f"source: {ct}\nvoxel_um: 1.0\n")
    mesh, flat = tmp_path / "mesh", tmp_path / "flat"
    argv = ["prepared", "mesh", "--config", str(path), "--output", str(mesh)]
    main([*argv, "--then-flatten", str(flat), "--flatten-cli-config", str(settings)])
    chain = json.loads((mesh / "CHAIN.json").read_text())
    step = chain["steps"][1]
    assert step["returncode"] == 0, step.get("stderr_tail")
    assert "--flattening-solver" not in step["command"]
    receipt = json.loads((flat / "SHEETS.json").read_text())
    assert receipt["route"] == "native-sheet" and len(receipt["levels"]) == 2
