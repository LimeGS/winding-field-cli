import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from winding_field.cli import main
from test_prepared_flags import surface_flags, pins
from test_run_sheet_export import saved_run as _saved_run, small_raster as _small_raster

saved_run = _saved_run
small_raster = _small_raster


def frame_flags():
    return [
        "--output-frame-id",
        "test",
        "--output-origin-xyz-um",
        "10",
        "20",
        "30",
        "--output-voxel-um",
        "9.6",
    ]


def resolved(tokens, tmp_path, *, solver=None, scope=None, method=None):
    from winding_field.export_settings import flattening_arguments, flattening_request

    parser = argparse.ArgumentParser()
    parser.add_argument("--flattening-config", type=Path)
    flattening_arguments(parser)
    args = parser.parse_args(tokens)
    args.flattening_solver = solver
    args.scope = scope
    args.flattening_method = method
    request = flattening_request(args)
    if solver:
        from winding_field.flattening_matrix import settings

        return settings(request, solver, scope)
    if method == "patches":
        from winding_field.patch_pipeline import settings
    else:
        from winding_field.original_pipeline import settings
    return settings(request)


@pytest.mark.parametrize("solver", ["lscm", "slim"])
@pytest.mark.parametrize("scope", ["sheet", "patches"])
def test_every_matrix_setting_matches_native_file(tmp_path, solver, scope):
    frame = dict(id="test", origin_xyz_um=[10.0, 20.0, 30.0], voxel_um=9.6)
    request = dict(
        output_frame=frame,
        pitch_um=9.6,
        tile_size=16,
        memory_gib=2.0,
        process_tree_gib=12.0,
        disk_reserve_gib=8.0,
        cache_dir=str(tmp_path / "cache"),
        cache_mib=0,
        cache_gib=0.0,
        ct_workers=2,
        raster_workers=2,
    )
    flags = frame_flags() + [
        "--pitch-um",
        "9.6",
        "--tile-size",
        "16",
        "--memory-gib",
        "2",
        "--process-tree-gib",
        "12",
        "--disk-reserve-gib",
        "8",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--cache-mib",
        "0",
        "--cache-gib",
        "0",
        "--ct-workers",
        "2",
        "--raster-workers",
        "2",
    ]
    if scope == "sheet":
        request.update(centerline=[[1.0, 2.0, 3.0], [1.0, 2.0, 9.0]], reference_z=6.0)
        flags += [
            "--centerline-point",
            "1",
            "2",
            "3",
            "--centerline-point",
            "1",
            "2",
            "9",
            "--reference-z",
            "6",
        ]
    else:
        request.update(max_patch_faces=128, max_charts=256)
        flags += ["--max-patch-faces", "128", "--max-charts", "256"]
    if solver == "slim":
        request.update(app_run=str(tmp_path / "AppRun"), native_recipe="installed-2f89bfc9")
        flags += ["--app-run", str(tmp_path / "AppRun"), "--native-recipe", "installed-2f89bfc9"]
    path = tmp_path / "native.json"
    path.write_text(json.dumps(request))
    assert resolved(flags, tmp_path, solver=solver, scope=scope) == resolved(
        ["--flattening-config", str(path)], tmp_path, solver=solver, scope=scope
    )


def test_historical_patches_all_settings_match_native_file(tmp_path):
    request = dict(
        output_frame=dict(id="test", origin_xyz_um=[10.0, 20.0, 30.0], voxel_um=9.6),
        pitch_um=19.2,
        app_run=str(tmp_path / "AppRun"),
        max_patch_faces=128,
        max_depth=0,
        max_charts=128,
        tile_size=16,
        memory_gib=2.0,
        timeout_seconds=12.0,
        cache_dir=str(tmp_path / "cache"),
        cache_mib=0,
        cache_gib=0.0,
        native_recipe="historical",
        workers=2,
        raster_workers=2,
        ct_workers=2,
        native_cpus=[1, 3],
        reuse_export=str(tmp_path / "reuse"),
        reuse_identity_sha256="a" * 64,
        process_tree_gib=11.0,
        disk_reserve_gib=8.0,
    )
    flags = frame_flags() + [
        "--pitch-um",
        "19.2",
        "--app-run",
        str(tmp_path / "AppRun"),
        "--max-patch-faces",
        "128",
        "--max-depth",
        "0",
        "--max-charts",
        "128",
        "--tile-size",
        "16",
        "--memory-gib",
        "2",
        "--timeout-seconds",
        "12",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--cache-mib",
        "0",
        "--cache-gib",
        "0",
        "--native-recipe",
        "historical",
        "--workers",
        "2",
        "--raster-workers",
        "2",
        "--ct-workers",
        "2",
        "--native-cpus",
        "1",
        "3",
        "--reuse-export",
        str(tmp_path / "reuse"),
        "--reuse-identity-sha256",
        "a" * 64,
        "--process-tree-gib",
        "11",
        "--disk-reserve-gib",
        "8",
    ]
    path = tmp_path / "native.json"
    path.write_text(json.dumps(request))
    assert resolved(flags, tmp_path, method="patches") == resolved(
        ["--flattening-config", str(path)], tmp_path, method="patches"
    )


def test_original_pitch_world_auto_and_explicit_frame_override(tmp_path):
    request = dict(
        output_frame=dict(id="test", origin_xyz_um=[10.0, 20.0, 30.0], voxel_um=9.6),
        pitch=0.25,
        memory_gib=5.0,
    )
    path = tmp_path / "native.json"
    path.write_text(json.dumps(request))
    assert resolved(frame_flags() + ["--pitch", ".25", "--memory-gib", "5"], tmp_path) == resolved(
        ["--flattening-config", str(path)], tmp_path
    )
    result = resolved(
        ["--flattening-config", str(path), "--pitch", "auto", "--output-voxel-um", "2"], tmp_path
    )
    assert result["pitch"] is None and result["output_frame"]["voxel_um"] == 2.0
    assert result["output_frame"]["origin_xyz_um"] == [10.0, 20.0, 30.0]


@pytest.mark.parametrize(
    "extra", [["--pitch", "1"], ["--max-depth", "0"], ["--max-patch-faces", "16"]]
)
def test_wrong_mode_settings_reject_in_existing_validator(tmp_path, extra):
    with pytest.raises(ValueError):
        resolved(frame_flags() + extra, tmp_path, solver="lscm", scope="sheet")


def test_missing_output_frame_is_not_invented(tmp_path):
    assert (
        main(
            [
                "export-sheets",
                "--mesh-bundle",
                "missing",
                "--source",
                "missing",
                "--flattening-method",
                "original",
                "--pitch",
                "1",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "out").exists()


def test_original_export_mesh_bundle_needs_no_user_recipe(saved_run, tmp_path, small_raster):
    from PIL import Image

    native = tmp_path / "native"
    flags = tmp_path / "flags"
    assert (
        main(
            [
                "export-sheets",
                "--run",
                str(saved_run),
                "--levels",
                ".5",
                "--output",
                str(native),
            ]
        )
        == 0
    )
    state = json.loads((saved_run / "RUN.json").read_bytes())
    before = pins(saved_run)
    assert (
        main(
            [
                "export-sheets",
                "--mesh-bundle",
                str(native / "meshes"),
                "--source",
                state["config"]["source"],
                "--flattening-method",
                "original",
                "--output-frame-id",
                "test",
                "--output-origin-xyz-um",
                "0",
                "0",
                "0",
                "--output-voxel-um",
                "1",
                "--output",
                str(flags),
            ]
        )
        == 0
    )
    for name in ("image.png", "mask.png", "tifxyz/x.tif", "tifxyz/y.tif", "tifxyz/z.tif"):
        with (
            Image.open(native / "original/level-0000" / name) as a,
            Image.open(flags / "original/level-0000" / name) as b,
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert before == pins(saved_run)
    kept = pins(flags)
    assert (
        main(
            [
                "export-sheets",
                "--mesh-bundle",
                str(native / "meshes"),
                "--source",
                state["config"]["source"],
                "--flattening-method",
                "original",
                "--output-frame-id",
                "test",
                "--output-origin-xyz-um",
                "0",
                "0",
                "0",
                "--output-voxel-um",
                "1",
                "--output",
                str(flags),
                "--resume",
            ]
        )
        == 0
    )
    after = pins(flags)
    assert {k: v[0] for k, v in kept.items()} == {k: v[0] for k, v in after.items()}
    assert {
        k: v
        for k, v in kept.items()
        if k not in ("original/IDENTITY.json", "original/ORIGINAL.json")
    } == {
        k: v
        for k, v in after.items()
        if k not in ("original/IDENTITY.json", "original/ORIGINAL.json")
    }


def test_saved_run_accepts_frame_flags_and_keeps_saved_ct(saved_run, tmp_path, small_raster):
    out = tmp_path / "flags"
    assert (
        main(
            [
                "export-sheets",
                "--run",
                str(saved_run),
                "--levels",
                ".5",
                "--output",
                str(out),
                "--output-frame-id",
                "override",
                "--output-origin-xyz-um",
                "10",
                "20",
                "30",
                "--output-voxel-um",
                "2",
            ]
        )
        == 0
    )
    cfg = json.loads((out / "flattening.json").read_bytes())
    assert cfg["output_frame"] == dict(
        id="override", origin_xyz_um=[10.0, 20.0, 30.0], voxel_um=2.0
    )


def test_range_flags_preserve_exact_lattice_and_geometry(tmp_path):
    from test_range_surface import recipe, canonical
    from winding_field.range_surface import extract_range

    request = recipe(tmp_path, samples=8)
    extract_range(request, ["-61", "-60.375"], tmp_path / "file")
    flags = surface_flags(
        tmp_path, dict(field=request["field"], geometry=request["geometry"], levels=[])
    )
    flags = flags[:-1]  # ranges use W bounds, not scalar mesh levels
    assert (
        main(
            [
                "extract-range",
                "--output",
                str(tmp_path / "flags"),
                "--w-range",
                "-61",
                "-60.375",
                *flags,
                "--handedness",
                "1",
                "--samples-per-w",
                "8",
                "--w-ref",
                "-60",
                "--theta-ref",
                "0",
                "--umbilicus-point",
                "19",
                "5",
                "19",
                "--umbilicus-point",
                "19",
                "5",
                "29",
            ]
        )
        == 0
    )
    assert canonical(tmp_path / "file") == canonical(tmp_path / "flags")


def test_uvw_flags_match_native_descriptor_outputs(tmp_path):
    from test_uvw_pipeline import inputs
    import tifffile

    args = inputs(tmp_path)
    descriptor = json.loads(Path(args.field_config).read_bytes())
    from winding_field.uvw_pipeline import export_existing

    export_existing(args)
    flags = surface_flags(
        tmp_path, dict(**{k: descriptor[k] for k in ("field", "geometry")}, levels=args.levels)
    )
    out = tmp_path / "flags"
    assert (
        main(
            [
                "export-uvw",
                *flags,
                "--reference-sheet",
                args.reference_sheet,
                "--source",
                args.source,
                "--output",
                str(out),
                "--canvas-samples",
                "7",
                "--workers",
                "1",
                "--cache-mib",
                "256",
            ]
        )
        == 0
    )
    for name in ("x", "y", "z"):
        np.testing.assert_array_equal(
            tifffile.imread(Path(args.output) / f"level-0000/tifxyz/{name}.tif"),
            tifffile.imread(out / f"level-0000/tifxyz/{name}.tif"),
        )
    before = pins(out)
    assert (
        main(
            [
                "export-uvw",
                *flags,
                "--reference-sheet",
                args.reference_sheet,
                "--source",
                args.source,
                "--output",
                str(out),
                "--canvas-samples",
                "7",
                "--workers",
                "1",
                "--cache-mib",
                "256",
                "--resume",
            ]
        )
        == 0
    )
    assert before == pins(out)


def test_uvw_legacy_descriptor_hash_remains_live_after_cli_resolution(tmp_path, monkeypatch):
    from test_uvw_pipeline import inputs
    from winding_field import uvw_pipeline

    args = inputs(tmp_path)
    original = uvw_pipeline.load_reference

    def changed(*a, **kw):
        result = original(*a, **kw)
        path = Path(args.field_config)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(uvw_pipeline, "load_reference", changed)
    assert (
        main(
            [
                "export-uvw",
                "--field-config",
                args.field_config,
                "--reference-sheet",
                args.reference_sheet,
                "--levels",
                "34",
                "35",
                "36",
                "--source",
                args.source,
                "--output",
                args.output,
                "--canvas-samples",
                "7",
            ]
        )
        == 1
    )
    assert not (Path(args.output) / "UVW.json").exists()


def test_partial_field_flags_report_a_configuration_error_without_output(tmp_path):
    assert (
        main(
            [
                "export-sheets",
                "--field-frame",
                "incomplete",
                "--source",
                "missing",
                "--flattening-method",
                "original",
                *frame_flags(),
                "--levels",
                "1",
                "--output",
                str(tmp_path / "bad"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "bad").exists()


def test_original_direct_field_flags_match_file_recipe(saved_run, tmp_path, small_raster):
    from PIL import Image

    baseline = tmp_path / "baseline"
    actual = tmp_path / "flags"
    assert (
        main(
            [
                "export-sheets",
                "--run",
                str(saved_run),
                "--levels",
                ".5",
                "--output",
                str(baseline),
            ]
        )
        == 0
    )
    descriptor = json.loads((baseline / "field.json").read_bytes())
    state = json.loads((saved_run / "RUN.json").read_bytes())
    direct = surface_flags(
        tmp_path, dict(field=descriptor["field"], geometry=descriptor["geometry"], levels=[0.5])
    )
    assert (
        main(
            [
                "export-sheets",
                *direct,
                "--source",
                state["config"]["source"],
                "--flattening-method",
                "original",
                "--output-frame-id",
                descriptor["geometry"]["frame"],
                "--output-origin-xyz-um",
                "0",
                "0",
                "0",
                "--output-voxel-um",
                "1",
                "--output",
                str(actual),
            ]
        )
        == 0
    )
    for name in ("image.png", "mask.png", "tifxyz/x.tif", "tifxyz/y.tif", "tifxyz/z.tif"):
        with (
            Image.open(baseline / "original/level-0000" / name) as a,
            Image.open(actual / "original/level-0000" / name) as b,
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_complete_cli_config_export_paths_are_relative_to_file(
    saved_run, tmp_path, small_raster, monkeypatch
):
    baseline = tmp_path / "baseline"
    assert (
        main(
            ["export-sheets", "--run", str(saved_run), "--levels", ".5", "--output", str(baseline)]
        )
        == 0
    )
    state = json.loads((saved_run / "RUN.json").read_bytes())
    options = dict(
        mesh_bundle="baseline/meshes",
        source=state["config"]["source"],
        flattening_method="original",
        output="from-config",
        output_frame_id="typed-file",
        output_origin_xyz_um=[10.0, 20.0, 30.0],
        output_voxel_um=2.0,
    )
    path = tmp_path / "options.json"
    path.write_text(json.dumps(options))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert main(["export-sheets", "--cli-config", str(path)]) == 0
    receipt = json.loads((tmp_path / "from-config/EXPORT.json").read_bytes())
    assert receipt["settings"]["output_frame"] == {
        "id": "typed-file",
        "origin_xyz_um": [10.0, 20.0, 30.0],
        "voxel_um": 2.0,
    }
    assert receipt["export_tifxyz"] is True
