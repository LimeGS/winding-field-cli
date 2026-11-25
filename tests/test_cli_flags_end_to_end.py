"""Public entry-point parity, real tiny artifacts and recovery boundaries."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from winding_field.cli import main, TOOLS
from winding_field.util import atomic_json, file_sha
from test_prepared_flags import surface_flags, pins


ROOT_COMMANDS = {
    "prepared",
    *TOOLS,
    "inspect",
    "run",
    "plan",
    "resume",
    "status",
    "prepare",
    "fibres",
    "field",
    "mesh",
    "flatten",
    "assemble",
    "render",
    "tifxyz",
    "verify-tifxyz",
    "extract-range",
    "join-ranges",
    "export-sheets",
    "export-uvw",
    "demo",
    "doctor",
    "models",
}
NESTED = (
    [("prepared", name) for name in ("build", "resume", "verify", "mesh", "surfaces", "render")]
    + [
        ("constant-sheet", name)
        for name in ("prepare", "flatten", "import-native", "validate", "render", "status")
    ]
    + [("sheet-atlas", name) for name in ("build", "query", "verify")]
)


@pytest.mark.parametrize("command", [(c,) for c in sorted(ROOT_COMMANDS)] + NESTED)
def test_public_dispatch_exposes_typed_config_entry(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([*command, "--help"])
    assert error.value.code == 0
    assert "--cli-config" in capsys.readouterr().out


def test_root_inventory_has_no_unqualified_public_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    help_text = capsys.readouterr().out
    choices = help_text.split("{", 1)[1].split("}", 1)[0].split(",")
    assert set(choices) == ROOT_COMMANDS


def test_numeric_render_view_json_rows_reach_real_renderer(tmp_path):
    from test_surface_workflow import prepared

    path, _ = prepared(tmp_path)
    assert (
        main(["prepared", "mesh", "--config", str(path), "--output", str(tmp_path / "mesh")]) == 0
    )
    options = tmp_path / "render.json"
    atomic_json(
        options,
        dict(
            mesh="mesh",
            output="render",
            render_view=[["top", 90, 45.5]],
            render_width=24,
            render_height=16,
            render_reserve_bytes=0,
        ),
    )
    assert main(["prepared", "render", "--cli-config", str(options)]) == 0
    receipt = json.loads((tmp_path / "render/RENDER.json").read_bytes())
    assert receipt["settings"]["views"] == [dict(name="top", azimuth=90.0, elevation=45.5)]


@pytest.mark.parametrize("value", [None, "auto"])
def test_cli_config_nullable_pitch_is_same_as_named_auto_flag(tmp_path, value):
    import argparse
    from winding_field.cli_settings import parse_options
    from winding_field.export_settings import flattening_arguments, flattening_request

    parser = argparse.ArgumentParser()
    flattening_arguments(parser)
    path = tmp_path / "cli.json"
    atomic_json(path, dict(pitch=value))
    args = parse_options(parser, ["--config", str(path)])
    assert flattening_request(args)["pitch"] is None
    atomic_json(path, dict(pitch=1.25))
    assert flattening_request(parse_options(parser, ["--config", str(path)]))["pitch"] == 1.25


def test_null_field_path_reports_input_error_without_traceback_or_output(tmp_path):
    options = tmp_path / "options.json"
    atomic_json(options, dict(field_path=None, output="bad"))
    try:
        result = main(["prepared", "mesh", "--cli-config", str(options)])
    except SystemExit as error:
        result = error.code
    assert result == 2 and not (tmp_path / "bad").exists()


def test_named_optional_field_artifact_clear_overrides_native_file(tmp_path):
    from test_field_workflow import prepared
    from winding_field.field_workflow import load_run_config

    path, _ = prepared(tmp_path, fixed=True, constraints=True)
    assert (
        main(
            [
                "prepared",
                "build",
                "--config",
                str(path),
                "--no-fixed",
                "--no-constraints",
                "--output",
                str(tmp_path / "field"),
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "field/RUN.json").read_bytes())
    request = load_run_config(saved["config_path"])
    assert "fixed" not in request and "constraints" not in request


@pytest.mark.parametrize("remote", [False, True])
def test_raw_paris4_style_mesh_matches_file_and_completed_resume_is_bound(
    tmp_path, monkeypatch, remote
):
    """No user JSON in the flags route; the remote fixture substitutes only HTTP."""
    if remote:
        from test_remote_field import export_fixture

        _, field, geometry, *_ = export_fixture(monkeypatch)
        levels = [1.5]
    else:
        values = np.broadcast_to(np.arange(-75.0, -68.0), (3, 4, 7)).copy()
        path = tmp_path / "field with spaces.npy"
        np.save(path, values)
        field = dict(kind="npy", path=str(path), sha256=file_sha(path))
        geometry = dict(
            frame="paris4/world",
            voxel_um=9.6,
            sheet_step=1.192497988541921,
            origin_xyz=[11.0, -4.0, 19.0],
            spacing_xyz=[32.0] * 3,
            shape_zyx=[3, 4, 7],
        )
        levels = [-71.54987931251526]
    request = dict(schema="winding-surfaces-v1", field=field, geometry=geometry, levels=levels)
    native = tmp_path / "native.json"
    atomic_json(native, request)
    assert (
        main(["prepared", "mesh", "--config", str(native), "--output", str(tmp_path / "file")]) == 0
    )
    flags = surface_flags(tmp_path, request)
    actual = tmp_path / "flag output"
    assert main(["prepared", "mesh", *flags, "--output", str(actual)]) == 0
    assert (actual / "level-0000.obj").read_bytes() == (
        tmp_path / "file/level-0000.obj"
    ).read_bytes()
    native.unlink()
    assert main(["prepared", "mesh", "--resume", "--output", str(actual)]) == 0
    for changed in (
        ["--levels", "-60"],
        ["--field-frame", "different"],
        ["--field-sha256", "a" * 64],
    ):
        before = pins(actual)
        assert main(["prepared", "mesh", "--resume", "--output", str(actual), *changed]) == 2
        assert before == pins(actual)


def test_flag_mesh_interrupted_after_admission_resumes_saved_request_from_other_cwd(
    tmp_path, monkeypatch
):
    from test_surface_workflow import prepared
    from winding_field import surface_workflow

    _, config = prepared(tmp_path)
    config["execution"]["tile"] = 1
    output = tmp_path / "interrupted mesh"
    original = surface_workflow.run_surfaces
    calls = 0

    def stopped(config_path, root, **kwargs):
        def stop():
            nonlocal calls
            calls += 1
            return calls > 1

        return original(config_path, root, stop=stop, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(surface_workflow, "run_surfaces", stopped)
        assert (
            main(["prepared", "mesh", *surface_flags(tmp_path, config), "--output", str(output)])
            == 0
        )
    assert json.loads((output / "STATUS.json").read_bytes())["status"] == "CHECKPOINTED_STOP"
    recipe = Path(json.loads((output / "REQUEST.json").read_bytes())["config_path"])
    pin = (file_sha(recipe), recipe.stat().st_mtime_ns)
    elsewhere = tmp_path / "other cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["prepared", "mesh", "--resume", "--output", str(output)]) == 0
    assert json.loads((output / "MESH.json").read_bytes())["status"] == "MESH_READY"
    assert pin == (file_sha(recipe), recipe.stat().st_mtime_ns)


def test_surface_snapshot_uses_same_canonical_output_on_resume(tmp_path):
    from test_surface_workflow import prepared

    _, config = prepared(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / ".." / "mesh"
    assert (
        main(["prepared", "mesh", *surface_flags(tmp_path, config), "--output", str(output)]) == 0
    )
    assert main(["prepared", "mesh", "--resume", "--output", str(output)]) == 0
    assert json.loads((output / "MESH.json").read_bytes())["status"] == "MESH_READY"


@pytest.mark.parametrize("family", ["field", "surface"])
def test_prepared_native_defaults_validate_after_explicit_overrides(tmp_path, monkeypatch, family):
    base = tmp_path / "recipe base"
    base.mkdir()
    if family == "field":
        from test_field_workflow import prepared

        path, config = prepared(base)
        config["solver"]["rtol"] = 0
        overrides = ["--rtol", "1e-8"]
        action = "build"
    else:
        from test_surface_workflow import prepared

        path, config = prepared(base)
        del config["levels"]
        overrides = ["--levels", "1.5"]
        action = "mesh"
    atomic_json(path, config)
    pin = file_sha(path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "output"
    assert (
        main(["prepared", action, "--config", str(path), *overrides, "--output", str(output)]) == 0
    )
    assert file_sha(path) == pin
    if family == "field":
        saved = json.loads((output / "RUN.json").read_bytes())
        assert saved["request_sources"] == {str(path): pin}
    else:
        assert json.loads((output / "MESH.json").read_bytes())["levels"][0]["level"] == 1.5


def test_max_bytes_auto_clears_inherited_prepared_memory_limit(tmp_path):
    from test_field_workflow import prepared

    path, config = prepared(tmp_path)
    config["execution"]["max_bytes"] = 1
    atomic_json(path, config)
    assert (
        main(
            [
                "prepared",
                "build",
                "--config",
                str(path),
                "--max-bytes",
                "auto",
                "--output",
                str(tmp_path / "field"),
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "field/RUN.json").read_bytes())
    request = json.loads(Path(saved["config_path"]).read_bytes())["request"]
    assert request["execution"]["max_bytes"] is None


def test_yaml_cli_recipe_runs_real_prepared_mesh_and_resume_without_yaml_file(
    tmp_path, monkeypatch
):
    import yaml

    source = tmp_path / "phi.npy"
    np.save(source, np.indices((3, 4, 5))[2].astype(float))
    options = tmp_path / "options.yml"
    options.write_text(
        yaml.safe_dump(
            dict(
                field_kind="npy",
                field_path="phi.npy",
                field_sha256=file_sha(source),
                field_frame="test/world",
                field_voxel_um=9.6,
                field_origin_xyz=[10.0, 20.0, 30.0],
                field_spacing_xyz=[2.0, 3.0, 4.0],
                field_shape_zyx=[3, 4, 5],
                sheet_step=1.0,
                levels=[1.5],
                tile=2,
                output="mesh output",
            )
        )
    )
    pin = file_sha(options)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["prepared", "mesh", "--cli-config", str(options)]) == 0
    output = tmp_path / "mesh output"
    manifest = json.loads((output / "MESH.json").read_bytes())
    assert manifest["metadata"]["voxel_um"] == 9.6 and manifest["levels"][0]["level"] == 1.5
    recipe = Path(json.loads((output / "REQUEST.json").read_bytes())["config_path"])
    assert json.loads(recipe.read_bytes())["sources"] == {str(options): pin}
    options.unlink()
    assert main(["prepared", "mesh", "--resume", "--output", str(output)]) == 0


@pytest.mark.skipif(
    sys.platform != "linux", reason="whole-sheet LSCM process-tree resource guard requires Linux"
)
def test_prepared_mesh_to_whole_sheet_lscm_ct_xyz_without_user_json(tmp_path):
    from test_flattening_matrix import source_bundle, request
    from winding_field.flattening.patch_mapping import PatchMapping

    _, ct = source_bundle(tmp_path)
    values = np.broadcast_to(np.arange(7.0), (7, 7, 7)).copy()
    source = tmp_path / "phi.npy"
    np.save(source, values)
    geometry = dict(
        frame="test",
        voxel_um=9.6,
        origin_xyz=[4.0] * 3,
        spacing_xyz=[1.0] * 3,
        shape_zyx=[7] * 3,
        sheet_step=1.0,
    )
    descriptor = dict(
        field=dict(kind="npy", path=str(source), sha256=file_sha(source)),
        geometry=geometry,
        levels=[3.5],
    )
    mesh = tmp_path / "prepared-mesh"
    assert (
        main(["prepared", "mesh", *surface_flags(tmp_path, descriptor), "--output", str(mesh)]) == 0
    )
    settings = tmp_path / "flattening.json"
    atomic_json(settings, request(tmp_path))
    common = [
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
    ]
    assert (
        main([*common, "--flattening-config", str(settings), "--output", str(tmp_path / "file")])
        == 0
    )
    settings.unlink()
    flags = [
        "--output-frame-id",
        "test",
        "--output-origin-xyz-um",
        "0",
        "0",
        "0",
        "--output-voxel-um",
        "9.6",
        "--pitch-um",
        "9.6",
        "--tile-size",
        "16",
        "--memory-gib",
        "2",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    actual = tmp_path / "flags"
    assert main([*common, *flags, "--output", str(actual)]) == 0
    with PatchMapping(actual / "level-0000") as a, PatchMapping(tmp_path / "file/level-0000") as b:
        for name in a.arrays:
            np.testing.assert_array_equal(a.arrays[name], b.arrays[name])
        np.testing.assert_array_equal(a.arrays["field_xyz"][:, 0], 7.5)
    before = pins(actual)
    assert main([*common, *flags, "--output", str(actual), "--resume"]) == 0
    assert before == pins(actual)
    assert main([*common, *flags, "--level", "1", "--output", str(actual), "--resume"]) == 1
    assert before == pins(actual)


def test_whole_sheet_fixture_reaches_export_with_a_fresh_prepared_mesh(tmp_path, monkeypatch):
    """Run the actual mesh stage even on hosts without the Linux export guard."""
    real_main = main

    class ExportReached(Exception):
        pass

    def mesh_then_export(argv):
        if argv[0] == "export-sheets":
            mesh = Path(argv[argv.index("--mesh-bundle") + 1])
            assert (mesh / "MESH.json").is_file()
            raise ExportReached
        return real_main(argv)

    monkeypatch.setitem(globals(), "main", mesh_then_export)
    with pytest.raises(ExportReached):
        test_prepared_mesh_to_whole_sheet_lscm_ct_xyz_without_user_json(tmp_path)
