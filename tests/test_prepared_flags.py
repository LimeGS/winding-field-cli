import json
from pathlib import Path

import numpy as np
import pytest

from test_field_workflow import prepared, sha, stored
from winding_field import field_workflow
from winding_field.util import file_sha


def run_flags(root, config):
    args = [
        "--field-frame",
        config["frame"],
        "--field-voxel-um",
        str(config["voxel_um"]),
        "--sheet-step",
        str(config["sheet_step"]),
        "--calibration",
        config["calibration"],
    ]
    for name in ("origin_xyz", "spacing_xyz", "shape_zyx"):
        args += ["--grid-" + name.replace("_", "-"), *map(str, config["grid"][name])]
    for name, value in {
        **config["observations"],
        **{k: config[k] for k in ("fixed", "constraints", "order") if k in config},
    }.items():
        args += [
            "--" + name,
            str((root / value["path"]).resolve()),
            "--" + name + "-sha256",
            value["sha256"],
        ]
    for section in ("solver", "execution"):
        for name, value in config[section].items():
            args += ["--" + name.replace("_", "-"), str(value)]
    for key, value in config.get("source_identity", {}).items():
        args += ["--source-identity", key, json.dumps(value)]
    return args


def pins(root):
    return {
        str(p.relative_to(root)): (file_sha(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_maximal_flag_request_matches_native_config_science(tmp_path):
    path, config = prepared(tmp_path, fixed=True, constraints=True)
    order = tmp_path / "inputs/order.npy"
    np.save(order, np.array([[0, 0, 0], [0, 0, 1]], dtype="i8"))
    config["order"] = dict(path="inputs/order.npy", sha256=sha(order))
    config["source_identity"] = {
        "instrument": "synthetic",
        "nested": {"channels": [1, False, None]},
    }
    config["solver"].update(device="cpu", preconditioner="jacobi")
    config["execution"].update(
        cache_bytes=0,
        read_workers=1,
        constraint_cache_bytes=0,
        prepared_cache_bytes=0,
        max_bytes=8 << 20,
        reserve_bytes=0,
    )
    path.write_text(json.dumps(config))
    reference = tmp_path / "file"
    actual = tmp_path / "flags"
    field_workflow.run_field(path, reference)
    assert (
        field_workflow.main(["build", "--output", str(actual), *run_flags(tmp_path, config)]) == 0
    )
    a, x, support = stored(reference)
    b, y, valid = stored(actual)
    assert a["identity"] == b["identity"]
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(support, valid)
    metadata = json.loads((actual / "RUN.json").read_bytes())
    assert Path(metadata["config_path"]).is_file()
    assert metadata["identity"]["source_identity"] == config["source_identity"]


def test_prepared_resume_without_user_config_reuses_saved_request(tmp_path):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 0
    before = pins(tmp_path)
    assert field_workflow.main(["resume", "--output", str(root)]) == 0
    after = pins(tmp_path)
    assert {k: v for k, v in before.items() if not k.endswith("/STATUS.json")} == {
        k: v for k, v in after.items() if not k.endswith("/STATUS.json")
    }


def test_verify_does_not_write_generated_request(tmp_path):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 0
    before = pins(tmp_path)
    assert field_workflow.main(["verify", "--output", str(root)]) == 0
    assert before == pins(tmp_path)


def test_changed_request_preserves_retained_outputs(tmp_path):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 0
    before = pins(tmp_path)
    assert field_workflow.main(["resume", "--output", str(root), "--rtol", "1e-3"]) == 2
    assert before == pins(tmp_path)


def test_mixed_native_file_and_flags_keep_provenance_and_overrides(tmp_path):
    path, config = prepared(tmp_path)
    original_sha = sha(path)
    root = tmp_path / "field"
    assert (
        field_workflow.main(
            [
                "build",
                "--config",
                str(path),
                "--output",
                str(root),
                "--cache-bytes",
                "0",
                "--read-workers",
                "2",
            ]
        )
        == 0
    )
    metadata = json.loads((root / "RUN.json").read_bytes())
    saved = Path(metadata["config_path"])
    assert saved != path and metadata["request_sources"] == {str(path): original_sha}
    assert field_workflow.load_run_config(saved)["execution"]["read_workers"] == 2
    assert sha(path) == original_sha
    path.unlink()
    assert field_workflow.main(["resume", "--output", str(root)]) == 0


def test_cli_config_supports_all_prepared_flags_from_its_own_directory(tmp_path, monkeypatch):
    _, config = prepared(tmp_path)
    args = run_flags(tmp_path, config)
    # Keep the independently specified CLI option names, not an argv generated by production code.
    options = {
        "field_frame": config["frame"],
        "field_voxel_um": 8.64,
        "sheet_step": 1.0,
        "calibration": config["calibration"],
        "grid_origin_xyz": [11.0, -4.0, 19.0],
        "grid_spacing_xyz": [2.0, 3.0, 5.0],
        "grid_shape_zyx": [3, 4, 6],
        "chunk": 4,
        "halo": 2,
        "sweeps": 2,
        "dtype": "float64",
        "rtol": 1e-10,
        "maxiter": 1000,
        "output": "from-config",
    }
    assert "--target" in args
    for name in ("target", "weights", "observed"):
        options[name] = config["observations"][name]["path"]
        options[name + "_sha256"] = config["observations"][name]["sha256"]
    file = tmp_path / "options.json"
    file.write_text(json.dumps(options))
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert field_workflow.main(["build", "--cli-config", str(file)]) == 0
    assert (tmp_path / "from-config/CHECKPOINT.json").is_file()


def test_gpu_lease_request_is_validated_without_acquiring_gpu(tmp_path):
    _, config = prepared(tmp_path)
    config["solver"]["device"] = "cuda:0"
    config["gpu"] = dict(
        index=0, uuid="GPU-synthetic", lock_path="lease.lock", min_free_bytes=0, deadline=4e9
    )
    validated = field_workflow.validate_run_config(config, base=tmp_path)
    assert validated["gpu"]["lock_path"] == str(tmp_path / "lease.lock")
    assert not (tmp_path / "lease.lock").exists()
    assert config["gpu"]["lock_path"] == "lease.lock"


def test_supervised_flag_request_survives_a_changed_working_directory(tmp_path, monkeypatch):
    _, config = prepared(tmp_path)
    root = tmp_path / "supervised"
    args = run_flags(tmp_path, config)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert field_workflow.main(["build", "--output", str(root), "--supervised", *args]) == 0
    saved = json.loads((root / "RUN.json").read_bytes())
    assert Path(saved["config_path"]).is_file()
    cp, values, valid = stored(root)
    assert [entry["sweep"] for entry in cp["progress"]["completed_sweeps"]] == [1, 2]
    np.testing.assert_allclose(
        values, np.broadcast_to(np.arange(6) * 0.25, values.shape), atol=2e-7
    )
    assert valid.all()


@pytest.mark.parametrize("failure", ["missing", "mutated"])
def test_retained_recipe_cannot_be_replaced_or_recreated_on_resume(tmp_path, failure):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 0
    recipe = Path(json.loads((root / "RUN.json").read_bytes())["config_path"])
    if failure == "missing":
        recipe.unlink()
    else:
        recipe.write_text("{}")
    before = pins(tmp_path)
    assert field_workflow.main(["resume", "--output", str(root)]) == 2
    assert before == pins(tmp_path)


def test_saved_recipe_cannot_initialize_a_different_output(tmp_path):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 0
    recipe = Path(json.loads((root / "RUN.json").read_bytes())["config_path"])
    before = pins(tmp_path)
    assert (
        field_workflow.main(
            ["build", "--output", str(tmp_path / "foreign"), "--config", str(recipe)]
        )
        == 2
    )
    assert before == pins(tmp_path)


def test_failed_input_admission_does_not_leave_a_recipe(tmp_path):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    config["observations"]["target"]["sha256"] = "a" * 64
    before = pins(tmp_path)
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 2
    assert before == pins(tmp_path)


def test_resource_only_continuation_keeps_scientific_checkpoint(tmp_path):
    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    assert field_workflow.main(["build", "--output", str(root), *run_flags(tmp_path, config)]) == 0
    checkpoint = (sha(root / "CHECKPOINT.json"), (root / "CHECKPOINT.json").stat().st_mtime_ns)
    assert (
        field_workflow.main(
            ["resume", "--output", str(root), "--read-workers", "2", "--cache-bytes", "1048576"]
        )
        == 0
    )
    assert checkpoint == (
        sha(root / "CHECKPOINT.json"),
        (root / "CHECKPOINT.json").stat().st_mtime_ns,
    )
    requests = [field_workflow.load_run_config(p) for p in tmp_path.glob("field.request-*.json")]
    assert any(
        r["execution"]["read_workers"] == 2 and r["execution"]["cache_bytes"] == 1048576
        for r in requests
    )


def test_mixed_recipe_reaches_supervised_worker_with_provenance(tmp_path):
    path, _ = prepared(tmp_path)
    root = tmp_path / "field"
    assert (
        field_workflow.main(
            [
                "build",
                "--config",
                str(path),
                "--output",
                str(root),
                "--supervised",
                "--read-workers",
                "2",
                "--cache-bytes",
                "0",
            ]
        )
        == 0
    )
    metadata = json.loads((root / "RUN.json").read_bytes())
    assert metadata["request_sources"] == {str(path): sha(path)}
    assert field_workflow.load_run_config(metadata["config_path"])["execution"]["read_workers"] == 2


def test_gpu_flags_survive_validation_before_resource_denial(tmp_path, monkeypatch):
    from field_runtime import GpuLease, ResourceUnavailable

    _, config = prepared(tmp_path)
    root = tmp_path / "field"
    config["solver"]["device"] = "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic")

    def unavailable(self):
        raise ResourceUnavailable("test has no hardware lease")

    monkeypatch.setattr(GpuLease, "__enter__", unavailable)
    assert (
        field_workflow.main(
            [
                "build",
                "--output",
                str(root),
                *run_flags(tmp_path, config),
                "--gpu-index",
                "3",
                "--gpu-uuid",
                "GPU-synthetic",
                "--gpu-lock-path",
                str(tmp_path / "lease.lock"),
                "--gpu-min-free-bytes",
                "0",
                "--gpu-deadline",
                "4000000000",
            ]
        )
        == 75
    )
    recipe = next(tmp_path.glob("field.request-*.json"))
    assert field_workflow.load_run_config(recipe)["gpu"] == {
        "index": 3,
        "uuid": "GPU-synthetic",
        "lock_path": str(tmp_path / "lease.lock"),
        "min_free_bytes": 0,
        "deadline": 4e9,
    }
    assert not (tmp_path / "lease.lock").exists()


def test_resume_missing_run_metadata_reports_input_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert field_workflow.main(["resume", "--output", str(root)]) == 2


def surface_flags(root, config):
    args = []
    for name, value in config["geometry"].items():
        flag = "--" + (
            name.replace("_", "-")
            if name in ("sheet_step", "calibration")
            else "field-" + name.replace("_", "-")
        )
        args += [flag, *(map(str, value) if isinstance(value, list) else [str(value)])]
    if "field" in config:
        for name, value in config["field"].items():
            if name == "mask":
                args += [
                    "--field-mask",
                    str(root / value["path"]),
                    "--field-mask-sha256",
                    value["sha256"],
                ]
            else:
                if name == "path":
                    value = str(root / value)
                args += ["--field-" + name.replace("_", "-"), str(value)]
        args += ["--levels", *map(str, config["levels"])]
    else:
        args += [
            "--mesh",
            str(root / config["mesh"]["path"]),
            "--mesh-sha256",
            config["mesh"]["sha256"],
        ]
    for name, value in config.get("execution", {}).items():
        args += ["--" + name.replace("_", "-"), str(value)]
    for name, value in config.get("render", {}).items():
        if name == "views":
            for view in value:
                args += [
                    "--render-view",
                    view["name"],
                    str(view["azimuth"]),
                    str(view["elevation"]),
                ]
        else:
            args += [
                "--render-" + name.replace("_", "-"),
                *(map(str, value) if isinstance(value, list) else [str(value)]),
            ]
    return args


def test_local_surface_flags_match_native_science_and_mask(tmp_path):
    from test_surface_workflow import prepared as surface_prepared
    from winding_field import surface_workflow

    path, config = surface_prepared(tmp_path)
    mask = tmp_path / "mask.npy"
    np.save(mask, np.ones((5, 6, 7), bool))
    config["field"].update(mask=dict(path="mask.npy", sha256=sha(mask)))
    config["geometry"]["calibration"] = "synthetic raw phi calibration"
    config["execution"].update(cache_bytes=0, reserve_bytes=0, mesh_bytes=8 << 20)
    config["render"].update(triangle_batch=64, reserve_bytes=0, levels=[3.5])
    path.write_text(json.dumps(config))
    surface_workflow.run_surfaces(path, tmp_path / "file")
    assert (
        surface_workflow.main(
            ["surfaces", "--output", str(tmp_path / "flags"), *surface_flags(tmp_path, config)]
        )
        == 0
    )
    for index in (0, 1):
        assert (tmp_path / f"file/level-{index:04d}.obj").read_bytes() == (
            tmp_path / f"flags/level-{index:04d}.obj"
        ).read_bytes()
    a = json.loads((tmp_path / "file/MESH.json").read_bytes())
    b = json.loads((tmp_path / "flags/MESH.json").read_bytes())
    assert a["identity"] == b["identity"] and a["levels"] == b["levels"]
    for view in ("front", "oblique"):
        assert (tmp_path / f"file/renders/{view}.png").read_bytes() == (
            tmp_path / f"flags/renders/{view}.png"
        ).read_bytes()
    before = pins(tmp_path)
    assert surface_workflow.main(["surfaces", "--resume", "--output", str(tmp_path / "flags")]) == 0
    after = pins(tmp_path)
    assert {k: v[0] for k, v in before.items()} == {k: v[0] for k, v in after.items()}
    assert {
        k: v for k, v in before.items() if k not in ("flags/STATUS.json", "flags/MESH.json")
    } == {k: v for k, v in after.items() if k not in ("flags/STATUS.json", "flags/MESH.json")}


def test_imported_mesh_flags_preserve_coordinates_and_components(tmp_path):
    from winding_field import surface_workflow

    mesh = tmp_path / "input.obj"
    mesh.write_text("v 0 0 0\nv 0 2 0\nv 0 0 2\nv 3 0 0\nv 3 2 0\nv 3 0 2\nf 1 2 3\nf 4 5 6\n")
    config = dict(
        geometry=dict(frame="mesh/world", voxel_um=9.6),
        mesh=dict(path="input.obj", sha256=sha(mesh)),
    )
    assert (
        surface_workflow.main(
            ["mesh", "--output", str(tmp_path / "flags"), *surface_flags(tmp_path, config)]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "flags/MESH.json").read_bytes())
    assert manifest["metadata"] == {"frame": "mesh/world", "voxel_um": 9.6}
    assert manifest["levels"][0]["components"] == 2 and manifest["levels"][0]["faces"] == 2


def test_paris4_w_minus60_keeps_raw_phi_and_9_6um(tmp_path):
    from winding_field import surface_workflow

    values = np.broadcast_to(np.arange(-75.0, -68.0), (3, 4, 7)).copy()
    source = tmp_path / "phi.npy"
    np.save(source, values)
    config = dict(
        field=dict(kind="npy", path="phi.npy", sha256=sha(source)),
        geometry=dict(
            frame="paris4/world",
            voxel_um=9.6,
            origin_xyz=[11.0, -4.0, 19.0],
            spacing_xyz=[32.0, 32.0, 32.0],
            shape_zyx=[3, 4, 7],
            sheet_step=1.192497988541921,
        ),
        levels=[-71.54987931251526],
    )
    assert (
        surface_workflow.main(
            ["mesh", "--output", str(tmp_path / "flags"), *surface_flags(tmp_path, config)]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "flags/MESH.json").read_bytes())
    assert manifest["metadata"]["voxel_um"] == 9.6
    assert manifest["identity"]["levels"] == [-71.54987931251526]
    from winding_field.surface_mesh import MeshReader

    with MeshReader(tmp_path / "flags") as mesh:
        triangles = list(mesh.triangles())[0][0]
        np.testing.assert_allclose(triangles[..., 0], 121.40386199951172, atol=1e-10)


def test_parallel_levels_reuses_generated_recipe_from_other_cwd(tmp_path, monkeypatch):
    from test_surface_workflow import prepared as surface_prepared
    from winding_field import surface_workflow

    _, config = surface_prepared(tmp_path)
    flags = surface_flags(tmp_path, config)
    root = tmp_path / "parallel"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert surface_workflow.main(["mesh", "--parallel-levels", "--output", str(root), *flags]) == 0
    binding = json.loads((root / "LEVELS.json").read_bytes())["binding"]
    assert Path(binding["config_path"]).is_file()
    assert json.loads((root / "LEVELS.json").read_bytes())["status"] == "LEVELS_READY"
    before = pins(tmp_path)
    assert (
        surface_workflow.main(["mesh", "--parallel-levels", "--resume", "--output", str(root)]) == 0
    )
    assert before == pins(tmp_path)


def test_render_flags_do_not_require_full_surface_config(tmp_path):
    from test_surface_workflow import prepared as surface_prepared
    from winding_field import surface_workflow

    path, _ = surface_prepared(tmp_path)
    surface_workflow.run_surfaces(path, tmp_path / "mesh", render=False)
    assert (
        surface_workflow.main(
            [
                "render",
                "--mesh",
                str(tmp_path / "mesh"),
                "--output",
                str(tmp_path / "render"),
                "--render-width",
                "32",
                "--render-height",
                "24",
                "--render-view",
                "top",
                "90",
                "45",
                "--render-triangle-batch",
                "64",
                "--render-levels",
                "1.5",
                "--render-reserve-bytes",
                "0",
            ]
        )
        == 0
    )
    from PIL import Image

    assert Image.open(tmp_path / "render/top.png").size == (32, 24)
    render = json.loads((tmp_path / "render/RENDER.json").read_bytes())
    assert render["settings"]["levels"] == [1.5]


def test_remote_surface_flags_use_both_http_pins(tmp_path, monkeypatch):
    from test_remote_field import export_fixture
    from winding_field import surface_workflow

    _, descriptor, geometry, *_ = export_fixture(monkeypatch)
    config = dict(
        field=descriptor,
        geometry=geometry,
        levels=[1.5, 3.5],
        execution=dict(tile=2, level_batch=1),
    )
    path = tmp_path / "remote.json"
    path.write_text(json.dumps(dict(schema="winding-surfaces-v1", **config)))
    surface_workflow.run_surfaces(path, tmp_path / "file", render=False)
    assert (
        surface_workflow.main(
            ["mesh", "--output", str(tmp_path / "flags"), *surface_flags(tmp_path, config)]
        )
        == 0
    )
    for i in (0, 1):
        assert (tmp_path / f"file/level-{i:04d}.obj").read_bytes() == (
            tmp_path / f"flags/level-{i:04d}.obj"
        ).read_bytes()


@pytest.mark.parametrize("failure", ["missing", "mutated", "conflict", "changed_resume"])
def test_surface_bad_pins_and_conflicts_do_not_write(tmp_path, failure):
    from test_surface_workflow import prepared as surface_prepared
    from winding_field import surface_workflow

    _, config = surface_prepared(tmp_path)
    root = tmp_path / "flags"
    if failure == "changed_resume":
        assert (
            surface_workflow.main(["mesh", "--output", str(root), *surface_flags(tmp_path, config)])
            == 0
        )
        flags = ["--resume", "--levels", "2.5"]
    else:
        if failure == "missing":
            config["field"].pop("sha256")
        if failure == "mutated":
            config["field"]["sha256"] = "a" * 64
        flags = surface_flags(tmp_path, config)
        if failure == "conflict":
            flags += [
                "--mesh",
                str(tmp_path / "phi.npy"),
                "--mesh-sha256",
                sha(tmp_path / "phi.npy"),
            ]
    before = pins(tmp_path)
    assert surface_workflow.main(["mesh", "--output", str(root), *flags]) == 2
    assert before == pins(tmp_path)


def test_surface_cli_config_paths_and_explicit_override(tmp_path, monkeypatch):
    from test_surface_workflow import prepared as surface_prepared
    from winding_field import surface_workflow

    _, config = surface_prepared(tmp_path)
    options = dict(
        field_kind="npy",
        field_path="phi.npy",
        field_sha256=config["field"]["sha256"],
        field_frame="other-scroll/world",
        field_voxel_um=3.0,
        field_origin_xyz=[17.0, -4.0, 29.0],
        field_spacing_xyz=[2.0, 3.0, 5.0],
        field_shape_zyx=[5, 6, 7],
        sheet_step=2.0,
        levels=[1.5, 3.5],
        output="from-file",
        tile=2,
    )
    path = tmp_path / "cli.json"
    path.write_text(json.dumps(options))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert surface_workflow.main(["mesh", "--cli-config", str(path), "--levels", "2.5"]) == 0
    manifest = json.loads((tmp_path / "from-file/MESH.json").read_bytes())
    assert [entry["level"] for entry in manifest["levels"]] == [2.5]


def test_render_rejects_unrelated_field_settings_before_writes(tmp_path):
    from test_surface_workflow import prepared as surface_prepared
    from winding_field import surface_workflow

    path, _ = surface_prepared(tmp_path)
    surface_workflow.run_surfaces(path, tmp_path / "mesh", render=False)
    before = pins(tmp_path)
    assert (
        surface_workflow.main(
            [
                "render",
                "--mesh",
                str(tmp_path / "mesh"),
                "--output",
                str(tmp_path / "render"),
                "--field-frame",
                "wrong-mode",
            ]
        )
        == 2
    )
    assert before == pins(tmp_path)


@pytest.mark.parametrize("family", ["field", "surface"])
def test_explicit_config_does_not_recreate_a_missing_retained_snapshot(tmp_path, family):
    if family == "field":
        path, _ = prepared(tmp_path)
        root = tmp_path / "field"
        owner = field_workflow
        assert owner.main(["build", "--config", str(path), "--output", str(root)]) == 0
        metadata = json.loads((root / "RUN.json").read_bytes())
        action = "resume"
        extra = []
    else:
        from test_surface_workflow import prepared as surface_prepared
        from winding_field import surface_workflow

        path, config = surface_prepared(tmp_path)
        root = tmp_path / "surface"
        owner = surface_workflow
        assert owner.main(["mesh", "--output", str(root), *surface_flags(tmp_path, config)]) == 0
        metadata = json.loads((root / "REQUEST.json").read_bytes())
        action = "mesh"
        extra = ["--resume"]
    Path(metadata["config_path"]).unlink()
    before = pins(tmp_path)
    assert owner.main([action, "--config", str(path), "--output", str(root), *extra]) == 2
    assert before == pins(tmp_path)
