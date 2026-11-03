import hashlib
import importlib.util
import itertools
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def api():
    assert importlib.util.find_spec("winding_field.surface_workflow"), "surface pipeline missing"
    from winding_field import surface_workflow

    return surface_workflow


def prepared(tmp_path):
    values = np.broadcast_to(np.arange(7.0), (5, 6, 7)).copy()
    p = tmp_path / "phi.npy"
    np.save(p, values)
    config = dict(
        schema="winding-surfaces-v1",
        field=dict(kind="npy", path="phi.npy", sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        geometry=dict(
            frame="other-scroll/world",
            voxel_um=3.0,
            sheet_step=2.0,
            origin_xyz=[17.0, -4.0, 29.0],
            spacing_xyz=[2.0, 3.0, 5.0],
            shape_zyx=list(values.shape),
        ),
        levels=[1.5, 3.5],
        execution=dict(tile=2, level_batch=1),
        render=dict(
            width=64,
            height=48,
            views=[dict(name="front", azimuth=0, elevation=0), dict(name="oblique", azimuth=35, elevation=20)],
        ),
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config))
    return path, config


def test_real_pipeline_mesh_obj_and_visible_png_keep_declared_frame(tmp_path):
    path, config = prepared(tmp_path)
    result = api().run_surfaces(path, tmp_path / "output")
    assert result["status"] == "SURFACES_READY" and result["physical_acceptance"] is False
    mesh = json.loads((tmp_path / "output/MESH.json").read_text())
    assert mesh["metadata"] == config["geometry"]
    image = np.asarray(Image.open(tmp_path / "output/renders/front.png"))
    assert image.shape == (48, 64, 4) and image[:, :, 3].any()
    assert (tmp_path / "output/level-0001.obj").is_file()
    render = json.loads((tmp_path / "output/renders/RENDER.json").read_text())
    assert render["metadata"]["frame"] == "other-scroll/world"


def test_pipeline_stop_then_resume_and_render_existing_bundle(tmp_path):
    path, _ = prepared(tmp_path)
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 2

    assert api().run_surfaces(path, tmp_path / "output", stop=stop)["status"] == "CHECKPOINTED_STOP"
    assert not (tmp_path / "output/renders").exists()
    assert api().run_surfaces(path, tmp_path / "output", resume=True)["status"] == "SURFACES_READY"
    result = api().render_bundle(tmp_path / "output", tmp_path / "new-render", settings=dict(width=32, height=32))
    assert result["status"] == "RENDERED_DIAGNOSTIC" and (tmp_path / "new-render/front.png").is_file()


@pytest.mark.parametrize(
    "change",
    [
        dict(schema="bad"),
        dict(unknown=True),
        dict(levels=[]),
        dict(levels=[True]),
        dict(execution=dict(tile=0)),
        dict(execution=dict(unknown=1)),
        dict(render=dict(width=True)),
        dict(render=dict(width=100000, height=100000)),
        dict(render=dict(views=[dict(name="../escape", azimuth=0)])),
        dict(render=dict(views=[dict(name="same"), dict(name="same")])),
    ],
)
def test_invalid_config_leaves_no_output(tmp_path, change):
    path, config = prepared(tmp_path)
    config.update(change)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        api().run_surfaces(path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("change", ["hash", "geometry", "levels"])
def test_changed_input_cannot_replace_committed_mesh(tmp_path, change):
    path, config = prepared(tmp_path)
    api().run_surfaces(path, tmp_path / "output", stop=lambda: True)
    db = tmp_path / "output/mesh.sqlite"
    before = db.read_bytes()
    if change == "hash":
        config["field"]["sha256"] = "f" * 64
    elif change == "levels":
        config["levels"] = [2.5]
    else:
        config["geometry"]["frame"] = "changed"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        api().run_surfaces(path, tmp_path / "output", resume=True)
    assert db.read_bytes() == before


@pytest.mark.parametrize("kind", ["obj", "npz"])
def test_existing_mesh_import_preserves_two_components_and_renders(tmp_path, kind):
    path, config = prepared(tmp_path)
    vertices = np.array([[0.0, 0, 0], [0, 2, 0], [0, 0, 2], [3, 0, 0], [3, 2, 0], [3, 0, 2]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = tmp_path / ("source." + kind)
    if kind == "obj":
        mesh.write_text("".join(f"v {p[0]} {p[1]} {p[2]}\n" for p in vertices) + "f 1 2 3\nf -3 -2 -1\n")
    else:
        np.savez(mesh, vertices_xyz=vertices, faces=faces)
    config.pop("field")
    config.pop("levels")
    config["geometry"] = {k: config["geometry"][k] for k in ("frame", "voxel_um")}
    config["mesh"] = dict(path=mesh.name, sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
    path.write_text(json.dumps(config))
    api().run_surfaces(path, tmp_path / "output")
    summary = json.loads((tmp_path / "output/MESH.json").read_text())["levels"][0]
    assert summary["components"] == 2 and summary["faces"] == 2
    with sqlite3.connect(tmp_path / "output/mesh.sqlite") as db:
        np.testing.assert_array_equal(np.array(list(db.execute("SELECT x,y,z FROM vertices ORDER BY id"))), vertices)


def test_empty_levels_are_honest_transparent_renders(tmp_path):
    path, config = prepared(tmp_path)
    config["levels"] = [-999.0]
    path.write_text(json.dumps(config))
    result = api().run_surfaces(path, tmp_path / "output")
    assert result["status"] == "SURFACES_READY"
    assert not np.asarray(Image.open(tmp_path / "output/renders/front.png"))[:, :, 3].any()


def test_actual_checkout_cli_surface_mesh_render_and_invalid_exit(tmp_path):
    path, _ = prepared(tmp_path)
    root = Path(__file__).resolve().parents[1]
    tool = root / "tools/run_surfaces.py"
    assert tool.exists(), "surface checkout CLI missing"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(*args):
        return subprocess.run(
            [sys.executable, str(tool), *map(str, args)], env=env, capture_output=True, text=True, timeout=30
        )

    result = run("mesh", "--config", path, "--output", tmp_path / "mesh")
    assert result.returncode == 0, result.stderr
    result = run("render", "--mesh", tmp_path / "mesh", "--output", tmp_path / "render")
    assert result.returncode == 0, result.stderr
    result = run("surfaces", "--config", path, "--output", tmp_path / "full")
    assert result.returncode == 0, result.stderr
    result = run("surfaces", "--config", tmp_path / "missing", "--output", tmp_path / "bad")
    assert result.returncode == 2 and not (tmp_path / "bad").exists()


def test_chunk_checkpoint_pipeline_uses_readonly_generic_geometry(tmp_path):
    from field_store import BlockStore

    path, config = prepared(tmp_path)
    root = tmp_path / "chunks"
    shape = (5, 6, 7)
    values = np.broadcast_to(np.arange(7.0) + 1.0, shape).copy()
    with BlockStore(root, shape, 3, dict(frame=config["geometry"]["frame"], spacing_xyz=[2.0, 3.0, 5.0])) as store:
        for key in itertools.product(range(2), range(2), range(3)):
            low = np.asarray(key) * 3
            high = np.minimum(low + 3, shape)
            region = tuple(slice(a, b) for a, b in zip(low, high))
            store.commit(key, values[region], dict(sweep=8, next_tile=1), np.ones(values[region].shape, bool))
    cp = root / "CHECKPOINT.json"
    before = cp.read_bytes()
    config["field"] = dict(
        kind="chunks",
        path="chunks/CHECKPOINT.json",
        sha256=hashlib.sha256(before).hexdigest(),
        support_policy="observed",
    )
    config["levels"] = [2.5]
    path.write_text(json.dumps(config))
    result = api().run_surfaces(path, tmp_path / "output")
    assert result["mesh"]["levels"][0]["faces"] == 40 and cp.read_bytes() == before
    assert result["mesh"]["metadata"]["spacing_xyz"] == [2.0, 3.0, 5.0]


@pytest.mark.parametrize("settings", [dict(elevation=91), dict(azimuth=float("inf"))])
def test_bad_render_angles_fail_before_new_output(tmp_path, settings):
    path, config = prepared(tmp_path)
    config["render"]["views"] = [dict(name="bad", **settings)]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        api().run_surfaces(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_installed_dispatch_function_reaches_mesh_pipeline(tmp_path):
    from winding_field import field_workflow

    path, _ = prepared(tmp_path)
    assert field_workflow.main(["mesh", "--config", str(path), "--output", str(tmp_path / "mesh")]) == 0
    assert (tmp_path / "mesh/MESH.json").is_file()


def test_unused_imported_vertices_do_not_shrink_actual_surface_out_of_view(tmp_path):
    path, config = prepared(tmp_path)
    mesh = tmp_path / "input.npz"
    np.savez(
        mesh,
        vertices_xyz=np.array([[0.0, 0, 0], [0, 2, 0], [0, 0, 2], [1e12, 1e12, 1e12]]),
        faces=np.array([[0, 1, 2]]),
    )
    config.pop("field")
    config.pop("levels")
    config["mesh"] = dict(path=mesh.name, sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
    config["geometry"] = {k: config["geometry"][k] for k in ("frame", "voxel_um")}
    path.write_text(json.dumps(config))
    result = api().run_surfaces(path, tmp_path / "output")
    assert all(view["nonempty_pixels"] > 0 for view in result["renders"]["outputs"])
    assert result["renders"]["bounds_xyz"] == [[0.0, 0.0, 0.0], [0.0, 2.0, 2.0]]
