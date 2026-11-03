"""Configured CT/prior sampling and real surface CLI with an owned native fixture."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_config import load_dataset
from patch_check import sha256


def configuration(frame):
    source = dict(frame=frame, origin_xyz=[10, -8, 4], spacing_xyz=[2, 2, 2])
    return dict(
        schema="winding-dataset-v1", frame=frame, voxel_um=6,
        grid=dict(origin_xyz=[10, -8, 4], spacing_xyz=[2, 2, 2], shape_zyx=[5, 6, 7]),
        sheet_step=1.5, calibration="synthetic analytic calibration",
        afv=dict(path="fibres.afv", native_frame="test-scroll/native", native_voxel_um=[3, 6, 9],
                 scale_xyz=[0.5, 1, 1.5], offset_xyz=[10, -8, 4]),
        sources={name: dict(source, path=name + ".npy") for name in ("ct", "surface", "nx", "ny", "density")},
        prediction=dict(normal_encoding="hemisphere_xy", density_scale=0.001, surface_threshold=128,
                        material_weight=1, background_weight=0.1),
        orientation=dict(mode="radial", centerline=dict(path="axis.json", frame=frame)),
    )


@pytest.fixture
def surface_case(tmp_path):
    frame = "another-scan/world"
    config = configuration(frame)
    config.update(voxel_um=12)
    config["afv"]["native_voxel_um"] = [6, 12, 18]
    config["grid"] = dict(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4], shape_zyx=[5, 6, 7])
    for source in config["sources"].values():
        source.update(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4])
    config["sources"]["ct"]["spacing_xyz"] = [1, 1, 2]
    config["context"] = dict(
        surface=dict(
            seed_margin=1,
            ct_padding_samples=0,
            pitch_world=2,
            depth_offsets_world=[0],
            mesh_step=1,
            ct_window=[0, 3000],
        )
    )
    ct = np.arange(10 * 18 * 14, dtype=np.uint16).reshape(10, 18, 14)
    np.save(tmp_path / "ct.npy", ct)
    np.save(tmp_path / "surface.npy", np.full((5, 6, 7), 255, np.uint8))
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(config))
    field = np.broadcast_to(np.arange(7, dtype=np.float32), (5, 6, 7)).copy()
    np.save(tmp_path / "field.npy", field)
    meta = dict(
        config["grid"],
        frame=frame,
        voxel_um=12,
        step=1.5,
        solver={"converged": True},
        field_sha256=sha256(tmp_path / "field.npy"),
    )
    (tmp_path / "field.npy.json").write_text(json.dumps(meta))
    return dict(path=path, dataset=load_dataset(path), field=field, metadata=meta, ct=ct, root=tmp_path)


def test_probe_preparation_preserves_ct_pixels_and_declared_grid(surface_case):
    from winding_field import surface_probe

    assert hasattr(surface_probe, "prepare_inputs"), "probe needs configured source preparation"
    case = surface_case
    work = case["root"] / "inputs"
    work.mkdir()
    selection, inputs = surface_probe.prepare_inputs(case["dataset"], case["field"], case["metadata"], work)
    np.testing.assert_array_equal(np.load(inputs["ct"]), case["ct"])
    meta = json.loads(Path(inputs["ct_metadata"]).read_text())
    assert meta["frame"] == "another-scan/world" and meta["voxel_um"] == 12
    assert meta["origin_xyz"] == [10, 20, 30] and meta["spacing_xyz"] == [1, 1, 2]
    assert selection["seed_index_zyx"] == [2, 2, 3] and selection["seed_xyz"] == [16, 26, 38]
    assert selection["level"] == 3 and not selection["human_labels_used"]
    for key in ("ct", "ct_metadata", "seed"):
        assert inputs[key + "_sha256"] == sha256(inputs[key])
    with pytest.raises(ValueError, match="frame"):
        surface_probe.prepare_inputs(case["dataset"], case["field"], dict(case["metadata"], frame="wrong"), work)


def test_context_image_uses_ct_spacing_and_rejects_outside_seed(tmp_path):
    from winding_field.surface_probe import context_image

    ct = np.zeros((7, 50, 60), np.uint16)
    meta = dict(
        frame="scan/world", origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4], voxel_um=12, display_window=[0, 1000]
    )
    seed = dict(seed_xyz=[30, 50, 42], frame="scan/world")
    vertices = np.array([[90, 140, 42]], float)
    path = tmp_path / "context.png"
    context_image(path, ct, meta, seed, vertices)
    with Image.open(path) as image:
        assert image.info["CT_z_world"] == "42.0"
        assert image.getpixel((40, 40)) == (255, 60, 60)
    with pytest.raises(ValueError, match="outside"):
        context_image(tmp_path / "bad.png", ct, meta, dict(seed, seed_xyz=[30, 50, 26]), vertices)


def native_fixture(root):
    app = root / "flatboi-fixture"
    app.write_text(
        f"#!{sys.executable}\n"
        + """import sys
from pathlib import Path
mesh = Path(sys.argv[2])
lines = mesh.read_text().splitlines()
vertices = [line for line in lines if line.startswith('v ')]
output = list(vertices)
for line in vertices:
    _, x, y, z = line.split()
    output.append(f'vt {y} {z}')
for line in lines:
    if line.startswith('f '):
        output.append('f ' + ' '.join(f'{v}/{v}' for v in line.split()[1:]))
mesh.with_name('inputmesh_flatboi.obj').write_text('\\n'.join(output) + '\\n')
"""
    )
    app.chmod(0o700)
    return app


def test_probe_cli_accepts_configured_frame_without_human_patch_or_tile_ownership(surface_case):
    from winding_field.surface_probe import main

    root = surface_case["root"]
    app = native_fixture(root)
    code = main(
        [
            "--config",
            str(surface_case["path"]),
            "--field",
            str(root / "field.npy"),
            "--metadata",
            str(root / "field.npy.json"),
            "--work",
            str(root / "probe"),
            "--output",
            str(root / "surface"),
            "--app-run",
            str(app),
        ]
    )
    result = json.loads((root / "probe/result.json").read_text())
    assert code == 0, result
    assert result["status"] == "EXPORTED_UNCERTIFIED"
    assert result["heldout_patch"]["status"] == "NOT_EVALUATED"
    assert result["seams"]["status"] == "NOT_APPLICABLE"
    assert result["seed"]["seed_xyz"] == [16, 26, 38]
    assert result["export_receipt"]["preview"]["window"] == [0, 3000]
    with np.load(root / "surface/ct_stack.npz") as stack:
        rows = np.argwhere(stack["valid"][0])
        row, col = rows[len(rows) // 3]
        expected = int(np.rint(float(stack["values"][0, row, col]) * 255 / 3000))
    with Image.open(root / "surface/ct_preview.png") as image:
        assert image.getpixel((int(col), int(row))) == (expected, expected, expected, 255)


def test_atlas_cli_reuses_prior_seed_and_configured_raster_without_human_labels(surface_case):
    from winding_field.surface_atlas import main
    from winding_field.surface_probe import prepare_inputs

    case = surface_case
    root = case["root"]
    work = root / "inputs"
    work.mkdir()
    _, inputs = prepare_inputs(case["dataset"], case["field"], case["metadata"], work)
    inputs.update(field_sha256=sha256(root / "field.npy"), metadata_sha256=sha256(root / "field.npy.json"))
    path = work / "inputs.json"
    path.write_text(json.dumps(inputs))
    app = native_fixture(root)
    code = main(
        [
            "--field",
            str(root / "field.npy"),
            "--metadata",
            str(root / "field.npy.json"),
            "--inputs",
            str(path),
            "--app-run",
            str(app),
            "--output",
            str(root / "atlas-output"),
            "--max-depth",
            "0",
            "--max-charts",
            "1",
            "--budget-seconds",
            "30",
        ]
    )
    result = json.loads((root / "atlas-output/result.json").read_text())
    assert code == 0, result
    assert result["human_seed"] is False and result["validation"]["status"] == "NOT_EVALUATED"
    assert result["atlas_status"] == "PASS"
    with np.load(root / "atlas-output/atlas/chart-000/grid.npz") as grid:
        np.testing.assert_array_equal(grid["pitch_um"], [24, 24])
    with np.load(root / "atlas-output/atlas/chart-000/ct_stack.npz") as stack:
        np.testing.assert_array_equal(stack["depth_offsets"], [0])


def human_patch(root, frame=None, unit=None):
    import tifffile

    root.mkdir()
    y, x = np.indices((3, 3))
    points = np.stack([20 + x, 20 + y, np.full_like(x, 20)], axis=-1).astype(float)
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(root / f"{axis}.tif", points[..., i])
    metadata = {"format": "tifxyz"}
    if frame is not None:
        metadata["frame"] = frame
    if unit is not None:
        metadata["voxel_um"] = unit
    (root / "meta.json").write_text(json.dumps(metadata))
    vertices = points[[0, 0, -1, -1], [0, -1, -1, 0]]
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def test_human_patch_conflicting_units_are_not_scored(tmp_path):
    from winding_field.surface_probe import evaluate_patch

    vertices, faces = human_patch(tmp_path / "patch", "scan/world", 24)
    meta = dict(frame="scan/world", voxel_um=12, origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1], shape_zyx=[64] * 3)
    with pytest.raises(ValueError, match="units"):
        evaluate_patch(tmp_path / "patch", vertices, faces, meta, None)


def test_legacy_patch_needs_explicit_coordinate_context(tmp_path):
    from winding_field.surface_probe import evaluate_patch

    vertices, faces = human_patch(tmp_path / "patch")
    meta = dict(frame="scan/world", voxel_um=12, origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1], shape_zyx=[64] * 3)
    before = (tmp_path / "patch/meta.json").read_bytes()
    with pytest.raises(ValueError, match="coordinate"):
        evaluate_patch(tmp_path / "patch", vertices, faces, meta, None)
    result = evaluate_patch(
        tmp_path / "patch", vertices, faces, meta, None, patch_frame="scan/world", patch_voxel_um=12
    )
    assert result["covered_human_points"] == 9
    assert (tmp_path / "patch/meta.json").read_bytes() == before


def test_atlas_checks_human_patch_units_before_seed_selection(surface_case):
    from winding_field.surface_atlas import main
    from winding_field.surface_probe import prepare_inputs

    case = surface_case
    root = case["root"]
    work = root / "inputs"
    work.mkdir()
    _, inputs = prepare_inputs(case["dataset"], case["field"], case["metadata"], work)
    inputs.update(field_sha256=sha256(root / "field.npy"), metadata_sha256=sha256(root / "field.npy.json"))
    (work / "inputs.json").write_text(json.dumps(inputs))
    human_patch(root / "patch", case["metadata"]["frame"], 24)
    code = main(
        [
            "--field",
            str(root / "field.npy"),
            "--metadata",
            str(root / "field.npy.json"),
            "--inputs",
            str(work / "inputs.json"),
            "--patch",
            str(root / "patch"),
            "--app-run",
            str(root / "not-launched"),
            "--output",
            str(root / "atlas"),
        ]
    )
    result = json.loads((root / "atlas/result.json").read_text())
    assert code == 2 and "units" in result["error"]
    assert not (root / "atlas/surface.npz").exists()
