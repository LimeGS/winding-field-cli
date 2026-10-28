"""Actual pointmap -> TIFXYZ frame/sentinel/scale contracts, no new geometry gate."""

import json

import numpy as np
import pytest
import tifffile


def test_tifxyz_keeps_l2_xyz_and_uses_minus_one_for_invalid_cells(tmp_path):
    from winding_field.surface_geometry_check import export_tifxyz

    points = np.array([[[0.0, 2500, 10096], [4123.123456, 2600, 10100]], [[np.nan] * 3, [0.0, 0, 0]]])
    valid = np.array([[True, True], [False, False]])
    grid = tmp_path / "grid.npz"
    np.savez(grid, points_xyz=points, valid=valid, pitch_um=[9.6, 9.6])
    metadata = dict(frame="PHercParis4/20260411134726:L2", voxel_um=9.6, origin_xyz=[4088, 2500, 10096])
    out = tmp_path / "tifxyz"
    receipt = export_tifxyz(grid, metadata, out)
    actual = np.stack([tifffile.imread(out / f"{a}.tif") for a in "xyz"], -1)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[~valid], -1)
    np.testing.assert_array_equal(actual[valid], points[valid].astype(np.float32))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["frame"] == metadata["frame"] and meta["voxel_um"] == 9.6
    assert meta["scale"] == [1.0, 1.0]
    assert receipt["valid_points"] == 2
    assert receipt["maximum_float32_error_um"] < 0.01
    with pytest.raises(FileExistsError):
        export_tifxyz(grid, metadata, out)


def test_scale_is_uv_cells_per_world_unit_and_axes_are_explicit(tmp_path):
    from winding_field.surface_geometry_check import export_tifxyz

    points = np.ones((3, 4, 3)) * 100
    grid = tmp_path / "grid.npz"
    np.savez(grid, points_xyz=points, valid=np.ones((3, 4), bool), pitch_um=[19.2, 9.6])
    out = tmp_path / "tifxyz"
    export_tifxyz(grid, dict(frame="PHercParis4/20260411134726:L2", voxel_um=9.6), out)
    assert json.loads((out / "meta.json").read_text())["scale"] == [1.0, 0.5]  # columns, rows


@pytest.mark.parametrize("damage", ["nonfinite_valid", "negative_valid", "mask_type", "frame"])
def test_conversion_rejects_invalid_declared_geometry(tmp_path, damage):
    from winding_field.surface_geometry_check import export_tifxyz

    points = np.ones((3, 4, 3)) * 100
    valid = np.ones((3, 4), bool)
    frame = "PHercParis4/20260411134726:L2"
    if damage == "nonfinite_valid":
        points[1, 1, 0] = np.nan
    if damage == "negative_valid":
        points[1, 1, 0] = -1
    if damage == "mask_type":
        valid = valid.astype(np.uint8)
    if damage == "frame":
        frame = ""
    grid = tmp_path / "grid.npz"
    np.savez(grid, points_xyz=points, valid=valid, pitch_um=[9.6, 9.6])
    with pytest.raises(ValueError):
        export_tifxyz(grid, dict(frame=frame, voxel_um=9.6), tmp_path / "tifxyz")


def test_unmeasured_gate_is_not_pass_even_when_gate_cli_returns_zero():
    from winding_field.surface_geometry_check import gate_outcome

    assert gate_outcome({"geometry_qc_state": "GEOMETRY_UNMEASURED", "measurement_complete": False}) == "UNMEASURED"
    assert gate_outcome({"geometry_qc_state": "GEOMETRY_CERTIFIED", "measurement_complete": False}) == "UNMEASURED"
    assert gate_outcome({"geometry_qc_state": "GEOMETRY_REJECTED_BRIDGE", "measurement_complete": True}) == "FAIL"
    assert (
        gate_outcome({"geometry_qc_state": "GEOMETRY_CERTIFIED", "measurement_complete": True, "status": "PASS"})
        == "PASS"
    )


def test_tifxyz_uses_another_declared_frame_and_physical_unit(tmp_path):
    from winding_field.surface_geometry_check import export_tifxyz

    points = np.ones((3, 4, 3)) * 50
    grid = tmp_path / "grid.npz"
    np.savez(grid, points_xyz=points, valid=np.ones((3, 4), bool), pitch_um=[24, 12])
    out = tmp_path / "tifxyz"
    receipt = export_tifxyz(grid, dict(frame="different-scroll/world", voxel_um=12), out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["frame"] == "different-scroll/world" and meta["voxel_um"] == 12
    assert meta["scale"] == [1, 0.5]
    assert receipt["maximum_float32_error_um"] == 0
