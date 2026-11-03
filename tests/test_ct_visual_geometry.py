"""Physical plane sampling regressions for the current CT renderer."""

import numpy as np
import pytest


def implementation():
    from winding_field import ct_visuals as module
    return module


def test_non_cubic_xyz_planes_match_zyx_coordinate_ramp():
    m = implementation()
    z, y, x = np.indices((4, 5, 6))
    field = 10 + x * 2 + 10 * (20 + y * 3) + 100 * (30 + z * 4)
    meta = dict(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4], shape_zyx=[4, 5, 6], step=2)
    for axis in range(3):
        xyz, remaining = m.plane_coordinates([10, 20, 30], [11, 13, 13], axis, 4)
        actual = m.sample_winding(field, meta, xyz)
        expected = (xyz[..., 0] + 10 * xyz[..., 1] + 100 * xyz[..., 2]) / 2
        np.testing.assert_allclose(actual, expected)
        assert np.all(xyz[..., axis] == [14, 24, 34][axis])
        assert remaining == [k for k in range(3) if k != axis]


def test_outside_or_unsupported_stencils_remain_missing():
    m = implementation()
    field = np.ones((3, 3, 3)) * 4
    meta = dict(origin_xyz=[0] * 3, spacing_xyz=[1] * 3, shape_zyx=[3] * 3, step=2)
    xyz = np.array([[[1, 1, 1], [3, 1, 1], [0.5, 0.5, 0.5]]], float)
    support = np.ones(field.shape, bool)
    support[0, 0, 0] = False
    result = m.sample_winding(field, meta, xyz, support)
    assert result[0, 0] == 2
    assert np.isnan(result[0, 1:]).all()
    meta["step"] = 0
    with pytest.raises(ValueError):
        m.sample_winding(field, meta, xyz)


def test_plane_rejects_nonfinite_coordinates_and_invalid_extent():
    m = implementation()
    with pytest.raises(ValueError):
        m.plane_coordinates([float("nan"), 0, 0], [3] * 3, 2, 1)
    with pytest.raises(ValueError):
        m.plane_coordinates([0] * 3, [3] * 3, 2, 3)


def test_visual_archive_supports_explicit_nonunit_grid_and_uint16_ct(tmp_path):
    import json

    from patch_check import sha256

    m = implementation()
    origin, size, spacing = [-5.25, 10.5, 20.0], [4, 5, 6], [0.5, 2, 3]
    planes = []
    for name, axis in (("XY", 2), ("XZ", 1), ("YZ", 0)):
        remaining = [k for k in range(3) if k != axis]
        shape = (size[remaining[1]], size[remaining[0]])
        ct = (1000 + np.arange(np.prod(shape))).astype(np.uint16).reshape(shape)
        np.savez(tmp_path / f"{name}-samples.npz", ct=ct)
        planes.append(
            dict(
                name=name,
                fixed_axis_xyz=axis,
                fixed_position_world=origin[axis] + (size[axis] // 2) * spacing[axis],
                ct_pixels=ct.size,
            )
        )
    receipt = dict(
        frame="another-volume/world",
        voxel_um=12,
        step=2,
        origin_xyz=origin,
        size_xyz=size,
        spacing_xyz=spacing,
        ct_window=[0, 4095],
        contour_levels=[0, 1, 2],
        planes=planes,
        field_sha256={},
        input_ct={},
        ct_url="synthetic://uint16",
        synthetic=True,
    )
    (tmp_path / "RECEIPT.json").write_text(json.dumps(receipt))
    manifest = {p.name: sha256(p) for p in tmp_path.iterdir() if p.is_file()}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    spec = dict(
        schema="ct-winding-overlays-v2",
        archive_manifest=dict(path="MANIFEST.json", sha256=sha256(tmp_path / "MANIFEST.json")),
    )
    visual = m.prepare(spec, tmp_path, receipt["frame"], 2, voxel_um=12)
    np.testing.assert_array_equal(visual["planes"][0]["xyz"][0, 1], [-4.75, 10.5, 29])
    field = np.broadcast_to(2 * np.arange(4, dtype=float), (6, 5, 4)).copy()
    np.save(tmp_path / "field.npy", field)
    meta = dict(
        frame=receipt["frame"],
        voxel_um=12,
        step=2,
        origin_xyz=origin,
        spacing_xyz=spacing,
        shape_zyx=list(field.shape),
        field_sha256=sha256(tmp_path / "field.npy"),
    )
    out = tmp_path / "visuals"
    out.mkdir()
    m.render(visual, field, meta, None, out)
    with np.load(out / "ct-winding-samples.npz") as arrays:
        assert arrays["XY_ct"].dtype == np.uint16
        np.testing.assert_allclose(arrays["XY_candidate"][0], [0, 1, 2, 3])
    recorded = json.loads((out / "VISUALS.json").read_text())
    assert recorded["frame"] == receipt["frame"] and recorded["voxel_um"] == 12
    assert recorded["ct_spacing_xyz"] == spacing
