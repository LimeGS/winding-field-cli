"""Analytic geometry/CT contracts; no downloads, GPU, or papyrus quality claims."""

import json

import numpy as np
import pytest


def metadata(shape, origin=(10, 20, 30), spacing=(2, 3, 4)):
    return dict(
        frame="synthetic:L2", origin_xyz=list(origin), spacing_xyz=list(spacing), shape_zyx=list(shape), voxel_um=9.6
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_actual_level_and_non_cubic_xyz_coordinates(axis, sign):
    from winding_field.surface_export import extract_surface

    field = sign * np.indices((6, 7, 8))[axis].astype(float)
    meta = metadata(field.shape)
    world_axis = 2 - axis
    for index in (1.25, 3.5):
        v, f, receipt = extract_surface(field, meta, sign * index)
        np.testing.assert_allclose(
            v[:, world_axis], meta["origin_xyz"][world_axis] + index * meta["spacing_xyz"][world_axis]
        )
        assert f.min() == 0 and f.max() == len(v) - 1
        assert receipt["level"] == sign * index
        assert receipt["geometry"]["basic_status"] == "PASS"
        assert receipt["geometry"]["self_intersection"] == "UNMEASURED"
        assert receipt["geometry"]["status"] == "UNMEASURED"


def test_seed_component_is_reindexed_and_largest_is_explicit():
    from winding_field.surface_export import extract_surface

    z, y, x = np.indices((12, 14, 22))
    field = np.minimum(
        (x - 5) ** 2 + (y - 6) ** 2 + (z - 5) ** 2 - 9, (x - 16) ** 2 + (y - 6) ** 2 + (z - 5) ** 2 - 4
    ).astype(float)
    meta = metadata(field.shape, (0, 0, 0), (1, 1, 1))
    v, f, r = extract_surface(field, meta, 0, seed_xyz=[18, 6, 5])
    assert v[:, 0].min() >= 14
    assert set(np.unique(f)) == set(range(len(v)))
    assert r["components"] == 2 and r["selection"] == "seed"
    v, _, r = extract_surface(field, meta, 0)
    assert v[:, 0].max() <= 8 and r["selection"] == "largest"
    assert r["sheet_identity"] == "UNMEASURED"


def plane():
    # Rectangle in an oblique plane; orthonormal intrinsic directions.
    u = np.array([0.8, 0, 0.6])
    v = np.array([0, 1, 0])
    points = np.array([20 + a * u + b * v for a, b in [(0, 0), (4, 0), (4, 3), (0, 3)]])
    return points, np.array([[0, 1, 2], [0, 2, 3]])


def test_planar_projection_raster_and_coordinate_coded_ct():
    from winding_field.surface_export import project_pca, rasterize_surface, sample_ct_stack

    vertices, faces = plane()
    uv, gate = project_pca(vertices, faces)
    assert gate["status"] == "PASS" and gate["method"] == "pca_projection"
    points, normals, valid, raster = rasterize_surface(vertices, faces, uv, 0.25)
    assert valid.any() and raster["multimapped_pixels"] == 0
    assert np.isnan(points[~valid]).all()
    for axis in (0, 1):
        pairs = np.take(valid, np.arange(valid.shape[axis] - 1), axis=axis) & np.take(
            valid, np.arange(1, valid.shape[axis]), axis=axis
        )
        np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=axis), axis=-1)[pairs], 0.25, atol=1e-10)
    np.testing.assert_allclose((points[valid] - 20) @ [-0.6, 0, 0.8], 0, atol=1e-10)
    np.testing.assert_allclose(normals[valid], np.tile([-0.6, 0, 0.8], (valid.sum(), 1)), atol=1e-10)
    # A CT stored at finer, anisotropic spacing in the SAME world frame.
    z, y, x = np.indices((40, 40, 48))
    cm = metadata(z.shape, (10, 10, 10), (0.5, 0.75, 1.25))
    ct = 2 * (10 + x * 0.5) + 3 * (10 + y * 0.75) + 5 * (10 + z * 1.25)
    offsets = np.array([-0.5, 0, 0.75])
    stack, mask, receipt = sample_ct_stack(ct, cm, points, normals, offsets, frame="synthetic:L2", voxel_um=9.6)
    for i, offset in enumerate(offsets):
        q = points[valid] + offset * np.array([-0.6, 0, 0.8])
        np.testing.assert_allclose(stack[i][valid], q @ [2, 3, 5], atol=1e-5)
    assert np.array_equal(mask, np.broadcast_to(valid, stack.shape))
    assert np.isnan(stack[~mask]).all()
    np.testing.assert_allclose(receipt["depth_offsets_um"], [-4.8, 0, 7.2])


def test_uv_overlap_inversion_and_distortion_are_rejected():
    from winding_field.surface_export import rasterize_surface, validate_uv

    tri = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    vertices = np.vstack([tri, tri + [0, 0, 1]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    uv = vertices[:, :2].copy()
    result = validate_uv(vertices, faces, uv)
    assert result["status"] == "FAIL" and result["overlap_pairs"] == 1
    with pytest.raises(ValueError, match="UV"):
        rasterize_surface(vertices, faces, uv, 0.25)
    vertices, faces = plane()
    uv = np.array([[0.0, 0], [4, 0], [4, 3], [0, 3]])
    flipped = faces.copy()
    flipped[1] = flipped[1, ::-1]
    assert validate_uv(vertices, flipped, uv)["status"] == "FAIL"
    assert validate_uv(vertices, faces, uv * 2)["status"] == "FAIL"


def test_subpixel_overlap_is_not_hidden_by_coarse_raster():
    from winding_field.surface_export import validate_uv

    uv = np.array([[0.0, 0], [1, 0], [0, 1], [0.99, 0], [1.99, 0], [0.99, 1]])
    vertices = np.column_stack([uv, [0, 0, 0, 1, 1, 1]])
    result = validate_uv(vertices, np.array([[0, 1, 2], [3, 4, 5]]), uv)
    assert result["status"] == "FAIL" and result["overlap_pairs"] == 1


def test_distortion_policy_is_p95_not_maximum_but_uniform_stretch_fails():
    from winding_field.surface_export import validate_uv

    uv = np.array([[3 * i + x, y] for i in range(101) for x, y in [(0, 0), (1, 0), (0, 1)]], float)
    vertices = np.column_stack([uv, np.zeros(len(uv))])
    vertices[-3:] = vertices[-3] + (vertices[-3:] - vertices[-3]) * 1.25
    faces = np.arange(len(uv)).reshape(-1, 3)
    gate = validate_uv(vertices, faces, uv)
    assert gate["status"] == "PASS"
    assert gate["p95_distortion"] == pytest.approx(0)
    assert gate["max_distortion"] == pytest.approx(0.25)
    assert gate["distortion_gate_statistic"] == "p95"
    gate = validate_uv(np.column_stack([uv * 1.25, np.zeros(len(uv))]), faces, uv)
    assert gate["status"] == "FAIL"


def test_supplied_global_normals_survive_chart_rasterization():
    from winding_field.surface_export import rasterize_surface

    vertices, faces = native_plane()
    uv = vertices[:, :2] - vertices[0, :2]
    global_normals = np.tile([0.6, 0, 0.8], (len(vertices), 1))
    _, normals, valid, _ = rasterize_surface(vertices, faces, uv, 1, vertex_normals_xyz=global_normals)
    np.testing.assert_allclose(normals[valid], np.tile([0.6, 0, 0.8], (valid.sum(), 1)))


def test_masked_field_hole_is_not_filled_by_extraction_or_raster():
    from winding_field.surface_export import extract_surface, project_pca, rasterize_surface

    field = np.broadcast_to(np.arange(5.0)[:, None, None], (5, 12, 14)).copy()
    support = np.ones(field.shape, bool)
    support[:, 4:8, 5:9] = False
    field[~support] = np.nan
    meta = metadata(field.shape, (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="finite"):
        extract_surface(field, meta, 2.5)
    v, f, r = extract_surface(field, {**meta, "support": support}, 2.5)
    uv, gate = project_pca(v, f)
    assert gate["status"] == "PASS"
    points, _, valid, _ = rasterize_surface(v, f, uv, 0.25)
    assert (~valid).any() and np.isnan(points[~valid]).all()
    assert not ((points[..., 0] > 5) & (points[..., 0] < 8) & (points[..., 1] > 4) & (points[..., 1] < 7)).any()
    assert r["unsupported_voxels"] == int((~support).sum())


def test_ct_outside_nan_support_and_frame_mismatch():
    from winding_field.surface_export import sample_ct_stack

    ct = np.arange(64.0).reshape(4, 4, 4)
    ct[2, 2, 2] = np.nan
    meta = metadata(ct.shape, (0, 0, 0), (1, 1, 1))
    points = np.array([[[1.0, 1, 1], [1.5, 1.5, 1.5], [-0.1, 1, 1], [np.nan] * 3]])
    normals = np.broadcast_to([0.0, 0, 1], points.shape).copy()
    stack, mask, _ = sample_ct_stack(ct, meta, points, normals, [0], frame="synthetic:L2", voxel_um=9.6)
    assert mask.tolist() == [[[True, False, False, False]]]
    assert stack[0, 0, 0] == 21 and np.isnan(stack[~mask]).all()
    with pytest.raises(ValueError, match="frame"):
        sample_ct_stack(ct, meta, points, normals, [0], frame="wrong:L0", voxel_um=9.6)
    with pytest.raises(ValueError, match="voxel_um"):
        sample_ct_stack(ct, meta, points, normals, [0], frame="synthetic:L2", voxel_um=2.4)


def test_geometry_does_not_certify_unmeasured_intersections():
    from winding_field.surface_export import mesh_geometry

    v, f = plane()
    assert mesh_geometry(v, f)["status"] == "UNMEASURED"
    for bad in (np.array([[0, 1, 99]]), np.array([[0, 0, 1]]), np.vstack([f, f[0]])):
        assert mesh_geometry(v, bad)["basic_status"] == "FAIL"
    assert mesh_geometry(v, f, [[0, 0, 0], [1, 1, 1]])["basic_status"] == "FAIL"


def test_uv_measurement_budget_and_tiny_pitch_fail_closed():
    from winding_field.surface_export import rasterize_surface, validate_uv

    uv = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    vertices = np.column_stack([uv, np.zeros(4)])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    with pytest.raises(ValueError, match="max_pixels"):
        rasterize_surface(vertices, faces, uv, 1e-30)
    tri = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    vertices = np.vstack([tri, tri + [0, 0, 1], tri + [0, 0, 2]])
    faces = np.arange(9).reshape(-1, 3)
    r = validate_uv(vertices, faces, vertices[:, :2], max_pairs=1)
    assert r["status"] == "FAIL" and r["overlap_status"] == "UNMEASURED"


def test_ct_float32_overflow_is_invalid_nan_and_integer_ct_interpolates():
    from winding_field.surface_export import sample_ct_stack

    ct = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    meta = metadata(ct.shape, (0, 0, 0), (1, 1, 1))
    points = np.array([[[0.25, 0.25, 0.25]]])
    normals = np.array([[[0.0, 0, 1]]])
    values, mask, _ = sample_ct_stack(ct, meta, points, normals, [0], frame="synthetic:L2", voxel_um=9.6)
    assert mask.all() and values.item() == 5.25
    values, mask, _ = sample_ct_stack(
        ct.astype(float) * 1e100, meta, points, normals, [0], frame="synthetic:L2", voxel_um=9.6
    )
    assert not mask.any() and np.isnan(values).all()


def test_native_ct_conversion_is_explicit_and_support_is_not_air():
    from winding_field.surface_export import sample_ct_stack

    z, y, x = np.indices((4, 4, 4))
    ct = (100 * z + 10 * y + x).astype(np.uint16)
    # Native origin [40,80,120] is supplied explicitly as L2 [10,20,30].
    meta = metadata(ct.shape, (10, 20, 30), (0.25, 0.25, 0.25))
    points = np.array([[[10.125, 20.125, 30.125]]])
    normals = np.array([[[0.0, 0, 1]]])
    values, valid, receipt = sample_ct_stack(ct, meta, points, normals, [0, 0.25], frame="synthetic:L2", voxel_um=9.6)
    np.testing.assert_allclose(values[:, 0, 0], [55.5, 155.5])
    np.testing.assert_allclose(receipt["depth_offsets_um"], [0, 2.4])
    assert valid.all()
    support = np.ones(ct.shape, bool)
    support[0, 0, 0] = False
    values, valid, _ = sample_ct_stack(
        ct, {**meta, "support": support}, points, normals, [0, 0.25], frame="synthetic:L2", voxel_um=9.6
    )
    assert np.isnan(values[0, 0, 0]) and not valid[0, 0, 0]
    assert values[1, 0, 0] == 155.5 and valid[1, 0, 0]


def test_cli_saves_real_artifacts_and_rejects_overwrite(tmp_path):
    from winding_field.surface_export import main

    field = np.indices((5, 6, 7))[0].astype(float)
    meta = metadata(field.shape, (0, 0, 0), (1, 1, 1))
    fp, mp, cp = tmp_path / "field.npy", tmp_path / "metadata.json", tmp_path / "ct.npy"
    np.save(fp, field)
    mp.write_text(json.dumps(meta))
    np.save(cp, field * 10)
    output = tmp_path / "new"
    args = [
        "--field",
        str(fp),
        "--metadata",
        str(mp),
        "--level",
        "2.5",
        "--output",
        str(output),
        "--pitch",
        ".5",
        "--ct",
        str(cp),
        "--ct-metadata",
        str(mp),
        "--depth-offsets",
        "-.5",
        "0",
        ".5",
    ]
    assert main(args) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "EXPORTED_UNCERTIFIED"
    assert receipt["raster"]["pitch_um"] == [4.8, 4.8]
    assert (output / "surface.obj").is_file() and (output / "surface.npz").is_file()
    with np.load(output / "grid.npz") as grid, np.load(output / "ct_stack.npz") as stack:
        assert np.isnan(grid["points_xyz"][~grid["valid"]]).all()
        np.testing.assert_allclose(stack["depth_offsets"], [-0.5, 0, 0.5])
        assert set(np.unique(stack["values"][stack["valid"]])).issubset({20, 25, 30})
    from PIL import Image

    with Image.open(output / "ct_preview.png") as image:
        assert image.mode == "RGBA"
        assert "not ink" in image.info["Description"]
    with pytest.raises(FileExistsError):
        main(args)


def test_failed_projection_keeps_receipt_not_fake_image(tmp_path):
    from winding_field.surface_export import main

    z, y, x = np.indices((10, 11, 12))
    field = ((x - 5) ** 2 + (y - 5) ** 2 + (z - 4) ** 2).astype(float)
    fp, mp = tmp_path / "field.npy", tmp_path / "meta.json"
    np.save(fp, field)
    mp.write_text(json.dumps(metadata(field.shape, (0, 0, 0), (1, 1, 1))))
    out = tmp_path / "rejected"
    assert (
        main(["--field", str(fp), "--metadata", str(mp), "--level", "6.25", "--output", str(out), "--pitch", ".5"]) == 2
    )
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["projection"]["status"] == "FAIL"
    assert (out / "surface.obj").exists()
    assert not (out / "grid.npz").exists() and not (out / "ct_preview.png").exists()


def write_flat_obj(path, vertices, faces, uv, mode="indexed"):
    lines = ["v " + " ".join(map(str, row)) for row in vertices]
    lines += ["vt " + " ".join(map(str, row)) for row in uv]
    for face in faces:
        lines.append("f " + " ".join(f"{i + 1}/{i + 1}" if mode == "indexed" else str(i + 1) for i in face))
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("mode", ["indexed", "per_vertex"])
def test_flatboi_obj_binding_and_single_physical_scale(tmp_path, mode):
    from winding_field.surface_export import read_flatboi_uv, scale_uv_to_world, validate_uv

    vertices, faces = plane()
    raw_uv = np.array([[7.0, 9], [7.4, 9], [7.4, 9.3], [7, 9.3]])
    obj = tmp_path / "flat.obj"
    # Row order and cyclic rotations do not change oriented topology.
    write_flat_obj(obj, vertices, np.roll(faces[::-1], 1, axis=1), raw_uv, mode)
    uv, binding = read_flatboi_uv(obj, vertices, faces)
    np.testing.assert_allclose(uv, raw_uv)
    assert binding["status"] == "PASS"
    scaled, scaling = scale_uv_to_world(vertices, faces, uv)
    assert scaling["world_units_per_uv_unit"] == pytest.approx(10)
    np.testing.assert_allclose(scaled, [[0, 0], [4, 0], [4, 3], [0, 3]], atol=1e-12)
    assert validate_uv(vertices, faces, scaled)["status"] == "PASS"
    raw_uv[:, 0] *= 2
    scaled, _ = scale_uv_to_world(vertices, faces, raw_uv)
    assert validate_uv(vertices, faces, scaled)["status"] == "FAIL"


@pytest.mark.parametrize("damage", ["xyz", "topology", "seam", "missing_uv", "nonfinite", "bad_index"])
def test_flatboi_obj_rejects_unbound_or_unsupported_uv(tmp_path, damage):
    from winding_field.surface_export import read_flatboi_uv

    vertices, faces = plane()
    uv = np.array([[0.0, 0], [4, 0], [4, 3], [0, 3]])
    obj = tmp_path / "flat.obj"
    changed_v, changed_f = vertices.copy(), faces.copy()
    if damage == "xyz":
        changed_v[0, 0] += 0.01
    if damage == "topology":
        changed_f[1] = [0, 1, 3]
    if damage == "nonfinite":
        uv[0, 0] = np.nan
    write_flat_obj(obj, changed_v, changed_f, uv)
    text = obj.read_text()
    if damage == "seam":
        text = text.replace("f 1/1 3/3 4/4", "f 1/2 3/3 4/4")
    if damage == "missing_uv":
        text = "\n".join(s for s in text.splitlines() if not s.startswith("vt "))
    if damage == "bad_index":
        text = text.replace("1/1", "1/99")
    obj.write_text(text)
    with pytest.raises(ValueError):
        read_flatboi_uv(obj, vertices, faces)


@pytest.mark.parametrize("delta", [0.0, 1e-12])
def test_observed_native_per_corner_indices_bind_only_exact_uv_aliases(tmp_path, delta):
    from winding_field.surface_export import read_flatboi_uv

    vertices, faces = native_plane()
    # Real native format observed: 3 vt records per face, identical coordinates
    # for repeated XYZ vertices. This is not a geometric UV seam.
    uv = np.array([[608.6471655820776, 383.4354435450956], [612.0, 383.0], [612.0, 386.0], [608.0, 386.0]])
    corner_uv = uv[faces].reshape(-1, 2).copy()
    corner_uv[3, 0] += delta
    obj = tmp_path / "flat.obj"
    lines = ["v " + " ".join(map(str, v)) for v in vertices]
    lines += ["vt " + " ".join(map(str, t)) for t in corner_uv]
    lines += ["f 1/1 2/2 3/3", "f 1/4 3/5 4/6"]
    obj.write_text("\n".join(lines) + "\n")
    if delta:
        with pytest.raises(ValueError, match="seam"):
            read_flatboi_uv(obj, vertices, faces)
    else:
        got, receipt = read_flatboi_uv(obj, vertices, faces)
        np.testing.assert_array_equal(got, uv)
        assert receipt["exact_uv_alias_occurrences"] == 2


def native_fixture(path, action="valid"):
    """Owned subprocess fixture implements the native file/argv boundary, not flattening."""
    import sys

    path.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json, sys, time
from pathlib import Path
print('native stdout', flush=True)
print('native stderr', file=sys.stderr, flush=True)
Path('seen-argv.json').write_text(json.dumps(sys.argv[1:]))
action = ACTION
if action == 'exit': sys.exit(7)
if action == 'timeout': time.sleep(10)
source = Path(sys.argv[2])
text = source.read_text()
if action == 'tamper': source.write_text(text + '# changed\\n')
lines = text.splitlines()
vertices = [list(map(float, s.split()[1:])) for s in lines if s.startswith('v ')]
# The integration fixture is a z-plane; return known, dimensionless UVs.
uv = ['vt %.17g %.17g' % (v[0]/10, v[1]/10) for v in vertices]
output = source.with_name(source.stem + '_flatboi.obj')
output.write_text('\\n'.join(lines+uv)+'\\n')
""".replace("ACTION", repr(action))
    )
    path.chmod(0o700)
    return path


def native_plane():
    return np.array([[10.0, 20, 30], [14, 20, 30], [14, 23, 30], [10, 23, 30]]), np.array([[0, 1, 2], [0, 2, 3]])


def test_native_adapter_command_hashes_logs_and_scaled_uv(tmp_path):
    import hashlib

    from winding_field.surface_export import flatten_native

    app = native_fixture(tmp_path / "AppRun")
    digest = hashlib.sha256(app.read_bytes()).hexdigest()
    vertices, faces = native_plane()
    out = tmp_path / "native"
    uv, receipt = flatten_native(vertices, faces, app, out, expected_sha256=digest)
    assert receipt["status"] == "PASS" and receipt["returncode"] == 0
    assert receipt["app_run_sha256"] == digest and not receipt["timed_out"]
    assert json.loads((out / "seen-argv.json").read_text()) == [
        "flatboi",
        str(out / "inputmesh.obj"),
        "200",
        "symmetric_dirichlet",
        "--tol=1e-6",
    ]
    assert "native stdout" in (out / "stdout.log").read_text()
    assert "native stderr" in (out / "stderr.log").read_text()
    for name in ("inputmesh.obj", "inputmesh_flatboi.obj", "stdout.log", "stderr.log"):
        assert receipt["artifacts"][name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert (out / "receipt.json").is_file()
    np.testing.assert_allclose(uv, [[0, 0], [4, 0], [4, 3], [0, 3]], atol=1e-12)


@pytest.mark.parametrize(
    "action,status",
    [("exit", "FAIL"), ("timeout", "TIMEOUT"), ("tamper", "FAIL"), ("wrong_hash", "FAIL"), ("missing", "UNAVAILABLE")],
)
def test_native_failures_never_pass_and_leave_receipts(tmp_path, action, status):
    from winding_field.surface_export import flatten_native

    app = tmp_path / "AppRun"
    if action != "missing":
        native_fixture(app, action)
    kwargs = {"expected_sha256": "0" * 64} if action == "wrong_hash" else {}
    out = tmp_path / "native"
    uv, receipt = flatten_native(*native_plane(), app, out, timeout=0.3 if action == "timeout" else 5, **kwargs)
    assert uv is None and receipt["status"] == status
    assert json.loads((out / "receipt.json").read_text())["status"] == status
    if action in ("wrong_hash", "missing"):
        assert receipt["returncode"] is None and not (out / "seen-argv.json").exists()
    if action == "exit":
        assert receipt["returncode"] == 7
    if action == "timeout":
        assert receipt["timed_out"] and receipt["returncode"] != 0


def test_mesh_step_reduces_resolution_without_coordinate_or_support_shortcuts():
    from winding_field.surface_export import extract_surface, project_pca, rasterize_surface

    field = np.indices((9, 11, 13))[0].astype(float)
    meta = metadata(field.shape)
    fine, fine_faces, _ = extract_surface(field, meta, 3.5)
    coarse, faces, r = extract_surface(field, meta, 3.5, step_size=2)
    assert len(faces) < len(fine_faces)
    np.testing.assert_allclose(coarse[:, 2], 44)
    np.testing.assert_allclose(coarse.min(0), fine.min(0))
    np.testing.assert_allclose(coarse.max(0), fine.max(0))
    assert r["step_size"] == 2 and r["geometry_resolution_reduced"]
    support = np.ones(field.shape, bool)
    support[:, 3, 3] = False
    vertices, faces, _ = extract_surface(field, {**meta, "support": support}, 3.5, step_size=2)
    uv, _ = project_pca(vertices, faces)
    points, _, valid, _ = rasterize_surface(vertices, faces, uv, 0.5)
    assert not np.any(valid & np.isclose(points[..., 0], 16) & np.isclose(points[..., 1], 29))
    with pytest.raises(ValueError, match="step_size"):
        extract_surface(field, meta, 3.5, step_size=0)


def test_cli_native_adapter_reaches_existing_render_gates(tmp_path):
    from winding_field.surface_export import main

    field = np.indices((7, 9, 11))[0].astype(float)
    fp, mp = tmp_path / "field.npy", tmp_path / "meta.json"
    np.save(fp, field)
    mp.write_text(json.dumps(metadata(field.shape, (0, 0, 0), (1, 1, 1))))
    app = native_fixture(tmp_path / "AppRun")
    out = tmp_path / "export"
    assert (
        main(
            [
                "--field",
                str(fp),
                "--metadata",
                str(mp),
                "--level",
                "2.5",
                "--pitch",
                ".5",
                "--output",
                str(out),
                "--app-run",
                str(app),
                "--flatten-iters",
                "10",
                "--flatten-timeout",
                "5",
                "--mesh-step",
                "2",
            ]
        )
        == 0
    )
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["projection"]["method"] == "native_flatboi"
    assert receipt["extraction"]["step_size"] == 2
    assert receipt["status"] == "EXPORTED_UNCERTIFIED"
    assert (out / "grid.npz").exists()
    assert "native_flatten/receipt.json" in receipt["artifacts"]


def test_cli_missing_native_tool_is_unavailable_without_silent_pca(tmp_path):
    from winding_field.surface_export import main

    field = np.indices((5, 6, 7))[0].astype(float)
    fp, mp = tmp_path / "field.npy", tmp_path / "meta.json"
    np.save(fp, field)
    mp.write_text(json.dumps(metadata(field.shape)))
    out = tmp_path / "export"
    assert (
        main(
            [
                "--field",
                str(fp),
                "--metadata",
                str(mp),
                "--level",
                "2.5",
                "--pitch",
                "1",
                "--output",
                str(out),
                "--app-run",
                str(tmp_path / "absent"),
            ]
        )
        == 2
    )
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["projection"]["status"] == "UNAVAILABLE"
    assert not (out / "grid.npz").exists()


def test_probe_seed_is_frozen_nearest_positive_prior_not_field_score():
    from winding_field.surface_probe import choose_seed

    field = np.arange(9 * 11 * 13, dtype=float).reshape(9, 11, 13)
    prior = np.zeros((11, 13), np.uint8)
    prior[5, 4] = 1
    prior[5, 8] = 255
    prior[0, 6] = 255
    meta = metadata(field.shape)
    selection = choose_seed(field, meta, prior, margin=2)
    assert selection["seed_index_zyx"] == [4, 5, 4]  # deterministic row-major tie
    np.testing.assert_allclose(selection["seed_xyz"], [18, 35, 46])
    assert selection["level"] == field[4, 5, 4]
    field[4, 5, 8] = -1e6
    assert choose_seed(field, meta, prior, margin=2)["seed_index_zyx"] == [4, 5, 4]
    with pytest.raises(ValueError, match="positive"):
        choose_seed(field, meta, np.zeros_like(prior), margin=2)


def test_probe_core_seams_use_world_coordinates_without_clamping():
    from winding_field.surface_probe import seam_registration

    owner = np.zeros((5, 6, 7), dtype=np.int16)
    owner[:, :, 3:] = 1
    meta = metadata(owner.shape, (10, 20, 30), (2, 3, 4))
    vertices = np.array([[12.0, 23, 34], [18, 23, 34], [18, 32, 34], [12, 32, 34]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    r = seam_registration(vertices, faces, meta, owner)
    assert r["tile_vertex_counts"] == {"0": 2, "1": 2}
    assert r["seam_edge_counts"] == {"0:1": 3}
    assert r["crosses_core_seams"] and r["physical_sheet_identity"] == "UNMEASURED"
    vertices[0] = [-100, -100, -100]
    assert seam_registration(vertices, faces, meta, owner)["unowned_vertices"] == 1


def test_probe_heldout_distance_is_labeled_upper_bound_and_normals_are_measured(tmp_path):
    import tifffile
    from winding_field.surface_probe import evaluate_patch

    y, x = np.indices((5, 5))
    xyz = np.stack([20 + x, 20 + y, np.full_like(x, 20)], axis=-1).astype(float)
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(tmp_path / f"{axis}.tif", xyz[..., i])
    (tmp_path / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [0.05, 0.05]}))
    vertices = xyz[[0, 0, -1, -1], [0, -1, -1, 0]]
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    owner = np.zeros((64, 64, 64), dtype=np.int8)
    meta = metadata(owner.shape, (0, 0, 0), (1, 1, 1))
    r = evaluate_patch(
        tmp_path, vertices, faces, meta, owner, patch_frame=meta["frame"], patch_voxel_um=meta["voxel_um"]
    )
    assert r["covered_human_points"] == 25
    assert r["human_to_mesh_vertex_upper_bound_um"]["p50"] > 0  # all lie on the actual triangles
    assert "UPPER BOUNDS" in r["distance_kind"]
    assert r["normal_absolute_cosine"]["n"] == 9
    assert r["normal_absolute_cosine"]["p50"] == 1
