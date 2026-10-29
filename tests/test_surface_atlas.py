"""Seed supervision, face-preserving cuts and exact-distance bound regressions."""

import json

import numpy as np
import pytest
from test_surface_export import metadata, native_fixture


def strip():
    y, x = np.indices((3, 9))
    vertices = np.stack([10 + x, 20 + y, np.full_like(x, 30)], axis=-1).reshape(-1, 3).astype(float)
    faces = []
    for r in range(2):
        for c in range(8):
            a = r * 9 + c
            faces += [[a, a + 1, a + 9], [a + 1, a + 10, a + 9]]
    return vertices, np.array(faces)


def test_human_seed_selection_is_geometry_only_and_excludes_uv_neighborhood():
    from winding_field.surface_atlas import choose_human_seed

    y, x = np.indices((9, 9))
    xyz = np.stack([26 + x, 26 + y, np.full_like(x, 31)], axis=-1).astype(float)
    selection, excluded = choose_human_seed(xyz, metadata((64, 64, 64), (0, 0, 0), (1, 1, 1)))
    assert selection["human_seed"] is True
    assert selection["seed_uv_rc"] == [5, 5]  # (31,31,31), row-major tie around 31.5
    assert selection["seed_xyz"] == [31, 31, 31]
    assert excluded.sum() == 25 and excluded[5, 5]
    assert "level" not in selection


def test_bisection_preserves_every_face_and_global_vertex_correspondence():
    from winding_field.surface_atlas import conservation, local_chart, split_faces

    vertices, faces = strip()
    pieces = split_faces(vertices, faces, np.arange(len(faces)))
    assert len(pieces) == 2
    result = conservation(vertices, faces, pieces)
    assert result["status"] == "PASS" and result["removed_faces"] == 0
    assert result["removed_area_world2"] == pytest.approx(0)
    mapped_vertices = []
    for ids in pieces:
        v, f, global_ids = local_chart(vertices, faces, ids)
        np.testing.assert_array_equal(global_ids[f], faces[ids])
        np.testing.assert_array_equal(v, vertices[global_ids])
        mapped_vertices.append(set(global_ids))
    assert mapped_vertices[0] & mapped_vertices[1]  # shared cut vertices are retained in both charts
    assert conservation(vertices, faces, [pieces[0]])["status"] == "FAIL"
    assert conservation(vertices, faces, [pieces[0], pieces[0], pieces[1]])["status"] == "FAIL"


def test_geometry_split_separates_disconnected_pieces_without_dropping_tiny_ones():
    from winding_field.surface_atlas import split_faces

    vertices = np.array([[i * 10 + x, y, 0] for i in range(4) for x, y in [(0, 0), (1, 0), (0, 1)]], float)
    faces = np.arange(12).reshape(-1, 3)
    pieces = split_faces(vertices, faces, np.arange(4))
    assert len(pieces) == 4
    assert sorted(np.concatenate(pieces).tolist()) == [0, 1, 2, 3]


def test_point_to_triangle_search_uses_a_proven_bound_not_k_nearest_faces():
    from winding_field.surface_probe import exact_point_to_mesh

    # The nearest centroid belongs to a worse triangle; the large triangle must
    # still enter the certified candidate set through its bounding radius.
    vertices = np.array([[0.0, 0, 0], [100, 0, 0], [0, 100, 0], [1, 1, 2], [2, 1, 2], [1, 2, 2]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    d, face_ids, receipt = exact_point_to_mesh(np.array([[1.0, 1, 0.1], [-1, -1, 0]]), vertices, faces)
    np.testing.assert_allclose(d, [0.1, np.sqrt(2)])
    np.testing.assert_array_equal(face_ids, [0, 0])
    assert receipt["complete"] is True
    with pytest.raises(ValueError, match="budget"):
        exact_point_to_mesh(np.array([[1.0, 1, 0.1]]), vertices, faces, max_candidates=1)


def test_seed_neighborhood_does_not_enter_heldout_distance_denominator(tmp_path):
    import tifffile
    from winding_field.surface_probe import evaluate_patch

    y, x = np.indices((9, 9))
    xyz = np.stack([20 + x, 20 + y, np.full_like(x, 20)], axis=-1).astype(float)
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(tmp_path / f"{axis}.tif", xyz[..., i])
    (tmp_path / "meta.json").write_text(json.dumps({"format": "tifxyz"}))
    vertices = xyz[[0, 0, -1, -1], [0, -1, -1, 0]]
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    owner = np.zeros((64, 64, 64), np.int8)
    meta = metadata(owner.shape, (0, 0, 0), (1, 1, 1))
    result = evaluate_patch(
        tmp_path,
        vertices,
        faces,
        meta,
        owner,
        exclude_uv_rc=[4, 4],
        exact=True,
        patch_frame=meta["frame"],
        patch_voxel_um=meta["voxel_um"],
    )
    assert result["excluded_seed_neighborhood_points"] == 25
    assert result["covered_human_points"] == 56
    assert result["human_to_mesh_surface_distance_um"]["max"] == 0


def test_context_image_declares_actual_integer_ct_plane_and_human_supervision(tmp_path):
    from PIL import Image
    from winding_field.surface_probe import context_image

    ct = np.ones((32, 32, 32), np.uint8) * 100
    selection = dict(seed_xyz=[10, 10, 20.3], human_seed=True)
    out = tmp_path / "context.png"
    context_image(out, ct, metadata(ct.shape, (0, 0, 0), (1, 1, 1)), selection, np.array([[10, 10, 20.3]]))
    with Image.open(out) as image:
        assert image.info["CT_z_world"] == "20.0"
        assert "human seed" in image.info["Description"]


@pytest.mark.parametrize("distort", [False, True])
def test_atlas_renders_valid_charts_or_retains_every_failed_leaf(tmp_path, distort):
    from winding_field.surface_atlas import build_atlas

    app = native_fixture(tmp_path / "AppRun")
    if distort:
        app.write_text(app.read_text().replace("v[0]/10, v[1]/10", "v[0]/5, v[1]/10"))
    vertices, faces = strip()
    z, y, x = np.indices((64, 64, 64))
    ct = (x + y + z).astype(np.uint8)
    meta = metadata(ct.shape, (0, 0, 0), (1, 1, 1))
    out = tmp_path / "atlas"
    r = build_atlas(vertices, faces, meta, ct, meta, app, out, max_depth=1, max_charts=2, budget_seconds=30)
    assert r["conservation"]["status"] == "PASS"
    assert r["conservation"]["removed_faces"] == 0
    leaf_ids = [np.load(out / c["directory"] / "chart.npz")["global_face_ids"] for c in r["leaves"]]
    assert sorted(np.concatenate(leaf_ids).tolist()) == list(range(len(faces)))
    if distort:
        assert r["atlas_status"] == "FAIL"
        assert len(r["leaves"]) == 2 and not any(c["render_status"] == "PASS" for c in r["leaves"])
    else:
        assert r["atlas_status"] == "PASS" and r["geometry_status"] == "UNMEASURED"
        assert all((out / c["directory"] / "ct_preview.png").is_file() for c in r["leaves"])
