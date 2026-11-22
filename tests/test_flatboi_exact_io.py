"""Exact flattening prep and OBJ I/O, including uncommon native corner mappings."""

import io
import tracemalloc

import numpy as np
import pytest


def _with_peak_bytes(function, *args):
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        result = function(*args)
        return result, tracemalloc.get_traced_memory()[1] - start
    finally:
        if not already_tracing:
            tracemalloc.stop()


@pytest.mark.parametrize("indexed", [True, False])
def test_block_writer_matches_savetxt_bytes(indexed):
    from winding_field.flattening.obj import write_obj

    rng = np.random.default_rng(21)
    v = rng.normal(size=(9137, 3))[::-1]
    v[1] = [-0.0, np.nextafter(0.0, 1.0), np.finfo(float).max]
    f = rng.integers(len(v), size=(10001, 3), dtype=np.int32)[:, ::-1]
    uv = v[:, 2::-2] if indexed else None
    old = io.StringIO()
    np.savetxt(old, v, fmt="v %.17g %.17g %.17g")
    if indexed:
        np.savetxt(old, uv, fmt="vt %.17g %.17g")
        np.savetxt(old, np.repeat(f + 1, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")
    else:
        np.savetxt(old, f + 1, fmt="f %d %d %d")
    new = io.StringIO()
    write_obj(new, v, f, uv)
    assert old.getvalue().encode() == new.getvalue().encode()


@pytest.mark.parametrize("dtype,offset", [(np.int32, 0), (np.uint64, 2**63 + 27)])
def test_edge_groups_preserve_lexicographic_order_orientation_and_face_owners(dtype, offset):
    from winding_field.flattening.topology import duplicate_faces, edge_groups

    f = (np.random.default_rng(42).integers(20, size=(100, 3)).astype(dtype) + offset)[::-1]
    f[3] = f[0, ::-1]
    old = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    unique, inverse, counts = np.unique(
        np.sort(old, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    signs = np.bincount(inverse, weights=np.where(old[:, 0] < old[:, 1], 1, -1))
    edges, sizes, inconsistent, order = edge_groups(f)
    np.testing.assert_array_equal(edges, unique)
    np.testing.assert_array_equal(sizes, counts)
    np.testing.assert_array_equal(order, np.argsort(inverse, kind="stable"))
    assert inconsistent == np.count_nonzero((counts == 2) & (signs != 0))
    assert duplicate_faces(f) == len(f) - len(np.unique(np.sort(f, axis=1), axis=0))


def test_native_sequential_corner_aliases_keep_last_signed_zero_and_reject_real_seam():
    from winding_field.surface_export import _bind_obj_uv

    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    tf = np.arange(6).reshape(2, 3)
    textures = v[f.ravel(), :2]
    textures[3, 0] = -0.0
    missing = np.zeros(f.shape, bool)
    got, receipt = _bind_obj_uv(v, f, v, textures, f, tf, missing, 0.0)
    assert got.tobytes() == textures[[3, 1, 4, 5]].tobytes()
    assert receipt["exact_uv_alias_occurrences"] == 2
    textures[3, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="UV seam"):
        _bind_obj_uv(v, f, v, textures, f, tf, missing, 0.0)


def test_nonsequential_alias_count_is_transitions_in_face_order_not_distinct_ids():
    from winding_field.surface_export import _bind_obj_uv

    v = np.array([[0, 0, 0.0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    tf = f.copy()
    tf[1:3, 0] = len(v)
    textures = np.vstack([v[:, :2], [-0.0, 0.0]])
    missing = np.zeros(f.shape, bool)
    got, receipt = _bind_obj_uv(v, f, v, textures, f, tf, missing, 0.0)
    assert got.tobytes() == v[:, :2].tobytes()
    assert receipt["exact_uv_alias_occurrences"] == 2
    unused = np.vstack([v, [2, 3, 4]])
    with pytest.raises(ValueError, match="unreferenced"):
        _bind_obj_uv(unused, f, unused, textures, f, tf, missing, 0.0)


def test_scaling_edges_follow_the_original_reduction_order():
    from scipy.sparse import coo_matrix

    from winding_field.flattening.native_sheet import _scaling_edges

    f = np.array([[0, 3, 2], [2, 1, 0]])
    i, j = f.T.ravel(), np.roll(f.T.ravel(), -len(f))
    adjacency = coo_matrix((np.ones(2 * i.size), (np.r_[i, j], np.r_[j, i]))).tocsc()
    first, second = _scaling_edges(adjacency)
    np.testing.assert_array_equal(
        np.column_stack([first, second]), [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
    )


def test_disk_validation_rejects_vertex_only_connection():
    from winding_field.flattening.native_sheet import _disk

    cut = {
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0], [-1.0, -1.0, 0]]),
        "faces": np.array([[0, 1, 2], [0, 3, 4]], np.int32),
        "boundary": np.arange(4),
    }
    with pytest.raises(ValueError, match="one edge component"):
        _disk(cut, None)


def test_topology_rejects_an_area_vector_for_a_different_number_of_faces():
    from winding_field.flattening.native_sheet import _topology_report

    v = np.eye(3)
    with pytest.raises(ValueError):
        _topology_report(v, np.array([[0, 1, 2]]), np.ones(2), None)


@pytest.mark.parametrize("per_corner", [False, True])
def test_order_preserving_binding_needs_no_sorted_face_or_corner_copies(per_corner):
    from winding_field.surface_export import _bind_obj_uv

    x, y = np.meshgrid(np.arange(200), np.arange(160))
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    k = (y[:-1, :-1] * 200 + x[:-1, :-1]).ravel()
    f = np.concatenate([np.stack([k, k + 1, k + 201], 1), np.stack([k, k + 201, k + 200], 1)])
    textures = v[::-1, :2].copy()
    tf = len(v) - 1 - f
    if per_corner:
        tf = np.arange(f.size).reshape(f.shape)
        textures = v[f.ravel(), :2]
    missing = np.zeros(f.shape, bool)
    (got, receipt), peak = _with_peak_bytes(_bind_obj_uv, v, f, v, textures, f, tf, missing, 0.0)
    assert got.tobytes() == v[:, :2].tobytes()
    assert receipt["exact_uv_alias_occurrences"] == (f.size - len(v) if per_corner else 0)
    assert peak < 3 * f.nbytes


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sheet_matrix_calibration_matches_original_reduction_without_rebuilding_topology(dtype):
    from winding_field.flattening.matrix import scale_uv_to_world

    y, x = np.indices((128, 128))
    v = np.column_stack((x.ravel(), y.ravel(), np.sin(x.ravel() / 13))).astype(dtype)
    uv = (v[:, :2] / 7 - 0.125)[:, ::-1]
    k = (y[:-1, :-1] * 128 + x[:-1, :-1]).ravel()
    f = np.concatenate([np.stack([k, k + 1, k + 129], 1), np.stack([k, k + 129, k + 128], 1)])
    f = f[np.random.default_rng(32).permutation(len(f))][:, ::-1]
    edges = np.unique(
        np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0
    )
    xyz_sum, uv_sum = [
        np.linalg.norm(a[edges[:, 1]] - a[edges[:, 0]], axis=1).sum() for a in (v, uv)
    ]
    scale = float(xyz_sum / uv_sum)
    (got, report), peak = _with_peak_bytes(scale_uv_to_world, v, f, uv)
    assert got.tobytes() == ((uv - uv.min(0)) * scale).tobytes()
    assert report == {
        "world_units_per_uv_unit": scale,
        "uv_origin_removed": uv.min(0).tolist(),
        "definition": "sum_unique_xyz_edge_lengths / sum_unique_uv_edge_lengths",
        "unique_edges": len(edges),
        "xyz_edge_length_sum": float(xyz_sum),
        "uv_edge_length_sum": float(uv_sum),
        "anisotropic_or_nonlinear_fit": False,
    }
    assert peak < 380 * len(f)


@pytest.mark.parametrize(
    "faces,error",
    [
        ([], "invalid global UV scale"),
        ([[0, 1, 1]], "invalid mesh or UVs for global scaling"),
        ([[0, 1, 3], [3, 1, 0]], "invalid mesh or UVs for global scaling"),
        ([[0, 1, 2], [0, 1, 3]], "invalid mesh or UVs for global scaling"),
        ([[0, 1, 2], [1, 0, 3], [0, 1, 4]], "invalid mesh or UVs for global scaling"),
    ],
)
def test_sheet_calibration_retains_empty_and_invalid_topology_errors(faces, error):
    from winding_field.flattening.original import scale_uv_to_world

    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 1]])
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match=f"^{error}$"):
        scale_uv_to_world(v, np.asarray(faces, np.int64).reshape(-1, 3), v[:, :2])


def test_sheet_calibration_retains_historical_extra_face_column_handling():
    from winding_field.flattening.original import scale_uv_to_world

    v = np.array([[0.0, 0, 3], [1, 0, 3], [0, 1, 3], [1, 1, 3]])
    # First three columns define edges; all columns participate in the duplicate check.
    f = np.array([[0, 1, 2, 2], [2, 1, 0, 3]])
    got, report = scale_uv_to_world(v, f, v[:, :2])
    assert got.tobytes() == v[:, :2].tobytes()
    assert report["unique_edges"] == 3


def test_original_map_area_gate_uses_only_the_verified_selected_faces():
    from winding_field.flattening.original import make_map

    v = np.array([[0.0, 0, 0], [1e-6, 0, 0], [0, 1e-6, 0]])
    f = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="invalid mesh or UVs for global scaling"):
        make_map(v, f)  # exactly the 1e-12 double-area threshold
    v = np.vstack([v, [[10.0, 10, 0], [12, 10, 0], [10, 12, 0]]])
    result = make_map(v, np.vstack([f, f + 3]))
    np.testing.assert_array_equal(result["source_face_rows"], [1])
    assert result["report"]["calibration"]["unique_edges"] == 3
