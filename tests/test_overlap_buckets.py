import time

import numpy as np
import pytest

import winding_field.flattening.legacy_charts as flatten


def mesh(triangles):
    uv = np.asarray(triangles, dtype=float).reshape(-1, 2)
    return uv, np.arange(len(uv)).reshape(-1, 3)


def brute_pairs(uv, faces):
    # Independent scalar clipping reference, including the original asymmetric area threshold.
    triangles = uv[faces]
    result = []

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for i, a in enumerate(triangles):
        for j in range(i + 1, len(triangles)):
            b = triangles[j]
            if np.any(np.minimum(a.max(0), b.max(0)) - np.maximum(a.min(0), b.min(0)) <= 1e-9):
                continue
            if any(
                max(cross(q - p, r - p) for r in other) <= 0
                for t, other in ((a, b), (b, a))
                for p, q in zip(t, np.roll(t, -1, axis=0))
            ):
                continue
            polygon = list(b)
            for p, q in zip(a, np.roll(a, -1, axis=0)):
                clipped = []
                for s, t in zip(polygon, polygon[1:] + polygon[:1]):
                    ds = cross(q - p, s - p)
                    dt = cross(q - p, t - p)
                    if ds >= 0:
                        clipped.append(s)
                    if (ds >= 0) != (dt >= 0):
                        clipped.append(s + (t - s) * ds / (ds - dt))
                polygon = clipped
            if len(polygon) < 3:
                continue
            polygon = np.asarray(polygon) - polygon[0]
            area = (
                abs(sum(cross(p, q) for p, q in zip(polygon, np.roll(polygon, -1, axis=0)))) * 0.5
            )
            if area > max(abs(cross(a[1] - a[0], a[2] - a[0])) * 1e-7, 1e-10):
                result.append((i, j))
    return result


def test_distant_large_face_does_not_inflate_every_small_face_query():
    uv, faces = mesh(
        [[[i * 10, 0], [i * 10 + 1, 0], [i * 10, 1]] for i in range(128)]
        + [[[100000, 100000], [200000, 100000], [100000, 200000]]]
    )
    report = {}
    assert list(flatten.iter_overlap_pairs(uv, faces, max_candidates=4, report=report)) == []
    assert report["complete"] and report["narrow_tests"] < 128


@pytest.mark.parametrize(
    "scale,shift", [(1.0, [0.0, 0.0]), (1e-3, [3.0, -2.0]), (1e3, [-500.0, 800.0])]
)
def test_crowded_aabb_chunk_matches_brute_force(scale, shift):
    rng = np.random.default_rng(4)
    crowd = rng.normal(size=(30, 3, 2)) * 0.3
    crowd[:, :, 0] += np.linspace(0, 0.4, 30)[:, None]
    sliver = np.array([[[0.0, 0.0], [2.0, 1e-9], [4.0, 0.0]]])
    contact = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]])
    giant = np.array([[[1000.0, 1000.0], [2000.0, 1000.0], [1000.0, 2000.0]]])
    triangles = (
        np.concatenate([crowd, sliver, contact, giant]) * scale + np.asarray(shift)
    ).reshape(-1, 3, 2)
    uv, faces = mesh(triangles)
    expected = brute_pairs(uv, faces)
    report = {}
    assert list(flatten.iter_overlap_pairs(uv, faces, max_candidates=3, report=report)) == expected
    assert report["complete"] and report["pairs"] == len(expected)


def test_scattered_face_order_does_not_multiply_traversal():
    # Real saved-domain aggregates are not spatially ordered; block bounding boxes are not usable.
    uv, faces = mesh([[[i * 3, 0], [i * 3 + 1, 0], [i * 3, 1]] for i in range(20000)])
    perm = np.random.default_rng(0).permutation(len(faces))
    shuffled = uv.reshape(-1, 3, 2)[perm].reshape(-1, 2)
    shuffled_faces = np.arange(len(faces) * 3).reshape(-1, 3)
    report = {}
    started = time.perf_counter()
    assert list(flatten.iter_overlap_pairs(shuffled, shuffled_faces, report=report)) == []
    assert time.perf_counter() - started < 5 and report["complete"] and report["narrow_tests"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_bucket_pairs_match_scalar_reference_in_order(seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(45, 1, 2)) * 3
    triangles = centres + rng.normal(size=(45, 3, 2)) * np.exp2(
        rng.integers(-4, 5, size=(45, 1, 1))
    )
    signed = flatten.cross2(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    triangles[signed < 0] = triangles[signed < 0, ::-1]
    uv, faces = mesh(triangles)
    expected = brute_pairs(uv, faces)
    report = {}
    assert (
        list(flatten.iter_overlap_pairs(uv, faces, max_candidates=128, report=report)) == expected
    )
    assert report["complete"] and report["pairs"] == len(expected)


def test_tiny_candidate_chunk_never_limits_auditable_mesh_size():
    uv, faces = mesh([[[0, 0], [s, 0], [0, s]] for s in (1, 2, 4, 8, 16, 32)])
    report = {}
    assert len(list(flatten.iter_overlap_pairs(uv, faces, max_candidates=5, report=report))) == 15
    assert report["complete"] and report["pairs"] == 15


def test_shared_vertices_edges_duplicates_and_tiny_intersections():
    uv = np.array([[0.0, 0.0], [2, 0], [0, 2], [2, 2], [1, 0], [0, 1], [2 - 1e-8, 0], [2, 1e-8]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 5], [0, 1, 2], [6, 1, 7], [2, 1, 0]])
    assert list(flatten.iter_overlap_pairs(uv, faces, max_candidates=36)) == brute_pairs(uv, faces)
