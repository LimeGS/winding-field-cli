"""Synthetic world-space geometry only; no fields, databases, or GPU contexts."""

import json
from collections.abc import Iterator
from itertools import permutations

import numpy as np
import pytest

from winding_field import surface_render


def batches(world, colors, batch_size=1):
    def stream() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(world), batch_size):
            yield world[start : start + batch_size], colors[start : start + batch_size]

    return stream


def rectangle(y0=0.0, y1=2.0, z0=0.0, z1=2.0, x=0.0):
    return np.array(
        [[[x, y0, z0], [x, y1, z0], [x, y1, z1]], [[x, y0, z0], [x, y1, z1], [x, y0, z1]]],
        dtype=np.float64,
    )


BOUNDS = [[-1.0, 0.0, 0.0], [1.0, 2.0, 2.0]]


def test_known_plane_is_filled_raster_with_world_camera_and_json_stats():
    world = rectangle(0.5, 1.5, 0.5, 1.5)
    original = world.copy()
    colors = np.tile([1.0, 0.5, 0.0], (2, 1))
    world.flags.writeable = colors.flags.writeable = False
    image, stats = surface_render.render_view(batches(world, colors), BOUNDS, width=8, height=8)

    expected_alpha = np.zeros((8, 8), dtype=np.uint8)
    expected_alpha[2:6, 2:6] = 255
    assert image.dtype == np.uint8 and image.shape == (8, 8, 4)
    np.testing.assert_array_equal(image[..., 3], expected_alpha)
    assert (image[expected_alpha == 0] == 0).all()
    assert (image[expected_alpha != 0, :3] == image[3, 3, :3]).all()
    assert 128 <= image[3, 3, 0] <= 255 and image[3, 3, 1] > 0 and image[3, 3, 2] == 0
    np.testing.assert_array_equal(world, original)
    assert stats["triangles_processed"] == 2
    assert stats["nonempty_pixels"] == 16
    assert stats["transparent_pixels"] == 48
    assert stats["camera"]["center_xyz"] == [0.0, 1.0, 1.0]
    assert stats["camera"]["scale"] == pytest.approx(4.0)
    np.testing.assert_allclose(stats["camera"]["basis_xyz"], [[0, 0, -1], [1, 0, 0], [0, 1, 0]])
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_nearest_face_and_first_equal_depth_win_across_batch_boundaries(batch_size):
    triangle = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]])
    world = np.repeat(triangle, 4, axis=0)
    world[:, :, 0] = np.array([-0.5, 0.5, 0.5, -0.8])[:, None]
    world[2] = world[2, ::-1]
    colors = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]])
    image, stats = surface_render.render_view(batches(world, colors, batch_size), BOUNDS, width=16, height=16)
    visible = image[..., 3] != 0
    assert visible.sum() > 80
    assert (image[visible, 0] == 0).all() and (image[visible, 2] == 0).all()
    assert (image[visible, 1] >= 128).all()
    assert stats["triangles_processed"] == 4


def test_depth_is_interpolated_per_pixel_for_intersecting_planes():
    flat = rectangle(x=0.0)
    slope = flat.copy()
    slope[..., 0] = slope[..., 1] - 1.0
    world = np.concatenate([flat, slope])
    colors = np.array([[1.0, 0, 0]] * 2 + [[0, 0, 1.0]] * 2)
    image, _ = surface_render.render_view(batches(world, colors, 3), BOUNDS, width=16, height=16)
    assert image[8, 3, 0] > 0 and image[8, 3, 2] == 0
    assert image[8, 12, 2] > 0 and image[8, 12, 0] == 0


def test_first_face_ties_for_oblique_duplicate_triangles_with_any_vertex_order():
    triangle = np.array([[0.13, 0.21, 0.31], [0.33, 1.61, 0.29], [0.19, 0.41, 1.73]])
    world = np.stack([triangle[list(order)] for order in permutations(range(3))])
    colors = np.array([[1.0, 0, 0]] + [[0, 0, 1.0]] * 5)
    image, _ = surface_render.render_view(
        batches(world, colors, 2), BOUNDS, width=61, height=53, azimuth=23, elevation=31
    )
    visible = image[..., 3] != 0
    assert visible.sum() > 100
    assert (image[visible, 0] > 0).all() and (image[visible, 2] == 0).all()


def test_transparent_hole_is_not_filled_between_disconnected_faces():
    world = np.concatenate([rectangle(0, 0.5), rectangle(1.5, 2)])
    colors = np.ones((4, 3))
    image, _ = surface_render.render_view(batches(world, colors, 3), BOUNDS, width=16, height=16)
    assert (image[:, :4, 3] == 255).all()
    assert (image[:, 12:, 3] == 255).all()
    assert (image[:, 4:12] == 0).all()


@pytest.mark.parametrize("factor", [1e-120, 0.125, 1.0, 8.0, 1e120])
def test_global_fit_is_invariant_under_translation_and_world_unit_scaling(factor):
    world = rectangle(0.5, 1.5, 0.5, 1.5)
    colors = np.ones((2, 3))
    bounds = np.array(BOUNDS)
    expected, _ = surface_render.render_view(
        batches(world, colors), bounds, width=37, height=29, azimuth=27, elevation=31
    )
    shift = np.array([4.0, -8.0, 16.0]) * factor
    transformed = world * factor + shift
    transformed_bounds = bounds * factor + shift
    actual, stats = surface_render.render_view(
        batches(transformed, colors), transformed_bounds, width=37, height=29, azimuth=27, elevation=31
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(stats["camera"]["center_xyz"], np.array([4, -7, 17]) * factor)


def test_framing_uses_full_aabb_even_for_empty_or_partial_streams():
    world = rectangle(0, 0.5, 0, 0.5)
    partial, stats = surface_render.render_view(batches(world, np.ones((2, 3))), BOUNDS, width=20, height=10)
    empty, empty_stats = surface_render.render_view(lambda: iter(()), BOUNDS, width=20, height=10)
    assert 0 < np.count_nonzero(partial[..., 3]) < 25
    assert not empty.any()
    assert empty_stats["camera"] == stats["camera"]
    assert empty_stats["triangles_processed"] == empty_stats["nonempty_pixels"] == 0
    assert empty_stats["transparent_pixels"] == 200


@pytest.mark.parametrize("azimuth,elevation", [(0, 0), (53, 37), (19, 90), (-70, -90), (81, 89.999999)])
def test_both_windings_and_pole_camera_have_finite_orthonormal_basis(azimuth, elevation):
    # Three orthogonal faces keep some nonzero projected area at every view.
    yz = rectangle()
    world = np.concatenate([yz, yz[..., [1, 0, 2]], yz[..., [1, 2, 0]]])
    colors = np.ones((6, 3))
    bounds = [[0, 0, 0], [2, 2, 2]]
    image, stats = surface_render.render_view(
        batches(world, colors), bounds, width=47, height=39, azimuth=azimuth, elevation=elevation
    )
    reversed_image, _ = surface_render.render_view(
        batches(world[:, ::-1], colors, 4), bounds, width=47, height=39, azimuth=azimuth, elevation=elevation
    )
    np.testing.assert_array_equal(image, reversed_image)
    basis = np.array(stats["camera"]["basis_xyz"])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.isfinite(basis).all() and np.count_nonzero(image[..., 3]) > 300
    if abs(elevation) < 89:
        assert basis[2, 1] > 0  # Camera up has positive world Z.
    json.dumps(stats, allow_nan=False)


def test_geometric_normal_shading_is_two_sided_and_has_ambient_floor():
    # Normal of this YZ-like plane is (0.8, 0.6, 0), perpendicular to (.3, -.4, sqrt(.75)).
    dark = rectangle()
    dark[..., 0] = -0.75 * dark[..., 1]
    bright = rectangle()
    colors = np.ones((2, 3))
    bounds = [[-2, 0, 0], [1, 2, 2]]
    dark_image, _ = surface_render.render_view(batches(dark, colors), bounds, width=32, height=32)
    bright_image, _ = surface_render.render_view(batches(bright, colors), bounds, width=32, height=32)
    dark_rgb = dark_image[dark_image[..., 3] > 0, :3]
    bright_rgb = bright_image[bright_image[..., 3] > 0, :3]
    assert dark_rgb.min() >= 128
    assert bright_rgb.min() > dark_rgb.max()


@pytest.mark.parametrize("footprint_scale", [1.0, 1e200])
def test_large_projected_triangle_is_clipped_and_rasterized_in_full(footprint_scale):
    world = np.array([[[0.0, -100, -100], [0.0, 1000, -100], [0.0, -100, 1000]]])
    world[..., 1:] *= footprint_scale
    image, stats = surface_render.render_view(batches(world, np.ones((1, 3))), BOUNDS, width=513, height=385)
    assert (image[..., 3] == 255).all()
    assert stats["nonempty_pixels"] == 513 * 385
    assert stats["transparent_pixels"] == 0


def test_degenerate_and_offscreen_faces_and_empty_batches_preserve_background():
    world = np.concatenate([np.zeros((1, 3, 3)), rectangle(20, 22), rectangle(z0=20, z1=22)])

    def stream():
        yield np.empty((0, 3, 3)), np.empty((0, 3))
        yield world, np.ones((len(world), 3))

    image, stats = surface_render.render_view(stream, BOUNDS, width=9, height=7)
    assert not image.any()
    assert stats["triangles_processed"] == 5


def test_flat_aabb_single_pixel_and_numpy_integer_dimensions_are_valid():
    image, stats = surface_render.render_view(
        batches(rectangle(), np.ones((2, 3))), [[0, 0, 0], [0, 2, 2]], width=np.int64(1), height=1
    )
    assert image.shape == (1, 1, 4) and image[0, 0, 3] == 255
    assert stats["nonempty_pixels"] == 1


@pytest.mark.parametrize("name", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(False), "8", None])
def test_invalid_dimensions_fail_before_consuming_stream(name, value):
    def stream():
        pytest.fail("invalid settings consumed geometry")

    with pytest.raises(ValueError):
        surface_render.render_view(stream, BOUNDS, **{name: value})


@pytest.mark.parametrize(
    "settings",
    [
        {"azimuth": np.nan},
        {"elevation": np.inf},
        {"elevation": -91},
        {"elevation": 91},
        {"azimuth": True},
        {"elevation": "30"},
        {"azimuth": [0]},
        {"azimuth": 1j},
    ],
)
def test_invalid_views_are_rejected(settings):
    with pytest.raises(ValueError):
        surface_render.render_view(lambda: iter(()), BOUNDS, **settings)


@pytest.mark.parametrize(
    "bounds",
    [
        [0, 1, 2],
        [[0, 0, 0], [1, 1, np.inf]],
        [[0, 0, np.nan], [1, 1, 1]],
        [[1, 0, 0], [0, 1, 1]],
        [[0, 0, 0], [0, 0, 0]],
        [["0"] * 3, ["1"] * 3],
        [[False] * 3, [True] * 3],
        np.zeros((2, 3), dtype=complex),
    ],
)
def test_invalid_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        surface_render.render_view(lambda: iter(()), bounds)


@pytest.mark.parametrize(
    "world,colors",
    [
        (np.zeros((3, 3)), np.ones((1, 3))),
        (np.zeros((1, 3, 2)), np.ones((1, 3))),
        (np.zeros((1, 3, 3), np.float32), np.ones((1, 3))),
        (np.zeros((1, 3, 3), int), np.ones((1, 3))),
        (np.zeros((1, 3, 3), complex), np.ones((1, 3))),
        (np.zeros((1, 3, 3), object), np.ones((1, 3))),
        (np.full((1, 3, 3), np.nan), np.ones((1, 3))),
        (np.full((1, 3, 3), np.inf), np.ones((1, 3))),
        (np.zeros((1, 3, 3)), np.ones((2, 3))),
        (np.zeros((1, 3, 3)), np.ones((1, 4))),
        (np.zeros((1, 3, 3)), np.ones((1, 3), int)),
        (np.zeros((1, 3, 3)), np.ones((1, 3), complex)),
        (np.zeros((1, 3, 3)), np.full((1, 3), -0.01)),
        (np.zeros((1, 3, 3)), np.full((1, 3), 1.01)),
        (np.zeros((1, 3, 3)), np.full((1, 3), np.nan)),
        (np.zeros((1, 3, 3)), np.full((1, 3), np.inf)),
    ],
)
def test_invalid_batches_are_rejected_even_for_degenerate_faces(world, colors):
    with pytest.raises(ValueError):
        surface_render.render_view(lambda: iter([(world, colors)]), BOUNDS, width=8, height=8)


@pytest.mark.parametrize("batch", [None, (), (np.zeros((1, 3, 3)),), ([], [])])
def test_malformed_batch_pairs_are_rejected(batch):
    with pytest.raises(ValueError):
        surface_render.render_view(lambda: iter([batch]), BOUNDS, width=8, height=8)


def test_noncallable_geometry_is_rejected():
    with pytest.raises(ValueError):
        surface_render.render_view(iter(()), BOUNDS)
