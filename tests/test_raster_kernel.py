import numpy as np
import pytest

from winding_field.flattening import original


def reference(v, f, uv, eligible, canvas, ids, tolerance):
    """Face-ordered scalar geometry oracle using the frozen recipe predicates."""
    height, width = canvas["height"], canvas["width"]
    xyz = np.full((height, width, 3), np.nan)
    bounds = np.full((height, width, 2, 3), np.nan)
    mask = np.zeros((height, width), np.uint8)
    owner = np.full(mask.shape, -1, np.int32)
    row, col = np.indices(mask.shape).reshape(2, -1)
    u0 = canvas.get("global_u0", canvas["u0"])
    top = canvas.get("global_v_top", canvas["v_top"])
    query = np.column_stack(
        (
            u0 + (col + canvas.get("column_offset", 0)) * canvas["pitch"],
            top - (row + canvas.get("row_offset", 0)) * canvas["pitch"],
        )
    )
    for i in ids:
        t = uv[f[i]]
        if not np.isfinite(t).all():
            continue
        a, b = t[1:] - t[0]
        area = a[0] * b[1] - a[1] * b[0]
        if area == 0:
            continue
        inside = original.inside_triangles(
            np.array([[0, 2, 1] if area < 0 else [0, 1, 2]]), t, np.zeros(len(row), int), query
        )
        r, c = row[inside], col[inside]
        if area < 0 or not eligible[i]:
            mask[r, c], owner[r, c], xyz[r, c], bounds[r, c] = 3, -2, np.nan, np.nan
        elif len(r):
            pts, known = original._points(v, f, uv, np.full(len(r), i), query[inside], tolerance)
            key = r[~known], c[~known]
            mask[key], owner[key], bounds[key] = 3, -2, np.nan
            for y, x, point in zip(r[known], c[known], pts[known]):
                if owner[y, x] == -1:
                    xyz[y, x], bounds[y, x], owner[y, x], mask[y, x] = point, point, i, 1
                else:
                    low, high = bounds[y, x]
                    bounds[y, x] = np.minimum(low, point), np.maximum(high, point)
                    if (
                        mask[y, x] == 3
                        or np.linalg.norm(bounds[y, x, 1] - bounds[y, x, 0]) > tolerance
                    ):
                        xyz[y, x], owner[y, x], mask[y, x] = np.nan, -2, 3
                    elif i < owner[y, x]:
                        xyz[y, x], owner[y, x] = point, i
    return xyz, mask, owner


def compare(v, f, uv, eligible, canvas, ids=None, tolerance=0.005):
    expected_xyz, expected_mask, expected_owner = reference(
        v, f, uv, eligible, canvas, range(len(f)) if ids is None else ids, tolerance
    )
    captured, calls = [], []

    def sample(xyz):
        calls.append(xyz)
        values = np.arange(len(xyz), dtype=np.float32) % 181
        values[::7] = np.nan
        return values, np.arange(len(xyz)) % 3 != 0

    image, mask, report = original.raster_sheet(
        v,
        f,
        uv,
        eligible,
        sample,
        canvas,
        capture=lambda *args: captured.extend(args),
        xyz_tolerance=tolerance,
        candidate_ids=None if ids is None else iter(ids),
    )
    expected_image = np.zeros_like(image)
    batch = max(1, 100000 // canvas["width"])
    for start in range(0, canvas["height"], batch):
        m = expected_mask[start : start + batch]
        r, c = np.where(m == 1)
        if len(r):
            assert calls.pop(0).tobytes() == expected_xyz[start + r, c].tobytes()
            k = np.arange(len(r))
            valid = (k % 7 != 0) & (k % 3 != 0)
            m[r[valid], c[valid]] = 2
            expected_image[start + r[valid], c[valid]] = np.rint((k[valid] % 181) * 255 / 180)
    assert not calls
    for got, want in zip(
        (image, mask, *captured),
        (expected_image, expected_mask, expected_xyz, expected_mask, expected_owner),
    ):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert got.tobytes() == want.tobytes()
    for code, name in enumerate(
        ("black_unmapped_pixels", "CT_missing_pixels", "CT_observed_pixels", "ambiguous_pixels")
    ):
        assert report[name] == np.count_nonzero(mask == code)
    return captured


@pytest.mark.parametrize("ids", [[0, 1], [1, 0], [0, 1, 1], [0, 1, 0], [0, 2], [0, 3], []])
def test_unknown_retains_prior_xyz_until_known_or_blocked_hit(ids):
    uv = np.tile([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], (4, 1))
    v = np.column_stack((np.full(len(uv), -0.0), uv))
    v[4, 1] = 1e60  # Uncertifiable away from the corners, even with binary128 long double.
    f = np.arange(len(uv)).reshape(-1, 3)
    f[3] = f[3, ::-1]
    compare(
        v,
        f,
        uv,
        [True, True, False, True],
        {"width": 5, "height": 5, "pitch": 1.0, "u0": 0.0, "v_top": 4.0},
        ids,
    )


@pytest.mark.parametrize("seed", [31, 72])
def test_shared_edges_ordered_overlap_and_clipped_canvas(seed):
    rng = np.random.default_rng(seed)
    uv = rng.integers(-3, 8, size=(51, 2)).astype(float)
    uv[:6] = [[0, 0], [5, 0], [5, 5], [0, 5], [np.nextafter(0.0, 1.0), 0], [np.nan, 4]]
    f = np.vstack(([[0, 1, 2], [0, 2, 3], [4, 2, 3]], rng.integers(0, len(uv), (30, 3))))
    v = np.column_stack((uv, rng.normal(size=len(uv))))
    eligible = rng.random(len(f)) > 0.1
    canvas = {
        "width": 5,
        "height": 5,
        "pitch": 0.5,
        "u0": 1.0,
        "v_top": 4.0,
        "global_u0": -1.0,
        "global_v_top": 5.0,
        "column_offset": 4,
        "row_offset": 2,
    }
    compare(v, f, uv, eligible, canvas)
    compare(v, f, uv, eligible, canvas, rng.permutation(len(f)))


def test_conflict_uses_accumulated_bounds_and_earliest_minimum_owner():
    uv = np.tile([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], (3, 1))
    f = np.arange(9).reshape(-1, 3)
    canvas = {"width": 5, "height": 5, "pitch": 1.0, "u0": 0.0, "v_top": 4.0}
    v = np.column_stack((uv, np.repeat([0.004, 0.0, 0.008], 3)))
    for ids in ([1, 0, 2], [1, 0], [0, 1], [2, 0, 1]):
        compare(v, f, uv, np.ones(3, bool), canvas, ids)


def test_large_face_spans_pixel_and_sampler_batches():
    uv = np.array([[0.0, 0.0], [540.0, 0.0], [0.0, 540.0]])
    compare(
        np.column_stack((uv, uv[:, 1] / 3)),
        np.array([[0, 1, 2]]),
        uv,
        [True],
        {"width": 420, "height": 251, "pitch": 1.0, "u0": 0.0, "v_top": 250.0},
    )


def test_index_superset_clips_near_misses_and_preserves_negative_zero():
    uv = np.array([[0, 0], [2, 0], [0, 2], [0.1, 0.1], [0.3, 0.1], [0.1, 0.3]], float)
    uv = np.vstack((uv, uv[3:] + 100))
    v = np.column_stack((uv, [-0.0, -1.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    xyz, _, owner = compare(
        v,
        np.arange(9).reshape(-1, 3),
        uv,
        [True] * 3,
        {"width": 3, "height": 3, "pitch": 1.0, "u0": 0.0, "v_top": 2.0},
        [1, 0, 2],
    )
    assert owner[2, 0] == 0 and np.signbit(xyz[2, 0, 2])


def test_far_index_false_positive_skipped_before_integer_cast_and_area():
    # Candidate scanning can reject this finite off-canvas box without evaluating
    # an unrepresentable signed area or converting its pixel bounds to integers.
    uv = np.array([[1e200, 1e200], [2e200, 1e200], [1e200, 2e200]])
    canvas = {"width": 1, "height": 1, "pitch": 1.0, "u0": 0.0, "v_top": 0.0}
    with np.errstate(over="raise", invalid="raise"):
        _, mask, _ = original.raster_sheet(
            np.eye(3),
            np.array([[0, 1, 2]]),
            uv,
            [True],
            lambda _: pytest.fail("sampled outside"),
            canvas,
            candidate_ids=iter([0]),
        )
    assert mask[0, 0] == 0


def test_column_face_mask_preserves_blocked_pixels():
    uv = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    compare(
        np.column_stack((uv, uv[:, 0])),
        f,
        uv,
        np.array([[True], [False]]),
        {"width": 5, "height": 5, "pitch": 1.0, "u0": 0.0, "v_top": 4.0},
    )


def test_float64_degenerate_even_if_longdouble_distinguishes_edges():
    n = float(2**27)
    uv = np.array([[0.0, 0.0], [n, n - 1], [n + 1, n]])
    _, mask, _ = compare(
        np.column_stack((uv, np.zeros(3))),
        np.array([[0, 1, 2]]),
        uv,
        [True],
        {"width": 2, "height": 2, "pitch": 1.0, "u0": 0.0, "v_top": 1.0},
    )
    assert not mask.any()


def test_ill_conditioned_corner_respects_platform_longdouble_precision():
    # det=1, but the recipe's denominator error bound is about 2 on binary64.
    n = float(2**24 + 1)
    uv = np.array([[0.0, 0.0], [n, n - 1], [n + 1, n]])
    _, mask, owner = compare(
        np.column_stack((uv, np.zeros(3))),
        np.array([[0, 1, 2]]),
        uv,
        [True],
        {"width": 1, "height": 1, "pitch": 1.0, "u0": 0.0, "v_top": 0.0},
    )
    if np.finfo(np.longdouble).nmant > 52:
        assert mask[0, 0] == 1 and owner[0, 0] == 0
    else:
        assert mask[0, 0] == 3 and owner[0, 0] == -2


def test_bounds_retained_across_candidate_batches():
    uv = np.tile([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], (4, 1))
    uv[9:] += 30.0
    v = np.column_stack((uv, np.repeat([0.004, 0.0, 0.008, 0.0], 3)))
    # The first two surfaces agree, but the range of all three exceeds tolerance.
    xyz, mask, owner = compare(
        v,
        np.arange(12).reshape(-1, 3),
        uv,
        [True] * 4,
        {"width": 3, "height": 3, "pitch": 1.0, "u0": 0.0, "v_top": 2.0},
        [0, 1] + [3] * 2046 + [2],
    )
    assert mask[2, 0] == 3 and owner[2, 0] == -2 and np.isnan(xyz[2, 0]).all()


def test_candidate_guard_can_cancel_before_sampling():
    checked = []

    def guard():
        checked.append(True)
        if len(checked) == 2:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError, match="cancel"):
        original.raster_sheet(
            np.eye(3),
            np.array([[0, 1, 2]]),
            np.eye(3)[:, :2],
            [True],
            lambda _: pytest.fail("sampled after cancellation"),
            {"width": 2, "height": 2, "pitch": 1.0, "u0": 0.0, "v_top": 1.0},
            candidate_ids=iter([0]),
            guard=guard,
        )
