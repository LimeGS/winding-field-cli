"""--field-prior: a completed field seeds and gauges a new (finer, shifted) solve."""

import json

import numpy as np
import pytest

from test_field import prepared_run


def _tile(root, shape, shift_x=0.0, frame="test"):
    run = prepared_run(root, shape=shape)
    meta = json.loads((run / "PREPARED.json").read_bytes())
    meta["origin_xyz"][0] += shift_x  # a CT ROI tile: same frame, shifted origin
    meta["frame"] = frame
    (run / "PREPARED.json").write_text(json.dumps(meta))
    return run


def _compare(fine, coarse):
    from winding_field import field
    from winding_field.field_prior import FieldPrior

    geometry = json.loads((fine / "FIELD.json").read_bytes())["geometry"]
    with field.open_field(fine) as source:
        values, valid = source.read([0, 0, 0], geometry["shape_zyx"])
    source = json.loads((fine / "PREPARED.json").read_bytes()).get("source")
    prior = FieldPrior(coarse, geometry, source)
    try:
        expected, supported = prior.sample(
            field.coordinates([0, 0, 0], geometry["shape_zyx"], geometry)
        )
    finally:
        prior.close()
    both = valid & supported
    assert both.any()
    return float(np.median(np.abs(values[both] - expected[both])))


@pytest.fixture
def coarse(tmp_path):
    from winding_field import field

    run = _tile(tmp_path / "coarse", (6, 6, 32))
    assert field.solve(run, dict(sweeps=3, field_chunk=8))["status"] == "FIELD_READY"
    return run


def test_shifted_fine_tile_inherits_the_prior_gauge(tmp_path, coarse):
    from winding_field import field

    config = dict(sweeps=3, field_chunk=8, field_spacing_um=1.0)
    alone = _tile(tmp_path / "alone", (6, 6, 16), shift_x=20.0)
    field.solve(alone, config)
    seeded = _tile(tmp_path / "seeded", (6, 6, 16), shift_x=20.0)
    receipt = field.solve(seeded, dict(config, field_prior=str(coarse)))
    assert receipt["status"] == "FIELD_READY"
    # Without a prior the tile pins its own first node to zero: a different W numbering.
    assert _compare(alone, coarse) > 0.5
    # With the prior, the same sheet has the same W in both fields.
    assert _compare(seeded, coarse) < 1e-6
    identity = json.loads((seeded / "field/CHECKPOINT.json").read_bytes())["identity"]
    assert identity["boundary"] == "caller-fixed"
    recorded = identity["source_identity"]["field_prior"]
    assert recorded["field_json_sha256"] and recorded["anchor"]["index_zyx"] == [0, 0, 0]
    receipt = json.loads((seeded / "FIELD.json").read_bytes())
    assert receipt["field_prior"] == recorded


def test_prior_in_another_frame_is_rejected(tmp_path, coarse):
    from winding_field import field

    other = _tile(tmp_path / "other", (6, 6, 16), frame="elsewhere")
    with pytest.raises(ValueError, match="frame"):
        field.solve(other, dict(sweeps=1, field_chunk=8, field_prior=str(coarse)))


@pytest.mark.parametrize("value", ["", 3, True])
def test_field_prior_must_name_a_run(value):
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="field_prior"):
        validate_config({"field_prior": value})


def _with_source(run, frame, **changes):
    meta = json.loads((run / "PREPARED.json").read_bytes())
    meta["frame"] = frame
    meta["source"] = dict(
        source="/data/scroll.zarr",
        level=0,
        shape=[6, 6, 32],
        spacing_xyz=[1.0, 1.0, 1.0],
        origin_xyz=[0.0, 0.0, 0.0],
        metadata_sha256="ab" * 32,
    )
    meta["source"].update(changes)
    (run / "PREPARED.json").write_text(json.dumps(meta))
    return run


def test_prior_from_a_release_without_reader_metadata_is_the_same_frame(tmp_path):
    """Earlier releases did not record the CT reader in the frame digest; the CT is the same."""
    from winding_field import field

    old = _with_source(_tile(tmp_path / "old", (6, 6, 32)), "old-reader")
    assert field.solve(old, dict(sweeps=3, field_chunk=8))["status"] == "FIELD_READY"
    reader = dict(backend="vc_reader-zarr-v2-checked-v1", native_runtime={"extension": "x"})
    new = _with_source(_tile(tmp_path / "new", (6, 6, 16), shift_x=20.0), "new-reader", **reader)
    config = dict(sweeps=3, field_chunk=8, field_spacing_um=1.0, field_prior=str(old))
    assert field.solve(new, config)["status"] == "FIELD_READY"
    assert _compare(new, old) < 1e-6
    recorded = json.loads((new / "FIELD.json").read_bytes())["field_prior"]["frame_match"]
    assert recorded["rule"] == "ct-identity-without-reader-runtime"

    other = _with_source(
        _tile(tmp_path / "other", (6, 6, 16)), "new-reader", **dict(reader, level=1)
    )
    with pytest.raises(ValueError, match="CT identity differs in level"):
        field.solve(other, dict(config, field_prior=str(old)))


def test_supported_scope_solves_and_stores_only_chunks_with_observations(tmp_path, coarse):
    from test_field import prepared_run
    from winding_field import field

    def half(name):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), support_x_below=16)
        meta = json.loads((run / "PREPARED.json").read_bytes())
        meta["frame"] = "test"
        (run / "PREPARED.json").write_text(json.dumps(meta))
        return run

    config = dict(sweeps=3, field_chunk=8, field_prior=str(coarse))
    full, scoped = half("full"), half("scoped")
    assert field.solve(full, config)["status"] == "FIELD_READY"
    receipt = field.solve(scoped, dict(config, field_scope="supported"))
    assert receipt["status"] == "FIELD_READY"
    assert receipt["field_scope"] == dict(
        mode="supported-chunks-v1", owned_chunks=2, total_chunks=4
    )
    checkpoint = json.loads((scoped / "field/CHECKPOINT.json").read_bytes())
    assert sorted(checkpoint["chunks"]) == ["0-0-0", "0-0-1"]
    assert checkpoint["identity"]["order_scope"] == "partial"
    shape = receipt["geometry"]["shape_zyx"]
    with field.open_field(scoped) as source:
        values, valid = source.read([0, 0, 0], shape)
    assert not valid[..., 16:].any() and valid[..., :16].all()
    # The skipped half is held at the prior and the solved half, with the same data as the
    # prior, stays on it: only the empty-side edge terms pull the last owned nodes toward
    # the held values (in scope "all" the empty half flattens instead).
    assert _compare(scoped, coarse) < 0.05
    assert _compare(full, coarse) < 0.05
    # A resume reuses the cached chunk scope.
    assert json.loads((scoped / "FIELD_SCOPE.json").read_bytes())["owned"] == [
        [0, 0, 0],
        [0, 0, 1],
    ]


@pytest.mark.parametrize(
    "config",
    [
        {"field_scope": "some"},
        {"field_scope": "supported"},
        {"field_scope": "supported", "field_prior": "coarse", "field_coarse_factor": 8},
    ],
)
def test_field_scope_is_validated(config):
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="field_scope"):
        validate_config(config)


def test_inherited_calibration_uses_the_priors_centre_and_is_recorded(tmp_path, coarse):
    from winding_field import field
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="requires field_prior"):
        validate_config({"field_prior_calibration": "inherit"})
    with pytest.raises(ValueError, match="own or inherit"):
        validate_config({"field_prior": str(coarse), "field_prior_calibration": "global"})
    config = dict(sweeps=3, field_chunk=8, field_spacing_um=1.0, field_prior=str(coarse))
    own = _tile(tmp_path / "own", (6, 6, 16), shift_x=20.0)
    field.solve(own, config)
    tile = _tile(tmp_path / "inherit", (6, 6, 16), shift_x=20.0)
    receipt = field.solve(tile, dict(config, field_prior_calibration="inherit"))
    assert receipt["status"] == "FIELD_READY"
    prior_cal = json.loads((coarse / "CALIBRATION.json").read_bytes())
    # A shifted tile's own centroid differs from the prior's; inheriting keeps the prior's.
    own_cal = json.loads((own / "CALIBRATION.json").read_bytes())
    assert own_cal["centre_xyz"] != prior_cal["centre_xyz"]
    assert json.loads((tile / "CALIBRATION.json").read_bytes()) == prior_cal
    identity = json.loads((tile / "field/CHECKPOINT.json").read_bytes())["identity"]
    assert identity["source_identity"]["calibration_source"]["kind"] == "field_prior"
    own_identity = json.loads((own / "field/CHECKPOINT.json").read_bytes())["identity"]
    assert "calibration_source" not in own_identity["source_identity"]


class _GridSource:
    """A FieldSource stand-in over an in-memory ZYX grid."""

    def __init__(self, values, support):
        self.values, self.support = values, support

    def read(self, lo, hi):
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return self.values[box], self.support[box]


@pytest.mark.parametrize("shape", [(9, 11, 8), (1, 6, 5)])
def test_lattice_points_take_the_same_values_as_scattered_points(shape):
    """A box of grid nodes is interpolated per axis; the result is bitwise the point path's,
    including nodes outside the prior and on unsupported corners."""
    from winding_field.field import coordinates
    from winding_field.field_prior import FieldPrior

    rng = np.random.default_rng(3)
    prior = object.__new__(FieldPrior)
    prior.origin = np.array([5.0, -3.0, 2.0])
    prior.spacing = np.array([4.0, 4.0, 4.0])
    prior.shape = np.asarray(shape, int)
    prior.source = _GridSource(rng.standard_normal(shape), rng.random(shape) > 0.2)
    meta = dict(spacing_xyz=np.array([1.5, 1.5, 1.5]), origin_xyz=np.array([1.0, -6.0, 0.5]))
    lattice = coordinates((0, 0, 0), (26, 34, 24), meta)
    values, valid = prior.sample(lattice)
    flat_values, flat_valid = prior.sample(lattice.reshape(-1, 3))
    assert np.array_equal(values.ravel(), flat_values, equal_nan=True)
    assert np.array_equal(valid.ravel(), flat_valid)
    assert valid.any() and not valid.all()


def test_coarse_factor_sets_the_production_field_defaults_for_unset_flags():
    from winding_field.config import field_defaults

    assert field_defaults({}) == dict(field_chunk=64)
    assert field_defaults(dict(field_coarse_factor=0, field_halo=3)) == dict(
        field_coarse_factor=0, field_halo=3, field_chunk=64
    )
    assert field_defaults(dict(field_coarse_factor=8)) == dict(
        field_coarse_factor=8, field_chunk=128, field_halo=8, field_kernel="fused"
    )
    explicit = dict(field_coarse_factor=8, field_chunk=64, field_halo=4, field_kernel="eager")
    assert field_defaults(dict(explicit)) == explicit


def _grid_prior(function, origin_xyz, spacing_xyz, shape_zyx, observed=None, stored=None):
    """A FieldPrior over an in-memory grid: sample() reads the ``observed`` nodes and
    gradient() the ``stored`` ones (both default to every node)."""
    from winding_field.field_prior import FieldPrior

    prior = object.__new__(FieldPrior)
    prior.origin = np.asarray(origin_xyz, float)
    prior.spacing = np.asarray(spacing_xyz, float)
    prior.shape = np.asarray(shape_zyx, int)
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape_zyx), indexing="ij")
    xyz = np.stack((x, y, z), -1) * prior.spacing + prior.origin
    values = function(xyz)
    everywhere = np.ones(values.shape, bool)
    prior.source = _GridSource(values, everywhere if observed is None else observed)
    prior._stored = _GridSource(values, everywhere if stored is None else stored)
    return prior


def test_prior_gradient_is_a_continuous_central_difference():
    slope = np.array([0.5, -0.25, 2.0])
    linear = _grid_prior(lambda p: p @ slope, (10.0, 20.0, 30.0), (2.0, 3.0, 4.0), (6, 7, 8))
    inside = np.array([[15.0, 30.0, 40.0], [16.3, 31.1, 41.7]])
    gradient, valid = linear.gradient(inside)
    assert valid.all() and np.allclose(gradient, slope)
    # Each axis needs its +-h samples inside the supported prior: one spacing from an edge
    # (x = 10 is the first node) is still valid, closer is not.
    gradient, valid = linear.gradient(np.array([[12.0, 30.0, 40.0], [11.0, 30.0, 40.0]]))
    assert valid.tolist() == [True, False] and (gradient[1] == 0).all()
    # A curved prior: no jump at a coarse cell face (x = 16), so a node shared by two tiles
    # gets the same sign whichever cell its coordinates round into.
    curved = _grid_prior(lambda p: p[..., 0] ** 2, (10.0, 20.0, 30.0), (2.0, 3.0, 4.0), (6, 7, 8))
    face = np.array([[16.0, 30.0, 40.0], [16.0 - 1e-9, 30.0, 40.0], [16.0 + 1e-9, 30.0, 40.0]])
    gradient, valid = curved.gradient(face)
    assert valid.all() and np.ptp(gradient[:, 0]) < 1e-6
    # Bounded batches give the same answer as one pass.
    points = np.random.default_rng(0).uniform([13, 24, 35], [18, 35, 50], (50, 3))
    whole, _ = curved.gradient(points)
    batched, _ = curved.gradient(points, max_points=7)
    assert np.array_equal(whole, batched)


def test_prior_gradient_reads_stored_values_observed_or_not():
    """A coarse solved over the whole scroll observes only part of it but stores its best
    estimate everywhere it solved: the gradient reads every stored, finite value, while
    sample() (initial field, gauge) keeps to the observed ones."""
    slope = np.array([0.5, -0.25, 2.0])
    shape = (6, 7, 8)  # x nodes at 10, 12, ..., 24
    observed = np.zeros(shape, bool)
    observed[..., :3] = True  # x <= 14 observed
    stored = np.ones(shape, bool)
    stored[..., 7] = False  # no payload at x = 24
    prior = _grid_prior(
        lambda p: p @ slope, (10.0, 20.0, 30.0), (2.0, 3.0, 4.0), shape, observed, stored
    )
    # x = 19 reads x = 16..22 (stored, unobserved); x = 21 reaches x = 24 (not stored).
    points = np.array([[19.0, 30.0, 40.0], [21.0, 30.0, 40.0]])
    _, sampled = prior.sample(points)
    assert not sampled.any()
    gradient, valid = prior.gradient(points)
    assert valid.tolist() == [True, False]
    assert np.allclose(gradient[0], slope) and (gradient[1] == 0).all()
    # A stored but non-finite corner leaves the gradient undefined.
    holed = _grid_prior(
        lambda p: np.where(p[..., 0] == 20.0, np.nan, p @ slope),
        (10.0, 20.0, 30.0),
        (2.0, 3.0, 4.0),
        shape,
    )
    gradient, valid = holed.gradient(points[:1])
    assert not valid[0] and (gradient[0] == 0).all()


def test_prior_gradient_orients_the_pressed_core_of_a_flattened_scroll():
    """Sheets pressed flat (normals ~ +-y, tilted 20 degrees) 22 mm along x from the centre.

    Radially, the sheet below the core plane takes the tilt's sign and points inward. The
    prior (W grows with |y|, a valley at the core plane) signs both sides outward, drops
    the valley floor where its gradient vanishes, and leaves the radial rule outside its
    support.
    """
    from winding_field.field import oriented

    prior = _grid_prior(
        lambda p: abs(p[..., 1]) / 103.0, (20000.0, -3000.0, -1000.0), (500.0,) * 3, (5, 13, 9)
    )
    cal = dict(mode="radial-z", centre_xyz=[0.0, 0.0, 0.0], axis_xyz=[0.0, 0.0, 1.0])
    tilt = np.deg2rad(20.0)
    sheet = np.array([np.sin(tilt), np.cos(tilt), 0.0])
    xyz = np.array(
        [
            [22000.0, 1000.0, 0.0],  # above the core plane: outward is +y
            [22000.0, -1000.0, 0.0],  # below: outward is -y
            [22000.0, 0.0, 0.0],  # the valley floor
            [30000.0, -1000.0, 0.0],  # outside the prior
        ]
    )
    # Normals carry no sign of their own.
    flips = np.array([1.0, -1.0, 1.0, -1.0])
    data = dict(
        normals=(sheet * flips[:, None]).astype(np.float32),
        density=np.full(4, 1 / 141.0, np.float32),
        support=np.ones(4, bool),
    )
    target, weights, valid = oriented(dict(data), xyz, cal)
    assert valid.tolist() == [True, True, True, True]
    assert target[1, 1] > 0  # radial: points inward below the core plane
    stats = {}
    target, weights, valid = oriented(dict(data), xyz, cal, prior, stats)
    assert valid.tolist() == [True, True, False, True]
    assert stats == dict(
        supported=4,
        prior_signed=2,
        prior_dropped=1,
        prior_inconsistent=0,
        radial_signed=1,
        radial_dropped=0,
    )
    assert target[0, 1] > 0 and target[1, 1] < 0
    assert (target[2] == 0).all() and weights[2] == np.float32(1e-3)
    radial, _, _ = oriented(dict(data), xyz[3:], cal)
    assert np.array_equal(target[3], radial[0])
    # A prior gradient weaker than 0.3 x the sheet density signs nothing.
    weak = _grid_prior(
        lambda p: abs(p[..., 1]) / 1000.0, (20000.0, -3000.0, -1000.0), (500.0,) * 3, (5, 13, 9)
    )
    _, _, valid = oriented(dict(data), xyz, cal, weak)
    assert valid.tolist() == [False, False, False, True]


_UNIT_LATTICE = dict(origin_xyz=[0.0, 0.0, 0.0], spacing_xyz=[1.0, 1.0, 1.0])
_PLANAR = dict(mode="planar", axis_xyz=[0.0, 1.0, 0.0], centre_xyz=[0.0, 0.0, 0.0])


def _sheet_box(lo, hi, normal):
    """Map data on lattice nodes lo..hi (ZYX): every node supported, unit density, and
    ``normal(xyz)`` with an arbitrary sign per node, as unsigned normals come."""
    from winding_field.field import coordinates

    xyz = coordinates(lo, hi, _UNIT_LATTICE)
    shape = xyz.shape[:-1]
    flips = np.where((xyz @ [7.0, 3.0, 1.0]) % 2 < 1, -1.0, 1.0)
    data = dict(
        normals=(normal(xyz) * flips[..., None]).astype(np.float32),
        density=np.ones(shape, np.float32),
        support=np.ones(shape, bool),
    )
    return data, xyz


def _sheet_oriented(prior, lo, hi, normal, share=0.3, least=0, stats=None):
    from winding_field.field import oriented

    data, xyz = _sheet_box(np.asarray(lo), np.asarray(hi), normal)
    sheet = dict(max_disagreement=share, min_probes=least)
    return oriented(data, xyz, _PLANAR, prior, stats, sheet, (np.asarray(lo), _UNIT_LATTICE))


def _horizontal(xyz):
    return np.broadcast_to(np.array([0.0, 1.0, 0.0]), xyz.shape)


def test_sheet_check_drops_the_nodes_where_the_prior_floor_crosses_a_sheet():
    """Horizontal sheets; the prior's valley floor (W = |h|) runs at 45 degrees to them, so
    its sign flips along every sheet where the floor crosses it. Only the nodes next to the
    crossing, on both sides, see more than 30% of their sheet against them."""
    from winding_field.field import ORIENTATION_CLASSES, oriented

    def height(p):  # above (+) or below (-) the floor, in nodes along y
        return p[..., 1] - 16.0 - (p[..., 0] - 16.0)

    prior = _grid_prior(lambda p: abs(height(p)), (-10.0,) * 3, (1.0,) * 3, (26, 50, 56))
    lo, hi = (0, 4, 0), (6, 29, 33)
    stats = {}
    target, weights, valid = _sheet_oriented(prior, lo, hi, _horizontal, stats=stats)
    data, xyz = _sheet_box(np.asarray(lo), np.asarray(hi), _horizontal)
    plain = oriented(dict(data), xyz, _PLANAR, prior)
    flagged = plain[2] & ~valid
    assert sorted(set(height(xyz[flagged]).tolist())) == [-1.0, 1.0]
    assert stats["prior_inconsistent"] == np.count_nonzero(flagged)
    assert sum(stats[k] for k in ORIENTATION_CLASSES) == stats["supported"]
    assert (target[flagged] == 0).all() and (weights[flagged] == np.float32(1e-3)).all()
    kept = ~flagged
    assert np.array_equal(target[kept], plain[0][kept])
    # The share is the threshold: 40% flags nothing here, and 1 is the v2 rule bit for bit.
    assert np.array_equal(_sheet_oriented(prior, lo, hi, _horizontal, share=0.4)[2], plain[2])
    for got, expected in zip(_sheet_oriented(prior, lo, hi, _horizontal, share=1.0), plain):
        assert np.array_equal(got, expected)
    # Pointwise on the global prior: two boxes whose faces cut the probes' reach (overlap
    # x 12..20 around the crossing at x 16) agree with one box node for node.
    for cut in (12, 16, 20):
        left = _sheet_oriented(prior, lo, (6, 29, cut + 4), _horizontal)
        right = _sheet_oriented(prior, (0, 4, cut - 4), hi, _horizontal)
        for whole, a, b in zip((target, weights, valid), left, right):
            assert np.array_equal(a, whole[:, :, : cut + 4])
            assert np.array_equal(b, whole[:, :, cut - 4 :])
    # min_probes: where probes leave the stored prior (x > 22), fewer usable ones remain
    # and requiring all 16 keeps those nodes.
    stored = np.ones((26, 50, 56), bool)
    stored[..., 33:] = False
    partial = _grid_prior(
        lambda p: abs(height(p)), (-10.0,) * 3, (1.0,) * 3, stored.shape, None, stored
    )
    some = ~_sheet_oriented(partial, lo, hi, _horizontal)[2]
    all16 = ~_sheet_oriented(partial, lo, hi, _horizontal, least=16)[2]
    assert (all16 <= some).all() and np.count_nonzero(all16) < np.count_nonzero(some)


def test_sheet_check_keeps_a_fold_the_prior_resolves():
    """A hairpin: sheets wrap a spine (x 12..28 at y 20) as stadiums. Probes in a node's
    tangent plane stay outside its stadium, where the prior still points away from the
    spine, so no node is flagged and every kept target points outward."""

    def offset(p):
        return np.stack(
            (p[..., 0] - np.clip(p[..., 0], 12, 28), p[..., 1] - 20, np.zeros(p.shape[:-1])), -1
        )

    def outward(p):
        d = offset(p)
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)

    prior = _grid_prior(
        lambda p: np.linalg.norm(offset(p), axis=-1), (-12.0,) * 3, (1.0,) * 3, (30, 60, 66)
    )
    stats = {}
    target, _, valid = _sheet_oriented(prior, (0, 4, 0), (6, 37, 41), outward, stats=stats)
    assert stats["prior_inconsistent"] == 0 and stats["prior_signed"] > 0.9 * stats["supported"]
    _, xyz = _sheet_box(np.array((0, 4, 0)), np.array((6, 37, 41)), outward)
    assert (np.sum(target[valid] * outward(xyz)[valid], axis=-1) > 0).all()


def test_prior_gradient_orientation_is_validated_and_recorded(tmp_path, coarse, capsys):
    from winding_field import field
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="requires field_prior"):
        validate_config({"field_orientation": "prior-gradient"})
    with pytest.raises(ValueError, match="radial or prior-gradient"):
        validate_config({"field_prior": str(coarse), "field_orientation": "normal"})
    gradient = {"field_prior": str(coarse), "field_orientation": "prior-gradient"}
    for key, value, message in (
        ("field_orientation_max_disagreement", 1.5, "from 0 to 1"),
        ("field_orientation_max_disagreement", True, "from 0 to 1"),
        ("field_orientation_min_probes", 17, "from 0 to 16"),
        ("field_orientation_min_probes", 2.0, "from 0 to 16"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_config({**gradient, key: value})
    with pytest.raises(ValueError, match="sheet check requires field_orientation prior-gradient"):
        validate_config({"field_prior": str(coarse), "field_orientation_min_probes": 4})
    validate_config(
        {**gradient, "field_orientation_max_disagreement": 0, "field_orientation_min_probes": 16}
    )
    config = dict(
        sweeps=3,
        field_chunk=8,
        field_spacing_um=1.0,
        field_prior=str(coarse),
        field_prior_calibration="inherit",
    )
    radial = _tile(tmp_path / "radial", (6, 6, 16), shift_x=20.0)
    assert field.solve(radial, config)["status"] == "FIELD_READY"
    tile = _tile(tmp_path / "prior", (6, 6, 16), shift_x=20.0)
    receipt = field.solve(tile, dict(config, field_orientation="prior-gradient"))
    assert receipt["status"] == "FIELD_READY"
    identity = json.loads((tile / "field/CHECKPOINT.json").read_bytes())["identity"]
    rule = identity["source_identity"]["orientation"]
    assert rule["rule"] == "prior-gradient-central-v3"
    assert rule["prior_values"] == "stored-finite"
    assert rule["sheet_check"] == dict(
        rule="tangent-probes-v1",
        radii_nodes=[3, 6],
        directions=8,
        max_disagreement=0.3,
        min_probes=0,
    )
    # The receipt splits the supported nodes by where their sign came from, and logs it.
    counts = json.loads((tile / "FIELD.json").read_bytes())["orientation"]
    assert counts["rule"] == rule["rule"] and counts["supported"] > 0
    assert sum(counts[name] for name in field.ORIENTATION_CLASSES) == counts["supported"]
    assert "prior_inconsistent" in field.ORIENTATION_CLASSES
    assert counts["radial_fallback"] == counts["radial_signed"] + counts["radial_dropped"]
    assert counts["radial_fallback_share"] == counts["radial_fallback"] / counts["supported"]
    log = capsys.readouterr().err
    assert "orientation prior-gradient over" in log and "dropped by the sheet check" in log
    # A share of 1 flags nothing and keeps the v2 identity, so v2 runs resume.
    v2 = _tile(tmp_path / "v2", (6, 6, 16), shift_x=20.0)
    off = dict(config, field_orientation="prior-gradient", field_orientation_max_disagreement=1)
    assert field.solve(v2, off)["status"] == "FIELD_READY"
    v2_rule = json.loads((v2 / "field/CHECKPOINT.json").read_bytes())["identity"]
    assert v2_rule["source_identity"]["orientation"] == dict(
        rule="prior-gradient-central-v2",
        prior_values="stored-finite",
        min_dot=0.15,
        min_gradient_per_density=0.3,
        fallback="radial",
    )
    radial_identity = json.loads((radial / "field/CHECKPOINT.json").read_bytes())["identity"]
    assert "orientation" not in radial_identity["source_identity"]
    # These sheets are planar and the prior agrees with the radial sign: the same field.
    geometry = json.loads((tile / "FIELD.json").read_bytes())["geometry"]
    with field.open_field(tile) as a, field.open_field(radial) as b:
        values, seen = a.read([0, 0, 0], geometry["shape_zyx"])
        expected, _ = b.read([0, 0, 0], geometry["shape_zyx"])
    assert seen.any() and np.max(np.abs(values - expected)) < 1e-6


def test_prior_gradient_orientation_reaches_cpu_workers(tmp_path, coarse):
    import torch

    from winding_field import field

    # One parent thread leaves the worker a CPU slot, as in test_field_cpu.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _solve_inline_and_pooled(tmp_path, coarse, field)
    finally:
        torch.set_num_threads(previous)


def test_cpu_workers_apply_the_sheet_check(tmp_path, monkeypatch):
    """A CPU worker signs its block with the parent's sheet check (spec orientation_prior
    sheet). The synthetic scroll agrees with its coarse everywhere, so this stub prior's
    floor crosses its sheets (planes x = const) instead."""
    from contextlib import closing

    from test_field_cpu import pool_spec
    from winding_field import field, field_cpu

    run = prepared_run(tmp_path / "run", shape=(6, 6, 24))
    spec = pool_spec(run)
    sheet = dict(max_disagreement=0.3, min_probes=0)
    spec.update(
        orientation_prior=dict(run="stub", field_json_sha256="stub", sheet=sheet),
        payload_bytes=1 << 30,
    )
    prior = _grid_prior(
        lambda p: abs(p[..., 0] - 33.0 - 2.0 * (p[..., 1] - 27.5)),
        (-10.0, -10.0, -10.0),
        (1.0,) * 3,
        (90, 70, 80),
    )
    lo, hi = np.zeros(3, int), np.array([6, 6, 24])
    with closing(field_cpu.Tiles(run / "maps", readonly=True)) as maps:
        monkeypatch.setattr(field_cpu, "_STATE", (spec, maps, None, prior))
        block = field_cpu.prepare_static_block(spec, lo, hi, build_rows=False)
        args = (maps, spec["meta"], spec["geometry"], spec["calibration"], lo, hi, prior)
        checked = field.sample_prior(*args, None, sheet)
        plain = field.sample_prior(*args)
    flagged = plain[2] & ~checked[2]
    assert flagged.any()
    for got, expected in zip(block.observations[:3], checked):
        assert np.array_equal(got, expected)
    assert block.worker_info["orientation"]["prior_inconsistent"] == np.count_nonzero(flagged)


def _solve_inline_and_pooled(tmp_path, coarse, field):
    config = dict(
        sweeps=2,
        field_chunk=8,
        field_spacing_um=1.0,
        field_prior=str(coarse),
        field_prior_calibration="inherit",
        field_orientation="prior-gradient",
    )
    inline = _tile(tmp_path / "inline", (6, 6, 16), shift_x=20.0)
    assert field.solve(inline, config)["status"] == "FIELD_READY"
    pooled = _tile(tmp_path / "pooled", (6, 6, 16), shift_x=20.0)
    workers = dict(config, field_cpu_workers=1, field_cpu_buffer_bytes=16 << 20)
    assert field.solve(pooled, workers)["status"] == "FIELD_READY"
    geometry = json.loads((inline / "FIELD.json").read_bytes())["geometry"]
    with field.open_field(inline) as a, field.open_field(pooled) as b:
        values, _ = a.read([0, 0, 0], geometry["shape_zyx"])
        expected, _ = b.read([0, 0, 0], geometry["shape_zyx"])
    assert np.array_equal(values, expected)
    # Workers count each box once, as the in-process path does.
    counts = [
        json.loads((run / "FIELD.json").read_bytes())["orientation"] for run in (inline, pooled)
    ]
    assert counts[0]["supported"] > 0 and counts[0] == counts[1]
