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


@pytest.mark.parametrize("config", [{"field_scope": "some"}, {"field_scope": "supported"}])
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
        supported=4, prior_signed=2, prior_dropped=1, radial_signed=1, radial_dropped=0
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


def test_prior_gradient_orientation_is_validated_and_recorded(tmp_path, coarse, capsys):
    from winding_field import field
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="requires field_prior"):
        validate_config({"field_orientation": "prior-gradient"})
    with pytest.raises(ValueError, match="radial or prior-gradient"):
        validate_config({"field_prior": str(coarse), "field_orientation": "normal"})
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
    assert rule["rule"] == "prior-gradient-central-v2"
    assert rule["prior_values"] == "stored-finite"
    # The receipt splits the supported nodes by where their sign came from, and logs it.
    counts = json.loads((tile / "FIELD.json").read_bytes())["orientation"]
    assert counts["rule"] == rule["rule"] and counts["supported"] > 0
    classes = ("prior_signed", "prior_dropped", "radial_signed", "radial_dropped")
    assert sum(counts[name] for name in classes) == counts["supported"]
    assert counts["radial_fallback"] == counts["radial_signed"] + counts["radial_dropped"]
    assert counts["radial_fallback_share"] == counts["radial_fallback"] / counts["supported"]
    assert "orientation prior-gradient over" in capsys.readouterr().err
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
