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
    prior = FieldPrior(coarse, geometry)
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
