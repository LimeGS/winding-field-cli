"""--field-boundary: a completed field on the same grid fixes a solve's internal faces."""

import json

import numpy as np
import pytest

from test_field_prior import _tile

# The synthetic scroll of test_field.prepared_run: planes x = const, W = x / 8 nodes on a
# grid of spacing (2, 3, 4) um at origin (10, 20, 30).
SPACING = [2.0, 3.0, 4.0]
ORIGIN = [10.0, 20.0, 30.0]


def write_field(run, shape, values, observed=None, origin=ORIGIN, frame="test", **meta):
    """A completed field directory holding ``values`` (ZYX) on the synthetic grid."""
    from field_store import BlockStore
    from winding_field.util import atomic_json, file_sha

    run.mkdir(parents=True)
    geometry = dict(
        frame=frame,
        voxel_um=1.0,
        origin_xyz=list(origin),
        spacing_xyz=SPACING,
        shape_zyx=list(shape),
        sheet_step=1.0,
    )
    observed = np.ones(shape, bool) if observed is None else observed
    with BlockStore(run / "field", shape, 8, dict(geometry=geometry)) as store:
        for key in np.ndindex(*[(n - 1) // 8 + 1 for n in shape]):
            box = tuple(slice(k * 8, min(k * 8 + 8, n)) for k, n in zip(key, shape))
            store.commit(key, values[box].copy(), {}, observed=observed[box].copy())
    atomic_json(
        run / "FIELD.json",
        dict(geometry=geometry, checkpoint_sha256=file_sha(run / "field/CHECKPOINT.json"), **meta),
    )
    return run


def planes(shape, first=0, level=0.0, slope=1 / 8):
    """W of the synthetic scroll on x nodes first.., plus ``level``."""
    return np.broadcast_to((np.arange(shape[2]) + first) * slope + level, shape).copy()


def read(run, policy="observed"):
    from winding_field import field

    shape = json.loads((run / "FIELD.json").read_bytes())["geometry"]["shape_zyx"]
    with field.open_field(run, support_policy=policy) as source:
        return source.read([0, 0, 0], shape)


CONFIG = dict(sweeps=3, field_chunk=8)


def test_internal_faces_take_the_other_fields_values_and_gauge_the_solve(tmp_path):
    """The tile covers x nodes 10..25 of a 32-node field 3 windings above the tile's own
    zero: with both x faces fixed there, the tile reports that field's W, with no prior."""
    from winding_field import field

    other = write_field(tmp_path / "other", (6, 6, 32), planes((6, 6, 32), level=3.0))
    alone = _tile(tmp_path / "alone", (6, 6, 16), shift_x=20.0)
    assert field.solve(alone, CONFIG)["status"] == "FIELD_READY"
    tile = _tile(tmp_path / "tile", (6, 6, 16), shift_x=20.0)
    receipt = field.solve(tile, dict(CONFIG, field_boundary=str(other)))
    assert receipt["status"] == "FIELD_READY"
    expected = planes((6, 6, 16), first=10, level=3.0)
    values, seen = read(tile)
    assert seen.all()
    np.testing.assert_allclose(values, expected, atol=1e-5)
    # On its own the tile pins its first node to zero instead.
    assert abs(read(alone)[0] - expected).min() > 2.5
    identity = json.loads((tile / "field/CHECKPOINT.json").read_bytes())["identity"]
    assert identity["boundary"] == "caller-fixed"
    recorded = identity["source_identity"]["field_boundary"]
    assert recorded["rule"] == "fixed-observed-layers-v2" and recorded["layers"] == 2
    assert recorded["offset_zyx"] == [0, 0, 10]
    # z and y end where the other field ends: only the two x faces are internal.
    assert recorded["fixed_nodes"] == {"x-": 72, "x+": 72}
    assert "run" not in recorded
    assert receipt["field_boundary"] == dict(recorded, run=str(other))


def test_fixed_layers_are_exact_and_the_interior_follows_the_data(tmp_path):
    """Layers that disagree with the data stay as given: the free nodes between them solve
    the tile's own observations against those fixed values."""
    from winding_field import field

    shape = (6, 6, 32)
    values = planes(shape)
    values[:, :, 10:12] += 0.25  # the tile's low x layers
    values[:, :, 24:26] -= 0.25  # and its high ones
    other = write_field(tmp_path / "other", shape, values)
    tile = _tile(tmp_path / "tile", (6, 6, 16), shift_x=20.0)
    assert field.solve(tile, dict(CONFIG, field_boundary=str(other)))["status"] == "FIELD_READY"
    solved, _ = read(tile)
    np.testing.assert_allclose(solved[:, :, :2], values[:, :, 10:12], atol=1e-6)
    np.testing.assert_allclose(solved[:, :, -2:], values[:, :, 24:26], atol=1e-6)
    # Half a winding less across the 11 free intervals than the data alone would give.
    slope = np.diff(solved[:, :, 2:14], axis=2)
    assert 1 / 8 - 0.5 / 11 - 0.02 < slope.mean() < 1 / 8
    with pytest.raises(ValueError, match="identity"):
        again = write_field(tmp_path / "again", shape, planes(shape))
        field.solve(tile, dict(CONFIG, field_boundary=str(again)))


def test_boundary_replaces_the_prior_gauge_and_starts_the_solve(tmp_path):
    from winding_field import field
    from winding_field.surface_sources import FieldSource
    from winding_field.util import file_sha

    coarse = _tile(tmp_path / "coarse", (6, 6, 32))
    assert field.solve(coarse, dict(sweeps=3, field_chunk=8))["status"] == "FIELD_READY"
    other = write_field(tmp_path / "other", (6, 6, 32), planes((6, 6, 32), level=3.0))
    tile = _tile(tmp_path / "tile", (6, 6, 16), shift_x=20.0)
    config = dict(CONFIG, field_prior=str(coarse), field_boundary=str(other))
    checkpoint = tile / "field/CHECKPOINT.json"

    def initialized():
        saved = json.loads(checkpoint.read_bytes()) if checkpoint.exists() else {}
        return saved.get("progress", {}).get("initialization_done", False)

    # Stopped before the first block: the stored field is the starting one.
    assert field.solve(tile, config, stop=initialized)["status"] == "PAUSED"
    identity = json.loads(checkpoint.read_bytes())["identity"]["source_identity"]
    descriptor = dict(
        kind="chunks", path=str(checkpoint), sha256=file_sha(checkpoint), support_policy="stored"
    )
    with FieldSource(descriptor, identity["geometry"]) as source:
        start, stored = source.read([0, 0, 0], [6, 6, 16])
    assert stored.all()
    np.testing.assert_allclose(start, planes((6, 6, 16), first=10, level=3.0), atol=1e-6)
    receipt = field.solve(tile, config)
    assert receipt["status"] == "FIELD_READY"
    np.testing.assert_allclose(read(tile)[0], planes((6, 6, 16), first=10, level=3.0), atol=1e-5)
    assert "anchor" not in receipt["field_prior"]
    assert receipt["field_boundary"]["fixed_nodes"] == {"x-": 72, "x+": 72}


def test_only_observed_nodes_are_fixed_and_the_rest_start_from_the_fallback(tmp_path):
    from winding_field.field_boundary import FieldBoundary

    shape = (6, 6, 32)
    observed = np.ones(shape, bool)
    observed[:3, :, 10] = False
    other = write_field(tmp_path / "other", shape, planes(shape), observed)
    geometry = dict(
        frame="test",
        origin_xyz=[ORIGIN[0] + 20.0, ORIGIN[1], ORIGIN[2] - 8.0],
        spacing_xyz=SPACING,
        shape_zyx=[8, 6, 16],
        sheet_step=1.0,
    )
    boundary = FieldBoundary(other, geometry, 2)
    try:
        # The grid starts 2 nodes below the field in z: its low z face is outside, and its
        # high z face (node 5 of the field) is the field's own border.
        assert boundary.internal == [(False, False), (False, False), (True, True)]
        assert boundary.identity["fixed_nodes"] == {"x-": 72 - 18, "x+": 72}
        fixed = boundary.provider()([0, 0, 0], [8, 6, 16])
        assert np.isnan(fixed[:2]).all() and np.isnan(fixed[:, :, 2:14]).all()
        assert np.isnan(fixed[2:5, :, 0]).all()
        np.testing.assert_allclose(fixed[5:, :, 0], 10 / 8)
        np.testing.assert_allclose(fixed[2:, :, 1], 11 / 8)
        np.testing.assert_allclose(fixed[2:, :, 14:], np.broadcast_to([24 / 8, 25 / 8], (6, 6, 2)))
        # A crop of a block and its base provider's values.
        crop = boundary.provider(lambda lo, hi: np.full(tuple(hi - lo), 7.0))(
            [2, 1, 12], [4, 3, 16]
        )
        np.testing.assert_allclose(crop[..., :2], 7.0)
        np.testing.assert_allclose(crop[..., 2:], np.broadcast_to([24 / 8, 25 / 8], (2, 2, 2)))
        start = boundary.initial([0, 0, 0], [8, 6, 16], lambda: np.full((8, 6, 16), -1.0))
        assert (start[:2] == -1.0).all()
        np.testing.assert_allclose(start[2:], planes((6, 6, 16), first=10), atol=1e-6)
        inside = boundary.initial([2, 0, 0], [8, 6, 16], lambda: pytest.fail("not needed"))
        np.testing.assert_allclose(inside, planes((6, 6, 16), first=10), atol=1e-6)
    finally:
        boundary.close()


def _geometry(**changes):
    return dict(
        dict(
            frame="test",
            origin_xyz=[ORIGIN[0] + 20.0, ORIGIN[1], ORIGIN[2]],
            spacing_xyz=SPACING,
            shape_zyx=[6, 6, 16],
            sheet_step=1.0,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "changes, layers, message",
    [
        (dict(origin_xyz=[ORIGIN[0] + 21.0, ORIGIN[1], ORIGIN[2]]), 2, "not aligned"),
        (dict(spacing_xyz=[1.0, 3.0, 4.0]), 2, "spacing differs"),
        (dict(sheet_step=2.0), 2, "sheet_step differs"),
        (dict(origin_xyz=[ORIGIN[0] + 200.0, ORIGIN[1], ORIGIN[2]]), 2, "does not overlap"),
        (dict(shape_zyx=[6, 6, 32], origin_xyz=ORIGIN), 2, "fixes no node"),
        ({}, 8, "no free node"),
        ({}, 0, "positive integer"),
        (dict(frame="elsewhere"), 2, "CT source identity is unavailable"),
    ],
)
def test_boundary_must_share_the_grid_and_leave_free_nodes(tmp_path, changes, layers, message):
    from winding_field.field_boundary import FieldBoundary

    other = write_field(tmp_path / "other", (6, 6, 32), planes((6, 6, 32)))
    with pytest.raises(ValueError, match=message):
        FieldBoundary(other, _geometry(**changes), layers)
    with pytest.raises(ValueError, match="no FIELD.json"):
        FieldBoundary(tmp_path / "missing", _geometry(), 2)


# The tile of _geometry() holds x nodes 10..25 of the other field.
SELF = [[0, 0, 10], [5, 5, 25]]
RIGHT = [[0, 0, 20], [5, 5, 31]]


def test_a_merge_holds_a_face_only_where_another_tile_continues_beyond_it(tmp_path):
    """A merge lists the boxes of its tiles. Layer nodes that only this tile's own earlier
    solution reached stay free: a tile that ends on the same face, or within the held
    layers of it, does not make them another tile's value."""
    from winding_field.field_boundary import FieldBoundary

    shape = (6, 6, 32)
    boxes = {
        "self": SELF,
        "right": RIGHT,
        "same-face": [[0, 0, 10], [5, 5, 31]],
        "left-part": [[0, 0, 0], [5, 2, 15]],
        "too-close": [[0, 0, 9], [5, 5, 20]],
    }
    merged = write_field(tmp_path / "merged", shape, planes(shape), tile_boxes=boxes)
    boundary = FieldBoundary(merged, _geometry(), 2)
    fixed = np.isfinite(boundary.provider()([0, 0, 0], [6, 6, 16]))
    boundary.close()
    assert boundary.identity["fixed_nodes"] == {"x-": 36, "x+": 72}
    assert boundary.identity["source_tiles"] == {"x-": ["left-part"], "x+": ["right", "same-face"]}
    expected = np.zeros((6, 6, 16), bool)
    expected[:, :3, :2] = expected[:, :, 14:] = True
    assert (fixed == expected).all()

    partial = write_field(
        tmp_path / "partial", shape, planes(shape), tile_boxes=dict(self=SELF, right=RIGHT)
    )
    boundary = FieldBoundary(partial, _geometry(), 2)
    boundary.close()
    assert boundary.identity["fixed_nodes"] == {"x-": 0, "x+": 72}
    assert boundary.identity["source_tiles"] == {"x-": [], "x+": ["right"]}

    alone = write_field(tmp_path / "alone", shape, planes(shape), tile_boxes=dict(self=SELF))
    with pytest.raises(ValueError, match="fixes no node: it has no observed value, from another"):
        FieldBoundary(alone, _geometry(), 2)
    old = write_field(tmp_path / "old", shape, planes(shape), schema="winding-scroll-merge-v1")
    with pytest.raises(ValueError, match="does not list its tiles' boxes"):
        FieldBoundary(old, _geometry(), 2)


def test_an_edge_between_a_free_and_a_held_face_is_held(tmp_path):
    """The tile is inside the merge along z and x; only x+ has a tile beyond it. The x+
    layers are held over the whole face, the z layers that belong to it included."""
    from winding_field.field_boundary import FieldBoundary

    shape = (12, 6, 32)
    boxes = {"self": [[3, 0, 10], [8, 5, 25]], "right": [[3, 0, 20], [8, 5, 31]]}
    merged = write_field(tmp_path / "merged", shape, planes(shape), tile_boxes=boxes)
    origin = [ORIGIN[0] + 20.0, ORIGIN[1], ORIGIN[2] + 3 * SPACING[2]]
    boundary = FieldBoundary(merged, _geometry(origin_xyz=origin), 2)
    fixed = np.isfinite(boundary.provider()([0, 0, 0], [6, 6, 16]))
    boundary.close()
    assert boundary.identity["offset_zyx"] == [3, 0, 10]
    # Each node counts once, under the first of z, y, x whose layers hold it.
    assert boundary.identity["fixed_nodes"] == {"z-": 24, "z+": 24, "x-": 0, "x+": 24}
    assert boundary.identity["source_tiles"] == {"z-": [], "z+": [], "x-": [], "x+": ["right"]}
    expected = np.zeros((6, 6, 16), bool)
    expected[:, :, 14:] = True
    assert (fixed == expected).all()


def test_a_face_toward_a_missing_tile_is_solved_not_frozen(tmp_path):
    """The merge carries this tile's own error on the x- layers and no tile beyond them:
    the solve corrects them from the tile's data, held only at x+."""
    from winding_field import field

    shape = (6, 6, 32)
    values = planes(shape, level=3.0)
    values[:, :, 10:12] += 0.25
    merged = write_field(
        tmp_path / "merged", shape, values, tile_boxes=dict(self=SELF, right=RIGHT)
    )
    tile = _tile(tmp_path / "tile", (6, 6, 16), shift_x=20.0)
    receipt = field.solve(tile, dict(CONFIG, sweeps=8, field_boundary=str(merged)))
    assert receipt["status"] == "FIELD_READY"
    assert receipt["field_boundary"]["fixed_nodes"] == {"x-": 0, "x+": 72}
    solved, _ = read(tile)
    np.testing.assert_allclose(solved, planes((6, 6, 16), first=10, level=3.0), atol=1e-3)


def test_boundary_accepts_a_frame_the_merged_field_accepted(tmp_path):
    """A merge of tiles read by two hosts lists both frames (FIELD.json frame_match): a
    tile carrying either one is on the same CT."""
    from winding_field.field_boundary import FieldBoundary
    from winding_field.field_prior import frame_match

    merged = write_field(
        tmp_path / "merged",
        (6, 6, 32),
        planes((6, 6, 32)),
        frame="host-a",
        frame_match=frame_match(frames={"host-a": ["000"], "host-b": ["001"]}),
    )
    boundary = FieldBoundary(merged, _geometry(frame="host-b"), 2)
    boundary.close()
    assert boundary.identity["frame_match"]["via"] == "field-boundary-frame-match"
    with pytest.raises(ValueError, match="frame 'host-a' differs from 'host-c'"):
        FieldBoundary(merged, _geometry(frame="host-c"), 2)


def test_field_can_be_written_apart_from_its_prepared_run(tmp_path):
    """``output``: the prepared run stays as it was and the field lands elsewhere, equal
    to the one solved in place."""
    from winding_field import field

    other = write_field(tmp_path / "other", (6, 6, 32), planes((6, 6, 32), level=3.0))
    config = dict(CONFIG, field_boundary=str(other))
    here = _tile(tmp_path / "here", (6, 6, 16), shift_x=20.0)
    assert field.solve(here, config)["status"] == "FIELD_READY"
    tile = _tile(tmp_path / "tile", (6, 6, 16), shift_x=20.0)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in (tile, *tile.rglob("*")))
    out = tmp_path / "passes/1/tile"
    assert field.solve(tile, config, output=out)["status"] == "FIELD_READY"
    # Nothing is created in the prepared run, not even for a moment.
    assert sorted((p.name, p.stat().st_mtime_ns) for p in (tile, *tile.rglob("*"))) == before
    assert (out / "FIELD.json").is_file() and (out / "CALIBRATION.json").is_file()
    np.testing.assert_array_equal(read(out)[0], read(here)[0])


@pytest.mark.parametrize(
    "config, message",
    [
        ({"field_boundary": ""}, "completed field directory"),
        ({"field_boundary": 3}, "completed field directory"),
        ({"field_boundary_layers": 2}, "requires field_boundary"),
        ({"field_boundary": "run", "field_boundary_layers": 0}, "positive integer"),
        ({"field_boundary": "run", "field_boundary_layers": 1.5}, "positive integer"),
        ({"field_boundary": "run", "bounded_preparation": True}, "ordinary"),
    ],
)
def test_boundary_options_are_validated(config, message):
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match=message):
        validate_config(config)
