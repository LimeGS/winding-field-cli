"""Exercise field.solve's actual SQL/clipping/constraint consumer with the pinned core."""

import io
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from winding_field import field
from winding_field.artifacts import Tiles
from winding_field.util import atomic_json, file_sha


def consume_constraints(
    root, monkeypatch, origin, spacing, lo, hi, curves, *, spatial=True, global_shape=None
):
    root.mkdir()
    shape = (np.asarray(hi) + 3).tolist() if global_shape is None else list(map(int, global_shape))
    with closing(
        Tiles(root / "maps", shape=shape, chunk=max(shape), identity={"test": "block bounds"})
    ) as maps:
        maps.write((0, 0, 0), dict(support=np.ones(shape, bool)))
        maps_sha = maps.identity()
    atomic_json(
        root / "PREPARED.json",
        dict(
            shape_zyx=shape,
            origin_xyz=list(origin),
            spacing_xyz=list(spacing),
            frame="synthetic",
            tiles_sha256=maps_sha,
            prediction="qual-v2" if spatial else "analytic-test",
        ),
    )
    (root / "fibres").mkdir()
    with sqlite3.connect(root / "fibres/fibres.sqlite") as db:
        db.executescript(
            "CREATE TABLE curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB);"
            "CREATE VIRTUAL TABLE curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);"
        )
        for index, points in enumerate(curves, 1):
            blob = io.BytesIO()
            np.save(blob, points, allow_pickle=False)
            db.execute("INSERT INTO curves VALUES (?,?,?)", (index, "0_0_0", blob.getvalue()))
            box = tuple(v for pair in zip(points.min(0), points.max(0)) for v in pair)
            db.execute("INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)", (index, *box))
    atomic_json(root / "FIBRES.json", dict(sha256=file_sha(root / "fibres/fibres.sqlite")))
    # Calibration and numerical iteration are unrelated: keep the real provider,
    # real R-tree, real clipping/sampling and unmodified core pair construction.
    monkeypatch.setattr(field, "calibration", lambda *_: dict(mode="planar"))
    captured = []

    class Consumed(Exception):
        pass

    def block_consumer(_root, _shape, _chunk, _identity, _load, **options):
        captured.append(options["constraints"](np.asarray(lo), np.asarray(hi), None))
        raise Consumed

    monkeypatch.setattr(field, "solve_blocks", block_consumer)
    with pytest.raises(Consumed):
        field.solve(root, dict(field_chunk=64, sweeps=1))
    return captured[0]


def test_real_172_8um_block_clips_to_the_same_physical_box_as_core(tmp_path, monkeypatch):
    points = np.array(
        [
            [45000.0, 32775.840000000004, 7027.225372391387],
            [44000.0, 32775.840000000004, 7027.225372391387],
        ]
    )
    rows = consume_constraints(
        tmp_path / "run",
        monkeypatch,
        [0.0, 0.0, 0.0],
        [172.8] * 3,
        [0, 126, 190],
        [66, 194, 258],
        [points],
    )
    assert rows.matrix.shape == (3, 66 * 68 * 68)
    assert np.isfinite(rows.targets).all()


CASES = [
    ([0.0, 0.0, 0.0], [172.8] * 3, [0, 126, 190], [66, 194, 258]),
    ([-137.8, 405.5, -99.2], [172.8, 9.6, 8.64], [3, 5, 7], [11, 16, 22]),
    ([17.0, -42.0, 103.0], [0.2, 3.3, 17.8], [0, 2, 3], [7, 9, 13]),
    ([2**30, -(2**28), 1.3], [0.1, 0.3, 1.7], [4, 4, 8], [8, 8, 11]),
    ([2**30, -(2**28), 1.3], [0.1, 0.3, 1.7], [2, 2, 6], [9, 10, 11]),
    ([2**30, -(2**28), 1.3], [0.1, 0.3, 1.7], [1, 1, 5], [9, 10, 11]),
]


@pytest.mark.parametrize("origin,spacing,lo,hi", CASES)
@pytest.mark.parametrize("edge", ["inside", "exact", "outside"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_local_box_exact_and_nextafter_boundaries(
    tmp_path, monkeypatch, origin, spacing, lo, hi, edge, axis, side
):
    origin = np.asarray(origin)
    spacing = np.asarray(spacing)
    low = origin + np.asarray(lo)[::-1] * spacing
    upper = low + (np.asarray(hi) - lo - 1)[::-1] * spacing
    boundary = low[axis] if side == "lower" else upper[axis]
    interior = upper[axis] if side == "lower" else low[axis]
    exterior = -np.inf if side == "lower" else np.inf
    value = {
        "inside": np.nextafter(boundary, interior),
        "exact": boundary,
        "outside": np.nextafter(boundary, exterior),
    }[edge]
    travel = (axis + 1) % 3
    a = low + 0.5 * spacing
    a[axis] = value
    a[travel] = low[travel] + 0.25 * spacing[travel]
    b = a.copy()
    b[travel] += 0.5 * spacing[travel]
    rows = consume_constraints(
        tmp_path / "run", monkeypatch, origin, spacing, lo, hi, [np.array([a, b])]
    )
    if edge == "outside":
        assert rows is None
    else:
        assert rows is not None and rows.matrix.shape[0] > 0
        from field_constraints import pair_constraints

        expected = pair_constraints(a[None], b[None], low, spacing, tuple(np.asarray(hi) - lo))
        assert rows.matrix.shape[1] == expected.matrix.shape[1]
        # The unmodified core still rejects even one physical ULP outside.
        exterior_point = a.copy()
        exterior_point[axis] = np.nextafter(boundary, exterior)
        with pytest.raises(ValueError, match="outside valid support"):
            pair_constraints(
                exterior_point[None], b[None], low, spacing, tuple(np.asarray(hi) - lo)
            )


def test_core_support_mask_is_not_weakened():
    from field_constraints import pair_constraints

    origin = np.array([32832.0, 21772.800000000003, 0.0])
    spacing = np.full(3, 172.8)
    shape = (3, 3, 3)
    upper = origin + (np.array(shape)[::-1] - 1) * spacing
    a = upper[None]
    b = (origin + spacing)[None]
    support = np.ones(shape, bool)
    assert pair_constraints(a, b, origin, spacing, shape, support=support).matrix.shape[0] == 1
    support[-1, -1, -1] = False
    with pytest.raises(ValueError, match="outside valid support"):
        pair_constraints(a, b, origin, spacing, shape, support=support)


@pytest.mark.parametrize("origin,spacing,lo,hi", CASES)
def test_nonspatial_constraints_keep_the_same_local_boundary(
    tmp_path, monkeypatch, origin, spacing, lo, hi
):
    origin = np.asarray(origin)
    spacing = np.asarray(spacing)
    low = origin + np.asarray(lo)[::-1] * spacing
    upper = low + (np.asarray(hi) - lo - 1)[::-1] * spacing
    a = np.array([upper[0], low[1] + 0.25 * spacing[1], low[2] + 0.5 * spacing[2]])
    points = np.linspace(a, a + [0.0, 0.5 * spacing[1], 0.0], 4)
    rows = consume_constraints(
        tmp_path / "run", monkeypatch, origin, spacing, lo, hi, [points], spatial=False
    )
    assert rows is not None and rows.matrix.shape[0] == 1


@pytest.mark.parametrize("halo", [0, 2, 3])
def test_incomplete_final_blocks_with_different_halos(tmp_path, monkeypatch, halo):
    shape = np.array([9, 10, 11])
    chunk = 4
    key = np.array([1, 1, 2])
    lo = np.maximum(key * chunk - halo, 0)
    hi = np.minimum(key * chunk + chunk + halo, shape)
    origin = np.array([-137.8, 405.5, -99.2])
    spacing = np.array([172.8, 9.6, 8.64])
    low = origin + lo[::-1] * spacing
    upper = low + (hi - lo - 1)[::-1] * spacing
    a = low + 0.5 * spacing
    a[0] = upper[0]
    b = a + [0.0, 0.5 * spacing[1], 0.0]
    rows = consume_constraints(
        tmp_path / "run",
        monkeypatch,
        origin,
        spacing,
        lo,
        hi,
        [np.array([a, b])],
        global_shape=shape,
    )
    assert rows is not None and rows.matrix.shape[1] == int(np.prod(hi - lo))


def test_fresh_field_can_reuse_copied_preparation_and_fibres_without_adopting_checkpoint(tmp_path):
    import shutil
    from test_field import prepared_run

    old = prepared_run(tmp_path / "old")
    checkpoint = old / "field/CHECKPOINT.json"
    config = dict(sweeps=1, field_chunk=8)

    def stop():
        import json

        return checkpoint.exists() and len(json.loads(checkpoint.read_bytes())["chunks"]) >= 1

    assert field.solve(old, config, stop=stop)["status"] == "PAUSED"
    before = {
        p.relative_to(old): (file_sha(p), p.stat().st_mtime_ns)
        for p in old.rglob("*")
        if p.is_file()
    }
    new = tmp_path / "recovery"
    new.mkdir()
    for name in ("maps", "fibres"):
        shutil.copytree(old / name, new / name)
    for name in ("PREPARED.json", "FIBRES.json"):
        shutil.copy2(old / name, new / name)
    assert not (new / "field").exists() and not (new / "RUN.json").exists()
    assert field.solve(new, config)["status"] == "FIELD_READY"
    assert before == {
        p.relative_to(old): (file_sha(p), p.stat().st_mtime_ns)
        for p in old.rglob("*")
        if p.is_file()
    }
    for relative, pin in before.items():
        if relative.parts[0] in ("maps", "fibres") or str(relative) in (
            "PREPARED.json",
            "FIBRES.json",
        ):
            p = new / relative
            assert (file_sha(p), p.stat().st_mtime_ns) == pin
