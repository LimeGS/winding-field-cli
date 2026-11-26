import io
import json
import sqlite3

import numpy as np
import pytest


def _fibres(path, curves):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE curves (id INTEGER PRIMARY KEY, points BLOB NOT NULL)")
    for i, points in enumerate(curves, start=1):
        blob = io.BytesIO()
        np.save(blob, points)
        db.execute("INSERT INTO curves VALUES (?, ?)", (i, blob.getvalue()))
    db.commit()
    db.close()


def _curves(rng, count, upper):
    curves = []
    for _ in range(count):
        start = rng.uniform(-5, upper + 5)
        steps = rng.normal(scale=4.0, size=(int(rng.integers(2, 40)), 3))
        curves.append(np.cumsum(np.vstack([start, steps]), 0))
    return curves


@pytest.mark.parametrize("workers", [1, 3])
def test_catalog_restrictions_equal_the_in_memory_global_pairs(tmp_path, monkeypatch, workers):
    from constraint_templates import StencilConstraints
    from winding_field import global_pairs
    from winding_field.field import spline_pairs

    monkeypatch.setattr(global_pairs, "BUCKET", 4)
    monkeypatch.setattr(global_pairs, "SHARD_BYTES", 2048)  # several shards
    rng = np.random.default_rng(11)
    shape = (21, 18, 26)  # ZYX
    geometry = dict(origin_xyz=[3.0, -2.0, 1.0], spacing_xyz=[1.5, 1.5, 1.5])
    origin, spacing = np.array(geometry["origin_xyz"]), np.array(geometry["spacing_xyz"])
    upper = origin + (np.array(shape[::-1]) - 1) * spacing
    curves = _curves(rng, 120, upper)
    _fibres(tmp_path / "fibres.sqlite", curves)

    status = global_pairs.build(
        tmp_path / "pairs", tmp_path / "fibres.sqlite", "sha", geometry, shape, workers=workers
    )
    assert status == "COMPLETE"
    manifest = json.loads((tmp_path / "pairs/PAIRS.json").read_bytes())
    assert len(manifest["shards"]) > 2
    catalog = global_pairs.GlobalPairs(tmp_path / "pairs", geometry, shape, 7.0)

    left, right = zip(*(spline_pairs(c, origin, upper, min(spacing)) for c in curves))
    a, b = np.concatenate(left), np.concatenate(right)
    assert len(catalog) == len(a)
    reference = StencilConstraints.pairs(a, b, origin, spacing, shape, weight=7.0)
    values = rng.normal(size=int(np.prod(shape)))
    for _ in range(40):
        lo = rng.integers(0, np.array(shape) - 1)
        hi = rng.integers(lo + 1, np.array(shape) + 1)
        expected = reference.restrict(lo, hi, lambda n: values[n])
        got = catalog(lo, hi, lambda n: values[n])
        if expected is None:
            assert got is None
            continue
        for name in ("targets", "weights", "diagonal"):
            np.testing.assert_array_equal(getattr(got, name), getattr(expected, name))
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(got.matrix, name), getattr(expected.matrix, name))
        ref_ext, got_ext = reference.prepare(lo, hi).exterior(), catalog.exterior(lo, hi)
        for x, y in zip(ref_ext, got_ext):
            np.testing.assert_array_equal(x, y)


def test_a_finished_catalog_is_reused_and_other_fibres_are_refused(tmp_path):
    from winding_field import global_pairs

    shape = (8, 8, 8)
    geometry = dict(origin_xyz=[0.0, 0.0, 0.0], spacing_xyz=[1.0, 1.0, 1.0])
    _fibres(tmp_path / "fibres.sqlite", [np.array([[1.0, 1, 1], [5, 5, 5]])])
    args = (tmp_path / "pairs", tmp_path / "fibres.sqlite")
    assert global_pairs.build(*args, "one", geometry, shape, workers=1) == "COMPLETE"
    stamp = (tmp_path / "pairs/PAIRS.json").stat().st_mtime_ns
    assert global_pairs.build(*args, "one", geometry, shape, workers=1) == "COMPLETE"
    assert (tmp_path / "pairs/PAIRS.json").stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="other fibres"):
        global_pairs.build(*args, "two", geometry, shape, workers=1)


def test_a_stopped_build_resumes_from_its_finished_shards(tmp_path, monkeypatch):
    from winding_field import global_pairs

    monkeypatch.setattr(global_pairs, "SHARD_BYTES", 256)
    shape = (10, 10, 10)
    geometry = dict(origin_xyz=[0.0, 0.0, 0.0], spacing_xyz=[1.0, 1.0, 1.0])
    curves = [np.array([[i, 1.0, 1.0], [i, 8.0, 8.0]]) for i in range(1, 9)]
    _fibres(tmp_path / "fibres.sqlite", curves)
    calls = iter([False, False, True])
    args = (tmp_path / "pairs", tmp_path / "fibres.sqlite", "s", geometry, shape)
    assert global_pairs.build(*args, workers=1, stop=lambda: next(calls, True)) == "PAUSED"
    finished = sorted(p.name for p in (tmp_path / "pairs").glob("shard-*"))
    assert finished and not (tmp_path / "pairs/PAIRS.json").exists()
    assert global_pairs.build(*args, workers=1) == "COMPLETE"
    total = json.loads((tmp_path / "pairs/PAIRS.json").read_bytes())["count"]
    fresh = tmp_path / "fresh"
    assert global_pairs.build(fresh, *args[1:], workers=1) == "COMPLETE"
    assert json.loads((fresh / "PAIRS.json").read_bytes())["count"] == total


def _with_spatial_index(run):
    """Give a traced test run the AFV layout: curve bounds and spatial_index in FIBRES.json."""
    from winding_field.util import atomic_json, file_sha

    path = run / "fibres/fibres.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE VIRTUAL TABLE curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z)"
    )
    for curve, raw in db.execute("SELECT id, points FROM curves").fetchall():
        points = np.load(io.BytesIO(raw), allow_pickle=False)
        low, high = points.min(0), points.max(0)
        db.execute(
            "INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)",
            (curve, *[float(v) for pair in zip(low, high) for v in pair]),
        )
    db.commit()
    db.close()
    fibres = json.loads((run / "FIBRES.json").read_bytes())
    atomic_json(run / "FIBRES.json", dict(fibres, spatial_index=True, sha256=file_sha(path)))


def test_a_two_level_run_from_the_catalog_equals_the_in_memory_pairs(tmp_path, monkeypatch):
    from test_field import prepared_run
    from winding_field import field
    from winding_field.field import open_field, solve

    def field_after(name):
        run = _with_spatial_index(prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8)) or (
            tmp_path / name
        )
        config = dict(
            sweeps=2, field_chunk=8, field_coarse_factor=4, field_kernel="eager", field_halo=2
        )
        result = solve(run, dict(config, rtol=1e-10, maxiter=10000, field_memory="disk"))
        assert result["status"] == "FIELD_READY" and result["fibre_constraint_rows"] > 0
        identity = json.loads((run / "field/CHECKPOINT.json").read_bytes())["identity"]
        with open_field(run) as source:
            values, _ = source.read([0, 0, 0], [6, 6, 32])
        return run, identity, result["fibre_constraint_rows"], values

    run, identity, rows, catalog = field_after("catalog")
    assert (run / "field-pairs/PAIRS.json").exists()
    monkeypatch.setattr(field, "_global_pair_catalog", lambda meta, fibres: False)
    other, identity_ram, rows_ram, in_memory = field_after("memory")
    assert not (other / "field-pairs").exists()
    # Same problem; the core names a provider's rows by the caller's identity (the fibres
    # hash and pair scope in source_identity) instead of hashing every global row.
    assert identity.pop("constraints") == "caller-bound-provider"
    identity_ram.pop("constraints")
    assert identity == identity_ram and rows == rows_ram
    np.testing.assert_array_equal(catalog, in_memory)


def test_prefetched_templates_equal_the_ones_built_in_place(tmp_path):
    from winding_field import global_pairs

    rng = np.random.default_rng(5)
    shape = (16, 16, 16)
    geometry = dict(origin_xyz=[0.0, 0.0, 0.0], spacing_xyz=[1.0, 1.0, 1.0])
    _fibres(tmp_path / "fibres.sqlite", _curves(rng, 60, np.full(3, 15.0)))
    global_pairs.build(tmp_path / "pairs", tmp_path / "fibres.sqlite", "s", geometry, shape)
    boxes = [
        (np.maximum(np.array(k) * 8 - 2, 0), np.minimum(np.array(k) * 8 + 10, shape))
        for k in np.ndindex(2, 2, 2)
    ]
    plain = global_pairs.GlobalPairs(tmp_path / "pairs", geometry, shape, 3.0)
    pooled = global_pairs.GlobalPairs(tmp_path / "pairs", geometry, shape, 3.0)
    pooled.prefetch(boxes, 2)
    values = rng.normal(size=16**3)
    try:
        for _ in range(2):  # two passes, as a sweep's coarse and local visits
            for lo, hi in boxes:
                a = plain(lo, hi, lambda n: values[n])
                b = pooled(lo, hi, lambda n: values[n])
                np.testing.assert_array_equal(a.targets, b.targets)
                assert (a.matrix != b.matrix).nnz == 0
                for x, y in zip(plain.exterior(lo, hi), pooled.exterior(lo, hi)):
                    np.testing.assert_array_equal(x, y)
    finally:
        pooled.close()
    assert pooled.stats["prefetched"] == 2 * len(boxes)
