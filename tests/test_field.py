import json

import numpy as np
import pytest


def prepared_run(root, shape=(6, 6, 24), chunk=8, variable=False, support_x_below=None):
    from winding_field.artifacts import Tiles
    from winding_field.fibres import trace
    from winding_field.util import atomic_json

    root.mkdir()
    maps = Tiles(
        root / "maps", shape=shape, chunk=chunk, identity={"test": "analytic physical layers"}
    )
    for key in maps.keys():
        lo, hi = maps.box(key)
        local = tuple(hi - lo)
        density = np.full(local, 1 / 16, np.float32)
        if variable:
            density = (
                np.broadcast_to(np.where(np.arange(lo[2], hi[2]) < 12, 1 / 8, 1 / 16), local)
                .astype(np.float32)
                .copy()
            )
        normals = np.zeros(local + (3,), np.float32)
        normals[..., 0] = 1
        tangent = np.zeros_like(normals)
        tangent[..., 2] = 1
        maps.write(
            key,
            dict(
                density=density,
                normals=normals,
                tangents=tangent,
                support=(
                    np.ones(local, bool)
                    if support_x_below is None
                    else np.broadcast_to(np.arange(lo[2], hi[2]) < support_x_below, local).copy()
                ),
                presence=np.ones(local, np.float32),
                ct=np.full(local, 128, np.uint8),
            ),
        )
    meta = dict(
        shape_zyx=list(shape),
        spacing_xyz=[2.0, 3.0, 4.0],
        origin_xyz=[10.0, 20.0, 30.0],
        frame="test",
        tiles_sha256=maps.identity(),
        prediction="analytic-test",
        physical_accuracy="synthetic",
    )
    atomic_json(root / "PREPARED.json", meta)
    maps.close()
    trace(root, {"seed_stride": 4})
    return root


def test_physical_planar_field_and_fibre_constraints(tmp_path):
    from winding_field.field import solve, open_field

    run = prepared_run(tmp_path / "run")
    receipt = solve(run, {"sweeps": 2, "field_chunk": 8})
    assert receipt["status"] == "FIELD_READY"
    assert receipt["fibre_constraint_rows"] > 0
    with open_field(run) as source:
        field, valid = source.read([0, 0, 0], [6, 6, 24])
    expected = np.broadcast_to(np.arange(24) / 8, field.shape)
    np.testing.assert_allclose(field, expected, atol=2e-4)
    assert valid.all()
    assert receipt["physical_acceptance"] is False


def test_fused_field_kernel_matches_eager_and_is_recorded(tmp_path):
    import torch

    from winding_field.field import solve, open_field

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fields, receipts = {}, {}
    for kernel in ("eager", "fused"):
        run = prepared_run(tmp_path / kernel)
        config = {"sweeps": 2, "field_chunk": 8, "device": device}
        if kernel == "fused":
            config["field_kernel"] = "fused"
        receipts[kernel] = solve(run, config)
        with open_field(run) as source:
            fields[kernel] = source.read([0, 0, 0], [6, 6, 24])[0]
        identity = json.loads((run / "field/CHECKPOINT.json").read_text())["identity"]
        assert identity.get("kernel") == (None if kernel == "eager" else "fused")
    np.testing.assert_allclose(fields["fused"], fields["eager"], atol=1e-6)
    assert "solver_kernels" not in receipts["eager"]["solver"]
    used = receipts["fused"]["solver"]["solver_kernels"]
    # Off CUDA (and outside Triton's interpreter) the core falls back to eager and says so.
    assert set(used) == {"fused-v1" if device == "cuda" else "eager"}


@pytest.mark.parametrize(
    "config",
    [
        {"preconditioner": "spectral"},
        {"field_kernel": "triton"},
        {"prepared_cache_bytes": True},
        {"prepared_cache_bytes": -1},
        {"prepared_cache_bytes": 1.5},
    ],
)
def test_general_solver_rejects_invalid_execution_before_io(tmp_path, config):
    from winding_field.config import validate_config
    from winding_field.field import solve

    with pytest.raises(ValueError):
        validate_config(config)
    with pytest.raises(ValueError):
        solve(tmp_path / "absent", config)
    assert not (tmp_path / "absent").exists()


def test_nonconstant_layer_separation_is_not_replaced_by_mean(tmp_path):
    from winding_field.field import solve, open_field

    run = prepared_run(tmp_path / "run", shape=(6, 6, 32), chunk=32, variable=True)
    solve(run, {"sweeps": 1, "field_chunk": 32})
    with open_field(run) as source:
        field, _ = source.read([0, 0, 0], [6, 6, 32])
    delta = np.diff(field[3, 3])
    assert np.mean(delta[2:10]) == pytest.approx(0.25, abs=0.002)
    assert np.mean(delta[16:28]) == pytest.approx(0.125, abs=0.002)


def test_static_constraint_cache_avoids_rebuilds_with_identical_field(tmp_path, monkeypatch):
    from winding_field import field

    original = field.pair_constraints
    builds = []

    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(field, "pair_constraints", counted)
    results = []
    counts = []
    rows = []
    for budget in (0, 1 << 20):
        builds.clear()
        run = prepared_run(tmp_path / str(budget))
        receipt = field.solve(
            run,
            dict(sweeps=3, field_chunk=8, constraint_cache_bytes=budget, prepared_cache_bytes=0),
        )
        with field.open_field(run) as source:
            results.append(source.read([0] * 3, [6, 6, 24])[0])
        counts.append(len(builds))
        rows.append(receipt["fibre_constraint_rows"])
    assert counts[0] > 0 and counts[1] * 3 == counts[0]
    assert rows[0] == rows[1]
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("budget", [0, 1, 16384, 1 << 20])
def test_constraint_cache_reports_bounded_retention_and_releases(tmp_path, budget):
    from winding_field import field

    run = prepared_run(tmp_path / "run")
    result = field.solve(run, dict(sweeps=3, field_chunk=8, constraint_cache_bytes=budget))
    stats = result["constraint_cache_stats"]
    assert stats["budget"] == budget
    assert stats["hits"] + stats["misses"] == 9
    assert stats["builds"] == stats["misses"]
    assert stats["peak_bytes"] <= budget
    assert stats["entries"] == stats["retained_bytes"] == 0
    assert json.loads((run / "FIELD.json").read_bytes())["constraint_cache_stats"] == stats
    if budget == 0:
        assert stats["hits"] == 0 and stats["builds"] == 9
    if budget == 1 << 20:
        assert stats["builds"] == 3 and stats["hits"] == 6 and stats["peak_entries"] == 3


def test_partial_cache_preserves_field_across_sweeps(tmp_path):
    from winding_field import field

    fields = []
    stats = {}
    for budget in (0, 16384, 1 << 20):
        run = prepared_run(tmp_path / str(budget))
        receipt = field.solve(
            run,
            dict(sweeps=4, field_chunk=8, constraint_cache_bytes=budget, prepared_cache_bytes=0),
        )
        with field.open_field(run) as source:
            fields.append(source.read([0] * 3, [6, 6, 24]))
        stats[budget] = (receipt["constraint_cache_stats"], receipt["fibre_constraint_rows"])
    for values, valid in fields[1:]:
        np.testing.assert_array_equal(values, fields[0][0])
        np.testing.assert_array_equal(valid, fields[0][1])
    assert len({rows for _, rows in stats.values()}) == 1
    assert stats[0][0]["hits"] == 0 and stats[0][0]["builds"] == 12
    partial = stats[16384][0]
    assert 0 < partial["peak_entries"] < 3 and partial["denied_entries"] > 0
    assert 0 < partial["hits"] < 12 and 0 < partial["builds"] < 12
    assert partial["hits"] + partial["misses"] == 12
    full = stats[1 << 20][0]
    assert full["builds"] == 3 and full["hits"] == 9 and full["peak_entries"] == 3
    assert full["denied_entries"] == 0
    for budget, (cache, _) in stats.items():
        assert cache["budget"] == budget and cache["peak_bytes"] <= budget
        assert cache["entries"] == cache["retained_bytes"] == 0


@pytest.mark.parametrize("budget", [True, -1, 1.5])
def test_constraint_cache_rejects_invalid_budget_before_io(tmp_path, budget):
    from winding_field.field import solve

    with pytest.raises(ValueError, match="constraint_cache_bytes"):
        solve(tmp_path / "missing", dict(constraint_cache_bytes=budget))
    assert not (tmp_path / "missing").exists()


def test_constraint_cache_retains_empty_blocks(tmp_path):
    import sqlite3
    from winding_field import field
    from winding_field.util import file_sha

    run = prepared_run(tmp_path / "run")
    with sqlite3.connect(run / "fibres/fibres.sqlite") as db:
        db.execute("DELETE FROM curves")
    receipt = json.loads((run / "FIBRES.json").read_bytes())
    receipt["sha256"] = file_sha(run / "fibres/fibres.sqlite")
    (run / "FIBRES.json").write_text(json.dumps(receipt))
    result = field.solve(run, dict(sweeps=3, field_chunk=8, constraint_cache_bytes=3 * 4096))
    assert result["fibre_constraint_rows"] == 0
    assert result["constraint_cache_stats"]["builds"] == 3
    assert result["constraint_cache_stats"]["hits"] == 6
    assert result["constraint_cache_stats"]["peak_bytes"] == 3 * 4096


def test_constraint_cache_resume_and_changed_source_preserve_checkpoint(tmp_path):
    from winding_field import field

    # The stop below watches the checkpoint advance block by block: block checkpoints.
    config = dict(sweeps=3, field_chunk=8, constraint_cache_bytes=1 << 20, field_memory="disk")
    run = prepared_run(tmp_path / "run")

    def stop():
        cp = run / "field/CHECKPOINT.json"
        return cp.exists() and json.loads(cp.read_bytes())["progress"].get("next_block") == 1

    paused = field.solve(run, config, stop=stop)
    assert paused["status"] == "PAUSED"
    assert paused["constraint_cache_stats"]["retained_bytes"] == 0
    assert field.solve(run, dict(config, constraint_cache_bytes=0))["status"] == "FIELD_READY"
    reference = prepared_run(tmp_path / "reference")
    field.solve(reference, dict(config, constraint_cache_bytes=0))
    with field.open_field(run) as a, field.open_field(reference) as b:
        np.testing.assert_array_equal(
            a.read([0] * 3, [6, 6, 24])[0], b.read([0] * 3, [6, 6, 24])[0]
        )
    before = (run / "field/CHECKPOINT.json").read_bytes()
    with (run / "fibres/fibres.sqlite").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        field.solve(run, config)
    assert (run / "field/CHECKPOINT.json").read_bytes() == before


def test_interrupted_solver_resumes_exact_problem(tmp_path):
    from winding_field.field import solve, open_field

    run = prepared_run(tmp_path / "run")
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 4

    receipt = solve(run, {"sweeps": 2, "field_chunk": 8}, stop=stop)
    assert receipt["status"] == "PAUSED"
    assert not (run / "FIELD.json").exists()
    with pytest.raises(ValueError, match="identity|differ"):
        solve(run, {"sweeps": 3, "field_chunk": 8})
    assert solve(run, {"sweeps": 2, "field_chunk": 8})["status"] == "FIELD_READY"
    with open_field(run) as source:
        field, _ = source.read([0, 0, 0], [6, 6, 24])
    np.testing.assert_allclose(field[3, 3], np.arange(24) / 8, atol=2e-4)


def test_failed_local_convergence_does_not_publish_field(tmp_path):
    from winding_field.field import solve

    run = prepared_run(tmp_path / "run", shape=(6, 6, 32), chunk=32, variable=True)
    with pytest.raises(RuntimeError, match="converge"):
        solve(run, {"sweeps": 1, "field_chunk": 32, "maxiter": 0})
    assert not (run / "FIELD.json").exists()


@pytest.mark.parametrize("limit", [0, 1])
def test_field_initialization_stops_at_block_boundary_and_resumes(tmp_path, limit):
    from winding_field.field import solve

    run = prepared_run(tmp_path / "run")
    checkpoint = run / "field/CHECKPOINT.json"

    def stop():
        return checkpoint.exists() and len(json.loads(checkpoint.read_bytes())["chunks"]) >= limit

    config = {"sweeps": 1, "field_chunk": 8}
    assert solve(run, config, stop=stop)["status"] == "PAUSED"
    saved = json.loads(checkpoint.read_bytes())
    assert len(saved["chunks"]) == limit
    assert not saved["progress"].get("initialization_done")
    assert not (run / "FIELD.json").exists()
    assert solve(run, config)["status"] == "FIELD_READY"


def test_field_initialization_preserves_disk_reserve(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import winding_field.field as field
    from winding_field.field import solve
    from field_store import StorageBudgetError

    run = prepared_run(tmp_path / "run")
    checkpoint = run / "field/CHECKPOINT.json"
    reserve = 16 << 20
    initializing = False
    Store = field.BlockStore

    def initialization_store(*args, **kwargs):
        nonlocal initializing
        store = Store(*args, **kwargs)
        initializing = True
        return store

    monkeypatch.setattr(field, "BlockStore", initialization_store)
    monkeypatch.setattr(
        "field_store.shutil.disk_usage",
        lambda path: SimpleNamespace(free=reserve if initializing else reserve * 10),
    )
    try:
        solve(run, {"sweeps": 1, "field_chunk": 8, "reserve_bytes": reserve})
    except StorageBudgetError:
        pass
    assert json.loads(checkpoint.read_bytes())["chunks"] == {}
    assert not (run / "FIELD.json").exists()


def test_initial_checkpoint_space_failure_can_resume_when_space_recovers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from winding_field.field import solve
    from field_store import StorageBudgetError

    run = prepared_run(tmp_path / "run")
    free = 0
    monkeypatch.setattr("field_store.shutil.disk_usage", lambda p: SimpleNamespace(free=free))
    config = dict(sweeps=1, field_chunk=8, reserve_bytes=16 << 20)
    with pytest.raises(StorageBudgetError):
        solve(run, config)
    assert not (run / "FIELD.json").exists()
    free = 1 << 30
    assert solve(run, config)["status"] == "FIELD_READY"


def test_resumed_fibre_archive_uses_indexed_tile_lookup(tmp_path):
    import sqlite3
    from winding_field.fibres import trace

    run = prepared_run(tmp_path / "run")
    count = json.loads((run / "FIBRES.json").read_bytes())["count"]
    with sqlite3.connect(run / "fibres/fibres.sqlite") as db:
        db.execute("DROP INDEX IF EXISTS curves_tile")
    (run / "FIBRES.json").unlink()
    assert trace(run, {"seed_stride": 4})["count"] == count
    with sqlite3.connect(run / "fibres/fibres.sqlite") as db:
        plan = db.execute(
            "EXPLAIN QUERY PLAN SELECT points FROM curves WHERE tile=?", ("0_0_0",)
        ).fetchall()
        assert all("SCAN curves" not in row[-1] for row in plan), plan
        assert db.execute("SELECT count(*) FROM curves WHERE tile=?", ("0_0_0",)).fetchone()[0] > 0


@pytest.mark.parametrize("pitch", [8.65, 17.0, 130.0, 300.0])
def test_exact_field_pitch_retains_fine_prior_and_strict_extent(tmp_path, pitch):
    from winding_field.config import field_geometry

    meta = dict(
        shape_zyx=[83, 92, 101],
        spacing_xyz=[8.64, 9.0, 10.0],
        origin_xyz=[13.0, 27.0, 45.0],
        frame="test",
    )
    geometry = field_geometry(meta, pitch)
    np.testing.assert_array_equal(geometry["spacing_xyz"], [pitch] * 3)
    np.testing.assert_array_equal(geometry["origin_xyz"], meta["origin_xyz"])
    extent = (np.array(meta["shape_zyx"])[::-1] - 1) * meta["spacing_xyz"]
    np.testing.assert_array_equal(
        geometry["shape_zyx"], (np.floor(extent / pitch).astype(int) + 1)[::-1]
    )
    assert np.all((np.array(geometry["shape_zyx"])[::-1] - 1) * pitch <= extent)
    assert meta["spacing_xyz"] == [8.64, 9.0, 10.0]


@pytest.mark.parametrize("spacing", [8.64, 8.65])
def test_explicit_native_pitch_retains_roundoff_boundary_plane(spacing):
    from winding_field.config import field_geometry

    meta = dict(shape_zyx=[128] * 3, spacing_xyz=[spacing] * 3, origin_xyz=[0.0] * 3, frame="test")
    assert field_geometry(meta, spacing)["shape_zyx"] == [128] * 3


def test_coarse_field_uses_physical_constraints_and_mesh_geometry(tmp_path):
    from winding_field.field import solve, open_field
    from winding_field.meshes import mesh

    run = prepared_run(tmp_path / "run", shape=(12, 12, 32), chunk=4)
    config = dict(sweeps=1, field_chunk=32, field_spacing_um=7.0, seed_stride=4)
    receipt = solve(run, config)
    assert receipt["geometry"]["spacing_xyz"] == [7.0] * 3
    assert receipt["geometry"]["shape_zyx"] == [7, 5, 9]
    assert receipt["fibre_constraint_rows"] > 0
    with open_field(run) as src:
        field, valid = src.read([0] * 3, list(src.shape))
        np.testing.assert_allclose(
            field, np.broadcast_to(np.arange(9) * 7 / 16, field.shape), atol=2e-4
        )
        assert valid.all()
    result = mesh(run, dict(config, mesh_tile=4))
    assert result["status"] == "MESH_READY"
    with pytest.raises(ValueError, match="identity|differ"):
        solve(run, dict(config, field_spacing_um=8.0))


def test_physical_sampling_orients_directors_and_preserves_support_holes(tmp_path):
    from winding_field.artifacts import Tiles
    from winding_field.field import sample_prior

    maps = Tiles(tmp_path / "maps", shape=(4, 4, 12), chunk=4, identity={})
    for key in maps.keys():
        lo, hi = maps.box(key)
        shape = tuple(hi - lo)
        normals = np.zeros(shape + (3,), np.float32)
        normals[..., 0] = (-1) ** key[2]
        support = np.ones(shape, bool)
        if key[2] == 1:
            support[:, :, 1] = False
        maps.write(
            key, dict(normals=normals, density=np.full(shape, 1 / 16, np.float32), support=support)
        )
    meta = dict(shape_zyx=[4, 4, 12], origin_xyz=[10.0, 20.0, 30.0], spacing_xyz=[2.0, 3.0, 4.0])
    geometry = dict(origin_xyz=meta["origin_xyz"], spacing_xyz=[3.0] * 3, shape_zyx=[4, 4, 8])
    cal = dict(mode="planar", axis_xyz=[1.0, 0.0, 0.0])
    target, weights, valid = sample_prior(maps, meta, geometry, cal, [0, 0, 0], [4, 4, 8])
    assert not valid[:, :, 3].any()  # x=4.5 interpolates an unsupported fine node.
    assert valid[:, :, :3].all() and valid[:, :, 4:].all()
    np.testing.assert_allclose(target[valid][:, 0], 1 / 16)
    assert not target[~valid].any()
    maps.close()


def qual_field_fixture(run, shape, points):
    import io
    import sqlite3
    from winding_field.artifacts import Tiles
    from winding_field.util import atomic_json, file_sha

    maps = Tiles(run / "maps", shape=shape, chunk=8, identity={})
    for key in maps.keys():
        lo, hi = maps.box(key)
        local = tuple(hi - lo)
        normals = np.zeros(local + (3,), np.float32)
        normals[..., 0] = 1
        maps.write(
            key,
            dict(
                density=np.full(local, 1 / 100, np.float32),
                normals=normals,
                support=np.ones(local, bool),
            ),
        )
    meta = dict(
        shape_zyx=list(shape),
        spacing_xyz=[9.0] * 3,
        origin_xyz=[10.0, 20.0, 30.0],
        frame="test",
        prediction="qual-v2",
        tiles_sha256=maps.identity(),
    )
    maps.close()
    atomic_json(run / "PREPARED.json", meta)
    (run / "fibres").mkdir()
    blob = io.BytesIO()
    np.save(blob, points, allow_pickle=False)
    with sqlite3.connect(run / "fibres/fibres.sqlite") as db:
        db.executescript(
            "CREATE TABLE curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT);"
            "CREATE VIRTUAL TABLE curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);"
        )
        db.execute("INSERT INTO curves VALUES (1,?,?,?)", ("qual", blob.getvalue(), "V"))
        box = tuple(v for pair in zip(points.min(0), points.max(0)) for v in pair)
        db.execute("INSERT INTO curve_bounds VALUES (1,?,?,?,?,?,?)", box)
    atomic_json(run / "FIBRES.json", dict(sha256=file_sha(run / "fibres/fibres.sqlite")))


@pytest.mark.parametrize("pitch", [17.0, 130.0])
@pytest.mark.parametrize("point_count", [2, 80])
def test_long_qual_spline_constrains_blocks_away_from_its_origin(
    tmp_path, monkeypatch, pitch, point_count
):
    import winding_field.field as field

    run = tmp_path / "run"
    points = np.column_stack([np.full(80, 41.5), np.full(80, 51.5), 30 + np.arange(80) * 9.0])
    if point_count == 2:
        points = points[68:70]
    qual_field_fixture(run, (80, 24, 24), points)
    calls = []
    pair = field.pair_constraints

    def constraints(a, b, origin, *args, **kwargs):
        calls.append((origin.copy(), a.copy(), b.copy()))
        return pair(a, b, origin, *args, **kwargs)

    monkeypatch.setattr(field, "pair_constraints", constraints)
    result = field.solve(run, dict(sweeps=1, field_chunk=4, field_spacing_um=pitch))
    assert result["fibre_constraint_rows"] > 0
    assert any(origin[2] > 30 and len(a) for origin, a, b in calls)
    for origin, a, b in calls:
        assert len(np.unique(np.c_[a, b], axis=0)) == len(a)
    with field.open_field(run) as source:
        values, support = source.read([0] * 3, list(source.shape))
        expected = np.broadcast_to(np.arange(source.shape[2]) * pitch / 100, values.shape)
        np.testing.assert_allclose(values, expected, atol=2e-4)
        assert support.all()


def test_qual_segments_constrain_every_intersected_small_block(tmp_path, monkeypatch):
    import winding_field.field as field

    run = tmp_path / "run"
    x = 10 + np.arange(0.5, 47, 1.6) * 9
    points = np.column_stack([x, np.full(len(x), 51.5), np.full(len(x), 61.5)])
    qual_field_fixture(run, (8, 8, 48), points)

    class Checked(Exception):
        pass

    # Exercise the real constraint provider over the solver's entire block schedule.
    def check_blocks(root, shape, chunk, identity, load, **options):
        checked = 0
        for key in options["order"]:
            lo = np.maximum(np.array(key) * chunk - options["halo"], 0)
            hi = np.minimum(np.array(key) * chunk + chunk + options["halo"], shape)
            low = np.array([10.0, 20.0, 30.0]) + lo[::-1] * 3
            high = np.array([10.0, 20.0, 30.0]) + (hi - 1)[::-1] * 3
            intersects = (
                low[1] < 51.5 < high[1]
                and low[2] < 61.5 < high[2]
                and max(low[0], x[0]) < min(high[0], x[-1])
            )
            rows = options["constraints"](lo, hi, None)
            if intersects:
                assert rows is not None, f"missing spline in block {key}"
                interior = np.zeros(tuple(hi - lo), bool)
                interior[1:-1, 1:-1, 1:-1] = True
                assert rows.matrix[:, interior.ravel()].nnz > 0, (
                    f"only fixed boundary nodes in block {key}"
                )
                checked += 1
            else:
                assert rows is None
        assert checked == 144
        raise Checked

    monkeypatch.setattr(field, "solve_blocks", check_blocks)
    with pytest.raises(Checked):
        field.solve(run, dict(field_spacing_um=3.0, field_chunk=4, sweeps=1))


def _checkpoint_digests(run):
    checkpoint = json.loads((run / "field/CHECKPOINT.json").read_bytes())
    return [record["sha256"] for _, record in sorted(checkpoint["chunks"].items())]


@pytest.mark.parametrize("budget", [0, 16384, 1 << 20])
def test_observation_cache_preserves_checkpoint_bytes_and_skips_reloads(
    tmp_path, monkeypatch, budget
):
    from winding_field import field

    original = field.sample_prior
    calls = []

    def counted(*args, **kwargs):
        calls.append(tuple(map(int, (*args[4], *args[5]))))
        return original(*args, **kwargs)

    monkeypatch.setattr(field, "sample_prior", counted)
    # Count the initialization reads in this process (workers would read them elsewhere).
    from winding_field import global_pairs

    monkeypatch.setattr(global_pairs, "default_workers", lambda: 1)
    results = {}
    for name, observation_budget in (("reference", 0), ("candidate", budget)):
        calls.clear()
        run = prepared_run(tmp_path / name)
        receipt = field.solve(
            run,
            dict(
                sweeps=3,
                field_chunk=8,
                constraint_cache_bytes=1 << 20,
                observation_cache_bytes=observation_budget,
            ),
        )
        with field.open_field(run) as source:
            values = source.read([0] * 3, [6, 6, 24])
        results[name] = (receipt, list(calls), _checkpoint_digests(run), values)
    reference, candidate = results["reference"], results["candidate"]
    assert candidate[2] == reference[2]
    for a, b in zip(candidate[3], reference[3]):
        np.testing.assert_array_equal(a, b)
    assert candidate[0]["fibre_constraint_rows"] == reference[0]["fibre_constraint_rows"]

    def solver(name):
        progress = json.loads((tmp_path / name / "field/CHECKPOINT.json").read_bytes())["progress"]
        return {k: v for k, v in progress["solver"].items() if k != "elapsed_s"}, progress[
            "completed_sweeps"
        ]

    assert solver("candidate") == solver("reference")
    stats = candidate[0]["observation_cache_stats"]
    assert stats["budget"] == budget and stats["hits"] + stats["misses"] == 9
    assert stats["peak_bytes"] <= budget and stats["entries"] == stats["retained_bytes"] == 0
    assert (
        json.loads((tmp_path / "candidate/FIELD.json").read_bytes())["observation_cache_stats"]
        == stats
    )
    # Three initialization cores are never retained; every solver miss reads the prior once.
    assert len(candidate[1]) == 3 + stats["misses"] and len(reference[1]) == 3 + 9
    if budget == 0:
        assert stats["hits"] == 0 and stats["peak_entries"] == 0 and stats["denied_entries"] == 0
    elif budget == 16384:
        assert stats["peak_entries"] == 1 and stats["hits"] == 2 and stats["denied_entries"] == 6
    else:
        assert stats["hits"] == 6 and stats["misses"] == 3 and stats["peak_entries"] == 3


@pytest.mark.parametrize("name", ["observation_cache_bytes", "field_cache_bytes"])
@pytest.mark.parametrize("budget", [True, -1, 1.5])
def test_new_cache_budgets_reject_invalid_values_before_io(tmp_path, name, budget):
    from winding_field.config import validate_config
    from winding_field.field import solve

    with pytest.raises(ValueError, match=name):
        solve(tmp_path / "missing", {name: budget})
    with pytest.raises(ValueError, match=name):
        validate_config({name: budget})
    assert not (tmp_path / "missing").exists()


def test_field_cache_budget_changes_reads_not_checkpoint(tmp_path):
    from winding_field import field

    results = {}
    for budget in (0, 64 << 20, 1 << 30):
        run = prepared_run(tmp_path / str(budget), shape=(6, 6, 32))
        receipt = field.solve(run, dict(sweeps=3, field_chunk=8, field_cache_bytes=budget))
        results[budget] = (_checkpoint_digests(run), receipt["solver"]["cache_stats"])
    assert results[0][0] == results[64 << 20][0] == results[1 << 30][0]
    assert results[0][1]["hits"] == 0 and results[0][1]["peak_cache_bytes"] == 0
    assert results[1 << 30][1]["hits"] > 0 and results[1 << 30][1]["loads"] < results[0][1]["loads"]


def test_observation_cache_arrays_are_sealed(tmp_path, monkeypatch):
    from winding_field import field
    import block_solver

    seen = []
    original = block_solver._observations

    def record(value, shape):
        seen.append(value)
        return original(value, shape)

    monkeypatch.setattr(block_solver, "_observations", record)
    run = prepared_run(tmp_path / "run")
    field.solve(run, dict(sweeps=2, field_chunk=8, observation_cache_bytes=1 << 20))
    assert len(seen) == 6
    assert all(not array.flags.writeable for value in seen for array in value[:3])
    assert seen[0][0] is seen[3][0]


def test_continue_publishes_complete_approximate_field(tmp_path):
    from winding_field.field import solve, open_field

    run = prepared_run(tmp_path / "run", shape=(6, 6, 32), chunk=8, variable=True)
    result = solve(run, dict(sweeps=2, field_chunk=8, maxiter=1, nonconvergence_policy="continue"))
    assert result["status"] == "FIELD_READY"
    assert result["solver"]["nonconverged_solves"] > 0
    assert result["solver"]["local_solves_converged"] is False
    assert len(result["solver"]["completed_sweeps"]) == 2
    assert "not converged" in result["numerical_status"]
    assert result["global_converged"] is False and result["physical_acceptance"] is False
    with open_field(run) as source:
        values, present = source.read([0] * 3, [6, 6, 32])
    assert present.all() and np.isfinite(values).all()


def test_one_block_covering_the_grid_is_a_global_solve_read_with_one_decode(tmp_path, monkeypatch):
    from winding_field import surface_sources
    from winding_field.config import validate_config
    from winding_field.field import open_field, solve

    validate_config(dict(field_chunk=640))
    run = prepared_run(tmp_path / "run")  # 6 x 6 x 24 nodes
    assert solve(run, {"sweeps": 1, "field_chunk": 640})["status"] == "FIELD_READY"
    assert json.loads((run / "field/CHECKPOINT.json").read_bytes())["chunk"] == 640
    decoded = []
    real = surface_sources._decode_chunk
    monkeypatch.setattr(surface_sources, "_decode_chunk", lambda *a: decoded.append(1) or real(*a))
    with open_field(run) as source:
        for _ in range(3):
            field, _ = source.read([0, 0, 0], [6, 6, 24])
    assert len(decoded) == 1
    np.testing.assert_allclose(field[3, 3] - field[3, 3, 0], np.arange(24) / 8, atol=2e-4)


def test_field_halo_is_part_of_the_problem_and_wider_overlap_converges_faster(tmp_path):
    from winding_field.config import validate_config
    from winding_field.field import open_field, solve

    with pytest.raises(ValueError):
        validate_config(dict(field_halo=0))

    def field_after(name, **config):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        solve(run, dict(config, rtol=1e-10, maxiter=10000))
        halo = json.loads((run / "field/CHECKPOINT.json").read_bytes())["identity"]["halo"]
        with open_field(run) as source:
            values, _ = source.read([0, 0, 0], [6, 6, 32])
        return run, halo, values - values.mean()

    _, _, exact = field_after("global", sweeps=1, field_chunk=64)
    narrow_run, narrow_halo, narrow = field_after("narrow", sweeps=2, field_chunk=8)
    _, wide_halo, wide = field_after("wide", sweeps=2, field_chunk=8, field_halo=8)
    assert (narrow_halo, wide_halo) == (2, 8)
    assert np.abs(wide - exact).max() < np.abs(narrow - exact).max()
    with pytest.raises(ValueError, match="identity|differ"):
        solve(narrow_run, dict(sweeps=2, field_chunk=8, field_halo=8, rtol=1e-10, maxiter=10000))


def test_general_solver_accepts_amg_preconditioner():
    from winding_field.config import validate_config

    validate_config({"preconditioner": "amg"})
    validate_config({"preconditioner": "jacobi"})


def test_general_solver_validates_field_status_policy():
    from winding_field.config import validate_config

    validate_config({"field_status_policy": "checkpoints"})
    validate_config({"field_status_policy": "all"})
    with pytest.raises(ValueError, match="field_status_policy"):
        validate_config({"field_status_policy": "never"})


def test_field_coarse_factor_is_part_of_the_problem_and_cuts_the_error(tmp_path):
    from winding_field.config import validate_config
    from winding_field.field import open_field, solve

    validate_config(dict(field_coarse_factor=0))
    validate_config(dict(field_coarse_factor=4))
    for bad in (1, -2):
        with pytest.raises(ValueError):
            validate_config(dict(field_coarse_factor=bad))

    def field_after(name, **config):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        result = solve(run, dict(config, rtol=1e-10, maxiter=10000))
        assert result["status"] == "FIELD_READY"
        identity = json.loads((run / "field/CHECKPOINT.json").read_bytes())["identity"]
        with open_field(run) as source:
            values, _ = source.read([0, 0, 0], [6, 6, 32])
        return run, identity, values - values.mean()

    _, _, exact = field_after("global", sweeps=1, field_chunk=64)
    one_run, one, one_level = field_after("one", sweeps=2, field_chunk=8)
    _, two, two_level = field_after("two", sweeps=2, field_chunk=8, field_coarse_factor=4)
    assert "coarse" not in one and two["coarse"] == dict(
        kind="trilinear", factor=4, revision=2, floor=1e-3, step="energy"
    )
    assert np.abs(two_level - exact).max() < 0.1 * np.abs(one_level - exact).max()
    status = json.loads((tmp_path / "two/field/STATUS.json").read_bytes())
    assert status["coarse"]["factor"] == 4
    residuals = [s["relative_residual"] for s in status["completed_sweeps"]]
    assert 0 < residuals[-1] < residuals[0] <= 1
    with pytest.raises(ValueError, match="identity|differ"):
        solve(
            one_run, dict(sweeps=2, field_chunk=8, field_coarse_factor=4, rtol=1e-10, maxiter=10000)
        )


def test_field_colours_are_opt_in_and_the_worker_count_leaves_the_field_unchanged(tmp_path):
    from winding_field.config import validate_config
    from winding_field.field import open_field, solve

    validate_config(dict(field_coarse_factor=4, field_colours=8, field_local_workers=2))
    validate_config(dict(field_coarse_factor=4, field_colours=8, field_local_devices=["cuda:0"]))
    for bad, message in (
        (dict(field_coarse_factor=4, field_colours=4), "field_colours"),
        (dict(field_colours=8), "field_coarse_factor"),
        (dict(field_coarse_factor=4, field_local_workers=2), "field_colours 8"),
        (
            dict(field_coarse_factor=4, field_colours=8, field_local_workers=0),
            "field_local_workers",
        ),
        (
            dict(field_coarse_factor=4, field_colours=8, field_local_devices=[]),
            "field_local_devices",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            validate_config(bad)

    def field_after(name, **config):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        base = dict(sweeps=2, field_chunk=8, field_coarse_factor=4, rtol=1e-10, maxiter=10000)
        result = solve(run, {**base, "field_memory": "field", **config})
        assert result["status"] == "FIELD_READY"
        identity = json.loads((run / "field/CHECKPOINT.json").read_bytes())["identity"]
        with open_field(run) as source:
            values, _ = source.read([0, 0, 0], [6, 6, 32])
        return identity, values

    plain, _ = field_after("plain")
    one, one_values = field_after("one", field_colours=8)
    three, three_values = field_after("three", field_colours=8, field_local_workers=3)
    assert "colours" not in plain and one["colours"] == 8 and one == three
    np.testing.assert_array_equal(one_values, three_values)
    with pytest.raises(ValueError, match="sweep checkpoints"):
        field_after("disk", field_colours=8, field_memory="disk")


def test_field_memory_tiers_give_the_same_field_and_are_reported(tmp_path):
    from winding_field.config import validate_config
    from winding_field.field import memory_plan, open_field, solve

    for tier in ("auto", "disk", "field", "all"):
        validate_config(dict(field_memory=tier))
    with pytest.raises(ValueError, match="field_memory"):
        validate_config(dict(field_memory="ram"))

    gib = 1 << 30
    shape = (1000, 1000, 1000)
    assert memory_plan({}, shape, 64, 8, available=2 * gib)["tier"] == "disk"
    assert memory_plan({}, shape, 64, 8, available=2 * gib)["warning"]
    assert memory_plan({}, shape, 64, 8, available=12 * gib)["tier"] == "field"
    assert memory_plan({}, shape, 64, 8, available=60 * gib)["tier"] == "all"
    # Configured caches count as workspace and can push a host down a tier.
    busy = dict(prepared_cache_bytes=40 * gib)
    assert memory_plan(busy, shape, 64, 8, available=60 * gib)["tier"] == "field"
    advised = memory_plan(dict(field_advice=True), shape, 64, 8, available=60 * gib)
    assert advised["tier"] == "all" and advised["checkpoint"] == "block" and advised["note"]
    bounded = memory_plan({}, shape, 64, 8, bounded=True, available=60 * gib)
    assert bounded["checkpoint"] == "block" and "bounded" in bounded["note"]
    small = memory_plan(dict(field_cache_bytes=0), shape, 64, 8, available=60 * gib)
    assert small["checkpoint"] == "block" and "field_cache_bytes" in small["note"]

    def field_after(name, **config):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        result = solve(run, dict(config, sweeps=2, field_chunk=8, rtol=1e-10, maxiter=10000))
        assert result["status"] == "FIELD_READY"
        status = json.loads((run / "field/STATUS.json").read_bytes())
        with open_field(run) as source:
            values, _ = source.read([0, 0, 0], [6, 6, 32])
        return status, values

    reference, expected = field_after("disk", field_memory="disk")
    assert reference["memory"]["tier"] == "disk"
    assert reference["checkpoint_policy"]["policy"] == "block"
    for tier in ("field", "all"):
        status, values = field_after(tier, field_memory=tier)
        np.testing.assert_array_equal(values, expected)
        assert status["memory"]["tier"] == tier and status["memory"]["requested"] == tier
        assert status["checkpoint_policy"]["policy"] == "sweep"
        assert status["checkpoint_policy"]["pending_chunks"] == 0
        assert status["memory"]["field_cache_bytes"] >= status["memory"]["field_bytes"]
    status, _ = field_after("auto")
    assert status["memory"]["requested"] == "auto"
    assert status["memory"]["tier"] in ("disk", "field", "all")


def test_global_pair_scope_solves_the_same_problem_as_one_block(tmp_path):
    from winding_field.config import validate_config
    from winding_field.field import open_field, solve

    validate_config(dict(field_pair_scope="global"))
    with pytest.raises(ValueError, match="field_pair_scope"):
        validate_config(dict(field_pair_scope="crop"))

    def field_after(name, **config):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        result = solve(run, dict(config, rtol=1e-10, maxiter=10000, field_memory="disk"))
        assert result["status"] == "FIELD_READY" and result["fibre_constraint_rows"] > 0
        identity = json.loads((run / "field/CHECKPOINT.json").read_bytes())["identity"]
        with open_field(run) as source:
            values, _ = source.read([0, 0, 0], [6, 6, 32])
        return identity, values - values.mean()

    one, exact = field_after("global", sweeps=1, field_chunk=64, field_pair_scope="global")
    assert "pair_scope" not in one["source_identity"]  # One block: the ordinary rows.
    import constraint_templates

    built = []
    original = constraint_templates.StencilConstraints.__init__

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        built.append(self.spatial_index_nbytes)

    constraint_templates.StencilConstraints.__init__ = record
    try:
        blocked, values = field_after(
            "blocked", sweeps=8, field_chunk=8, field_coarse_factor=4, field_pair_scope="global"
        )
    finally:
        constraint_templates.StencilConstraints.__init__ = original
    assert built and all(n > 0 for n in built)
    assert blocked["source_identity"]["pair_scope"] == "global-grid-v1"
    assert np.abs(values - exact).max() < 1e-5


def test_coarse_factor_defaults_to_global_pairs(tmp_path, capsys):
    from winding_field.field import solve

    def identity_of(name, **config):
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        base = dict(rtol=1e-10, maxiter=10000, field_memory="disk", sweeps=2, field_chunk=8)
        assert solve(run, dict(base, **config))["status"] == "FIELD_READY"
        return json.loads((run / "field/CHECKPOINT.json").read_bytes())["identity"]

    auto = identity_of("auto", field_coarse_factor=4)
    assert auto["source_identity"]["pair_scope"] == "global-grid-v1"
    # Only the block-scope stall warning matters here; macOS also warns that host
    # memory cannot be measured (no /proc), which is unrelated to pair scope.
    assert "--field-pair-scope block stalls" not in capsys.readouterr().err
    block = identity_of("block", field_coarse_factor=4, field_pair_scope="block")
    assert "pair_scope" not in block["source_identity"]
    assert "pair-scope block stalls" in capsys.readouterr().err
    one_level = identity_of("one-level")
    assert "pair_scope" not in one_level["source_identity"]


def test_global_pairs_get_a_spatial_index_when_it_fits(capsys):
    from winding_field.field import pair_index_bytes

    need = 4096 + 1000 * 16 * 64
    assert pair_index_bytes(1000, available=4 * need) == need
    assert pair_index_bytes(1000, available=4 * need - 4) == 0
    assert "pair index needs" in capsys.readouterr().err
