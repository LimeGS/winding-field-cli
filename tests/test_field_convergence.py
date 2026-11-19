"""Automatic-mode acceptance; dense expectations do not use core operators."""

import io
import itertools
import json
import sqlite3

import numpy as np
import pytest

from winding_field.util import atomic_json, file_sha


def dense_problem(shape, spacing, target, weights, pairs, weight):
    """Independently assemble each global edge and trilinear fibre row once."""
    target = target.astype(float)
    weights = weights.astype(float)
    n = int(np.prod(shape))
    h = np.zeros((n, n))
    rhs = np.zeros(n)
    for ij in np.ndindex(shape):
        u = np.ravel_multi_index(ij, shape)
        for axis in range(3):
            other = list(ij)
            other[axis] += 1
            if other[axis] >= shape[axis]:
                continue
            other = tuple(other)
            v = np.ravel_multi_index(other, shape)
            row = np.zeros(n)
            row[u] = -1
            row[v] = 1
            w = (weights[ij] + weights[other]) / 2
            goal = spacing[2 - axis] * (target[ij][2 - axis] + target[other][2 - axis]) / 2
            h += w * np.outer(row, row)
            rhs += w * goal * row
    for a, b in pairs:
        row = np.zeros(n)
        for point, sign in ((a, 1), (b, -1)):
            base = np.minimum(np.floor(point).astype(int), np.array(shape)[::-1] - 2)
            fraction = np.asarray(point) - base
            for corner in itertools.product((0, 1), repeat=3):
                xyz = base + corner
                value = np.prod(
                    [fraction[i] if c else 1 - fraction[i] for i, c in enumerate(corner)]
                )
                row[np.ravel_multi_index(tuple(xyz[::-1]), shape)] += sign * value
        h += weight * np.outer(row, row)
    return h, rhs


def test_reference_global_pairs_matches_independent_assembled_system(tmp_path):
    from winding_field.bounded_constraints import build, PairCatalog
    from winding_field.global_residual import verify

    shape = (3, 4, 11)
    spacing = np.array([1.2, 2.0, 3.0])
    grid = dict(shape_zyx=list(shape), origin_xyz=[10.0, 20.0, 30.0], spacing_xyz=spacing.tolist())
    points = np.array([[0.2, 1.2, 0.3], [9.8, 2.7, 1.7]])
    physical = points * spacing + grid["origin_xyz"]
    blob = io.BytesIO()
    np.save(blob, physical, allow_pickle=False)
    dbpath = tmp_path / "fibres.sqlite"
    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE curves (id INTEGER PRIMARY KEY,points BLOB)")
        db.execute("INSERT INTO curves VALUES (1,?)", (blob.getvalue(),))
    build(tmp_path / "pairs", dbpath, file_sha(dbpath), grid, weight=0.7, quota_bytes=8 << 20)
    # The single global edge is sampled uniformly in physical arclength.
    count = int(np.ceil(np.linalg.norm(physical[1] - physical[0]) / spacing.min()))
    samples = np.array([points[0] + i / count * (points[1] - points[0]) for i in range(count + 1)])
    pairs = list(zip(samples[:-1], samples[1:]))
    rng = np.random.default_rng(178)
    target = rng.normal(size=(*shape, 3)).astype("f4")
    weights = rng.uniform(0.1, 1.7, size=shape).astype("f4")
    weights[1, 2, 5] = 1e-3
    h, rhs = dense_problem(shape, spacing, target, weights, pairs, 0.7)
    phi = np.zeros(np.prod(shape))
    phi[1:] = np.linalg.solve(h[1:, 1:], rhs[1:])

    def observations(lo, hi):
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))
        return target[crop], weights[crop], weights[crop] > 0.001, {}

    with PairCatalog(tmp_path / "pairs", grid) as provider:
        for perturb in (0.0, 0.03):
            stored = phi.reshape(shape).copy()
            stored[1, 2, 4] += perturb
            result = verify(
                shape,
                4,
                spacing,
                lambda lo, hi: stored[tuple(slice(a, b) for a, b in zip(lo, hi))],
                observations,
                constraints=provider,
                rtol=1e-10,
            )
            expected = np.linalg.norm((h @ stored.ravel() - rhs)[1:])
            assert result["residual_l2"] == pytest.approx(expected, abs=2e-13)
            assert result["rhs_l2"] == pytest.approx(np.linalg.norm(rhs[1:]), rel=1e-13)
            assert result["free_nodes"] == 131
            assert result["global_converged"] == (perturb == 0)
            assert not result["physical_acceptance"]


def prepared(root, *, bounded=False, constant=False):
    from winding_field.artifacts import Tiles

    root.mkdir()
    shape = (3, 3, 9)
    maps = Tiles(
        root / ("coarse/maps" if bounded else "maps"),
        shape=shape,
        chunk=4,
        identity={"test": "convergence"},
    )
    for key in maps.keys():
        lo, hi = maps.box(key)
        local = tuple(hi - lo)
        density = (
            np.broadcast_to(np.where(np.arange(lo[2], hi[2]) < 4, 0.125, 0.03125), local)
            .astype("f4")
            .copy()
        )
        if constant:
            density.fill(0.125)
        normals = np.zeros((*local, 3), "f4")
        normals[..., 0] = 1
        target = normals * density[..., None]
        maps.write(
            key,
            dict(
                density=density,
                normals=normals,
                support=np.ones(local, bool),
                target=target,
                weights=np.ones(local, "f4"),
            ),
        )
    meta = dict(
        shape_zyx=list(shape),
        spacing_xyz=[1.0, 2.0, 3.0],
        origin_xyz=[0.0, 0.0, 0.0],
        frame="synthetic",
        tiles_sha256=maps.identity(),
        prediction="qual-v2",
    )
    if bounded:
        meta.update(
            bounded_preparation=True,
            calibration=dict(
                mode="planar",
                axis_xyz=[1.0, 0.0, 0.0],
                centre_xyz=[4.0, 2.0, 3.0],
                mean_density_per_um=0.125 if constant else 7 / 96,
            ),
        )
    maps.close()
    atomic_json(root / "PREPARED.json", meta)
    (root / "fibres").mkdir()
    points = np.array([[2.3, 2.2, 0.2], [2.3, 2.2, 5.8]])
    raw = io.BytesIO()
    np.save(raw, points, allow_pickle=False)
    with sqlite3.connect(root / "fibres/fibres.sqlite") as db:
        db.execute("CREATE TABLE curves (id INTEGER PRIMARY KEY,points BLOB)")
        db.execute("INSERT INTO curves VALUES (1,?)", (raw.getvalue(),))
    atomic_json(root / "FIBRES.json", dict(sha256=file_sha(root / "fibres/fibres.sqlite")))
    return root


def automatic(root, max_sweeps=40, global_rtol=1e-4, stop=None):
    from winding_field.field import solve
    from winding_field.field_convergence import Controller, policy

    settings = policy(max_sweeps, global_rtol)
    return solve(
        root,
        dict(sweeps=max_sweeps, field_chunk=4, rtol=1e-10, reserve_bytes=0),
        stop,
        convergence=Controller(root, settings),
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_automatic_initial_convergence_publishes_only_verified_field(tmp_path, bounded):
    root = prepared(tmp_path / "run", bounded=bounded, constant=True)
    result = automatic(root)
    assert result["reason"] == "VERIFIED_CONVERGED"
    assert result["completed_sweeps"] == []
    assert result["global_converged"] and not result["physical_acceptance"]
    assert result["verification"]["residual_l2"] <= 1e-4 * result["verification"]["rhs_l2"]
    assert json.loads((root / "FIELD.json").read_bytes())["global_converged"]


def test_local_convergence_and_budget_exhaustion_are_not_global_convergence(tmp_path):
    root = prepared(tmp_path / "run")
    result = automatic(root, max_sweeps=1, global_rtol=1e-10)
    assert result["reason"] == "SWEEP_LIMIT"
    assert len(result["completed_sweeps"]) == 1
    saved = json.loads((root / "field/CHECKPOINT.json").read_bytes())
    assert saved["progress"]["solver"]["converged"]
    assert not result["global_converged"] and not result["physical_acceptance"]
    assert result["verification"]["relative_residual"] > 1e-10
    assert not (root / "FIELD.json").exists()


def test_automatic_stops_before_budget_and_checks_stored_codec(tmp_path):
    root = prepared(tmp_path / "run")
    result = automatic(root, max_sweeps=120, global_rtol=1e-3)
    assert result["reason"] == "VERIFIED_CONVERGED"
    assert 0 < len(result["completed_sweeps"]) < 120
    assert result["verification"]["checkpoint_sha256"] == file_sha(root / "field/CHECKPOINT.json")
    assert result["verification"]["fixed_storage_max_error"] == 0


def test_stagnation_persists_instead_of_false_precision_convergence(tmp_path):
    root = prepared(tmp_path / "run")
    result = automatic(root, max_sweeps=150, global_rtol=1e-15)
    assert result["reason"] == "STAGNATED"
    assert len(result["completed_sweeps"]) < 150
    assert not result["global_converged"] and not (root / "FIELD.json").exists()
    before = (root / "convergence/STATE.json").read_bytes()
    assert automatic(root, max_sweeps=150, global_rtol=1e-15)["reason"] == "STAGNATED"
    assert (root / "convergence/STATE.json").read_bytes() == before


@pytest.mark.parametrize("phase", ["partial", "boundary", "verification"])
def test_pause_resume_does_not_repeat_sweeps_or_verifications(tmp_path, monkeypatch, phase):
    from winding_field import global_residual

    root = prepared(tmp_path / "paused")
    reference = prepared(tmp_path / "reference")

    def stop():
        path = root / "field/CHECKPOINT.json"
        if not path.exists():
            return False
        p = json.loads(path.read_bytes())["progress"]
        return bool(p.get("completed_sweeps")) if phase == "boundary" else bool(p.get("solver"))

    original = global_residual.verify
    if phase == "verification":

        def interrupt(*args, **kwargs):
            if len(
                json.loads((root / "field/CHECKPOINT.json").read_bytes())["progress"][
                    "completed_sweeps"
                ]
            ):
                return dict(status="PAUSED", global_converged=False, physical_acceptance=False)
            return original(*args, **kwargs)

        monkeypatch.setattr(global_residual, "verify", interrupt)
        stop = lambda: False
    assert automatic(root, max_sweeps=12, global_rtol=1e-8, stop=stop)["reason"] == "PAUSED"
    monkeypatch.setattr(global_residual, "verify", original)
    resumed = automatic(root, max_sweeps=12, global_rtol=1e-8)
    full = automatic(reference, max_sweeps=12, global_rtol=1e-8)
    assert resumed["reason"] == full["reason"]
    assert resumed["completed_sweeps"] == full["completed_sweeps"]
    for path in (root, reference):
        state = json.loads((path / "convergence/STATE.json").read_bytes())["data"]
        assert [r["sweep"] for r in state["records"]] == list(range(13))
    from winding_field.surface_sources import FieldSource

    a = json.loads((root / "field/CHECKPOINT.json").read_bytes())
    b = json.loads((reference / "field/CHECKPOINT.json").read_bytes())
    assert {k: v["sha256"] for k, v in a["chunks"].items()} == {
        k: v["sha256"] for k, v in b["chunks"].items()
    }


def workflow_source(root, **kwargs):
    root = prepared(root, **kwargs)
    (root / ".run.lock").touch()
    stages = {
        name: dict(receipt=receipt, sha256=file_sha(root / receipt))
        for name, receipt in (("prepare", "PREPARED.json"), ("fibres", "FIBRES.json"))
    }
    atomic_json(
        root / "RUN.json",
        dict(
            schema="winding-field-run-v1",
            config=dict(field_chunk=4, rtol=1e-10, device="cpu", reserve_bytes=0, threads=1),
            stages=stages,
        ),
    )
    return root


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("bounded", [False, True])
def test_end_to_end_new_mode_reuses_inputs_and_preserves_source(tmp_path, monkeypatch, bounded):
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source", bounded=bounded)
    before = snapshot(source)
    from winding_field import fibres, predict

    monkeypatch.setattr(fibres, "trace", lambda *a, **k: pytest.fail("must not retrace fibres"))
    monkeypatch.setattr(predict, "prepare", lambda *a, **k: pytest.fail("must not prepare CT"))
    out = tmp_path / "auto"
    result = converge(source, out, max_sweeps=40, global_rtol=1e-3)
    assert result["reason"] == "VERIFIED_CONVERGED"
    assert snapshot(source) == before
    assert not (out / "RUN.json").exists()
    assert not any((out / p).exists() for p in ("meshes", "flat", "surfaces"))
    assert (out / "fibres/fibres.sqlite").stat().st_ino != (
        source / "fibres/fibres.sqlite"
    ).stat().st_ino
    completed = snapshot(out)
    assert converge(out, resume=True)["reason"] == "VERIFIED_CONVERGED"
    assert snapshot(out) == completed


def test_reference_actual_multiblock_controller_reaches_independent_dense_solution(tmp_path):
    from winding_field.field_convergence import converge
    from winding_field.field import open_field

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    a = np.array([0.2, 0.2, 0.2])
    b = np.array([7.8, 1.8, 1.8])
    spacing = np.array([1.0, 2.0, 3.0])
    points = np.array([a, b]) * spacing
    blob = io.BytesIO()
    np.save(blob, points, allow_pickle=False)
    with sqlite3.connect(source / "fibres/fibres.sqlite") as db:
        db.execute("UPDATE curves SET points=?", (blob.getvalue(),))
    atomic_json(source / "FIBRES.json", dict(sha256=file_sha(source / "fibres/fibres.sqlite")))
    run = json.loads((source / "RUN.json").read_bytes())
    run["stages"]["fibres"]["sha256"] = file_sha(source / "FIBRES.json")
    atomic_json(source / "RUN.json", run)
    result = converge(source, out, max_sweeps=300, global_rtol=1e-5)
    assert result["reason"] == "VERIFIED_CONVERGED"
    shape = (3, 3, 9)
    target = np.zeros((*shape, 3), "f4")
    target[..., 0] = np.broadcast_to(np.where(np.arange(9) < 4, 0.125, 0.03125), shape)
    count = int(np.ceil(np.linalg.norm(points[1] - points[0])))
    samples = np.array([a + (b - a) * i / count for i in range(count + 1)])
    h, rhs = dense_problem(
        shape, spacing, target, np.ones(shape), list(zip(samples[:-1], samples[1:])), 1.0
    )
    expected = np.zeros(81)
    expected[1:] = np.linalg.solve(h[1:, 1:], rhs[1:])
    with open_field(out) as field:
        values, _ = field.read([0] * 3, list(shape))
    residual = np.linalg.norm((h @ values.ravel() - rhs)[1:])
    assert residual <= 1e-5 * np.linalg.norm(rhs[1:])
    assert residual == pytest.approx(result["verification"]["residual_l2"], abs=2e-14)
    np.testing.assert_allclose(values.ravel(), expected, atol=2e-4)


def test_exhausted_automatic_generation_continues_without_repeating_completed_sweeps(tmp_path):
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"
    full = tmp_path / "full"
    exhausted = converge(source, first, max_sweeps=2, global_rtol=1e-8)
    assert exhausted["reason"] == "SWEEP_LIMIT"
    before = snapshot(first)
    continued = converge(first, second, max_sweeps=12)
    reference = converge(source, full, max_sweeps=12, global_rtol=1e-8)
    assert continued["completed_sweeps"] == reference["completed_sweeps"]
    assert continued["completed_sweeps"][:2] == exhausted["completed_sweeps"]
    assert snapshot(first) == before
    a = json.loads((second / "field/CHECKPOINT.json").read_bytes())
    b = json.loads((full / "field/CHECKPOINT.json").read_bytes())
    assert {k: v["sha256"] for k, v in a["chunks"].items()} == {
        k: v["sha256"] for k, v in b["chunks"].items()
    }
    assert a["sweep_extensions"][0]["from"] == 2 and a["sweep_extensions"][0]["to"] == 12


@pytest.mark.parametrize("target", ["FIELD.json", "field/CHECKPOINT.json"])
def test_legacy_fields_are_not_silently_migrated(tmp_path, target):
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    path = source / target
    path.parent.mkdir(exist_ok=True)
    atomic_json(path, {"legacy": True})
    before = snapshot(source)
    with pytest.raises(ValueError, match="legacy"):
        converge(source, tmp_path / "output", max_sweeps=12)
    assert not (tmp_path / "output").exists() and snapshot(source) == before


@pytest.mark.parametrize(
    "change", ["input", "state", "checkpoint", "policy", "implementation", "field"]
)
def test_changed_auto_generation_is_rejected_before_retained_writes(tmp_path, monkeypatch, change):
    from winding_field import field_convergence, workflow

    source = workflow_source(tmp_path / "source", constant=change == "field")
    out = tmp_path / "auto"
    field_convergence.converge(source, out, max_sweeps=1)
    if change == "implementation":
        monkeypatch.setattr(workflow, "implementation_sha", lambda: "changed")
    elif change == "input":
        (out / "fibres/fibres.sqlite").write_bytes(b"corrupt")
    elif change == "state":
        atomic_json(out / "convergence/STATE.json", {})
    elif change == "field":
        atomic_json(out / "FIELD.json", {"global_converged": True})
    elif change == "checkpoint":
        cp = json.loads((out / "field/CHECKPOINT.json").read_bytes())
        cp["identity"]["rtol"] = 0.5
        atomic_json(out / "field/CHECKPOINT.json", cp)
    before = snapshot(out)
    with pytest.raises(ValueError):
        field_convergence.converge(
            out, resume=True, **({"global_rtol": 0.1} if change == "policy" else {})
        )
    assert snapshot(out) == before


@pytest.mark.parametrize("which", ["source", "output", "field"])
def test_concurrent_writer_is_rejected(tmp_path, which):
    import fcntl
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    if which != "source":
        converge(source, out, max_sweeps=1)
    lock = (
        source / ".run.lock"
        if which == "source"
        else out / (".run.lock" if which == "output" else "field/.writer.lock")
    )
    with lock.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(source if which == "source" else out)
        with pytest.raises((RuntimeError, BlockingIOError)):
            converge(source, out, max_sweeps=1) if which == "source" else converge(out, resume=True)
        assert snapshot(source if which == "source" else out) == before


@pytest.mark.parametrize(
    "sweeps,tolerance",
    [(None, 1e-6), (0, 1e-6), (-1, 1e-6), (True, 1e-6), (2, 0), (2, float("inf"))],
)
def test_invalid_limits_fail_before_output_creation(tmp_path, sweeps, tolerance):
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    with pytest.raises(ValueError):
        converge(source, tmp_path / "auto", max_sweeps=sweeps, global_rtol=tolerance)
    assert not (tmp_path / "auto").exists()


def test_cli_automatic_and_typed_yaml_settings(tmp_path, capsys):
    from winding_field.cli import main

    workflow_source(tmp_path / "source", constant=True)
    config = tmp_path / "settings.yaml"
    config.write_text("source: source\noutput: auto\nmax_sweeps: 2\nglobal_rtol: 0.001\n")
    assert main(["field-converge", "--cli-config", str(config), "--max-sweeps", "3"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "VERIFIED_CONVERGED" and result["policy"]["max_sweeps"] == 3
    assert main(["field-converge", str(tmp_path / "auto"), "--resume"]) == 0


def test_source_provenance_corruption_rejects_resume_without_writes(tmp_path):
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    converge(source, out, max_sweeps=1)
    (out / "source/RUN.json").write_bytes(b"changed")
    before = snapshot(out)
    with pytest.raises(ValueError):
        converge(out, resume=True)
    assert snapshot(out) == before


def test_input_mutation_during_verification_cannot_publish_field(tmp_path, monkeypatch):
    from winding_field.field_convergence import converge
    from winding_field import global_residual

    source = workflow_source(tmp_path / "source", constant=True)
    out = tmp_path / "auto"
    original = global_residual.verify

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (out / "FIBRES.json").write_bytes(b"{}")
        return result

    monkeypatch.setattr(global_residual, "verify", mutate)
    with pytest.raises(ValueError):
        converge(source, out, max_sweeps=2)
    assert not (out / "FIELD.json").exists()
    assert not (out / "CONVERGENCE_RESULT.json").exists()


def test_copy_interruption_resumes_pinned_admission_and_rejects_changed_runtime(
    tmp_path, monkeypatch
):
    from winding_field import field_convergence as fc, workflow

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    original = fc._copy

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("interrupted before admission commit")

    monkeypatch.setattr(fc, "_copy", interrupt)
    with pytest.raises(OSError):
        fc.converge(source, out, max_sweeps=2)
    before = snapshot(out)
    implementation = workflow.implementation_sha()
    monkeypatch.setattr(fc, "_copy", original)
    monkeypatch.setattr(workflow, "implementation_sha", lambda: "changed")
    with pytest.raises(ValueError):
        fc.converge(out, resume=True)
    assert snapshot(out) == before
    monkeypatch.setattr(workflow, "implementation_sha", lambda: implementation)
    assert fc.converge(out, resume=True)["reason"] == "SWEEP_LIMIT"


def test_convergence_input_symlink_is_rejected_without_copying(tmp_path):
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    original = source / "maps"
    moved = source / "elsewhere"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    before = snapshot(source)
    with pytest.raises(ValueError):
        converge(source, tmp_path / "out", max_sweeps=2)
    assert snapshot(source) == before and not (tmp_path / "out").exists()


def test_pending_admission_rejects_path_escape_before_any_copy(tmp_path, monkeypatch):
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    monkeypatch.setattr(fc, "_copy", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        fc.converge(source, out, max_sweeps=2)
    request = fc._checked(out / "CONVERGENCE.json")
    request["files"]["../escaped.json"] = dict(
        source="PREPARED.json",
        sha256=file_sha(source / "PREPARED.json"),
        bytes=(source / "PREPARED.json").stat().st_size,
    )
    fc._save(out / "CONVERGENCE.json", request)
    monkeypatch.undo()
    before = snapshot(out)
    with pytest.raises(ValueError):
        fc.converge(out, resume=True)
    assert snapshot(out) == before and not (tmp_path / "escaped.json").exists()


def test_cli_nonconverged_and_pause_are_not_success(tmp_path, capsys):
    from winding_field.cli import main

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    assert main(["field-converge", str(source), "--output", str(out), "--max-sweeps", "1"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "SWEEP_LIMIT"
    paused = tmp_path / "paused"
    # A stop present at the first controller callback is cooperative and resumable.
    from winding_field.field_convergence import converge

    result = converge(source, paused, max_sweeps=2, stop=lambda: True)
    assert result["reason"] == "PAUSED" and not (paused / "FIELD.json").exists()
    (paused / "STOP").touch()
    assert main(["field-converge", str(paused), "--resume"]) == 130
    (paused / "STOP").unlink()
    assert main(["field-converge", str(paused), "--resume"]) == 2


def test_numerical_failure_is_persisted_without_field_publication(tmp_path, monkeypatch):
    from winding_field.field_convergence import converge
    import block_solver

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"

    def fail(*args, **kwargs):
        raise RuntimeError("numerical failure")

    monkeypatch.setattr(block_solver, "solve_field", fail)
    with pytest.raises(RuntimeError):
        converge(source, out, max_sweeps=2)
    error = json.loads((out / "CONVERGENCE_ERROR.json").read_bytes())["data"]
    assert error["reason"] == "NUMERICAL_ERROR" and not error["global_converged"]
    assert not (out / "FIELD.json").exists()


@pytest.mark.parametrize("point", ["before", "after"])
def test_verification_journal_crash_does_not_repeat_committed_sweep(tmp_path, monkeypatch, point):
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    original = fc._save

    def crash(path, data):
        if Path(path).name == "STATE.json" and data["records"][-1]["sweep"] == 1:
            if point == "after":
                original(path, data)
            raise OSError("simulated journal crash")
        return original(path, data)

    from pathlib import Path

    monkeypatch.setattr(fc, "_save", crash)
    with pytest.raises(OSError):
        fc.converge(source, out, max_sweeps=2)
    completed = json.loads((out / "field/CHECKPOINT.json").read_bytes())["progress"][
        "completed_sweeps"
    ]
    assert len(completed) == 1
    monkeypatch.setattr(fc, "_save", original)
    result = fc.converge(out, resume=True)
    assert result["reason"] == "SWEEP_LIMIT" and result["completed_sweeps"][:1] == completed
    records = fc._checked(out / "convergence/STATE.json")["records"]
    assert [r["sweep"] for r in records] == [0, 1, 2]


def test_real_sigterm_during_local_solve_resumes_same_stored_field(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import winding_field
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "paused"
    before = snapshot(source)
    # Deterministically deliver a real signal inside a real local solve.
    script = """
import os,signal,sys
import block_solver
from winding_field.cli import main
original=block_solver.solve_field
def signal_during_solve(*args,**kwargs):
    os.kill(os.getpid(),signal.SIGTERM)
    return original(*args,**kwargs)
block_solver.solve_field=signal_during_solve
sys.exit(main(sys.argv[1:]))
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(Path(winding_field.__file__).parent.parent), os.environ.get("PYTHONPATH", "")]
        ),
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "field-converge",
            str(source),
            "--output",
            str(out),
            "--max-sweeps",
            "2",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 130, child.stderr
    paused = json.loads(child.stdout)
    assert paused["reason"] == "PAUSED" and not (out / "FIELD.json").exists()
    progress = json.loads((out / "field/CHECKPOINT.json").read_bytes())["progress"]
    assert progress["next_block"] == 1 and not progress["completed_sweeps"]
    resumed = converge(out, resume=True)
    reference = converge(source, tmp_path / "reference", max_sweeps=2)
    assert resumed["completed_sweeps"] == reference["completed_sweeps"]
    assert resumed["verification"]["residual_l2"] == reference["verification"]["residual_l2"]
    assert snapshot(source) == before


def test_writer_interleaved_before_publication_cannot_reuse_old_certification(
    tmp_path, monkeypatch
):
    from winding_field import field_convergence as fc
    from field_store import BlockStore

    source = workflow_source(tmp_path / "source", constant=True)
    out = tmp_path / "auto"
    original = fc.solve_blocks

    def interleave(root, *args, **kwargs):
        result = original(root, *args, **kwargs)
        saved = fc._read(root / "CHECKPOINT.json")
        with BlockStore(
            root, saved["shape"], saved["chunk"], saved["identity"], resume=True, reserve_bytes=0
        ) as writer:
            writer.commit(
                (0, 0, 0), np.full((3, 3, 4), 99.0), saved["progress"], np.ones((3, 3, 4), bool)
            )
        return result

    monkeypatch.setattr(fc, "solve_blocks", interleave)
    with pytest.raises(ValueError, match="verified|checkpoint"):
        fc.converge(source, out, max_sweeps=2)
    assert not (out / "FIELD.json").exists()


def test_crash_after_field_publication_recovers_over_stale_paused_result(tmp_path, monkeypatch):
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source", constant=True)
    out = tmp_path / "auto"
    assert fc.converge(source, out, max_sweeps=2, stop=lambda: True)["reason"] == "PAUSED"
    original = fc._save

    def crash(path, data):
        if path.name == "CONVERGENCE_RESULT.json":
            raise OSError("crash after field publication")
        return original(path, data)

    monkeypatch.setattr(fc, "_save", crash)
    with pytest.raises(OSError):
        fc.converge(out, resume=True)
    before = (out / "FIELD.json").read_bytes()
    monkeypatch.setattr(fc, "_save", original)
    assert fc.converge(out, resume=True)["reason"] == "VERIFIED_CONVERGED"
    assert (out / "FIELD.json").read_bytes() == before
    assert fc._checked(out / "CONVERGENCE_RESULT.json")["field_sha256"] == file_sha(
        out / "FIELD.json"
    )


def test_actual_mid_copy_process_death_does_not_poison_map_inventory(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import winding_field
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    script = """
import os,sys
from winding_field import field_convergence as fc
original=fc.shutil.copyfileobj
def die(src,dst,*args,**kwargs):
    if '/maps/' in src.name:
        dst.write(src.read(16));dst.flush();os.fsync(dst.fileno())
        os._exit(77)
    return original(src,dst,*args,**kwargs)
fc.shutil.copyfileobj=die
fc.converge(sys.argv[1],sys.argv[2],max_sweeps=1)
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(Path(winding_field.__file__).parent.parent), os.environ.get("PYTHONPATH", "")]
        ),
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(source), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 77, child.stderr
    assert converge(out, resume=True)["reason"] == "SWEEP_LIMIT"
    assert not any(p.name.startswith(".copy-") for p in (out / "maps").iterdir())


@pytest.mark.parametrize("directory", ["maps", "pairs", "convergence", "field", "source"])
def test_pending_copy_never_writes_through_managed_directory_symlink(
    tmp_path, monkeypatch, directory
):
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    external = tmp_path / "external"
    original = fc._copy
    monkeypatch.setattr(fc, "_copy", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        fc.converge(source, out, max_sweeps=1)
    external.mkdir()
    (out / directory).symlink_to(external, target_is_directory=True)
    before = snapshot(external)
    monkeypatch.setattr(fc, "_copy", original)
    with pytest.raises((ValueError, OSError)):
        fc.converge(out, resume=True)
    assert snapshot(external) == before


def test_unrelated_extra_map_file_remains_an_integrity_error(tmp_path, monkeypatch):
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    original = fc._copy

    def extra(*args, **kwargs):
        original(*args, **kwargs)
        (out / "maps/unrelated").write_bytes(b"user data")

    monkeypatch.setattr(fc, "_copy", extra)
    with pytest.raises(ValueError):
        fc.converge(source, out, max_sweeps=1)
    assert (out / "maps/unrelated").read_bytes() == b"user data"


def test_copy_directory_swap_race_does_not_redirect_payload_publication(tmp_path, monkeypatch):
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    external = tmp_path / "external"
    external.mkdir()
    original = fc.shutil.copyfileobj
    swapped = False

    def swap(src, dst, *args, **kwargs):
        nonlocal swapped
        if "/maps/" in src.name and not swapped:
            swapped = True
            (out / "maps").rename(out / "moved-maps")
            (out / "maps").symlink_to(external, target_is_directory=True)
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(fc.shutil, "copyfileobj", swap)
    with pytest.raises((ValueError, OSError)):
        fc.converge(source, out, max_sweeps=1)
    assert swapped and snapshot(external) == {}


@pytest.mark.parametrize("kind", ["resource", "storage"])
def test_core_resource_errors_are_not_reported_as_numerical_failure(tmp_path, monkeypatch, kind):
    from winding_field import field_convergence as fc
    from field_runtime import ResourceUnavailable
    from field_store import StorageBudgetError

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    error = (ResourceUnavailable if kind == "resource" else StorageBudgetError)(
        "disk reserve unavailable"
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(fc, "solve_blocks", fail)
    with pytest.raises(type(error)):
        fc.converge(source, out, max_sweeps=1)
    assert fc._checked(out / "CONVERGENCE_ERROR.json")["reason"] == "RESOURCE_ERROR"


def test_copy_crash_after_exclusive_link_recovers_its_own_temporary_alias(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import winding_field
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    before = snapshot(source)
    script = """
import os,sys
from winding_field import field_convergence as fc
original=fc.os.link
def die(*args,**kwargs):
    original(*args,**kwargs)
    os._exit(78)
fc.os.link=die
fc.converge(sys.argv[1],sys.argv[2],max_sweeps=1)
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(Path(winding_field.__file__).parent.parent), os.environ.get("PYTHONPATH", "")]
        ),
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(source), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 78, child.stderr
    assert converge(out, resume=True)["reason"] == "SWEEP_LIMIT"
    assert (out / "PREPARED.json").stat().st_nlink == 1 and snapshot(source) == before


def test_pending_admission_does_not_adopt_external_hardlinks(tmp_path, monkeypatch):
    import os
    from winding_field import field_convergence as fc

    source = workflow_source(tmp_path / "source")
    out = tmp_path / "auto"
    original = fc._copy
    monkeypatch.setattr(fc, "_copy", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        fc.converge(source, out, max_sweeps=1)
    os.link(source / "PREPARED.json", out / "PREPARED.json")
    before = snapshot(source)
    monkeypatch.setattr(fc, "_copy", original)
    with pytest.raises(ValueError, match="alias"):
        fc.converge(out, resume=True)
    assert snapshot(source) == before


def test_field_publication_itself_holds_the_core_writer_lease(tmp_path, monkeypatch):
    from winding_field import field_convergence as fc
    from field_store import BlockStore

    source = workflow_source(tmp_path / "source", constant=True)
    out = tmp_path / "auto"
    original = fc._publish
    checked = []

    def publish(path, data):
        if path.name == "FIELD.json":
            saved = fc._read(out / "field/CHECKPOINT.json")
            with pytest.raises(RuntimeError, match="writer"):
                with BlockStore(
                    out / "field",
                    saved["shape"],
                    saved["chunk"],
                    saved["identity"],
                    resume=True,
                    reserve_bytes=0,
                ):
                    pass
            checked.append(True)
        return original(path, data)

    monkeypatch.setattr(fc, "_publish", publish)
    assert fc.converge(source, out, max_sweeps=1)["reason"] == "VERIFIED_CONVERGED"
    assert checked
