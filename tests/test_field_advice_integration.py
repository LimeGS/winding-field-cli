"""Advisor on/off must preserve science and the existing verification cadence."""

import json

import numpy as np
import pytest

from winding_field import field, field_advice, global_residual


@pytest.mark.parametrize("route", ["ordinary", "bounded", "automatic"])
def test_opt_in_changes_no_scientific_work_or_verification_schedule(tmp_path, monkeypatch, route):
    from test_field_convergence import prepared
    from test_field import prepared_run
    from winding_field.field_convergence import Controller, policy
    from field_store import BlockStore

    original_verify = global_residual.verify
    checks = []
    commits = []
    original_commit = BlockStore.commit

    def verify(*args, **kwargs):
        checks.append(1)
        return original_verify(*args, **kwargs)

    def commit(store, key, *args, **kwargs):
        commits.append(tuple(key))
        return original_commit(store, key, *args, **kwargs)

    monkeypatch.setattr(global_residual, "verify", verify)
    monkeypatch.setattr(BlockStore, "commit", commit)
    snapshots = []
    results = []
    counts = []
    orders = []
    for enabled in (False, True):
        checks.clear()
        commits.clear()
        root = (
            prepared_run(tmp_path / str(enabled), shape=(3, 3, 9), chunk=4)
            if route == "ordinary"
            else prepared(tmp_path / str(enabled), bounded=route == "bounded")
        )
        config = dict(sweeps=3, field_chunk=4, rtol=1e-10, reserve_bytes=0)
        if enabled:
            config["field_advice"] = True
        controller = Controller(root, policy(3, 1e-12)) if route == "automatic" else None
        result = field.solve(root, config, convergence=controller)
        saved = json.loads((root / "field/CHECKPOINT.json").read_bytes())
        saved["progress"]["solver"].pop("elapsed_s")
        snapshots.append(saved)
        results.append(result)
        counts.append(len(checks))
        orders.append(list(commits))
        if enabled:
            sink = field_advice.Advice(root / "field", result.get("geometry"))
            assert len(sink.records()) == 3
            assert all(r["metrics"]["observed_count"] == 81 for r in sink.records())
            measured = list(sink.path.glob("verification-*.json"))
            assert len(measured) == counts[-1]
            assert all(field_advice._read(p)["elapsed_s"] >= 0 for p in measured)
        else:
            assert not (root / "field-advice").exists()
        assert not (root / "STOP").exists()
    assert counts == [dict(ordinary=0, bounded=1, automatic=4)[route]] * 2
    assert orders[0] == orders[1]
    a, b = snapshots
    assert a["identity"] == b["identity"] and a["progress"] == b["progress"]
    assert {k: v["sha256"] for k, v in a["chunks"].items()} == {
        k: v["sha256"] for k, v in b["chunks"].items()
    }
    assert results[0]["status"] == results[1]["status"]
    assert (
        results[0].get("verification", results[0])["global_converged"]
        == results[1].get("verification", results[1])["global_converged"]
    )
    assert not results[1].get("verification", results[1])["physical_acceptance"]


def test_sink_failure_keeps_science_and_valid_resume_counters(tmp_path, monkeypatch):
    from test_field import prepared_run

    root = prepared_run(tmp_path / "run", shape=(3, 3, 9), chunk=4)
    config = dict(sweeps=3, field_chunk=4, rtol=1e-10, reserve_bytes=0, field_advice=True)

    def stop():
        path = root / "field/CHECKPOINT.json"
        return path.exists() and json.loads(path.read_bytes())["progress"].get("next_block") == 1

    assert field.solve(root, config, stop)["status"] == "PAUSED"
    sink = field_advice.Advice(root / "field")
    assert (sink.path / "JOURNAL.json").is_file()
    result = field.solve(root, config)
    assert result["status"] == "FIELD_READY"
    assert [r["metrics"]["observed_count"] for r in sink.records()] == [81] * 3
    broken = prepared_run(tmp_path / "broken", shape=(3, 3, 9), chunk=4)

    def fail(*args, **kwargs):
        raise OSError("diagnostics unavailable")

    monkeypatch.setattr(field_advice.Advice, "_write", fail)
    with pytest.warns(RuntimeWarning, match="advice"):
        result = field.solve(broken, config)
    assert result["status"] == "FIELD_READY" and (broken / "FIELD.json").is_file()


def test_unsupported_core_is_rejected_before_field_creation(tmp_path, monkeypatch):
    import block_solver
    from test_field import prepared_run

    root = prepared_run(tmp_path / "run", shape=(3, 3, 9), chunk=4)
    monkeypatch.setattr(block_solver, "solve_blocks", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="core.*observer"):
        field.solve(root, dict(field_advice=True))
    assert not (root / "field").exists()


def test_read_only_report_never_invokes_solver_or_verifier(tmp_path, monkeypatch, capsys):
    from test_field import prepared_run
    from winding_field.cli import main

    root = prepared_run(tmp_path / "run", shape=(3, 3, 9), chunk=4)
    field.solve(root, dict(sweeps=3, field_chunk=4, rtol=1e-10, reserve_bytes=0, field_advice=True))
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("report attempted scientific work")

    monkeypatch.setattr(field, "solve_blocks", forbidden)
    monkeypatch.setattr(global_residual, "verify", forbidden)
    assert main(["field-advice", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["actions"] == [] and result["physical_usefulness"] == "precisión útil no calibrada"
    )
    assert result["current_closure_recorded"]
    assert result["equivalent_global_verifier_available"] is False
    assert result["estimated"]["remaining_error"] is None
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_regional_statistics_share_the_existing_global_pass():
    shape = (3, 3, 5)
    phi = np.zeros(shape)
    calls = []

    def read(lo, hi):
        return phi[tuple(slice(a, b) for a, b in zip(lo, hi))]

    def observations(lo, hi):
        calls.append(1)
        local = tuple(hi - lo)
        return np.ones((*local, 3), "f4"), np.ones(local, "f4"), np.ones(local, bool), {}

    plain = global_residual.verify(shape, 3, [1.0, 2.0, 3.0], read, observations)
    count = len(calls)
    calls.clear()
    regions = []
    annotated = global_residual.verify(
        shape, 3, [1.0, 2.0, 3.0], read, observations, regions=regions
    )
    assert annotated == plain and len(calls) == count == 2
    assert sum(r["free_nodes"] for r in regions) == 44
    assert np.hypot(*(r["residual_l2"] for r in regions)) == plain["residual_l2"]


def test_prepared_opt_in_preserves_science_and_explicit_verification(tmp_path):
    from test_field_workflow import prepared, workflow, stored

    path, config = prepared(tmp_path)
    workflow().run_field(path, tmp_path / "off")
    config["execution"]["field_advice"] = True
    path.write_text(json.dumps(config))
    workflow().run_field(path, tmp_path / "on")
    a, va, oa = stored(tmp_path / "off")
    b, vb, ob = stored(tmp_path / "on")
    assert a["identity"] == b["identity"]
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_array_equal(oa, ob)
    sink = field_advice.Advice(tmp_path / "on")
    assert len(sink.records()) == 2
    assert not list(sink.path.glob("verification-*.json"))
    result = workflow().verify_field(path, tmp_path / "on")
    assert result["global_converged"]
    assert len(list(sink.path.glob("verification-*.json"))) == 1


def test_run_plan_flag_config_parity_and_default_absence(tmp_path, monkeypatch, capsys):
    from winding_field import cli, workflow

    monkeypatch.setattr(workflow, "plan", lambda config: config)
    common = [
        "plan",
        "--source",
        str(tmp_path / "ct"),
        "--output",
        str(tmp_path / "run"),
        "--device",
        "cpu",
    ]
    assert cli.main(common) == 0
    assert "field_advice" not in json.loads(capsys.readouterr().out)
    assert cli.main([*common, "--field-advice"]) == 0
    explicit = json.loads(capsys.readouterr().out)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(dict(field_advice=True)))
    assert cli.main([*common, "--cli-config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == explicit
    assert explicit["field_advice"] is True
    from winding_field.config import validate_config

    with pytest.raises(ValueError):
        validate_config(dict(field_advice="true"))


@pytest.mark.parametrize("unavailable", [False, True])
def test_optional_regions_are_bounded_and_allocation_failure_keeps_global_result(unavailable):
    shape = (2, 2, 130)
    phi = np.zeros(shape)

    def read(lo, hi):
        return phi[tuple(slice(a, b) for a, b in zip(lo, hi))]

    def observations(lo, hi):
        local = tuple(hi - lo)
        return np.zeros((*local, 3), "f4"), np.ones(local, "f4"), np.ones(local, bool), {}

    plain = global_residual.verify(shape, 2, [1.0, 2.0, 3.0], read, observations)

    class Unavailable(list):
        def append(self, item):
            raise MemoryError("optional region allocation")

    regions = Unavailable() if unavailable else []
    assert (
        global_residual.verify(shape, 2, [1.0, 2.0, 3.0], read, observations, regions=regions)
        == plain
    )
    assert len(regions) == (0 if unavailable else 64)


def test_automatic_resume_delivers_committed_closure_before_terminal_stop(tmp_path, monkeypatch):
    from test_field_convergence import workflow_source
    from winding_field.field_convergence import converge

    source = workflow_source(tmp_path / "source")
    root = tmp_path / "automatic"
    complete = field_advice.Advice.complete
    verify = global_residual.verify
    checks = []

    class Crash(BaseException):
        pass

    def crash(observer, snapshot):
        raise Crash("after scientific commit, before closure delivery")

    def counted(*args, **kwargs):
        checks.append(1)
        return verify(*args, **kwargs)

    monkeypatch.setattr(global_residual, "verify", counted)
    monkeypatch.setattr(field_advice.Advice, "complete", crash)
    with pytest.raises(Crash):
        converge(source, root, max_sweeps=1, global_rtol=1e-12, config=dict(field_advice=True))
    checkpoint = (root / "field/CHECKPOINT.json").read_bytes()
    assert len(json.loads(checkpoint)["progress"]["completed_sweeps"]) == 1
    assert len(checks) == 1
    monkeypatch.setattr(field_advice.Advice, "complete", complete)
    result = converge(root, resume=True)
    assert result["reason"] == "SWEEP_LIMIT" and len(checks) == 2
    assert (root / "field/CHECKPOINT.json").read_bytes() == checkpoint
    sink = field_advice.Advice(root / "field")
    assert [r["sweep"] for r in sink.records()] == [1]
    assert sink.records()[0]["metrics"]["observed_count"] == 81
    assert converge(root, resume=True)["reason"] == "SWEEP_LIMIT"
    assert len(checks) == 2 and len(sink.records()) == 1
