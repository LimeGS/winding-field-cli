"""A field retry consumes verified upstream data without re-running its producer."""

import fcntl
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from test_lasagna_input import source_case
from test_splines import upstream_curves
from winding_field import workflow
from winding_field.util import atomic_json, file_sha


@pytest.fixture
def donor(tmp_path, monkeypatch):
    volume, geometry, config = source_case(tmp_path, shape=(9, 10, 11), whole=True)
    volume.close()
    afv = tmp_path / "external.afv"
    upstream_curves(afv, (9, 10, 11), [(1, np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]]))])
    root = tmp_path / "donor"
    config.update(
        output=str(root),
        fibres=str(afv),
        fibres_geometry=config["lasagna_geometry"],
        maxiter=2,
        sweeps=1,
        field_chunk=16,
        threads=1,
        previews=False,
    )
    monkeypatch.setattr(workflow, "implementation_sha", lambda: "old-producer")
    state = workflow.run(config, until="fibres")
    state.update(status="FAILED", error="old field failure")
    atomic_json(root / "RUN.json", state)
    (root / "field").mkdir()
    (root / "field/old-checkpoint").write_text("must not copy")
    monkeypatch.setattr(workflow, "implementation_sha", lambda: "new-consumer")
    return root, config


def request(donor, tmp_path):
    root, config = donor
    return dict(
        config,
        output=str(tmp_path / "retry"),
        reuse_from=str(root),
        reuse_from_sha256=file_sha(root / "RUN.json"),
    )


def snapshot(root):
    return {str(p.relative_to(root)): file_sha(p) for p in root.rglob("*") if p.is_file()}


def forbidden(*args, **kwargs):
    pytest.fail("reuse must not invoke an upstream producer/export")


def test_field_retry_reuses_complete_stages(donor, tmp_path, monkeypatch):
    from winding_field import predict, fibres, exports

    root, _ = donor
    before = snapshot(root)
    monkeypatch.setattr(predict, "prepare", forbidden)
    monkeypatch.setattr(fibres, "trace", forbidden)
    monkeypatch.setattr(exports, "export_lasagna", forbidden)
    monkeypatch.setattr(exports, "export_afv", forbidden)
    config = request(donor, tmp_path)
    config.update(maxiter=2000, nonconvergence_policy="continue")
    result = workflow.run(config, until="field")
    out = Path(config["output"])
    assert set(result["stages"]) == {"prepare", "fibres", "field"}
    assert result["implementation_sha256"] == "new-consumer"
    assert result["reuse"]["implementation_sha256"] == "old-producer"
    assert result["reuse"]["run_sha256"] == config["reuse_from_sha256"]
    for name in ("prepare", "fibres"):
        assert result["stages"][name]["mode"] == "reused"
        assert result["stages"][name]["seconds"] == 0
    for name in ("PREPARED.json", "FIBRES.json", "fibres/fibres.sqlite"):
        assert file_sha(out / name) == before[name]
        assert (out / name).stat().st_ino != (root / name).stat().st_ino
    assert not (out / "field/old-checkpoint").exists()
    assert (out / "FIELD.json").exists()
    assert snapshot(root) == before
    assert workflow.run(result["config"], resume=True, until="field")["stages"] == result["stages"]


def test_offline_retry_reuses_an_online_donor(donor, tmp_path, monkeypatch):
    # --offline only selects how CT bytes are reached: the reuse still re-checks every chunk
    # in the copied ledger and must not refuse it as a changed preparation setting.
    from winding_field import predict, fibres, exports

    root, donor_config = donor
    assert not donor_config.get("offline")
    monkeypatch.setattr(predict, "prepare", forbidden)
    monkeypatch.setattr(fibres, "trace", forbidden)
    monkeypatch.setattr(exports, "export_lasagna", forbidden)
    monkeypatch.setattr(exports, "export_afv", forbidden)
    config = request(donor, tmp_path)
    config.update(offline=True, maxiter=2000, nonconvergence_policy="continue")
    result = workflow.run(config, until="field")
    assert result["config"]["offline"] is True
    for name in ("prepare", "fibres"):
        assert result["stages"][name]["mode"] == "reused"
    assert (Path(config["output"]) / "FIELD.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing_prepare",
        "missing_fibres",
        "roi",
        "pitch",
        "input",
        "chunk",
        "tile",
        "payload",
        "receipt",
        "manifest",
        "ct_chunk",
        "ledger",
        "export",
        "export_escape",
        "symlink",
        "journal",
        "schema",
        "active",
        "digest",
        "bounded",
        "ct_truncated",
        "ct_missing_row",
        "afv_missing",
        "afv_unbound",
        "surface_missing",
        "chained",
    ],
)
def test_reuse_rejects_incomplete_incompatible_or_corrupt_source(
    donor, tmp_path, monkeypatch, change
):
    from winding_field import predict, field

    root, _ = donor
    config = request(donor, tmp_path)
    state = json.loads((root / "RUN.json").read_bytes())
    if change.startswith("missing_"):
        state["stages"].pop(change.removeprefix("missing_"))
    elif change == "roi":
        config.update(roi_lo=[0, 0, 0], roi_hi=[8, 10, 11])
    elif change == "pitch":
        config["field_spacing_um"] = 20.0
    elif change == "input":
        config["fibres_geometry"] = None
    elif change == "chunk":
        config["chunk"] = 8
    elif change == "bounded":
        state["config"]["bounded_preparation"] = True
    elif change == "chained":
        state["reuse"] = dict(
            implementation_sha256="original-producer", stages=state["stages"].copy()
        )
        state["stages"] = {k: dict(v, mode="reused", seconds=0) for k, v in state["stages"].items()}
    elif change == "ct_truncated":
        (root / "source.sqlite").write_bytes(b"")
    elif change == "ct_missing_row":
        with sqlite3.connect(root / "source.sqlite") as db:
            db.execute("DELETE FROM chunks WHERE key=(SELECT key FROM chunks LIMIT 1)")
    elif change in ("afv_missing", "afv_unbound", "surface_missing"):
        name = (
            "fibres/fibres.afv"
            if change.startswith("afv_")
            else "lasagna/surface_nx.ome.zarr/0/0.0.0"
        )
        with sqlite3.connect(root / "products.sqlite") as db:
            if change == "afv_unbound":
                (root / name).write_bytes(b"corrupt-but-ledger-updated")
                db.execute("UPDATE exports SET sha=? WHERE path=?", (file_sha(root / name), name))
            else:
                db.execute("DELETE FROM exports WHERE path=?", (name,))
                (root / name).unlink()
    elif change == "schema":
        state["schema"] = "unknown-v99"
    elif change == "active":
        state["status"] = "RUNNING"
    elif change == "tile":
        next((root / "maps").glob("*.npz")).write_bytes(b"corrupt")
    elif change == "payload":
        (root / "fibres/fibres.sqlite").write_bytes(b"corrupt")
    elif change == "receipt":
        (root / "FIBRES.json").write_text("{}")
    elif change == "manifest":
        Path(config["lasagna_input"]).write_text("{}")
    elif change == "ct_chunk":
        (Path(config["source"]) / "0/0.0.0").write_bytes(bytes(64))
    elif change == "ledger":
        (root / "lasagna-source/WORK.json").unlink()
    elif change == "export":
        (root / "fibres/fibres.afv").write_bytes(b"corrupt")
    elif change == "export_escape":
        with sqlite3.connect(root / "products.sqlite") as db:
            db.execute(
                "INSERT INTO exports VALUES ('../external.afv','fibres',?)",
                (file_sha(Path(config["fibres"])),),
            )
    elif change == "symlink":
        tile = next((root / "maps").glob("*.npz"))
        other = tmp_path / "escaped.npz"
        tile.rename(other)
        tile.symlink_to(other)
    elif change == "journal":
        (root / "fibres/fibres.sqlite-wal").write_bytes(b"live")
    atomic_json(root / "RUN.json", state)
    if change != "digest":
        config["reuse_from_sha256"] = file_sha(root / "RUN.json")
    else:
        config["reuse_from_sha256"] = "0" * 64
    monkeypatch.setattr(predict, "prepare", forbidden)
    monkeypatch.setattr(field, "solve", forbidden)
    before = snapshot(root)
    with pytest.raises((ValueError, OSError, KeyError, sqlite3.Error)):
        workflow.run(config, until="field")
    assert snapshot(root) == before
    out = Path(config["output"])
    assert not (out / "FIELD.json").exists()
    if (out / "RUN.json").exists():
        failed = json.loads((out / "RUN.json").read_bytes())
        assert not failed["stages"]
        with pytest.raises(ValueError, match="reuse"):
            workflow.run(failed["config"], resume=True, until="field")


def test_reuse_refuses_locked_donor(donor, tmp_path):
    root, _ = donor
    with (root / ".run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            workflow.run(request(donor, tmp_path), until="field")


def test_reuse_refuses_overlapping_output(donor, tmp_path):
    config = request(donor, tmp_path)
    config["output"] = str(donor[0] / "nested")
    with pytest.raises(ValueError, match="overlap"):
        workflow.run(config, until="field")


def test_stopped_copy_cannot_fall_back_to_import_or_resume(donor, tmp_path, monkeypatch):
    from winding_field import predict
    import shutil

    config = request(donor, tmp_path)
    original = shutil.copy2

    def stop_after_copy(src, dst, *a, **kw):
        result = original(src, dst, *a, **kw)
        (Path(config["output"]) / "STOP").touch()
        return result

    monkeypatch.setattr(shutil, "copy2", stop_after_copy)
    monkeypatch.setattr(predict, "prepare", forbidden)
    result = workflow.run(config, until="field")
    assert result["status"] == "PAUSED" and not result["stages"]
    with pytest.raises(ValueError, match="reuse"):
        workflow.run(result["config"], resume=True, until="field")


def test_cli_forwards_reuse_identity_and_field_goal(tmp_path, monkeypatch):
    from winding_field.cli import main

    captured = {}

    def execute(config, **kw):
        captured.update(config=config, options=kw)
        return dict(status="PAUSED")

    monkeypatch.setattr(workflow, "run", execute)
    assert (
        main(
            [
                "run",
                "--source",
                str(tmp_path / "ct"),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
                "--until",
                "field",
                "--reuse-from",
                str(tmp_path / "donor"),
                "--reuse-from-sha256",
                "a" * 64,
            ]
        )
        == 0
    )
    assert captured["config"]["reuse_from"] == str(tmp_path / "donor")
    assert captured["config"]["reuse_from_sha256"] == "a" * 64
    assert captured["options"] == {"until": "field"}


@pytest.mark.parametrize(
    "supplied",
    [
        {"reuse_from_sha256": "a" * 64},
        {"reuse_from": "missing"},
        {"reuse_from": "missing", "reuse_from_sha256": "bad"},
    ],
)
def test_reuse_requires_path_and_digest_pair(tmp_path, supplied):
    with pytest.raises(ValueError, match="reuse"):
        workflow.plan(dict(source="missing", output=str(tmp_path / "new"), **supplied))


def test_field_retry_reuses_streamed_lasagna_import(tmp_path, monkeypatch):
    # A manifest without a physical frame streams into lasagna-import (scroll tiles do this);
    # its provenance must be in the prepare exports or every --reuse-from is refused.
    from test_lasagna_stream import fixture
    from winding_field import predict, fibres, exports

    ct, _, config = fixture(tmp_path)
    afv = tmp_path / "external.afv"
    upstream_curves(afv, (9, 10, 11), [(1, np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]]))])
    descriptor = tmp_path / "geometry.json"
    atomic_json(
        descriptor,
        dict(source=str(ct), origin_xyz_um=[100.0, 200.0, 300.0], spacing_xyz_um=[5.0, 10.0, 20.0]),
    )
    root = tmp_path / "donor"
    config.update(
        source=str(ct),
        output=str(root),
        fibres=str(afv),
        fibres_geometry=str(descriptor),
        maxiter=2,
        sweeps=1,
        field_chunk=16,
        previews=False,
    )
    monkeypatch.setattr(workflow, "implementation_sha", lambda: "old-producer")
    state = workflow.run(config, until="fibres")
    assert (root / "lasagna-import/IDENTITY.json").is_file()
    with sqlite3.connect(root / "products.sqlite") as db:
        recorded = {path for (path,) in db.execute("SELECT path FROM exports")}
    assert "lasagna-import/IDENTITY.json" in recorded
    documents = {
        str(p.relative_to(root)) for p in (root / "lasagna-import").glob("document-*.json")
    }
    assert documents and documents <= recorded
    state.update(status="FAILED", error="old field failure")
    atomic_json(root / "RUN.json", state)
    monkeypatch.setattr(workflow, "implementation_sha", lambda: "new-consumer")
    before = snapshot(root)
    monkeypatch.setattr(predict, "prepare", forbidden)
    monkeypatch.setattr(fibres, "trace", forbidden)
    monkeypatch.setattr(exports, "export_lasagna", forbidden)
    monkeypatch.setattr(exports, "export_afv", forbidden)
    retry = dict(
        config,
        output=str(tmp_path / "retry"),
        reuse_from=str(root),
        reuse_from_sha256=file_sha(root / "RUN.json"),
        maxiter=2000,
        nonconvergence_policy="continue",
    )
    result = workflow.run(retry, until="field")
    out = Path(retry["output"])
    for name in ("prepare", "fibres"):
        assert result["stages"][name]["mode"] == "reused"
    assert file_sha(out / "lasagna-import/IDENTITY.json") == before["lasagna-import/IDENTITY.json"]
    assert (out / "FIELD.json").exists()
    assert snapshot(root) == before
    assert workflow.run(result["config"], resume=True, until="field")["stages"] == result["stages"]
