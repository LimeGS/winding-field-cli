"""Newer code continues a run only for stages it computes identically."""

import json
from pathlib import Path

import pytest

from test_field_extension import _workflow_case


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from winding_field import workflow

    monkeypatch.setattr(workflow, "implementation_sha", lambda: "v1")
    state = workflow.run(_workflow_case(tmp_path, "case", 1), until="fibres")
    assert set(state["stages"]) == {"prepare", "fibres"} and "compat" in state
    monkeypatch.setattr(workflow, "implementation_sha", lambda: "v2")
    return state


def test_newer_code_continues_pending_stages_with_unchanged_epochs(prepared):
    from winding_field import workflow

    state = workflow.run(prepared["config"], resume=True, until="field")
    assert "field" in state["stages"]
    assert state["implementation_sha256"] == "v1"  # the creating code stays recorded
    (entry,) = state["continued_with"]
    assert entry["implementation_sha256"] == "v2" and entry["stages"][0] == "field"


def test_an_epoch_bump_of_a_pending_stage_refuses(prepared, monkeypatch):
    from winding_field import compat, workflow

    monkeypatch.setitem(compat.STAGE_EPOCHS, "field", 2)
    with pytest.raises(ValueError, match="stage field epoch 1 -> 2"):
        workflow.run(prepared["config"], resume=True, until="field")


def test_an_epoch_bump_of_a_completed_stage_does_not_refuse(prepared, monkeypatch):
    from winding_field import compat, workflow

    monkeypatch.setitem(compat.STAGE_EPOCHS, "prepare", 2)
    assert "field" in workflow.run(prepared["config"], resume=True, until="field")["stages"]


def test_a_new_core_major_version_refuses(prepared, monkeypatch):
    from winding_field import compat, workflow

    monkeypatch.setattr(compat, "core_major", lambda: 99)
    with pytest.raises(ValueError, match="core major version"):
        workflow.run(prepared["config"], resume=True, until="field")


def test_core_1_runs_continue_on_core_2(monkeypatch):
    from winding_field import compat

    monkeypatch.setattr(compat, "core_major", lambda: 2)
    saved = dict(compat.record(), core_major=1)
    assert compat.incompatibilities(saved, ["field"]) == []
    assert compat.incompatibilities(dict(saved, core_major=3), ["field"])


def test_runs_without_a_record_keep_the_exact_implementation_rule(prepared):
    from winding_field import workflow

    run = Path(prepared["config"]["output"])
    state = json.loads((run / "RUN.json").read_bytes())
    del state["compat"]
    (run / "RUN.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="implementation changed"):
        workflow.run(prepared["config"], resume=True, until="field")
