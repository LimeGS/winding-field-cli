"""Ordinary fields continue from their checkpoint to a larger sweep budget."""

import json

import numpy as np
import pytest

from test_field import prepared_run


def _values(run):
    from winding_field import field

    geometry = json.loads((run / "FIELD.json").read_bytes())["geometry"]
    with field.open_field(run) as source:
        return source.read([0, 0, 0], geometry["shape_zyx"])


def test_extending_a_finished_field_equals_solving_the_larger_budget_directly(tmp_path):
    from winding_field import field

    config = dict(sweeps=2, field_chunk=8)
    direct = prepared_run(tmp_path / "direct", shape=(6, 6, 32))
    assert field.solve(direct, dict(config, sweeps=4))["status"] == "FIELD_READY"

    extended = prepared_run(tmp_path / "extended", shape=(6, 6, 32))
    assert field.solve(extended, config)["status"] == "FIELD_READY"
    first = json.loads((extended / "field/CHECKPOINT.json").read_bytes())
    assert len(first["progress"]["completed_sweeps"]) == 2

    receipt = field.solve(extended, config, sweep_target=4)
    assert receipt["status"] == "FIELD_READY"
    manifest = json.loads((extended / "field/CHECKPOINT.json").read_bytes())
    assert len(manifest["progress"]["completed_sweeps"]) == 4
    assert [(e["from"], e["to"]) for e in manifest["sweep_extensions"]] == [(2, 4)]
    a, va = _values(direct)
    b, vb = _values(extended)
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_array_equal(a, b)

    # The extension is part of the checkpoint: later resumes keep the larger budget,
    # a smaller target is refused, and an equal target adds nothing.
    with pytest.raises(ValueError, match="sweep target"):
        field.solve(extended, config, sweep_target=3)
    again = field.solve(extended, config, sweep_target=4)
    assert again["status"] == "FIELD_READY"
    assert (
        len(
            json.loads((extended / "field/CHECKPOINT.json").read_bytes())["progress"][
                "completed_sweeps"
            ]
        )
        == 4
    )


def _workflow_case(tmp_path, name, sweeps):
    from test_lasagna_input import source_case
    from test_splines import upstream_curves

    volume, _, config = source_case(tmp_path / name, shape=(9, 10, 11), whole=True)
    volume.close()
    afv = tmp_path / name / "external.afv"
    upstream_curves(afv, (9, 10, 11), [(1, np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]]))])
    config.update(
        output=str(tmp_path / name / "run"),
        fibres=str(afv),
        fibres_geometry=config["lasagna_geometry"],
        sweeps=sweeps,
        field_chunk=4,
        threads=1,
        previews=False,
        nonconvergence_policy="continue",
    )
    return config


def test_resume_extends_a_finished_ordinary_run_and_refuses_after_later_stages(tmp_path):
    from pathlib import Path

    from winding_field import workflow

    direct = workflow.run(_workflow_case(tmp_path, "direct", 3), until="field")
    config = _workflow_case(tmp_path, "extended", 1)
    first = workflow.run(config, until="field")
    assert "field" in first["stages"]
    run = Path(config["output"])

    state = workflow.run(first["config"], resume=True, until="field", field_sweeps=3)
    assert "field" in state["stages"]
    assert state["field_extensions"][0]["from_sweeps"] == 1
    assert state["field_extensions"][0]["to_sweeps"] == 3
    assert (run / "FIELD.sweeps-1.json").is_file() and (run / "FIELD.json").is_file()
    a, va = _values(Path(direct["config"]["output"]))
    b, vb = _values(run)
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_array_equal(a, b)

    # A field that later stages were built from is never reopened.
    state["stages"]["mesh"] = dict(receipt="MESHES.json")
    (run / "RUN.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="later stages"):
        workflow.run(state["config"], resume=True, until="field", field_sweeps=5)
