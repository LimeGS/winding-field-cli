"""prepared mesh --then-flatten: flattening chained as a separate second step."""

import json
import sys

import pytest

from winding_field import mesh_chain


def _stub(monkeypatch, code):
    seen = {}

    def command(mesh_output, flatten_output, cli_config=None):
        seen["args"] = (mesh_output, flatten_output, cli_config)
        return [sys.executable, "-c", code]

    monkeypatch.setattr(mesh_chain, "flatten_command", command)
    return seen


def test_ready_mesh_starts_the_flatten_step_and_records_both(tmp_path, monkeypatch):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    seen = _stub(monkeypatch, "print('log'); print('{\"status\": \"EXPORTED\"}')")
    record = mesh_chain.chain_flatten(mesh, {"status": "MESH_READY"}, tmp_path / "flat")
    assert [s["step"] for s in record["steps"]] == ["mesh", "flatten"]
    assert record["steps"][0]["status"] == "MESH_READY"
    flatten = record["steps"][1]
    assert flatten["status"] == "EXPORTED" and flatten["returncode"] == 0
    assert seen["args"][1] == (tmp_path / "flat").absolute()
    assert json.loads((mesh / "CHAIN.json").read_text()) == record


def test_unready_mesh_never_starts_the_flatten_step(tmp_path, monkeypatch):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    seen = _stub(monkeypatch, "raise SystemExit(9)")
    record = mesh_chain.chain_flatten(mesh, {"status": "CHECKPOINTED_STOP"}, tmp_path / "flat")
    assert record["steps"][1]["status"] == "NOT_STARTED"
    assert "args" not in seen
    assert json.loads((mesh / "CHAIN.json").read_text()) == record


def test_failed_flatten_keeps_its_exit_code_and_stderr(tmp_path, monkeypatch):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    _stub(monkeypatch, "import sys; sys.stderr.write('boom'); raise SystemExit(2)")
    record = mesh_chain.chain_flatten(mesh, {"status": "MESH_READY"}, tmp_path / "flat")
    flatten = record["steps"][1]
    assert flatten["status"] == "FAILED" and flatten["returncode"] == 2
    assert flatten["stderr_tail"] == "boom"


def test_flatten_command_is_export_sheets_and_resumes_an_existing_output(tmp_path):
    config = tmp_path / "flatten.yaml"
    config.write_text("source: ct.zarr\n")
    fresh = mesh_chain.flatten_command(tmp_path / "mesh", tmp_path / "flat", config)
    assert fresh[1:4] == ["-m", "winding_field", "export-sheets"]
    assert fresh[4:6] == ["--cli-config", str(config.absolute())]
    assert fresh[6:] == [
        "--mesh-bundle",
        str(tmp_path / "mesh"),
        "--output",
        str(tmp_path / "flat"),
    ]
    (tmp_path / "flat").mkdir()
    assert mesh_chain.flatten_command(tmp_path / "mesh", tmp_path / "flat")[-1] == "--resume"


def test_chain_refuses_an_output_inside_the_mesh_or_a_missing_config(tmp_path):
    with pytest.raises(ValueError, match="outside the mesh output"):
        mesh_chain.check_chain(tmp_path / "mesh", tmp_path / "mesh" / "flat")
    with pytest.raises(ValueError, match="not a file"):
        mesh_chain.check_chain(tmp_path / "mesh", tmp_path / "flat", tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--flatten-cli-config", "x.yaml"], "requires --then-flatten"),
        (["--then-flatten", "flat", "--parallel-levels"], "without --parallel-levels"),
    ],
)
def test_prepared_mesh_rejects_invalid_chains_before_extraction(
    tmp_path, capsys, monkeypatch, extra, message
):
    from winding_field.surface_workflow import main

    monkeypatch.chdir(tmp_path)
    assert main(["mesh", "--output", str(tmp_path / "mesh"), *extra]) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "mesh").exists()


def test_render_action_rejects_then_flatten(tmp_path, capsys, monkeypatch):
    from winding_field.surface_workflow import main

    monkeypatch.chdir(tmp_path)
    assert main(["render", "--output", str(tmp_path / "r"), "--then-flatten", "flat"]) == 2
    assert "requires the mesh action" in capsys.readouterr().err


def test_prepared_mesh_then_flatten_runs_a_real_uv_export_as_its_own_step(tmp_path, monkeypatch):
    from test_surface_workflow import prepared
    from winding_field.cli import main

    monkeypatch.chdir(tmp_path)
    path, _ = prepared(tmp_path)
    settings = tmp_path / "flatten.yaml"
    settings.write_text("flattening_solver: lscm\nscope: sheet\nstages: uv\n")
    mesh, flat = tmp_path / "mesh", tmp_path / "flat"
    argv = ["prepared", "mesh", "--config", str(path), "--output", str(mesh)]
    code = main([*argv, "--then-flatten", str(flat), "--flatten-cli-config", str(settings)])
    chain = json.loads((mesh / "CHAIN.json").read_text())
    assert chain["steps"][0]["status"] == "MESH_READY"
    if sys.platform != "linux":
        # Matrix export needs the Linux /proc RSS guard; the mesh step still publishes.
        assert code == 1 and chain["steps"][1]["status"] == "FAILED"
        assert "requires Linux /proc" in chain["steps"][1]["stderr_tail"]
        return
    assert code == 0
    assert chain["steps"][1]["returncode"] == 0
    # Each step keeps its own receipts; the frame comes from the mesh publication.
    assert json.loads((mesh / "MESH.json").read_text())["status"] == "MESH_READY"
    assert (flat / "EXPORT.json").is_file() and (flat / "UV-MATRIX.json").is_file()
    assert "--output-frame-id" in chain["steps"][1]["command"]
