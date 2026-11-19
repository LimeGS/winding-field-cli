"""Keep the commented run/plan YAML examples aligned with the public parser."""

import argparse
from pathlib import Path
import re

import pytest
import yaml

from winding_field import cli, cli_settings

TEMPLATE = Path(__file__).resolve().parents[1] / "examples/run-plan.template.yaml"


def run_parser(monkeypatch, command):
    captured = []

    class Captured(Exception):
        pass

    def capture(parser, *args, **kwargs):
        captured.append(parser)
        raise Captured

    with monkeypatch.context() as patch:
        patch.setattr(cli_settings, "parse_options", capture)
        with pytest.raises(Captured):
            cli.main([command])
    commands = next(a for a in captured[0]._actions if isinstance(a, argparse._SubParsersAction))
    return commands.choices[command]


@pytest.mark.parametrize("command", ["run", "plan"])
def test_template_documents_every_option_with_a_typed_example(monkeypatch, command):
    parser = run_parser(monkeypatch, command)
    actions = {}
    for action in parser._actions:
        if action.dest != "help":
            actions.setdefault(action.dest, action)
    entries = {}
    for line in TEMPLATE.read_text().splitlines():
        match = re.match(r"^(?:# )?([a-z][a-z0-9_]*):", line)
        if not match:
            continue
        name = match[1]
        assert name not in entries, f"duplicate template option: {name}"
        assert "# Opciones:" in line, f"missing alternatives/type comment: {name}"
        example = line.removeprefix("# ")
        entries[name] = (yaml.safe_load(example)[name], line.split("# Opciones:", 1)[1])
    assert entries.keys() == actions.keys()
    for name, (value, comment) in entries.items():
        action = actions[name]
        cli_settings._value(action, value)
        if action.choices:
            assert all(str(choice) in comment for choice in action.choices), name
        if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
            assert "true" in comment and "false" in comment, name


def test_active_template_can_plan_tiny_ct_without_creating_output(tmp_path, capsys):
    import json
    import numpy as np
    from ct_fixture import write_ct

    source = write_ct(tmp_path / "ct", np.full((8, 8, 8), 100, "u1"), spacing=(9.6,) * 3)
    output = tmp_path / "not-created"
    assert (
        cli.main(
            [
                "plan",
                "--cli-config",
                str(TEMPLATE),
                "--source",
                str(source),
                "--output",
                str(output),
                "--prediction",
                "classical",
                "--device",
                "cpu",
                "--until",
                "prepare",
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["prediction"] == "classical" and result["shape_zyx"] == [8, 8, 8]
    assert not output.exists()
