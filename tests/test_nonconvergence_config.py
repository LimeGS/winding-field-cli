import argparse

import pytest

from winding_field.cli import main
from winding_field.config import validate_config


def test_run_policy_is_explicit_and_validated(monkeypatch):
    class Captured(Exception):
        pass

    parsers = []

    def capture(parser, *args, **kwargs):
        parsers.append(parser)
        raise Captured

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(Captured):
            main(["run"])
    p = parsers[0]
    base = ["run", "--output", "unused"]
    assert p.parse_args(base).nonconvergence_policy == "fail"
    assert (
        p.parse_args(base + ["--nonconvergence-policy", "continue"]).nonconvergence_policy
        == "continue"
    )
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--nonconvergence-policy", "other"])
    with pytest.raises(ValueError, match="nonconvergence_policy"):
        validate_config({"nonconvergence_policy": "other"})
