"""CLI processes default OpenMP to passive waiting; an explicit setting wins."""

import os
import subprocess
import sys

import pytest

PROBE = (
    "import sys, os; import winding_field.cli; "
    "assert 'torch' not in sys.modules, 'torch loaded before the policy'; "
    "import torch; print(os.environ['OMP_WAIT_POLICY'])"
)


@pytest.mark.parametrize("given,expected", [(None, "PASSIVE"), ("ACTIVE", "ACTIVE")])
def test_cli_sets_passive_wait_before_torch_unless_user_chose(given, expected):
    env = {k: v for k, v in os.environ.items() if k != "OMP_WAIT_POLICY"}
    if given:
        env["OMP_WAIT_POLICY"] = given
    done = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == expected
