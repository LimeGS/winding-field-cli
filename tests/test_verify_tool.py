"""The verifier and its rules stay usable: tools/verify.py, REVIEW.md, approvals."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify  # noqa: E402

# Public options whose silent loss already happened once (3556a4b) or would break runs.
CRITICAL = [
    "run --nonconvergence-policy=continue",
    "run --reuse-from",
    "run --reuse-from-sha256",
    "run --flattening-method=slim",
    "run --flattening-method=default",
    "run --surface-output=native",
    "export-sheets --scope=sheet",
    "export-sheets --flattening-solver=slim",
    "resume",
    "prepared <action>=build",
    "prepared mesh --output",
]


def test_flag_inventory_sees_every_command_and_critical_option(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools/verify.py"), "--probe", "flags", "--work", tmp_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    inventory = set(json.loads(done.stdout.splitlines()[-1]))
    assert len(inventory) > 500
    assert not [entry for entry in CRITICAL if entry not in inventory]


def test_review_items_parse_with_valid_triggers():
    items = verify.review_items(ROOT / "REVIEW.md")
    ids = [item["id"] for item in items]
    assert ids == [f"R{i}" for i in range(1, len(ids) + 1)] and len(ids) >= 10
    for item in items:
        assert item["check"] and item["origin"]
        pattern = item["pattern"].strip(" `")
        if pattern not in ("—", ""):
            re.compile(pattern)
        for glob in (p.strip(" `") for p in item["paths"].split(",")):
            # The container suite mounts tests, tools and docs but installs src/ as a package.
            if (ROOT / "src").is_dir() or (ROOT / glob.split("/")[0]).exists():
                assert any(ROOT.glob(glob)) or (ROOT / glob).exists(), (item["id"], glob)


def test_review_step_matches_paths_and_changed_lines_only(tmp_path):
    (tmp_path / "REVIEW.md").write_text(
        "| R1 | `src/a.py` | — | path check | x |\n"
        "| R2 | `src/*.py` | `^-.*add_argument` | flag check | y |\n"
        "| R3 | `src/*.py` | `identity` | identity check | z |\n"
        "| R4 | `tests/*.py` | `add_argument` | other file check | w |\n"
    )
    diff = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n"
        "-    p.add_argument('--gone')\n identity in context only\n"
        "diff --git a/docs/x.md b/docs/x.md\n+add_argument mentioned in prose\n"
    )
    step = verify.Step("review")
    verify.step_review(step, dict(base_tree=tmp_path, lines=verify.changed_lines(diff)))
    assert step.evidence["checklist"] == ["R1: path check", "R2: flag check"]


def test_approvals_are_explicit_and_reasoned():
    approved = json.loads((ROOT / "tools/verify-approved.json").read_text())
    assert set(approved) >= {"flags", "golden"}
    for section in ("flags", "golden"):
        assert all(isinstance(r, str) and len(r) > 10 for r in approved[section].values())


@pytest.mark.parametrize("name", sorted(verify.RUNNERS))
def test_every_step_is_documented_in_the_skill(name):
    skill = (ROOT / ".claude/skills/verify/SKILL.md").read_text()
    assert f"`{name}`" in skill
