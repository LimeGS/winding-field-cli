"""Source archives must carry the experimental tools required by their tests."""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


def test_sdist_contains_flattening_tools_with_their_tests(tmp_path):
    pytest.importorskip("build")
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "THIRD_PARTY.md", "uv.toml"):
        shutil.copyfile(root / name, project / name)
    for name in ("src", "tools", "tests", "provenance", "licenses", "docs", "examples", "docker"):
        shutil.copytree(
            root / name,
            project / name,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.egg-info", ".pytest_cache", "superpowers", "verification"
            ),
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(tmp_path / "dist"),
        ],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    for tool in (
        "bench_flattening",
        "flattening_ct",
        "flattening_metrics",
        "flattening_review",
        "bench_local_imports",
        "bench_afv_import",
    ):
        assert f"tools/{tool}.py" in names
    for test in ("benchmark", "ct", "metrics", "review"):
        assert f"tests/test_flattening_{test}.py" in names
    for module in ("cli_settings", "export_settings", "input_settings", "tool_settings"):
        assert f"src/winding_field/{module}.py" in names
    for name in (
        "docs/CLI-FLAGS.md",
        "examples/paris4-cli.json",
        "examples/run-plan.template.yaml",
        "tests/test_cli_flags_end_to_end.py",
    ):
        assert name in names
