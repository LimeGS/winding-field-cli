import importlib.metadata as metadata
import importlib.util
import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_core_dependency_and_provenance_use_renamed_distribution():
    from packaging.requirements import Requirement

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    provenance = json.loads((ROOT / "provenance/core.json").read_text())
    assert provenance["distribution"] == "winding-field"
    core = next(
        Requirement(value)
        for value in config["project"]["dependencies"]
        if Requirement(value).name == "winding-field"
    )
    # A git commit, never a GitHub archive: archives are not served everywhere and
    # carry no content identity of their own.
    assert core.url == f"git+{provenance['origin']}@{provenance['revision']}"
    assert core.url == provenance["install_url"]
    assert "archive_sha256" not in provenance
    assert not any(
        "winding-field-experiments" in value for value in config["project"]["dependencies"]
    )


def test_core_constraint_matches_provenance_and_stays_out_of_hash_locks():
    provenance = json.loads((ROOT / "provenance/core.json").read_text())
    pinned = f"winding-field @ {provenance['install_url']}"
    assert pinned in (ROOT / "docker/cli.in").read_text().splitlines()
    for variant in ("cpu", "cu128"):
        text = (ROOT / f"docker/cli-{variant}.txt").read_text()
        # A git source has no archive hash; install-runtime.sh installs and verifies it.
        assert "--no-emit-package winding-field" in text.splitlines()[1]
        assert not any(line.startswith("winding-field @") for line in text.splitlines())


def test_core_verifier_reports_commit_source_and_origin_mismatches(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("verify_core", ROOT / "tools/verify-core.py")
    verify_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_core)
    source = tmp_path / "field_runtime.py"
    source.write_text("x = 1\n")
    provenance = {
        "distribution": "winding-field",
        "version": "1.5.4",
        "origin": "https://github.com/LimeGS/winding-field",
        "revision": "a" * 40,
        "source_sha256": {"field_runtime.py": hashlib.sha256(b"x = 1\n").hexdigest()},
    }

    class Dist:
        version = "1.5.4"

        def __init__(self, direct):
            self.direct = direct

        def read_text(self, name):
            return json.dumps(self.direct)

    def install(direct):
        monkeypatch.setattr(verify_core.metadata, "distribution", lambda name: Dist(direct))

    monkeypatch.setattr(
        verify_core.importlib.util,
        "find_spec",
        lambda name: type("Spec", (), {"origin": str(source)})(),
    )
    good = {"url": provenance["origin"], "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}
    install(good)
    assert verify_core.problems(provenance) == []
    install({**good, "vcs_info": {"vcs": "git", "commit_id": "b" * 40}})
    assert any("commit" in p for p in verify_core.problems(provenance))
    archive = {"url": f"{provenance['origin']}/archive/{'a' * 40}.tar.gz", "archive_info": {}}
    install(archive)
    assert any("by git" in p for p in verify_core.problems(provenance))
    install(good)
    source.write_text("x = 2\n")
    assert any("sha256" in p for p in verify_core.problems(provenance))


def test_spline_dependency_matches_provenance_and_frozen_archives():
    from packaging.requirements import Requirement
    from winding_field.splines import SPLINE_REV

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    provenance = json.loads((ROOT / "provenance/qual.json").read_text())
    spline = next(
        Requirement(value)
        for value in config["project"]["dependencies"]
        if Requirement(value).name == "avf-spline-generator"
    )
    assert SPLINE_REV == provenance["spline_revision"]
    archive = f"{provenance['spline_repository']}/archive/{provenance['spline_revision']}.tar.gz"
    assert spline.url == archive
    installed = next(
        Requirement(value)
        for value in metadata.requires("winding-field-cli")
        if Requirement(value).name == "avf-spline-generator"
    )
    assert installed.url == archive
    for variant in ("cpu", "cu128"):
        assert (
            "avf-spline-generator @ " + archive + " \\"
            in (ROOT / f"docker/cli-{variant}.txt").read_text().splitlines()
        )


def test_installed_core_matches_frozen_source():
    provenance = json.loads((ROOT / "provenance/core.json").read_text())
    assert metadata.version("winding-field") == provenance["version"]
    for name, expected in provenance["source_sha256"].items():
        path = Path(importlib.util.find_spec(Path(name).stem).origin)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, (
            f"core source differs: {path}"
        )


def test_installed_cli_version_matches_package_metadata():
    from winding_field import __version__

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert __version__ == config["project"]["version"] == metadata.version("winding-field-cli")


def test_tifxyz_identity_records_renamed_core_version():
    from winding_field.tifxyz import _code_identity

    identity = _code_identity()
    assert "winding-field" in identity["numeric"]
    assert identity["numeric"]["winding-field"] == metadata.version("winding-field")
    assert "winding-field-experiments" not in identity["numeric"]
