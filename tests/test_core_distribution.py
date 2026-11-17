import importlib.metadata as metadata
import importlib.util
import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_core_dependency_and_provenance_use_renamed_distribution():
    from packaging.requirements import Requirement
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    provenance = json.loads((ROOT / 'provenance/core.json').read_text())
    assert provenance['distribution'] == 'winding-field'
    core = next(Requirement(value) for value in config['project']['dependencies']
                if Requirement(value).name == 'winding-field')
    assert core.url == f"{provenance['origin']}/archive/{provenance['revision']}.tar.gz"
    assert not any('winding-field-experiments' in value for value in config['project']['dependencies'])


def test_core_archive_constraint_matches_provenance():
    provenance = json.loads((ROOT / 'provenance/core.json').read_text())
    archive = f"winding-field @ {provenance['origin']}/archive/{provenance['revision']}.tar.gz"
    assert archive in (ROOT / 'docker/cli.in').read_text().splitlines()
    for variant in ('cpu', 'cu128'):
        lines = (ROOT / f'docker/cli-{variant}.txt').read_text().splitlines()
        index = lines.index(archive + ' \\')
        assert lines[index + 1].strip() == '--hash=sha256:' + provenance['archive_sha256']


def test_installed_core_matches_frozen_source():
    provenance = json.loads((ROOT / 'provenance/core.json').read_text())
    assert metadata.version('winding-field') == provenance['version']
    for name, expected in provenance['source_sha256'].items():
        path = Path(importlib.util.find_spec(Path(name).stem).origin)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'core source differs: {path}'


def test_installed_cli_version_matches_package_metadata():
    from winding_field import __version__
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert __version__ == config['project']['version'] == metadata.version('winding-field-cli')


def test_tifxyz_identity_records_renamed_core_version():
    from winding_field.tifxyz import _code_identity
    identity = _code_identity()
    assert 'winding-field' in identity['numeric']
    assert identity['numeric']['winding-field'] == metadata.version('winding-field')
    assert 'winding-field-experiments' not in identity['numeric']
