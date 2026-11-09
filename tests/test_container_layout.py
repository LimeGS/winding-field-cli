"""Library sharing must reduce bytes without changing installed package contents."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / 'docker/deduplicate-libraries.py'


def run_helper(*roots, check=True):
    result = subprocess.run([sys.executable, str(HELPER), *map(str, roots)],
                            capture_output=True, text=True)
    if not check:
        return result
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def library(root, name, data=b'library', mode=0o644):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    return path


@pytest.mark.parametrize('name', ['libcuda.so', 'libcuda.so.12.8', 'libcuda.a'])
def test_identical_libraries_share_storage_and_second_run_is_noop(tmp_path, name):
    a = library(tmp_path / 'py312', name)
    b = library(tmp_path / 'py314', name)
    metadata = library(tmp_path / 'py314', 'package.dist-info/RECORD', b'unchanged')
    assert not a.samefile(b)
    assert run_helper(tmp_path) == {'files': 1, 'bytes': 7}
    assert a.samefile(b) and a.read_bytes() == b.read_bytes() == b'library'
    assert a.stat().st_mode & 0o777 == b.stat().st_mode & 0o777 == 0o644
    assert metadata.read_bytes() == b'unchanged'
    assert run_helper(tmp_path) == {'files': 0, 'bytes': 0}


def test_equal_size_different_abi_bytes_and_nonlibraries_are_not_shared(tmp_path):
    a = library(tmp_path / 'py312', 'module.so', b'cp312')
    b = library(tmp_path / 'py314', 'module.so', b'cp314')
    c = library(tmp_path / 'py312', 'module.py', b'python')
    d = library(tmp_path / 'py314', 'module.py', b'python')
    assert run_helper(tmp_path) == {'files': 0, 'bytes': 0}
    assert not a.samefile(b) and not c.samefile(d)
    assert a.read_bytes() == b'cp312' and b.read_bytes() == b'cp314'


def test_different_permissions_are_preserved(tmp_path):
    a = library(tmp_path / 'a', 'lib.so', mode=0o644)
    b = library(tmp_path / 'b', 'lib.so', mode=0o755)
    assert run_helper(tmp_path) == {'files': 0, 'bytes': 0}
    assert not a.samefile(b)
    assert a.stat().st_mode & 0o777 == 0o644
    assert b.stat().st_mode & 0o777 == 0o755


def test_symlinks_and_symlink_directories_are_not_followed(tmp_path):
    root = tmp_path / 'image'
    a = library(root, 'lib.so')
    outside = library(tmp_path / 'outside', 'lib.so')
    link = root / 'alias.so'
    link.symlink_to(outside)
    (root / 'directory').symlink_to(outside.parent, target_is_directory=True)
    assert run_helper(root) == {'files': 0, 'bytes': 0}
    assert link.is_symlink() and not outside.samefile(a)


def test_symlink_roots_are_rejected_before_any_modification(tmp_path):
    root = tmp_path / 'real'
    a = library(root / 'a', 'lib.so')
    b = library(root / 'b', 'lib.so')
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    result = run_helper(alias, check=False)
    assert result.returncode != 0 and 'symlink' in result.stderr.lower()
    assert not a.samefile(b)


def test_existing_hardlinks_and_overlapping_roots_are_not_double_counted(tmp_path):
    a = library(tmp_path / 'a', 'lib.so')
    b = tmp_path / 'a' / 'alias.so'
    os.link(a, b)
    c = library(tmp_path / 'b', 'lib.so')
    assert run_helper(tmp_path, a.parent) == {'files': 1, 'bytes': 7}
    assert a.samefile(b) and b.samefile(c)


def test_different_ownership_is_not_shared(tmp_path):
    if os.geteuid() != 0:
        pytest.skip('ownership fixture requires root')
    a = library(tmp_path / 'a', 'lib.so')
    b = library(tmp_path / 'b', 'lib.so')
    os.chown(b, 10001, 10001)
    assert run_helper(tmp_path) == {'files': 0, 'bytes': 0}
    assert not a.samefile(b) and b.stat().st_uid == 10001
