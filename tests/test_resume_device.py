"""A run prepared on one host continues its field on another device (CPU prepare, GPU field)."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


def invoke(*args, cwd):
    env = dict(os.environ, OMP_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-m", "winding_field", *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _config(tmp_path, source):
    return dict(
        source=source,
        output=str(tmp_path / "run"),
        prediction="classical",
        reserve_bytes=0,
        cache=str(tmp_path / "cache"),
        device="cpu",
        field_chunk=64,
        sweeps=1,
        seed_stride=4,
        chart_faces=1024,
    )


def _files(root):
    from winding_field.util import file_sha

    return {p.relative_to(root): file_sha(p) for p in Path(root).rglob("*") if p.is_file()}


def test_device_changes_before_the_field_and_enters_its_identity(tmp_path):
    from winding_field.demo import synthetic
    from winding_field.workflow import run

    config = _config(tmp_path, synthetic(tmp_path / "ct.zarr"))
    run(config, until="fibres")
    state = run(config, resume=True, until="field", device="cpu:0")
    assert state["config"]["device"] == "cpu:0"
    assert [(c["from"], c["to"]) for c in state["device_changes"]] == [("cpu", "cpu:0")]
    saved = json.loads((tmp_path / "run/RUN.json").read_bytes())
    assert saved["config"]["device"] == "cpu:0"
    checkpoint = json.loads((tmp_path / "run/field/CHECKPOINT.json").read_bytes())
    assert checkpoint["identity"]["device"] == "cpu:0"
    # The same device again is not a change.
    again = run(saved["config"], resume=True, until="field", device="cpu:0")
    assert len(again["device_changes"]) == 1


def test_device_override_is_refused_once_the_field_started(tmp_path):
    from winding_field.demo import synthetic
    from winding_field.workflow import run

    config = _config(tmp_path, synthetic(tmp_path / "ct.zarr"))
    run(config, until="field")
    before = _files(tmp_path / "run")
    with pytest.raises(ValueError, match="field has started"):
        run(config, resume=True, device="cpu:0")
    assert _files(tmp_path / "run") == before


def test_device_override_needs_prepared_fibres_a_resume_and_a_usable_device(tmp_path):
    from winding_field.demo import synthetic
    from winding_field.workflow import run

    config = _config(tmp_path, synthetic(tmp_path / "ct.zarr"))
    with pytest.raises(ValueError, match="resumed run"):
        run(config, device="cpu:0")
    run(config, until="prepare")
    before = _files(tmp_path / "run")
    with pytest.raises(ValueError, match="missing fibres"):
        run(config, resume=True, device="cpu:0")
    assert _files(tmp_path / "run") == before
    run(config, resume=True, until="fibres")
    before = _files(tmp_path / "run")
    with pytest.raises(ValueError, match="not usable"):
        run(config, resume=True, device="nonexistent-device")
    assert _files(tmp_path / "run") == before


def test_split_cpu_host_then_field_host_through_the_cli(tmp_path):
    """Host A prepares; the run is copied (tar, new inodes) to host B at the same path."""
    from winding_field.demo import synthetic

    host_a = tmp_path / "host"
    host_a.mkdir()
    source = synthetic(host_a / "ct.zarr")
    common = [
        "--source", source, "--cache", host_a / "cache", "--reserve-gib", "0",
        "--prediction", "classical", "--field-chunk", "64", "--sweeps", "1",
        "--seed-stride", "4", "--chart-faces", "1024", "--device", "cpu",
    ]  # fmt: skip
    done = invoke("run", *common, "--output", host_a / "run", "--until", "fibres", cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    archive = tmp_path / "transfer.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(host_a, arcname="host")
    shutil.rmtree(host_a)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path, filter="tar")
    done = invoke("resume", host_a / "run", "--device", "cpu:0", "--offline", cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    state = json.loads((host_a / "run/RUN.json").read_bytes())
    assert state["status"] == "COMPLETE" and state["config"]["device"] == "cpu:0"
    # Once the field exists the device is fixed.
    done = invoke("resume", host_a / "run", "--device", "cpu", cwd=tmp_path)
    assert done.returncode == 1 and "field has started" in done.stderr
