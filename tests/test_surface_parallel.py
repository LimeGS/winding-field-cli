import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from test_surface_workflow import prepared


def test_cpu_limit_honors_nested_quota_and_affinity(tmp_path):
    from winding_field.surface_parallel import cpu_limit

    proc, groups = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/parent/leaf\n")
    (groups / "parent/leaf").mkdir(parents=True)
    (groups / "cpu.max").write_text("max 100000\n")
    (groups / "parent/cpu.max").write_text("250000 100000\n")
    (groups / "parent/leaf/cpu.max").write_text("max 100000\n")
    assert cpu_limit(proc, groups, affinity=8) == 2
    assert cpu_limit(proc, groups, affinity=1) == 1
    (groups / "parent/leaf/cpu.max").write_text("invalid")
    assert cpu_limit(proc, groups, affinity=8) == 1


def unlimited(config, render, output, active=()):
    return dict(slots=max(0, 2 - len(active)), cpus=2, memory_known=True,
                worker_bytes=512 << 20, staging_bytes=32 << 20)


def run(tmp_path, monkeypatch, **kwargs):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, _ = prepared(tmp_path)
    return surface_parallel, path, surface_parallel.run_parallel_levels(path, tmp_path / "parallel", **kwargs)


def test_real_parallel_levels_match_serial_and_completed_resume_is_readonly(tmp_path, monkeypatch):
    from winding_field.surface_workflow import run_surfaces

    parallel, path, result = run(tmp_path, monkeypatch)
    assert result["status"] == "LEVELS_READY"
    assert result["peak_workers"] == 2
    serial = tmp_path / "serial"
    run_surfaces(path, serial, render=False)
    records = result["levels"]
    assert [r["level"] for r in records] == [1.5, 3.5]
    for i, record in enumerate(records):
        child = tmp_path / "parallel" / record["path"]
        assert (child / "level-0000.obj").read_bytes() == (serial / f"level-{i:04d}.obj").read_bytes()
        with sqlite3.connect(child / "mesh.sqlite") as db:
            assert db.execute("SELECT count(*) FROM levels").fetchone()[0] == 1
    root = tmp_path / "parallel"
    before = {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")}
    parallel.run_parallel_levels(path, root, resume=True)
    assert {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in before} == before


@pytest.mark.parametrize("target", ["level-0000/level-0000.obj", "level-0000/MESH.json", "configs/level-0000.json"])
def test_completed_collection_rejects_changed_retained_products(tmp_path, monkeypatch, target):
    parallel, path, _ = run(tmp_path, monkeypatch)
    root = tmp_path / "parallel"
    changed = root / target
    changed.write_text("foreign")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        parallel.run_parallel_levels(path, root, resume=True)
    assert {name: (root / name).read_bytes() for name in before} == before


def test_stop_then_resume_and_missing_committed_child_rejected(tmp_path, monkeypatch):
    parallel, path, result = run(tmp_path, monkeypatch, stop=lambda: True)
    assert result["status"] == "CHECKPOINTED_STOP"
    result = parallel.run_parallel_levels(path, tmp_path / "parallel", resume=True)
    assert result["status"] == "LEVELS_READY"
    (tmp_path / "parallel/level-0000/RUN.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        parallel.run_parallel_levels(path, tmp_path / "parallel", resume=True)


def test_render_filter_applies_only_to_selected_children(tmp_path, monkeypatch):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, config = prepared(tmp_path)
    config["render"]["levels"] = [3.5]
    path.write_text(json.dumps(config))
    result = surface_parallel.run_parallel_levels(path, tmp_path / "parallel", render=True)
    assert result["status"] == "LEVELS_READY"
    assert not (tmp_path / "parallel/level-0000/renders").exists()
    assert (tmp_path / "parallel/level-0001/renders/front.png").is_file()


def test_cli_parallel_flag_and_invalid_render_action(tmp_path):
    path, _ = prepared(tmp_path)
    args = [sys.executable, "-m", "winding_field", "prepared"]
    p = subprocess.run(args + ["mesh", "--parallel-levels", "--config", str(path),
                              "--output", str(tmp_path / "parallel")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    result = json.loads(p.stdout)
    assert result["status"] == "LEVELS_READY"
    from winding_field.surface_parallel import cpu_limit
    from winding_field.tifxyz_runtime import memory_report

    if cpu_limit() >= 2 and memory_report()["known"]:
        assert result["peak_workers"] == 2
    p = subprocess.run(args + ["render", "--parallel-levels", "--mesh", str(tmp_path / "parallel"),
                              "--output", str(tmp_path / "bad")], capture_output=True, text=True)
    assert p.returncode == 2
    assert not (tmp_path / "bad").exists()


def test_resource_admission_reserves_host_ram_and_inflight_allocations(tmp_path, monkeypatch):
    from winding_field import surface_parallel, tifxyz_runtime

    path, _ = prepared(tmp_path)
    config = surface_parallel.load_surface_config(path)
    monkeypatch.setattr(surface_parallel, "cpu_limit", lambda: 16)
    monkeypatch.setattr(surface_parallel, "_v1_memory", lambda: dict(known=False), raising=False)
    monkeypatch.setattr(tifxyz_runtime, "memory_report", lambda: dict(
        known=True, effective_headroom_bytes=1 << 30, host_available_bytes=8 << 30))
    observation = surface_parallel.resources(config, False, tmp_path)
    assert 1 <= observation["slots"] <= 3
    worker = type("Worker", (), {"pid": 1})()
    monkeypatch.setattr(tifxyz_runtime, "process_rss", lambda pid: 0)
    assert surface_parallel.resources(config, False, tmp_path, [worker])["slots"] < observation["slots"]
    monkeypatch.setattr(tifxyz_runtime, "memory_report", lambda: dict(
        known=True, effective_headroom_bytes=1 << 30, host_available_bytes=2 << 30))
    assert surface_parallel.resources(config, False, tmp_path)["slots"] == 0
    monkeypatch.setattr(tifxyz_runtime, "memory_report", lambda: dict(known=False))
    assert surface_parallel.resources(config, False, tmp_path)["slots"] == 1
    assert surface_parallel.resources(config, False, tmp_path, [worker])["slots"] == 0


def test_disk_admission_and_no_resources_do_not_launch(tmp_path, monkeypatch):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", lambda *a: dict(slots=0))
    path, _ = prepared(tmp_path)
    result = surface_parallel.run_parallel_levels(path, tmp_path / "parallel")
    assert result["status"] == "RESOURCE_UNAVAILABLE"
    assert not (tmp_path / "parallel/level-0000").exists()


def test_stop_during_active_children_checkpoints_and_reaps(tmp_path, monkeypatch):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, _ = prepared(tmp_path)
    processes = []
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(surface_parallel.subprocess, "Popen", launch)
    result = surface_parallel.run_parallel_levels(path, tmp_path / "parallel", stop=lambda: bool(processes))
    assert result["status"] == "CHECKPOINTED_STOP"
    assert len(processes) == 1
    assert processes and all(p.poll() is not None for p in processes)
    result = surface_parallel.run_parallel_levels(path, tmp_path / "parallel", resume=True)
    assert result["status"] == "LEVELS_READY"


def test_child_failure_reaps_peers_and_preserves_logs(tmp_path, monkeypatch):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, _ = prepared(tmp_path)
    popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        if not processes:
            command = [sys.executable, "-c", "raise SystemExit(7)"]
        process = popen(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(surface_parallel.subprocess, "Popen", launch)
    result = surface_parallel.run_parallel_levels(path, tmp_path / "parallel")
    assert result["status"] == "LEVELS_FAILED"
    assert all(p.poll() is not None for p in processes)
    assert list((tmp_path / "parallel/logs").glob("*.log"))


@pytest.mark.parametrize("target", ["LEVELS.json", "configs/level-0000.json", "level-0000"])
def test_resume_rejects_symlink_paths(tmp_path, monkeypatch, target):
    parallel, path, _ = run(tmp_path, monkeypatch)
    root = tmp_path / "parallel"
    item = root / target
    moved = tmp_path / "retained"
    item.rename(moved)
    item.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        parallel.run_parallel_levels(path, root, resume=True)
    assert item.is_symlink() and moved.exists()


def test_source_change_rejected_on_completed_resume(tmp_path, monkeypatch):
    import numpy as np

    parallel, path, _ = run(tmp_path, monkeypatch)
    np.save(tmp_path / "phi.npy", np.zeros((5, 6, 7)))
    before = (tmp_path / "parallel/LEVELS.json").read_bytes()
    with pytest.raises(ValueError, match="hash"):
        parallel.run_parallel_levels(path, tmp_path / "parallel", resume=True)
    assert (tmp_path / "parallel/LEVELS.json").read_bytes() == before


@pytest.mark.parametrize("value", [None, [], [None], ["bad"]])
def test_malformed_level_inventory_is_rejected_cleanly(tmp_path, monkeypatch, value):
    parallel, path, _ = run(tmp_path, monkeypatch)
    inventory = tmp_path / "parallel/LEVELS.json"
    saved = json.loads(inventory.read_bytes())
    saved["levels"] = value
    inventory.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        parallel.run_parallel_levels(path, tmp_path / "parallel", resume=True)


def test_completed_render_tampering_is_rejected(tmp_path, monkeypatch):
    parallel, path, _ = run(tmp_path, monkeypatch, render=True)
    png = tmp_path / "parallel/level-0000/renders/front.png"
    png.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="products changed"):
        parallel.run_parallel_levels(path, tmp_path / "parallel", render=True, resume=True)
    assert png.read_bytes() == b"foreign"


def test_unrecorded_completed_child_cannot_recreate_missing_obj(tmp_path, monkeypatch):
    parallel, path, _ = run(tmp_path, monkeypatch)
    root = tmp_path / "parallel"
    inventory = root / "LEVELS.json"
    saved = json.loads(inventory.read_bytes())
    saved["levels"][0].update(state="running", products=None)
    saved["status"] = "EXTRACTING"
    inventory.write_text(json.dumps(saved))
    obj = root / "level-0000/level-0000.obj"
    obj.unlink()
    before = inventory.read_bytes()
    with pytest.raises(ValueError):
        parallel.run_parallel_levels(path, root, resume=True)
    assert not obj.exists() and inventory.read_bytes() == before


@pytest.mark.parametrize("render", [False, True])
def test_unrecorded_sealed_child_is_adopted_without_rewriting(tmp_path, monkeypatch, render):
    parallel, path, _ = run(tmp_path, monkeypatch, render=render)
    root = tmp_path / "parallel"
    inventory = root / "LEVELS.json"
    saved = json.loads(inventory.read_bytes())
    saved["levels"][0].update(state="running", products=None)
    saved["status"] = "EXTRACTING"
    inventory.write_text(json.dumps(saved))
    before = {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in (root / "level-0000").rglob("*") if p.is_file()}
    result = parallel.run_parallel_levels(path, root, resume=True, render=render)
    assert result["status"] == "LEVELS_READY"
    assert {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in before} == before


def test_mesh_ready_render_incomplete_continues_without_reexporting_mesh(tmp_path, monkeypatch):
    parallel, path, _ = run(tmp_path, monkeypatch, render=True)
    root = tmp_path / "parallel"
    inventory = root / "LEVELS.json"
    saved = json.loads(inventory.read_bytes())
    saved["levels"][0].update(state="running", products=None)
    saved["status"] = "EXTRACTING"
    inventory.write_text(json.dumps(saved))
    child = root / "level-0000"
    (child / '.parallel-ready.json').unlink()
    (child / 'renders/RENDER.json').unlink()
    before = {name: ((child / name).read_bytes(), (child / name).stat().st_mtime_ns)
              for name in ('mesh.sqlite', 'level-0000.obj', 'MESH.json')}
    assert parallel.run_parallel_levels(path, root, resume=True, render=True)["status"] == "LEVELS_READY"
    assert {name: ((child / name).read_bytes(), (child / name).stat().st_mtime_ns) for name in before} == before


def test_generated_config_mutation_cannot_publish_success(tmp_path, monkeypatch):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, _ = prepared(tmp_path)
    popen = subprocess.Popen

    def launch(command, **kwargs):
        config = tmp_path / "parallel/configs/level-0000.json"
        saved = json.loads(config.read_bytes())
        saved["execution"]["tile"] = 1
        config.write_text(json.dumps(saved))
        return popen(command, **kwargs)

    monkeypatch.setattr(surface_parallel.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match="config"):
        surface_parallel.run_parallel_levels(path, tmp_path / "parallel")


def test_disk_failure_still_reaps_children_and_restores_signal_handlers(tmp_path, monkeypatch):
    import errno
    import signal
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, _ = prepared(tmp_path)
    processes = []
    popen, atomic = subprocess.Popen, surface_parallel.atomic_json
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def launch(*args, **kwargs):
        p = popen(*args, **kwargs)
        processes.append(p)
        return p

    def full(*args, **kwargs):
        if processes:
            raise OSError(errno.ENOSPC, "disk full")
        return atomic(*args, **kwargs)

    monkeypatch.setattr(surface_parallel.subprocess, "Popen", launch)
    monkeypatch.setattr(surface_parallel, "atomic_json", full)
    try:
        with pytest.raises(OSError):
            surface_parallel.run_parallel_levels(path, tmp_path / "parallel")
        assert processes and all(p.poll() is not None for p in processes)
        assert {s: signal.getsignal(s) for s in previous} == previous
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        for s, handler in previous.items():
            signal.signal(s, handler)


def test_failed_child_prevents_further_launches(tmp_path, monkeypatch):
    from winding_field import surface_parallel

    monkeypatch.setattr(surface_parallel, "resources", unlimited)
    path, _ = prepared(tmp_path)
    popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        p = popen([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)
        p.wait()
        processes.append(p)
        return p

    monkeypatch.setattr(surface_parallel.subprocess, "Popen", launch)
    result = surface_parallel.run_parallel_levels(path, tmp_path / "parallel")
    assert result["status"] == "LEVELS_FAILED" and len(processes) == 1


def test_cli_sigterm_checkpoints_then_resumes(tmp_path):
    import signal
    import time

    path, _ = prepared(tmp_path)
    root = tmp_path / "parallel"
    command = [sys.executable, '-m', 'winding_field', 'prepared', 'mesh', '--parallel-levels',
               '--config', str(path), '--output', str(root)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if (root / 'LEVELS.json').exists():
                saved = json.loads((root / 'LEVELS.json').read_bytes())
                if any(r['state'] == 'running' for r in saved['levels']):
                    break
            assert process.poll() is None
            time.sleep(.01)
        else:
            pytest.fail('parallel child did not start')
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 3, stdout + stderr
        assert json.loads(stdout)['status'] == 'CHECKPOINTED_STOP'
        resumed = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=30)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_hybrid_cgroup_namespace_uses_mounted_v1_limits(tmp_path):
    from winding_field import surface_parallel

    proc, groups = tmp_path / 'proc', tmp_path / 'groups'
    (proc / 'self').mkdir(parents=True)
    (groups / 'cpu,cpuacct').mkdir(parents=True)
    (groups / 'memory').mkdir()
    (proc / 'self/cgroup').write_text('4:cpu,cpuacct:/docker/id\n6:memory:/docker/id\n0::/docker/id\n')
    (proc / 'self/mountinfo').write_text(
        f'1 0 0:1 /docker/id {groups}/cpu,cpuacct ro - cgroup cgroup rw,cpu,cpuacct\n'
        f'2 0 0:2 /docker/id {groups}/memory ro - cgroup cgroup rw,memory\n')
    (groups / 'cpu,cpuacct/cpu.cfs_quota_us').write_text('3071999')
    (groups / 'cpu,cpuacct/cpu.cfs_period_us').write_text('100000')
    (groups / 'memory/memory.limit_in_bytes').write_text(str(64 << 30))
    (groups / 'memory/memory.usage_in_bytes').write_text(str(4 << 30))
    (proc / 'meminfo').write_text(f'MemAvailable: {128 << 20} kB\n')
    assert surface_parallel.cpu_limit(proc, groups, affinity=64) == 30
    memory = surface_parallel._v1_memory(proc, groups)
    assert memory['known'] and memory['effective_headroom_bytes'] == 60 << 30
    (proc / 'self/mountinfo').write_text('')
    assert surface_parallel.cpu_limit(proc, groups, affinity=64) == 1
    assert not surface_parallel._v1_memory(proc, groups)['known']
