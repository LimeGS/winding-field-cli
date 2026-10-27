import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("supervised", [False, True])
def test_offline_example_stops_resumes_and_prefetches_without_parent_repo(tmp_path, supervised):
    root = Path(__file__).resolve().parents[1]
    script = root / "examples/block_field.py"
    assert script.is_file(), "runnable block example is missing"
    env = dict(
        os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    output = tmp_path / "example"
    command = [sys.executable, str(script), "--output", str(output)]
    stopped = subprocess.run(
        [*command, "--stop-after-blocks", "1"], env=env, capture_output=True, text=True, timeout=30
    )
    assert stopped.returncode == 0, stopped.stderr
    assert json.loads((output / "field/STATUS.json").read_text())["status"] == "CHECKPOINTED_STOP"
    resumed = subprocess.run(
        [*command, "--resume", "--prefetch", *(["--supervised"] if supervised else [])],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((output / "RESULT.json").read_text())
    assert result["synthetic_max_error"] < 2e-7 and result["prefetch_parity"]
    assert result["global_converged"] is False
    field = np.load(output / "synthetic-field.npy", allow_pickle=False)
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


def test_small_cycle_probe_reports_parity_without_a_timing_gate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "tools/benchmark_blocks.py"
    assert script.is_file(), "complete-cycle performance probe is missing"
    env = dict(
        os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path / "probe"), "--repetitions", "1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "probe/REPORT.json").read_text())
    assert report["parity"] is True and len(report["runs"]) == 2
    assert report["prepared_parity"] is True
    assert {r["variant"] for r in report["runs"]} == {"serial-uncached", "bounded-parallel"}
    assert all(r["complete_cycle_seconds"] > 0 for r in report["runs"])


def test_example_supervisor_owns_a_real_worker_and_finishes_at_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/block_field.py"),
            "--output",
            str(tmp_path / "supervised"),
            "--supervised",
            "--prefetch",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    service = json.loads((tmp_path / "supervised/service/SERVICE.json").read_text())
    assert service["status"] == "SWEEP_BUDGET_REACHED" and service["worker_pid"] is None
    assert service["last_worker_pid"] > 0
    receipt = json.loads((tmp_path / "supervised/RESULT.json").read_text())
    assert receipt["synthetic_max_error"] < 2e-7
