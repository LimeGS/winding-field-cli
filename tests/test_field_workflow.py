import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from constraint_templates import StencilConstraints
from field_store import BlockStore


def workflow():
    assert importlib.util.find_spec("winding_field.field_workflow"), "principal construction/resume workflow missing"
    from winding_field import field_workflow

    return field_workflow


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared(tmp_path, *, fixed=False, constraints=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shape = (3, 4, 6)
    for name, values in {
        "target": np.broadcast_to(np.array([0.125, 0, 0], np.float32), (*shape, 3)),
        "weights": np.ones(shape, np.float32),
        "observed": np.ones(shape, bool),
    }.items():
        np.save(inputs / (name + ".npy"), values)
    config = dict(
        schema="winding-field-run-v1",
        frame="synthetic/world",
        voxel_um=8.64,
        sheet_step=1.0,
        calibration="analytic phi = .125 * world X offset",
        grid=dict(origin_xyz=[11.0, -4.0, 19.0], spacing_xyz=[2.0, 3.0, 5.0], shape_zyx=list(shape)),
        observations={
            k: dict(path="inputs/" + k + ".npy", sha256=sha(inputs / (k + ".npy")))
            for k in ("target", "weights", "observed")
        },
        execution=dict(chunk=4, halo=2, sweeps=2),
        solver=dict(dtype="float64", rtol=1e-10, maxiter=1000),
    )
    if fixed:
        values = np.full(shape, np.nan)
        values[0, 0, 0] = 1.234567890123
        np.save(inputs / "fixed.npy", values)
        config["fixed"] = dict(path="inputs/fixed.npy", sha256=sha(inputs / "fixed.npy"))
    if constraints:
        rows = StencilConstraints.pairs(
            [[11.5, -1.0, 24.0]],
            [[20.5, 2.0, 24.0]],
            [11.0, -4.0, 19.0],
            [2.0, 3.0, 5.0],
            shape,
            weight=2.0,
            targets=-1.125,
        )
        rows.save(inputs / "constraints.npz")
        config["constraints"] = dict(path="inputs/constraints.npz", sha256=sha(inputs / "constraints.npz"))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config))
    return path, config


def stored(path):
    cp = json.loads((path / "CHECKPOINT.json").read_text())
    with BlockStore(path, cp["shape"], cp["chunk"], cp["identity"], resume=True) as store:
        return cp, store.read([0] * 3, store.shape), store.read_observed([0] * 3, store.shape)


def test_real_prepared_run_uses_core_and_exact_physical_geometry(tmp_path):
    path, config = prepared(tmp_path, constraints=True)
    result = workflow().run_field(path, tmp_path / "field")
    cp, field, observed = stored(tmp_path / "field")
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)
    assert observed.all() and result["status"] == "SWEEP_BUDGET_REACHED"
    assert result["global_converged"] is False and cp["identity"]["source_identity"]["frame"] == config["frame"]
    metadata = json.loads((tmp_path / "field/RUN.json").read_text())
    assert metadata["frame"] == config["frame"] and metadata["sheet_step"] == 1.0
    assert metadata["grid"] == config["grid"]


def test_exact_resume_matches_uninterrupted_and_allows_reader_budget_changes(tmp_path):
    path, config = prepared(tmp_path)
    workflow().run_field(path, tmp_path / "full")
    root = tmp_path / "split"

    def stop():
        cp = root / "CHECKPOINT.json"
        return cp.exists() and json.loads(cp.read_text())["progress"].get("next_block") == 1

    assert workflow().run_field(path, root, stop=stop)["status"] == "CHECKPOINTED_STOP"
    config["execution"].update(cache_bytes=1024**2, read_workers=2)
    path.write_text(json.dumps(config))
    result = workflow().run_field(path, root, resume=True, prefetch=True)
    full, expected, support = stored(tmp_path / "full")
    actual, values, observed = stored(root)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(observed, support)
    assert actual["progress"]["completed_sweeps"] == full["progress"]["completed_sweeps"]
    assert result["prefetch_parity"] is True


def test_fixed_values_keep_double_precision_and_zero_gauge_is_not_substituted(tmp_path):
    path, _ = prepared(tmp_path, fixed=True)
    workflow().run_field(path, tmp_path / "field")
    _, field, _ = stored(tmp_path / "field")
    np.testing.assert_allclose(field, 1.234567890123 + np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


@pytest.mark.parametrize("change", ["observations", "fixed", "constraints", "frame", "spacing", "solver"])
def test_changed_problem_cannot_overwrite_resumable_checkpoint(tmp_path, change):
    path, config = prepared(tmp_path, fixed=True, constraints=True)
    root = tmp_path / "field"
    workflow().run_field(path, root, stop=lambda: True)
    before = (root / "CHECKPOINT.json").read_bytes()
    if change in ("fixed", "constraints"):
        config[change]["sha256"] = "f" * 64
    elif change == "observations":
        config["observations"]["target"]["sha256"] = "f" * 64
    elif change == "frame":
        config["frame"] = "different/world"
    elif change == "spacing":
        config["grid"]["spacing_xyz"] = [1.0, 1.0, 1.0]
    else:
        config["solver"]["rtol"] = 1e-7
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow().run_field(path, root, resume=True)
    assert (root / "CHECKPOINT.json").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"unexpected": True},
        {"grid": {"shape_zyx": [2, 2, 2]}},
        {"solver": {"dtype": "float16"}},
        {"solver": {"maxiter": True}},
        {"solver": {"device": "cuda:1"}},
        {"execution": {"chunk": False, "halo": 1, "sweeps": 2}},
        {"voxel_um": 0},
        {"calibration": ""},
        {"sheet_step": float("nan")},
        {"execution": {"chunk": 4, "halo": 0, "sweeps": 1}},
    ],
)
def test_invalid_configuration_fails_before_output(tmp_path, change):
    path, config = prepared(tmp_path)
    config.update(change)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow().run_field(path, tmp_path / "field")
    assert not (tmp_path / "field").exists()


def test_paths_belong_to_configuration_file_and_unknown_input_keys_fail(tmp_path, monkeypatch):
    path, config = prepared(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    parsed = workflow().load_run_config(path)
    assert parsed["observations"]["target"]["path"] == str(tmp_path / "inputs/target.npy")
    config["observations"]["target"]["scale"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow().load_run_config(path)


def test_duplicate_json_members_are_rejected(tmp_path):
    path, _ = prepared(tmp_path)
    text = path.read_text()
    path.write_text(text.replace('"schema":', '"schema":"wrong", "schema":', 1))
    with pytest.raises(ValueError):
        workflow().load_run_config(path)


def test_bad_observation_shape_or_hash_never_creates_output(tmp_path):
    path, config = prepared(tmp_path)
    np.save(tmp_path / "inputs/weights.npy", np.ones((2, 2, 2), np.float32))
    with pytest.raises(ValueError):
        workflow().run_field(path, tmp_path / "field")
    config["observations"]["weights"]["sha256"] = sha(tmp_path / "inputs/weights.npy")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow().run_field(path, tmp_path / "field")
    assert not (tmp_path / "field").exists()


def test_checkout_cli_build_resume_and_error_statuses(tmp_path):
    path, _ = prepared(tmp_path)
    root = tmp_path / "field"

    def stop():
        cp = root / "CHECKPOINT.json"
        return cp.exists() and json.loads(cp.read_text())["progress"].get("next_block") == 1

    workflow().run_field(path, root, stop=stop)
    script = Path(__file__).resolve().parents[1] / "tools/run_field.py"
    assert script.exists(), "principal checkout CLI missing"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [sys.executable, str(script), "resume", "--config", str(path), "--output", str(root)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((root / "STATUS.json").read_text())["status"] == "SWEEP_BUDGET_REACHED"
    path.write_text("{}")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "configuration" in result.stderr.lower()


def test_nonconvergence_is_terminal_and_retains_exact_cursor(tmp_path):
    path, config = prepared(tmp_path)
    config["solver"]["maxiter"] = 0
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="converge"):
        workflow().run_field(path, tmp_path / "field")
    assert stored(tmp_path / "field")[0]["progress"]["next_block"] == 0


def test_all_nan_fixed_input_uses_the_global_numerical_zero_gauge(tmp_path):
    path, config = prepared(tmp_path, fixed=True)
    np.save(tmp_path / "inputs/fixed.npy", np.full((3, 4, 6), np.nan))
    config["fixed"]["sha256"] = sha(tmp_path / "inputs/fixed.npy")
    path.write_text(json.dumps(config))
    workflow().run_field(path, tmp_path / "field")
    _, field, _ = stored(tmp_path / "field")
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


def test_config_change_during_preparation_fails_before_first_commit(tmp_path):
    path, config = prepared(tmp_path)

    def change():
        config["sheet_step"] = 2.0
        path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="source changed"):
        workflow().run_field(path, tmp_path / "field", admit=change)
    assert stored(tmp_path / "field")[0]["progress"]["next_block"] == 0


def test_cli_has_distinct_success_numerical_and_resource_exit_statuses(tmp_path, monkeypatch):
    import field_runtime

    path, config = prepared(tmp_path)
    api = workflow()
    assert api.main(["build", "--config", str(path), "--output", str(tmp_path / "good")]) == 0
    config["solver"]["maxiter"] = 0
    path.write_text(json.dumps(config))
    assert api.main(["build", "--config", str(path), "--output", str(tmp_path / "failed")]) == 1
    config["solver"].update(maxiter=1000, device="cuda:0")
    config["gpu"] = dict(index=0, uuid="GPU-test", lock_path="gpu.lock", min_free_bytes=0)
    path.write_text(json.dumps(config))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")

    def missing(*args, **kwargs):
        raise FileNotFoundError("synthetic NVIDIA inventory unavailable")

    monkeypatch.setattr(field_runtime.subprocess, "check_output", missing)
    assert api.main(["build", "--config", str(path), "--output", str(tmp_path / "busy")]) == 75
    assert json.loads((tmp_path / "busy/STATUS.json").read_text())["status"] == "RESOURCE_UNAVAILABLE"
    assert stored(tmp_path / "busy")[0]["progress"]["next_block"] == 0


def test_cuda_cannot_select_a_device_without_an_explicit_matching_lease(tmp_path, monkeypatch):
    path, config = prepared(tmp_path)
    config["solver"]["device"] = "cuda:0"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="lease"):
        workflow().run_field(path, tmp_path / "field")
    config["gpu"] = dict(index=1, uuid="GPU-test", lock_path="gpu.lock")
    path.write_text(json.dumps(config))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        workflow().run_field(path, tmp_path / "field")
    assert not (tmp_path / "field").exists()


@pytest.mark.parametrize("kind", ["float", "duplicate", "npz"])
def test_invalid_order_source_does_not_create_output(tmp_path, kind):
    path, config = prepared(tmp_path)
    if kind == "npz":
        order = tmp_path / "inputs/order.npz"
        np.savez(order, order=[[0, 0, 0], [0, 0, 1]])
    else:
        order = tmp_path / "inputs/order.npy"
        np.save(
            order,
            np.array([[0, 0, 0], [0, 0, 0 if kind == "duplicate" else 1]], dtype=float if kind == "float" else int),
        )
    config["order"] = dict(path=str(order), sha256=sha(order))
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        workflow().run_field(path, tmp_path / "field")
    assert not (tmp_path / "field").exists()


@pytest.mark.parametrize("action", ["build", "resume"])
def test_supervised_checkout_cli_runs_the_same_prepared_problem(tmp_path, action):
    path, _ = prepared(tmp_path, constraints=True)
    root = tmp_path / "field"
    script = Path(__file__).resolve().parents[1] / "tools/run_field.py"
    env = dict(
        os.environ,
        PYTHONPATH=str(script.parents[1] / "src"),
        CUDA_VISIBLE_DEVICES="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    if action == "resume":
        stopped = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from winding_field.field_workflow import run_field; run_field(sys.argv[1],sys.argv[2],stop=lambda:True)",
                str(path),
                str(root),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert stopped.returncode == 0, stopped.stderr
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            action,
            "--config",
            str(path),
            "--output",
            str(root),
            "--supervised",
            "--prefetch",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "field.service/SERVICE.json").read_text())
    assert state["status"] == "SWEEP_BUDGET_REACHED" and state["worker_pid"] is None
    assert state["last_worker_pid"] != os.getpid()
    _, field, _ = stored(root)
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


@pytest.mark.parametrize("failure", ["hash", "order"])
def test_supervised_preflight_rejects_bad_inputs_without_creating_output(tmp_path, failure):
    path, config = prepared(tmp_path)
    if failure == "hash":
        config["observations"]["target"]["sha256"] = "f" * 64
    else:
        order = tmp_path / "inputs/order.npy"
        np.save(order, [[0, 0, 0], [0, 0, 0]])
        config["order"] = dict(path=str(order), sha256=sha(order))
    path.write_text(json.dumps(config))
    assert workflow().main(["build", "--config", str(path), "--output", str(tmp_path / "field"), "--supervised"]) == 2
    assert not (tmp_path / "field").exists() and not (tmp_path / "field.service").exists()


@pytest.mark.parametrize("name", ["constraints", "order"])
def test_atomic_replacement_between_hash_and_load_cannot_change_the_pinned_problem(tmp_path, monkeypatch, name):
    path, config = prepared(tmp_path, constraints=True)
    source = tmp_path / ("inputs/constraints.npz" if name == "constraints" else "inputs/order.npy")
    replacement = tmp_path / ("replacement.npz" if name == "constraints" else "replacement.npy")
    if name == "constraints":
        StencilConstraints([[0, 71]], [[1.0, -1.0]], (3, 4, 6), targets=4).save(replacement)
    else:
        np.save(source, [[0, 0, 0], [0, 0, 1]])
        np.save(replacement, [[0, 0, 1], [0, 0, 0]])
        config["order"] = dict(path=str(source), sha256=sha(source))
        path.write_text(json.dumps(config))
    digest = hashlib.file_digest
    replaced = False

    def replace_after_hash(stream, algorithm):
        nonlocal replaced
        result = digest(stream, algorithm)
        if Path(stream.name) == source and not replaced:
            replacement.replace(source)
            replaced = True
        return result

    monkeypatch.setattr(hashlib, "file_digest", replace_after_hash)
    with pytest.raises(ValueError, match="source.*changed|source.*hash"):
        workflow().run_field(path, tmp_path / "field")
    assert replaced and not (tmp_path / "field").exists()


@pytest.mark.parametrize(
    "name,raw",
    [
        ("constraints", b""),
        ("constraints", b"PK\x03\x04broken zip"),
        ("order", b""),
        ("fixed", b""),
    ],
)
def test_malformed_prepared_artifacts_are_input_errors_without_output(tmp_path, name, raw):
    path, config = prepared(tmp_path)
    source = tmp_path / "inputs" / (name + (".npz" if name == "constraints" else ".npy"))
    source.write_bytes(raw)
    config[name] = dict(path=str(source), sha256=sha(source))
    path.write_text(json.dumps(config))
    assert workflow().main(["build", "--config", str(path), "--output", str(tmp_path / "field")]) == 2
    assert not (tmp_path / "field").exists()
