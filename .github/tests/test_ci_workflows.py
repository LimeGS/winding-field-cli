"""Scheduling/cache optimizations must retain the expensive qualification gates."""

import json
from pathlib import Path
import re
import shlex

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    # BaseLoader keeps GitHub's `on` key and expressions as strings (not YAML1.1 booleans).
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


@pytest.mark.parametrize("name", ["tests.yml", "docker.yml"])
def test_only_main_pushes_and_prs_schedule_automatic_suites(name):
    triggers = workflow(name)["on"]
    assert triggers["push"]["branches"] == ["main"]
    assert "pull_request" in triggers


@pytest.mark.parametrize("name", ["tests.yml", "docker.yml"])
def test_only_obsolete_pr_runs_are_canceled(name):
    concurrency = workflow(name)["concurrency"]
    assert (
        concurrency["group"]
        == "${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}"
    )
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


@pytest.mark.parametrize(
    "name,heavy_jobs", [("tests.yml", ["installed-package"]), ("docker.yml", ["cpu", "cuda"])]
)
def test_lightweight_failures_block_expensive_jobs(name, heavy_jobs):
    jobs = workflow(name)["jobs"]
    preflight = jobs["preflight"]
    commands = "\n".join(step.get("run", "") for step in preflight["steps"])
    for test in ["test_ci_workflows.py", "test_container_layout.py", "test_install_script.py"]:
        assert test in commands
    assert "pytest" in commands and "PyYAML" in commands
    for job in heavy_jobs:
        assert jobs[job]["needs"] == "preflight"
        assert "always()" not in jobs[job].get("if", "")


def test_cpu_cache_uses_pinned_bake_action_and_explicit_builder_and_path():
    steps = workflow("docker.yml")["jobs"]["cpu"]["steps"]
    buildx = next(step for step in steps if step.get("id") == "buildx")
    assert buildx["uses"].startswith("docker/setup-buildx-action@")
    bake = next(step for step in steps if step.get("uses", "").startswith("docker/bake-action@"))
    assert re.fullmatch(r"docker/bake-action@[0-9a-f]{40}", bake["uses"])
    inputs = bake["with"]
    assert inputs["builder"] == "${{ steps.buildx.outputs.name }}"
    assert inputs["source"] == "." and inputs["targets"] == "cpu" and inputs["load"] == "true"
    assert inputs["set"].splitlines() == [
        "cpu.cache-from=type=gha,scope=winding-field-cli-cpu",
        "cpu.cache-to=type=gha,scope=winding-field-cli-cpu,mode=max,ignore-error=true",
    ]
    assert inputs.get("push", "false") == "false"


def test_test_image_build_uses_runner_dns_without_enabling_runtime_network():
    steps = workflow("docker.yml")["jobs"]["cpu"]["steps"]
    commands = [line.strip() for step in steps for line in step.get("run", "").splitlines()]
    build = next(
        shlex.split(line)
        for line in commands
        if line.startswith("docker build ") and "docker/testing.Dockerfile" in line
    )
    assert "--network=host" in build
    offline = next(
        step["run"]
        for step in steps
        if step.get("name") == "Installed non-root runtime and offline workflows"
    )
    assert "--network=none" in offline
    assert "APT::Update::Error-Mode=any" in (ROOT / "docker/testing.Dockerfile").read_text()
    assert (
        "https://snapshot.ubuntu.com/ubuntu/20261101T000000Z/" in (ROOT / "Dockerfile").read_text()
    )


def test_full_suites_and_optional_gpu_checks_are_preserved_and_timed():
    package = workflow("tests.yml")["jobs"]["installed-package"]
    assert matrix_for(package, "push") == [
        {"os": "ubuntu-latest", "python": "3.11"},
        {"os": "macos-latest", "python": "3.12"},
    ]
    assert matrix_for(package, "pull_request") == [{"os": "ubuntu-latest", "python": "3.11"}]
    package_commands = "\n".join(step.get("run", "") for step in package["steps"])
    assert "python -m pytest -q --durations=30 -o pythonpath=" in package_commands
    assert "winding-field resume demo" in package_commands
    docker = workflow("docker.yml")
    commands = "\n".join(step.get("run", "") for step in docker["jobs"]["cpu"]["steps"])
    assert "--durations=30 -o pythonpath=" in commands and "/tests" in commands
    assert "dst=/benchmarks,readonly" in commands
    # sdist tests copy the source layout; the container mounts every input they need.
    for path in (
        "/MANIFEST.in",
        "/THIRD_PARTY.md",
        "/licenses",
        "/docs",
        "/README.md",
        "/uv.toml",
        "/REVIEW.md",
        "/.claude/skills",
        "/compose.yaml",
        "/docker-bake.hcl",
    ):
        assert f"dst={path},readonly" in commands
    assert "build==" in (ROOT / "docker/testing.Dockerfile").read_text()
    suite = next(
        step
        for step in docker["jobs"]["cpu"]["steps"]
        if step.get("name") == "Installed suite with official native checkpoint qualification"
    )
    assert suite["if"] == "github.event_name != 'pull_request'"
    smoke = next(
        step
        for step in docker["jobs"]["cpu"]["steps"]
        if step.get("name") == "Installed non-root runtime and offline workflows"
    )
    assert "if" not in smoke
    assert "verify-runtime.py --variant cpu" in commands and "smoke.py" in commands
    gpu = docker["jobs"]["cuda"]
    assert "github.event_name == 'workflow_dispatch' && inputs.cuda" == gpu["if"]
    assert gpu["runs-on"] == ["self-hosted", "linux", "x64", "nvidia"]
    assert "cuda" in docker["on"]["workflow_dispatch"]["inputs"]
    assert "gpu" in docker["on"]["workflow_dispatch"]["inputs"]
    assert any(
        "--models --device cuda:0" in step.get("run", "") and step.get("if") == "inputs.gpu"
        for step in gpu["steps"]
    )


def matrix_for(job, event):
    expression = job["strategy"]["matrix"]["include"]
    match = re.fullmatch(
        r"\$\{\{ fromJSON\(github.event_name == 'pull_request' && '(.+?)' \|\| '(.+?)'\) \}\}",
        expression,
    )
    assert match, expression
    return json.loads(match[1] if event == "pull_request" else match[2])


def test_pull_requests_reuse_vc_reader_and_main_resolves_the_real_wheel():
    steps = workflow("tests.yml")["jobs"]["installed-package"]["steps"]
    full = [step for step in steps if "python -m pip install dist/*.whl" in step.get("run", "")]
    assert len(full) == 1 and full[0]["if"] == "github.event_name != 'pull_request'"
    cache = next(
        step for step in steps if step.get("uses", "").startswith("actions/cache/restore@")
    )
    assert cache["if"] == "github.event_name == 'pull_request'"
    assert (
        "${{ steps.vc.outputs.key }}" in cache["with"]["key"]
        and "${{ matrix.python }}" in cache["with"]["key"]
    )
    save = next(step for step in steps if step.get("uses", "").startswith("actions/cache/save@"))
    assert save["with"]["key"] == "${{ steps.vc-cache.outputs.cache-primary-key }}"
    assert steps.index(save) < next(
        i for i, step in enumerate(steps) if step.get("run", "").startswith("python -m pytest")
    )
    fast = next(
        step
        for step in steps
        if step.get("name") == "Install wheel with cached vc-reader (pull requests)"
    )
    assert "--no-deps dist/*.whl" in fast["run"] and "vc-reader-wheel" in fast["run"]
    assert any(step.get("run") == "python -m pip check" for step in steps)
    pytest_step = next(
        step["run"] for step in steps if step.get("run", "").startswith("python -m pytest")
    )
    assert "-n auto --dist loadfile" in pytest_step


def test_linux_jobs_provision_and_require_flatboi():
    steps = workflow("tests.yml")["jobs"]["installed-package"]["steps"]
    provision = [
        s for s in steps if "winding_field.flattening.flatboi --provision" in s.get("run", "")
    ]
    assert len(provision) == 1 and "runner.os == 'Linux'" in provision[0]["if"]
    pytest_step = next(s for s in steps if s.get("run", "").startswith("python -m pytest"))
    assert (
        pytest_step["env"]["WINDING_REQUIRE_FLATBOI"] == "${{ runner.os == 'Linux' && '1' || '0' }}"
    )
    assert steps.index(provision[0]) < steps.index(pytest_step)
    docker = workflow("docker.yml")
    commands = "\n".join(step.get("run", "") for step in docker["jobs"]["cpu"]["steps"])
    assert "-e WINDING_REQUIRE_FLATBOI=1" in commands


def test_installed_sdist_checks_have_their_build_dependencies():
    steps = workflow("tests.yml")["jobs"]["installed-package"]["steps"]
    setup = next(
        step["run"]
        for step in steps
        if step.get("run", "").startswith("python -m pip install build ")
    )
    packages = set(shlex.split(setup)[4:])
    assert {"build", "pytest", "wheel", "pytest-xdist", "packaging"} <= packages
