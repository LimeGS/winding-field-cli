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
    # Every apt source in the image uses the one dated snapshot of the security baseline.
    dockerfile = (ROOT / "Dockerfile").read_text()
    snapshots = set(re.findall(r"https://snapshot\.ubuntu\.com/ubuntu/(\d{8}T\d{6}Z)/", dockerfile))
    assert snapshots == {"20261121T000000Z"}


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
    # The in-image suite was the longest step of main's CI while it ran serially.
    assert "-n auto --dist loadfile" in commands
    assert "pytest-xdist==" in (ROOT / "docker/testing.Dockerfile").read_text()
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


def test_only_qualified_main_builds_are_published():
    docker = workflow("docker.yml")
    publish = docker["jobs"]["publish"]
    # Publishing waits for the qualified CPU build and never runs for pull requests.
    assert publish["needs"] == "cpu"
    assert publish["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    # Only the publish job may write packages; the build that tests run under stays read-only.
    assert docker["permissions"] == {"contents": "read"}
    for name, job in docker["jobs"].items():
        if name not in ("publish", "release", "release-cuda"):
            assert "packages" not in job.get("permissions", {})
    steps = publish["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
    bake = next(step for step in steps if step.get("uses", "").startswith("docker/bake-action@"))
    assert bake["with"]["targets"] == "cpu" and bake["with"].get("push", "false") == "false"
    assert "cache-to" not in bake["with"]["set"]
    login = next(step for step in steps if step.get("uses", "").startswith("docker/login-action@"))
    assert login["with"]["registry"] == "ghcr.io"
    push = "\n".join(step.get("run", "") for step in steps)
    # Rolling tags only: a version tag must never follow main.
    assert 'docker push "$image:$tag"' in push
    assert 'for tag in edge-cpu "sha-${GITHUB_SHA::7}-cpu"; do' in push and "version" not in push


def test_release_tags_promote_the_qualified_image_without_rebuilding():
    docker = workflow("docker.yml")
    assert docker["on"]["push"] == {"branches": ["main"], "tags": ["v*"]}
    release = docker["jobs"]["release"]
    assert release["if"] == "github.event_name == 'push' && github.ref_type == 'tag'"
    assert "needs" not in release
    assert release["permissions"] == {"contents": "read", "packages": "write"}
    # A tag push builds and tests nothing: every other job hangs off preflight or is manual.
    jobs = docker["jobs"]
    assert jobs["preflight"]["if"] == "github.ref_type != 'tag'"
    assert jobs["cpu"]["needs"] == "preflight" and jobs["publish"]["needs"] == "cpu"
    assert "workflow_dispatch" in jobs["cuda"]["if"]
    for step in release["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
            assert not step["uses"].startswith(("docker/bake-action@", "docker/build-push-action@"))
    run = "\n".join(step.get("run", "") for step in release["steps"])
    assert "docker build " not in run and "bake" not in run and "docker push" not in run
    # The tag must be this version, the commit's image must exist, and a release is immutable.
    assert '"$GITHUB_REF_NAME" != "v$version"' in run
    assert 'source="$image:sha-${GITHUB_SHA::7}-cpu"' in run
    assert '[ "$released" != "$qualified" ]' in run
    assert 'imagetools create --tag "$image:$version-cpu" "${rolling[@]}" "$source"' in run
    # The rolling cpu tag follows stable releases only, never a pre-release.
    assert 'rolling=(--tag "$image:cpu")' in run
    assert 'if [[ "$version" =~ (a|b|rc)[0-9]+$ ]]; then rolling=(); fi' in run


def assert_configurable_gpu(run):
    assert 'device="${GPU_DEVICE_REQUEST:-${WF_GPU_DEVICE:-0}}"' in run
    assert "^([0-9]+(,[0-9]+)*|GPU-[0-9a-fA-F-]+)$" in run
    assert '--gpus "\\"device=$device\\""' in run
    assert "device=0" not in run


def test_self_hosted_cuda_jobs_build_with_the_daemon_driver():
    # A docker-container builder runs on the bridge network, where DNS timed out on the GPU
    # runner (v2.0.0b1 release-cuda: lookup ghcr.io: i/o timeout); the daemon driver does not.
    jobs = workflow("docker.yml")["jobs"]
    for name in ("release-cuda", "cuda"):
        setup = next(
            s for s in jobs[name]["steps"] if s.get("uses", "").startswith("docker/setup-buildx")
        )
        assert setup["with"] == {"driver": "docker"}, name


def test_manual_cuda_gpu_step_takes_a_configurable_device():
    docker = workflow("docker.yml")
    inputs = docker["on"]["workflow_dispatch"]["inputs"]
    assert inputs["gpu_device"]["type"] == "string" and inputs["gpu_device"]["default"] == ""
    step = next(s for s in docker["jobs"]["cuda"]["steps"] if "--gpus" in s.get("run", ""))
    assert step["if"] == "inputs.gpu"
    assert step["env"] == {"GPU_DEVICE_REQUEST": "${{ inputs.gpu_device || vars.WF_GPU_DEVICE }}"}
    assert_configurable_gpu(step["run"])


def test_release_default_image_is_cuda_qualified_on_a_gpu_before_publishing():
    docker = workflow("docker.yml")
    job = docker["jobs"]["release-cuda"]
    # Only for release tags, after the tag was validated, and only on the GPU runner.
    assert job["needs"] == "release"
    assert job["if"] == "github.event_name == 'push' && github.ref_type == 'tag'"
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "nvidia"]
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    steps = job["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
    runs = [step.get("run", "") for step in steps]
    order = {
        key: next(i for i, run in enumerate(runs) if needle in run)
        for key, needle in {
            "refuse": "releases are never overwritten",
            "build": "docker buildx bake cuda --load",
            "gpu": "--variant cu128 --gpu",
            "push": 'docker push "$image:$tag"',
        }.items()
    }
    assert order["refuse"] < order["build"] < order["gpu"] < order["push"]
    # GPU qualification is unconditional here, with real model inference.
    gpu = steps[order["gpu"]]
    assert "if" not in gpu and "--gpus" in gpu["run"]
    assert "--models --device cuda:0" in gpu["run"]
    # The GPU is configurable: repository variable, else the runner's environment, else 0.
    assert gpu["env"] == {"GPU_DEVICE_REQUEST": "${{ vars.WF_GPU_DEVICE }}"}
    assert_configurable_gpu(gpu["run"])
    # The default tags carry no variant suffix; the CPU variant keeps -cpu.
    assert 'tags=("$version" latest)' in runs[order["push"]]
    assert 'for tag in "${tags[@]}"; do' in runs[order["push"]]
    # A pre-release publishes its version tag only; latest stays on the last stable release.
    assert (
        'if [[ "$version" =~ (a|b|rc)[0-9]+$ ]]; then tags=("$version"); fi' in runs[order["push"]]
    )
    assert "winding-field:cuda" in runs[order["push"]]


def test_installed_suite_dumps_stacks_when_a_test_hangs():
    # Three of five main runs stalled mid-suite until the job limit with no output; a test
    # running past 10 minutes now prints every thread's stack (the result is unchanged).
    run = "\n".join(step.get("run", "") for step in workflow("docker.yml")["jobs"]["cpu"]["steps"])
    assert (
        "-o pythonpath= -o faulthandler_timeout=600 -o cache_dir=/scratch/pytest-cache /tests"
        in run
    )


def test_a_stalled_level_worker_prints_its_stack():
    # faulthandler_timeout only dumps the pytest process; the stall is in a spawned worker.
    run = "\n".join(step.get("run", "") for step in workflow("docker.yml")["jobs"]["cpu"]["steps"])
    assert "--name wf-suite" in run and "-e PYTHONFAULTHANDLER=1" in run
    assert "sleep 2700" in run and "multiprocessing[.]spawn" in run and "kill -ABRT" in run
    # The suite's own exit status is kept, and the watchdog never outlives the step.
    assert "/tests || status=$?" in run and 'kill "$watchdog"' in run and 'exit "$status"' in run
