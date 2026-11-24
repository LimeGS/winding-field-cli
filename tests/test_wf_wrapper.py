"""tools/wf installs the CLI (Docker or native) and runs it with the installed runtime."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

RECORDER = """#!{python}
import json, os, sys
args = sys.argv[1:]
if os.path.basename(sys.argv[0]) == 'docker':
    if args[:1] == ['info']:
        print(os.environ.get('FAKE_RUNTIMES', '{{"runc":{{}}}}'))
        sys.exit(0)
    if args[:2] == ['compose', 'version']:
        sys.exit(0)
with open({log!r}, 'a') as f:
    env = {{k: os.environ.get(k) for k in ('WF_HOME', 'WF_UID', 'WF_GID')}}
    f.write(json.dumps(dict(tool=os.path.basename(sys.argv[0]), args=args, cwd=os.getcwd(), env=env)) + '\\n')
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
"""


@pytest.fixture
def wf(tmp_path):
    """A checkout copy whose docker, installer and native CLI only record their calls."""
    checkout = tmp_path / "src" / "winding-field-cli"
    (checkout / "tools").mkdir(parents=True)
    shutil.copy2(ROOT / "tools/wf", checkout / "tools/wf")
    shutil.copy2(ROOT / "compose.yaml", checkout / "compose.yaml")
    log = tmp_path / "calls.jsonl"
    recorder = RECORDER.format(python=sys.executable, log=str(log))
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "docker").write_text(recorder)
    (bin / "docker").chmod(0o755)
    # wf runs the installer with bash, as the real tools/install.sh.
    (checkout / "tools/install.sh").write_text(
        f'printf \'{{"tool":"install.sh","args":[],"cwd":"%s","env":{{}}}}\\n\' "$PWD" >> {log}\n'
    )
    work = tmp_path / "work"
    work.mkdir()

    def run(*args, gpu=False, native_cli=False, **env):
        if gpu:
            smi = bin / "nvidia-smi"
            smi.write_text("#!/bin/sh\necho 'GPU 0: NVIDIA'\n")
            smi.chmod(0o755)
        if native_cli:
            cli = checkout / ".venv/bin/winding-field"
            cli.parent.mkdir(parents=True, exist_ok=True)
            cli.write_text(recorder)
            cli.chmod(0o755)
        environment = {k: v for k, v in os.environ.items() if not k.startswith("WF_")}
        environment.update(PATH=f"{bin}:/usr/bin:/bin", **env)
        done = subprocess.run(
            [str(checkout / "tools/wf"), *args],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return done, calls

    return run, checkout, work


def test_running_before_installing_explains_the_two_install_commands(wf):
    run, _, _ = wf
    done, calls = run("demo")
    assert done.returncode == 2 and "wf docker-install" in done.stderr and calls == []


def test_docker_install_builds_checks_and_later_runs_the_selected_image(wf):
    run, checkout, work = wf
    done, calls = run("docker-install")
    assert done.returncode == 0, done.stderr
    compose = [
        "compose",
        "--project-directory",
        str(checkout),
        "--file",
        str(checkout / "compose.yaml"),
    ]
    assert calls[0]["args"] == [*compose, "--profile", "cpu", "build", "cpu"]
    assert calls[1]["args"] == [*compose, "run", "--rm", "-T", "cpu", "doctor"]
    assert (checkout / ".wf-runtime").read_text().split() == ["docker", "cpu"]

    done, calls = run("run", "--source", "s3://x y", "--output", "/outputs/a b", FAKE_EXIT="7")
    assert done.returncode == 7
    assert calls[-1]["args"] == [
        *compose,
        "run",
        "--rm",
        "-T",
        "cpu",
        "run",
        "--source",
        "s3://x y",
        "--output",
        "/outputs/a b",
    ]
    env = calls[-1]["env"]
    assert env["WF_HOME"] == str(work.resolve())
    assert env["WF_UID"] == str(os.getuid()) and env["WF_GID"] == str(os.getgid())
    for name in ("data", "outputs", "models", "cache", "scratch"):
        assert (work / name).is_dir()


def _commit(checkout):
    git = ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    head = ["git", "-C", str(checkout), "rev-parse", "--short=7", "HEAD"]
    return subprocess.run(head, check=True, capture_output=True, text=True).stdout.strip()


def test_docker_install_pulls_the_cpu_image_published_for_this_commit(wf):
    run, checkout, _ = wf
    commit = _commit(checkout)
    done, calls = run("docker-install")
    assert done.returncode == 0, done.stderr
    image = f"ghcr.io/limegs/winding-field-cli:sha-{commit}-cpu"
    assert calls[0]["args"] == ["pull", image]
    assert calls[1]["args"] == ["tag", image, "winding-field:cpu"]
    assert not any("build" in call["args"] for call in calls)
    assert calls[-1]["args"][-3:] == ["-T", "cpu", "doctor"]
    assert (checkout / ".wf-runtime").read_text().split() == ["docker", "cpu"]


def test_docker_install_builds_when_nothing_matching_is_published(wf):
    run, checkout, _ = wf
    _commit(checkout)
    # The pull fails: an unpublished commit (FAKE_EXIT makes every recorded call fail).
    done, calls = run("docker-install", FAKE_EXIT="1")
    assert done.returncode != 0 and calls[0]["args"][0] == "pull"
    assert calls[1]["args"][-4:] == ["--profile", "cpu", "build", "cpu"]
    assert "no published image" in done.stdout
    # Local changes or WF_BUILD=1 never pull.
    (checkout / "compose.yaml").write_text((checkout / "compose.yaml").read_text() + "\n")
    _, calls = run("docker-install")
    assert calls[-2]["args"][-2:] == ["build", "cpu"] and calls[-3]["args"][0] != "pull"
    _commit(checkout)
    _, calls = run("docker-install", WF_BUILD="1")
    assert (
        calls[-2]["args"][-2:] == ["build", "cpu"] and calls[-3]["args"][-1] != "winding-field:cpu"
    )


def test_docker_install_selects_cuda_only_with_a_gpu_and_the_nvidia_runtime(wf):
    run, checkout, _ = wf
    run("docker-install", gpu=True)
    assert (checkout / ".wf-runtime").read_text().split() == ["docker", "cpu"]
    run("docker-install", gpu=True, FAKE_RUNTIMES='{"nvidia":{}}')
    assert (checkout / ".wf-runtime").read_text().split() == ["docker", "cuda"]
    _, calls = run("doctor", WF_VARIANT="cpu", WF_BUILD="1")
    assert calls[-1]["args"][-4:] == ["--build", "-T", "cpu", "doctor"]
    done, _ = run("doctor", WF_VARIANT="tpu")
    assert done.returncode == 2 and "WF_VARIANT" in done.stderr


def test_native_install_uses_the_host_installer_and_translates_container_paths(wf):
    run, checkout, work = wf
    done, calls = run("native-install")
    assert done.returncode == 0, done.stderr
    assert calls[-1]["tool"] == "install.sh" and calls[-1]["cwd"] == str(checkout.parent)
    assert (checkout / ".wf-runtime").read_text().split() == ["native"]
    done, _ = run("doctor")
    assert done.returncode == 2 and "native-install" in done.stderr  # no CLI yet

    done, calls = run(
        "run",
        "--lasagna-input=/data/l.json",
        "--output",
        "/outputs/r",
        "--levels",
        "-31",
        native_cli=True,
    )
    assert done.returncode == 0, done.stderr
    home = str(work.resolve())
    assert calls[-1]["tool"] == "winding-field" and calls[-1]["cwd"] == f"{home}/outputs"
    assert calls[-1]["args"] == [
        "run",
        f"--lasagna-input={home}/data/l.json",
        "--output",
        f"{home}/outputs/r",
        "--levels",
        "-31",
    ]


def test_compose_matches_bake_and_documented_mounts():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    services = compose["services"]
    bake = (ROOT / "docker-bake.hcl").read_text()
    cuda = dict(re.findall(r'(\w+) = "([^"]+)"', bake.split('target "cuda"', 1)[1]))
    assert services["cuda"]["build"]["args"] == {
        k: cuda[k] for k in services["cuda"]["build"]["args"]
    }
    assert set(services["cuda"]["build"]["args"]) == {
        "BASE_IMAGE",
        "TORCH_VARIANT",
        "NVIDIA_REQUIRE_CUDA",
    }
    assert (
        services["cpu"]["image"] == "winding-field:cpu"
        and services["cuda"]["image"] == "winding-field:cuda"
    )
    documented = re.findall(r"src=\$PWD/(\w+),dst=([^,\"]+)", (ROOT / "docs/DOCKER.md").read_text())
    for name in ("cpu", "cuda"):
        mounts = {
            (Path(v["source"]).name, v["target"], v.get("read_only", False))
            for v in services[name]["volumes"]
        }
        assert {(src, dst) for src, dst, _ in mounts} == set(documented)
        assert ("data", "/data", True) in mounts
        assert services[name]["init"] is True and services[name]["platform"] == "linux/amd64"
