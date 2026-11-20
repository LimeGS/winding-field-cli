"""tools/wf runs the CLI through compose.yaml with the workspace bound as documented."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "tools/wf"


@pytest.fixture
def wf(tmp_path):
    bin = tmp_path / "bin"
    bin.mkdir()
    log = tmp_path / "docker.jsonl"
    docker = bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['info']:\n"
        "    print(os.environ.get('FAKE_RUNTIMES', '{\"runc\":{}}'))\n"
        "    sys.exit(0)\n"
        f"with open({str(log)!r}, 'a') as f:\n"
        "    env = {k: os.environ.get(k) for k in ('WF_HOME', 'WF_UID', 'WF_GID')}\n"
        "    f.write(json.dumps(dict(args=args, env=env)) + '\\n')\n"
        "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"
    )
    docker.chmod(0o755)

    def run(*args, gpu=False, cwd=tmp_path, **env):
        if gpu:
            smi = bin / "nvidia-smi"
            smi.write_text("#!/bin/sh\necho 'GPU 0: NVIDIA'\n")
            smi.chmod(0o755)
        environment = dict(os.environ, PATH=f"{bin}:/usr/bin:/bin", **env)
        done = subprocess.run(
            [str(WF), *args], cwd=cwd, env=environment, capture_output=True, text=True, check=False
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return done, calls

    return run


def test_wf_runs_the_cpu_service_with_the_workspace_and_arguments(wf, tmp_path):
    done, calls = wf("run", "--source", "s3://x y", "--output", "/outputs/a b")
    assert done.returncode == 0, done.stderr
    args = calls[-1]["args"]
    assert args[:5] == [
        "compose",
        "--project-directory",
        str(ROOT),
        "--file",
        str(ROOT / "compose.yaml"),
    ]
    assert args[5:] == [
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
    assert env["WF_HOME"] == str(tmp_path.resolve())
    assert env["WF_UID"] == str(os.getuid()) and env["WF_GID"] == str(os.getgid())
    for name in ("data", "outputs", "models", "cache", "scratch"):
        assert (tmp_path / name).is_dir()


def test_wf_selects_cuda_only_with_a_gpu_and_the_nvidia_runtime(wf):
    _, calls = wf("doctor", gpu=True)
    assert "cpu" in calls[-1]["args"]  # GPU visible but Docker lacks the NVIDIA runtime
    _, calls = wf("doctor", gpu=True, FAKE_RUNTIMES='{"nvidia":{},"runc":{}}')
    assert "cuda" in calls[-1]["args"] and "cpu" not in calls[-1]["args"]
    _, calls = wf("doctor", gpu=True, FAKE_RUNTIMES='{"nvidia":{}}', WF_VARIANT="cpu")
    assert "cpu" in calls[-1]["args"]


def test_wf_options_exit_status_and_invalid_variant(wf, tmp_path):
    home = tmp_path / "workspace"
    home.mkdir()
    done, calls = wf("demo", WF_HOME=str(home), WF_BUILD="1", FAKE_EXIT="7")
    assert done.returncode == 7
    assert "--build" in calls[-1]["args"] and calls[-1]["env"]["WF_HOME"] == str(home.resolve())
    done, _ = wf("doctor", WF_VARIANT="tpu")
    assert done.returncode == 2 and "WF_VARIANT" in done.stderr


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
