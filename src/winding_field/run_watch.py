#!/usr/bin/env python3
"""Own one job, heartbeat it to disk, and record failure/timeout/resource limits."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def supervise(command, output, timeout=1800, reserve_bytes=10 * 1024**3, interval=2):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    started = time.time()
    state = dict(
        state="starting", command=list(command), started_utc=started, timeout_s=timeout, supervisor_pid=os.getpid()
    )

    def record():
        state.update(
            updated_utc=time.time(),
            elapsed_s=time.time() - started,
            free_bytes=shutil.disk_usage(output).free,
            log_bytes=(output / "stdout.log").stat().st_size if (output / "stdout.log").exists() else 0,
        )
        temp = output / "status.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(output / "status.json")

    record()
    if state["free_bytes"] < reserve_bytes:
        state.update(state="resource_blocked", returncode=None)
        record()
        return state
    with (output / "stdout.log").open("xb") as log:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        state.update(state="running", pid=child.pid)
        try:
            while child.poll() is None:
                record()
                cause = (
                    "timed_out"
                    if state["elapsed_s"] > timeout
                    else "resource_blocked"
                    if state["free_bytes"] < reserve_bytes
                    else None
                )
                if cause:
                    state["state"] = cause
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(interval)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state["state"] = "cancelled"
            state["returncode"] = child.returncode
            record()
            raise
        state["returncode"] = child.wait()
    if state["state"] == "running":
        state["state"] = "completed" if state["returncode"] == 0 else "failed"
    record()
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--reserve-gib", type=float, default=10)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd or a.timeout <= 0 or a.reserve_gib < 0:
        p.error("require command, positive timeout and nonnegative reserve")
    status = supervise(cmd, a.output, a.timeout, a.reserve_gib * 1024**3)
    print(json.dumps(status), flush=True)
    return 0 if status["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
