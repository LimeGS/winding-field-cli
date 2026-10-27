"""Explicit POSIX resource leases and source-bound, single-child supervision.

No numerical imports or implicit host memory/GPU policies. Deadlines are Unix
seconds; elapsed I/O and shutdown budgets use a monotonic clock.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Self

_JSON_LIMIT = 1024 * 1024
_TERMINAL = {"CHECKPOINTED_STOP", "SWEEP_BUDGET_REACHED"}


class ResourceUnavailable(RuntimeError):
    """An explicitly retryable capacity, occupancy, backend or deadline failure."""


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _seconds(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise ResourceUnavailable("resource deadline expired")


def _same_file(fd, path):
    actual, expected = os.fstat(fd), Path(path).lstat()
    if not stat.S_ISREG(expected.st_mode) or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("lease file identity differs")


class GpuLease:
    """Exclusive advisory FD lease; the caller must bind CUDA_VISIBLE_DEVICES.

    ``inherited_fd`` is duplicated, never unlocked or consumed. Closing either
    owner leaves the lease held until the final inherited/duplicated FD closes.
    Call ``recheck()`` immediately before work, including after slow preparation.
    NVIDIA queries are bounded and exclude only this process's own GPU context.
    """

    def __init__(
        self,
        index: int,
        uuid: str,
        lock_path: str | Path,
        *,
        min_free_bytes: int = 0,
        deadline: float | None = None,
        inherited_fd: int | None = None,
    ) -> None:
        self.index: int = _nonnegative(index, "index")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or any(c.isspace() for c in uuid):
            raise ValueError("explicit GPU UUID required")
        self.uuid, self.lock_path = uuid, Path(lock_path)
        if not self.lock_path.is_absolute():
            raise ValueError("absolute GPU lock_path required")
        self.min_free_bytes: int = _nonnegative(min_free_bytes, "min_free_bytes")
        self.deadline: float | None = None if deadline is None else _seconds(deadline, "deadline")
        self.inherited_fd: int | None = None if inherited_fd is None else _nonnegative(inherited_fd, "inherited_fd")
        self._fd: int | None = None

    def __enter__(self) -> Self:
        if self._fd is not None:
            raise ValueError("lease already entered")
        _deadline(self.deadline)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != self.uuid:
            raise ValueError("GPU must be UUID-bound before importing CUDA")
        self._fd = (
            os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            if self.inherited_fd is None
            else os.dup(self.inherited_fd)
        )
        try:
            _same_file(self._fd, self.lock_path)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ResourceUnavailable("GPU lease is busy") from error
            self.recheck()
            return self
        except BaseException:
            self.close()
            raise

    def fileno(self) -> int:
        if self._fd is None:
            raise ValueError("GPU lease is closed")
        return self._fd

    def recheck(self) -> None:
        _deadline(self.deadline)
        _same_file(self.fileno(), self.lock_path)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != self.uuid:
            raise ValueError("GPU environment identity changed")

        def query(fields, *extra):
            _deadline(self.deadline)
            timeout = 5.0 if self.deadline is None else min(5.0, self.deadline - time.time())
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", fields, "--format=csv,noheader,nounits", *extra],
                    text=True,
                    timeout=max(0.001, timeout),
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ResourceUnavailable("GPU inventory unavailable") from error
            _deadline(self.deadline)
            return result.strip()

        row = query("--query-gpu=index,uuid,memory.free", f"--id={self.index}").split(",")
        if len(row) != 3:
            raise ResourceUnavailable("GPU memory/identity inventory unavailable")
        index, uuid, free = (v.strip() for v in row)
        if index != str(self.index) or uuid != self.uuid:
            raise ValueError("physical GPU identity differs")
        try:
            enough = int(free) * 1024**2 >= self.min_free_bytes
        except ValueError as error:
            raise ResourceUnavailable("GPU free memory unknown") from error
        if not enough:
            raise ResourceUnavailable("GPU free memory unavailable")
        for line in query("--query-compute-apps=gpu_uuid,pid").splitlines():
            try:
                used, pid = (v.strip() for v in line.split(","))
                foreign = int(pid) != os.getpid()
            except ValueError as error:
                raise ResourceUnavailable("GPU occupancy unknown") from error
            if used == self.uuid and foreign:
                raise ResourceUnavailable("foreign compute process on GPU")
        _deadline(self.deadline)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # LOCK_UN would also unlock inherited owners.
            self._fd = None

    def __exit__(self, *exc: object) -> None:
        self.close()


def check_resources(
    path: str | Path,
    *,
    reserve_bytes: int = 0,
    memory_available: Callable[[], int | None] | None = None,
    required_memory_bytes: int = 0,
) -> dict[str, int | None]:
    """Check disk reserve and caller-supplied effective (e.g. cgroup) RAM bytes.

    No host-wide memory estimate substitutes for a missing effective-memory
    callback. Return the measured byte counts; this is admission, not reservation.
    """
    _nonnegative(reserve_bytes, "reserve_bytes")
    _nonnegative(required_memory_bytes, "required_memory_bytes")
    free = shutil.disk_usage(path).free
    if free < reserve_bytes:
        raise ResourceUnavailable("disk reserve unavailable")
    available = memory_available() if memory_available is not None else None
    if available is not None:
        _nonnegative(available, "memory_available result")
    if required_memory_bytes and (available is None or available < required_memory_bytes):
        raise ResourceUnavailable("effective memory unavailable or unknown")
    return {"disk_free_bytes": free, "memory_available_bytes": available}


def _json_bytes(value, limit=_JSON_LIMIT):
    raw = json.dumps(value, allow_nan=False, sort_keys=True).encode()
    if len(raw) > limit:
        raise ValueError("runtime JSON exceeds byte bound")
    return raw


def _atomic_json(path, value):
    """Publish file data, rename, then parent directory; durability errors propagate."""
    raw = _json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this unpublished temporary belongs to us.


def _read_json(path, limit=_JSON_LIMIT):
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
    except FileNotFoundError:
        return None
    if len(raw) > limit:
        raise ValueError("runtime JSON exceeds byte bound")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    _json_bytes(value, limit)
    return value


def _alive(pid):
    if pid is None:
        return False
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid recorded PID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _status(path):
    value = _read_json(path)
    if value is not None and (
        type(value.get("pid")) is not int
        or value["pid"] <= 0
        or not isinstance(value.get("status"), str)
        or not value["status"]
    ):
        raise ValueError("invalid worker status receipt")
    return value


def _checkpoint(path):
    """Validate the resume envelope; the runner validates identity/data hashes.

    Required: nonempty identity object and progress with integer sweep and
    next_block >= 0. Extra store/runner metadata is preserved, never rewritten.
    """
    # Chunk inventories are larger than small process-status receipts.
    value = _read_json(path, limit=64 * 1024**2)
    if value is None:
        return False
    progress = value.get("progress")
    if not isinstance(value.get("identity"), dict) or not value["identity"] or not isinstance(progress, dict):
        raise ValueError("invalid checkpoint identity/progress")
    for key in ("sweep", "next_block"):
        _nonnegative(progress.get(key), f"checkpoint {key}")
    return True


def _stop_child(child, run, grace):
    (run / "STOP").touch(exist_ok=True)
    try:
        child.wait(timeout=grace)
        return False
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.getpgid(child.pid) != child.pid or os.getsid(child.pid) != child.pid:
            raise RuntimeError("owned worker lost its private process group")
    except ProcessLookupError:
        # An exited, unreaped start_new_session leader still reserves our group ID.
        if child.returncode is not None:
            return False
    # Do not reap the leader until escalation: its reserved PID prevents group-ID reuse.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        if child.poll() is None:
            raise
        return False
    except PermissionError:
        if sys.platform != "darwin" or child.poll() is None:
            raise
        return False
    time.sleep(1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin reports EPERM for groups whose remaining members are zombies.
        if sys.platform != "darwin" or child.poll() is None:
            raise
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("owned child did not exit after bounded shutdown") from error
    return True


def run_supervised(
    command: Sequence[str],
    resume_command: Sequence[str],
    run_dir: str | Path,
    service_dir: str | Path,
    *,
    source_sha256: dict[str, str],
    retry_seconds: float = 30.0,
    stop_grace_seconds: float = 60.0,
    resume: bool = False,
) -> int:
    """Run absolute executable argv; retry only exit75 + same-child resource status.

    The service owns SERVICE.json, append-only service.log/worker.log and lifetime
    locks in both directories. Existing run data needs a valid CHECKPOINT.json.
    Commands must tolerate the run directory containing .supervisor.lock. Source
    inventory pins every caller-selected code/config input, not arbitrary argv
    literals. Invalid configuration/previous ownership raises before launch;
    operational or integrity failures persist FAILED and return 1. STOP returns
    0 (STOPPED if forced, never a claim that a checkpoint was written); a signal
    returns 130. Forced shutdown signals the verified private process group of
    the live Popen child, including its prefetch descendants; no recorded PID or
    unrelated process group is signalled, and no run data is removed.
    Explicit ``resume=True`` can hand off a dead stopped worker with a checkpoint;
    it never makes a scientific failure retryable. Default terminal idempotence
    is preserved, and live owners or unverified resource receipts remain errors.
    """

    def argv(value):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
            raise ValueError("command must be a nonempty argv sequence")
        if any(not isinstance(v, str) or not v or "\0" in v for v in value) or not Path(value[0]).is_absolute():
            raise ValueError("argv requires an absolute executable and nonempty strings")
        return list(value)

    command, resume_command = argv(command), argv(resume_command)
    run, service = Path(run_dir), Path(service_dir)
    if not run.is_absolute() or not service.is_absolute():
        raise ValueError("absolute run/service paths required")
    run, service = run.resolve(), service.resolve()
    if run == service or run in service.parents or service in run.parents:
        raise ValueError("run and service must be separate directories")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if resume and not _checkpoint(run / "CHECKPOINT.json"):
        raise ValueError("explicit resume requires an existing checkpoint")
    retry_seconds = _seconds(retry_seconds, "retry_seconds", zero=True)
    stop_grace_seconds = _seconds(stop_grace_seconds, "stop_grace_seconds", zero=True)
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise ValueError("nonempty source SHA256 inventory required")
    sources = dict(source_sha256)
    for path, digest in sources.items():
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("source inventory requires absolute paths and SHA256 hex")
    binding = dict(
        command=command,
        resume_command=resume_command,
        run_dir=str(run),
        service_dir=str(service),
        source_sha256=sources,
        retry_seconds=retry_seconds,
        stop_grace_seconds=stop_grace_seconds,
    )
    _json_bytes(binding)
    service.mkdir(exist_ok=True)
    metadata = service / "SERVICE.json"
    if not metadata.exists() and any(p.name not in {"STOP", "SERVICE.lock"} for p in service.iterdir()):
        raise ValueError("unverified service ownership")
    with ExitStack() as stack:
        lock = stack.enter_context((service / "SERVICE.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("service already active") from error
        previous = _read_json(metadata)
        if previous is not None:
            if previous.get("schema") != "winding-supervisor-v1" or previous.get("binding") != binding:
                raise ValueError("service ownership/config binding differs")
            if _alive(previous.get("supervisor_pid")) or _alive(previous.get("worker_pid")):
                raise ValueError("recorded owner is still alive")
            if previous.get("status") in {"STARTING", "RUNNING"}:
                raise ValueError("ambiguous interrupted launch ownership")
        state = dict(
            schema="winding-supervisor-v1",
            binding=binding,
            supervisor_pid=os.getpid(),
            worker_pid=None,
            last_worker_pid=None,
        )

        def emit(status, **values):
            state.update(values, status=status, updated_epoch=time.time())
            _atomic_json(metadata, state)
            with (service / "service.log").open("ab") as log:
                log.write(_json_bytes(state) + b"\n")
                log.flush()
                os.fsync(log.fileno())

        def verify():
            if _read_json(metadata) != state:
                raise ValueError("service ownership/config changed")
            _same_file(lock.fileno(), service / "SERVICE.lock")
            if run_lock is not None:
                _same_file(run_lock.fileno(), run / ".supervisor.lock")
            for path, wanted in sources.items():
                with open(path, "rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != wanted:
                    raise ValueError(f"source hash changed: {path}")

        interrupted = False

        def interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True  # Popen must return before its owned PID is acted on.

        handlers = {}
        if threading.current_thread() is threading.main_thread():
            handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}

        def stopped():
            return interrupted or (run / "STOP").exists() or (service / "STOP").exists()

        child, run_lock = None, None
        try:
            emit("READY")
            verify()
            receipt = _status(run / "STATUS.json")
            if receipt is not None and _alive(receipt["pid"]):
                raise ValueError("run has a live owner")
            if stopped():
                emit("STOPPED")
                return 130 if interrupted else 0
            if previous is not None and previous.get("status") == "FAILED":
                raise ValueError("previous service failure is terminal")
            if previous is not None and previous.get("status") == "RETRYING" and receipt is None:
                raise ValueError("missing previous resource receipt")
            if previous is not None and previous.get("status") in _TERMINAL:
                if (
                    not receipt
                    or receipt["pid"] != previous.get("last_worker_pid")
                    or receipt["status"] != previous["status"]
                ):
                    raise ValueError("stale terminal receipt")
                if not resume:
                    emit(receipt["status"], last_worker_pid=receipt["pid"])
                    return 0
            owned_stop = bool(
                previous is not None
                and previous.get("status") == "STOPPED"
                and receipt is not None
                and receipt["pid"] == previous.get("last_worker_pid")
                and receipt["status"] in {"RUNNING", "RESOURCE_UNAVAILABLE", *_TERMINAL}
            )
            handoff = bool(
                resume
                and receipt is not None
                and (
                    owned_stop
                    or (receipt["status"] in _TERMINAL and (previous is None or previous.get("status") in _TERMINAL))
                )
            )
            if receipt is not None and not handoff and receipt["status"] != "RESOURCE_UNAVAILABLE":
                raise ValueError("terminal or stale previous worker status")
            if (
                receipt is not None
                and not handoff
                and (
                    not previous or previous.get("last_worker_pid") != receipt["pid"] or previous.get("exit_code") != 75
                )
            ):
                raise ValueError("unverified previous resource receipt")
            if (
                run.exists()
                and any(p.name not in {".supervisor.lock", "STOP"} for p in run.iterdir())
                and not _checkpoint(run / "CHECKPOINT.json")
            ):
                raise ValueError("unowned run without checkpoint")
            run.mkdir(exist_ok=True)
            run_lock = stack.enter_context((run / ".supervisor.lock").open("a"))
            try:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("run already supervised") from error
            while True:
                verify()
                resume = _checkpoint(run / "CHECKPOINT.json")
                if stopped():
                    emit("STOPPED")
                    return 130 if interrupted else 0
                old_receipt = _status(run / "STATUS.json")
                if old_receipt != receipt:
                    raise ValueError("worker status ownership changed before launch")
                emit("STARTING", worker_pid=None)
                verify()
                if stopped():
                    emit("STOPPED")
                    return 130 if interrupted else 0
                with (service / "worker.log").open("ab") as log:
                    child = subprocess.Popen(
                        resume_command if resume else command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    emit("RUNNING", worker_pid=child.pid, last_worker_pid=child.pid)
                    forced = False
                    while child.poll() is None:
                        if stopped():
                            forced = _stop_child(child, run, stop_grace_seconds)
                            break
                        try:
                            child.wait(timeout=0.05)
                        except subprocess.TimeoutExpired:
                            pass
                    os.fsync(log.fileno())
                verify()
                receipt = _status(run / "STATUS.json")
                code = child.returncode
                if forced:
                    if receipt is not None and receipt != old_receipt:
                        if receipt["pid"] != child.pid:
                            raise ValueError("stale status receipt during STOP")
                        if receipt["status"] == "FAILED":
                            raise RuntimeError(f"worker failed during STOP: {receipt}")
                    emit("STOPPED", worker_pid=None, exit_code=code)
                    return 130 if interrupted else 0
                if receipt is None or receipt == old_receipt or receipt["pid"] != child.pid:
                    raise ValueError("missing or stale same-child status receipt")
                if code == 0 and receipt["status"] in _TERMINAL:
                    emit(receipt["status"], worker_pid=None, exit_code=code)
                    return 130 if interrupted else 0
                if code != 75 or receipt["status"] != "RESOURCE_UNAVAILABLE":
                    raise RuntimeError(f"terminal worker exit {code}: {receipt}")
                _checkpoint(run / "CHECKPOINT.json")
                emit("RETRYING", worker_pid=None, exit_code=code, failure=receipt)
                until = time.monotonic() + retry_seconds
                while not stopped() and time.monotonic() < until:
                    time.sleep(min(0.05, max(0, until - time.monotonic())))
        except Exception as error:
            emit("FAILED", error=str(error))
            return 1
        finally:
            try:
                if child is not None and child.poll() is None:
                    _stop_child(child, run, stop_grace_seconds)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                emit(
                    state["status"],
                    supervisor_pid=None,
                    worker_pid=child.pid if child is not None and child.poll() is None else None,
                )
