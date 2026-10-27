"""One immutable observation request over a private inherited socketpair.

The trusted, importable loader is pickled (never accept network/untrusted input).
No Torch import occurs here. Primary/secondary numerical parity is the runner's
gate; this transport validates shapes, values, frame hashes and receipt stability.
"""

import hashlib
import io
import math
import os
import pickle
import re
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NotRequired, Self, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .field_runtime import GpuLease, ResourceUnavailable

_INIT_LIMIT = 1024 * 1024

BoundsZYX: TypeAlias = tuple[int, int, int]
ObservationPayload: TypeAlias = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], dict[str, str]]
# The loader may return tensors; only the worker converts them without importing Torch here.
ObservationLoader: TypeAlias = Callable[[BoundsZYX, BoundsZYX], tuple[Any, Any, Any, dict[str, str]]]


class GpuConfig(TypedDict):
    """Explicit secondary-device lease configuration accepted by prefetch."""

    index: int
    uuid: str
    lock_path: str | Path
    min_free_bytes: NotRequired[int]
    deadline: NotRequired[float | None]


def _bounds(lo, hi, shape):
    try:
        lo, hi, shape = tuple(lo), tuple(hi), tuple(shape)
        if (
            len(lo) != 3
            or len(hi) != 3
            or len(shape) != 3
            or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in lo + hi + shape)
            or any(not 0 <= a < b <= n for a, b, n in zip(lo, hi, shape))
        ):
            raise ValueError("invalid observation bounds/shape")
    except TypeError as error:
        raise ValueError("invalid observation bounds/shape") from error
    return tuple(map(int, lo)), tuple(map(int, hi))


def _end(timeout, deadline):
    seconds = timeout if deadline is None else min(timeout, deadline - time.time())
    if seconds <= 0:
        raise TimeoutError("prefetch deadline expired")
    return time.monotonic() + seconds


def _remaining(end):
    if end is None:
        return None
    seconds = end - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("prefetch frame timeout")
    return seconds


def _serialize(value, limit):
    class BoundedBytes(io.BytesIO):
        def write(self, data):
            if self.tell() + memoryview(data).nbytes > limit:
                raise ValueError("serialized prefetch payload exceeds byte bound")
            return super().write(data)

    with BoundedBytes() as stream:
        pickle.Pickler(stream, protocol=5).dump(value)
        return stream.getvalue()


def _send(channel, raw, end):
    header = struct.pack("!Q", len(raw)) + hashlib.sha256(raw).digest()
    for data in (header, raw):
        channel.settimeout(_remaining(end))
        channel.sendall(data)
    _remaining(end)


def _receive(channel, limit, end):
    def exact(size):
        data = bytearray(size)
        view = memoryview(data)
        while view:
            channel.settimeout(_remaining(end))
            count = channel.recv_into(view)
            if not count:
                raise ValueError("unexpected EOF in prefetch frame")
            view = view[count:]
        return data

    size = struct.unpack("!Q", exact(8))[0]
    if not 0 < size <= limit:
        raise ValueError("prefetch frame exceeds byte bound")
    digest, raw = exact(32), exact(size)
    if hashlib.sha256(raw).digest() != digest:
        raise ValueError("prefetch frame digest mismatch")
    try:
        value = pickle.loads(raw)
    except Exception as error:
        raise ValueError("malformed prefetch frame") from error
    _remaining(end)
    return value


def _payload(values, bounds):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("invalid observation payload")
    target, weights, observed, receipts = values
    shape = tuple(b - a for a, b in zip(*bounds))
    for value, expected, dtype in (
        (target, (*shape, 3), np.float32),
        (weights, shape, np.float32),
        (observed, shape, np.bool_),
    ):
        if not isinstance(value, np.ndarray) or value.shape != expected or value.dtype != dtype:
            raise ValueError("invalid observation shape or dtype")
    if not np.isfinite(target).all() or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("invalid observation finite values or positive weights")
    if not isinstance(receipts, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
        for k, v in receipts.items()
    ):
        raise ValueError("invalid observation SHA256 receipts")
    return target, weights, observed, receipts


def _reply(value):
    if not isinstance(value, tuple) or not value:
        raise ValueError("malformed worker reply")
    if value[0] == "unavailable" and len(value) == 2:
        raise ResourceUnavailable(value[1])
    if value[0] == "error" and len(value) == 3:
        error = ValueError if value[1] == "ValueError" else RuntimeError
        raise error(f"prefetch worker {value[1]}: {value[2]}")
    return value


class ObservationPrefetch:
    """Single-caller, one-pending-frame prefetch (CPU unless an explicit GPU is set).

    ``max_bytes`` bounds each serialized frame, including pickle overhead; init
    is additionally capped at 1 MiB. Array bytes alone must fit before submit.
    These are transport bounds, not whole-process memory limits for a trusted
    loader. ``timeout`` bounds complete sends/reads, including startup. ``deadline``
    is Unix seconds. GPU keys are index, uuid, lock_path, min_free_bytes, deadline
    (the last two optional); the earlier GPU/client deadline applies to both.
    Source receipt keys must be stable or include crop bounds, as appropriate.
    """

    def __init__(
        self,
        loader: ObservationLoader,
        shape: Sequence[int],
        *,
        max_bytes: int,
        timeout: float = 30.0,
        deadline: float | None = None,
        env: dict[str, str] | None = None,
        gpu: GpuConfig | None = None,
    ) -> None:
        self.shape: BoundsZYX = _bounds((0, 0, 0), shape, shape)[1]
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive integer max_bytes required")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite timeout required")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("finite deadline required")
        if not callable(loader):
            raise ValueError("importable callable loader required")
        child_env = dict(os.environ)
        if env is not None:
            if not isinstance(env, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
            ):
                raise ValueError("env must contain string keys and values")
            child_env.update(env)
        if gpu is not None:
            if (
                not isinstance(gpu, dict)
                or not {"index", "uuid", "lock_path"} <= gpu.keys()
                or gpu.keys() - {"index", "uuid", "lock_path", "min_free_bytes", "deadline"}
            ):
                raise ValueError("invalid GPU configuration keys")
            validated = GpuLease(**gpu)  # Validate without touching a device in the parent.
            gpu_deadline = validated.deadline
            if gpu_deadline is not None:
                deadline = gpu_deadline if deadline is None else min(deadline, gpu_deadline)
            gpu = GpuConfig(
                index=validated.index,
                uuid=str(validated.uuid),
                lock_path=str(validated.lock_path),
                min_free_bytes=validated.min_free_bytes,
                deadline=deadline,
            )
        child_env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else gpu["uuid"]
        # The module's own location works in both a flat checkout and site-packages.
        # Propagate import roots for the caller's trusted pickled loader as well.
        paths = [str(Path(__file__).resolve().parent), *(str(Path(p or os.getcwd()).resolve()) for p in sys.path)]
        paths.extend(child_env.get("PYTHONPATH", "").split(os.pathsep))
        child_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
        self.max_bytes, self.timeout, self.deadline = max_bytes, float(timeout), deadline
        self.pending: tuple[BoundsZYX, BoundsZYX] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._channel: socket.socket | None = None
        self._available = False
        self._receipts: dict[str, str] = {}
        init_limit = min(max_bytes, _INIT_LIMIT)
        loader_bytes = _serialize(loader, init_limit)
        initial = _serialize((loader_bytes, self.shape, max_bytes, self.timeout, deadline, gpu), init_limit)
        try:
            end = _end(self.timeout, self.deadline)
            parent, child = socket.socketpair()
            self._channel = parent
            try:
                try:
                    self._process = subprocess.Popen(
                        [sys.executable, "-m", "field_prefetch", str(child.fileno()), str(min(max_bytes, _INIT_LIMIT))],
                        pass_fds=(child.fileno(),),
                        env=child_env,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError as error:
                    raise ResourceUnavailable("prefetch backend could not start") from error
            finally:
                child.close()
            _send(parent, initial, end)
            if _reply(_receive(parent, max_bytes, end)) != ("ready",):
                raise ValueError("invalid worker startup reply")
            self._available = True
        except (TimeoutError, ResourceUnavailable):
            self.close()
        except BaseException:
            self.close()
            raise

    @property
    def available(self) -> bool:
        if self._available and self.deadline is not None and time.time() >= self.deadline:
            self.close()
        return self._available

    def submit(self, lo: ArrayLike, hi: ArrayLike) -> bool:
        bounds = _bounds(lo, hi, self.shape)
        if not self.available or self.pending is not None:
            return False
        if math.prod(b - a for a, b in zip(*bounds)) * 17 > self.max_bytes:
            raise ValueError("observation arrays exceed byte bound")
        try:
            _send(self._channel, _serialize(bounds, self.max_bytes), _end(self.timeout, self.deadline))
            self.pending = bounds
            return True
        except (TimeoutError, ResourceUnavailable):
            self.close()
            return False
        except BaseException:
            self.close()
            raise

    def receive(self, lo: ArrayLike, hi: ArrayLike) -> ObservationPayload | None:
        bounds = _bounds(lo, hi, self.shape)
        if self.pending is not None and bounds != self.pending:
            raise ValueError("requested bounds differ from pending bounds")
        if not self.available or self.pending is None:
            return None
        try:
            end = _end(self.timeout, self.deadline)
            reply = _reply(_receive(self._channel, self.max_bytes, end))
            if (
                len(reply) != 3
                or reply[0] != "result"
                or _bounds(*cast(tuple[BoundsZYX, BoundsZYX], reply[1]), self.shape) != bounds
            ):
                raise ValueError("worker returned different bounds or malformed result")
            result = _payload(reply[2], bounds)
            for key, digest in result[3].items():
                if key in self._receipts and self._receipts[key] != digest:
                    raise ValueError(f"source receipt changed: {key}")
            _remaining(end)
            self._receipts.update(result[3])
            self.pending = None
            return result
        except (TimeoutError, ResourceUnavailable):
            self.close()
            return None
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._available, self.pending = False, None
        if self._channel is not None:
            try:
                self._channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._channel.close()
            self._channel = None
        if self._process is not None:
            for action in (self._process.terminate, self._process.kill):
                if self._process.poll() is not None:
                    break
                action()
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if self._process.poll() is None:
                raise RuntimeError("owned prefetch worker did not exit")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _worker(fd, init_limit):
    with socket.socket(fileno=fd) as channel, ExitStack() as stack:
        limit, timeout, deadline = init_limit, 30.0, None
        try:
            loader_bytes, shape, limit, timeout, deadline, gpu = _receive(channel, init_limit, _end(timeout, None))
            lease = None if gpu is None else stack.enter_context(GpuLease(**gpu))
            # Unpickling can import Torch or allocate device tensors; admission comes first.
            loader = pickle.loads(loader_bytes)
            _send(channel, _serialize(("ready",), limit), _end(timeout, deadline))
            while True:
                idle_end = None if deadline is None else _end(max(0, deadline - time.time()), deadline)
                bounds = _bounds(*_receive(channel, limit, idle_end), shape)
                if lease is not None:
                    lease.recheck()
                _end(timeout, deadline)
                values = loader(*bounds)
                values = tuple(v.detach().cpu().numpy() if hasattr(v, "detach") else v for v in values[:3]) + (
                    values[3],
                )
                result = _payload(values, bounds)
                end = _end(timeout, deadline)
                _send(channel, _serialize(("result", bounds, result), limit), end)
        except Exception as error:
            retryable = isinstance(error, (ResourceUnavailable, TimeoutError))
            reply = (
                ("unavailable", str(error)[:1000]) if retryable else ("error", type(error).__name__, str(error)[:1000])
            )
            try:
                _send(channel, _serialize(reply, limit), _end(min(timeout, 1.0), None))
            except (OSError, ValueError):
                pass  # The owning parent may already have shut down its socket.


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]))
