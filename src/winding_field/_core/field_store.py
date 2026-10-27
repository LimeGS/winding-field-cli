"""Single-writer, recoverable ZYX fields with bounded NPZ reader retention.

Chunks encode float32 deltas about a float64 mean. This is quantized storage,
not a convergence certificate or a lossless float64 export. Missing chunks are
numerical zero with no observations; an explicit boolean mask survives even a
zero-valued commit. Progress is caller-owned JSON (including cumulative sweep
statistics), published atomically with the owned core.

The directory and its immutable generations must remain exclusively owned by
this store, including while cached. Restore checks every referenced payload;
uncached reads hash and decode the same bytes. Cache budgets bound retained
arrays plus an overhead charge, not returned arrays or whole-process memory.
Reader batches contain at most ``workers`` chunks; defaults retain nothing and
use one reader. Filesystem durability requires local POSIX flock/rename/fsync
semantics. After a publication or storage-admission error, close and reopen.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import stat
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Key = tuple[int, int, int]
_Chunk = tuple[NDArray[np.float32], float, NDArray[np.uint8]]


class StorageBudgetError(RuntimeError):
    """Terminal storage admission failure; close the store before retrying."""


def _integer(value: Any, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("expected an integer within bounds")
    return int(value)


def _triple(value: Sequence[int], minimum: int = 0) -> _Key:
    try:
        parts = tuple(value)
    except TypeError as error:
        raise ValueError("expected three integer ZYX coordinates") from error
    if len(parts) != 3:
        raise ValueError("expected three integer ZYX coordinates")
    return _integer(parts[0], minimum), _integer(parts[1], minimum), _integer(parts[2], minimum)


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _json_load(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_unique_object, parse_float=_finite_float, parse_constant=_finite_float)


def _json_bytes(value: Any) -> bytes:
    try:
        raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        _json_load(raw)
        return raw
    except (TypeError, OverflowError) as error:
        raise ValueError("expected finite JSON data") from error


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    """Publish finite JSON via a unique same-directory temporary and two fsyncs.

    The parent must exist and the target cannot be a symlink. An exception after
    replace may mean publication happened: callers must stop and inspect/reopen.
    Failed temporaries are left for inspection, never silently pruned.
    """
    path = Path(path)
    raw = _json_bytes(value)
    if path.is_symlink():
        raise ValueError("JSON target cannot be a symlink")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    _write_file(temporary, raw)
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _read_file(path: Path, expected_bytes: int | None = None) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("checkpoint member must be a regular file")
        if expected_bytes is not None and metadata.st_size != expected_bytes:
            raise ValueError("checkpoint chunk size differs")
        raw = stream.read() if expected_bytes is None else stream.read(expected_bytes + 1)
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError("checkpoint chunk size changed")
    return raw


def _array(raw: bytes, shapes: tuple[tuple[int, ...], ...], dtype: Any) -> NDArray[Any]:
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, actual_dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported checkpoint NPY version")
    if shape not in shapes or actual_dtype != np.dtype(dtype):
        raise ValueError("checkpoint array shape/dtype differs")
    data = stream.read()
    if len(data) != math.prod(shape) * actual_dtype.itemsize:
        raise ValueError("checkpoint array byte count differs")
    return np.frombuffer(data, dtype=actual_dtype).reshape(shape, order="F" if fortran else "C")


def _decode(raw: bytes, shape: _Key) -> _Chunk:
    count = math.prod(shape)
    specifications = {
        "delta.npy": ((shape,), np.float32, count * 4),
        "offset.npy": (((),), np.float64, 8),
        "observed.npy": (((0,), ((count + 7) // 8,)), np.uint8, (count + 7) // 8),
    }
    arrays = {}
    try:
        with ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 3 or {member.filename for member in members} != set(specifications):
                raise ValueError("checkpoint NPZ members differ")
            for member in members:
                shapes, dtype, size = specifications[member.filename]
                if member.file_size > size + 10016 or member.flag_bits & 1:
                    raise ValueError("oversized or encrypted checkpoint member")
                arrays[member.filename] = _array(archive.read(member), shapes, dtype)
    except (BadZipFile, EOFError, OSError, RuntimeError) as error:
        raise ValueError("invalid checkpoint NPZ") from error
    delta, offset, observed = arrays["delta.npy"], float(arrays["offset.npy"]), arrays["observed.npy"]
    with np.errstate(over="ignore", invalid="ignore"):
        finite = np.isfinite(delta.astype(np.float64) + offset).all()
    if not finite or not math.isfinite(offset):
        raise ValueError("nonfinite decoded checkpoint")
    if observed.size and count % 8 and int(observed[-1]) & ((1 << (8 - count % 8)) - 1):
        raise ValueError("nonzero observation padding bits")
    return delta, offset, observed


class BlockStore:
    """Lifetime writer lease over ``root/CHECKPOINT.json`` and immutable chunks.

    ``key`` is a three-integer chunk index; read boxes are nonempty half-open
    integer ZYX bounds. ``nodes`` preserves arbitrary integer-array shape/order
    and duplicates. ``observed=None`` means unsupported, not observed zero.

    ``max_bytes`` includes existing files (also orphans), old generations, new
    payloads and replacement manifests at publication peak. ``reserve_bytes``
    protects free filesystem bytes at admission; it cannot reserve disk against
    unrelated writers. Both are logical byte budgets, not filesystem block usage.
    ``stored_bytes`` in a commit receipt counts only manifest-referenced payloads.
    ``manifest`` returns a detached snapshot; no caller may mutate store files.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        shape: Sequence[int],
        chunk: int,
        identity: Any,
        resume: bool = False,
        *,
        cache_bytes: int = 0,
        workers: int = 1,
        max_bytes: int | None = None,
        reserve_bytes: int = 0,
    ) -> None:
        self.root = Path(root)
        self.shape = _triple(shape, 1)
        self.chunk = _integer(chunk, 1)
        if max(math.prod(self.shape), self.chunk) > np.iinfo(np.intp).max:
            raise ValueError("grid exceeds integer indexing capacity")
        self.max_cache_bytes = _integer(cache_bytes)
        self.workers = _integer(workers, 1)
        self.max_bytes = None if max_bytes is None else _integer(max_bytes)
        self.reserve_bytes = _integer(reserve_bytes)
        identity = _json_load(_json_bytes(identity))
        if not isinstance(resume, bool) or self.root.is_symlink():
            raise ValueError("invalid resume flag or symlink store root")
        self.stats = dict(hits=0, loads=0, zeros=0, evictions=0, peak_cache_bytes=0)
        self._cache: OrderedDict[str, _Chunk] = OrderedDict()
        self._cache_nbytes = 0
        self._operation, self._cache_lock = RLock(), Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._lease: int | None = None
        self._closed = self._poisoned = False
        if not resume:
            try:
                self.root.mkdir(exist_ok=False)
            except FileExistsError:
                # The owned supervisor creates only its lease before the first worker.
                if not self.root.is_dir() or {p.name for p in self.root.iterdir()} != {".supervisor.lock"}:
                    raise
                marker = (self.root / ".supervisor.lock").lstat()
                if not stat.S_ISREG(marker.st_mode) or marker.st_size:
                    raise
            _sync_directory(self.root.parent.resolve())
        metadata = self.root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("store root must be a directory")
        self._root_identity = (metadata.st_dev, metadata.st_ino)
        try:
            self._lease = os.open(
                self.root / ".writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
            )
            if not stat.S_ISREG(os.fstat(self._lease).st_mode):
                raise ValueError("writer lease must be a regular file")
            try:
                fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("field store already has a writer") from error
            if resume:
                self._manifest = _json_load(_read_file(self.root / "CHECKPOINT.json"))
                self._validate_manifest(identity)
                for label, record in self._manifest["chunks"].items():
                    key = self._label_key(label)
                    _decode(self._payload(record), self._shape(key))
            else:
                self._manifest = dict(
                    identity=identity,
                    shape=list(self.shape),
                    chunk=self.chunk,
                    chunks={},
                    progress={},
                    initialization="numerical zero, not an observed value",
                )
                self._publish(self._manifest)
        except BaseException:
            self.close()
            raise

    @property
    def manifest(self) -> dict[str, Any]:
        with self._operation:
            return deepcopy(self._manifest)

    @property
    def cache_nbytes(self) -> int:
        return self._cache_nbytes

    def _ensure_open(self) -> None:
        if self._closed or self._poisoned:
            raise RuntimeError("field store is closed or failed; close and resume from disk")
        root = self.root.lstat()
        lease = (self.root / ".writer.lock").lstat()
        assert self._lease is not None
        owned = os.fstat(self._lease)
        if (root.st_dev, root.st_ino) != self._root_identity or (lease.st_dev, lease.st_ino) != (
            owned.st_dev,
            owned.st_ino,
        ):
            self._poisoned = True
            raise RuntimeError("field store ownership changed")

    def _key(self, key: Sequence[int]) -> _Key:
        key = _triple(key)
        if any(k * self.chunk >= n for k, n in zip(key, self.shape)):
            raise ValueError("chunk key outside field")
        return key

    def _label_key(self, label: str) -> _Key:
        if (
            not isinstance(label, str)
            or re.fullmatch(r"(?:0|[1-9][0-9]*)-(?:0|[1-9][0-9]*)-(?:0|[1-9][0-9]*)", label) is None
        ):
            raise ValueError("invalid checkpoint chunk label")
        return self._key(tuple(int(v) for v in label.split("-")))

    def _shape(self, key: _Key) -> _Key:
        z, y, x = (min(self.chunk, n - k * self.chunk) for k, n in zip(key, self.shape))
        return z, y, x

    def _validate_manifest(self, identity: Any) -> None:
        manifest = self._manifest
        required = {"identity", "shape", "chunk", "chunks", "progress"}
        if not isinstance(manifest, dict) or not required <= manifest.keys():
            raise ValueError("malformed checkpoint manifest")
        if (
            _triple(manifest["shape"], 1) != self.shape
            or _integer(manifest["chunk"], 1) != self.chunk
            or _json_bytes(manifest["identity"]) != _json_bytes(identity)
        ):
            raise ValueError("checkpoint identity/shape/chunk differs")
        if not isinstance(manifest["chunks"], dict) or not isinstance(manifest["progress"], dict):
            raise ValueError("malformed checkpoint chunks/progress")
        for label, record in manifest["chunks"].items():
            key = self._label_key(label)
            if not isinstance(record, dict) or not {"path", "sha256", "bytes", "quantization_max"} <= record.keys():
                raise ValueError("malformed checkpoint chunk record")
            if (
                not isinstance(record["path"], str)
                or re.fullmatch(re.escape(label) + r"\.g[0-9a-f]+\.npz", record["path"]) is None
            ):
                raise ValueError("invalid checkpoint generation path")
            if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
                raise ValueError("invalid checkpoint digest")
            if _integer(record["bytes"], 1) > math.prod(self._shape(key)) * 5 + 65536:
                raise ValueError("oversized checkpoint payload")
            error = record["quantization_max"]
            if type(error) not in (float, int) or not math.isfinite(error) or error < 0:
                raise ValueError("invalid checkpoint quantization error")

    def _payload(self, record: dict[str, Any]) -> bytes:
        raw = _read_file(self.root / record["path"], record["bytes"])
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("checkpoint chunk hash differs")
        return raw

    @staticmethod
    def _charge(value: _Chunk) -> int:
        return value[0].nbytes + value[2].nbytes + 512

    def _load(self, key: _Key) -> _Chunk | None:
        record = self._manifest["chunks"].get("-".join(map(str, key)))
        with self._cache_lock:
            if record is None:
                self.stats["zeros"] += 1
                return None
            name = record["path"]
            if name in self._cache:
                self._cache.move_to_end(name)
                self.stats["hits"] += 1
                return self._cache[name]
        value = _decode(self._payload(record), self._shape(key))
        charge = self._charge(value)
        with self._cache_lock:
            self.stats["loads"] += 1
            if charge <= self.max_cache_bytes:
                while self._cache_nbytes + charge > self.max_cache_bytes:
                    _, retired = self._cache.popitem(last=False)
                    self._cache_nbytes -= self._charge(retired)
                    self.stats["evictions"] += 1
                self._cache[name] = value
                self._cache_nbytes += charge
                self.stats["peak_cache_bytes"] = max(self.stats["peak_cache_bytes"], self._cache_nbytes)
        return value

    def _chunks(self, keys: Iterable[_Key]) -> Iterator[tuple[_Key, _Chunk | None]]:
        if self.workers == 1:
            for key in keys:
                yield key, self._load(key)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="field-read")
        iterator = iter(keys)
        while batch := list(itertools.islice(iterator, self.workers)):
            futures = [self._pool.submit(self._load, key) for key in batch]
            try:
                values = [future.result() for future in futures]
            finally:
                # An integrity failure must also drain the other readers before publication.
                wait(futures)
            yield from zip(batch, values)

    def _read(self, lo: Sequence[int], hi: Sequence[int], observed: bool) -> NDArray[Any]:
        with self._operation:
            self._ensure_open()
            low, high = np.array(_triple(lo)), np.array(_triple(hi))
            if (high > self.shape).any() or (high <= low).any():
                raise ValueError("field read outside grid")
            result = np.zeros(tuple(high - low), dtype=bool if observed else np.float64)
            keys = (
                (z, y, x)
                for z, y, x in itertools.product(
                    *(range(int(a), int(b) + 1) for a, b in zip(low // self.chunk, (high - 1) // self.chunk))
                )
            )
            for key, value in self._chunks(keys):
                if value is None:
                    continue
                origin = np.asarray(key) * self.chunk
                a, b = np.maximum(low, origin), np.minimum(high, origin + np.asarray(self._shape(key)))
                destination = tuple(slice(int(x), int(y)) for x, y in zip(a - low, b - low))
                source = tuple(slice(int(x), int(y)) for x, y in zip(a - origin, b - origin))
                delta, offset, mask = value
                if observed:
                    if mask.size:
                        result[destination] = np.unpackbits(mask, count=delta.size).reshape(delta.shape)[source]
                else:
                    result[destination] = delta[source]
                    result[destination] += offset
            return result

    def read(self, lo: Sequence[int], hi: Sequence[int]) -> NDArray[np.float64]:
        """Read a nonempty half-open integer ZYX box as a new float64 array."""
        return self._read(lo, hi, False)

    def read_observed(self, lo: Sequence[int], hi: Sequence[int]) -> NDArray[np.bool_]:
        """Read explicit observation support, including supported numerical zeros."""
        return self._read(lo, hi, True)

    def nodes(self, flat: ArrayLike) -> NDArray[np.float64]:
        """Read flat ZYX integer indices, preserving duplicates and input shape."""
        with self._operation:
            self._ensure_open()
            if not isinstance(flat, np.ndarray):
                for index in np.asarray(flat, dtype=object).flat:
                    _integer(index)
            indices = np.asarray(flat)
            if indices.size == 0 and not isinstance(flat, np.ndarray):
                indices = indices.astype(np.int64)
            if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= math.prod(self.shape)).any():
                raise ValueError("node indices must be integers inside field")
            result = np.zeros(indices.shape, np.float64)
            if not indices.size:
                return result
            coords = np.array(np.unravel_index(indices.astype(np.intp).ravel(), self.shape)).T
            grid_shape = tuple((n + self.chunk - 1) // self.chunk for n in self.shape)
            labels = np.ravel_multi_index(tuple((coords // self.chunk).T), grid_shape)
            order = np.argsort(labels, kind="stable")
            groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
            keys = (_triple(np.unravel_index(labels[group[0]], grid_shape)) for group in groups)
            for group, (_, value) in zip(groups, self._chunks(keys)):
                if value is not None:
                    delta, offset, _ = value
                    result.ravel()[group] = delta[tuple((coords[group] % self.chunk).T)].astype(np.float64) + offset
            return result

    def _admit(self, additional: int) -> None:
        used = 0
        if self.max_bytes is not None:
            for directory, subdirs, files in os.walk(self.root):
                names = files + [name for name in subdirs if (Path(directory) / name).is_symlink()]
                used += sum((Path(directory) / name).lstat().st_size for name in names)
        if (self.max_bytes is not None and used + additional > self.max_bytes) or shutil.disk_usage(
            self.root
        ).free < additional + self.reserve_bytes:
            self._poisoned = True
            raise StorageBudgetError("field checkpoint exceeds storage cap or free-space reserve")

    @staticmethod
    def _progress(progress: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(progress, dict):
            raise ValueError("checkpoint progress must be a JSON object")
        return _json_load(_json_bytes(progress))

    def _publish(self, manifest: dict[str, Any], payload: tuple[str, bytes] | None = None) -> None:
        self._admit(len(_json_bytes(manifest)) + (len(payload[1]) if payload else 0))
        try:
            if payload is not None:
                name, raw = payload
                destination = self.root / name
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError("immutable checkpoint generation already exists")
                temporary = self.root / (name + ".pending")
                _write_file(temporary, raw)
                os.replace(temporary, destination)
                _sync_directory(self.root)
            atomic_json(self.root / "CHECKPOINT.json", manifest)
        except BaseException:
            self._poisoned = True
            raise
        self._manifest = manifest

    def commit(
        self,
        key: Sequence[int],
        values: ArrayLike,
        progress: dict[str, Any],
        observed: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Durably replace one entire owned core and progress in one transaction.

        The receipt's ``quantization_max`` is the maximum absolute decoded-core
        error, not a stored/global residual or a decoded fixed-value guarantee.
        Caller validation of the pre-encoding solve remains the acceptance gate.
        """
        with self._operation:
            self._ensure_open()
            key = self._key(key)
            shape = self._shape(key)
            array = np.asarray(values)
            if array.dtype.kind not in "iuf" or array.shape != shape:
                raise ValueError("invalid owned core shape or real dtype")
            array = np.array(array, dtype=np.float64, copy=True)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite owned core values")
            mask: NDArray[np.uint8] = np.empty(0, np.uint8)
            if observed is not None:
                support = np.asarray(observed)
                if support.shape != shape or support.dtype != np.bool_:
                    raise ValueError("observed must be a matching boolean array")
                mask = np.packbits(support)
            next_manifest = deepcopy(self._manifest)
            next_manifest["progress"] = self._progress(progress)
            label = "-".join(map(str, key))
            old = next_manifest["chunks"].pop(label, None)
            payload = None
            error = 0.0
            if array.any() or observed is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    offset = np.float64(array.mean())
                    delta = (array - offset).astype(np.float32)
                    restored = delta.astype(np.float64) + offset
                    error = float(np.max(np.abs(restored - array)))
                if not np.isfinite(delta).all() or not np.isfinite(restored).all() or not math.isfinite(error):
                    raise ValueError("nonfinite checkpoint encoding/decoding")
                stream = io.BytesIO()
                np.savez_compressed(stream, delta=delta, offset=offset, observed=mask)
                raw = stream.getvalue()
                _decode(raw, shape)
                name = f"{label}.g{uuid.uuid4().hex}.npz"
                next_manifest["chunks"][label] = dict(
                    path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), quantization_max=error
                )
                payload = name, raw
            if old is not None:
                try:
                    self._payload(old)
                except BaseException:
                    self._poisoned = True
                    raise
            self._publish(next_manifest, payload)
            if old is not None:
                with self._cache_lock:
                    retired = self._cache.pop(old["path"], None)
                    if retired is not None:
                        self._cache_nbytes -= self._charge(retired)
                try:
                    self._payload(old)
                    (self.root / old["path"]).unlink()
                    _sync_directory(self.root)
                except BaseException:
                    self._poisoned = True
                    raise
            # shortcut: interrupted/unreferenced generations require verified manual cleanup.
            return dict(quantization_max=error, stored_bytes=sum(v["bytes"] for v in self._manifest["chunks"].values()))

    def save_progress(self, progress: dict[str, Any]) -> None:
        """Replace all progress atomically; preserve cumulative sweep statistics."""
        with self._operation:
            self._ensure_open()
            manifest = deepcopy(self._manifest)
            manifest["progress"] = self._progress(progress)
            self._publish(manifest)

    def close(self) -> None:
        """Drain readers, drop retention, and release the writer lease; idempotent."""
        with self._operation:
            try:
                if self._pool is not None:
                    self._pool.shutdown(wait=True)
                    self._pool = None
            finally:
                with self._cache_lock:
                    self._cache.clear()
                    self._cache_nbytes = 0
                if self._lease is not None:
                    os.close(self._lease)
                    self._lease = None
                self._closed = True

    def __enter__(self) -> BlockStore:
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
