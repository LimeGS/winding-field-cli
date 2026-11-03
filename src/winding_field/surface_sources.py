"""Read-only SHA-bound field crops; arrays are ZYX, geometry vectors are XYZ.

NPY files are hashed once, mapped through that same open inode, and checked for
immutable inode/size/mtime/ctime/header identity before and after each crop.
Checkpoint metadata and touched payloads are checked on every read, including
cache hits. Concurrent writers are unsupported: detected changes fail closed.
No source directory entries, writer leases or checkpoints are ever created.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
import stat
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import field_store

_Key = tuple[int, int, int]
_Chunk = tuple[NDArray[np.float32], float, NDArray[np.uint8]]


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA256 hex hash")
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("expected a positive finite geometry number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("geometry number exceeds float64") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError("expected a positive finite geometry number")
    return number


def _vector(value: Any, *, positive: bool = False) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "iuf" or any(isinstance(v, (bool, np.bool_)) for v in value):
        raise ValueError("expected a finite numeric XYZ vector")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError("invalid geometry XYZ vector")
    result.flags.writeable = False
    return result


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("field source must be a regular file, not a symlink")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _File:
    def __init__(self, path: Path, digest: str, size: int | None = None) -> None:
        self.path, self.digest = path, _digest(digest)
        self.signature = _signature(path.lstat())
        self.size = self.signature[2]
        if size is not None and self.size != size:
            raise ValueError("field source byte count differs")

    def check(self) -> None:
        if _signature(self.path.lstat()) != self.signature:
            raise ValueError("field source changed: " + str(self.path))

    def read(self) -> bytes:
        self.check()
        raw = field_store._read_file(self.path, self.size)
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise ValueError("field source hash differs: " + str(self.path))
        self.check()
        return raw


def _file(descriptor: dict[str, Any]) -> _File:
    if not isinstance(descriptor, dict) or not {"path", "sha256"} <= descriptor.keys():
        raise ValueError("source requires path and sha256")
    path = descriptor["path"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError("source path must be absolute")
    return _File(Path(path), descriptor["sha256"])


class _Npy:
    def __init__(self, file: _File, shape: _Key, *, mask: bool = False) -> None:
        self.file = file
        fd = os.open(file.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            with os.fdopen(fd, "rb") as stream:
                if _signature(os.fstat(stream.fileno())) != file.signature:
                    raise ValueError("NPY source inode changed")
                if hashlib.file_digest(stream, "sha256").hexdigest() != file.digest:
                    raise ValueError("NPY source hash differs")
                stream.seek(0)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    actual_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    actual_shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported field NPY version")
                self.offset = stream.tell()
                if actual_shape != shape or (dtype != np.bool_ if mask else dtype.kind not in "iuf"):
                    raise ValueError("NPY shape or real/boolean dtype differs")
                if self.offset + math.prod(shape) * dtype.itemsize != file.size:
                    raise ValueError("NPY header/data byte count differs")
                stream.seek(0)
                self.header = stream.read(self.offset)
                file.check()
                # Map the hashed descriptor, never reopen its potentially replaced pathname.
                self.array = np.memmap(
                    stream, dtype=dtype, mode="r", offset=self.offset, shape=shape, order="F" if fortran else "C"
                )
                self.check()
        except BaseException:
            self.close()
            raise

    def check(self) -> None:
        self.file.check()
        fd = os.open(self.file.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if _signature(os.fstat(stream.fileno())) != self.file.signature or stream.read(self.offset) != self.header:
                raise ValueError("NPY source inode/header changed")
        self.file.check()

    def close(self) -> None:
        if hasattr(self, "array"):
            self.array._mmap.close()  # type: ignore[attr-defined]


class FieldSource:
    """Bounded owned float64 values and boolean admissibility from pinned files.

    ``field`` is ``{kind: 'npy', path, sha256, mask?: {path, sha256}}`` or
    ``{kind: 'chunks', path: '/.../CHECKPOINT.json', sha256, support_policy}``.
    Paths must already be absolute. Chunk policies are ``observed`` (packed
    observation bits) and ``stored`` (every referenced payload node). Missing
    chunks always return zero and false, regardless of policy.

    ``geometry`` requires frame, voxel_um, origin_xyz, spacing_xyz, shape_zyx and
    sheet_step. Known geometry in checkpoint/identity/grid/source_identity or
    geometry envelopes must match; omitted metadata never implies units.
    NPY NaN/Inf values remain in values but are always inadmissible.

    Manifest paths, keys and all payload sizes are checked at open. Payload
    hashes and safe decoding are lazy and consume the same byte snapshot.
    ``validate`` checks immutable metadata and all referenced file stats.
    Cache budgets charge retained delta/bitpack arrays plus 512 bytes per chunk;
    working memory also includes the requested crop and one decoded chunk.
    Instances are serial readers, with no concurrent read/close guarantee.
    """

    def __init__(self, field: dict[str, Any], geometry: dict[str, Any], *, cache_bytes: int = 0) -> None:
        if not isinstance(field, dict) or not isinstance(geometry, dict):
            raise ValueError("field and geometry must be JSON objects")
        field = field_store._json_load(field_store._json_bytes(field))
        geometry = field_store._json_load(field_store._json_bytes(geometry))
        required = {"frame", "voxel_um", "origin_xyz", "spacing_xyz", "shape_zyx", "sheet_step"}
        if not required <= geometry.keys():
            raise ValueError("explicit geometry fields are required")
        self.frame = geometry["frame"]
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("geometry frame must be a nonempty string")
        self.voxel_um, self.sheet_step = _number(geometry["voxel_um"]), _number(geometry["sheet_step"])
        self.origin = _vector(geometry["origin_xyz"])
        self.spacing = _vector(geometry["spacing_xyz"], positive=True)
        self.shape = field_store._triple(geometry["shape_zyx"], 1)
        if math.prod(self.shape) > np.iinfo(np.intp).max:
            raise ValueError("field shape exceeds indexing capacity")
        self._budget = field_store._integer(cache_bytes)
        self._cache: OrderedDict[str, _Chunk] = OrderedDict()
        self._cache_bytes = 0
        self._arrays: list[_Npy] = []
        self._payloads: dict[str, _File] = {}
        self._closed = False
        self._geometry = geometry
        self._kind = field.get("kind")
        self._file = _file(field)
        self._identity = dict(field=field, geometry=geometry)
        try:
            if self._kind == "npy":
                if field.keys() - {"kind", "path", "sha256", "mask"}:
                    raise ValueError("unknown NPY descriptor fields")
                self._arrays.append(_Npy(self._file, self.shape))
                if "mask" in field:
                    mask = field["mask"]
                    if not isinstance(mask, dict) or set(mask) != {"path", "sha256"}:
                        raise ValueError("NPY mask requires path and sha256")
                    self._arrays.append(_Npy(_file(mask), self.shape, mask=True))
                self._identity["source_identity"] = {k: v for k, v in field.items() if k != "kind"}
            elif self._kind == "chunks":
                if field.keys() - {"kind", "path", "sha256", "support_policy"}:
                    raise ValueError("unknown chunk descriptor fields")
                self._policy = field.get("support_policy")
                if self._policy not in ("observed", "stored") or self._file.path.name != "CHECKPOINT.json":
                    raise ValueError("chunks require CHECKPOINT.json and observed/stored support_policy")
                self._root = self._file.path.parent
                info = self._root.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError("checkpoint root must be a directory, not a symlink")
                self._root_identity = info.st_dev, info.st_ino
                self._manifest = field_store._json_load(self._file.read())
                self._prepare_chunks()
                self._identity["source_identity"] = self._manifest["identity"]
            else:
                raise ValueError("field kind must be npy or chunks")
        except BaseException:
            self.close()
            raise

    @property
    def identity(self) -> dict[str, Any]:
        """Detached JSON configuration, geometry and embedded source identity."""
        return deepcopy(self._identity)

    def _match_geometry(self, envelope: Any) -> None:
        if not isinstance(envelope, dict):
            return
        for key, value in envelope.items():
            name = "shape_zyx" if key == "shape" else key
            if name in self._geometry and name in {
                "frame",
                "voxel_um",
                "sheet_step",
                "origin_xyz",
                "spacing_xyz",
                "shape_zyx",
            }:
                expected = self._geometry[name]
                if name == "shape_zyx":
                    matches = field_store._triple(value, 1) == self.shape
                elif name in ("origin_xyz", "spacing_xyz"):
                    matches = np.array_equal(_vector(value, positive=name == "spacing_xyz"), expected)
                elif name in ("voxel_um", "sheet_step"):
                    matches = _number(value) == expected
                else:
                    matches = isinstance(value, str) and value == expected
                if not matches:
                    raise ValueError("embedded source geometry differs: " + key)
            elif key in {"identity", "source_identity", "grid", "geometry"}:
                self._match_geometry(value)

    def _chunk_shape(self, key: _Key) -> _Key:
        z, y, x = (min(self._chunk, n - k * self._chunk) for k, n in zip(key, self.shape))
        return z, y, x

    def _prepare_chunks(self) -> None:
        manifest = self._manifest
        if (
            not isinstance(manifest, dict)
            or not {"identity", "shape", "chunk", "chunks", "progress"} <= manifest.keys()
        ):
            raise ValueError("malformed checkpoint manifest")
        self._chunk = field_store._integer(manifest["chunk"], 1)
        if self._chunk > np.iinfo(np.intp).max:
            raise ValueError("chunk exceeds indexing capacity")
        self._match_geometry(manifest)
        if not isinstance(manifest["chunks"], dict) or not isinstance(manifest["progress"], dict):
            raise ValueError("malformed checkpoint chunks/progress")
        for label, record in manifest["chunks"].items():
            if re.fullmatch(r"(?:0|[1-9][0-9]*)-(?:0|[1-9][0-9]*)-(?:0|[1-9][0-9]*)", label) is None:
                raise ValueError("invalid checkpoint chunk key")
            key = field_store._triple(tuple(int(v) for v in label.split("-")))
            if any(k * self._chunk >= n for k, n in zip(key, self.shape)):
                raise ValueError("checkpoint key outside field")
            if not isinstance(record, dict) or not {"path", "sha256", "bytes"} <= record.keys():
                raise ValueError("malformed checkpoint payload record")
            name = record["path"]
            if not isinstance(name, str) or re.fullmatch(re.escape(label) + r"\.g[0-9a-f]+\.npz", name) is None:
                raise ValueError("invalid checkpoint generation path")
            size = field_store._integer(record["bytes"], 1)
            if size > math.prod(self._chunk_shape(key)) * 5 + 65536:
                raise ValueError("oversized checkpoint payload")
            if "quantization_max" in record:
                error = record["quantization_max"]
                if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                    raise ValueError("invalid checkpoint quantization error")
            self._payloads[label] = _File(self._root / name, record["sha256"], size)

    def _validate(self, *, payloads: bool) -> None:
        if self._closed:
            raise RuntimeError("field source is closed")
        if self._kind == "npy":
            for array in self._arrays:
                array.check()
        else:
            info = self._root.lstat()
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._root_identity:
                raise ValueError("checkpoint directory changed")
            self._file.read()
            if payloads:
                for file in self._payloads.values():
                    file.check()

    def validate(self, *, hashes: bool = False) -> None:
        """Fail on changed metadata or referenced files, including cached files."""
        self._validate(payloads=True)
        if hashes:
            for file in self._payloads.values():
                file.read()

    @staticmethod
    def _charge(value: _Chunk) -> int:
        return value[0].nbytes + value[2].nbytes + 512

    def _load(self, label: str, shape: _Key) -> _Chunk:
        file = self._payloads[label]
        file.check()
        if label in self._cache:
            self._cache.move_to_end(label)
            return self._cache[label]
        value = field_store._decode(file.read(), shape)
        charge = self._charge(value)
        if charge <= self._budget:
            while self._cache_bytes + charge > self._budget:
                _, retired = self._cache.popitem(last=False)
                self._cache_bytes -= self._charge(retired)
            self._cache[label] = value
            self._cache_bytes += charge
        return value

    def read(self, lo: Sequence[int], hi: Sequence[int]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Copy a nonempty half-open integer ZYX crop and its admissible mask."""
        self._validate(payloads=False)
        low, high = field_store._triple(lo), field_store._triple(hi)
        if any(b <= a or b > n for a, b, n in zip(low, high, self.shape)):
            raise ValueError("field read outside grid")
        region = tuple(slice(a, b) for a, b in zip(low, high))
        if self._kind == "npy":
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.array(self._arrays[0].array[region], dtype=np.float64, order="C", copy=True)
            valid = np.isfinite(values)
            if len(self._arrays) == 2:
                valid &= self._arrays[1].array[region]
            self._validate(payloads=False)
            return values, valid
        shape = tuple(b - a for a, b in zip(low, high))
        values, valid = np.zeros(shape, np.float64), np.zeros(shape, bool)
        keys = itertools.product(*(range(a // self._chunk, (b - 1) // self._chunk + 1) for a, b in zip(low, high)))
        for z, y, x in keys:
            key = (z, y, x)
            label = "-".join(map(str, key))
            if label not in self._payloads:
                continue
            block_shape = self._chunk_shape(key)
            origin = tuple(k * self._chunk for k in key)
            first = tuple(max(a, o) for a, o in zip(low, origin))
            last = tuple(min(b, o + n) for b, o, n in zip(high, origin, block_shape))
            target = tuple(slice(a - lo, b - lo) for a, b, lo in zip(first, last, low))
            source = tuple(slice(a - o, b - o) for a, b, o in zip(first, last, origin))
            delta, offset, packed = self._load(label, block_shape)
            values[target] = delta[source]
            values[target] += offset
            if self._policy == "stored":
                valid[target] = True
            elif packed.size:
                valid[target] = np.unpackbits(packed, count=delta.size).reshape(block_shape)[source]
            self._payloads[label].check()
            del delta, offset, packed
        self._validate(payloads=False)
        return values, valid

    def close(self) -> None:
        """Release mappings and cache retention; safe to call repeatedly."""
        for array in self._arrays:
            array.close()
        self._arrays.clear()
        self._cache.clear()
        self._cache_bytes = 0
        self._closed = True

    def __enter__(self) -> FieldSource:
        self.validate()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
