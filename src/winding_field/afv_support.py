"""Verified dense support snapshot shared by AFV clipping workers.

The snapshot is the maps' Boolean ``support`` channel as one read-only ZYX ``.npy``
memory map. Each tile is read once; the same bytes are hashed against the tile
index and decoded, so a cache miss never re-hashes or re-decompresses a tile, and
workers share one set of physical pages instead of splitting a small per-process
cache. Publication is atomic: the array is fsynced and renamed into place before
the metadata that names its digest; a reader that finds metadata, file or digest
inconsistent rebuilds. The snapshot depends only on the support content and grid
(``maps`` identity, shape, chunk, ZYX order, representation version), not on AFV
input. The final import validation re-checks the snapshot digest and the maps.
"""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from .util import atomic_json, file_sha

SCHEMA = "afv-support-snapshot-v1"
REPRESENTATION = "dense-bool-zyx-npy-v1"


class DenseSupport:
    """Read-only support with the ``SupportTiles``/``Tiles`` load interface."""

    def __init__(self, path, meta):
        self.path = Path(path)
        self.meta = meta
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self.array.dtype != np.bool_ or self.array.shape != tuple(meta["shape"]):
            raise ValueError("support snapshot geometry differs")
        if not self.array.flags.c_contiguous:
            raise ValueError("support snapshot must be C-contiguous")
        self.shape = tuple(meta["shape"])
        self.chunk = meta["chunk"]
        self.dense = self.array
        self.loads = 0

    def load(self, key, names=("support",)):
        if names != ("support",):
            raise ValueError("support snapshot only reads support")
        lo = np.asarray(key) * self.chunk
        hi = np.minimum(lo + self.chunk, self.shape)
        return {"support": self.array[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]}

    def verify(self):
        if file_sha(self.path) != self.meta["sha256"]:
            raise ValueError("AFV support snapshot changed during import")

    def close(self):
        mapping = getattr(self.array, "_mmap", None)
        self.array = self.dense = None
        if mapping is not None:
            try:
                mapping.close()
            except (BufferError, ValueError):
                pass  # still referenced by a view; released with it


def _fsync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _expected(maps, maps_identity):
    return dict(
        schema=SCHEMA,
        representation=REPRESENTATION,
        maps_sha256=maps_identity,
        shape=list(maps.shape),
        chunk=maps.chunk,
    )


def _read_tile(maps, key):
    """A tile's support, hashed against the tile index and decoded from the same bytes."""
    stem = maps.path(key).stem
    record = maps.db.execute("SELECT sha FROM tiles WHERE key=?", (stem,)).fetchone()
    if record is None:
        raise ValueError("incomplete tile set")
    raw = maps.path(key).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record[0]:
        raise ValueError("corrupt tile: " + str(key))
    lo, hi = maps.box(key)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        support = data["support"]
    if support.dtype != np.bool_ or support.shape != tuple(hi - lo):
        raise ValueError("support tile geometry differs: " + str(key))
    return support


def open_or_build(maps, directory, maps_identity, *, limit_bytes, reserve_bytes):
    """A verified ``DenseSupport``, or None when it exceeds the byte or disk budget."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nbytes = math.prod(maps.shape)
    if nbytes > limit_bytes:
        return None, dict(snapshot="bounded-tiles", reason="support exceeds snapshot budget")
    expected = _expected(maps, maps_identity)
    name = f"support-{maps_identity[:24]}.npy"
    path = directory / name
    meta_path = directory / "SUPPORT.json"
    if meta_path.is_file() and path.is_file():
        try:
            meta = json.loads(meta_path.read_bytes())
        except ValueError:
            meta = None
        if (
            isinstance(meta, dict)
            and {k: meta.get(k) for k in expected} == expected
            and meta.get("file") == name
            and file_sha(path) == meta.get("sha256")
        ):
            return DenseSupport(path, meta), dict(snapshot="dense", reused=True, bytes=nbytes)
    for stale in directory.glob("support-*.npy*"):
        stale.unlink()
    meta_path.unlink(missing_ok=True)
    if shutil.disk_usage(directory).free - reserve_bytes < nbytes + (8 << 20):
        return None, dict(snapshot="bounded-tiles", reason="insufficient disk for snapshot")
    partial = directory / (name + ".partial")
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.bool_, shape=maps.shape)
    try:
        for key in maps.keys():
            lo, hi = maps.box(key)
            array[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] = _read_tile(maps, key)
        array.flush()
    finally:
        del array
    _fsync(partial)
    digest = file_sha(partial)
    os.replace(partial, path)
    _fsync(directory)
    meta = dict(expected, file=name, sha256=digest, bytes=nbytes)
    atomic_json(meta_path, meta)
    return DenseSupport(path, meta), dict(snapshot="dense", reused=False, bytes=nbytes)


def remove(directory):
    directory = Path(directory)
    (directory / "SUPPORT.json").unlink(missing_ok=True)
    for path in directory.glob("support-*.npy*"):
        path.unlink()
