"""Exact native AFV clipping: build, load and call ``native/afv_clip.cpp``.

The shared library is compiled on first use with the host C++ compiler into a
content-addressed cache (source bytes, compiler identity, flags and platform), with
strict IEEE flags: no fast-math, no FMA contraction, no host-specific ``-march``.
Without a compiler the importer keeps its NumPy implementation; ``native`` must
then be requested explicitly to fail instead. Blocks whose exact outcome the kernel
cannot reproduce return ``REFERENCE`` and are clipped by ``afv_clip.pieces``.
"""

import ctypes as C
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

SOURCE = Path(__file__).parent / "native" / "afv_clip.cpp"
FLAGS = (
    "-O2",
    "-std=c++17",
    "-fPIC",
    "-shared",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-strict-aliasing",
)
ABI = 1
OK, REFERENCE, CAPACITY = 0, 1, 2
BACKENDS = ("auto", "native", "numpy", "reference")
MAX_VERTEX_CAPACITY = 4 << 20  # 96 MiB of vertices; one in-budget block needs far less

_LIBRARY = {}


class NativeUnavailable(RuntimeError):
    pass


def backend(config):
    value = config.get("afv_clip_backend") or os.environ.get("WINDING_AFV_CLIP_BACKEND") or "auto"
    if value not in BACKENDS:
        raise ValueError("afv_clip_backend must be one of " + ", ".join(BACKENDS))
    return value


def _compiler():
    for name in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if name and shutil.which(name):
            return shutil.which(name)
    raise NativeUnavailable("no C++ compiler for the native AFV clipper")


def _cache_dir():
    configured = os.environ.get("WINDING_FIELD_NATIVE_CACHE")
    root = Path(configured) if configured else Path.home() / ".cache/winding-field/native"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=root, delete=True)
        probe.close()
        return root
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f"winding-field-native-{os.getuid()}"
        fallback.mkdir(mode=0o700, exist_ok=True)
        return fallback


def build():
    """Path of the compiled library for this source/compiler, building it once."""
    compiler = _compiler()
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, timeout=60, check=False
    ).stdout
    source = SOURCE.read_bytes()
    key = hashlib.sha256(
        b"\0".join(
            [
                source,
                version.encode(),
                " ".join(FLAGS).encode(),
                platform.system().encode(),
                platform.machine().encode(),
                str(ABI).encode(),
            ]
        )
    ).hexdigest()[:24]
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = _cache_dir() / f"afv_clip-{key}{suffix}"
    if library.is_file():
        return library
    with tempfile.TemporaryDirectory(dir=library.parent) as scratch:
        staged_source = Path(scratch) / "afv_clip.cpp"
        staged_source.write_bytes(source)
        staged = Path(scratch) / library.name
        result = subprocess.run(
            [compiler, *FLAGS, str(staged_source), "-o", str(staged)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode or not staged.is_file():
            raise NativeUnavailable("native AFV clipper build failed:\n" + result.stderr[-4000:])
        os.replace(staged, library)
    return library


def load(path):
    path = str(path)
    if path in _LIBRARY:
        return _LIBRARY[path]
    lib = C.CDLL(path)
    lib.afv_clip_abi.restype = C.c_int32
    if lib.afv_clip_abi() != ABI:
        raise NativeUnavailable("native AFV clipper ABI differs")
    p = C.c_void_p
    i64 = C.c_int64
    lib.afv_clip_blocks.restype = C.c_int32
    lib.afv_clip_blocks.argtypes = [p, i64, p, i64, p, i64, i64, i64, i64, p, i64, p, i64, p, p, p]
    _LIBRARY[path] = lib
    return lib


def _pointer(array):
    return array.ctypes.data_as(C.c_void_p)


def clip_blocks(lib, blocks, support, max_intervals):
    """Clip a batch of blocks with the native kernel.

    ``blocks`` is a list of (n, 3) float64 point arrays; ``support`` a C-contiguous
    Boolean/uint8 ZYX array. Returns, per block, a list of fragment arrays or None
    when that block must be clipped by the Python reference.
    """
    if support.dtype not in (np.bool_, np.uint8) or support.ndim != 3:
        raise ValueError("native AFV clipping needs a 3D Boolean support volume")
    if not support.flags.c_contiguous:
        raise ValueError("native AFV clipping needs C-contiguous support")
    if type(max_intervals) is not int or max_intervals < 1:
        raise ValueError("invalid interval budget")
    results = [None] * len(blocks)
    start = 0
    capacity = None
    while start < len(blocks):
        chunk = blocks[start:]
        sizes = []
        for points in chunk:
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("AFV block points must be (n, 3)")
            sizes.append(len(points))
        offsets = np.zeros(len(chunk) + 1, np.int64)
        np.cumsum(sizes, out=offsets[1:])
        packed = (
            np.ascontiguousarray(np.concatenate(chunk), dtype="<f8")
            if offsets[-1]
            else np.zeros((0, 3), "<f8")
        )
        if packed.dtype.byteorder not in ("=", "<", "|") or not np.isfinite(packed).all():
            raise ValueError("AFV block points must be finite native float64")
        if capacity is None:
            capacity = min(MAX_VERTEX_CAPACITY, 4 * int(offsets[-1]) + 4096)
        vertices = np.empty((capacity, 3), np.float64)
        fragment_offsets = np.empty(capacity + 1, np.int64)
        block_fragments = np.empty(len(chunk), np.int64)
        status = np.empty(len(chunk), np.int32)
        totals = np.zeros(2, np.int64)
        code = lib.afv_clip_blocks(
            _pointer(packed),
            int(offsets[-1]),
            _pointer(offsets),
            len(chunk),
            _pointer(support),
            *(int(n) for n in support.shape),
            max_intervals,
            _pointer(vertices),
            capacity,
            _pointer(fragment_offsets),
            capacity,
            _pointer(block_fragments),
            _pointer(status),
            _pointer(totals),
        )
        if code:
            raise RuntimeError(f"native AFV clipper failed with code {code}")
        fragment = 0
        done = 0
        for index, block_status in enumerate(status.tolist()):
            if block_status == CAPACITY:
                break
            count = int(block_fragments[index])
            if block_status == OK:
                results[start + index] = [
                    vertices[fragment_offsets[f] : fragment_offsets[f + 1]].copy()
                    for f in range(fragment, fragment + count)
                ]
            elif block_status != REFERENCE:
                raise RuntimeError("native AFV clipper returned an unknown status")
            fragment += count
            done += 1
        if fragment != totals[1]:
            raise RuntimeError("native AFV clipper inventory mismatch")
        if done == 0:
            # The first remaining block alone does not fit: grow, or leave it to the
            # reference once the bounded capacity is exhausted.
            if capacity >= MAX_VERTEX_CAPACITY:
                start += 1
            else:
                capacity = min(MAX_VERTEX_CAPACITY, 4 * capacity)
            continue
        start += done
    return results
