"""Per-chunk value ranges of a chunk-checkpoint field, so mesh extraction skips tiles.

A level is meshed only where a tile's admissible values span it (``surface_mesh._tile_mesh``
returns no faces otherwise), yet every tile is still read and decoded first. With a ranges
file the extractor asks ``tile_bounds`` before reading and skips tiles whose chunks cannot
contain the level. The bounds are the union over the chunks the tile touches, a superset
of the tile's own range, so skipping never removes a face: meshes are identical.
"""

import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from field_store import atomic_json

from .surface_mesh import sha
from .surface_sources import FieldSource

SCHEMA = "winding-field-chunk-ranges-v1"
# Returned for a tile with no admissible node: excludes every finite level except this one,
# and a tile with no admissible node yields no faces for any level, so including it is safe.
EMPTY = (-sys.float_info.max, -sys.float_info.max)

_worker = {}


def _binding(field, geometry):
    if not isinstance(field, dict) or field.get("kind") != "chunks":
        raise ValueError("tile ranges require a chunk-checkpoint field")
    manifest = json.loads(Path(field["path"]).read_bytes())
    return dict(
        checkpoint_sha256=field["sha256"],
        support_policy=field["support_policy"],
        shape_zyx=[int(n) for n in geometry["shape_zyx"]],
        chunk=int(manifest["chunk"]),
    ), manifest


def _start(field, geometry):
    _worker["source"] = FieldSource(field, geometry, cache_bytes=0)


def _scan(task):
    label, low, high = task
    values, valid = _worker["source"].read(low, high)
    if not valid.any():
        return label, None
    admissible = values[valid]
    return label, [float(admissible.min()), float(admissible.max())]


def build_ranges(field, geometry, output, *, workers):
    """Decode every stored chunk once (hash-checked by the reader) and write its range."""
    if type(workers) is not int or workers < 1:
        raise ValueError("ranges workers must be a positive integer")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("ranges output exists; keep it or choose a new path")
    binding, manifest = _binding(field, geometry)
    with FieldSource(field, geometry, cache_bytes=0) as source:
        source.validate()
    chunk, shape = binding["chunk"], binding["shape_zyx"]
    tasks = []
    for label in sorted(manifest["chunks"]):
        key = [int(k) for k in label.split("-")]
        low = [k * chunk for k in key]
        tasks.append((label, low, [min(a + chunk, n) for a, n in zip(low, shape)]))
    started = time.monotonic()
    ranges = {}
    with ProcessPoolExecutor(
        min(workers, max(1, len(tasks))), initializer=_start, initargs=(field, geometry)
    ) as pool:
        for label, value in pool.map(_scan, tasks, chunksize=max(1, len(tasks) // (workers * 8))):
            if value is not None:
                ranges[label] = value
    with FieldSource(field, geometry, cache_bytes=0) as source:
        source.validate()
    result = dict(
        schema=SCHEMA,
        binding=binding,
        chunks=len(tasks),
        admissible_chunks=len(ranges),
        raw_min=min((v[0] for v in ranges.values()), default=None),
        raw_max=max((v[1] for v in ranges.values()), default=None),
        seconds=time.monotonic() - started,
        ranges=ranges,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {k: v for k, v in result.items() if k != "ranges"} | dict(
        path=str(output), sha256=sha(output)
    )


def load_ranges(path, field, geometry):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("tile ranges file missing")
    document = json.loads(path.read_bytes())
    binding, _ = _binding(field, geometry)
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or document.get("binding") != binding
        or not isinstance(document.get("ranges"), dict)
    ):
        raise ValueError("tile ranges were built for another field, policy or chunking")
    ranges = {}
    for label, value in document["ranges"].items():
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in value)
            or value[0] > value[1]
        ):
            raise ValueError("invalid tile range entry")
        ranges[label] = (float(value[0]), float(value[1]))
    return binding["chunk"], ranges


class RangedSource:
    """A FieldSource whose extractor may skip tiles outside the stored chunk ranges."""

    def __init__(self, source, chunk, ranges):
        self._source, self._chunk, self._ranges = source, chunk, ranges

    def __getattr__(self, name):
        return getattr(self._source, name)

    def tile_bounds(self, low, high):
        """Union of the ranges of the chunks under the half-open crop [low, high)."""
        lowest, highest = math.inf, -math.inf
        keys = itertools.product(
            *(
                range(int(a) // self._chunk, (int(b) - 1) // self._chunk + 1)
                for a, b in zip(low, high)
            )
        )
        for key in keys:
            value = self._ranges.get("-".join(map(str, key)))
            if value is not None:
                lowest, highest = min(lowest, value[0]), max(highest, value[1])
        return EMPTY if lowest > highest else (lowest, highest)


def default_workers():
    from .surface_parallel import cpu_limit

    return max(1, cpu_limit())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("use: winding-field prepared ranges --config FILE --output RANGES.json")
