"""Distance of field runs from a reference run on the nodes both observe.

    compare_fields.py REFERENCE_RUN CANDIDATE [CANDIDATE ...]

A candidate is a run directory, or ``label=CHECKPOINT.json,FIELD.json`` for an archived
checkpoint (e.g. a field before it was extended). All fields must share one grid. The
constant gauge offset (median difference) is removed; the rest is |dW| in windings.
A blocked coarse can look settled per sweep and still be windings away from the one-block
solution: on PHerc0175A, halo 2 after 8 sweeps had a median |dW| of 11.0, halo 32 after 16
sweeps 0.38.
"""

import hashlib
import json
import sys

import numpy as np

from winding_field.surface_sources import FieldSource


def source(checkpoint, field_json):
    geometry = json.load(open(field_json))["geometry"]
    sha = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
    descriptor = dict(kind="chunks", path=checkpoint, sha256=sha, support_policy="observed")
    return FieldSource(descriptor, geometry, cache_bytes=12 << 30), geometry


def parse(text):
    if "=" in text:
        label, files = text.split("=", 1)
        return (label, *files.split(","))
    return text, text + "/field/CHECKPOINT.json", text + "/FIELD.json"


if len(sys.argv) < 3:
    sys.exit(__doc__)
reference, grid = source(*parse(sys.argv[1])[1:])
shape = grid["shape_zyx"]
for label, checkpoint, field_json in map(parse, sys.argv[2:]):
    candidate, geometry = source(checkpoint, field_json)
    if any(geometry[k] != grid[k] for k in ("shape_zyx", "origin_xyz", "spacing_xyz")):
        sys.exit(f"{label}: grid differs from the reference")
    parts = []
    for z in range(0, shape[0], 64):
        lo, hi = [z, 0, 0], [min(z + 64, shape[0]), shape[1], shape[2]]
        expected, seen = reference.read(lo, hi)
        values, observed = candidate.read(lo, hi)
        both = seen & observed
        parts.append((values - expected)[both].astype(np.float32))
    difference = np.concatenate(parts)
    distance = np.abs(difference - np.median(difference))
    q = np.percentile(distance, [50, 90, 95, 99, 99.9])
    print(
        f"{label}: nodes {len(distance):,} | median {q[0]:.3f} p90 {q[1]:.3f} p95 {q[2]:.3f} "
        f"p99 {q[3]:.3f} p99.9 {q[4]:.3f} max {distance.max():.2f} | "
        f">0.5: {100 * (distance > 0.5).mean():.2f}%  >1: {100 * (distance > 1).mean():.2f}%",
        flush=True,
    )
