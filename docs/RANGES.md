# Continuous W ranges

`extract-range` extracts the explicit hypothesis
`theta = 2*pi*h*(W - (-60))`, with angular origin +X toward +Y and `h` equal
to +1 or -1. It solves `field(X,Y,Z) = sheet_step * W` along each forward
radial ray. This is **not** scalar `--levels` extraction, physical sheet
validation, or a claim that arbitrary surfaces have one radial root.

## Recipe and commands

Use absolute source paths and SHA256 hashes, as for `FieldSource`:

```json
{
  "field": {
    "kind": "npy",
    "path": "/absolute/field.npy",
    "sha256": "<64 lowercase hex characters>",
    "mask": {
      "path": "/absolute/support.npy",
      "sha256": "<64 lowercase hex characters>"
    }
  },
  "geometry": {
    "frame": "scan/world",
    "voxel_um": 9.6,
    "origin_xyz": [0, 0, 0],
    "spacing_xyz": [32, 32, 32],
    "shape_zyx": [593, 256, 256],
    "sheet_step": 1.192497988541921
  },
  "umbilicus": [[4000, 3900, 0], [4010, 3890, 18944]],
  "W_ref": -60,
  "theta_ref": 0,
  "h": 1,
  "samples_per_w": 1000,
  "max_edge_world": 128
}
```

The numbers above illustrate the format, **not a verified scan calibration**.
Supply the actual saved geometry and umbilicus. `mask` is optional; omitted
means all finite NPY samples are supported. SHA-bound chunk checkpoints with
explicit `support_policy` are also accepted by the existing FieldSource.
Umbilicus entries are XYZ in the declared world frame, strictly increasing in
Z. Centers interpolate only inside that Z support and must lie within XY
source bounds. Source arrays use ZYX; geometry vectors use XYZ. `voxel_um`
converts one world unit to micrometres. `max_edge_world` is optional and is
measured in those world units, not micrometres or grid samples.

```sh
winding-field extract-range --config recipe.json --w-range -61 -60 --output left
winding-field extract-range --config recipe.json --w-range -60 -59 --output right
winding-field join-ranges right left --output joined
# Resume the identical recipe and bounds after a crash:
winding-field extract-range --config recipe.json --w-range -61 -60 --output left --resume
```

Python API:

```python
from winding_field.range_surface import extract_range, join_ranges, RangeReader
extract_range(recipe, ["-61", "-60.375"], "left", block_ticks=16)
extract_range(recipe, ["-60.375", "-59"], "right", block_ticks=8)
join_ranges(["right", "left"], "joined")
with RangeReader("joined") as reader:
    print(reader.receipt)
    # reader.mesh is an ordinary MeshReader over joined/mesh.
```

## Determinism and supported domain

- Default: 1000 intervals per W unit. Endpoints use Decimal and must be exact
  lattice members; they are never rounded. Bounds must increase. Python callers
  should use strings to preserve decimal input literals. `-60.375` is valid.
- Global node keys are `recipe_sha:n:tickW:tickZ`; cell keys are
  `recipe_sha:c:tickW:tickZ`, with face suffix `:0` or `:1`. Z ticks are source
  sample indices. Both endpoint columns are present; cells own `[A,B)`.
- Trigonometry uses the global W tick modulo samples-per-W; cardinal directions
  are exact. Coordinates and raw target levels never depend on the request's
  start, transaction grouping, or partitioning.
- Rays are divided at every XY grid-cell boundary. The fixed-Z bilinear
  interpolant restricted to each segment is quadratic; both roots are solved,
  including same-sign endpoint cases. Boundary roots are deduplicated. Tangent,
  zero-radius, identically-zero, and numerically unresolved roots are rejected.
- Numeric policy, implementation file hashes, Python, NumPy and platform are
  part of the recipe identity. Source content/mask hashes, units, calibration,
  umbilicus, handedness, lattice and edge policy are bound too. Source paths,
  request bounds and `block_ticks` are not recipe identity. Different numeric
  runtimes are conservatively incompatible.
- A missing/nonfinite source cell anywhere on a ray makes uniqueness
  `unsupported`, even if another supported portion contains one root. This
  conservative rule avoids claiming uniqueness through a masked gap.
- Node states: `unique`, `no_root`, `multiple`, `degenerate`, `unsupported`,
  `center_unsupported`. All finite radial candidates remain in the audit.
- Cells with four unique nodes use `(00,10,11)` and `(00,11,01)` in increasing
  W/Z order. Cells record `accepted`, `node_rejected`, `long_edge` or
  `degenerate`. Optional maximum-edge rejection includes the diagonal.

## Artifacts, crash recovery and joining

Each output contains `RANGE.json` and an inner `mesh/` MeshDB bundle. The
wrapper is published only after completion. Inner identity is
`kind=continuous_range`; **slot 0 is a storage slot, not W=0**. Scalar original
export rejects both the wrapper and a detached inner mesh path before CT or
optional flattening dependencies are accessed.

`mesh/mesh.sqlite` contains all geometry, topology, node and cell audit tables,
per-column audit hashes and the tile cursor. Each column's writes share one
SQLite transaction; `block_ticks` groups at most 256 columns per transaction
(default 16). Resume verifies committed geometry and audit checksums, and
rebuilds topology using the existing MeshDB mechanism. Abrupt death rolls the
unfinished transaction back. No full W-range array is materialized: working
geometry consists of two Z columns, radial crossing/root lists, a 2x2 source
crop and SQLite's bounded cache. Chunk sources additionally decode one source
chunk. Space on disk scales with the full requested domain, including holes.

Joining accepts complete, recipe-identical contiguous or overlapping ranges
in any input order. Duplicates are idempotent. Every shared node record and
cell record must match exactly, including XYZ, root/state audit, rejection,
triangle presence and orientation. No tolerance welding or averaging occurs.
Source fields need not remain online to join. Join writes a new output; failed
joins leave an incomplete directory without `RANGE.json` and require a fresh
output path for retry.

An output with no faces is complete `EMPTY`, with domain and rejection causes
retained. EMPTY can be joined but is never positive surface evidence. Unique
isolated vertices can exist in EMPTY. Successful positive tests require both
faces and positive area. Compare direct and partitioned outputs by canonical
keyed node/cell/vertex/face data, **not SQLite bytes or local integer row IDs**.

## Validation and limits

`tests/test_range_surface.py` exercises independent analytic geometry,
calibration and anisotropic/nonzero-origin coordinates, roots and tangencies,
fractional cuts, duplicate/reversed/overlapping joins, incompatible recipes,
gaps, masks, EMPTY, checksum-resealed conflicts, scalar guards, CLI execution,
and abrupt process death inside a real write transaction followed by resume.

This initial extractor is a correctness-first one-root radial domain. It does
not certify physical continuity, self-intersection freedom, topology between
sampled nodes, or output flattenability. Real scan/GPU performance and memory
admission need the planned remote smoke test. Patch flattening, CT and TIFXYZ
publication are the separate Task 2 deliverable.
