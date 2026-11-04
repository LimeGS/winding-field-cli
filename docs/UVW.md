# Aligned UVW bands

`export-uvw` transports **one existing original global UV map** along the
winding gradient. Every W uses the same UV origin, pitch, dimensions and row
direction. It does not flatten each neighbouring sheet again.

Requires the existing `tifxyz` extra (`pip install 'winding-field-cli[tifxyz]'`).

```sh
winding-field export-uvw \
  --field-config field.json \
  --reference-sheet exports/ref/original/level-0000 \
  --levels -32 -31 -30 \
  --source /data/scroll.zarr --voxel-um 8.64 \
  --canvas-samples 512 --output exports/uvw
```

For read-only process geometry parallelism, add `--workers 2` (1–8, default 1).
`--cache-mib 256` is the total retained decoded-CT plus field-region RAM cache
budget (0–8192 MiB; 0 disables both). Half goes to CT in the coordinator;
the remainder is divided between field readers, including the coordinator.
A chunk/region larger than its allowance is read without retention. Allow for
decoded payload bytes plus 512 bytes per retained entry when choosing a budget.
This is not a total RSS limit: each process also owns the reference map/index,
and output arrays, one transient CT decode and bounded in-flight batches use
additional memory. Start with 1/2 workers and measure before using 4/8.

Workers use `spawn` and inherit one-thread native-library limits before imports.
They open their own read-only field/reference; only the coordinator reads CT or
writes cache/source/publication SQLite databases and checkpoints. A window of
at most `workers * 1024` samples feeds the existing chunk-sorted trilinear
sampler, preserving output order and positive-weight missingness. CT grouping
is reused, not duplicated. SIGINT/STOP cancels pending geometry and signals
running workers; shutdown waits for their bounded work to exit.

Decoded CT hits check encoded-file identity (inode/size/mtime/ctime); cold reads
retain SHA and source-ledger validation. Field-region hits validate metadata
and every touched payload before and after copying cells. `validate_used()`
clears decoded RAM entries and forces fresh authoritative CT checks, even in
the same reader. UVW publication uses `validate_used(fresh=False)` within that
invocation: it hashes encoded local/cache bytes against the ledger while reusing
each chunk's remote verification. New or evicted chunks still require a fetch.
Resume starts a fresh verification epoch. This per-chunk read snapshot is not
an atomic snapshot of the remote volume; remote changes after a chunk's read
are detected on explicit fresh validation/resume, not every level publication.
Only advisory encoded-cache access-time updates are batched;
new source-ledger entries and published checkpoint commits remain durable.

Repeat exactly the same command with `--resume` to continue. A file named
`exports/uvw/STOP`, SIGINT or SIGTERM pauses at bounded work boundaries. Remove
`STOP` before resuming. Completed batches and publications are checked against
the SQLite ledgers; changed inputs, settings, implementation, CT availability
or products are rejected. Preserve the whole output directory, including hidden
checkpoint directories and ledgers. Use a new output for changed inputs.

## Inputs and coordinates

The reference must retain its `LEVEL.json`, full product inventory, sibling
`publications.sqlite` and `ORIGINAL.json`. These are read-only. The CT identity
must match the reference's original CT identity. The field descriptor uses the
same SHA-bound `field` and explicit `geometry` objects as `export-sheets`:

```json
{
  "field": {"kind": "npy", "path": "/data/w.npy", "sha256": "<64 lowercase hex characters>"},
  "geometry": {
    "frame": "scan-frame", "voxel_um": 8.64,
    "origin_xyz": [0, 0, 0], "spacing_xyz": [16, 16, 16],
    "shape_zyx": [128, 256, 256], "sheet_step": 1
  }
}
```

Chunked descriptors and optional SHA-bound NPY support masks are also supported.
Geometry origins and spacings in that descriptor are in `voxel_um` units.
The numerical API uses physical XYZ **micrometres**, UV **micrometres** and W
in field units. Original map XYZ are converted using both output-frame scale
and origin; UV lengths and canvas pitch use the scale only.

`--canvas-samples` selects a shared square grid, default 512, maximum 1024.
The reference canvas top-left and physical span are preserved; U increases
with columns, V decreases with rows. Target W values may be fractional; the
requested band must contain reference W. `--level` selects the CT pyramid
level, not W. Up to 16 distinct W targets are admitted.

## Products

- `UVW.json`, `reference.json`, `quality.json`: source identities, original
  receipts, shared grid, per-level and common coverage, residuals, deformation,
  elapsed level time and process peak RSS.
- `level-NNNN/`: `image.png`, `geometry-mask.png`, `ct-mask.png`,
  `cause-mask.png`, `cell-mask.png`, `quality.json`, `LEVEL.json`.
- `level-NNNN/tifxyz/{x,y,z}.tif`: float32 **output-frame voxel coordinates**;
  `meta.json` supplies origin and voxel size. Convert to physical XYZ by
  `xyz_um = xyz_tiff * voxel_um + origin_xyz_um`.
- `tifxyz/valid-mask.png`: geometry validity, including points with unavailable
  CT. Invalid coordinates are NaN. Validity masks use 255 valid / 0 invalid.
  `cell-mask.png` has dimensions `(height-1, width-1)`.

`cause-mask.png` codes: 0 valid, 1 reference outside/excess projection,
2 ambiguous or unresolved reference, 3 invalid initial field, 4 zero gradient,
5 unsupported integration path, 6 nonconvergence, 7 roundtrip failure,
8 sampled Jacobian failure, 9 invalid target, 10 coordinate precision loss.
Unsupported paths include
out-of-field or unusable intermediate samples. CT missingness is separate.
The preview uses the original fixed intensity window 0–180. Black pixels
are **not** evidence of air; inspect both masks.

## Python API

```python
from winding_field.surface_sources import FieldSource
from winding_field.tifxyz_field import PhysicalField
from winding_field.uvw_pipeline import load_reference
from winding_field.uvw import sample_w_gradient, transport, uvw_to_xyz, xyz_to_uvw

ref, metadata = load_reference("exports/ref/original/level-0000", [-32, -31, -30])
with PhysicalField(FieldSource(descriptor["field"], descriptor["geometry"])) as field:
    xyz_um, valid = uvw_to_xyz(ref, field, [[u_um, v_um, -30.5]])
    uvw, inverse_valid = xyz_to_uvw(ref, field, xyz_um)
    values, gradient_w_per_um, supported = sample_w_gradient(field, xyz_um)
    moved, moved_valid, diagnostics = transport(field, xyz_um, -31)
```

Batches are at most 16,384 points; the exporter checkpoints 1,024 at a time.
`Reference(world_um, faces, uv_um, reference_w, band=(lo, hi))` also accepts
programmatically constructed maps. `export_existing(args, stop=callback)` in
`uvw_pipeline` is the orchestration entry point; `args` has the CLI fields.

### Performance measurements

Per-level `quality.json` contains `timings` for geometry wall time, recorded
geometry/reference worker time, CT sampling, checkpoint work, Jacobian checks,
publication preparation and fresh source validation. Preparation excludes the
final receipt/rename itself; aggregate invocation time includes completed
level publication. Resumed worker-time counters include saved batches, while
wall-time counters describe the current invocation. Cache/read counters and
per-worker PID/peak RSS are included; individual process peaks are not a
measurement of simultaneous aggregate RSS. Worker/cache settings are pinned
in the resume identity, but do not change numerical coordinates or masks.

For fixed physical-UV point benchmarks, use:

```python
from winding_field.uvw_workers import GeometryWorkers

jobs = [{"uv": uv_um_batch, "w": 32., "reference_data": None}]
with GeometryWorkers(descriptor, reference_path, [30., 31., 32.], 512,
                     workers=2, region_cache_bytes=64 << 20) as pool:
    cold = list(pool.map(jobs))
    warm = list(pool.map(jobs))
# Each result is (numeric_arrays, timing_and_cache_metrics).
```

Use a `__main__` guard in scripts that spawn workers. Batches are <=16,384;
at most twice the worker count is queued. The low-level serial equivalent is
`uvw_workers.compute_geometry(reference, field, frame, uv, w, ...)`.
CT-only measurements use `Volume(..., decoded_cache_bytes=...)` with
`original_pipeline.sample_ct`; default `Volume` caching remains disabled.
Cold/warm RAM timings must be distinguished from encoded disk, OS and HTTP
cache state. Compare identical frozen points and masks before reporting speedups;
synthetic timings do not establish performance or physical coverage on a scroll.

## Numerical scope

The field is trilinear with analytic physical gradients. Adaptive midpoint
step doubling integrates `dX/dW = grad(W)/|grad(W)|²`; trial spatial steps are
at most 0.25 of minimum field spacing. Every cell in each accepted swept box
must have all eight finite supported nodes. Original extraction residuals
may be projected at most 0.25 spacing; final normal residual is at most 0.05
spacing and checked roundtrip error at most 0.25 spacing. Integration has a
4,096-iteration bound.

The exporter checks the actual float32 coordinate representation before CT
sampling: rounding must be at most 0.01 spacing, and the stored points must
pass the field residual and backward roundtrip checks. TIFFs are read back
before publication.

Forward mapping, inverse mapping and export share the same ambiguity admission.
The inverse flows back to reference W, enumerates nearby reference-triangle
normal-line intersections, rejects distinct UV solutions, and forward-checks
the remaining candidates without recursively calling the public forward API.
A triangle within the allowed projection distance that the normal line misses
is **unresolved**, not ruled out: curved flow might reach it. Such points are
masked unless every supported field cell enclosing that neighbourhood has
identical nodal edge slopes, certifying affine field/straight flow there.
Failed candidate checks are treated conservatively in the same way. This
strict certificate can exclude valid curved regions; the cause mask records
that exclusion rather than claiming uniqueness. Nearest-triangle distance
alone is never accepted. Raster cells additionally check both triangle
orientations and the sampled ODE W Jacobian against the reference Jacobian,
using gradients at each surface's own XYZ; their area ratios are reported.
These are **sampled local checks, not a proof of global injectivity**. Missing
or excluded regions remain masked; coverage denominators include the full
canvas and excluded reference samples.

Outputs are diagnostic, not physically accepted sheets. A real-data pilot
and visual/scientific assessment remain necessary before operational use.
