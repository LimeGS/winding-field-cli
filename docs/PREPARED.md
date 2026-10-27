# Prepared fields, surfaces and rendering

These workflows live in **winding-field-cli** and call the installed original
`winding-field-experiments` library. The library owns numerical operators,
constraints, block storage, field sources and geometric extraction. The CLI owns
recipes, command dispatch, prepared execution and PNG rendering.

## Build and resume a prepared field

```bash
winding-field prepared build --config /path/to/field-run.json --output /path/to/new-field
winding-field prepared resume --config /path/to/field-run.json --output /path/to/new-field --prefetch

# Owned worker, immutable code/config binding and resource-only retries:
winding-field prepared build --config /path/to/field-run.json --output /path/to/other-field --supervised
```

The `prepared` prefix separates these recipes from `winding-field resume RUN`
and `winding-field mesh RUN`, which operate on the end-to-end CT pipeline.
Checkout wrappers remain `python tools/run_field.py build ...` and
`python tools/run_surfaces.py surfaces ...`.

Start from [field-run.example.json](../examples/field-run.example.json), replacing
the placeholder paths, SHA-256 hashes and calibration with your prepared inputs.
Paths resolve relative to the JSON. Raw CT/AFV preparation and physical
calibration belong to the caller's recipe.

`winding-field-run-v1` requires:

- `frame`, positive `voxel_um` and `sheet_step`, and explicit `calibration`
  provenance. Raw field values are retained; `sheet_step` records conversion to
  nominal sheet units rather than implicitly rescaling them.
- `grid.origin_xyz`, positive `spacing_xyz`, integer `shape_zyx` with at least
  two nodes per axis. Coordinates/gradients are XYZ, arrays/traversal ZYX.
- `observations.target`, `weights`, `observed`: local `path` and lowercase full
  `sha256`. Target is floating ZYX3, weights floating ZYX, observed boolean ZYX.
  Target/weights are float32/float64 inputs read as float32 crops. Targets must
  be finite and weights positive even where observed is false. Unsupported is
  not a measured zero or a physical boundary anchor.
- `execution.chunk`, `halo`, `sweeps`; multiple cores require positive halo.
  Optional `cache_bytes=0`, `read_workers=1`, `constraint_cache_bytes=0`,
  `reserve_bytes=0`, `max_bytes=null` retain the library's resource controls.
- Optional `solver`: `device="cpu"`, `dtype="float64"`, `rtol=1e-5`,
  `maxiter=4000`; float32 also supported.

Optional `fixed` pins a float32/float64 ZYX NPY: finite values are exact fixed
nodes, NaN means free. Without finite anchors the first node fixes the numerical
zero gauge. Optional `constraints` pins an archive from
`constraint_templates.StencilConstraints.save(path)`, preserving rows, targets,
weights and groups. This archive is resident; cache limits do not bound its input
arrays. Large inputs can use the library's `solve_blocks` provider interface.
Optional `order` pins an integer Nx3 NPY containing every chunk key exactly once;
default traversal is lexicographic ZYX. `source_identity` accepts JSON provenance.

CUDA requires `solver.device="cuda:0"`, a `gpu` object with explicit `index`,
`uuid`, `lock_path`, and matching `CUDA_VISIBLE_DEVICES` UUID before launch.
Optional `min_free_bytes` and Unix `deadline` use the library's lease checks.
Prefetch is CPU-only here. Admission does not infer safe RAM/VRAM limits; Python
callers can provide `admit=` for their own policy.

`OUTPUT/CHECKPOINT.json` is recovery authority. `RUN.json` binds physical/source
identity at first admission; `STATUS.json` records progress. Resume rejects
changed inputs, geometry, calibration, numerical settings or implementation.
Reader/cache budgets may change for direct resume.

With `--supervised`, `OUTPUT.service/` holds source-bound state and worker logs.
Automatic retry needs exit 75 and that child's resource receipt. Keep service
code/config immutable. Touch `OUTPUT/STOP` or `OUTPUT.service/STOP` for a bounded
checkpoint request; remove those markers before `prepared resume --supervised`.
A long solve is not an immediate cancellation point. The supervisor returns
0 for budget/STOP, 1 for failure, 130 for interruption.

Python: `winding_field.field_workflow.run_field(config, output, resume=False,
prefetch=False, stop=None, admit=None)` and `load_run_config(config)`.

## Extract or import surfaces, then render

```bash
winding-field prepared surfaces --config /path/to/surfaces.json --output /path/to/new-bundle
winding-field prepared surfaces --config /path/to/surfaces.json --output /path/to/new-bundle --resume

winding-field prepared mesh --config /path/to/surfaces.json --output /path/to/another-bundle
winding-field prepared render --mesh /path/to/another-bundle --output /path/to/new-views
```

Start from [surface-run.example.json](../examples/surface-run.example.json).
`winding-surfaces-v1` has exactly one `field` or `mesh`, explicit `geometry`, and
optional `execution`/`render`. Paths belong to the configuration, not CWD.

- Field geometry: `frame`, positive `voxel_um`, `origin_xyz`, positive
  `spacing_xyz`, `shape_zyx` (at least two nodes/axis), positive `sheet_step`;
  optional `calibration`. Levels are distinct finite raw phi values (max 4096).
  To request winding W, declare phi = W × sheet_step explicitly.
- NPY field: `{"kind":"npy","path":"phi.npy","sha256":"…"}` with optional
  `mask={"path":"support.npy","sha256":"…"}`. Crops are bounded memmaps;
  masks boolean ZYX; NaN/Inf field samples are excluded.
- Chunk field: `{"kind":"chunks","path":"field/CHECKPOINT.json",
  "sha256":"…","support_policy":"observed"}`. Supports native delta32 +
  offset64 + observation-bitpack storage. `observed` requires observation bits;
  explicit `stored` admits referenced payload nodes as inferred geometry.
  Absent chunks remain unsupported. Paris4's legacy sparse Blosc bricks need
  their existing reader or an explicit adapter.
- Existing mesh: replace `field` and `levels` with a local pinned OBJ/NPZ
  `mesh` descriptor. Declare `geometry.frame` and positive `voxel_um`;
  vertices must already use that frame. NPZ: numeric `vertices_xyz` Nx3,
  integer zero-based `faces` Mx3. OBJ: streamed triangles referencing already
  defined vertices, including negative indices. UV/materials are not textures;
  imported group 0 is not a measured phi=0 level.

Execution defaults: `tile=128` owned cells (max 256), `level_batch=4`,
`cache_bytes=0`, `reserve_bytes=0`, `mesh_bytes=536870912`. Each read includes
border nodes. NPZ declared lengths must fit the allowance before allocation;
use streaming OBJ for larger meshes. Cache retention is not total memory.

`mesh.sqlite` is the authoritative mesh/checkpoint; every tile transaction binds
geometry, source progress and hashes. Shared edges/exact-level nodes use global
keys. Every component is retained. `MESH.json` reports counts, area and topology;
`level-*.obj` exports the explicit level inventory. Remove `OUTPUT/STOP` before
resume. Complete resume verifies all referenced source payloads and preserves
mesh bytes. Changed OBJ/PNG/JSON files are preserved and rejected. Source,
geometry, levels, tile size and numerical-code changes fail; cache/reserve and
level-batch settings may change. Inputs must be immutable frozen checkpoints.

Views are bounded CPU orthographic triangle rasterizations with depth, two-sided
shading and transparent background. Defaults: 1024×1024, `triangle_batch=8192`,
front (+X, Z up) and oblique (35° azimuth, 20° elevation). Optional raw `levels`
filter retains a full-bundle camera/palette. Limits: 16M pixels, 65536 triangles
per batch, 32 views; names must be simple/unique, excluding `PLAN` and `RENDER`.
Empty or edge-on geometry may be blank. These are geometry images, not CT/ink.

`prepared render --config views.json` accepts the **render settings only**,
including an independent `reserve_bytes`. Output includes PNGs, per-view JSON,
`PLAN.json`, `RENDER.json` with mesh/code hashes, metadata and camera statistics.
The combined workflow propagates `execution.reserve_bytes` to rendering.

Python: `winding_field.surface_workflow.run_surfaces(config, output,
resume=False, render=True, stop=None)`, `render_bundle(mesh, output,
settings=None)`; `winding_field.surface_render.render_view(...)` streams RGBA.
Numerical geometry APIs remain `surface_sources.FieldSource`,
`surface_mesh.extract_mesh`, `import_mesh` and `MeshReader` in the library.

## Offline examples

After installing both packages, run these from the CLI checkout with new paths:

```bash
mkdir -p runs
python -I examples/block_field.py --output runs/blocks --stop-after-blocks 1
python -I examples/block_field.py --output runs/blocks --resume --prefetch
python -I examples/block_field.py --output runs/supervised --supervised --prefetch
python -I examples/surface_pipeline.py --output runs/surfaces --stop-after-tiles 3
python -I examples/surface_pipeline.py --output runs/surfaces --resume
python -I examples/surface_pipeline.py --output runs/surfaces --resume
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/benchmark_blocks.py \
  --output runs/performance-probe --repetitions 2
```

The block example checks an anisotropic translated analytic field, fixed source
hashes and prefetch parity. The surface example extracts two objects/levels,
checks visible PNGs and imports its OBJ. The probe compares complete serial and
bounded-parallel cycles, including durable storage, plus prepared-operator reuse;
it makes no universal speedup claim.

Direct prepared commands return 0 for budget/STOP, 75 for explicit resource
unavailability, 2 for input/integrity errors, 1 for numerical/operational failure.
`SWEEP_BUDGET_REACHED` is not global convergence. Quantized checkpoints and mesh
diagnostics do not establish physical sheet correctness. Disk admission measures
free space; it cannot reserve it against other jobs. Additional working memory
and Unix locking/fsync semantics are required.

## Migration

Replace imports of `field_workflow`, `surface_workflow`, `surface_render` with
their `winding_field.*` names and prefix former prepared commands with
`winding-field prepared`. Install both distributions together; only the CLI
registers the executable. Resume identities include the external library's code.
Use the original packages to resume runs created before this separation, or start
a new output. Existing receipts are never rewritten to bypass that check.
