# Flatten fields or meshes and sample CT

Start with [README section 6](../README.md). Choose the route before creating an
output directory:

| Goal | Use |
| --- | --- |
| Export selected levels from a saved CLI run | `export-sheets --run RUN --levels ...` below |
| Choose LSCM or SLIM on patches or whole sheets | `export-sheets --flattening-solver ... --scope ...` below |
| Run the original global LSCM recipe | [Original mapping](#new-ct-run-with-original-mapping) |
| Continue a normal CT run | [Native output](#native-output); use the saved run configuration |
| Align neighbouring W levels on one UV grid | [UVW](UVW.md) |
| Export directly from the field, without a mesh | [Experimental direct-field TIFXYZ](#experimental-direct-field-tifxyz) |

CT images and optional XYZ TIFFs are different products. Missing CT does not
necessarily mean missing geometry; always inspect masks and coverage.

## Flag-first external mesh export

Native recipe files remain supported, but are no longer required. On Linux:

```bash
winding-field export-sheets --mesh-bundle runs/paris4-flags-W-60 \
  --source /data/paris4-original-ct.zarr --voxel-um 9.6 --level 0 \
  --flattening-solver lscm --scope sheet \
  --output-frame-id 'PHercParis4/20260411134726:L2' \
  --output-origin-xyz-um 0 0 0 --output-voxel-um 9.6 \
  --pitch-um 19.2 --memory-gib 6 --output exports/paris4-W-60
```

Replace CT path/level with the **actual calibrated dataset**, not the field's
coarse Zarr. The pitch/memory above are an explicit example, not measured
full-scroll settings or the user's production flattening recipe. Reuse your
approved values. LSCM needs no native Flatboi; SLIM always runs Flatboi (Linux
x86_64; see [SLIM runtime](#slim-runtime-flatboi)) with Linux admission. Complete mappings, automatic values and file/flag
precedence are in [CLI-FLAGS](CLI-FLAGS.md).

## Export from a saved run

After the README's `run --until field`, export without creating configuration files:

```sh
winding-field export-sheets --run runs/175a-roi --levels -31 \
  --output exports/W-31
```

`--run` selects the original global LSCM recipe. It loads the completed field
receipt/checkpoint, CT source, CT pyramid level and scale from the saved run.
It verifies field hashes and geometry, current CT metadata and previously
consumed encoded CT chunks before creating the export. Keep `RUN.json`,
`FIELD.json`, `field/` and `source.sqlite` intact. This is a new read-only export,
not a resume of the producer; the old producer implementation need not match
the exporter. Revalidating the saved field and CT history can take time on large runs.
The export retains a copy of the producer's CT hash ledger, so those expected
hashes also protect rendering after admission. Resume rejects lost producer rows
or a missing/symlinked export ledger instead of accepting fresh source bytes.

Default output frame: the saved field's frame identity, `voxel_um: 1`,
`origin_xyz_um: [0,0,0]`. Thus XYZ TIFFs contain physical micrometres; nonzero
CT origins and anisotropic spacing remain accounted for. Original mapping's
existing automatic raster pitch and 4 GiB memory-admission estimate are unchanged.
Do not treat a successful diagnostic export as physical-sheet acceptance.

The exporter creates `field.json` and `flattening.json` inside `OUT`, binding their
hashes and source receipts in `EXPORT.json`. Repeat the identical command with
`--resume`; edited/missing configuration or changed sources are rejected, never
silently regenerated. Use a fresh output for changed settings. The output must
be separate from the source run, not inside or above it.

To override output coordinates, pitch or memory admission, supply output-frame
flags, `--pitch NUMBER|auto`, `--memory-gib`, or optional `--flattening-config`
using the original recipe's format below. Do not combine
`--run` with `--source`, `--level`, `--voxel-um`, another input selector, patches,
or the solver/scope matrix: saved inputs are authoritative. `--levels` retains
the existing **raw field isovalue** contract, not automatic W recalibration.

The shared manual/quick/Docker installation already includes `plot` and `tifxyz`.
Only for an environment installed without these extras, run from the checkout:

```sh
uv pip install --constraint docker/cli.in '.[tifxyz]'
```

For external fields/meshes, keep using the explicit configuration routes below.

Remote field exports can use the same `zarr-v2` field descriptor documented in
[Prepared: remote Zarr v2](PREPARED.md#read-a-remote-zarr-v2-field), including
`--field-config` inputs. A flatten/export field-config contains `field` and
`geometry` from the [Paris4 example](../examples/paris4-remote.json); levels are
still supplied with `--levels` in raw phi units. This adds remote **field** reads,
not CT texturing or a different flattening algorithm.

## SLIM runtime (Flatboi)

SLIM means Flatboi's symmetric-Dirichlet optimizer in every pipeline (native runs,
`export-sheets` patches and sheets). The default runtime is the hash-pinned VC3D
AppImage in [native provenance](../provenance/flatboi.json) (recipe `historical`),
which bundles its own libraries and loader. When no `app_run` is given it is
resolved in this order:

1. `WINDING_FLATBOI_APP_RUN` (an extracted `AppRun`);
2. `/opt/vc3d/AppRun` (the container image);
3. the user cache (`~/.cache/winding-field/flatboi`, or `WINDING_FLATBOI_CACHE`);
4. a one-time verified download and extraction of the AppImage
   (`python -m winding_field.flattening.flatboi --provision` does it ahead of time).

Downloads are disabled with `--offline` runs or `WINDING_FLATBOI_OFFLINE=1`. The launcher
and binary SHA256 are verified on every use. Flatboi is Linux x86_64 only: on other
platforms SLIM is rejected before a run is created and native runs default to LSCM.
The vendored `installed-2f89bfc9` binary lacks its runtime libraries, so that recipe
requires an explicit `app_run` from an installation that provides them.

## Native run flattening

New `winding-field run` and `plan` requests with `--surface-output native`
default to `--flattening-method slim` on Linux x86_64 (`default` elsewhere), and
native SLIM defaults to `--flattening-scope sheet`: each edge-connected piece is cut to
a disk, seeded with a global LSCM and solved by one Flatboi SLIM call, so there are no
welded chart seams (same route as `scroll render-ct`, see docs/SCROLL.md). A piece that
cannot be flattened whole uses the chart route below and is listed in `LEVEL.json`
`sheet_flatten.fallback`. A piece with holes keeps them (no cut from a hole to the
outline). A piece with handles (tunnels a few voxels wide, common in a fine scroll
field) first loses the faces around each tunnel: every handle is found along its own
short loop and the faces within a quarter of that loop's length leave the piece, as do
the few faces on which the seed folds. Those faces are flattened as small pieces of their
own and counted in `sheet_flatten.pieces[].aside_faces`; `handles` records the genus, the
loops and their lengths. What is left of such a piece can still close around the scroll
axis: a rim at least a quarter as long as the longest one is an end of the wrap, not a
hole, and gets the shortest seam to the outline, so the wrap unrolls to a strip instead
of mapping as a ring (`seam_vertex_copies` counts the seam). The seed is the global LSCM.
A piece that bends around one axis has a second seed, its unrolled map (arc length of its
median cross-section against the position along the axis, `initialization: unroll`): it
is used when LSCM folds on more than 5% of the area, and for a second solve when the
LSCM-seeded one ends outside the stretch gate. It folds on more faces than LSCM, so it is
not the first choice. When neither is valid, the seed is the mean-value map of the piece
with its holes capped.
`--flattening-scope charts` selects the chart route for the
whole run; runs created before 2.0.0 keep their saved chart scope on resume.

With `--flattening-scope charts`, each
bounded local chart gets the existing near-planar projection or LSCM seed; seeded
charts from consecutive face batches (up to 65,536 faces) are then optimized together
by **one Flatboi call** (symmetric Dirichlet, at most 200 iterations, tolerance 1e−6).
Charts are laid out in disjoint UV boxes for the call; translation does not change
the energy, and overlapping seeds would make the solve ~30× slower.

Seeds and optimized UVs must pass the unchanged finite, positive-area,
non-overlap and edge-stretch gates (0.25–4). A chart whose Flatboi result fails is
split and re-solved; one that cannot be split further is recorded as failed. A native
failure (for example PaStiX's negative diagonal) bisects the batch until the failing
chart is isolated. There is no silent fallback to seed UVs. Exact source geometry/IDs
and the existing seam welding, cuts, recovery and rendering remain unchanged (the
bounded weld recovery step still uses libigl). This is local optimization, not a
whole-sheet or globally injective SLIM map.

Measured on 120 curved 1,024-face charts: Flatboi reached the same symmetric-Dirichlet
energy as 10 libigl SLIM iterations (mean 4.00057, no flips) in 9.7 s versus 3.1 s, i.e.
about 80 µs per face including process startup.

`--flattening-method default` keeps projection/LSCM charts;
`--flattening-method original --flattening-config FILE` retains global LSCM.
Explicit `slim` is rejected for `--surface-output atlas|tifxyz`; omitting the
method on those branches keeps their historical behavior. Mesh-derived TIFFs are
always added to native renders.

The CLI saves the resolved method in `RUN.json`; native identities bind the
solver (`flatboi-slim`), recipe, launcher/binary hashes, iteration bound, tolerance
and batch size. Implementation/runtime changes require the original package or a new
output. Completed charts are verified/reused; interruption within a SLIM group restarts
only that uncommitted group on continuation.

**Host-side exact acceleration (comparison to `eea0cbb`).** Seed OBJ formatting,
verified UV readback and integer mesh-topology checks use bounded batches or
reuse sorted edges. The numerical solver, seed selection, threads and checkpoint
batches retain their existing settings. On a 130,050-triangle grid on macOS arm64,
four interleaved runs per version gave median seed writes of 273→96 ms, indexed
seed readback 471→166 ms, and component cutting/topology 849→285 ms. Run
`PYTHONPATH=src python benchmarks/flatboi_prep.py` in a clone containing the
baseline to compare stage times and exact products. On a separate real 1,007-face
175A chart, Linux x86_64 Flatboi (pinned historical binary, one numerical thread)
converged in four steps with identical UV bytes and final energy `8.0004073106`
before and after; the small solve itself took about 3.9 s on both under emulation.

## Explicit solver × scope matrix

`export-sheets` selects `--flattening-solver lscm-slim|lscm|slim` and
`--scope patches|sheet`. Without a selection (and without `--run` or a historical
`--flattening-method`), it uses **`lscm-slim` on whole sheets** on Linux x86_64 and
`lscm` elsewhere; an omitted `--scope` is `sheet`. Every combination uses the same
source-preserving preparation and tiled CT exporter:

| Selection | Domain | Algorithm |
| --- | --- | --- |
| `--flattening-solver lscm-slim --scope sheet` (default) | Whole connected component | Global libigl LSCM seed → native Flatboi/SLIM |
| `--flattening-solver lscm-slim --scope patches` | Deterministic independent patches | LSCM seed per patch → native Flatboi/SLIM |
| `--flattening-solver lscm --scope patches` | Deterministic independent patches | libigl LSCM |
| `--flattening-solver lscm --scope sheet` | Whole connected component | libigl LSCM |
| `--flattening-solver slim --scope patches` | The same patches and cuts | Positive mean-value initialization → native Flatboi/SLIM |
| `--flattening-solver slim --scope sheet` | Whole connected component | Positive mean-value initialization → native Flatboi/SLIM |

`lscm-slim` falls back to the mean-value seed for a domain whose LSCM map folds (Flatboi
needs a valid seed) and records `lscm_rejected` in that domain's `INITIAL.json`. On a
38k-face rolled sheet the LSCM seed converged in 8 Flatboi steps (p95 area distortion
0.026) where the mean-value seed did not converge in 200 (0.041) and took 6x longer.

Minimal `matrix.json` for LSCM:

```json
{
  "output_frame": {
    "id": "my-scroll/my-scan",
    "origin_xyz_um": [0, 0, 0],
    "voxel_um": 9.6
  },
  "pitch_um": 19.2,
  "memory_gib": 6
}
```

For SLIM, `app_run` is optional: the pinned `historical` AppImage is resolved as in
[SLIM runtime](#slim-runtime-flatboi). To use the `installed-2f89bfc9` binary instead,
set `"native_recipe": "installed-2f89bfc9"` and an `app_run` whose installation
provides its libraries. Both launcher and binary hashes must match the selected recipe. Its settings remain cap200, batches10, two threads, symmetric
Dirichlet and tolerance1e−6. SLIM runs require a cgroup v2 envelope:

```sh
systemd-run --user --wait --pipe -p MemoryMax=21G -p MemorySwapMax=64G \
  winding-field export-sheets --mesh-bundle /data/meshes \
  --flattening-solver slim --scope sheet --flattening-config matrix.json \
  --source /data/scan.zarr --voxel-um 9.6 --output exports/slim-sheet
```

Change the two selection flags and output directory to choose another cell.
`--field-config field.json --levels -59 -60 -61` may replace `--mesh-bundle`;
it uses the existing supported field formats and preserves unsupported holes.
Validated RANGE wrappers are also accepted; their stored slot is not a constant
W label. Separate stored mesh levels are always exported independently.

### Preparation and comparison

- Patch membership depends only on geometry and `max_patch_faces` (default4096),
  not on solver success. `max_charts` defaults to4096. These two settings are
  accepted only for `patches`.
- Sheet scope uses each complete connected component. It never switches to
  patches to evade a memory/topology limit. Disconnected components are retained
  and placed separately.
- Both solvers use the same existing-edge disk cuts. With sheet scope, optional
  `centerline` (an XYZ array with increasing Z) selects the current +X reference
  seam. Optional `reference_z` records the anchor for every component crossing
  that height. These coordinates use the declared **output frame's voxel units**.
- Every source face and its orientation survive the cuts. Seam copies retain
  the source vertex IDs and exact source coordinates in `field_xyz`.
- UVs receive isotropic edge-length calibration and increasing-Z orientation
  where Z varies. Independent domains are then translated into separate atlas
  regions, without welding or nonlinear seam deformation. Their layout is
  recorded; patch mode does not reproduce whole-sheet optimization.
- Solver failures abort publication. Inverted/distorted/overlapping regions are
  reported with diagnostic and scientific masks; no physical acceptance is
  claimed. Neither algorithm silently substitutes for the other.

### Outputs and resume

```text
exports/slim-sheet/
├── EXPORT.json, CT-IDENTITY.json, MATRIX.json
├── ct/ct-chunks.sqlite
└── level-0000/
    ├── LEVEL.json, map.npz, MAPPING.json, keys.sqlite, ct-used.sqlite
    ├── flattening/                 # partition/cut/solver reports and checkpoints
    └── render/tile-*/              # image.png, mask.png, mapping.npz, TILE.json
```

The map contains exact source face/vertex IDs, face rows and domain IDs; use the
existing `winding_field.flattening.patch_mapping.PatchMapping` reader for original source
keys. Each tile's `mapping.npz` stores XYZ and source face rows. The existing
validated XYZ TIFF products are always added to nonempty tiles.

Repeat the identical command with `--resume`. Solver, scope, source geometry,
configuration, runtime, code and CT identity are bound. Completed levels are
hash-verified and reused; interrupted mapping retains completed domains and
native UV checkpoints. Interrupted rendering retains its previous directory
as `render-interrupted-*` and restarts that level's raster. CT ledger entries,
including previously missing chunks, are checked before reuse; committed levels
carry a bound CT ledger snapshot. Native continuation is a UV warm start, not
bit-identical uninterrupted SLIM state.

Use SIGTERM or an output-root `STOP` file to stop extraction, native solving or
rendering. The native runner checks the outer command's resource/stop guard as
well as its own checkpoint bounds. It also honors
`level-*/flattening/domain-*/native/STOP`. Remove stop markers before resuming.
As with existing checkpoint code identities, a checkpoint from an older runner
must use its frozen runtime or the explicit `constant-sheet import-native`
migration with the original runner SHA; it is never relabeled as a new run.
Changing `native_recipe` requires a new output; import-native also retains the
original binary identity and cannot migrate checkpoints between these releases.

Sampling stays isotropic at `pitch_um` (default19.2), window0–255, with bounded
native CT caching. Common rendering settings are `tile_size`, `raster_workers`,
`ct_workers`, `cache_dir`, `cache_mib`, and `cache_gib`; they retain
the patch export validation. At least8GiB disk reserve is required. Geometry/
raster memory remains capped at6GiB, and existing topology/map bounds still
apply. Selecting the matrix does **not** establish feasibility for full fine
sheets or bypass their current admission limits.

### Historical compatibility

`--flattening-method original`, `--flattening-method patches` and the staged
[`constant-sheet`](CONSTANT-SHEET.md) commands remain available with their
existing recipes. Do not mix `--flattening-method` with the new pair. In
particular, historical patches perform registration/welding, whereas the
explicit matrix keeps independent domains; `original` retains its historical
domain selection and canvas. The new matrix is not a bit-identical alias for
either historical exporter.

## Configuration files

### `field.json`: the input field

Replace paths, hashes and geometry with your actual values:

```json
{
  "field": {
    "kind": "npy",
    "path": "/data/field.npy",
    "sha256": "REPLACE_WITH_FILE_SHA256"
  },
  "geometry": {
    "frame": "my-scroll/my-scan",
    "voxel_um": 1.0,
    "origin_xyz": [0.0, 0.0, 0.0],
    "spacing_xyz": [17.28, 17.28, 17.28],
    "shape_zyx": [100, 200, 200],
    "sheet_step": 1.0
  }
}
```

- Get the hash with `shasum -a 256 /data/field.npy`. Use absolute paths.
- The array is ZYX; origin and spacing are XYZ. With `voxel_um: 1`, coordinates
  are micrometres. With `voxel_um: 8.64`, spacing 2 means 17.28 µm.
- Optional `field.mask` is `{"path":"/data/mask.npy","sha256":"MASK_SHA256"}`:
  a boolean array of the same shape. Nonfinite field values are unsupported.

For a CLI block checkpoint, replace the `field` object with:

```json
{
  "kind": "chunks",
  "path": "/data/run/field/CHECKPOINT.json",
  "sha256": "REPLACE_WITH_CHECKPOINT_SHA256",
  "support_policy": "observed"
}
```

Copy `geometry` from that run's `FIELD.json`. Keep the immutable checkpoint and
its chunk files together. The existing field determines coarse/fine resolution.
Missing chunks are unsupported; `observed` also requires observation support.
Scalar `--levels` are raw field isovalues. For a nominal winding value W, use
`raw_level = W * sheet_step`; the values coincide only when `sheet_step` is 1.

### `flattening.json`: the output coordinates

```json
{
  "output_frame": {
    "id": "my-scroll/my-scan",
    "origin_xyz_um": [0.0, 0.0, 0.0],
    "voxel_um": 8.64
  },
  "memory_gib": 4.0
}
```

Set the ID, origin and voxel size to the destination scan's frame. This writes
XYZ in 8.64 µm voxels; it does not change field resolution. Conversion is
`xyz = (physical_xyz_um - output_origin_um) / output_voxel_um`. Mesh-derived TIFFs
use float32, nonnegative valid XYZ and −1 for invalid nodes. The separate
[UVW exporter](UVW.md#products) uses float64 and NaN instead; do not interchange
their validity conventions.

`--source` must be the matching CT. `--voxel-um` supplies its scale if metadata
lacks units, separately from the output frame. Select CT pyramid level with
`--level`; select winding values with `--levels`.

## Reuse an existing mesh

In the README export command replace `--field-config field.json --levels -31`
with `--mesh-bundle /data/run/meshes`. The immutable bundle must contain
`MESH.json`, `RUN.json` and `mesh.sqlite`; it already records its levels.

**Chained after extraction.** `prepared mesh ... --then-flatten DIR` runs that export
as a second, separate step once the extraction is `MESH_READY`, and never otherwise:

```sh
wf prepared mesh --config surface-run.json --output /outputs/meshes \
  --then-flatten /outputs/flat --flatten-cli-config flatten.yaml
```

The second step is an ordinary `export-sheets --mesh-bundle /outputs/meshes --output
DIR` process with its own receipts and exit code; `--flatten-cli-config` is its
`--cli-config` (`source`, `voxel_um`, `flattening_solver`, `scope`, `stages`,
...). Without `output_frame_id` or `flattening_config` there, the output frame is the
mesh publication's (`MESH.json` → `metadata`). An existing DIR is resumed with
`--resume`. `CHAIN.json` in the mesh output records both steps (status, command,
return code); a failed flatten step keeps the mesh step's receipts and returns its own
exit code. DIR must be outside the mesh output. It is mesh-only: not with
`--parallel-levels`, `surfaces` or `render`.

## New CT run with original mapping

After preparing `flattening.json` and installing the `tifxyz` extra:

```bash
winding-field run --source /data/scroll.zarr --voxel-um 8.64 \
  --field-spacing-um 17.28 --levels -31 \
  --flattening-method original --flattening-config flattening.json \
  --output runs/original
winding-field resume runs/original
```

Add `--roi` for a region or `--until mesh` to checkpoint before mapping.
Choose the method when creating the run; do not edit saved configuration.
TIFXYZ is always written; 2.0 removed `--export-tifxyz` and `--no-export-tifxyz`.

## Resume and inspect results

For standalone `export-sheets`, repeat the same command with `--resume`.
The output's parent must already exist (for example, `mkdir -p exports`).
Remove an `OUT/STOP` marker before resuming. Changed inputs or settings need a
new output directory.

Each requested level produces `original/level-NNNN/`:

- `image.png`: flattened original CT, fixed 4096² canvas, contrast 0–180.
- `mask.png`: 0 unmapped, 1 mapped/CT missing, 2 observed, 3 ambiguous.
- `map.npz`, `LEVEL.json`: source correspondence, mapping diagnostics and hashes.
- Optional `tifxyz/`: XYZ TIFFs, `valid-mask.png`, metadata and readback receipt.

`level-0000` means the first requested level, not W=0. Copy the **whole**
`tifxyz/` directory. CT missingness and geometry validity are separate; black
image pixels alone cannot distinguish them. Readback is not physical acceptance.

The recipe selects the largest-area connected component, retains holes and
reports excluded source area. It tries curvature cuts to obtain a genus-zero
domain before global LSCM; unresolved handles/degeneracy fail. Fine geometry can
exceed available memory: `memory_gib` is an admission estimate, not a hard cap.
The full component and sparse solver are resident; no automatic decimation occurs.

The default fits eligible UV extent into the canvas. Optional `"pitch": NUMBER`
in `flattening.json` is a minimum in output-frame voxels per pixel; larger domains
increase it to fit. This is independent of CT and field resolution.

## Native output

`mesh RUN`, `flatten RUN` and `render RUN` honor the run's saved branch. New runs
default to native, with meshes and CT under `surfaces/`; `flat/` and `index.html`
belong to `--surface-output atlas`. Choose W values with `--levels` at creation.

Native `--coverage-target` measures mapped mesh area, not observed CT or certified
field coverage. `--surface-time-limit` defaults to 1800 active seconds per W.
`resume RUN --extra-surface-seconds 600` extends unfinished work, but a published
PARTIAL level is immutable. Optional native TIFXYZ can be unavailable or partial
even when OBJ/PNG products exist. Inspect `LEVEL.json` and `TIFXYZ.json`.

## Experimental direct-field TIFXYZ

This separate **Linux-only** route projects a bounded coarse guide onto the fine
field without native OBJ extraction. Install `.[tifxyz]` first. Save `export.json`:

```json
{
  "schema": "winding-field-tifxyz-v1",
  "levels": [-4.0],
  "output_frame": "selected-source",
  "memory_gib": 3,
  "scratch_gib": 16,
  "reserve_disk_gib": 12.25,
  "max_pixels": 64000000
}
```

Choose a W present in your field. `selected-source` means its SHA-bound isotropic
CT level, not necessarily L0. The output parent must exist, outside the source run:

```bash
mkdir -p exports
winding-field tifxyz runs/old-field --output exports/direct --config export.json
winding-field tifxyz runs/old-field --output exports/direct --resume
winding-field verify-tifxyz exports/direct
```

Standalone export never opens CT and preserves the producer run. For integrated
new runs select `--surface-output tifxyz --tifxyz-config export.json`; do not
combine with original mapping. `mesh`/`flatten` do not apply to that branch.

Ctrl-C requests a stop. Standalone exports also honor `OUT/STOP`; remove it before
resume/verification. If admission created no output root, retry the initial command.

Strict coverage is default. Set `"coverage_policy":"allow_partial"` **before**
starting to permit partial coverage after geometry gates pass. READY and PARTIAL
are distinct final markers; only READY passes the exporter's coverage checks for
the declared supported ROI at the recorded tolerance. Verification exits 0 for verified READY, 2 for verified
PARTIAL; resource blocks exit 75. Neither proves physical-sheet correctness.
macOS production runtime is unsupported. No whole-W throughput guarantee is made.

## Technical reference

- [Original mapping and coordinate conventions](FORMATS.md#original-recipe-products)
- [Native masks, time and coverage](FORMATS.md#native-global-surfaces)
- [Direct request, resource and verification contracts](FORMATS.md#direct-tifxyz-experimental)

### Code layout

All parameterization code lives in `src/winding_field/flattening/`. The old top-level
module names (`winding_field.patch_pipeline`, `python -m winding_field.patch_continue`, …)
were removed in 2.0.0; import and run the modules below as `winding_field.flattening.<module>`.

| Module | Was | Role |
|---|---|---|
| `selection` | — | Valid `--flattening-method`, `--flattening-solver`, `--scope` and export method names |
| `lscm` | — | Shared libigl LSCM primitive; every LSCM solve goes through it |
| `flatboi`, `flatboi_checkpoint` | `flatboi_checkpoint` | Flatboi SLIM runtime, batched solves, checkpointed batches (only SLIM backend) |
| `native_charts` | `native_geometry` | Native-run local charts, seam welding, assembly |
| `native_sheet` | `native_global` | Cut-to-disk, mean-value and ARAP whole-sheet maps |
| `handles` | — | Short loop per handle of a sheet piece; faces set aside around each tunnel |
| `matrix`, `matrix_pipeline` | `flattening_matrix`, `matrix_pipeline` | `export-sheets` solver × scope matrix |
| `patches`, `patch_mapping`, `patch_continue`, `patch_visuals` | `patch_pipeline`, … | Historical Flatboi patches |
| `original`, `original_pipeline`, `original_tifxyz` | `original_recipe`, … | Original global LSCM recipe |
| `sheet_prepare`, `sheet_finish`, `sheet_atlas*`, `constant_sheet`, `run_sheet_export` | same | Whole-sheet preparation, publication, atlases |
| `legacy_charts` | `flatten` | Legacy linked-chart LSCM for `--surface-output atlas` |

Kernel and code identities hash the moved files, so the refactor changes recorded code
hashes but not numerical outputs.
