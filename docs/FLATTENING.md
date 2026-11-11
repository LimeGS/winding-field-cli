# Flatten fields or meshes and sample CT

Start with [README section 5](../README.md). Choose the route before creating an
output directory:

| Goal | Use |
| --- | --- |
| Export selected levels from a saved CLI run | `export-sheets --run RUN --levels ... --export-tifxyz` below |
| Choose LSCM or SLIM on patches or whole sheets | `export-sheets --flattening-solver ... --scope ...` below |
| Run the original global LSCM recipe | [Original mapping](#new-ct-run-with-original-mapping) |
| Continue a normal CT run | [Native output](#native-output); use the saved run configuration |
| Align neighbouring W levels on one UV grid | [UVW](UVW.md) |
| Export directly from the field, without a mesh | [Experimental direct-field TIFXYZ](#experimental-direct-field-tifxyz) |

CT images and optional XYZ TIFFs are different products. Missing CT does not
necessarily mean missing geometry; always inspect masks and coverage.

## Export from a saved run

After the README's `run --until field`, export without creating configuration files:

```sh
winding-field export-sheets --run runs/175a-roi --levels -31 \
  --export-tifxyz --output exports/W-31
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

To override output coordinates, pitch or memory admission, supply an explicit
`--flattening-config` using the original recipe's format below. Do not combine
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

## Native run flattening

New `winding-field run` and `plan` requests with `--surface-output native`
default to `--flattening-method slim`. Each bounded local chart uses the existing
near-planar projection or LSCM initialization, then **10 libigl SLIM iterations**
with symmetric-Dirichlet energy. The base package requires `libigl==2.6.3`;
`.[tifxyz]` adds TIFF export only. No Flatboi executable or external export
pipeline is involved.

Seeds and optimized UVs must pass the unchanged finite, positive-area,
non-overlap and edge-stretch gates (0.25–4). Failed charts remain accounted for;
there is no silent fallback to unoptimized UVs. Exact source geometry/IDs and
the existing seam welding, cuts, recovery and rendering remain unchanged.
This is local optimization, not a whole-sheet or globally injective SLIM map.
Ten iterations are a work bound, not a convergence or physical-acceptance claim.

`--flattening-method default` retains historical local mapping;
`--flattening-method original --flattening-config FILE` retains global LSCM.
Explicit `slim` is rejected for `--surface-output atlas|tifxyz`; omitting the
method on those branches keeps their historical behavior. `--export-tifxyz`
still adds mesh-derived TIFFs to native renders.

The CLI saves the resolved method in `RUN.json`; native identities bind the
algorithm, initialization, energy and iteration bound. Resume uses the saved
method without backfilling missing fields: historical configs with no method
still use historical mapping. As before, implementation/runtime changes require
the original package or a new output. Completed charts are verified/reused;
interruption within SLIM restarts only the uncommitted chart on continuation.

## Explicit solver × scope matrix

`export-sheets` accepts **both** `--flattening-solver lscm|slim` and
`--scope patches|sheet`. Follow the [installation guide](../README.md)
and run on Linux. The four
combinations use the same source-preserving preparation and tiled CT exporter:

| Selection | Domain | Algorithm |
| --- | --- | --- |
| `--flattening-solver lscm --scope patches` | Deterministic independent patches | libigl LSCM |
| `--flattening-solver slim --scope patches` | The same patches and cuts | Positive mean-value initialization → native Flatboi/SLIM |
| `--flattening-solver lscm --scope sheet` | Whole connected component | libigl LSCM |
| `--flattening-solver slim --scope sheet` | Whole connected component | Positive mean-value initialization → native Flatboi/SLIM |

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

For SLIM, add `"app_run": "/opt/vc3d/AppRun"`. The default is the pinned
`installed-2f89bfc9` Flatboi binary used by `constant-sheet`. To select the
official Villa stable release recorded in [native provenance](../provenance/flatboi.json),
also add `"native_recipe": "historical"`. Both launcher and binary hashes
must match the selected recipe. Its settings remain cap200, batches10, two threads, symmetric
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
existing `winding_field.patch_mapping.PatchMapping` reader for original source
keys. Each tile's `mapping.npz` stores XYZ and source face rows. `--export-tifxyz`
adds the existing validated XYZ TIFF products to nonempty tiles.

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

## New CT run with original mapping

After preparing `flattening.json` and installing the `tifxyz` extra:

```bash
winding-field run --source /data/scroll.zarr --voxel-um 8.64 \
  --field-spacing-um 17.28 --levels -31 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output runs/original
winding-field resume runs/original
```

Add `--roi` for a region or `--until mesh` to checkpoint before mapping.
Choose the method when creating the run; do not edit saved configuration.
Omit `--export-tifxyz` for CT/mask/map products only.

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
