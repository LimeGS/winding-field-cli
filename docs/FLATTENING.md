# Field-to-TIFXYZ setup

Start with the [commands in README section 5](../README.md). Use original mapping
for global CT plus optional XYZ TIFFs; native runs use their saved configuration.

## Explicit solver × scope matrix

`export-sheets` accepts **both** `--flattening-solver lscm|slim` and
`--scope patches|sheet`. Install `.[tifxyz,ct]` and run on Linux. The four
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
`ct_workers`, `cache_dir`, `cache_mib`, `cache_gib`, and `ct_sampler`; they retain
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
`xyz = (physical_xyz_um - output_origin_um) / output_voxel_um`. Valid TIFF XYZ
must be nonnegative; −1 marks invalid nodes.

`--source` must be the matching CT. `--voxel-um` supplies its scale if metadata
lacks units, separately from the output frame. Select CT pyramid level with
`--level`; select winding values with `--levels`.

## Reuse an existing mesh

In the README export command replace `--field-config field.json --levels -31`
with `--mesh-bundle /data/run/meshes`. The immutable bundle must contain
`MESH.json`, `RUN.json` and `mesh.sqlite`; it already records its levels.

## New CT run with original mapping

After preparing `flattening.json` and installing `.[tifxyz]`:

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
are distinct final markers; only READY certifies the declared supported ROI at
the recorded tolerance. Verification exits 0 for verified READY, 2 for verified
PARTIAL; resource blocks exit 75. Neither proves physical-sheet correctness.
macOS production runtime is unsupported. No whole-W throughput guarantee is made.

## Technical reference

- [Original mapping and coordinate conventions](FORMATS.md#original-recipe-products)
- [Native masks, time and coverage](FORMATS.md#native-global-surfaces)
- [Direct request, resource and verification contracts](FORMATS.md#direct-tifxyz-experimental)
