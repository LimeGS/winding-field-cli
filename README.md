# winding-field-cli
![Illustrated workflow: original CT → Lasagna surfaces + Qual V2 fibres/splines → winding field → 3D meshes → flattened charts](docs/assets/workflow-banner.png)

Build a winding field from a scroll's CT, then extract 3D meshes and CT-textured
global surfaces. By default, Lasagna supplies surface observations, Qual V2 supplies
fibres and splines, and native surfaces supply the output pipeline. Runs save each
stage so you can stop and continue later.

Follow sections **1–2** for a full scroll, or **1 → 3** for a smaller region.
Section **4** covers stopping and resuming; **5** extracts meshes; **6** covers
inputs you already have.

<details>
<summary><strong>1. Install</strong></summary>

**CLI and numerical library.** Requires Python **3.11 or 3.12**, Linux/macOS.
Run these commands in a terminal; keep the virtual environment active for the
examples below. Substitute `python3.12` for `python3.11` if needed.

```bash
git clone https://github.com/LimeGS/winding-field-cli.git
cd winding-field-cli
python3.11 -m venv .venv
source .venv/bin/activate
CORE_REV=$(python -c "import json; print(json.load(open('provenance/core.json'))['revision'])")
python -m pip install \
  "winding-field-experiments @ https://github.com/LimeGS/winding-field/archive/$CORE_REV.tar.gz" .
winding-field doctor
```

**Lasagna runtime.** Default CT runs also need the official Lasagna package. It uses
Python 3.14 in a separate environment; install it once without activating it:

```bash
python -m pip install uv
uv python install 3.14
uv venv --python 3.14 ~/.cache/winding-field/lasagna-venv
uv pip install --python ~/.cache/winding-field/lasagna-venv/bin/python \
  'vesuvius-lasagna @ https://github.com/ScrollPrize/villa/archive/b29f310957c7b29e7d332f9876e0e487819cfaa3.tar.gz#subdirectory=lasagna'
```

The CLI finds this runtime automatically. Lasagna and Qual weights download on
first use. For CUDA, both environments need a compatible CUDA-enabled PyTorch:
[runtime setup](docs/LASAGNA.md#one-time-runtime-setup).

**Optional installation check:** run the synthetic pipeline offline, without
model downloads, then open `runs/demo/index.html` in a browser:

```bash
winding-field demo --output runs/demo
```

</details>

<details>
<summary><strong>2. Run a winding field for a full scroll</strong></summary>

Select the original CT volume. This example uses the public PHerc0175A scan;
you can replace its URL with a local 3D Zarr v2 path. **No `--roi` means the entire
selected CT level.**

First inspect geometry and storage estimates with `plan`, then start the run:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'

winding-field plan --source "$CT" --voxel-um 8.64 \
  --field-spacing-um 300 --device cuda:0 --ct-mode stream \
  --until field --output runs/175a-full

winding-field run --source "$CT" --voxel-um 8.64 \
  --field-spacing-um 300 --device cuda:0 --ct-mode stream \
  --until field --output runs/175a-full
```

- `--until field` runs Lasagna → Qual predictions/splines → field, then stops.
- `--field-spacing-um 300` selects an example **coarse 300 µm solve grid**.
  Choose your field resolution here; it does not reduce observation preparation.
- `--ct-mode stream` fetches CT chunks as needed. Prediction products and tracing
  scratch still need disk/RAM; inspect the estimates before a whole-scroll run.
- `--device cuda:0` uses the first GPU. Use `--device cpu` in both commands for CPU.
- `--output` must be a **new directory**. `plan` does not create the run.

**Results:** `runs/175a-full/field/` holds the field and `CHECKPOINT.json`;
`FIELD.json` records geometry and solver status. `fibres/fibres.afv` retains the
splines, and `lasagna-source/output/` retains Lasagna predictions.

The default Qual model requires **uint8, isotropic 8.64–9.6 µm observations**.
For another acquisition, set `--voxel-um` to its actual spacing and choose
`--level` / `--stride` accordingly. Field spacing is independent of CT spacing.
`--field-spacing-um` accepts any positive pitch that fits at least two nodes per
axis; omitting it uses the observation grid, including its anisotropy. Unitless
CT needs `--voxel-um`. For catalog inputs use `--scroll ID --scan ID`; ambiguous
acquisitions need an exact scan or `--source`. A matching local chunk mirror can
be supplied with `--local /data/mirror.zarr` alongside a remote source.

</details>

<details>
<summary><strong>3. Run a field for a specific section of a scroll</strong></summary>

Add `--roi Z0 Z1 Y0 Y1 X0 X1` to limit the output to a region. Bounds are in
**voxels of the selected CT level**, with exclusive upper bounds. This example
selects a **48 × 48 × 48 voxel** region and solves a **17 µm field** on CPU:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'

winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 \
  --field-spacing-um 17 --device cpu --ct-mode stream \
  --until field --output runs/175a-roi
```

- Replace the six bounds to select your region; use `plan` instead of `run` to
  inspect its grids first. Inference reads context around the ROI, but the output
  stays inside it.
- Choose `--field-spacing-um` independently of the CT resolution. The ROI must
  contain at least two field nodes per axis.
- Results are in `runs/175a-roi/field/`. Use this run path in sections 4 and 5.

To create intermediate inputs from scratch, replace `--until field` with
`--until prepare` for Lasagna + Qual predictions, or `--until fibres` to also trace
and export `fibres/fibres.afv`. Continue that same run with `winding-field field RUN`.

</details>

<details>
<summary><strong>4. Stop, inspect and resume a run</strong></summary>

Point `RUN` at the run directory, not its `field/` subdirectory. While a job is
running, press **Ctrl-C** or create `STOP` from another terminal. It saves
committed work and stops at a bounded operation; wait for the process to exit.

```bash
RUN=runs/175a-full
winding-field status "$RUN"               # stage status, timings and saved settings
touch "$RUN/STOP"                         # request a checkpointed stop
```

After it exits, choose how far to continue:

```bash
# Continue only through field construction:
winding-field field "$RUN"

# Or continue to the selected pipeline's terminal stage:
winding-field resume "$RUN"
```

Both commands clear `STOP`, validate saved work and reuse completed stages. A
run intentionally stopped with `--until field` reports `PAUSED`; check `FIELD.json`
for its field result. Keep the original inputs, saved settings and runtime.
Use a new output directory to change them. `resume` finishes native surfaces by
default, original mapping for `--flattening-method original`, the legacy atlas for
`--surface-output atlas`, or direct export for `--surface-output tifxyz`.

</details>

<details>
<summary><strong>5. Extract meshes and flattened CT from an existing field</strong></summary>

For a field produced in section 2 or 3, use its complete run directory. The
saved field is reused; completed inference and solving stages are not rerun.

```bash
RUN=runs/175a-roi                         # or runs/175a-full
winding-field mesh "$RUN"                # native mesh checkpoints
winding-field flatten "$RUN"             # native local UV checkpoints
winding-field render "$RUN"              # assembly and global CT render
```

**Native results:** `surfaces/level-*/meshes/level-0000.obj` and
`surfaces/level-*/component-*/{sheet,overview,mask}.png`. Isovalues are automatic
unless you pass `--levels 1 2 3` when creating the run. Native runs do not produce
the legacy `flat/` directory or `index.html` gallery.

For that gallery, create a new run with `--surface-output atlas`.
Its `mesh` command writes `meshes/level-*.obj`; `flatten` writes UV charts,
flat OBJ files and CT previews in `flat/`, plus the run's `index.html`.
For original whole-sheet mapping, select the method at run creation as below,
or reuse an existing mesh/field with `export-sheets`.

## Original global flattening: coarse or fine, any scroll

**`--flattening-method original` selects the November 3 whole-sheet recipe.**
It does not select coarse/fine resolution. Existing field geometry determines
that resolution; when computing a new field, use `--field-spacing-um` as above.
The default flattening path remains available with `--flattening-method default`.

Install the pinned mapping dependency in your environment:

```bash
python -m pip install '.[tifxyz]'
```

Create `flattening.json` with the **output coordinate frame**, independently of the
field spacing and selected CT pyramid level. This example exports L0 coordinates
for a scan with 8.64 µm isotropic L0 voxels:

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

Replace the ID, origin and voxel size with your scan's frame. A nonzero origin is
supported. Package-generated mesh and field origins/spacings are in physical
micrometres (`geometry.voxel_um=1`). External descriptor coordinates are in their
declared world units: multiply by `geometry.voxel_um` to get micrometres. Exported
XYZ is `(physical_xyz - output_origin_um) / output_voxel_um`. The CT transform
uses the CT metadata's origin and spacing, including anisotropic CT. Exported
valid XYZ must be nonnegative; `-1` is reserved for invalid nodes.

### New CT-to-field-to-TIFXYZ run

```bash
winding-field run --source /data/scroll.zarr --voxel-um 8.64 \
  --field-spacing-um 17.28 --levels 1 2 3 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output runs/original-fine

# Same recipe at a coarser field resolution: change only the field spacing
winding-field run --source /data/scroll.zarr --voxel-um 8.64 \
  --field-spacing-um 138.24 --levels 1 2 3 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output runs/original-coarse

winding-field resume runs/original-fine
```

Add `--roi` for a local domain, `--prediction classical` for the classical
predictor, or `--device cuda:0` for GPU computation. Qual V2/Lasagna remain the
default observation providers. Omit `--export-tifxyz` for
PNG/mask/map products only. `--until mesh` checkpoints extraction before mapping;
`resume` continues to original flattening. Set the method when creating the run:
editing an existing saved run's configuration is deliberately rejected.
Use `--export-tifxyz` with this recipe, **not** `--surface-output tifxyz`, which
selects the separate experimental direct-field algorithm.

### Reuse an existing mesh or field

An immutable `winding-mesh-v1` bundle contains `MESH.json`, `RUN.json` and
`mesh.sqlite`. Its saved levels and physical geometry are reused without changing
the source:

```bash
winding-field export-sheets --mesh-bundle /data/existing-run/meshes \
  --source /data/scroll.zarr --voxel-um 8.64 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output exports/original
```

For an existing field, pass `--field-config field.json --levels 1 2 3` instead
of `--mesh-bundle`. The descriptor supports SHA-bound real-valued ZYX NPY arrays
and the package's chunk checkpoints. Example `field.json`:

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

This descriptor expresses origins and spacings in micrometres, hence
`geometry.voxel_um=1`. For L0-unit descriptors, use the physical L0 voxel size
and origins/spacings in L0 units instead (e.g. spacing 2 at voxel_um 8.64 is
17.28 µm). Use absolute source paths and actual geometry, shape and SHA256 (`shasum -a 256
/data/field.npy`). An optional `field.mask` object with absolute `path` and
`sha256` binds a boolean NPY support mask of the same shape. Nonfinite field
values are unsupported. For chunk checkpoints use `kind: "chunks"`, `path` to
`CHECKPOINT.json`, its `sha256`, and `support_policy: "observed"` or `"stored"`;
missing chunks are always unsupported. Checkpoint metadata must agree with the
declared geometry. Arbitrary proprietary field formats and implicit composite
corrections are not inferred: supply a supported, already-corrected field.

```bash
winding-field export-sheets --field-config field.json --levels 1 2 3 \
  --source /data/scroll.zarr --voxel-um 8.64 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output exports/from-field

# Resume with exactly the same arguments and configuration, plus --resume
winding-field export-sheets --field-config field.json --levels 1 2 3 \
  --source /data/scroll.zarr --voxel-um 8.64 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output exports/from-field --resume
```

### Products, canvas and limits

End-to-end control: the installed CLI reconstructed 175A W+31 from its SHA-bound
field and support mask, with fresh extraction and mapping. The 4096² image and CT
mask matched the historical reference exactly; independent `tifffile` readback
also found identical X/Y/Z arrays. See
[`original-field-e2e-W31.json`](docs/verification/original-field-e2e-W31.json).
This validates that control, not every level or physical sheet continuity.

Each `original/level-NNNN/` contains:

- `image.png`: 4096×4096 original-CT image, normal offset zero, contrast 0–180.
- `mask.png`: **0** unmapped, **1** mapped but CT missing, **2** CT observed,
  **3** ambiguous/invalid mapping. A black pixel alone does not identify its state.
- `map.npz`: original-recipe XYZ/UV, faces, eligibility and source vertex/face IDs.
- `LEVEL.json`: input identity, source-component exclusions, distortion, coverage
  and file hashes. `original/ORIGINAL.json` aggregates completed levels.
- With `--export-tifxyz`, `tifxyz/{x,y,z}.tif` (float32), `meta.json`,
  `valid-mask.png` and `VERIFIED.json`. Copy the whole directory to a consumer.

TIFXYZ validity is independent of CT availability: mapped CT-missing geometry can
remain valid. Boundary-adjacent, jumping, collapsed/folded cells and orphan nodes
are excluded. Invalid XYZ is `-1`; `meta.json` records XYZ frame, origin,
`pitch_um`, and `scale: [1/pitch, 1/pitch]`. TIFF readback is checked during export;
this is not a certification that VC3D opened the product or that it is one
physically continuous papyrus sheet.

By default each level fits its eligible UV extent into a centered 4096² canvas.
For a common minimum pitch add `"pitch": NUMBER` to `flattening.json`, in output
frame voxels per pixel; larger domains increase it to fit. To reproduce the
November 3 175A canvas floor, use `pitch: 3.5358847757241643` with its 8.64 µm
L0 frame. Other scans should choose their own sampling pitch.

The recipe selects the largest-area connected source component, tries curvature
cuts −0.1/−0.05/−0.02 until genus zero, and uses the original two boundary pins
with global LSCM. Diagnostic eligibility requires anisotropy ≤1.25 and metric
scale 0.5–2. All excluded source area is reported. Degenerate domains and
unresolved handles fail visibly; no decimation, replacement anchors or chart
packing is substituted. This preserves the original recipe's known distortions.

`memory_gib` is an **admission estimate**, not a hard OS memory cap. Mapping loads
a complete component and sparse solver fill can exceed estimates; use a suitably
sized worker/cgroup for large fine meshes. Admission checks the level inventory
before loading arrays and checks available host/cgroup memory. Native meshes
with tens of millions of faces may be rejected. Completed levels are hash-checked
on resume; failed/pending attempt directories are retained. Changed inputs,
settings or package implementation require a new output directory.

## Generate native meshes and global CT renders

Continue from the default native field run in section 3:

```bash
winding-field mesh runs/175a-roi          # Checkpoint native OBJ meshes
winding-field flatten runs/175a-roi       # Checkpoint local UVs
winding-field render runs/175a-roi        # Assemble and render global surfaces
```

For CT → meshes in one command, use `--until mesh` on `run`.
For the complete selected pipeline, omit `--until`.

New `run`/`plan` commands default to `--surface-output native`. Each W is processed
through mesh → linked charts → shared-edge UV assembly → tiled XYZ → original CT.
The same XYZ tiles can also supply **mesh-derived TIFXYZ**; no field-root
projection or exhaustive volumetric coverage audit is required by this branch.
Disconnected source components remain separate, with native faces preserved.
Near-planar patches use a metric-checked tangent projection; others retain LSCM.
Native LSCM uses libigl when the existing `tifxyz` extra is installed, otherwise
SciPy; the backend and numerical-library versions are frozen in the run identity.
The native path processes small batches and reserves 40% of the post-mesh time
remaining for rendering/publication. A difficult run can therefore publish a
partial image below its requested coverage; the receipt makes that unmet target
explicit. Once rasterization starts, its geometry is frozen for safe resume.

Choose the coverage/time policy when creating a run:

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1504 1536 4032 4064 6048 6080 --field-spacing-um 17.28 \
  --coverage-target 0.90 --surface-time-limit 1800 \
  --export-tifxyz --output runs/native
```

`--coverage-target` is mapped source mesh **area**, greater than 0 and at most 1.
It is not certified field coverage or a guarantee of CT support. Coverage and time
are not proportional. The 1800-second default is a **cumulative active budget per W**,
not a demonstrated throughput guarantee. The field solve is outside that budget;
mesh, UV, render and publication work are inside. Native renders sample at the
minimum field spacing, recorded in physical micrometres per pixel.
The recorded whole-W−4 acceptance attempt stopped at 30 minutes during meshing
(445/2527 volume tiles, versus 196 previously). See [measured results and limits](docs/verification/fast-native-surfaces.md).

`--export-tifxyz` requires separately installed `tifffile` (available in the
existing `tifxyz` extra). Missing dependency fails before preparing the run.
Valid geometry and CT have separate masks. Unsafe reader cells remove their four
corner nodes before TIFXYZ publication; remaining cells pass the same geometry
checks. Receipts separately count unsafe cells, removed nodes, collateral cells
and orphans, and label the subset `partial_geometry`. An incompatible output
frame or empty retained surface makes TIFXYZ unavailable while valid OBJ/PNG
products remain. These products do not carry the
certified-direct branch's `VERIFIED.json` or field-coverage claims.

For an explicit **experimental global map from an existing indexed mesh**, use
[`tools/bench_global_native.py`](docs/verification/native-global-export.md).
It uses one global UV solution, complete bounded overlap qualification and the
same tiled renderer, with separately supervised source/cut/MVC/ARAP/qualification/
render/verification workers. Component, CT frame and sampling pitch are explicit;
all omitted components and faces remain in the source-area denominator. Its timing
is mesh-to-render, separate from the whole-field W−4 goal.

### Cancel and resume native work

```bash
# Ctrl+C, or from another terminal:
touch runs/native/STOP
winding-field status runs/native
winding-field resume runs/native

# If the cumulative time budget was exhausted, explicitly add more time:
winding-field resume runs/native --extra-surface-seconds 600
```

The native supervisor requests a cooperative stop, then can terminate **its own
worker** after a 2-second grace period if native code or IO is unresponsive.
Completed mesh tiles, charts and XYZ/CT tiles survive; incomplete transactions
roll back and unpublished temporary files are never presented as finished output.
Resume validates the source/configuration and committed hashes before continuing.
Elapsed time survives cancellation. Extra allowance is recorded separately and
does not turn a run exceeding its original budget into a speed success.
An already published PARTIAL level is immutable: resume verifies it rather than
silently increasing its coverage. A different target or mapping needs a new run.
PNG and its render receipt are published before optional TIFF work; incomplete
TIFF channels remain under `.tifxyz.pending/` until the full segment is verified.

`status` includes current phase, per-level progress, elapsed time and observed RSS.
Native exit codes: complete 0, published partial 2, resource/time pause 75, user
stop/stage pause 0, failure 1. Inspect JSON status; a pause is not completion.

For the original small-chart/gallery workflow, select `--surface-output atlas`.
Existing configurations without a selector and the offline `demo` keep that
legacy behavior; they are not silently migrated to the native branch.

## Experimental direct TIFXYZ export

The alternative branch is **prepare → fibres → field → tifxyz**. It constructs a
bounded coarse guide and projects onto the fine field, producing one global UV
grid per explicit level and ROI. It does not produce native OBJ meshes or an atlas
gallery. Explicit `--surface-output atlas` retains the legacy pipeline; `native`
is the default for new CLI runs.

This is **quality-first experimental integration**. Comparative performance is
`UNMEASURED` / `NO_COMPARISON` after reference SLIM OOM; there is no native-mesh
versus direct speedup or whole-W acceptance claim. Final handoff still requires the separate
geometry/integrity and ≤3 GiB acceptance gate.

Export and verification require Linux supervision and the opt-in `tifxyz` extra
(`libigl==2.6.3`, `tifffile>=2023.7`), provisioned separately in your environment.
Commands never install these dependencies. macOS reports `BLOCKED` (exit 75).

Save this as `export.json` (choose levels present in your field):

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

```bash
# Integrated branch, optionally pausing after the field
winding-field run --source /data/scroll.zarr --field-spacing-um 17.28 \
  --surface-output tifxyz --tifxyz-config export.json \
  --until field --output runs/direct
winding-field resume runs/direct
winding-field verify-tifxyz runs/direct/tifxyz

# Reuse a previous producer's FIELD_READY run with CT unavailable
winding-field tifxyz runs/old-field --output runs/direct-export --config export.json
winding-field tifxyz runs/old-field --output runs/direct-export --resume
winding-field verify-tifxyz runs/direct-export
```

Omitting `--until` finishes at `tifxyz` for this branch; `mesh`/`flatten` are
incompatible. JSON levels are authoritative; an additional `--levels` must agree
in both order and values. The normalized request and digest are frozen in
`RUN.config` at creation, even with `--until field`. Editing the original JSON
afterward has no effect. Field-dependent defaults resolve at first export:
pitch = minimum physical field spacing, tolerance = maximum spacing. Effective
settings are saved in `tifxyz/EXPORT.json` and reused on resume.

Standalone export preserves the source run byte-for-byte and never opens CT. Its
output must be outside the source run, with an existing parent directory. Resume
uses `EXPORT.json`; an optional `--config` must match the original normalized
request digest. Missing/foreign export roots are rejected rather than repaired.
This allows reuse of an older producer without weakening legacy resume's strict
implementation-SHA check.

`levels` are **W values**, not distances. `pitch_um` and `tolerance_um` are physical
micrometres. Delivered XYZ are **voxel coordinates in the declared output frame**,
not micrometres: `(xyz_um - output_origin_um) / output_voxel_um`.
`selected-source` means the SHA-bound, isotropic **selected CT level**, not
automatically L0. For 8.64 µm destination voxels and 17.28 µm UV pitch,
`meta.scale` is `[0.5, 0.5]`. An export JSON `roi` uses half-open **field-node ZYX
indices**, unlike `run --roi`, which uses selected CT voxels. Explicit destination
frames and the verification scope are described in [FORMATS.md](docs/FORMATS.md#direct-tifxyz-experimental).

Coverage policy defaults to `"coverage_policy": "strict"`. To explicitly keep
usable geometry whose source coverage cannot be fully certified, add
`"coverage_policy": "allow_partial"` to `export.json` **before creating a new
output**, then use the same export/run/verify commands above. The policy is frozen
in the request and effective settings; it cannot be changed on resume.

Only `SOURCE_UNCOVERED` or `COVERAGE_DEPTH_LIMIT` after all delivered-geometry gates
pass can publish partial output. Root ambiguity, invalid geometry, resource and
integrity failures still block publication. XYZ and `-1` masks are preserved.
The first uncertified source cell/subcell and depth are recorded in the per-level
audit, with `source_scan_complete=false`. Remaining coverage is **UNMEASURED**:
that record is not an exhaustive list of holes or a missing-area estimate.

Every publication has exactly one final, SHA-bound marker: `READY.json` for full
coverage or `PARTIAL.json` for incomplete coverage. `TIFXYZ.json`, its level records
and each `VERIFIED.json` carry `status` and `coverage_complete`. A mixed-level
export is PARTIAL. `allow_partial` can still produce READY when every level passes.
Both publications register and snapshot the integrated stage, but partial output
leaves `RUN.status=PARTIAL`, never COMPLETE; resume revalidates that quality.

Exit codes: READY / `TIFXYZ_VERIFIED` / COMPLETE / PAUSED = **0**;
PARTIAL / `TIFXYZ_PARTIAL_VERIFIED` = **2** (usable files, incomplete coverage);
resource or work-limit BLOCKED = **75**; geometry BLOCKED = **2**;
integrity/IO failure = **1**. Check JSON status to distinguish these cases.
Exit 0 with PAUSED is not acceptance. `verify-tifxyz` is read-only and reports
`TIFXYZ_VERIFIED` or `TIFXYZ_PARTIAL_VERIFIED` after fresh independent verification.
All accepted results retain `physical_acceptance=false`,
`native_face_identity=false`, `native_topology_equivalence="UNMEASURED"`.

The bounded reader check accepts either publication without upgrading its quality:

```bash
python -B tests/test_tifxyz_vc3d.py runs/direct-export/level-0000 \
  --work /existing-parent/new-reader-proof --apprun /path/to/VC3D/AppRun
```

Run this on Linux after independent verification, with a new work directory outside
the export. Its `READER_VERIFIED.json` records publication `coverage_complete` and
checks OBJ/TIFF correspondence; it proves neither source coverage nor independent
root reprojection. A PARTIAL publication stays PARTIAL.

### Optional hybrid audit acceleration

The export JSON accepts `"audit_backend": "numpy"` (default), `"cpu"`, or `"cuda"`.
All modes retain the same geometric gates and use exact Python predicates for
uncertain cases. Audit STATUS now reports `index`, `geometry`, local/nonlocal
`intersections`, and `coverage`, with counters. Native artifacts are built explicitly
using an existing compiler; export/resume/verify never compile or install anything.

```bash
# New output directories; the parent directory must already exist.
python -m winding_field.tifxyz_build --backend cpu --output ./audit-cpu
export WINDING_TIFXYZ_ACCELERATOR="$PWD/audit-cpu/accelerator.json"
# Set audit_backend to "cpu" in export.json, then use the normal export command.

# CUDA example: provide the compatible host compiler and your assigned GPU UUID.
python -m winding_field.tifxyz_build --backend cuda --host-compiler /usr/bin/g++-12 \
  --output ./audit-cuda
export WINDING_TIFXYZ_ACCELERATOR="$PWD/audit-cuda/accelerator.json"
export WINDING_TIFXYZ_CUDA_DEVICE=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# Set audit_backend to "cuda"; the UUID must be a real explicitly selected device.
```

Use only a native artifact you built/trust. Its ABI, platform, source and binary
hashes are checked and pinned in the export; paths are taken from your environment,
never from a downloaded export. Keep the same artifact available for resume/verify.
Missing or changed explicit acceleration fails rather than silently changing it.
The CUDA artifact includes compiled CPU predicates for contacts and small batches;
larger coverage/orientation batches use CUDA. Only one explicitly selected GPU is
visible to its worker, with bounded buffers and a 256 MiB free-VRAM reserve.

CPU workers keep their address-space ceiling. CUDA workers instead enforce a
heap/data ceiling plus aggregate worker/controller RSS supervision, because CUDA
reserves nonresident virtual mappings. The 3 GiB real-RAM budget and 2 GiB RAM reserve
remain. Old frozen runs require their original code/artifacts; acceleration does
not change a run already in progress.

### Stop and resume direct exports
For direct exports, Ctrl-C/SIGTERM cooperatively stop the supervised job. Integrated
jobs also honor `RUN/STOP`; `resume RUN` clears that marker as in the legacy flow.
Standalone jobs honor `OUT/STOP`: explicitly remove that marker before retrying
`tifxyz ... --resume` or `verify-tifxyz OUT`. Verification never clears markers or
rewrites accepted output. If admission stopped before any export root was created,
retry the new-export command with `--config` (there is no export to resume).

</details>

<details>
<summary><strong>6. Custom input options</strong></summary>

### Fibres / splines (`.afv`)

Use an existing AFV with the **same CT acquisition** to skip Qual inference and
tracing. Lasagna still supplies the surface observations. Set the fibre path and
ROI to match your data; this example uses the region from section 3:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'

winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --fibres /path/fibres.afv --until field --output runs/from-fibres
```

CLI-exported AFVs include their calibrated coordinates. If yours lacks that
metadata, also pass `--fibres-geometry geometry.json` with the CT source, XYZ
origin and spacing: [geometry descriptor](docs/LASAGNA.md#external-afv).

### External field (`.npy` or block checkpoint)

Extract surfaces directly from an existing field using a prepared recipe. This
route needs the field and its geometry; it does not run CT inference or solve a
new field.

Copy the template:

```bash
cp examples/surface-run.example.json surface-run.json
```

Edit `surface-run.json` before running; input paths are relative to this JSON
unless absolute:

- Set `field.path` and replace its placeholder SHA-256 with the file's hash.
- Set the `field.mask` path/hash to your support mask, or remove this optional
  entry if your field has no mask.
- Set `geometry` to the field's frame, origin, XYZ spacing, ZYX shape and calibration.
- Set `levels` to the raw field values you want to extract.
- For a frozen native block field, replace the `field` object with
  `{"kind":"chunks","path":"/path/field/CHECKPOINT.json","sha256":"…","support_policy":"observed"}`.

Then extract meshes and geometry views:

```bash
winding-field prepared surfaces --config surface-run.json --output runs/external-surfaces
```

**Results:** `runs/external-surfaces/level-*.obj` and `renders/*.png`.
These PNGs show geometry; CT-textured export needs a matching CT source.
For original global CT/TIFXYZ, use the [complete `export-sheets` recipe](#reuse-an-existing-mesh-or-field).
[Field recipe details](docs/PREPARED.md#extract-or-import-surfaces-then-render)
· [CT-textured export](docs/SURFACE_TOOLS.md#export-and-atlas)

### External Lasagna predictions — Coming soon

Importing precomputed `.lasagna.json` / OME-Zarr predictions into a new run is
not available yet. Current runs generate Lasagna predictions automatically and
reuse them when resumed. `--lasagna-model-path` selects model weights, not
prediction files.

</details>

<details>
<summary><strong>Options and output reference</strong></summary>

## Main options

| Option | Purpose / default |
| --- | --- |
| `--prediction qual-v2\|sd2\|classical` | Qual V2 by default; learned predictors use Lasagna surfaces; classical uses CT intensity |
| `--model-path /path/model.pt` | Local pinned checkpoint for the selected fibre predictor; `winding-field models --prediction qual-v2` prefetches Qual |
| `--lasagna-model-path /path/model.pt` | Local pinned Lasagna surface checkpoint, not precomputed predictions |
| `--lasagna-python /path/venv/bin/python` | Override the isolated official Python 3.14 Lasagna runtime |
| `--fibres /path/fibres.afv --fibres-geometry geometry.json` | Import calibrated fibres instead of inference/tracing; descriptor needed if AFV frame metadata is absent |
| `--device cpu\|cuda:0` | Compute device; default `cpu` |
| `--source PATH_OR_URL` | CT Zarr input; alternatively use public catalog `--scroll ID --scan ID` |
| `--local PATH` | Local mirror of source CT chunks |
| `--output PATH` | Run or export directory, depending on the command |
| `--level 0` | CT pyramid level, **not** winding W |
| `--levels -31 -30` | Winding W isovalues to extract, **not** CT pyramid levels |
| `--voxel-um 8.64` | Micrometre scale for unitless CT geometry; explicit metadata units take precedence |
| `--stride 1` | CT observation sampling step, 1–8; not an output TIFXYZ pitch |
| `--field-spacing-um 17.28` | Isotropic field-grid pitch; default: observation pitch |
| `--roi Z0 Z1 Y0 Y1 X0 X1` | Half-open voxel bounds at the selected CT pyramid level; default: whole volume |
| `--chunk 64 --field-chunk 64` | Observation tile edge in sampled voxels / solver block edge in field samples; learned inference windows stay fixed |
| `--mesh-tile 64` | Mesh extraction tile edge in field-grid cells |
| `--threshold VALUE` | Classical predictor CT intensity threshold; default: calibrated Otsu; unused by learned predictors |
| `--seed-stride 8 --fibre-weight 1` | Fibre seed spacing in observation samples / solver fibre constraint weight |
| `--threads 4` | PyTorch CPU threads; not a global worker or RAM limit |
| `--sweeps 4 --maxiter 2000 --rtol 1e-6` | Solver sweep budget, per-block iterations and tolerance |
| `--ct-mode auto\|download\|stream` | Default `auto`: download if cache/storage permits, otherwise stream |
| `--cache /disk/cache --cache-gib 2` | Shared, bounded CT payload cache |
| `--reserve-gib 1` | Free-space reserve on output and cache filesystems |
| `--offline` | CT network access disabled; `--scroll` needs a cached catalog; provision selected model weights and Lasagna runtime separately |
| `--invalid-value 0` | Declare acquisition fill explicitly; otherwise zero is valid CT |
| `--chart-faces 1024 --no-previews` | Native/atlas local chart budget, not original global flattening; `--no-previews` skips only legacy per-chart images |
| `--surface-output native\|atlas\|tifxyz` | Pipeline branch, not file format: native (default), legacy atlas, or experimental direct-field TIFXYZ |
| `--flattening-method default\|original` | Select flattening mathematics independently of coarse/fine resolution; default preserves the selected output branch |
| `--flattening-config flattening.json` | Original recipe's explicit output frame, optional minimum canvas pitch and memory admission estimate |
| `--coverage-target 1 --surface-time-limit 1800` | Native mapped-area target and cumulative active seconds per W |
| `--export-tifxyz` | Optional mesh-derived TIFXYZ alongside native or original-recipe CT renders |
| `--tifxyz-config export.json` | Frozen direct-export request; separate memory, scratch and disk reserves |
| `--until STAGE` | Stop after a stage supported by the selected branch; default: its terminal stage |
| `resume RUN --extra-surface-seconds N` | Add N seconds to the saved native surface budget |

**Input and output resolution:** `--level`/`--voxel-um` describe CT input,
`--stride` controls observation sampling, and `--field-spacing-um` controls the
field grid. The output XYZ coordinate scale and canvas pitch belong in
`--flattening-config`; they do not change the field resolution.

**Two TIFXYZ routes:** `--export-tifxyz` adds mesh-derived XYZ TIFFs to native
or original-recipe CT renders. `--surface-output tifxyz` instead selects the
experimental direct-field pipeline and uses `--tifxyz-config`. The standalone
`tifxyz FIELD_RUN` command calls that latter JSON `--config`; its `--resume`
continues the saved request. `export-sheets --resume` requires the same input
arguments and configuration as the initial export.

For existing inputs, `export-sheets` requires either `--field-config field.json`
with `--levels`, or `--mesh-bundle PATH` with its saved levels, plus `--source`,
`--output`, `--flattening-method original` and `--flattening-config`. Add
`--export-tifxyz` for XYZ TIFFs and `--resume` to continue the same export.
See the [complete examples above](#reuse-an-existing-mesh-or-field).

## Outputs

Paths are relative to the run/export directory. Package-generated field/mesh
coordinates are XYZ µm; external bundles use their declared world unit.
Original-recipe maps/TIFXYZ use the explicit output frame. Arrays and ROI bounds
are ZYX.

| Path | Contents |
| --- | --- |
| `RUN.json` | Saved configuration, stage status, timings and I/O |
| `field/CHECKPOINT.json`, `FIELD.json` | Field blocks, support and physical geometry |
| `SURFACES.json`, `surfaces/STATUS.json` | Native publication summary and live worker progress |
| `surfaces/level-*/meshes/level-0000.obj` | Complete native mesh for that W |
| `surfaces/level-*/component-*/{sheet,overview,mask}.png` | Global CT canvas, reduced view and validity mask |
| `surfaces/level-*/component-*/tifxyz/` | Optional mesh-derived XYZ TIFFs and frame metadata |
| `meshes/level-*.obj` | Atlas/original-recipe 3D meshes at each selected isovalue |
| `original/ORIGINAL.json` | Original-recipe summary; integrated runs also publish root `ORIGINAL.json` |
| `original/level-NNNN/{image,mask}.png` | Original global CT image and 0/1/2/3 coverage mask |
| `original/level-NNNN/map.npz`, `LEVEL.json` | XYZ/UV, source correspondence, coverage and product hashes |
| `original/level-NNNN/tifxyz/` | Optional float32 XYZ TIFFs, separate validity mask, frame metadata and readback receipt |
| `flat/chart-*-flat.obj`, `flat/chart-*.png` | Atlas-only flattened charts and original-CT previews |
| `flat/chart-*.npz`, `flat/atlas.sqlite` | Atlas-only UV coordinates and correspondence to source mesh faces |
| `tifxyz/TIFXYZ.json`, exactly one of `tifxyz/READY.json` / `tifxyz/PARTIAL.json` | Direct-branch quality receipt and final publication binding |
| `tifxyz/level-NNNN/{x,y,z}.tif`, `meta.json`, `VERIFIED.json` | Direct coordinates and per-level verification (standalone: relative to OUT) |
| `maps/`, `lasagna/`, `fibres/fibres.afv` | Observations, Lasagna maps and traced fibre fragments |
| `index.html` | Local interactive gallery after the atlas pipeline |

Research output: physical sheet correctness and global convergence remain
unvalidated. Whole-scroll native-resolution throughput has not been benchmarked.
See [formats and geometry](docs/FORMATS.md) and [attribution](THIRD_PARTY.md).

</details>

Full command-specific help:

```bash
winding-field run --help
winding-field plan --help
winding-field resume --help
winding-field export-sheets --help
winding-field tifxyz --help
winding-field verify-tifxyz --help
```

[Formats](docs/FORMATS.md) · [Lasagna and AFV](docs/LASAGNA.md) · [Streaming](docs/STREAMING.md) ·
[Prepared workflows](docs/PREPARED.md) · [Surface tools](docs/SURFACE_TOOLS.md) ·
[Attribution](THIRD_PARTY.md)

Research output: physical sheet correctness and global convergence remain unvalidated.
