# Lasagna surfaces and custom inputs

Use this guide to choose where surface observations and fibres come from:

| Inputs | Options | Model inference |
| --- | --- | --- |
| CT only | Default run | Lasagna surfaces + Qual fibres |
| Existing surface predictions | `--lasagna-input` | Only fibre inference |
| Existing fibres | `--fibres` | Only Lasagna surface inference |
| Both imported | `--lasagna-input` and `--fibres` | None |

Start with the [CLI installation](../README.md). Imports use a fresh output
directory; they do not replace inputs inside an existing run.

The default CT pipeline uses official Lasagna for unsigned surface normals and
target winding density, with Qual V2 supplying fibres/splines. `--fibres` replaces
fibre inference/tracing with an external AFV. `--lasagna-input` independently
replaces surface inference with existing predictions. Explicit `--prediction sd2`
also uses Lasagna; `--prediction classical` without either imported input uses the
offline local baseline.

## One-time runtime setup

Skip this setup when using `--lasagna-input` or the Docker images, which already
include Lasagna. The CLI/core use Python 3.11/3.12.
Pinned Villa Lasagna inference requires Python 3.14,
so install it separately with [uv](https://docs.astral.sh/uv/):

```bash
python -m pip install --upgrade pip uv
uv python install 3.14
uv venv --python 3.14 ~/.cache/winding-field/lasagna-venv
uv pip install --python ~/.cache/winding-field/lasagna-venv/bin/python \
  --constraint docker/cli.in \
  'vesuvius-lasagna @ https://github.com/ScrollPrize/villa/archive/b29f310957c7b29e7d332f9876e0e487819cfaa3.tar.gz#subdirectory=lasagna'
winding-field doctor
winding-field models --prediction lasagna
winding-field models --prediction qual-v2
```

`doctor` reports the CLI runtime. A CT run additionally checks the selected
Lasagna interpreter, pinned archive provenance, source inventory and checkpoint
before creating its output directory. `plan` estimates storage without installing
or running models. Use `--lasagna-python /another/venv/bin/python` to override the
default interpreter. Keep the venv executable path, including its symlink.

Run the installation commands from the checkout. Its `uv.toml` selects a
CPU/CUDA PyTorch build automatically; `docker/cli.in` keeps the supported version.
Inference uses CPU unless you pass `--device cuda:0` to the CT run. Both the CLI
and Lasagna environments must support the selected device.
Runtime sources and installed dependency versions are bound to each run; changing
them requires a new output directory.

The public model is `scrollprize/lasagna/model_current.pt`, revision
`de45f01147adea9f4ad0f621b1ccd885a1f03760`, SHA-256
`4cc0c40a846b296769b9f7982470559078662a1a0b0e1f74ff41ed9341c0d600`.
It is 558,287,587 bytes. `--lasagna-model-path` accepts a local copy of these same
weights. Prefetch first, then use `--offline` with cached/local CT. The CLI accepts
only this pinned checkpoint; model and runtime identities are recorded in
[provenance/lasagna.json](../provenance/lasagna.json).

## Normal CT run

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --sweeps 1 --ct-mode stream --until field \
  --output runs/175a-lasagna
```

Results: `lasagna-source/output/` holds official predictions,
`fibres/fibres.afv` holds calibrated splines, and `FIELD.json` describes the field.
This example stops after field construction. `winding-field render
runs/175a-lasagna` continues the saved native surface pipeline.

Qual requires uint8, isotropic 8.64–9.6 µm observations. Lasagna accepts upstream's
uint8/uint16 input normalization (uint16 is quantized by integer division by 257).
No universal scientific resolution range is asserted for Lasagna. Field pitch is
independent of observation pitch. Explicit SD2 uses Lasagna surfaces with the
existing local fibre tracer; explicit classical remains the local offline baseline.

## Reuse existing predictions

Keep the complete CLI run, its CT source and original runtime. Completed
preparation is reused without rerunning model inference:

```bash
winding-field fibres /path/to/run   # trace the saved Qual predictions
winding-field field /path/to/run    # build the field; also finishes fibres if needed
winding-field resume /path/to/run   # finish the saved output branch
```

## External Lasagna predictions

This guide covers explicit/embedded physical-frame inputs and trilinear interpolation.
For pooled base-grid inputs, remote channels and offline replay, see [LASAGNA.md](LASAGNA.md).

Import a local `.lasagna.json` into a new CT run:

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --lasagna-input /path/surface.lasagna.json \
  --lasagna-geometry /path/lasagna-geometry.json \
  --until field --output runs/imported-lasagna
```

Use `plan` instead of `run` to check geometry and storage before creating a run.
The manifest and channels are read-only. Import requires:

- Lasagna manifest **version 2**, positive integer `base_shape_zyx`, and one scalar
  **3D ZYX Zarr v2 array** per required channel: `cos`, `grad_mag`, `nx`, `ny`.
  Optional `presence` adds a support mask. Legacy packed CZYX groups, Zarr v3
  and remote prediction URLs are unsupported.
- Each group's `zarr` points directly to its array directory (for example
  `surface_cos.ome.zarr/0`), relative to the manifest or absolute. `channels` is
  the single channel name in a list. `scaledown` is a nonnegative exponent:
  its grid step is **`2**scaledown` base voxels**. Channel shape must be
  `ceil(base_shape_zyx / 2**scaledown)`. The paired normal components `nx`/`ny`
  must share their grid; cos, magnitude and presence may use other resolutions.
- `cos`, `nx`, `ny` and optional `presence` are finite values in 0..255.
  `grad_mag` is finite and nonnegative; float32 values above 255 are supported.
  All consumed local chunks must exist, including all-zero chunks. Missing files
  fail explicitly rather than becoming implicit fill.
- If `crops` is declared, the observation ROI must fit inside one XYZ/size crop.
  A referenced `provenance` JSON is retained and a reported unfinished producer
  is rejected.

**Geometry:** `--lasagna-geometry` describes the manifest's **base grid**, not
the chosen CT level, individual channel grid or requested ROI. For a full-volume
base grid with zero origin, save this as `lasagna-geometry.json`:

```json
{
  "source": "s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr",
  "origin_xyz_um": [0, 0, 0],
  "spacing_xyz_um": [8.64, 8.64, 8.64]
}
```

The mapping is `world = origin_xyz_um + base_XYZ * spacing_xyz_um`; origins are
voxel centres. For ROI-local predictions, supply that crop's physical origin.
For a local CT, use its absolute acquisition path in `source`. CLI exports at
`RUN/lasagna/surfaces.lasagna.json` already contain `original_ct`,
`world_origin_xyz_um` and `world_spacing_xyz_um`: omit `--lasagna-geometry` for
these. Conflicting embedded/explicit frames and mismatched CT sources fail.

Scalar channels are trilinearly sampled onto the CT observation grid. Normals are
decoded, sign-aligned and interpolated as vectors, then normalized; opposite
directions describe the same unsigned surface. Every contributing corner must be
covered, with positive `grad_mag` and, when present,
positive `presence`. Import does not bridge support gaps or extrapolate beyond
stored channel centres. Coarser channels can therefore leave unsupported edge
planes. CT acquisition fill declared with `--invalid-value` also excludes samples.

Density uses `grad_mag / grad_mag_encode_scale * grad_mag_factor` (defaults
1000 and 1), then converts base-index normals/density to physical units using the
base XYZ pitch. `source_to_base` is producer metadata; `grad_mag_factor` already
accounts for producer input-to-base density conversion. Neither channel scaledown
nor observation/field pitch is applied again to density.

`--lasagna-input` skips the Lasagna interpreter and checkpoint entirely, and
conflicts with `--lasagna-python` / `--lasagna-model-path`. Fibre selection remains
independent: default Qual, explicit SD2/classical, or `--fibres` with an external
AFV. Qual's CT dtype/pitch requirements still apply when it supplies fibres.
Combining both imported inputs skips all model inference and downloads; local
inputs can run with `--offline`. A geometry option alone is rejected.

Omit `--bounded-preparation` when importing predictions: that backend recomputes
native Lasagna/Qual windows and cannot consume external surfaces. The combination
is rejected before source I/O. The standard importer still reads bounded tiles.

## External AFV

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --fibres runs/175a-lasagna/fibres/fibres.afv \
  --sweeps 1 --ct-mode stream --output runs/175a-import
```

This example requires the AFV from the earlier run. For an AFV from another
producer, supply its calibrated frame as described below; do not guess the crop
origin from its array shape.

An AFV replaces fibre prediction and tracing, including their downloads and dense
scratch. Surface observations come from Lasagna inference or `--lasagna-input`,
even when a fibre `--prediction` was specified.
Do not combine it with `--model-path`; surface weights use `--lasagna-model-path`.

AFV v1 SQLite input must be complete, closed/checkpointed and contain contiguous
binary64 XYZ blocks. Input remains read-only. A complete CLI `frame` identifies
its selected source level and physical transform. A Qual ROI-local header with
only voxel size and shape is insufficient to locate a crop. Supply a descriptor:

```json
{
  "source": "s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr",
  "origin_xyz_um": [52254.72, 34836.48, 12856.32],
  "spacing_xyz_um": [8.64, 8.64, 8.64]
}
```

Pass it with `--fibres-geometry geometry.json`. The mapping is exactly
`world = origin_xyz_um + AFV_XYZ * spacing_xyz_um`. The example origin is for the
ROI above, assuming zero source origin. Use `[0,0,0]` / `[1,1,1]` for coordinates
already in world micrometres. Conflicting complete embedded metadata is rejected.

Import clips intersecting edges to the observation ROI and traverses voxel cells
to split at unsupported corners. It preserves original vertices wherever no
clipping/splitting is needed, without smoothing or retracing. Source IDs, original
annotations and available family labels survive export; unlabeled fibres remain
`unknown`. An annotation's `hv_classification.manual_tag` supplies V/H when the
family column is missing, empty or `unknown`; an existing family takes precedence.
`FIBRES.json` reports source/imported/excluded counts. Exported imported
curves have `automatically_traced=false` and their source AFV hash.

## Products, units and recovery

For native inference, `lasagna-source/input.zarr` stages only the sampled ROI plus 16 observations of
context per side, clipped at source bounds. The official subprocess uses fixed
64³ tiles, overlap 32 and border 16. Output reductions default to Chase's
**cos 2×, grad_mag 4×, nx 4×, ny 4× per axis**. Override them with:

```bash
--lasagna-cos-downsample 2 --lasagna-grad-mag-downsample 4 \
--lasagna-nx-downsample 4 --lasagna-ny-downsample 4
```

Each accepts **1, 2, 4 or 8**; `nx` and `ny` must match because they form one
vector. These reduce predictions, **not model input or CT texture**. Grids keep
index zero and use ceil-shaped axes. The ordinary route uses official scalar
and paired-normal pyramids and retains only selected levels. Bounded preparation
Gaussian-reduces globally anchored raw predictions before encoding, then
calibrates on the finer density/normal lattice. Interpolation decodes and aligns
unsigned normals before averaging; samples beyond a channel's last grid point
remain unsupported, rather than extrapolated. Fill-value CT samples remain invalid.

`plan` reports `lasagna_output_grids` with shapes, pitches and estimated array
bytes. Native staging, inference workspace and dense observation maps are **not**
eliminated by these flags; bounded output remains a quota, not a full-volume
storage estimate. Existing inputs retain their declared grids: generation flags
conflict with `--lasagna-input`. Factors are bound into generated identities;
use a **new run** to change them. Old runs require their original package.

The official
runtime loads the pinned checkpoint strictly. CPU uses FP32
to avoid emulated BF16; pre-Ampere CUDA also uses FP32, while Ampere/newer uses
upstream BF16. The private worker disables
upstream's unrelated global training-pause service. Concurrent GPU jobs need
their own scheduling.

`lasagna-source/output/` retains official cos, grad_mag, nx and ny OME-Zarrs and
their manifest. `LASAGNA.json` binds runtime, weights, transforms and hashes.
The private worker persists all-zero chunks too, so an absent file remains an
incomplete/corrupt product rather than being silently interpreted as zero.
Density is `grad_mag / 1000` converted from input-index gradient to world units:
divide normal components by input XYZ pitch, multiply density by that vector's
length, then normalize the vector. Density is cycles/µm; output grid pitch is not
applied a second time. Unsigned normals still need the existing global orientation
and sheet calibration step.

Surface `support` and `surface_phase` (official cos / 255) are independent of
fibre probabilities. Missing fibre prediction does not erase valid surfaces.
Declared acquisition fill is excluded at its samples; zero is otherwise valid.
Automatic mesh phase uses surface cos. Lasagna predictions and nominal sheet
calibration are research observations, not established physical truth.

For imported predictions, `lasagna-source/maps/` instead retains checksummed 32³
tiles of physical surfaces and CT. `IDENTITY.json` binds the manifest, descriptor,
metadata and observation grid; per-channel SQLite ledgers bind every consumed
source chunk. `WORK.json` seals the source/tile inventories on STOP and handled
errors; resume rejects missing or altered records before reopening writable stores.
`LASAGNA.json` records `method: external-lasagna`, tile, ledger and seal hashes.
The planner reports `estimated_lasagna_import_bytes` and zero native
Lasagna staging/workspace. Imported surfaces plus AFV need no preparation halo.

STOP/Ctrl-C reaps children and seals retained work. Resume rechecks source chunks,
AFV/descriptor bytes, runtime identity and products, including completed stages.
Imported runs recheck prediction manifests, geometry, metadata, consumed chunks
and normalized tiles too; keep those original local files in place. A paused
import reuses committed tiles and continues without replacing valid work.
An unsealed hard-killed parent requires a new output directory. Old implementation-
bound receipts require their original environment and are never rewritten.
