# Lasagna surfaces and external AFV fibres

The default CT pipeline uses official Lasagna for unsigned surface normals and
target winding density, with Qual V2 supplying fibres/splines. `--fibres` replaces
fibre inference/tracing with an external AFV. Explicit `--prediction sd2` also uses
Lasagna; `--prediction classical` without an AFV uses the offline local baseline.

## One-time runtime setup

The CLI/core retain Python 3.11/3.12. Pinned Villa Lasagna requires Python 3.14,
so install it separately with [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.14
uv venv --python 3.14 ~/.cache/winding-field/lasagna-venv
uv pip install --python ~/.cache/winding-field/lasagna-venv/bin/python \
  'vesuvius-lasagna @ https://github.com/ScrollPrize/villa/archive/b29f310957c7b29e7d332f9876e0e487819cfaa3.tar.gz#subdirectory=lasagna'
~/.cache/winding-field/lasagna-venv/bin/python -I -c \
  'from lasagna.preprocess_cos_omezarr import cli_main; raise SystemExit(cli_main())' predict3d --help
winding-field doctor
winding-field models --prediction lasagna
winding-field models --prediction qual-v2
```

`doctor` reports the CLI runtime. A CT run additionally checks the selected
Lasagna interpreter, pinned archive provenance, source inventory and checkpoint
before creating its output directory. `plan` estimates storage without installing
or running models. Use `--lasagna-python /another/venv/bin/python` to override the
default interpreter. Keep the venv executable path, including its symlink.

CUDA users must install a driver-compatible PyTorch build in this environment
and check `torch.cuda.is_available()` there, independently of the CLI environment.
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
This example pauses after field construction. `winding-field render
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

### Import local, custom or community Lasagna predictions

`--lasagna-input /path/to/scroll.lasagna.json` or an HTTP(S)/S3 manifest URL
replaces **surface inference**. No Lasagna interpreter, checkpoint or model
download is required. Relative channel and provenance paths resolve against the
manifest. The default Qual fibre provider still needs its own checkpoint; use
`--prediction classical` for CT-derived fibres or `--fibres` for existing AFV.
`--lasagna-input` conflicts with `--lasagna-python` / `--lasagna-model-path`.

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --lasagna-input /data/custom/scroll.lasagna.json \
  --fibres /data/custom/fibres.afv --fibres-geometry /data/custom/frame.json \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --cache /ssd/prediction-and-ct-cache --cache-gib 2 --reserve-gib 8 \
  --ct-mode stream --until prepare --output runs/imported-lasagna
```

Accepted format is v2, with positive `source_to_base`,
`grad_mag_encode_scale`, `grad_mag_factor`, and `base_shape_zyx`. Required groups
are `grad_mag`, `nx`, `ny`, `cos`; each names a direct 3-D Zarr-v2 **uint8** array,
zero fill, `channels: ["<group name>"]`, and integer `scaledown` in 0..16.
Channel shapes must be `ceil(base_shape_zyx / 2**scaledown)`. Mixed resolutions
are supported. Optional `mask` or `validity` groups use the same contract;
nonzero values are valid. Compressed channels use the shared Volume decoder.

**CT binding is explicit.** Supplying `--lasagna-input` asserts that its base
grid belongs to the selected CT's level-zero scan. The importer validates the
base shape, selected-level scale/shape relationship, zero relative origin and
channel OME axes/transforms. Equal shapes alone do **not** prove scan identity.
Receipts label this `user-selected-ct-base-grid` and retain both CT identities.
For machine-verifiable binding, custom manifests may include `ct_source` equal
to the CLI's complete level-zero Volume identity; mismatches fail closed.
Use the CT pyramid root with `--level` for a coarser CT level, not an unlabelled
direct downsampled array. Missing OME channel metadata uses the manifest's
explicit base-grid/scaledown contract; present metadata must agree. Pixel or
unitless scales are base-index units; micrometre metadata must match CT geometry.

Each target sample selects its pooled channel cell using
`floor(base_index / 2**scaledown)`. No normal-vector interpolation across sheets
is introduced. Physical density is
`grad_mag / grad_mag_encode_scale * grad_mag_factor * norm(n / input_pitch_xyz)`,
where `input_pitch_xyz = CT_base_pitch_xyz * source_to_base`.
Channel downsample pitch is **never** used as the inference-input pitch.
Density scaling uses float64 intermediates and checks the unmasked result before
float32 publication, so extreme finite encoding scales cannot silently erase support.
Normals remain unsigned and pass through the existing orientation/calibration.

`crops` is the authoritative union of half-open base-grid XYZ-width-height-depth
boxes, even when `inference.json` describes only the last write. Omitted crops
declare the full base extent; an empty list declares no coverage. Outside this
coverage, zero density, masked CT, explicit validity-mask zeros, and missing
remote prediction chunks produce unsupported observations. Coverage is
conservative at crop seams: every sampled pooled cell must fit within a single
declared crop, so a cell spanning separate crops can be excluded. No prediction
chunks are fetched for wholly uncovered regions. An absent **local** chunk in
claimed coverage is an error, not implicit zero data.

Encoded prediction chunks and CT share `--cache` / `--cache-gib` and obey
`--reserve-gib`; cache eviction is allowed. Reads are on demand, bounded to
128³ requested samples, without a native-volume staging copy. Each metadata
document is capped at 4 MiB. `--offline` requires retained metadata and every
previously consumed chunk needed for validation; an uncached object is an error,
not an authoritative 404. `lasagna-import/` records manifest/OME/provenance
snapshots and per-channel encoded-byte ledgers. Resume checks those identities,
consumed bytes and their inventory before reusing prepared products.
Missing import identity/ledgers alongside partial observation or coarse-calibration
work fail closed; they are never recreated around existing products. Automatically
discovered CT mirrors (`WINDING_CT_ROOTS`) are also used during retained validation.

After an online import, continue that same run without network access:

```bash
winding-field resume runs/import --offline
# Or validate/finish just preparation:
winding-field prepare runs/import --offline
```

This is a per-invocation transport override: saved run configuration, source identity
and product bindings are unchanged. A new offline output cannot reuse another run's
metadata snapshots merely by sharing its payload cache.

The sidecar revision, dirty flag, checkpoint and other provenance are retained
verbatim as **external** evidence; importing a community model does not certify
the pinned official CLI runtime. `PREPARED.json` retains the compatible
`surface_method: "lasagna"` for downstream phase/export consumers and adds
`surface_origin: "external"`.

`--bounded-preparation --lasagna-input ...` supplies the same on-demand reader
to native surface maps, retaining coarse observations instead of a native
prediction volume. Existing bounded-mode restrictions still apply: native
Qual V2, stride 1, explicit field pitch and no external AFV in that mode. A
full-scroll job still needs a credible runtime/storage admission estimate;
bounded reading alone is not a <=24-hour runtime claim.

#### Bounded community check (2026-11-05)

Verified the [PHerc0483A community manifest](https://huggingface.co/buckets/SharpMeow/lasagna-fields/resolve/PHerc0483A/representations/predictions/lasagna/20250521140913-lasagna-de45f011/PHerc0483A.lasagna.json)
against the user-bound `20250521140913-8.640um-1.2m-116keV-masked.zarr`
CT base grid on gpu-1, CPU 7 only, after the final provenance/encoding fixes. The sampled ROI was 64³ at ZYX
`[8192,4096,4096]`; cos used level 1 and normals/density level 2.

| Measurement | Result |
| --- | --- |
| Supported samples | 262,144 / 262,144 |
| Density range | 0.01018518–0.01886574 cycles/µm |
| Cold read including metadata | 19.197 s |
| Offline replay including validation | 0.166 s; all five output-array hashes identical |
| Encoded prediction + CT cache | 2,630,947 bytes; 64 MiB cap |
| Offline payload downloads | 0 |
| Peak process RSS | 488,532 KiB |
| Free disk after check | 182,320,955,392 bytes; 8 GiB reserve |

The retained sidecar reports dirty revision
`db3dbea26185a9e09ab997bfa21a2a00477f0d56`; this is external provenance,
not exact pinned-runtime equivalence. Evidence is retained at
`gpu-1:/mnt/fast2/paris4-vcreader2440-20261105/lasagna-review-01/community-representable/EVIDENCE.json`,
including source-module and harness hashes. Earlier checks remain retained.
This establishes bounded reader/calibration/replay behavior, not full-volume
runtime or physical sheet correctness. No full-volume PHerc0483A job was launched.

## External AFV

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --fibres runs/175a-lasagna/fibres/fibres.afv \
  --sweeps 1 --ct-mode stream --output runs/175a-import
```

An AFV replaces fibre prediction and tracing, including their downloads and dense
scratch. It still uses Lasagna surfaces (imported when `--lasagna-input` is set),
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

`lasagna-source/input.zarr` stages only the sampled ROI plus 16 observations of
context per side, clipped at source bounds. The official subprocess uses fixed
64³ tiles, overlap 32, border 16 and full-input-lattice outputs. The official
runtime loads the pinned checkpoint strictly. CPU uses FP32
to avoid emulated BF16; CUDA retains upstream BF16. The private worker disables
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

STOP/Ctrl-C reaps children and seals retained work. Resume rechecks source chunks,
AFV/descriptor bytes, runtime identity and products, including completed stages.
An unsealed hard-killed parent requires a new output directory. Old implementation-
bound receipts require their original environment and are never rewritten.
