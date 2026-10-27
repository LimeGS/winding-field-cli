# Lasagna surfaces and external AFV fibres

The CT pipeline uses official Lasagna for unsigned surface normals and target
winding density. Qual V2 supplies fibres/splines unless `--fibres` is supplied.
Lasagna runs automatically; no precomputed surface predictions are needed.

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

The verified CPU environment used Python 3.14.10 and Torch 2.14.1 on macOS arm64.
CUDA users must install a driver-compatible PyTorch build in this environment
and check `torch.cuda.is_available()` there, independently of the CLI environment.
Runtime sources and installed dependency versions are bound to each run; changing
them requires a new output directory.

The public model is `scrollprize/lasagna/model_current.pt`, revision
`de45f01147adea9f4ad0f621b1ccd885a1f03760`, SHA-256
`4cc0c40a846b296769b9f7982470559078662a1a0b0e1f74ff41ed9341c0d600`.
It is 558,287,587 bytes. `--lasagna-model-path` accepts a local copy of these same
weights. Prefetch first, then use `--offline` with cached/local CT. The official
repository lists one public checkpoint; its model card does not document accuracy
or coverage for every scroll.

## Normal CT run

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --sweeps 1 --ct-mode stream --cache-gib .25 --reserve-gib .25 \
  --output runs/175a-lasagna
```

Qual requires uint8, isotropic 8.64–9.6 µm observations. Lasagna accepts upstream's
uint8/uint16 input normalization (uint16 is quantized by integer division by 257).
No universal scientific resolution range is asserted for Lasagna. Field pitch is
independent of observation pitch. Explicit SD2 uses Lasagna surfaces with the
existing local fibre tracer; explicit classical remains the local offline baseline.

## External AFV

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --fibres runs/175a-lasagna/fibres/fibres.afv \
  --sweeps 1 --ct-mode stream --output runs/175a-import
```

An AFV replaces fibre prediction and tracing, including their downloads and dense
scratch. It still uses Lasagna, even when a fibre `--prediction` was specified.
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
`unknown`. `FIBRES.json` reports source/imported/excluded counts. Exported imported
curves have `automatically_traced=false` and their source AFV hash.

## Products, units and recovery

`lasagna-source/input.zarr` stages only the sampled ROI plus 16 observations of
context per side, clipped at source bounds. The official subprocess uses fixed
64³ tiles, overlap 32, border 16 and full-input-lattice outputs. Strict model
loading reconstructs architecture from all 306 checkpoint tensors. CPU uses FP32
to avoid emulated BF16; CUDA retains upstream BF16. The private worker disables
upstream's unrelated global training-pause service. Concurrent GPU jobs need
their own scheduling.

`lasagna-source/output/` retains official cos, grad_mag, nx and ny OME-Zarrs and
their manifest. `LASAGNA.json` binds runtime, weights, transforms and hashes.
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
