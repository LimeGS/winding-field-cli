# winding-field-cli

Original CT → Lasagna surfaces + Qual V2 fibres/splines → winding field → 3D meshes → flattened
charts. Resumable CLI for local or remote 3D Zarr v2 volumes, with automatic
observation and fibre preparation.

## Quick start

Python 3.11 or 3.12, Linux or macOS. Clone this branch and install the original
library at its recorded revision (`winding-field-experiments==1.3.0.dev1`).
Use `python3.12` instead of `python3.11` if preferred:

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

The [original numerical library](https://github.com/LimeGS/winding-field)
is a real dependency: this package contains no solver copy. Only the CLI installs
the `winding-field` executable. The revision in
[provenance/core.json](provenance/core.json) is also used by CI. A separate local
library checkout is not required. To reproduce a fixed source snapshot, check
out the CLI commit from the verification receipt or PR before installing.

Run the complete pipeline on synthetic CT, offline:

```bash
winding-field demo --output runs/demo
open runs/demo/index.html                 # macOS; open in a browser on Linux
```

## Calculate a field

Small real-CT example: native **8.64 µm observations**, **17 µm field**, CPU.
Install the isolated official Lasagna runtime once using
[these instructions](docs/LASAGNA.md). Normal CT runs then execute Lasagna
automatically for surface normals and target density. Its pinned public weights
(558 MB) are downloaded and verified on first use.

The default **Qual AFV fibre V2** fibre predictor expects uint8 CT at isotropic
**8.64–9.6 µm** observation spacing. Its weights (~540 MiB) are downloaded and
SHA-256 verified on first use. The official Qual spline generator is installed
automatically with the package; its H/V splines constrain the field.

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'

winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 \
  --field-spacing-um 17 --until field --sweeps 1 \
  --ct-mode stream --cache-gib .25 --reserve-gib .25 \
  --output runs/175a-17um
```

The field is in `runs/175a-17um/field/`; `FIELD.json` records its geometry and
solver status. `--sweeps 1` is a short example budget; the default is 4.

### Import existing fibres

Add `--fibres /path/fibres.afv` to the same command to replace Qual inference and
tracing. Lasagna still runs. CLI-exported AFVs carry their calibrated frame;
files with ambiguous geometry need `--fibres-geometry geometry.json`:

```json
{"source":"/absolute/path/scroll.zarr","origin_xyz_um":[100,200,300],"spacing_xyz_um":[9,9,9]}
```

This declares `world_XYZ = origin + AFV_XYZ * spacing`. The source must match the
CT acquisition. See [AFV geometry and examples](docs/LASAGNA.md#external-afv).

### Choose the resolution

**`--field-spacing-um P` accepts any positive pitch in micrometres** that fits the
ROI: `8.64`, `17.28`, `130`, `309`, `250.5`, etc. It sets equal spacing in X/Y/Z.
Omit it to use the observation grid, including its anisotropy.

- Keep `--level 0 --stride 1` for native-resolution CT observations.
- For finer CT, explicitly select a level/stride in Qual's range (e.g. 4.32 µm
  CT with `--stride 2` gives 8.64 µm observations).
- Increasing field spacing reduces the solve grid; fine preparation still costs the same.
- `--voxel-um` declares input units. `--field-chunk` sets block size in nodes.
- The ROI must contain at least two field nodes per axis. Nodes stay inside its
  observed extent; `plan` shows both grids before computation.

### Local CT or a whole scroll

```bash
# Local CT with physical units in its metadata
winding-field run --source /data/scroll.zarr \
  --field-spacing-um 130 --until field --output runs/local-130um

# Whole scroll: inspect geometry/storage estimates, then compute on CUDA
winding-field plan --scroll PHerc0175A --field-spacing-um 300 --output runs/175a-300um
winding-field run --scroll PHerc0175A --field-spacing-um 300 \
  --device cuda:0 --until field --output runs/175a-300um
```

Omitting `--roi` selects the entire source level. ROI syntax is
`--roi Z0 Z1 Y0 Y1 X0 X1`, with exclusive upper bounds in selected-level voxels.
Unitless CT needs `--voxel-um`. Ambiguous scroll acquisitions need `--scan` or
an exact `--source`. Use `--local /data/mirror.zarr` alongside a remote source to
reuse matching local chunks and fetch missing ones.

## Generate meshes and flattened charts

Continue from the field run above:

```bash
winding-field mesh runs/175a-17um          # 3D OBJ meshes
winding-field flatten runs/175a-17um       # UV charts, flat OBJ and CT previews
open runs/175a-17um/index.html             # gallery after flattening
```

For **CT → meshes in one command**, use `--until mesh` on `run`.
For the **complete pipeline**, omit `--until`.

Isovalues are selected automatically. To choose them, supply `--levels 1 2 3`
when creating the run. Stage commands reuse its saved configuration.

## Stop and resume

```bash
winding-field status runs/175a-17um
touch runs/175a-17um/STOP                  # or press Ctrl-C
winding-field field runs/175a-17um         # resume only through field construction
winding-field resume runs/175a-17um        # continue through the complete pipeline
```

Stopping finishes the current bounded operation and retains committed work.
Lasagna and Qual children are stopped and reaped too; resume validates retained work.
Resume checks input/product integrity and clears `STOP`. A deliberate `--until`
stop reports `PAUSED`; check the corresponding stage receipt for completion.
Use a new output directory to change settings, input data or package implementation.

## Main options

| Option | Purpose / default |
| --- | --- |
| `--prediction qual-v2\|sd2\|classical` | Qual V2 + global splines by default; explicit SD2/classical use the legacy local tracer |
| `--model-path /path/qual-v2.pt` | Use the selected model's pinned checkpoint; `winding-field models` prefetches Qual V2 |
| `--fibres input.afv` | Import external fibres; skip fibre models and tracing |
| `--fibres-geometry geometry.json` | Explicit transform when the AFV lacks a complete calibrated frame |
| `--lasagna-python /venv/bin/python` | Override the isolated official runtime |
| `--lasagna-model-path model.pt` | Local pinned surface weights; `models --prediction lasagna` prefetches them |
| `--device cpu\|cuda:0` | Inference/solver device; default `cpu`. Spline tracing uses CPU |
| `--level 0 --stride 1` | CT observation sampling; stride accepts 1–8 |
| `--chunk 64 --field-chunk 64` | Storage tile / solve block sizes; inference windows stay fixed |
| `--sweeps 4 --maxiter 2000 --rtol 1e-6` | Solver sweep budget, per-block iterations and tolerance |
| `--ct-mode auto\|download\|stream` | Default `auto`: download if cache/storage permits, otherwise stream |
| `--cache /disk/cache --cache-gib 2` | Shared, bounded CT payload cache |
| `--reserve-gib 1` | Free-space reserve on output and cache filesystems |
| `--offline` | Cached/local data only; `--scroll` needs a cached catalog |
| `--invalid-value 0` | Declare acquisition fill explicitly; otherwise zero is valid CT |
| `--chart-faces 1024 --no-previews` | Chart size; optionally skip CT preview images |

## Prepared inputs and geometry rendering

Already have a field recipe or a mesh? The workflows moved here from the
original library:

```bash
winding-field prepared build --config field-run.json --output runs/prepared
winding-field prepared resume --config field-run.json --output runs/prepared --prefetch
winding-field prepared surfaces --config surfaces.json --output runs/surfaces
winding-field prepared render --mesh runs/surfaces --output runs/views
```

See [prepared workflows](docs/PREPARED.md) for recipes, supervised workers,
Python imports and runnable examples. Implementation-bound runs made before
the split require the original packages or a new output directory.

Surface extraction, Flatboi/PCA export, atlases, geometry checks and CT/patch
pictures are also owned by this package. See [surface and visualization tools](docs/SURFACE_TOOLS.md)
for `surface-export`, `surface-probe`, `surface-atlas`, `surface-geometry-check`,
`ct-visuals`, `patch-visuals` and `run-watch`. Numerical benchmarks stay in the core;
their optional CT pictures are generated separately with `winding-field ct-visuals`.

## Outputs

Paths are relative to the run directory. Field/mesh coordinates are XYZ µm;
arrays and ROI bounds are ZYX.

| Path | Contents |
| --- | --- |
| `RUN.json` | Saved configuration, stage status, timings and I/O |
| `field/CHECKPOINT.json`, `FIELD.json` | Field blocks, support and physical geometry |
| `meshes/level-*.obj` | 3D meshes at each selected isovalue |
| `flat/chart-*-flat.obj`, `flat/chart-*.png` | Flattened charts and original-CT previews |
| `flat/chart-*.npz`, `flat/atlas.sqlite` | UV coordinates and correspondence to source mesh faces |
| `maps/`, `lasagna/`, `fibres/fibres.afv` | Observations, Lasagna maps and family-labelled Qual splines (AFV schema v1) |
| `LASAGNA.json`, `lasagna-source/` | Raw official surface products, staged CT, transforms and integrity records |
| `fibres/upstream/generator.log` | Official tracer progress; resumable scratch is alongside it |
| `index.html` | Local interactive gallery after the full pipeline |

For fibres alone, use `--until fibres` on a new run. Qual traces across storage
tiles and splits curves at unsupported gaps. Its V2 name refers to the model,
not a new AFV container version.

`plan` includes the dense spline arrays: 8 bytes per observation voxel plus
approximately 18 bytes per half-resolution tracing voxel. Seed sorting also uses
RAM proportional to seed count (~64 bytes/seed); saved curves need extra disk.
External AFV runs omit that dense tracing scratch. Planning also includes Lasagna
staging, raw outputs and conservative upstream workspace estimates. Explicit
`--prediction sd2` uses Lasagna surfaces with SD2 fibres; `classical` keeps the
local offline surface/fibre baseline. An external AFV always selects Lasagna.

Research output: physical sheet correctness and global convergence remain
unvalidated. Whole-scroll native-resolution throughput has not been benchmarked.
See [formats and geometry](docs/FORMATS.md),
[Lasagna/AFV verification](docs/verification/lasagna-afv-inputs.md),
[core/CLI verification](docs/verification/core-cli-separation.md),
[Qual verification](docs/verification/qual-v2-splines.md) and
[attribution](THIRD_PARTY.md) for details.

## Development

With the quick-start environment active:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=4 python -m pytest -q
python -m build
```
