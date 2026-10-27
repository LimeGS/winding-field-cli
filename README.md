# winding-field-cli

Original CT → winding field → 3D meshes → flattened charts. Resumable CLI for
local or remote 3D Zarr v2 volumes, with automatic observation and fibre preparation.

## Quick start

Python 3.11+, Linux or macOS. Install from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
winding-field doctor
```

Run the complete pipeline on synthetic CT, offline:

```bash
winding-field demo --output runs/demo
open runs/demo/index.html                 # macOS; open in a browser on Linux
```

## Calculate a field

Small real-CT example: native **8.64 µm observations**, **17 µm field**, CPU.
The default SD2 predictor expects uint8 CT; its weights (~329 MiB) are downloaded
and verified on first use.

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'

winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1504 1536 4032 4064 6048 6080 \
  --field-spacing-um 17 --until field --sweeps 1 \
  --ct-mode stream --cache-gib .25 --reserve-gib .25 \
  --output runs/175a-17um
```

The field is in `runs/175a-17um/field/`; `FIELD.json` records its geometry and
solver status. `--sweeps 1` is a short example budget; the default is 4.

### Choose the resolution

**`--field-spacing-um P` accepts any positive pitch in micrometres** that fits the
ROI: `8.65`, `17`, `130`, `300`, `250.5`, etc. It sets equal spacing in X/Y/Z.
Omit it to use the observation grid, including its anisotropy.

- Keep `--level 0 --stride 1` for native-resolution CT observations.
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
Resume checks input/product integrity and clears `STOP`. A deliberate `--until`
stop reports `PAUSED`; check the corresponding stage receipt for completion.
Use a new output directory to change settings, input data or package implementation.

## Main options

| Option | Purpose / default |
| --- | --- |
| `--prediction sd2\|classical` | Pretrained SD2 by default; `classical` uses CT threshold/structure tensor |
| `--model-path /path/sd2.pt` | Use a pinned, hash-verified checkpoint; `winding-field models` prefetches it |
| `--device cpu\|cuda:0` | Compute device; default `cpu` |
| `--level 0 --stride 1` | CT observation sampling; stride accepts 1–8 |
| `--chunk 64 --field-chunk 64` | Storage tile / solve block sizes; SD2 inference windows stay fixed |
| `--sweeps 4 --maxiter 2000 --rtol 1e-6` | Solver sweep budget, per-block iterations and tolerance |
| `--ct-mode auto\|download\|stream` | Default `auto`: download if cache/storage permits, otherwise stream |
| `--cache /disk/cache --cache-gib 2` | Shared, bounded CT payload cache |
| `--reserve-gib 1` | Free-space reserve on output and cache filesystems |
| `--offline` | Cached/local data only; `--scroll` needs a cached catalog |
| `--invalid-value 0` | Declare acquisition fill explicitly; otherwise zero is valid CT |
| `--chart-faces 1024 --no-previews` | Chart size; optionally skip CT preview images |

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
| `maps/`, `lasagna/`, `fibres/fibres.afv` | Observations, Lasagna maps and traced fibre fragments |
| `index.html` | Local interactive gallery after the full pipeline |

Research output: physical sheet correctness and global convergence remain
unvalidated. Whole-scroll native-resolution throughput has not been benchmarked.
See [formats and geometry](docs/FORMATS.md), [verification](docs/verification/audit-fixes.md)
and [attribution](THIRD_PARTY.md) for details.

## Development

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=4 python -m pytest -q
python -m build
```
