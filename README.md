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

### Continue a CLI run

For the default native run from section 2 or 3, continue through rendering:

```bash
winding-field render runs/175a-roi
```

Use `runs/175a-full` for the full-scroll example. This automatically runs any
preceding stages still needed and reuses completed work. Results are under the
run's `surfaces/`: OBJ meshes in `level-*/meshes/` and CT images, overviews and
masks in `level-*/component-*/`.

### Export a chosen W level as TIFXYZ

Install the export dependencies once in your active environment:

```bash
python -m pip install '.[tifxyz]'
```

Prepare two [configuration files](docs/FLATTENING.md#configuration-files):

- `field.json` describes the saved field data and its geometry.
- `flattening.json` defines the output coordinates.

Replace the paths and CT scale (`--voxel-um`) with those for your scan:

```bash
winding-field export-sheets --field-config field.json --levels -31 \
  --source /data/scroll.zarr --voxel-um 8.64 \
  --flattening-method original --flattening-config flattening.json \
  --export-tifxyz --output exports/W-31
```

Choose more W values with `--levels -31 -30`. To resume, repeat the same command
with the same configuration and add `--resume`.

Main outputs:

```text
exports/W-31/original/level-0000/
├── image.png
├── mask.png
└── tifxyz/{x,y,z}.tif
```

`level-0000` means the first requested W (here −31), not W=0.
See [flattening setup and advanced reference](docs/FLATTENING.md) for details.

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
For original global CT/TIFXYZ, use the [complete `export-sheets` recipe](docs/FLATTENING.md).
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
See the [complete examples](docs/FLATTENING.md).

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
