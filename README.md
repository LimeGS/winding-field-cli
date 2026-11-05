# winding-field-cli
![Illustrated workflow: original CT → Lasagna surfaces + Qual V2 fibres/splines → winding field → 3D meshes → flattened charts](docs/assets/workflow-banner.png)

Build a winding field from scroll CT, then extract meshes and CT-textured surfaces.
Defaults: Lasagna surface observations, Qual V2 fibres/splines, native surface
output. Saved stages let you stop and continue later.

Follow **1–2** for a full scroll or **1 → 3** for a region; **4** resumes work,
**5** exports surfaces, and **6** reuses existing inputs.

<details>
<summary><strong>1. Install</strong></summary>

Use **Python 3.11 or 3.12** on Linux/macOS. Install the numerical core at the
revision recorded in this checkout, then the CLI, in the same environment:

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

Default CT runs also need official Lasagna under **Python 3.14**, in a separate
environment. Install it once without activating it:

```bash
python -m pip install uv
uv python install 3.14
uv venv --python 3.14 ~/.cache/winding-field/lasagna-venv
uv pip install --python ~/.cache/winding-field/lasagna-venv/bin/python \
  'vesuvius-lasagna @ https://github.com/ScrollPrize/villa/archive/b29f310957c7b29e7d332f9876e0e487819cfaa3.tar.gz#subdirectory=lasagna'
```

The CLI finds this runtime automatically. Model weights download on first use.
CUDA requires compatible PyTorch in both environments; see
[runtime setup](docs/LASAGNA.md#one-time-runtime-setup).

Optional offline installation check (no model downloads):

```bash
winding-field demo --output runs/demo
```

Open `runs/demo/index.html`. Use a new output directory for each demo.

</details>

<details>
<summary><strong>2. Run a winding field for a full scroll</strong></summary>

Select the original CT, inspect storage/grid estimates, then run. Replace this
public scan URL with your own URL or local 3D Zarr v2 path as needed.
**No `--roi` means the entire selected CT level.**

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
- `--field-spacing-um 300` chooses an example coarse solve grid; it does not
  reduce observation preparation or tracing scratch. Check `plan` before running.
- `--ct-mode stream` fetches CT chunks on demand; products still need disk/RAM.
- Use `--device cpu` in both commands for CPU. `plan` does not create the run;
  `run --output` must name a new directory.

**Results:** `field/CHECKPOINT.json` and field blocks, root `FIELD.json` with
geometry/status, `fibres/fibres.afv`, and `lasagna-source/output/` predictions.

Qual requires **uint8, isotropic 8.64–9.6 µm observations**. Set `--voxel-um`
to the actual CT scale and choose `--level`/`--stride` accordingly. Field pitch
is independent: any positive pitch fitting at least two nodes per axis is allowed;
omitting it uses the observation grid, including its anisotropy. Unitless CT
requires `--voxel-um`. Catalog selection uses `--scroll ID --scan ID`; ambiguous
acquisitions need an exact scan or `--source`. `--local /data/mirror.zarr` adds a
matching local chunk mirror to a remote source.

</details>

<details>
<summary><strong>3. Run a field for a specific section of a scroll</strong></summary>

ROI bounds are **selected CT-level voxels**, ZYX, with exclusive upper bounds.
This example requests a 48³ region and a 17 µm field on CPU:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 \
  --field-spacing-um 17 --device cpu --ct-mode stream \
  --until field --output runs/175a-roi
```

Replace the bounds for your region. Use `plan` instead of `run` to inspect grids
first. Inference reads surrounding context; outputs remain inside the ROI.
The field needs at least two nodes per axis.

**Results:** `runs/175a-roi/field/`. To stop earlier, use `--until prepare` for
predictions or `--until fibres` for traced/exported AFV. Continue with
`winding-field field runs/175a-roi`.

</details>

<details>
<summary><strong>4. Stop, inspect and resume a run</strong></summary>

Point `RUN` at the complete run, not its `field/` subdirectory. Press Ctrl-C or
create `STOP` from another terminal, then wait for the process to exit:

```bash
RUN=runs/175a-full
winding-field status "$RUN"
touch "$RUN/STOP"
```

Continue through field construction or through the selected terminal stage:

```bash
winding-field field "$RUN"
# Or:
winding-field resume "$RUN"
```

Both clear `STOP`, validate saved work and reuse completed stages. A run stopped
at `--until field` reports `PAUSED`; inspect `FIELD.json` for the field result.
Keep the original inputs, runtime, settings and run location. Changed inputs or
implementation require a new output directory.

`resume` finishes native surfaces by default, original mapping for
`--flattening-method original`, atlas for `--surface-output atlas`, or direct
export for `--surface-output tifxyz`. To extend an exhausted native time budget:

```bash
winding-field resume "$RUN" --extra-surface-seconds 600
```

Already published partial native levels are immutable; extra time does not remap
them. Inspect status and coverage rather than treating a pause as completion.

If a `--bounded-preparation` field exhausted its sweeps without passing global
verification, increase its **absolute total** budget (for example, from 4 to 8):

```bash
winding-field resume "$RUN" --field-sweeps 8
```

The original configuration stays immutable. At an exhausted sweep boundary, the
CLI atomically records the budget extension in the checkpoint, preserving all
field chunks and solver progress. Repeating the same target does not add sweeps;
lower targets and changes to published fields are rejected. Numerical verification
still gates publication. The embedded history records previous-state hashes and
budget changes, not historical copies of every field generation.

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
masks in `level-*/component-*/`. The saved run configuration selects the branch.

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
mkdir -p exports
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

For aligned neighbouring W levels, use [`export-uvw`](docs/UVW.md) with an
existing original map as reference. It exports CT and TIFXYZ on one shared UV
grid, with validity masks and resumable processing. Outputs are diagnostic;
inspect coverage and deformation before using side cuts through the stack.

</details>

<details>
<summary><strong>6. Custom input options</strong></summary>

### Fibres / splines (`.afv`)

An AFV from the **same CT acquisition** skips fibre inference/tracing. Lasagna
still supplies surfaces. Set the ROI to match your data:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --fibres /path/fibres.afv --until field --output runs/from-fibres
```

CLI AFVs carry calibrated coordinates. Otherwise supply `--fibres-geometry
geometry.json` with CT identity, XYZ origin and spacing:
[descriptor](docs/LASAGNA.md#external-afv). V2 names the model; its AFV container
uses schema v1.

### External field (`.npy` or block checkpoint)

For geometry-only extraction, copy and edit the prepared template:

```bash
cp examples/surface-run.example.json surface-run.json
# Replace paths, SHA-256 hashes, geometry, calibration and levels before running.
winding-field prepared surfaces --config surface-run.json --output runs/external-surfaces
```

Paths resolve relative to the JSON. Remove `field.mask` if there is no support
mask. The template's `voxel_um: 8` makes spacing 2 equal to 16 µm; use
`voxel_um: 1` for micrometre origins/spacings. For chunk checkpoints and raw
field-level calibration see [prepared inputs](docs/PREPARED.md#extract-or-import-surfaces-then-render).

**Results:** `level-*.obj` and `renders/*.png` under the output directory. These
are geometry views; original CT/TIFXYZ needs the matching CT and the
[export-sheets recipe](docs/FLATTENING.md).

### External Lasagna predictions

Importing precomputed `.lasagna.json` / OME-Zarr predictions into a new CT run is
not supported. Resume reuses predictions generated by that run.
`--lasagna-model-path` selects weights, not prediction files.

</details>

## Options and results at a glance

| Option | Meaning |
| --- | --- |
| `--level` / `--levels` | CT pyramid level / winding W isovalues |
| `--stride` / `--field-spacing-um` | Observation sampling / independent solve-grid pitch |
| `--surface-output native\|atlas\|tifxyz` | Native default / legacy charts and gallery / experimental direct-field export |
| `--flattening-method original --flattening-config FILE` | Global LSCM with explicit output frame; independent of field resolution |
| `--export-tifxyz` | Add mesh-derived XYZ TIFFs to native or original renders |
| `--tifxyz-config FILE` | Separate request for the Linux-only experimental direct route |
| `--coverage-target 1 --surface-time-limit 1800` | Native mapped mesh-area target / cumulative active seconds per W |
| `--cache-gib 2 --reserve-gib 1` | CT payload cache cap / free disk reserve, not RAM |
| `--threads 4` | PyTorch threads, not a global worker limit |
| `--offline` | Disable CT networking; provision models/runtime separately |
| `--invalid-value 0` | Explicitly exclude acquisition fill; otherwise zero CT is valid |

| Output | Where to look |
| --- | --- |
| Saved configuration, status and timings | `RUN.json` |
| Field geometry and solver status | `FIELD.json`, `field/CHECKPOINT.json` |
| Native results and progress | `SURFACES.json`, `surfaces/STATUS.json` |
| Native meshes and CT | `surfaces/level-*/meshes/`, `surfaces/level-*/component-*/` |
| Original global CT/masks/maps | `original/level-NNNN/`, `original/ORIGINAL.json` |
| Atlas-only charts and gallery | `flat/`, `index.html` |
| Direct-field publication | `tifxyz/TIFXYZ.json` and `READY.json` or `PARTIAL.json` |

Native output does not create the atlas gallery. XYZ units, masks, coverage,
resume contracts and resource limits are documented in [FORMATS](docs/FORMATS.md).
For complete flags use command-specific help:

```bash
winding-field run --help
winding-field plan --help
winding-field resume --help
winding-field export-sheets --help
winding-field tifxyz --help
winding-field verify-tifxyz --help
```

[Flattening](docs/FLATTENING.md) · [Lasagna and AFV](docs/LASAGNA.md) ·
[Streaming](docs/STREAMING.md) · [Prepared workflows](docs/PREPARED.md) ·
[Surface tools](docs/SURFACE_TOOLS.md) · [Attribution](THIRD_PARTY.md)

Research output: physical sheet correctness and global convergence remain
unvalidated; whole-scroll native-resolution throughput is not established.
