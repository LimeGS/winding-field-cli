# winding-field-cli
![Illustrated workflow: original CT → Lasagna surfaces + Qual V2 fibres/splines → winding field → 3D meshes → flattened charts](docs/assets/workflow-banner.png)

Turn local or remote 3D Zarr v2 CT into winding fields, 3D meshes and CT-textured
flat charts. Start from CT or continue from existing results.

<details>
<summary><strong>Install + offline demo</strong></summary>

Python **3.11 or 3.12**, Linux/macOS. Install the CLI and its pinned numerical library:

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

For real CT, also install the [isolated Lasagna runtime](docs/LASAGNA.md#one-time-runtime-setup)
once. Model weights download automatically on first use.

Try the complete pipeline offline, without model downloads:

```bash
winding-field demo --output runs/demo
```

Open `runs/demo/index.html` in a browser.

</details>

<details>
<summary><strong>Create Lasagna predictions, fibres or a field from CT</strong></summary>

Choose a final `STAGE`; earlier stages run automatically. This example uses a
small PHerc0175A ROI on CPU, with 8.64 µm observations and a 17 µm field.

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
STAGE=field  # prepare | fibres | field | mesh | flatten

winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 \
  --field-spacing-um 17 --until "$STAGE" \
  --ct-mode stream --cache-gib .25 --reserve-gib .25 \
  --output "runs/175a-$STAGE"
```

| `STAGE` | New results inside the run |
| --- | --- |
| `prepare` | Lasagna surfaces + Qual fibre predictions: `lasagna-source/output/`, `lasagna/`, `maps/` |
| `fibres` | Traced fibres/splines: `fibres/fibres.afv` |
| `field` | Winding field: `field/CHECKPOINT.json`, geometry/status in `FIELD.json` |
| `mesh` | 3D meshes: `meshes/level-*.obj` |
| `flatten` | UV charts, flat OBJ, CT previews: `flat/`, gallery: `index.html` |

Use a **new output directory** for each new run. Omit `--until` for the full pipeline.
`prepare` runs both models; fibre tracing starts at `fibres`.

Qual V2 requires **uint8, isotropic 8.64–9.6 µm observations**. Select `--level` /
`--stride` to match your CT; `--field-spacing-um` only changes the solve grid.

</details>

<details>
<summary><strong>I already have a field → meshes and charts</strong></summary>

**Field inside a CLI run:** keep the complete run directory and continue:

```bash
winding-field mesh runs/175a-field
winding-field flatten runs/175a-field
```

Open `runs/175a-field/index.html`. Isovalues are automatic unless you set
`--levels 1 2 3` when creating the run.

**Standalone NPY field or frozen block checkpoint:** fill the
[surface recipe](examples/surface-run.example.json) with its path, SHA-256,
geometry and levels, then extract meshes + geometry PNGs:

```bash
winding-field prepared surfaces --config surface-run.json --output runs/surfaces
```

[Field formats and recipes](docs/PREPARED.md#extract-or-import-surfaces-then-render)
· [CT-textured export and visualization](docs/SURFACE_TOOLS.md)

</details>

<details>
<summary><strong>I already have Lasagna predictions → fibres and field</strong></summary>

**From a CLI run:** reuse the complete run with its saved predictions:

```bash
winding-field fibres runs/175a-prepare     # trace fibres from saved predictions
winding-field field runs/175a-prepare      # solve using retained observations/fibres
```

**External `.lasagna.json` / OME-Zarr predictions:** direct import is not supported
yet. If you already converted them into oriented, calibrated `target.npy`,
`weights.npy` and `observed.npy`, fill the
[prepared field recipe](examples/field-run.example.json):

```bash
winding-field prepared build --config field-run.json --output runs/from-predictions
```

[Reuse limits and preparation requirements](docs/LASAGNA.md#reuse-existing-predictions)

</details>

<details>
<summary><strong>I already have fibres → field</strong></summary>

**Inside a CLI run:** continue directly:

```bash
winding-field field runs/175a-fibres
```

**Existing AFV → new run:** use the same CT acquisition. With `$CT` from the
new-run example:

```bash
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --fibres /path/fibres.afv --until field --output runs/from-fibres
```

This skips Qual inference and tracing; Lasagna still runs. CLI-exported AFVs
include geometry. Other AFVs may need `--fibres-geometry geometry.json`:
[coordinate mapping](docs/LASAGNA.md#external-afv).

</details>

<details>
<summary><strong>Stop, inspect and resume a run</strong></summary>

```bash
winding-field status runs/175a-field
touch runs/175a-field/STOP                 # or Ctrl-C
winding-field field runs/175a-field        # continue only through field
winding-field resume runs/175a-field       # continue through flattening
```

Stage commands reuse saved settings and completed work. Resume validates inputs
and products and clears `STOP`. A deliberate `--until` stop reports `PAUSED`.
Keep the original inputs and runtime; changed settings/code require a new run.

</details>

<details>
<summary><strong>Essential options</strong></summary>

| Option | Use |
| --- | --- |
| `--source /data/scroll.zarr` | Local CT; also accepts HTTP/S3 URLs |
| `--roi Z0 Z1 Y0 Y1 X0 X1` | Selected-level voxels, exclusive upper bounds; omit for the whole volume |
| `--voxel-um 8.64` | Physical voxel size when input metadata has no units |
| `--field-spacing-um 17` | Positive XYZ field pitch in µm; ROI must fit ≥2 nodes/axis |
| `--device cuda:0` | GPU inference/solve; default `cpu`, spline tracing stays on CPU |
| `--levels 1 2 3` | Mesh isovalues, chosen when creating the run |
| `--invalid-value 0` | Exclude acquisition fill; zero is otherwise valid CT |

Use `winding-field plan` with your `run` arguments to inspect grids/storage first.
Use `winding-field run --help` for all flags.

</details>

<details>
<summary><strong>Development</strong></summary>

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=4 python -m pytest -q
python -m build
```

</details>

[Formats](docs/FORMATS.md) · [Streaming](docs/STREAMING.md) ·
[Prepared workflows](docs/PREPARED.md) · [Surface tools](docs/SURFACE_TOOLS.md) ·
[Verification](docs/verification/lasagna-afv-inputs.md) · [Attribution](THIRD_PARTY.md)

Research output: physical sheet correctness and global convergence remain unvalidated.
