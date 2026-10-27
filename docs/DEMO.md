# Reproducible demonstration

Use a fresh output name for each new run. `resume` continues the exact original
package, source and settings; it does not update an old run to a new algorithm.

## Install and verify

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'git+https://github.com/LimeGS/winding-field-cli.git@v0.1.0'
winding-field doctor
```

Linux CUDA users should install the appropriate PyTorch build for their driver
before installing this package. CPU is enough for the demonstrations below.

## Offline synthetic CT

```bash
winding-field demo --output runs/demo
winding-field status runs/demo
winding-field resume runs/demo
```

Open `runs/demo/index.html` in a browser. Drag the mesh preview to rotate it;
download individual 3D/flat OBJs and correspondence NPZs from the chart cards.
The generator writes curved layers with known 8-voxel pitch at 4 µm per voxel.
This route uses the explicitly labelled classical CT baseline, without a model
or network. The CT is persisted at `runs/demo-ct.zarr` beside the run.

## Original PHerc0175A CT + pretrained SD2

```bash
winding-field models --path models/sd2.pt
winding-field run \
  --source 's3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr' \
  --voxel-um 8.64 --roi 1503 1535 4038 4070 6062 6094 \
  --output runs/175a-demo --prediction sd2 --model-path models/sd2.pt \
  --device cpu --cache .cache/175a --cache-gib .25 --ct-mode stream \
  --sweeps 1 --reserve-gib .25
winding-field status runs/175a-demo
```

This is a **32 × 32 × 32 native-level voxel ROI**, not a whole-scroll demonstration.
Preparation reads a 40-voxel halo; source chunks may extend further. Original CT
is read from public S3. The downloaded checkpoint is approximately 329 MiB and
is SHA-256 verified before loading. Cache/model downloads need network access on
the first run. A prior matching local copy can be supplied with `--local`.

The initial CPU reference run produced 26 fibre fragments, 269 evaluated
constraint rows, 2,382 mesh faces across 12 components and 19 linked charts,
covering all mesh faces. Preparation took 22.56 s; fibres 0.05 s, field 0.25 s,
mesh 0.19 s and flattening 1.16 s. **Those are warm-cache figures**, excluding model
download: 6,291,456 newly downloaded CT bytes, 3 downloads and 20 cache hits.
Final release verification is retained under [verification](verification/).

## Show and interpret the result

1. `index.html`: bounded rotatable preview and flattened original-CT images.
2. `meshes/level-*.obj`: all extracted components for each nominal isovalue.
3. `flat/chart-*.obj`, `*-flat.obj`: matching 3D and flat chart geometry.
4. `flat/chart-*.npz`: UVs and original mesh vertex/face IDs for correspondence.
5. `FIELD.json`: calibration, numerical budget and explicit acceptance status.
6. `FLAT.json`: exact mesh-face coverage; 100% is coverage of the extracted mesh,
   not evidence that every physical sheet in the CT was recovered.

`COMPLETE` means that all requested stages finished. It does not mean the field
has globally converged or that physical sheet identity is correct. A one-sweep
demo intentionally exhausts its sweep budget. Current tracing creates fragments;
flattening creates an atlas with cuts. SD2 was trained on Paris4, so its transfer
to this acquisition is also unvalidated.

## Stop, resume and inspect

```bash
winding-field run --source /data/original.zarr --output runs/partial \
  --prediction classical --until prepare
winding-field fibres runs/partial
winding-field resume runs/partial
```

Ctrl-C, SIGTERM or a `STOP` file in the run requests a durable stop at a bounded
operation boundary. `resume` removes the stop marker and validates source and
products before continuing. Keep the original source available. Runs contain
absolute source/output paths and are not portable resumable checkpoints after
moving directories; exported geometry and the gallery can be copied independently.
