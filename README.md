# winding-field-cli

An installable, resumable **original CT → observations → fibres → winding field →
3D meshes → linked flat charts** pipeline. It accepts an entire CT volume or an
explicit ROI. Local CT is preferred; public S3/HTTP originals can be downloaded
into a bounded cache or streamed on demand.

**Research preview:** a completed run means the requested processing stages ran.
It does not establish correct physical sheet identity or whole-scroll continuity.
Numerical receipts distinguish conditional block convergence from global convergence.
Flattening produces a linked atlas with cuts, not a claim of one seamless sheet.

## Install

Python 3.11+; Linux or macOS. For large runs use Linux, CUDA and adequate storage.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
winding-field doctor
```

From the published repository:

```bash
python -m pip install 'git+https://github.com/LimeGS/winding-field-cli.git@v0.1.0'
```

The numerical solver and extractor are reused from `winding-field-gh`; their
mathematics are unchanged. Source hashes and namespace-only changes are recorded
in `provenance/`. Five pinned Villa modules implement the pretrained SD2 network.

## Offline demonstration

```bash
winding-field demo --output runs/demo
open runs/demo/index.html        # macOS; open this file in a browser on Linux
winding-field resume runs/demo
```

This creates synthetic CT, runs the complete **classical CT baseline**, and writes
3D OBJ, flattened OBJ, correspondence NPZ, original-CT PNG previews and a local
rotatable gallery. It downloads no model. Synthetic CT is saved beside the run as
`demo-ct.zarr`, and the source/cache are separate from persistent products.

## Original CT

```bash
winding-field models                    # download and verify pinned SD2 (about 329 MiB)

winding-field run \
  --source 's3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr' \
  --voxel-um 8.64 \
  --roi 1503 1535 4038 4070 6062 6094 \
  --output runs/175a-roi --prediction sd2 --device cpu \
  --sweeps 1 --ct-mode stream --cache-gib .25 --reserve-gib .25
```

`--roi` is **Z0 Z1 Y0 Y1 X0 X1**, half-open, in voxels of the explicitly selected
source level. Output samples stay inside it. Preparation reads a recorded 40-grid-
voxel halo around it; source chunks may contain additional compressed data.
Arrays are ZYX; all field/mesh coordinates are XYZ **micrometres**, with voxel-centre
origins. Unitless CT metadata requires `--voxel-um` or catalog spacing.

For an entire scroll, omit `--roi`:

```bash
winding-field plan --scroll PHerc0175A --output runs/175a --cache-gib 32
winding-field run --scroll PHerc0175A --output runs/175a \
  --device cuda:0 --chunk 64 --field-chunk 128 --cache-gib 32
```

Catalog ambiguity requires `--scan` or an exact `--source`; the program never
selects a different scan or pyramid level to make a job fit. Whole-scroll execution
is supported by tiled processing, but has **not** been validated end-to-end at
native resolution. A large full-volume job is not a next-morning laptop demo.

### Important controls

- `--prediction sd2` (default): pinned pretrained fibre presence/directions.
  The model was trained on Paris4 data; transfer to other acquisitions is unvalidated.
- `--prediction classical`: explicitly labelled Otsu/structure-tensor baseline.
  One threshold is estimated from bounded ROI samples and persisted for all tiles;
  `--threshold VALUE` supplies an explicit calibrated threshold instead.
- `--local /matching/original.zarr`: use existing matching CT chunks first; absent
  local chunks are fetched from the original remote source. A local-only missing
  chunk raises instead of silently becoming air. Online mirrors are content-verified;
  S3 single-part ETags can verify local bytes without downloading the payload again.
- `WINDING_CT_ROOTS=/data:/workspace/ct`: search matching source basenames locally.
- `--ct-mode auto|download|stream`: auto prefetches the required input domain if it
  fits both the cache cap and available space after estimated products/reserve.
- `--cache-gib 2`, `--cache /disk/cache`, `--reserve-gib 1`: disposable CT cache,
  shared compressed-payload cap and output free-space reserve. Metadata/index and
  transient decode RAM are separate from the payload cap.
- `--level 0`: explicit source pyramid level. `--stride 1`: explicit output-grid
  sampling; larger values reduce computation but can lose sheet/fibre detail.
- `--until prepare|fibres|field|mesh|flatten`: stop after an intermediate stage.
- `--sweeps 4`, `--maxiter 2000`: numerical budgets. Exhausting sweeps is not proof
  of global convergence. Settings are pinned on resume.
- `--levels 1 2 3`: explicit isovalues; otherwise CT-presence phase selects them.
- `--chart-faces 1024`: bounded chart size. Difficult charts are split, preserving
  source face/vertex correspondence and complete mesh-face coverage.

## Resume and products

```bash
winding-field status runs/175a-roi
winding-field resume runs/175a-roi
winding-field mesh runs/175a-roi       # continue through a named stage
```

Ctrl-C or `touch RUN_DIRECTORY/STOP` requests a durable stop at stage/block
boundaries. Resume checks source metadata, consumed chunk hashes and completed
products; changed numerical inputs require a new run. The first interrupt waits
for the current bounded operation to finish.

| Path | Contents |
| --- | --- |
| `RUN.json` | Source, effective settings, domain, stages, timings and I/O counts |
| `maps/`, `PREPARED.json` | Physical density, normals, tangents, presence, support |
| `lasagna/` | OME-Zarr channels and separate fibre/surface Lasagna v2 manifests |
| `fibres/fibres.sqlite`, `fibres/fibres.afv` | Traced fragments, physical constraints and AFV export |
| `field/CHECKPOINT.json`, `FIELD.json` | Persistent block field, support and numerical receipt |
| `meshes/` | Globally indexed, all-component mesh database and per-level OBJ |
| `flat/` | Linked atlas, 3D/flat OBJ, UV/source-ID NPZ and original-CT previews |
| `index.html` | Self-contained local gallery (bounded visual preview) |

Lasagna manifests use an **ROI-local output grid**, with explicit world transform
extras and NGFF physical metadata. AFV coordinates are in the selected original
source grid; inspect its `frame` metadata before combining it with an L0-only tool.
Fibres terminate at tile boundaries and unsupported gaps; these are fragments,
not globally joined anatomical fibres.

## Current limits

- Portable reader supports numeric, 3D Zarr **v2** with numcodecs. Unsupported
  v3/sharded layouts fail explicitly. `vc_reader` was researched, but its native
  build failed on this Mac's libc++; it is not a required dependency or a claimed
  tested backend. See [streaming research](docs/STREAMING.md).
- Automatic orientation uses a planar/radial heuristic and density from the
  existing fractional-crossing recipe. Folds, topology and sheet-count calibration
  can be wrong; inspect CT previews and receipts before interpreting the geometry.
- Fibre tracing and flattening use bounded fragments/charts. One continuous,
  physically verified full-scroll unroll is not demonstrated.
- Stage outputs and solver state persist; budgets do not guarantee a hard process
  RSS ceiling. Choose smaller chunks for RAM/VRAM limits.

## Development

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=4 python -m pytest -q
python -m build
```

See [formats](docs/FORMATS.md), [demo recipe](docs/DEMO.md), and [attribution](THIRD_PARTY.md).
