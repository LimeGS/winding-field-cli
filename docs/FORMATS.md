# Geometry and persisted products

## Coordinates

- Arrays and ROI arguments: ZYX. `--roi Z0 Z1 Y0 Y1 X0 X1` is half-open in voxels
  of the explicitly selected source level, never implicitly level 0.
- Field grids, fibre constraints and 3D mesh vertices: XYZ micrometres.
- Origins are voxel-centre coordinates. At source index `i_xyz`,
  `world_xyz = origin_xyz + i_xyz * spacing_xyz`.
- Observation origin is the source origin plus reversed ROI-low times source
  spacing; observation spacing is source spacing times explicit `--stride`.
- Field origin is the observation origin. `--field-spacing-um P` sets exact
  isotropic field spacing. Per-axis node count is
  `floor((observation_count-1)*observation_spacing/P)+1`. The last field centre
  never exceeds the last observation centre. Default: the full observation grid.
- OME-NGFF scale/translation and axis units are interpreted. Unitless metadata
  requires explicit `--voxel-um` or catalog spacing. No silent resolution change.
- Preparation has a 40-observation-grid-voxel input halo, clipped to the source volume.
  Qual V2 rounds input core coverage out to source-anchored 48³ tiles (SD2: 32³);
  `RUN.json.plan.input_lo_zyx/input_hi_zyx` records this extra input-only context.
  Persistent maps and extracted meshes remain within the output grid/ROI.

## Run and integrity

`RUN.json` (`winding-field-run-v1`) stores the original acquisition/metadata hash,
effective settings, implementation SHA-256, stage receipt hashes, timings and I/O
counts. `source.sqlite` records consumed source-chunk hashes. `products.sqlite`
records exported-product hashes. A run is single-writer locked.
`exports.sqlite` retains input identity and hashes of individually completed
Lasagna chunks and AFV publication. A partial AFV database commits each fibre
record and retains its last committed ID. Resume verifies the retained inventory,
coordinates, metadata and spatial bounds against the pinned source before skipping
records; only the complete database is renamed.

Statuses are `STARTING`, `RUNNING`, `PAUSED`, `FAILED` and `COMPLETE`.
`physical_acceptance: false` is independent of successful stage completion.
Source metadata, consumed content, package implementation and completed products
must agree on resume. The CT cache is disposable, not the source of truth for
checkpoints. Retain the original package and data for reproducibility.
The implementation identity also includes bundled model JSON, installed numerical
dependency versions, and the source hashes of the Qual tracer and network architecture.

## Observations and fibres

`maps/` is a transactional tile inventory plus checksummed NPZ payloads.
`PREPARED.json` describes its physical grid, source, ROI, model provenance and halo.
Maps contain CT-derived sheet density (inverse micrometres), unsigned surface
normals (XYZ), separate fibre tangents (XYZ), presence and support.

SD2 inference owns fixed 32³ cores on the selected-source sampling lattice. Its
phase is `roi_low % stride`, context is 40 samples per side, volume-edge padding
is clamped, and normalization is the pinned model's window z-score. Storage
`--chunk` and ROI edges do not change the canonical window for overlapping samples
on the same sampling lattice. Up to eight cropped canonical tiles are cached in RAM.
This defines a repeatable tiled estimator, not equality with whole-volume inference.

The default `qual-v2` predictor requires isotropic uint8 observations at 8.64–9.6 µm.
It uses fixed 128³ windows (48³ owned core plus 40-voxel halos), with the same
source anchoring and bounded cache as SD2. Normalization is upstream's sample
z-score with standard-deviation floor 10. Segmentation softmax is ordered
background/V/H/intersection; presence is `1 - P(background)`. `probabilities`
stores raw float32 V/H channels as `(Z,Y,X,2)`; intersection is not assigned to
either family. The direction head supplies normalized XYZ fibre axes.
On CPU backends without oneDNN, convolutions are evaluated in depth slabs with
exact padding/halos to limit im2col scratch to 256 MiB (or one output plane).
Instance normalization still sees each complete feature window; weights and
the network topology are unchanged. This is not smaller-window inference.

`--invalid-value` is an explicit acquisition-fill policy. By default every finite
CT measurement, including zero, is valid. Declared fill zeros predicted presence;
support requires the full 40-voxel observation-filter footprint to be valid.
Unsupported density, normals and tangents are zero. The policy is identity-bound.

For the classical route, `THRESHOLD.json` persists a single Otsu threshold sampled
from at most 27 fixed 16³-output-grid patches within the ROI. Tile size cannot
change that threshold. Rare intensity populations may be absent from this bounded
sample; `--threshold` allows an acquisition-calibrated value. SD2 supplies its own
presence predictions. Strided preparation assembles sampled grids from bounded
native-chunk reads rather than allocating their dense native envelope.

`lasagna/` exports six uint8 OME-Zarr v2 arrays and two Lasagna v2 manifests:
`fibres.lasagna.json` and `surfaces.lasagna.json`. Direction XY is encoded around
128 with scale 127 after folding the sign into the positive-Z hemisphere. The
manifests address an **ROI-local output grid** and include world origin, spacing
and original-source metadata; align this frame before combining with other data.

`fibres/fibres.sqlite` stores curves in physical XYZ units.
`FIBRES.json` includes their count and database SHA. `fibres/fibres.afv` is an AFV
v1 SQLite export with application ID 1447249475. Its `frame` metadata specifies
selected-source voxel coordinates, source level and physical transform. It is
not automatically native L0. Blocks contain at most 256 segments (257 XYZ
binary64 points), sharing one endpoint. H/V labels, voxel lengths and physical
millimetre lengths are retained for Qual curves. The published V2 model uses the
same AFV schema v1 container.

Qual tracing consumes `fibres/probabilities.npy`, a disk-backed `(2,Z,Y,X)` float32
array in V/H order with unsupported observations zeroed. Its tile ledger and
completion hash prevent changed input on resume. The pinned official generator
runs as an interruptible CPU subprocess, with work/logs under `fibres/upstream/`.
Model inference and field solving still use the selected `--device`.

The generator pools 2×2×2 observations before quantizing, excludes unpaired final
planes on odd axes, and traces globally across storage tiles. Its intermediate
AFV already applies `local_observation_XYZ = 2 * tracing_XYZ + 0.5`; the importer
adds the prepared origin and spacing exactly once. Curves split wherever any
contributing fine-grid support corner is invalid, testing edges at intervals no
larger than 0.25 observation voxels. SD2/classical streamlines retain their original
tile-boundary stopping behavior.

Qual's dense scratch needs approximately 18 bytes per tracing voxel, besides
the native float32 input (8 bytes per observation voxel). `plan` accounts for
these arrays. Seed sorting additionally requires RAM proportional to seed count
(upstream estimate: 64 bytes/seed), and curve batches/exports need additional
disk. Disk reserve is monitored while the child runs; stopped preparation may
replay its current upstream phase, and tracing replays only uncommitted batches.

## Field

`field/CHECKPOINT.json` and block NPYs use the reused numerical core's durable
block format. `FIELD.json` records physical grid, orientation/calibration,
constraint rows and conditional local-solver convergence. Sweeps are a budget;
`global_converged: false` means no independent global convergence is established.
The nominal unit sheet step and automatic planar/radial-Z calibration are
scientifically unvalidated.

The grid adapter orients fine unsigned normals before trilinearly sampling the
physical density-normal target. All contributing interpolation nodes must be
supported. Tiles are loaded jointly for density/normals/support, once per request;
coarse sampling reads only touched tiles, never the dense fine-grid envelope.
Fibre constraints and mesh phase sampling use the respective physical transforms.
Qual curves have an R-tree of world-coordinate extents. Each solve block selects
overlapping curves once, irrespective of their original preparation tile; short
two-point pieces and the final endpoint survive constraint downsampling.

## Meshes and linked atlas

`meshes/mesh.sqlite` is the globally indexed vertex/face store. `meshes/MESH.json`
and `MESHES.json` identify the frozen field and per-isovalue mesh statistics.
Each `meshes/level-*.obj` exports every supported component at that nominal level.

`flat/atlas.sqlite` maps each original face ID to a chart and local face index.
Each chart has:

| File | Meaning |
| --- | --- |
| `chart-N.npz` | `vertices` XYZ µm, `faces` zero-based indices, `uv` µm, `source_face_ids`, `source_vertex_ids` |
| `chart-N.obj` | 3D vertices with UV texture coordinates |
| `chart-N-flat.obj` | Identical connectivity, vertices `(u,v,0)` |
| `chart-N.png` | Optional original-CT samples at barycentrically mapped chart points |

Charts use LSCM. Flipped, overlapping or excessively stretched charts are split;
degenerate triangles fail explicitly. Shared original vertex IDs connect chart
cuts. `FLAT.json` records coverage of mesh faces, chart count and atlas identity.
Each chart's quality is also in the atlas database. UV coverage and triangle
integrity do not establish physical sheet correctness or 3D nonintersection.
Committed atlas rows also retain the expected OBJ/PNG hashes. Partial-stage
recovery restores missing/corrupt companions from the verified chart payload and
original CT before it can publish completion.
