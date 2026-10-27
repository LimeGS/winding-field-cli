# Geometry and persisted products

## Coordinates

- Arrays and ROI arguments: ZYX. `--roi Z0 Z1 Y0 Y1 X0 X1` is half-open in voxels
  of the explicitly selected source level, never implicitly level 0.
- Field grids, fibre constraints and 3D mesh vertices: XYZ micrometres.
- Origins are voxel-centre coordinates. At source index `i_xyz`,
  `world_xyz = origin_xyz + i_xyz * spacing_xyz`.
- Output origin is the source origin plus reversed ROI-low times source spacing;
  output spacing is source spacing times explicit `--stride`.
- OME-NGFF scale/translation and axis units are interpreted. Unitless metadata
  requires explicit `--voxel-um` or catalog spacing. No silent resolution change.
- Preparation has a 40-output-grid-voxel input halo, clipped to the source volume.
  Persistent maps and extracted meshes remain within the output grid/ROI.

## Run and integrity

`RUN.json` (`winding-field-run-v1`) stores the original acquisition/metadata hash,
effective settings, implementation SHA-256, stage receipt hashes, timings and I/O
counts. `source.sqlite` records consumed source-chunk hashes. `products.sqlite`
records exported-product hashes. A run is single-writer locked.

Statuses are `STARTING`, `RUNNING`, `PAUSED`, `FAILED` and `COMPLETE`.
`physical_acceptance: false` is independent of successful stage completion.
Source metadata, consumed content, package implementation and completed products
must agree on resume. The CT cache is disposable, not the source of truth for
checkpoints. Retain the original package and data for reproducibility.

## Observations and fibres

`maps/` is a transactional tile inventory plus checksummed NPZ payloads.
`PREPARED.json` describes its physical grid, source, ROI, model provenance and halo.
Maps contain CT-derived sheet density (inverse micrometres), unsigned surface
normals (XYZ), separate fibre tangents (XYZ), presence and support.

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

`fibres/fibres.sqlite` stores bounded traced fragments in physical XYZ units.
`FIBRES.json` includes their count and database SHA. `fibres/fibres.afv` is an AFV
v1 SQLite export with application ID 1447249475. Its `frame` metadata specifies
selected-source voxel coordinates, source level and physical transform. It is
not automatically native L0. Fragments stop at unsupported gaps or tile edges.

## Field

`field/CHECKPOINT.json` and block NPYs use the reused numerical core's durable
block format. `FIELD.json` records physical grid, orientation/calibration,
constraint rows and conditional local-solver convergence. Sweeps are a budget;
`global_converged: false` means no independent global convergence is established.
The nominal unit sheet step and automatic planar/radial-Z calibration are
scientifically unvalidated.

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
