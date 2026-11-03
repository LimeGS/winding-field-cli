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
  SD2 rounds the input core coverage out to source-anchored 32³ tiles first;
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

Statuses are `STARTING`, `RUNNING`, `PAUSED`, `FAILED` and `COMPLETE`; the direct
TIFXYZ branch also records `BLOCKED` with structured reasons and `PARTIAL` for
published geometry with incomplete source coverage.
`physical_acceptance: false` is independent of successful stage completion.
Source metadata, consumed content, package implementation and completed products
must agree on resume. The CT cache is disposable, not the source of truth for
checkpoints. Retain the original package and data for reproducibility.

## Original-recipe products

`--flattening-method original` uses `MESHES.json` then
`original/ORIGINAL.json` (also published as run-root `ORIGINAL.json`). It is
independent of field resolution. The source `winding-mesh-v1` bundle is immutable.
Unlike package-generated micrometre geometry (`voxel_um=1`), external bundles may
use other world units: `xyz_um = bundle_xyz * metadata.voxel_um`. The output
frame is explicit and separate from field spacing and CT pyramid resolution.

Each `original/level-NNNN` publishes `map.npz`, `image.png`, `mask.png`, optional
`tifxyz/`, and `LEVEL.json`. `publications.sqlite` binds each level receipt hash
and its owned pending directory for crash recovery; the aggregate binds the
receipt inventory. Failed attempts remain hidden sibling directories. Resume
checks receipt hashes, the required product inventory, product hashes, settings,
mesh, CT metadata and numerical dependency versions.

Maps contain frame-voxel XYZ, UV, triangles, diagnostic/scientific eligibility,
and original source face/vertex IDs. The 4096-square raster preserves global UV
positions; PNG masks encode 0 unmapped, 1 mapped/CT unavailable, 2 observed,
3 ambiguous. Missing local/cache chunk names are bound separately in the level
receipt; a later available payload requires a new export.

Optional TIFXYZ uses float32 XYZ with sentinel −1 and a separate conservative
validity mask. `meta.json` declares output frame, origin, physical voxel size,
UV pitch and reciprocal pitch scale. `VERIFIED.json` records TIFF readback and
excluded node/cell counts, not physical acceptance or actual VC3D opening.

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

The grid adapter orients fine unsigned normals before trilinearly sampling the
physical density-normal target. All contributing interpolation nodes must be
supported. Tiles are loaded jointly for density/normals/support, once per request;
coarse sampling reads only touched tiles, never the dense fine-grid envelope.
Fibre constraints and mesh phase sampling use the respective physical transforms.

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

## Native global surfaces

New CLI runs select `surface_output=native`. The parent stage path is
`prepare/fibres/field/surfaces`; mesh, flatten, assemble and render are independently
checkpointed per level inside `surfaces/level-NNNN/`. Stage commands/`--until`
accept those internal stage names. Each completed W is published before work on
the next one. Existing unselected/atlas configurations retain the old stage path.

`surfaces/NATIVE.json` binds field, CT, effective options, implementation and actual
level selection; `LEVELS.json` must agree. `IDENTITY.json` binds each level to that
parent. `work.sqlite` records committed payload hashes, charts, source IDs, shared
edges, assembly frontier, explicit cut edges, raster tiles and cumulative active
time. Atomic files are registered only after their bytes are durable. Recovery
never replaces a changed committed payload. Temporary files are not products.

`LEVEL.json` describes completed products, mapped/source area, target and outcome,
phase timings, CT counts and TIFXYZ availability. `SURFACES.json` aggregates levels.
Incomplete user/time stops return PAUSED without registering an unfinished parent
stage. Published partial geometry remains PARTIAL even when a lower requested
coverage target was met. `physical_acceptance=false` and
`field_coverage_certified=false` always remain explicit.

`coverage_target` is in `(0,1]`, default1. It measures original mesh area with valid
assembled UV, not unique observed CT area or geometric coverage of the implicit
field. All source components/faces remain in the denominator. Parent-edge welding
preserves shared coordinates; non-tree cycle edges are explicit cuts. Pixel-level
XYZ conflicts are masked; there is no global subpixel-overlap or3D-intersection
certificate. Chart distortion checks are retained.

`surface_time_limit_s` defaults to1800 active seconds per W, accumulated across
resumes. `resume --extra-surface-seconds N` appends an allowance record without
changing the original request. `original_deadline_met` compares against the
original limit. Interrupted/cold-start validation and worker overhead are reported
by supervision; wall time paused between invocations is not active compute time.

Each component's `RENDER.json` records pitch, grid origin, shared intensity window,
source mapping and mask counts. `sheet.png` is grayscale+alpha from original CT;
`overview.png` samples that same canvas. `mask.png` values are0 background/unmapped,
1 geometry without supported CT,2 observed CT,3 conflicting XYZ. Geometry-valid
samples are1/2; a missing CT sample does not by itself remove a TIFXYZ position.

Optional mesh-derived TIFXYZ uses the existing float32 tiled writer and explicit
output-frame conversion. It requires an isotropic output frame and usable local
four-corner cells; invalid optional output is reported UNAVAILABLE with a reason.
Its metadata/provenance identifies `native-mesh-v1`. It is not accepted by the
certified-direct `verify-tifxyz` protocol and does not fabricate its audit receipts.
Orphan grid nodes that belong to no four-valid-corner cell are omitted only from
TIFXYZ; the receipt records their count. The shared XYZ coordinates are unchanged.
Each active cell also requires mapped support at its center and no unpaired
source boundary/hole/cut crossing its interior. Exactly coincident, oppositely
oriented shared UV edges are internal seams. Original float32 cells are classified
in bounded, journaled masks bound by `TIFXYZ-MASK.json`. The four corner nodes of
every unsafe cell are removed, then orphan normalization and all geometry gates
run again on the exact delivered subset. Counts distinguish unsafe cells, removed
nodes, collateral cells and orphans; `partial_geometry=true` prevents COMPLETE.
Frame failures or no remaining surface still yield UNAVAILABLE. Source XYZ and
PNG geometry remain unchanged.
`RENDER.json` binds the image/grid before optional TIFF publication;
`TIFXYZ.json` reports the optional output. `.tifxyz.pending/` is atomically renamed
only after channels and metadata validate. The immutable `CT-SNAPSHOT.sqlite`
snapshot binds consumed logical CT chunks; authoritative Zarr fill can be valid
data, so `observed_ct` is not evidence of papyrus/ink. `ct_value_range` reports
the actual sampled intensity range, including all-zero patches.
`usable_render` requires contrast within at least one observed component; an
all-zero or constant canvas stays valid data but is reported `ct-no-contrast`.
The snapshot has a distinct name from the mutable `ct-source.sqlite`, including
on case-insensitive filesystems. Completed CLI resume revalidates snapshot-bound
CT bytes inside its owned worker. A final `LEVEL.json` is published only when its
ledger entry is committed; interruption between rename and registration is
recoverable rather than mistaken for a completed level.

Native local charts first try an orthogonal tangent projection with <=2% measured
local metric change; otherwise they retain LSCM. Persisted chart quality reports
both local method and assembled edge stretch. UV work is streamed in small batches
and stops before using the render reserve (40% of time remaining after mesh).
The render identity binds all mapped chart payloads, UV hashes and component bounds.
Resume of an interrupted raster keeps that geometry frozen; it never reuses tiles
at a shifted origin or after adding new charts. A completed partial publication
does not silently start a new mapping attempt on resume.

Native compute is supervised in an owned process group: STOP/SIGINT requests a
bounded stop; the worker may be killed after2seconds if unresponsive. Transactional
mesh/chart/tile checkpoints survive. RSS sampling covers worker group plus parent;
the3GiB envelope and16GiB intermediate-work quota do not certify whole-W viability.
Products/cache and the caller's disk reserve are accounted separately.

### Experimental standalone global mapping

`tools/bench_global_native.py` consumes an explicitly selected component of an
existing single-level indexed mesh in micrometres. It computes one global cut,
positive mean-value initialization and direct ARAP mapping, then qualifies faces
with the native metric/orientation gates and a complete bounded UV-overlap audit.
Both participants of every detected overlap are excluded and listed. All input
components/faces remain in the denominator. The same shared renderer consumes
bounded chunks of this ONE UV solution rather than independently placed charts.

Request schema `experimental-global-native-v3` binds the mesh manifest, exact frame
bytes, CT identity and requested options. Worker-bound `RUNTIME.json` fixes code
and numerical versions. Each stage has its own committed `PHASE.json` and bounded
numeric units. `PUBLICATION-RECORD.json` fixes authoritative render/timing evidence;
`PUBLICATION.json` seals the final `RESULT.json` last. Resume checks source identity,
phase artifacts, CT snapshots and summary claims in an owned supervised worker.
STOP/deadline admission applies before this verification too.

Source-triangle coverage, retained-grid connectivity and raster coverage are
reported separately. The command is experimental and may publish PARTIAL; it does
not produce direct-field certification or establish physical-sheet correctness.
See [the command and measured scope](verification/native-global-export.md).

## Direct TIFXYZ (experimental)

The saved `surface_output=atlas` selects `prepare/fibres/field/mesh/flatten`
(also old configurations without the key); `tifxyz` selects
`prepare/fibres/field/tifxyz`. Old configuration JSON is not backfilled on resume.
`RUN.config.tifxyz_request` contains the normalized JSON by content and
`tifxyz_request_sha256` binds it. Unknown/duplicate keys and nonfinite JSON are
rejected. Levels are explicit W isovalues, independent of physical XYZ units.
The JSON `coverage_policy` enum is `strict` (default) or explicitly `allow_partial`;
it is also frozen in `EXPORT.effective_settings`. Resume cannot change this policy.

The integrated export root is `RUN/tifxyz/`; standalone export uses its separate
`OUT`. `EXPORT.json` binds effective settings, source identity, exporter code and
numeric versions, reader policy and verification scope. Geometry-dependent
defaults are resolved by the supervised worker and persisted once. Standalone
resume requires this saved identity; it never creates a replacement root for
missing or foreign work. The exporter owns its own lock and transactional work.

`effective_settings.audit_backend` is `numpy`, `cpu`, or `cuda`. `audit_accelerator`
binds the selected backend and, for native modes, ABI1 plus manifest/library SHA256.
Every audit and final publication/verification gate checks that frozen binding.
Native manifests are accepted only through `WINDING_TIFXYZ_ACCELERATOR` supplied
by the operator; no native path from EXPORT.json is executed. CUDA visibility is
likewise chosen only through `WINDING_TIFXYZ_CUDA_DEVICE`, an explicit GPU UUID.
Native sources ship with the wheel; `python -m winding_field.tifxyz_build` builds
an artifact into a new external directory. It is not called by runtime commands.

### Coordinates and request settings

`roi` is null for the whole persisted field, or
`{"lo_zyx":[0,0,0],"hi_zyx":[513,513,513]}` for an explicit half-open subset of
**field nodes** (within the checkpoint, at least two nodes per axis). It is not a
CT voxel crop. Internal XYZ are physical micrometres; the ROI retains absolute
coordinates and cannot borrow interpolation support from outside its bounds.

`pitch_um` defaults to minimum physical field spacing; `tolerance_um` defaults to
maximum spacing. `output_frame="selected-source"` derives only the SHA-bound
isotropic selected CT level. It never infers L0 from a pathname. Alternatively,
provide all fields of an explicit destination frame:

```json
{
  "id": "declared-volume-level",
  "origin_xyz_um": [86.4, 172.8, 259.2],
  "voxel_um": 8.64,
  "shape_zyx": [100, 100, 100],
  "source_identity_sha256": "<64 lowercase hex characters identifying the destination volume>"
}
```

The placeholder must be replaced by the actual destination identity, not a guessed
frame. Source and destination origins are voxel-centre coordinates. Conversion is
`xyz_tif=(xyz_um-origin_xyz_um)/voxel_um`, once. For the frame above, physical
`[112.32,250.56,423.36]` µm becomes destination XYZ `[3,9,19]`. Valid nodes must lie
within destination bounds. Invalid nodes have all three coordinates `-1`.

`level-NNNN/` follows the requested level order and contains tiled 256² float32
`x.tif`, `y.tif`, `z.tif` (deflate; BigTIFF when needed), `meta.json`,
`.segment.json` and `VERIFIED.json`. Metadata records the exact level, frame, UV
origin, pitch, shape and scale. For physical UV pitch `p`,
`scale=[voxel_um/p,voxel_um/p]`; the last row/column is not stretched to fit a
bounding box. UV pitch does not imply physical isometry.

### Publication, verification and limits

`TIFXYZ.json` aggregates the levels, artifact hashes, work inventory and source
scope. Its `status` is READY iff every requested level is READY, otherwise PARTIAL;
`coverage_complete` is the corresponding boolean. Each level descriptor and its
SHA-bound `VERIFIED.json` carry the same per-level `status` and `coverage_complete`.
All requested levels must have valid delivered geometry.

Exactly one marker, `READY.json` or `PARTIAL.json`, is written last. It carries the
matching `status`, `coverage_complete`, `tifxyz_sha256` and `identity_sha256`.
Partial publication requires effective `coverage_policy=allow_partial`; explicit
permission does not force a partial outcome when all coverage passes. Conflicting
or missing markers, inconsistent root/result/level quality, forged complete flags
and invalid SHA bindings fail before parent registration.

A bound READY or PARTIAL child registers the integrated `tifxyz` stage. The stage
records `status`, `coverage_complete`, and either `ready_sha256` (complete) or
`partial_sha256` (partial), alongside its ordinary receipt hash. READY permits
`RUN.status=COMPLETE`; PARTIAL leaves `RUN.status=PARTIAL` and
`RUN.coverage_complete=false`. `products.sqlite` snapshots the new
artifacts, including retained work, effective settings, producer snapshot and
per-level metadata/TIFFs and the selected marker. Resume checks these hashes and
invokes fresh supervised verification; partial stays partial. It does not assume
a `FLAT.json` or atlas gallery.

Only post-geometry `SOURCE_UNCOVERED` or `COVERAGE_DEPTH_LIMIT` qualifies for partial
publication. The partial audit has `geometry_complete=true`,
`blocked_phase="coverage"`, `source_scan_complete=false`, and structured
`reasons[].region` identifying the first uncertified source cell, dyadic subcell
and depth. Unexamined coverage remains **UNMEASURED**; this is neither an exhaustive
hole map nor an assertion that only one region is missing. Empty/invalid geometry,
ambiguous roots, resource exhaustion and integrity failures remain hard failures.
Partial publication does not fill gaps or discard extra points: XYZ and `-1`
masks are unchanged. Export alone does not claim independent root reprojection.

Reader scope is four valid corners per cell, both `nw-se` and `ne-sw` diagonal
policies, no inpainting. Orphan nodes do not count as delivered coverage. READY
certifies the declared supported ROI at the recorded spatial tolerance, including
delivered float32 rounding, support, root/UV mapping and intersection checks. It
does **not** establish physical sheet correctness, identical native faces or
native topology. The immutable flags remain `physical_acceptance=false`,
`native_face_identity=false`, `native_topology_equivalence="UNMEASURED"`.

Source scope is `metadata_scope="global-bound-source"`,
`payload_hash_scope="consumed-roi-payloads"` and
`all_checkpoint_payloads_verified=false`. Coverage is the explicitly recorded
field-node ROI, not automatically the whole W. The historical
`producer-run.snapshot.json` preserves the initial RUN bytes. Stable source
binding includes producer schema/implementation, source/frame, FIELD receipt and
checkpoint/geometry; it excludes mutable RUN status, timings and later stages.
Consequently parent PAUSED/PARTIAL/COMPLETE updates do not break export resume/verification.
Changed source bindings or payloads consumed by verification still fail. The
producer's implementation may be older than the exporter; legacy workflow resume
continues to require the original implementation.

`verify-tifxyz OUT` reports `TIFXYZ_VERIFIED` for READY or
`TIFXYZ_PARTIAL_VERIFIED` for PARTIAL after a fresh admitted worker independently
reprojects, rebuilds and audits the publication, including exact quality/region
receipt comparisons. Both mean file-valid; only the former certifies full declared
coverage. It does not promote candidates, repair roots, remove STOP, or
change files in OUT or the source. Verification may use a temporary sibling
directory under its admitted scratch/disk budget.

The bounded VC3D reader helper accepts exactly one SHA-bound READY/PARTIAL marker.
Reader proof copies `publication_status` and `coverage_complete` from the root,
plus the selected `level_status` / `level_coverage_complete`. It records
`source_coverage_verified=false` and `independent_root_reprojection=false`: geometric
OBJ/TIFF agreement proves neither. A fixture with no publication is a reader-only
control and has no `coverage_complete` claim. The reader never upgrades PARTIAL.

| Child result | Integrated status | New completed stage | CLI exit |
| --- | --- | --- | --- |
| READY | COMPLETE | Yes, TIFXYZ + READY hashes | 0 |
| PARTIAL | PARTIAL | Yes, TIFXYZ + PARTIAL hashes | 2 |
| PAUSED | PAUSED | No | 0 |
| Resource/work-limit BLOCKED | BLOCKED | No | 75 |
| Geometry BLOCKED | BLOCKED | No | 2 |
| Integrity/unrecoverable IO failure | FAILED | No | 1 |

Standalone verification exits 0 for `TIFXYZ_VERIFIED`, 2 for
`TIFXYZ_PARTIAL_VERIFIED`; existing PAUSED/BLOCKED/FAILED exit behavior is unchanged.

Linux admission uses at most 3 GiB for worker plus controller, a separate 2 GiB
RAM reserve, scratch capped at 16 GiB, and `reserve_disk_gib` (default 12.25 GiB).
`run --reserve-gib` remains the CT/cache/product reserve, not the RAM reserve.
The planner reports branch-only product allowances and the separate reserves;
direct estimates contain no native-OBJ allowance. Estimates are not acceptance:
the worker enforces admission and actual quotas. macOS production runtime is
unsupported. Optional libigl/tifffile dependencies must already be provisioned.

Hybrid filters process at most65536rows; audit queues are bounded to4096. Uncertain
numeric/contact filters retain the exact implementation. CUDA uses FP64 outward
rounding with FMA/FTZ/fast-math disabled; its host library also supplies bounded
exact integer contact filters. Unsupported numerical ranges fall back to Python.
Read-only audit checkpoints poll STOP at0.1s and resource accounting at1s; all
pre-write quota/journal checks remain. SQL batches stay<=128rows, grouped into
transactions of<=2048rows. No global SQL materialization or output-sized RAM index.
STATUS.audit records subphase and counters without altering accepted audit receipts.
CPU supervision records `enforced_limit=RLIMIT_AS` and `child_address_ceiling`;
CUDA records `enforced_limit=RLIMIT_DATA` and `child_data_ceiling`, plus the same
aggregate RSS observation. Native GPU buffers are fixed-size and require256MiB
free VRAM. Large virtual CUDA mappings are not reported as resident RAM.

This integration is experimental preparation under the approved quality-first
amendment. Geometry/integrity and ≤3 GiB remain final acceptance requirements;
comparative RAM/time are `UNMEASURED` (`NO_COMPARISON`, reference OOM). Neither
measured speedup nor whole-W acceptance follows from CLI tests or an ROI result.
