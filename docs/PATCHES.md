# Historical patch flattening and CT/TIFXYZ

`export-sheets --flattening-method patches` consumes an immutable mesh bundle or
a **validated RANGE wrapper**. Join ranges in 3D first with `join-ranges`.
Do not pass the wrapper's inner `mesh/`: slot 0 is storage, not W=0.
Ordinary mesh levels are exported separately; this command never invents joins
between different constant-W surfaces. Imported meshes retain their saved slot
values, which do not by themselves establish a calibrated winding value.

```sh
winding-field export-sheets --mesh-bundle joined-range \
  --flattening-method patches --flattening-config patches.json \
  --source https://example.org/scan.zarr/2 --voxel-um 9.6 \
  --output new-patch-export --export-tifxyz
```

Example `patches.json` (paths must refer to existing installed dependencies):

```json
{
  "output_frame": {
    "id": "Paris4/L2",
    "origin_xyz_um": [0, 0, 0],
    "voxel_um": 9.6
  },
  "app_run": "/path/to/historical-vc3d/AppRun",
  "pitch_um": 19.2,
  "max_patch_faces": 4096,
  "max_depth": 3,
  "max_charts": 4096,
  "timeout_seconds": 120,
  "tile_size": 1024,
  "memory_gib": 2,
  "cache_dir": "/shared/disk/patch-ct-cache",
  "cache_mib": 64,
  "cache_gib": 2
}
```

`pitch_um` is physical UV sampling pitch, **not field spacing**. It defaults to
19.2 µm. Select 76.8 explicitly for the older preview scale; a field sampled at
307.2 µm does not imply either image pitch. UVs are edge-length calibrated by
the existing Flatboi kernel; metric distortion still changes physical sampling
locally and is reported, not certified away. Frame coordinates obey
`XYZ_output = (XYZ_field * source_voxel_um - origin_xyz_um) / output_voxel_um`.

## Bounded source-key mapping

### Continue verified PNG tiles

For a single saved `level-0000` PNG export, continue in a fresh directory:

```sh
python -m winding_field.patch_continue \
  --source-export old-export/patches --mesh-bundle original-joined-range \
  --source-identity-sha256 SHA256_OF_OLD_IDENTITY_JSON \
  --config resource-overrides.json --output new-export
```

The source needs its hash-bound `map.npz`, `keys.sqlite`, `MAPPING.json`, original
mesh, CT ledger at `old-export/ct-chunks.sqlite`, and either a saved
`MAP-REPORT.json` or completed `LEVEL.json`. Ordinary interrupted exports without
either report cannot be continued this way. Paris4's gated runner saves the map
report before rasterization. Stop the old producer before operational handoff.

Overrides are limited to cache location/budgets, raster workers and memory/disk
resource limits. Geometry, lattice, CT identity and numerical runtime remain
bound to the source. Complete tiles are copied after receipt/payload validation;
their original receipts and identity are retained as provenance. Incomplete tiles
are recomputed. Source face row indices are reconstructed from sorted original
mesh IDs. The prior CT hash ledger is inherited so reloads still detect changes.

The command performs no extraction or flattening. It supports PNG/mapping
continuation only; TIFXYZ and multiple levels are rejected. The existing
`export-sheets --resume` behavior is unchanged. Results remain diagnostic.

### Parallel native calls and verified reuse

The same public `--flattening-config` accepts `"workers": 1` (default) through
`4`. Parallel execution requires an explicit `"native_cpus": [2,3,4,5]` CPU
pool (Linux `taskset`), with at least one CPU per worker. Each native child uses
one BLAS/OpenMP thread. This overrides a narrowly inherited parent affinity.
`process_tree_gib` defaults to 11 and `disk_reserve_gib` to 8; parallel execution
checks descendant-process RSS and disk headroom every 0.1 seconds. These sampled
guards complement, rather than replace, the external supervisor/hard host limit.

At most `workers` native results/jobs are held in lookahead. Results commit in
the original DFS order, including failure subdivisions and chart-budget omissions.
Speculative jobs never acquire chart IDs or budget. `_pending-*` directories
retain uncommitted/cancelled work; only `patch-*` directories are committed.
Native command paths record the actual invocation before the directory rename.
Seams and registration remain serial. Rasterization defaults to serial and is
deterministic for the same saved UVs. Independent native runs are **not** promised bit-identical.

### Bounded raster workers and verified CT cache

`raster_workers` accepts 1 (default) through 4 independently of native `workers`.
Threads share immutable geometry and a synchronized CT reader; at most that many
tiles are in flight. The coordinator publishes receipts in original tile order.
Raster buffers, CT workspace and scratch admission scale with worker count.
Threads inherit the parent CPU affinity; `native_cpus` only configures native
children. Keep the serial default until a representative workload shows a gain.

`cache_mib` budgets a shared LRU of verified, read-only decoded CT chunks,
including per-entry overhead. The native RAM cache is disabled to avoid duplicate
retention. RAM hits reuse the verified immutable bytes; eviction/reload checks
the saved SHA256 again, including missing-chunk identity. Native loads and SQLite
access are serialized; interpolation runs outside the lock.

`cache_dir` may reside on a dedicated SSD; `cache_gib` caps the native disk cache.
Cache misses require `disk_reserve_gib` free space plus two decoded chunks on that
filesystem. Output scratch has its own reserve check. Paris4 experiments use
`/mnt/fast2/vesuvius-ct-cache/paris4`, 64 GiB disk and 1024 MiB RAM with an 8 GiB
reserve. These are benchmark settings, not revised defaults. Use a fresh output
namespace; do not alter an active export's frozen implementation or cache.

To reuse completed native work, set both `reuse_export` to the old export's
`patches/` directory and `reuse_identity_sha256` to that directory's
`IDENTITY.json` SHA256. Choose a fresh, disjoint output namespace. The new identity
binds the old identity and a snapshot of completed receipt hashes. Source mesh,
frame, partition budgets, native pins, exact input OBJ, output/log hashes and
replayed binding/scaling/UV diagnostics must agree. Copied outputs retain a
`source-receipt.json` and `reused_from` provenance. Invalid artifacts abort;
missing/incomplete/failed native attempts are counted and recomputed, not claimed
as reusable. Finite UVs rejected by the UV gate remain eligible for replay as
the original ambiguity blockers. Final receipts distinguish copied from committed
reuse counts. Recheck the inventory after the old owner reaches a safe checkpoint.

Mesh loading gathers at most 4096 numeric rows at a time; IDs stay int64 (never
float64). Original keys remain in SQLite. Each exported level includes
`keys.sqlite` (exact source ID/key projection, including omitted faces) and
`MAPPING.json`, binding its hash, `map.npz`, source mesh manifest and immutable
source database hash. A copied delivery does not require the source path online.
Small maps also retain legacy `source_vertex_keys`/`source_face_keys` arrays;
large maps omit these arrays explicitly and require the mapping reader:

```python
from winding_field.patch_mapping import PatchMapping
with PatchMapping(level_directory) as mapping:
    xyz = mapping.arrays['field_xyz']
    for source_id, key in mapping.iter_keys('vertices', batch_size=512):
        pass  # consume keys incrementally; faces use iter_keys('faces')
```

The reader also accepts old maps with embedded keys. It rejects missing key
provenance and changed map/key-database hashes. Key lookup preserves original
integer IDs and mapped order; it does not synthesize replacement keys.

The existing partition, sparse solver, UV placement, folded-footprint blockers
and global two-node TIFF halo are unchanged. Numeric assembly now preallocates
arrays and releases chart buffers after their last use. Whole-mesh admission is
still conservative: these reductions alone do **not** establish full-height
three-turn admission, and the memory guard has not been relaxed.

For image-first delivery, omit `--export-tifxyz` explicitly: PNG, masks and
mapping/provenance still retain the requested isotropic pitch and domain. This
does not deliver TIFXYZ; a requested TIFF export is never silently skipped.
Budget PNG/mapping and optional TIFF disk usage separately before a full launch.

## Dependency and algorithm contract

- Existing `vc_reader` is required for CLI CT reads. There is no installation or
  automatic substitute. Native transport uses one thread/connection, a bounded
  RAM cache, a shared bounded lossless disk cache and an on-disk chunk hash ledger.
  Missing chunks remain missing; exact zero-weight interpolation corners do not
  invalidate a sample. CT is central trilinear sampling with fixed **0..255** display.
- Historical launcher SHA256:
  `c1e0ca2ea1616967f6da5907c60ea3fe114e840834154ccde758856f85510517`.
  Dispatched `AppRun` sibling `usr/bin/flatboi` SHA256:
  `d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38`.
  Both are checked before/after native calls. The default `native_recipe` is
  `historical` and still rejects the installed binary below. No pin-override switch.
- Explicitly authorized alternative: `"native_recipe": "installed-2f89bfc9"`
  selects binary SHA256
  `2f89bfc96512bebf7ef44b25de01c5afd01c7af058e452515b03bc52cf6afd6a`
  with the same launcher pin and numerical parameters. Receipts bind this separate
  recipe with `historical_reproduction=false`; upstream release/commit is unknown.
  Unknown recipes and any different binary hash are rejected. This is a newly
  validated version, not reproduction of the historical executable.
- Native invocation is `AppRun flatboi inputmesh.obj 200 symmetric_dirichlet --tol=1e-6`.
  This reuses `surface_export.flatten_native`, including source OBJ correspondence,
  world edge-length scaling, timeout/process-group handling and UV diagnostics.
- All edge-connected source components are split with the existing balanced PCA
  face partition. Failed native charts are recursively split within explicit depth
  and attempt budgets. Rejected finite-UV faces remain in their original chart,
  following its registration, deformation and final placement as ineligible raster
  blockers. Their footprints mask overlapping CT and XYZ as ambiguous even if a
  later weld makes their UV orientation positive. Faces without any UV remain
  explicitly unmapped; rejected area and source IDs remain accounted for.
- Registration, common seam targets and cotangent displacement extension adapt
  `six-surfaces-opt-20261016/global_sheet.py`. The historical seam regularizer's
  256 L2² / 1e-4 L2 constants are converted to the declared physical output units.
  Recovery uses the positive-face partition from `face_recovery.py`; splitting
  reuses the existing `surface_atlas` kernel used by the historical wrappers.
- Seams use **original global vertex/edge IDs and exact XYZ**. Coincident geometry
  without shared IDs does not join. No atan2 branch pruning or new global UV
  optimizer. Shared seam vertices are welded in output connectivity as well as UV.
- Every source component is counted. Disconnected mapped components are translated
  into separate horizontal regions; this is disclosed layout, not physical joining.
  Native/UV failures and remaining folded faces are counted, including omitted area.

## Products and masks

`PATCHES.json` binds the input, settings, native pins and product hashes.
Each `patches/level-NNNN/` contains:

- `map.npz`: original `field_xyz`, converted `world`, indexed `faces`, UVs,
  source vertex/face IDs and keys, scientific/display masks and omitted face IDs.
  Finite-UV blockers remain in `faces` with display/scientific false;
  `omitted_source_face_ids` means no UV map, while `omitted_uv_source_face_ids`
  also includes these mapped-but-ineligible blockers.
- `LEVEL.json`: input slot semantics, all-component/area accounting, native attempt
  receipts, seams, registration, distortion, omitted-UV count and render inventory.
- `patches/`: preserved native OBJ/log/receipt intermediates; no scratch deletion.
- `render/tile-.../`: `image.png`, `mask.png`, barycentric XYZ plus source-face-row
  mapping, and `TILE.json`. Optional `tifxyz/` contains float32 XYZ TIFFs, a validity
  mask, frame/pitch metadata and exact TIFF readback verification.

Masks are **0** unmapped, **1** mapped but CT unavailable, **2** observed CT,
**3** ambiguous/invalid mapping. Geometry remains valid independently of missing CT.
TIFXYZ conservatively excludes boundary-crossing, folded/collapsed/jumping cells
and orphan nodes, using the existing `original_tifxyz` implementation. A tile with
no safe cells explicitly reports `EMPTY_TIFXYZ`; it is not a positive export.
No retained safe cell is fabricated to make a tiny chart pass.

Rows follow **decreasing UV V** in CT, mapping buffers and TIFFs alike. Metadata
records this convention; this is not the historical increasing-V per-chart grid.
TIFF readback is tested, but external VC3D readback is not claimed.

Large canvases are stored as exact-pitch bounded tiles with one-node publication
overlap. Computation uses an additional **two-node halo on every side**, clipped
only at the true global canvas boundary. Orphan membership depends on neighboring
pruned nodes (one step); those depend on incident unsafe-cell corners (another
step). Two nodes therefore cover the full exclusion dependency. Unsafe-cell
testing, corner pruning and orphan removal run before cropping publication.
Full source topology is used for boundary exclusions; storage edges create none.
Global pixel origins/offsets keep fractional-pitch arithmetic consistent between
tiles and the single canvas. TIFF metadata describes the cropped published region.
Shared valid nodes can be published even when a narrow edge tile contains no
complete cell (`TIFXYZ_SHARED_NODES_READBACK_VERIFIED`, zero retained cells).
Counts explicitly include publication overlap but exclude computational halo. The
renderer never silently coarsens the requested pitch. It does not create a single
huge PNG/TIFF or an incremental/global-UVW map.

## Resource and execution limits

Geometry and the seam graph are in memory, admitted conservatively at 2048 bytes
per source vertex/face after reserving halo-expanded raster/cache/runtime memory
and CT chunk workspace. After CT metadata, before opening native transport,
admission additionally reserves **four decoded chunk sizes**: native decode buffer,
codec workspace, newly materialized Python bytes, and the previous chunk still held
by the sampler during the next callback. This is independent of native cache size.
Insufficient whole budgets are rejected before transport; the same reservation is
subtracted from geometry capacity and recorded in `IDENTITY.json`. Oversized meshes
are rejected, not truncated to the largest component. Pixel buffers are tile-bounded;
native charts have bounded face count and timeout. Disk is checked between stages
and tiles, but native scratch and total output disk use are not preallocated.
Use a shared data-disk cache and set `SQLITE_TMPDIR` to that disk where appropriate.
For a one-CPU run, set BLAS/OpenMP threads to 1 and use CPU affinity on the launcher.

`--field-config`, `--levels`, and `--resume` are rejected for patches. Extraction is
a separate first step. A failed run retains diagnostics; retry into a **fresh**
output. Publication has no completed receipt on failure. The full-range real-scale
run and independent review remain separate from synthetic software tests.

All outputs are **unaccepted diagnostics**. Software correctness does not gate on
physical CT agreement and does not establish physical-sheet identity or flattening
quality. A missing historical binary blocks historical native smoke; a different
installed binary must not be described as the pinned recipe.
