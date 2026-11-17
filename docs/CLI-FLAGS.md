# CLI flags and config parity

This feature is under development after v1.5.7. The coverage/mappings below
describe the implementation contract, not flags already available in that release.
Existing published options and native config formats remain supported.

## Rules

- Explicit flags override the optional config, which overrides existing defaults.
  Omission, `false`, zero and a supported automatic/null setting are distinct.
- A list supplied by flags replaces the config list; repeated flags preserve order.
- Config-relative paths use the config directory. Flag paths use invocation CWD.
- Existing scientific validators remain authoritative. No field/frame/hash,
  calibration, support, runtime or resource requirement disappears in either mode.
- Saved requests and receipts are still generated automatically. “No config”
  means no manually written settings file, not removal of scientific metadata.
- All public commands receive `--cli-config FILE` for defaults using their argparse
  argument names. Where `--config` has no existing meaning, it is an alias. Where
  it already names a native recipe, that meaning is unchanged; `--cli-config`
  supplies complete CLI options and may coexist with that legacy recipe.
- Config mode is typed defaults, not JSON-to-argv translation: required options
  and positionals, arity, choices, booleans, finite numbers and source exclusivity
  still validate before work. A config cannot change the selected command.
- Neither mode accepts credentials as new flags or logs secret values.

## Coverage inventory

Each row is a required acceptance family. Settings-only content is not exempt
merely because its current argument is called metadata/case/inputs/manifest.
The corresponding family tests must compare maximal file and flag requests.

| Public family | Existing authority | Required settings/flag groups |
|---|---|---|
| `run`, `plan` | `cli.py`, `config.py` | Every existing option; nested original mapping and direct-TIFXYZ requests; imported input geometry and manifest settings |
| `resume`, `prepare`, `fibres`, `field`, `mesh`, `flatten`, `assemble`, `render`, `status` | `cli.py`, `workflow.py` | Run path; existing offline, sweep and explicit extra-time controls only; config mode must not make frozen scientific settings mutable |
| `prepared build/resume/verify` | `field_workflow.load_run_config` | Frame/voxel/grid/sheet/calibration; target, weights, observed, fixed, constraints and order artifacts+SHA; solver; execution; GPU lease; source identity |
| `prepared mesh/surfaces` | `surface_workflow.load_surface_config` | Field or mesh descriptor; field geometry; raw levels; extraction execution; render settings; parallel levels; resume |
| `prepared render` | `surface_workflow._render_settings` | Mesh path, output, dimensions, views, triangle batch, levels, reserve |
| `export-sheets` | `original_pipeline.settings`, `patch_pipeline.settings`, `flattening_matrix.settings` | Run/field/mesh exclusivity, levels, CT and selected level, output frame, mode-specific pitch, budgets, cache/workers, backend/native settings, centerline/reference controls, TIFF and resume |
| `extract-range` | `range_surface._recipe`, `_bounds` | Field+geometry, repeated umbilicus points, handedness, W/theta reference, samples per W, maximum edge, exact decimal W bounds |
| `join-ranges` | `range_surface` | Actual input range artifacts and output; no extra settings recipe |
| `export-uvw` | `uvw_pipeline`, field descriptors | Field+geometry, reference-sheet data, levels, CT/calibration/selected level, canvas, workers, cache, output and resume |
| `tifxyz`, `verify-tifxyz` | `tifxyz_contract.validate_request`, `config.tifxyz_settings` | Full direct request or saved output; verification remains read-only |
| `constant-sheet prepare/validate` | `sheet_prepare`, `sheet_finish` | Mesh/prepared/native data, W, repeated centerline points, reference Z and explicit frame bindings |
| `constant-sheet flatten/import-native/status` | `flatboi_checkpoint` | Existing prepared/native/checkpoint/binary inputs and all existing execution flags; generic config entry |
| `constant-sheet render` | `sheet_finish.render` | CT/source/frame, sampling/render/cache/budget settings currently inside `--config` |
| `sheet-atlas build/query/verify` | `sheet_atlas` | Existing sheets/atlas data and parameters; complete config entry |
| `surface-export` | `surface_export` | Field/CT data, field/CT geometry currently inside metadata, support, seed, raw level, mesh step, backend, iterations/deadline, WORLD pitch/depth offsets |
| `surface-atlas` | `surface_atlas` | Mesh/data inputs, frame declarations, prior/CT/cache/selection settings in mixed inputs/metadata, chart/triangulation/budget controls |
| `surface-probe` | `surface_probe` | Field/metadata/data, dataset CT/prior declaration, optional human patch/frame, native binary, reusable frozen input artifacts |
| `surface-geometry-check` | `surface_geometry_check` | Existing input/gate/policy artifacts and pins; complete config entry, no replacement for external policy data |
| `ct-visuals` | `ct_visuals` | Existing result/case data, explicit field/CT geometry and dataset/calibration declarations; plotting/render settings |
| `patch-visuals` | `patch_visuals` | Field/human-patch data, metadata geometry, frame, output, step and margin |
| `run-watch` | `run_watch` | Existing supervision flags/command; config entry without permitting shell reinterpretation |
| `inspect`, `doctor`, `models`, `demo` | `cli.py` | All existing positionals/options and typed config entry; inspection/doctor retain existing side-effect contracts |

### Closed schema leaves and units

| Settings object | Accepted leaves to cover | Mapping/type contract |
|---|---|---|
| Prepared field geometry | `frame`, `voxel_um`, `grid.origin_xyz`, `grid.spacing_xyz`, `grid.shape_zyx`, `sheet_step`, `calibration` | Field strings/scales, XYZ numeric triples, integer ZYX triple; distinct from CT selected level |
| Prepared input artifacts | `observations.target`, `.weights`, `.observed`, optional `fixed`, `constraints`, `order` | Each contains an explicit local path and lowercase SHA256; corresponding named artifact/pin flags |
| Prepared solver | `device`, `dtype`, `rtol`, `maxiter`, `preconditioner` | Existing choices/types/defaults; no new backend or bypass |
| Prepared execution | `chunk`, `halo`, `sweeps`, `cache_bytes`, `read_workers`, `constraint_cache_bytes`, `prepared_cache_bytes`, `max_bytes`, `reserve_bytes` | Integer counts/bytes; distinguish missing/zero where accepted |
| Prepared GPU lease | `index`, `uuid`, `lock_path`, `min_free_bytes`, `deadline` | Existing `GpuLease` validator; path has its own base; CPU/GPU mismatch still errors |
| Extra provenance | `source_identity` | Repeated `--source-identity KEY JSON_VALUE`, preserving arbitrary value types; duplicate/nonfinite values reject |
| Remote field | `kind`, `url`, `sha256`, `complete_sha256`, `support_policy` | `--field-*` flags, unchanged remote metadata/COMPLETE pins |
| Local field | `kind`, `path`, `sha256`, `mask.path`, `mask.sha256`, `support_policy` | `--field-*` paths/pins; mask remains optional scientific data |
| Surface geometry | `frame`, `voxel_um`, `origin_xyz`, `spacing_xyz`, `shape_zyx`, `sheet_step`, `calibration` | `--field-*` geometry and sheet/calibration flags; preserve original coordinate units |
| Surface execution | `tile`, `level_batch`, `cache_bytes`, `reserve_bytes`, `mesh_bytes` | Existing integer ranges/defaults and unit-specific help |
| Surface render | Existing `_render_settings` leaves, including named view entries | Fixed-arity repeated views, dimensions/angle units, raw-level filters and byte budgets |
| Original mapping | `output_frame.id`, `.origin_xyz_um`, `.voxel_um`, `pitch`, `memory_gib` | Explicit output-frame flags; `pitch` remains WORLD units |
| Matrix/patch mapping | Existing validator's selected-mode leaves | `pitch_um`, geometry/process-tree/disk budgets, raster/cache workers, domain/chart/backend/centerline controls; reject wrong-mode fields |
| Direct TIFXYZ | `schema`, `levels`, `roi.lo_zyx`, `.hi_zyx`, `pitch_um`, `tolerance_um`, `memory_gib`, `scratch_gib`, `reserve_disk_gib`, `max_pixels`, `output_frame`, `audit_backend`, `coverage_policy` | Fixed schema; levels raw; ROI ZYX; lengths µm; memory GiB; coverage defaults unchanged |
| Explicit TIFXYZ frame | `id`, `origin_xyz_um`, `voxel_um`, `shape_zyx`, `source_identity_sha256` | Full frame/pin required together; selected-source mode stays available |
| AFV/Lasagna declarations | Accepted source-identity geometry and manifest leaves in current validators | Per-source geometry, repeated typed channel/crop declarations, encoding and validity; hash-bound arrays remain data |

As each family is implemented, its exhaustive leaf-to-flag table and maximal
parity fixture complete this inventory. Newly accepted validator fields must
receive either a typed flag or an explicit scientific-artifact disposition,
never a hidden requirement for another handwritten JSON file.

### Prepared build/resume/verify mappings

| Native setting | Flag | Type / units |
|---|---|---|
| `frame`, `voxel_um` | `--field-frame`, `--field-voxel-um` | frame string, µm |
| `grid.origin_xyz`, `.spacing_xyz`, `.shape_zyx` | `--grid-origin-xyz`, `--grid-spacing-xyz`, `--grid-shape-zyx` | 3 WORLD coordinates, 3 WORLD spacings, 3 ZYX integer node counts |
| `sheet_step`, `calibration` | `--sheet-step`, `--calibration` | raw field units/sheet, provenance string |
| `observations.target/weights/observed` | `--target/--weights/--observed` and each `--NAME-sha256` | local array path and lowercase SHA256 |
| `fixed`, `constraints`, `order` | `--fixed/--constraints/--order` and each `--NAME-sha256` | local NPY/NPZ data path and SHA256 |
| `solver.*` | `--device`, `--dtype`, `--rtol`, `--maxiter`, `--preconditioner` | existing solver types/defaults; preconditioner stays Jacobi |
| `execution.*` | `--chunk`, `--halo`, `--sweeps`, `--cache-bytes`, `--read-workers`, `--constraint-cache-bytes`, `--prepared-cache-bytes`, `--max-bytes`, `--reserve-bytes` | integer node counts, counts or bytes as named |
| `gpu.*` | `--gpu-index`, `--gpu-uuid`, `--gpu-lock-path`, `--gpu-min-free-bytes`, `--gpu-deadline` | index, explicit UUID, local lock path, bytes, epoch seconds |
| `source_identity` | repeated `--source-identity KEY JSON_VALUE` | strict JSON values; duplicate keys reject |

The fixed schema is `winding-field-run-v1`. Native `--config` remains optional;
`--cli-config` may supply every option using the flag's underscore destination.
Explicit paths use the caller's CWD, file paths use that file's directory.
CPU/GPU lease checks and all SHA checks remain mandatory.

CLI requests are saved as immutable, output-bound `OUTPUT.request-SHA256.json`
siblings, not inside the solver/service-owned directories. Preserve these files
alongside the run. Plain `resume --output OUTPUT` and `verify --output OUTPUT`
read the recipe pinned by RUN; verify never creates a recipe. Missing/changed
saved recipes and changed science reject before writes. Existing cache/reader/
resource-only continuation remains allowed; its effective recipe is another
immutable sibling. Historical source-file hashes stay in `request_sources`;
the original handwritten file is not needed to resume a saved CLI request.

### Prepared mesh/surfaces/render mappings

Field descriptors use `--field-kind`, `--field-path`/`--field-url`,
`--field-sha256`, `--field-complete-sha256`, `--field-support-policy`,
`--field-mask` and `--field-mask-sha256`. NPY sources accept masks; only chunk/
remote sources accept their existing stored/observed policies. Imported mesh
data uses `--mesh PATH --mesh-sha256 SHA256`, exclusive with a field source.

Geometry uses `--field-frame`, `--field-voxel-um`, `--field-origin-xyz`,
`--field-spacing-xyz`, `--field-shape-zyx`, `--sheet-step`, `--calibration`.
Origins/spacings are WORLD XYZ triples, shape is integer ZYX. `--levels` is
always raw phi, never winding W; W−60 in the checked Paris4 setup is
`--levels -71.54987931251526 --sheet-step 1.192497988541921`, not `--levels -60`.
The 9.6 µm frame voxel calibration is not multiplied into those WORLD vectors.

Execution leaves map to `--tile` (cells per axis, ≤256), `--level-batch`,
`--cache-bytes`, `--reserve-bytes`, `--mesh-bytes`. Rendering uses
`--render-width`, `--render-height`, `--render-triangle-batch`,
`--render-reserve-bytes`, `--render-levels`, and repeated
`--render-view NAME AZIMUTH ELEVATION` (degrees). View flags replace the whole
file view list. Combined surfaces retain the existing execution reserve policy;
standalone render uses its own render reserve. Standalone `render --mesh BUNDLE`
needs no field geometry or full surface recipe.

`--parallel-levels` uses the same generated sibling recipe and existing digest-
bound per-level worker protocol. Plain `--resume --output OUTPUT` locates a
generated serial recipe through `REQUEST.json` or a parallel recipe through
`LEVELS.json`. Preserve both the output and its sibling request. Parallel
collections keep full recipe bindings. Native legacy recipe invocations remain
supported without translation.
