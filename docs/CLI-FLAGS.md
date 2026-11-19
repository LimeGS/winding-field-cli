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

### YAML configuration

Every command accepts `--cli-config options.yaml` or `options.yml` in addition
to JSON. The `--config` alias accepts YAML wherever it aliases `--cli-config`;
native legacy recipe options retain their own formats and meanings.
For the complete commented `run`/`plan` option reference, copy
[`examples/run-plan.template.yaml`](../examples/run-plan.template.yaml).
It includes typed examples, choices/ranges/units and conditional-mode notes;
optional or incompatible modes are commented rather than activated together.

```yaml
field_kind: npy
field_path: phi.npy
field_sha256: "REPLACE_WITH_THE_FILE_SHA256"
field_frame: my-scroll/world
field_voxel_um: 9.6
field_origin_xyz: [0, 0, 0]
field_spacing_xyz: [32, 32, 32]
field_shape_zyx: [3, 4, 5]
sheet_step: 1.0
levels: [1.5]
output: mesh
```

Run with `winding-field prepared mesh --cli-config options.yaml`. Use the
actual geometry and file hash. Explicit flags still override the file; relative
file paths still use its directory. Booleans, nulls, numbers and nested lists use
the same typed validation as JSON. Quote strings that resemble YAML booleans,
dates or numbers. Duplicate keys, aliases/merge keys, custom object tags,
nonfinite numbers and values outside JSON's data types are rejected.
Generated requests, checkpoints and receipts remain JSON.

## Coverage inventory

Each row is a required acceptance family. Settings-only content is not exempt
merely because its current argument is called metadata/case/inputs/manifest.
The corresponding family tests must compare maximal file and flag requests.

| Public family | Existing authority | Required settings/flag groups |
|---|---|---|
| `run`, `plan` | `cli.py`, `config.py` | Every existing option; nested original mapping and direct-TIFXYZ requests; imported input geometry and manifest settings |
| `resume`, `prepare`, `fibres`, `field`, `mesh`, `flatten`, `assemble`, `render`, `status` | `cli.py`, `workflow.py` | Run path; existing offline, sweep and explicit extra-time controls only; config mode must not make frozen scientific settings mutable |
| `field-converge` | `field_convergence.py` | Prepared source or automatic generation, fresh output/resume, total sweep limit, separate global tolerance and inherited field overrides; no legacy migration |
| `prepared build/resume/verify` | `field_workflow.load_run_config` | Frame/voxel/grid/sheet/calibration; target, weights, observed, fixed, constraints and order artifacts+SHA; solver; execution; GPU lease; source identity |
| `prepared mesh/surfaces` | `surface_workflow.load_surface_config` | Field or mesh descriptor; field geometry; raw levels; extraction execution; render settings; parallel levels; resume |
| `prepared render` | `surface_workflow._render_settings` | Mesh path, output, dimensions, views, triangle batch, levels, reserve |
| `export-sheets` | `flattening.original_pipeline.settings`, `flattening.patches.settings`, `flattening.matrix.settings` | Run/field/mesh exclusivity, levels, CT and selected level, output frame, mode-specific pitch, budgets, cache/workers, backend/native settings, centerline/reference controls, TIFF and resume |
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

`field-converge` accepts JSON/YAML `--cli-config` (`--config` alias), with keys
`source`, `output`, `resume`, `max_sweeps`, `global_rtol` and the field overrides
`field_chunk`, `field_spacing_um`, `rtol`, `maxiter`, `fibre_weight`, `device`,
`threads`, `reserve_bytes`, `bounded_bytes`, `prepared_cache_bytes`,
`constraint_cache_bytes`. Explicit flags override file defaults. On resume every
provided value must equal the saved setting; omission reuses it. See
[automatic convergence](FIELD-CONVERGENCE.md) for units, defaults and limits.

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

The family leaf-to-flag tables and maximal parity fixtures cover this inventory.
Newly accepted validator fields must
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
`--no-fixed`, `--no-constraints`, `--no-order` explicitly remove optional
prepared artifacts inherited from a native recipe. The corresponding CLI-config
values may be `null`; an artifact pin cannot be supplied alongside its clear flag.
`--max-bytes auto` clears an inherited optional prepared-memory limit (`null` in
config mode); it does not disable disk reserves or other resource admission.

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

### Export, range and UVW mappings

`export-sheets` accepts either a saved `--run`, an immutable `--mesh-bundle`,
or the field descriptor/geometry flags above (optionally merged over
`--field-config`). Direct field export also requires raw `--levels`. Mesh/run
bindings cannot be silently replaced with field geometry flags.

`--flattening-config` is optional. Its leaves map directly as follows; the
existing selected-mode validator rejects flags not supported by that mode.

| Settings | Flags / units |
|---|---|
| `output_frame.id`, `.origin_xyz_um`, `.voxel_um` | `--output-frame-id`, `--output-origin-xyz-um X Y Z`, `--output-voxel-um`; µm |
| Original `pitch`, `memory_gib` | `--pitch NUMBER\|auto` (WORLD units), `--memory-gib` (GiB) |
| Matrix/patch sampling and budgets | `--pitch-um`, `--tile-size`, `--memory-gib`, `--process-tree-gib`, `--disk-reserve-gib`; µm, pixels, GiB |
| Cache and readers | `--cache-dir`, `--cache-mib`, `--cache-gib`, `--ct-workers`, `--raster-workers` |
| Patch domains | `--max-patch-faces`, `--max-charts`; historical patches also `--max-depth`, `--timeout-seconds`, `--workers` |
| Whole-sheet reference | repeated `--centerline-point X Y Z`, `--reference-z`; WORLD coordinates |
| Native backend | `--app-run`, `--native-recipe`, historical `--native-cpus CPU...` / `--auto-native-cpus` |
| Historical reuse binding | `--reuse-export`, `--reuse-identity-sha256`; both required together; clear both with `--no-reuse-export --no-reuse-identity` |

`--flattening-solver lscm-slim|lscm|slim --scope patches|sheet` (default `lscm-slim` × `sheet` on Linux x86_64) remains distinct from
historical `--flattening-method original|patches`. Native requirements, resource
guards and Linux admission do not change. `--pitch` is not `--pitch-um`.
CT still uses `--source`, `--voxel-um`, `--level` (pyramid index); no field/CT
unit is inferred from the other. Existing `--run` uses its saved CT binding,
while output-frame flags may override its original-mapping defaults.

`extract-range` accepts optional native `--config` plus field/geometry flags,
repeated `--umbilicus-point X Y Z`, `--handedness -1|1`, `--w-ref`,
`--theta-ref`, `--samples-per-w`, `--max-edge-world NUMBER|auto`.
`--w-range A B` preserves exact decimal text on the established W lattice;
it is not the scalar mesh `--levels` option. `join-ranges` still takes actual
range artifacts, not a recipe.

`export-uvw` accepts field/geometry flags in place of `--field-config`. Its
reference-sheet artifact, CT binding, levels, canvas samples, workers and
cache MiB retain their established meanings and validation. The same settings
are available through `--cli-config`; all known file paths use that file's
directory and explicit flag paths use the current directory.

### Direct TIFXYZ and nested run/plan requests

Direct `tifxyz FIELD_RUN --output OUTPUT` uses optional native `--config` or:

- `--levels PHI...` (raw field values).
- `--roi-lo-zyx Z Y X --roi-hi-zyx Z Y X` (half-open field node ROI);
  `--whole-field` clears a file ROI.
- `--pitch-um NUMBER|auto`, `--tolerance-um NUMBER|auto`.
- `--memory-gib`, `--scratch-gib`, `--reserve-disk-gib` (zero reserve remains
  explicit zero, not an omitted/default value), `--max-pixels`.
- `--audit-backend numpy|cpu|cuda`, `--coverage-policy strict|allow_partial`.
- `--output-frame selected-source`, or the complete explicit frame:
  `--output-frame-id`, `--output-origin-xyz-um X Y Z`, `--output-voxel-um`,
  `--output-shape-zyx Z Y X`, `--output-source-identity-sha256`.

New requests retain the fixed `winding-field-tifxyz-v1` schema and strict coverage
default. Selected-source mode is not an invented frame; it uses the existing
producer's validated source binding. Linux worker admission is unchanged.

`run`/`plan --surface-output tifxyz` share raw `--levels` and output-frame flags.
Their direct-export-only settings use `--tifxyz-` prefixes, for example
`--tifxyz-pitch-um`, `--tifxyz-memory-gib`, `--tifxyz-roi-lo-zyx`,
`--tifxyz-reserve-disk-gib`. These settings cannot leak into other branches.
`--tifxyz-config` remains optional and merges before explicit leaves.
The producer's existing `--levels` must still agree with a file's direct-export
levels. Use `--tifxyz-levels` to explicitly replace that nested list (and, when
also supplied, give the same producer `--levels`). File-free requests may use
producer `--levels` directly. `--no-export-tifxyz` and `--no-previews` preserve
explicit false overrides rather than falling back to true file defaults.

`run`/`plan --flattening-method original` accept the output-frame flags,
`--pitch NUMBER|auto` in WORLD units, and `--original-memory-gib`. They merge
over optional `--flattening-config`; no frame is invented if neither supplies one.
These output settings remain separate from CT `--roi`, `--voxel-um`, `--level`
and the observation/field resource budgets.

### AFV/Lasagna input declarations

`run` and `plan` may keep the existing native `--fibres-geometry`,
`--lasagna-geometry`, and `--lasagna-input` files, or supply their settings inline:

| Setting | Flag |
|---|---|
| AFV physical frame | `--fibres-source`, `--fibres-origin-xyz-um X Y Z`, `--fibres-spacing-xyz-um X Y Z` |
| Explicit Lasagna physical frame | `--lasagna-source`, `--lasagna-origin-xyz-um X Y Z`, `--lasagna-spacing-xyz-um X Y Z` |
| Base dimensions / encoding | `--lasagna-base-shape-zyx Z Y X`, `--lasagna-source-to-base`, `--lasagna-grad-mag-encode-scale`, `--lasagna-grad-mag-factor` |
| Scalar channel declarations | repeated `--lasagna-channel GROUP CHANNEL ZARR SCALEDOWN` (scaledown is an exponent, including zero) |
| Optional per-channel metadata pins | repeated `--lasagna-channel-metadata-sha256 GROUP SHA256` |
| Coverage crops | repeated `--lasagna-crop X Y Z WIDTH HEIGHT DEPTH`; `--no-lasagna-crops` explicitly clears the list |
| Producer receipt | `--lasagna-provenance PATH` (actual producer data, not a handwritten settings recipe) |
| Transport policy | `--lasagna-required-transport vc_reader.DiskCache` |
| Frozen metadata inventory | repeated `--lasagna-metadata-pin LOCATION SHA256`; `--clear-lasagna-metadata-pins` clears that map |
| Declared CT identity | `--lasagna-ct-source CT_DATA`, optional `--lasagna-ct-level`, `--lasagna-ct-voxel-um`; identity is read from the actual CT artifact |
| Embedded physical-frame variant | the declared CT source plus `--lasagna-embedded-origin-xyz-um X Y Z` and `--lasagna-embedded-spacing-xyz-um X Y Z` |

The manifest schema remains version 2. CT identity documents are scientific
provenance: the flags derive those complete identities from actual dataset
metadata rather than requiring users to copy backend/library hashes manually.
Prediction arrays and any existing producer receipts remain read-only inputs.

Omitted physical geometry still means base-grid binding; explicit/embedded
physical geometry still selects calibrated interpolation. Neither route rescales
density because channel grids are downsampled. All source/channel metadata pins,
support/crop checks, input ledgers and final revalidation remain active. Mixed
file+flag inputs preserve raw recipe hashes in `input_recipe_sources`; changing
one during a run or before continuation fails rather than relabeling provenance.
HTTP/S3 manifests may also be combined with typed overrides; relative channel and
provenance locations retain the remote URL base and the original raw manifest pin.
Online continuation revalidates that pin. Offline continuation reuses the frozen
declaration and still checks cached channel metadata and consumed-byte ledgers.

All leaves are also available through `--cli-config`. Channel rows use typed
`[group, channel, path, integer_scaledown]` arrays; the path element uses the
config directory, not the worker CWD. Executable paths such as `lasagna_python`
retain virtual-environment symlinks instead of resolving to a different Python.

### Public tool settings and artifact boundaries

Every tool, including all `constant-sheet` and `sheet-atlas` subcommands,
accepts `--cli-config` with typed underscore option names. `--config` aliases it
only when that tool has no native recipe named `--config`.

| Tool / native object | Flag mapping |
|---|---|
| Surface export/probe/atlas field metadata | `--field-frame`, `--field-voxel-um`, `--field-origin-xyz`, `--field-spacing-xyz`, `--field-shape-zyx`, `--field-sha256`, `--sheet-step`, `--calibration`; WORLD XYZ / ZYX, µm per WORLD unit |
| Support metadata | `--support`, `--support-sha256`, `--support-array-sha256`; probe/atlas also `--core-ownership` |
| Surface export/atlas CT metadata | `--ct-frame`, `--ct-voxel-um`, `--ct-origin-xyz`, `--ct-spacing-xyz`, `--ct-shape-zyx`, `--ct-display-window LOW HIGH` |
| Probe dataset view | `--dataset-frame`, `--dataset-voxel-um`, `--ct-source`, `--ct-sha256`, CT geometry above; `--prior-source`, `--prior-sha256`, `--prior-frame`, `--prior-origin-xyz`, `--prior-spacing-xyz` |
| Probe/atlas render context | `--seed-margin`, `--ct-padding-samples` (sample counts), `--render-pitch-world`, `--render-depth-offsets-world`, `--render-mesh-step`, `--ct-window`; display-window aliases must agree |
| Atlas frozen inputs | `--seed`, `--seed-sha256`, `--ct`, `--ct-sha256`, `--ct-metadata`, `--ct-metadata-sha256`, `--inputs-field-sha256`, `--inputs-metadata-sha256` |
| CT visual case settings | `--visualization-schema`, `--archive-manifest`, `--archive-manifest-sha256`, `--reference-key` / `--no-reference-key`, `--reference-label`, `--requires-support` / `--no-requires-support`, `--case-step`, `--dataset-frame`, `--dataset-voxel-um` |
| Patch visual geometry | Field geometry above; legacy WORLD-only metadata remains valid without invented µm calibration; existing `--step`, `--margin` retain their meanings |
| Constant-sheet centerline | Repeated `--centerline-point X Y Z` replaces optional `--centerline` JSON; strictly increasing WORLD Z, for prepare and validate |
| Constant-sheet render | `--ct-source`, `--ct-level`, `--ct-voxel-um`, all historical flattening settings above, `--prior-identity`, `--prior-ledger`, `--prior-ledger-sha256` |

Original native metadata/case/inputs recipes may be combined with explicit
settings. Raw file pins remain separately recorded in `request_sources`; a
changed file is never silently repinned. Generated effective metadata lives in
the tool's new output/work directory, never overwrites the user's input.

The following document content is **scientific data/provenance**, not an
additional settings recipe to author:

- Field receipts' solver/pilot results, byte/array hashes, source ownership,
  saved grids, seeds, CT crops, support arrays and original observation history.
- CT visual archive manifests, receipts, NPY/PNG planes and their hashes; a case's
  unrelated benchmark results are not consumed as plotting controls.
- Geometry-check metadata, registration reports, pointmaps and source receipts
  already SHA-bound to those artifacts; the external gate/policy files and their
  pins remain explicit inputs, not settings synthesized by this CLI.
- Atlas/query sheets, validated maps, native checkpoints, producer/prior ledgers,
  backend executables and their identities.
- Legacy full dataset AFV, prediction, orientation, constraints, execution and
  solver sections: probe consumes only frame/units, CT/prior sources and surface
  render context. Inline requests expose that consumed view; legacy full files
  still pass the original full dataset validator.

`run-watch` config supports its literal `command` string array. Explicit child
arguments (with or without `--`) stay opaque, including child `--config` options;
no shell interpolation or JSON-to-shell translation is introduced.
