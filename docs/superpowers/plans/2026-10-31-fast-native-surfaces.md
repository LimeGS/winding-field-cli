# Fast Native Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Native meshes, global flattened CT images and optional mesh-derived TIFXYZ, with configurable coverage/time, durable cancellation and measured performance.

**Architecture:** Keep the existing atlas and certified-direct branches. A native orchestration stage processes levels independently through mesh, local charts, connected UV assembly and a shared tiled XYZ raster. SQLite transactions and atomic payloads checkpoint each bounded unit; a supervised process makes slow native calls and network I/O cancellable without corrupting committed work.

**Tech Stack:** Existing NumPy/SciPy/Pillow, SQLite, standard-library subprocess/zlib, optional existing tifffile. CPU first; no new solver/runtime dependency.

**Spec:** `docs/superpowers/specs/2026-10-31-fast-native-surfaces-design.md`

## Global Constraints

- User approved implementation inline, in the current checkout; no additional approval loop for execution method, no commits or worktree.
- Preserve existing modifications, `_core/`, `provenance/core.json`, frozen remote runs and source data.
- <=1800s goal from computed field to full native mesh and requested-coverage CT image for one whole W; reused-mesh/UV timings are separate.
- `coverage_target` in `(0,1]`, default1; `surface_time_limit_s` positive, default1800, per W. Report cumulative active elapsed across resumes; paused wall time is excluded.
- Target coverage is mapped source mesh area, not certified field/TIFXYZ coverage. All source faces stay accounted; no silent largest-component selection.
- Initial physical pitch is minimum field spacing; real175A17.28um, CT L0 at8.64um. Reuse physical coordinate transforms, never infer them from filenames.
- Pilot <=3GiB aggregate RSS, <=16GiB scratch and12.25GiB disk reserve. Preserve larger existing source/product files; admit generated products and cache separately, honestly.
- Ctrl+C/STOP => PAUSED with committed work retained. Timeout => PAUSED/TIME_LIMIT if pipeline unfinished; complete usable partial renders retain measured coverage and unmet target flags.
- A resume does not silently reset cumulative time. An explicit additional time allowance, recorded separately from original config, may extend the budget; it cannot make an over30min cumulative run satisfy the original goal.
- No per-output-node field roots, whole-field coverage audit or certified-direct verifier in the native path.

## Review Focus

1. Cancellation during numeric/native work or CT fetch: owned worker can be terminated, SQLite rolls back current transaction, previous payload hashes survive.
2. Seam closure/non-disk surfaces: retain explicit cuts; never force incompatible shared vertices together or merge disconnected components.
3. Resume after source/payload corruption: refuse mutation rather than overwriting committed geometry or images.
4. Mapped geometry with missing/ambiguous CT: keep geometry and CT validity distinct; TIFXYZ must not bridge holes or claim field certification.
5. Multiple levels and budget exhaustion: per-level checkpoints/products/timing, no hidden timer reset or small-ROI performance claim.

## File responsibilities and interfaces

- `native_store.py`: `Paused(reason)`, `Journal(root, identity, stop=None, seconds=1800, reserve_bytes=0)`; owns SQLite/file ledger, checks, status and timing. `save_arrays(name, **arrays)`, `load_arrays(name)`, `register(name)`, `check()`, `close()`; validates immutable committed files on resume.
- `native_geometry.py`: `flatten_mesh(reader, journal, chart_faces, until=None)` creates linked charts/edge records; `assemble(journal, coverage_target)` places connected charts using shared original edges and stores UVs/component bounds. All pieces retain source IDs and area.
- `native_render.py`: `render(journal, volume, pitch_um, export_tifxyz=False)` streams shared XYZ/CT tiles and PNG/mask/overview; optional writer adapter reuses `tifxyz_store.write_segment` without direct certificates.
- `native_pipeline.py`: `work(run, config, until='render', stop=None)` handles the per-level pipeline; `run_native(run, config, stop=None, until='render')` supervises the worker and reports terminal/paused states. Module worker entry is private. A reusable `render_mesh(reader, output, volume, config, stop=None)` path accepts genuine mesh bundles for precomputed-input tests.
- `meshes.py`: factor `select_levels(run, config, source, stop=None)` from existing selection without changing legacy results.
- `workflow.py`, `cli.py`, `config.py`: native defaults for new CLI runs, unchanged fallback for old dictionaries, options and stage aliases. The parent records one native aggregate stage; level receipts expose mesh/flatten/assemble/render checkpoints to avoid duplicating global stages for every level.
- `flatten.py`: replace all-pairs candidate search with bounded cKDTree broad phase; preserve exact overlap predicate and caller behavior.
- `tests/test_native_*.py`: geometry, rendering, persistence/cancel, CLI and budgets.
- `tools/bench_native_surfaces.py`: bounded real benchmark, explicit historical adapter and recorded cold/warm scope.

### Task 1: Geometry primitives and safe checkpoint ledger

**Files:** `flatten.py`, new `native_store.py`, `native_geometry.py`, `tests/test_native_geometry.py`, `tests/test_native_store.py`.

**Interfaces:** consumes immutable `MeshReader.db/manifest`; produces persistent charts with original face/vertex IDs, source area, shared boundary edges and global UVs.

- [ ] RED: overlap tests for overlapping slivers, contacts-only, far triangles and long triangles; original curved-strip tests remain exact.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_native_geometry.py tests/test_meshes.py` and observe failing new assertions/imports.
- [ ] Implement bounded overlap broad phase, Journal atomic payload + committed hash ledger, per-batch SQLite transactions and source identity binding.
- [ ] RED: interrupted chart insertion retains previous hashes; an uncommitted file can be regenerated; changed committed payload or input identity is refused.
- [ ] Implement local charts and shared-edge indexing, then deterministic connected placement. Fit rigid poses at parent seams and harmonic displacement only within bounded charts; leave incompatible cycle edges as explicit cuts and record invalid charts. Never load every chart's vertices at once.
- [ ] RED: two planar/curved neighbouring charts share exact saved seam coordinates; separate components yield separate canvases; faces are neither lost nor duplicated; unsafe welding is reported rather than accepted.
- [ ] Run focused tests to GREEN, inspect coverage accounting and cancellation at committed boundaries.

### Task 2: Shared tiled raster, CT and optional TIFXYZ

**Files:** new `native_render.py`, `tests/test_native_render.py`; reuse `preview.sample_ct`, `tifxyz_store.write_segment`, `util` and `io.Volume`.

**Interfaces:** consumes Journal placed charts/component bounds; produces committed XYZ/geometry-mask/CT tiles, global PNG/overview/mask, optional TIFXYZ and per-component receipts. Tile size256; bounded vectorized triangle/pixel batches, not Python per-pixel loops.

- [ ] RED: an analytic plane over CT `x+2*y+3*z` rasterizes exact physical coordinates and trilinear values; different origins/anisotropic CT transform correctly.
- [ ] RED: multi-chart overlap with different XYZ is sticky/transparent; roundoff sharing is consistent; missing CT does not delete valid geometry.
- [ ] Implement bounded triangle bbox expansion/barycentric batches and per-tile conflict resolution; persist each completed tile before advancing.
- [ ] Implement streaming PNG rows and a bounded overview with a common intensity window. Resume skips committed raster/CT tiles and rejects corruption.
- [ ] Implement the small tiled reader adapter for mesh-derived TIFXYZ; explicit producer metadata and local cell checks. Unsupported TIFXYZ cases report a reason without discarding valid OBJ/PNG.
- [ ] RED/GREEN: TIFF arrays/frame/scale match the shared XYZ, holes do not acquire bridges, partial publication never uses a certified-direct status; canceled publication exposes no truncated final file.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_native_render.py tests/test_sampling.py tests/test_tifxyz_store.py`.

### Task 3: Per-level orchestration and interruptible worker

**Files:** new `native_pipeline.py`, `tests/test_native_runtime.py`; `meshes.py`.

**Interfaces:** `work` produces `SURFACES.json` aggregate and per-level checkpoints; `run_native` returns `COMPLETE`, `PARTIAL`, `PAUSED` or failure with reason, elapsed/peak RSS. Mesh and CT sources remain immutable.

- [ ] RED: two selected levels publish separately; stop during first level preserves it and does not touch later levels; retry does not redo committed charts/tiles.
- [ ] Factor level discovery without altering legacy output. Reuse `_core.extract_mesh` for one level at a time; native orchestration uses actual per-level mesh bundles, never fabricated field/run receipts.
- [ ] Start the entire heavy path in an owned subprocess. Parent watches STOP/signals, RSS and persisted progress; request cooperative stop then terminate only its process group after a bounded grace period.
- [ ] Persist elapsed active time and original request. Resume uses remaining cumulative budget; explicit added seconds are separate recorded allowance, never a configuration/geometry rewrite.
- [ ] RED/GREEN: force-kill during a worker operation, reopen/checkpoint recovery, committed hashes unchanged; source mutation rejected; tiny timeout exits promptly and saves progress; zero remaining budget does no work.
- [ ] Test terminal publication/read-only resume, pending-file recovery and original30min performance flag after an explicit budget extension.

### Task 4: Public CLI, compatibility and documentation

**Files:** `cli.py`, `config.py`, `workflow.py`, `README.md`, `docs/FORMATS.md`, `THIRD_PARTY.md`, `tests/test_native_cli.py`.

**Interfaces:** new CLI default `--surface-output native`; `--coverage-target` fraction, `--surface-time-limit` seconds, `--export-tifxyz`; `resume --extra-surface-seconds` explicitly extends remaining budget. Existing `atlas`/`tifxyz` and config dictionaries without selector retain their contracts. Stage commands/`--until mesh|flatten|assemble|render` route to the per-level native stage when selected.

- [ ] RED: invalid target/time/bool values fail before IO; new default/native stage path; old dictionaries and direct branch unchanged.
- [ ] Wire native stage without rerunning producer stages or requiring the certified-export settings. On cancellation record PAUSED, do not register incomplete stages as complete.
- [ ] Snapshot only immutable committed products; completed native resume verifies source and products without rewriting them. Expose per-level current phase, completed counts, coverage and elapsed in status.
- [ ] RED/GREEN: installed CLI from another cwd, field->native, cancellation through actual SIGINT/STOP and resumed completion, multi-level output, optional TIFXYZ unavailable preflight.
- [ ] Document exact options/output/coverage/time semantics, real commands, and mesh-derived versus field-certified guarantees. Retain legacy invocation tests.

### Task 5: Acceptance, review and real performance

**Files:** `tools/bench_native_surfaces.py`, `docs/verification/fast-native-surfaces.md`, evidence receipts.

- [ ] Run focused then full local suite with single-thread BLAS and `git diff --check`; confirm `_core/` and pinned provenance unchanged.
- [ ] Freeze the implemented runtime before remote execution. Reuse verified historical source bindings; do not change the running direct verifier or its files.
- [ ] Run Linux synthetic end-to-end/cancel-resume and real VC3D TIFXYZ roundtrip where TIFXYZ is published.
- [ ] Benchmark real representative blocks first, with phase timings and complete source accounting. Resume this work rather than repeatedly starting over.
- [ ] Run the admitted whole-W acceptance attempt with explicit time/coverage and cache condition, preserving checkpoints if30min is exhausted. If extraction/core/resource limits block the target, report measured evidence and the constraint instead of calling a small test success.
- [ ] Request independent whole-change review, fix findings with regression tests, and rerun affected checks.
- [ ] Summarize implemented behavior, tests and measured real result; distinguish feature correctness from the not-yet-met performance target if applicable.

### Task 5a: Profile-driven batched mesh storage

**Files:** new `src/winding_field/native_mesh.py`, `tests/test_native_mesh.py`;
switch native orchestration/benchmark to that driver, keep `_core/` unchanged.

- [ ] RED: byte-identical OBJ vertices/faces and equal component/edge counts versus
  original extraction on multi-tile/multi-level analytic fields; area agrees to1e-12.
- [ ] RED: cancel/resume keeps geometry, corrupted immutable rows fail, corrupted
  derived topology is rebuilt; vertex-only contact stays disconnected.
- [ ] RED: large connected tile assembly uses a bounded number of SELECT statements,
  rather than per-face SQLite union/find queries.
- [ ] Implement `MeshDB(core.MeshDB)` with batched vertex/face insertion, vectorized
  areas/local edge-connected groups and batched seam/component merges. Use the same
  schema, vertex numbering and OBJ export; helper tables stay in the owned DB.
- [ ] Implement canonical JSON row hashing in batches with a safe fallback for
  delimiter-containing strings, producing the original digest byte-for-byte.
- [ ] Implement `extract_mesh(source, output, levels, *, tile, resume, stop,
  reserve_bytes)` reusing `core._tile_mesh` without monkeypatching the core.
  Bind the storage implementation separately in identity/provenance.
- [ ] Run parity/integrity tests, profile the retained dense real tile and freeze
  a new runtime for complete acceptance. Never overwrite runtimes01/02.

## Plan self-review

## Overnight continuation: reproducible global control

The user's continued instruction is to work until large coherent flattened exports
are practical. Measured local-chart repairs improved coverage but left1295grid
patches in W-50. A source-preserving global control now qualifies95.860%of the
whole input mesh area and has99.212%of retained TIF cells in one patch. This is
PARTIAL native geometry, not physical acceptance or the whole-field W-4 speed goal.

Builder ruling: consolidate the demonstrated global recipe as an explicit
experimental command, keeping the current public native default intact until
this new path has reproducible end-to-end evidence. Input is a genuine indexed
single-level mesh with explicit component selection and original CT/frame; all
other components and excluded faces stay in the area denominator and ledger.

- [ ] Add `native_global.py` with bounded source-preserving cut, positive mean-value
  initialization (bounded ILU+GMRES), existing direct ARAP20, original native face
  quality gates, complete bounded overlap participant removal, and source-area/
  connectivity accounting. Share the existing overlap predicate rather than change it.
- [ ] Add a reproducible experimental tool that executes fresh source→cut→MVC→ARAP→
  qualification→native renderer phases, using separate owned workers to release
  sparse-factor memory. Bind source/runtime/config and intermediate hashes;
  retain checkpoints and cumulative active time, STOP, RSS3GiB, RAM reserve2GiB,
  scratch16GiB and disk reserve12.25GiB. No downloads/installations of dependencies.
- [ ] Register qualified faces in bounded internal chunks with UVs from the ONE
  global map, exact original XYZ/face IDs and cut ancestry. Reuse native_render
  safe-cell masks and writer. Record all exclusions and PARTIAL status; do not
  silently select the largest source component, promote field certification, or
  claim source-area coverage for the retained raster.
- [ ] Verify mathematical/identity/boundary/cancel controls RED→GREEN, independent
  review, full applicable suite and a fresh W-50 execution without precomputed UVs.
  Run actual VC3D readback and measure grid connectivity. Keep image inspection
  and physical-sheet interpretation separate from software/format acceptance.

Coverage: cancellation/kill, source and payload integrity, per-level budgets, coverage denominator,
CT/TIFXYZ mask differences, global seams, complete native meshes, CLI legacy behavior,
historical source binding and real timing each have an owning task. No commits or
changes to frozen runtimes are part of execution. Exact internal receipt keys may
follow existing conventions but the stated interfaces and user-visible semantics
must remain consistent between tasks.
