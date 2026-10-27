# Audit fixes and independent field spacing implementation plan

> **For agentic workers:** Use executing-plans to implement this plan task-by-task.

**Goal:** Resolve all ten findings in the approved business-logic audit.

**Architecture:** Keep the numerical core unchanged. Add a physical-grid adapter
between fine observations and the solver; source-anchored inference tiles and
explicit validity keep storage layout from defining observation semantics.

**Tech Stack:** Python 3.11+, NumPy/SciPy, Torch, SQLite, existing portable Zarr v2.

**Spec:** The ten findings and “Fix all findings” approval in the conversation;
reproductions in `.runtime/audit/PLAN.md`, `probes.py`, `resources.py`, `neural_tiles.py`.

## Global constraints and interface decisions

- Preserve `_core`, `_vendor`, original checkouts, historical runs and v0.1.0.
- Arrays are ZYX; coordinates are XYZ micrometres at voxel centres.
- `--field-spacing-um P` is an exact isotropic pitch, independent of `--stride`
  (CT observation sampling). Default field geometry remains the observation grid.
- Field origin is the first ROI observation centre. Shape is
  `floor((observation_shape-1)*observation_spacing/P)+1`; never extend the ROI.
- Sample the already smoothed fine prior with trilinear interpolation. Orient
  unsigned normals before interpolation; require every contributing support node.
  Read only the touched tiles, never a dense native envelope for coarse sampling.
- SD2 uses fixed 32³ owned cores on the selected-source sampling lattice, 40-voxel
  halo and its pinned z-score recipe. Cache at most eight cropped canonical cores
  in memory. `--chunk` only changes persisted map storage. No new dependency.
- `--invalid-value V` explicitly marks acquisition fill, otherwise zero remains
  a valid measurement. `observations(..., validity=bool_zyx)` zeros invalid
  presence and excludes the complete 40-voxel filter footprint from support.
- Exports return `None` when stopped; workflow only records a completed stage
  after its export completes. Resume verifies retained chunks/record identity.
- Configuration checks precede source/catalog I/O; geometric checks precede
  CT reads and inference. Plan displays both geometries and both disk budgets.

## Review focus

- Nonintegral pitches and anisotropic source coordinates must not drift or leave ROI.
- A coarse solve spanning many observation tiles must retain physical fibre constraints.
- A stopped export must retain progress, reject altered inputs and avoid false completion.
- Offline mirrors with cached conflicting metadata must still be rejected.
- Full cache filesystems must not receive payload writes despite a generous cache limit.

## Task 1: Input, admission and concurrent downloads (findings 1, 4, 5, 8, 9)

Files: `io.py`, `models.py`, `workflow.py`, `cli.py`, new `config.py`;
tests: `test_io.py`, `test_cli.py`, new `test_audit_io.py`.

Interfaces: `validate_config(config)`; `resolve_source(..., cache=None, offline=False)`;
`Volume(..., reserve_bytes=0)`; plan adds observation/field geometry and cache free bytes.

- [x] Add regressions: bounded gzip decoding, offline mirror and catalog, early invalid
  options, distinct cache-disk admission and write rejection, two simultaneous downloads.
- [x] Run regressions and observe their specific failures.
- [x] Implement shared validation, bounded decoding, cached catalog, local metadata
  fallback with conflict checking, per-filesystem admission, unique checkpoint temporaries.
- [x] Run targeted tests; commit the verified batch.

## Task 2: Stable observations and validity (findings 3, 7)

Files: `predict.py`, `config.py`, `cli.py`; tests: `test_prepare.py`.

Interfaces: `observations(..., validity=None)`; preparation records canonical tile,
normalization, padding, validity and halo policies in its identity.

- [x] Add tests comparing nonlinear window-dependent predictions across chunk sizes
  and overlapping ROIs, plus fill-mask footprint and valid-zero controls.
- [x] Observe failures, implement fixed source-anchored tiles and explicit validity.
- [x] Run preparation tests and actual pinned-SD2 chunk comparison; commit.

## Task 3: Independent solve grid and joint map reads (findings 2, 10)

Files: `artifacts.py`, `field.py`, `meshes.py`, `config.py`, `cli.py`;
tests: `test_field.py`, `test_prepare.py`, `test_cli.py`.

Interfaces: `field_geometry(meta, spacing_um)`; `Tiles.read_many(names,lo,hi)`;
bounded trilinear sampling of oriented prior/mesh presence in physical coordinates.

- [x] Add tests for exact 8.65/17/130/300 pitches, analytical physical ramps,
  support gaps, distant fine-tile constraints, resume identity and one NPZ traversal.
- [x] Observe failures; implement geometry and sampling adapter, adapt constraints,
  initialization, extraction phase and plan estimates to the field grid.
- [x] Run targeted tests and coarse-grid end-to-end extraction; commit.

## Task 4: Interruptible exports (finding 6)

Files: `exports.py`, `workflow.py`; tests: new `test_exports.py`.

Interfaces: `export_lasagna(run, stop=None)` returns completion boolean or None;
`export_afv(run, stop=None)` returns final SHA or None. Lasagna chunk hashes live
in `exports.sqlite`; AFV partial SQLite commits each bounded fibre record.

- [x] Add stop/resume equivalence, retained-file timestamp, corrupted export and
  changed-source identity tests; verify failures.
- [x] Implement hash-bound chunk resume and transactional record resume, then wire
  stop callbacks into stage completion.
- [x] Run exporter and workflow tests; commit.

## Task 5: Documentation, installed verification and whole-change review

Files: `README.md`, `docs/FORMATS.md`, `docs/verification/audit-fixes.md`.

- [x] Document independent pitches, input masks, canonical inference and resume rules.
- [x] Run full suite, build and install wheel, run suite outside checkout.
- [x] Exercise live gzip catalog; fresh synthetic and original-CT demos; repeat resume.
- [x] Obtain fresh-context review; fix important findings with regressions.
- [x] Record exact evidence and limitations; commit delivery without moving v0.1.0.

Acceptance: 80 source tests / 80 isolated installed-wheel tests. All ten findings,
three review findings and the actual low-space checkpoint retry regression resolved.
See `docs/verification/audit-fixes.md` and its machine-readable receipt.
