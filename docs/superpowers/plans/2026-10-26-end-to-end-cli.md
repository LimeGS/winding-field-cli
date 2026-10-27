# End-to-end winding-field CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a public, runnable original-CT-to-unrolled-mesh CLI with resumable prerequisites and bounded streaming.

**Architecture:** A single run manifest binds source geometry and stage settings. A common CT reader and chunked intermediates feed the existing least-squares/block solver and global mesh extractor; independent chart flattening preserves correspondence and coverage. Upstream neural inference and an explicitly labelled classical baseline share the same observation contract.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Torch, scikit-image, Zarr/numcodecs, vc_reader optional native backend, pytest, argparse.

**Spec:** `docs/superpowers/specs/2026-10-26-end-to-end-design.md`

## Global Constraints
- Arrays ZYX; coordinates and spacing XYZ. Record original acquisition, units and transforms.
- Use original CT; no silent scan, level or codec substitution.
- Local-first; download when required working set fits, otherwise bounded on-demand caching.
- Input halos allowed; output ROI remains exclusive.
- Reuse mathematical code and preserve source attribution.
- No modification of source repositories, existing runs or machine-wide environments.
- Failures and resource exhaustion retain resumable progress.
- Scientific correctness is not implied by a converged solver or an exported mesh.

## Review Focus
- Incomplete local Zarr must fetch missing chunks, not create artificial air.
- Remote metadata/source changes must invalidate cached identities and resume.
- Nonzero origins, anisotropic pitches and ROI clipping must preserve XYZ.
- Interruptions may leave temporary files but cannot mark incomplete stages complete.
- Zero/ambiguous observations and invalid UVs cannot be reported as recovered sheets.

### Task 1: Installable package and source/CT contract
**Files:** `pyproject.toml`, `src/winding_field/{__init__,cli,io}.py`, `tests/test_io.py`.
**Interfaces:** `Volume(source, local=None, cache=..., cache_bytes=..., offline=False, voxel_um=None, level=0)` exposes `shape`, `spacing_xyz`, `origin_xyz`, `identity`, `read(lo_zyx,hi_zyx)`, `stats()`; `resolve_source(scroll,source,scan)` returns exact original source.
- [ ] Test HTTP/local equality, missing local fallback, bounded eviction, offline miss, metadata mismatch, invalid coordinates and physical metadata.
- [ ] Run `python -m pytest tests/test_io.py -q`; expect missing feature failures.
- [ ] Implement bounded source access and source discovery; configure packaging and CLI.
- [ ] Repeat the command; expect all pass; commit.

### Task 2: Chunked preparation, prediction and fibre tracing
**Files:** `src/winding_field/{artifacts,predict,fibres,models}.py`, `tests/test_prepare.py`.
**Interfaces:** `Tiles(root,shape,chunk)` supports `write(key,arrays)`, `read(name,lo,hi)` and resumable completion; `prepare(volume,run,config)` emits geometrically bound maps; `trace(run,config)` writes fibre database.
- [ ] Test CT-derived planar normals/density, normal/tangent distinction, prediction core/halo continuity, stable traced coordinates, unsupported gaps, and model identity.
- [ ] Run `python -m pytest tests/test_prepare.py -q`; expect feature failures.
- [ ] Integrate labelled classical and pinned upstream neural observation paths; chunk persistence and bounded tracing.
- [ ] Repeat command; expect all pass; commit.

### Task 3: Winding field and resumable numerical execution
**Files:** `src/winding_field/_core/*.py`, `src/winding_field/field.py`, `tests/test_field.py`, `provenance/core.json`.
**Interfaces:** `solve(run,config)` consumes prepared physical observations/fibres and emits core-compatible CHECKPOINT plus diagnostic metadata; no complete stage on numerical failure.
- [ ] Test an analytically known layered CT, physical units, constraints, nonconstant separation, interrupted block resume and unchanged-input requirements.
- [ ] Run `python -m pytest tests/test_field.py -q`; expect feature failures.
- [ ] Import selected generic MIT core modules with provenance and namespace imports; connect tiled observations and fibre constraints.
- [ ] Repeat command; expect all pass; commit.

### Task 4: Global meshes, flattening and CT correspondence
**Files:** `src/winding_field/{meshes,flatten}.py`, `tests/test_meshes.py`.
**Interfaces:** `mesh(run,config)` consumes a frozen field and emits all components; `flatten(run,config)` emits chart OBJ/UV and face-to-source mapping plus distortion/coverage.
- [ ] Test plane/curved-chart flattening, shared tile vertices, all components, exact ROI bounds and invalid/degenerate meshes.
- [ ] Run `python -m pytest tests/test_meshes.py -q`; expect feature failures.
- [ ] Connect native extraction and bounded chart parametrization; optional native Flatboi adapter and CT previews.
- [ ] Repeat command; expect all pass; commit.

### Task 5: One-command orchestration and demos
**Files:** `src/winding_field/{workflow,demo,cli}.py`, `tests/test_cli.py`, `README.md`.
**Interfaces:** `run(config,resume=False,until='flatten')`, CLI `run/resume/status/plan/doctor/demo` and stage commands. Manifest stores stage outputs and effective settings.
- [ ] Test subprocess invocation from another directory, full synthetic CT pipeline, stop/resume, changed inputs, and empty-volume failure reporting.
- [ ] Run `python -m pytest tests/test_cli.py -q`; expect feature failures.
- [ ] Implement stages/progress/space admission and demo outputs with honest numerical/physical statuses.
- [ ] Repeat command; expect all pass; commit.

### Task 6: Installation, real CT verification and public delivery
**Files:** `.github/workflows/tests.yml`, `docs/DEMO.md`, `docs/FORMATS.md`, `THIRD_PARTY.md`, release artifacts.
**Interfaces:** wheel/sdist installation supports `winding-field demo --output NEW` without repository-relative inputs.
- [ ] Run full suite, build wheel/sdist and install into a fresh environment.
- [ ] Run synthetic demo and a bounded original-S3-CT demo; preserve measured receipts and screenshots.
- [ ] Exercise vc_reader where buildable and document fallback/installation accurately.
- [ ] Fresh-context whole-repo review; fix important issues with regressions.
- [ ] Publish new GitHub repository and tagged release after verification; record remaining scientific/runtime limitations.
