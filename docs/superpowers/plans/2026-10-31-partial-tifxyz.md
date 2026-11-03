# Partial TIFXYZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a usable, verifiable partial TIFXYZ without pretending complete coverage.
**Architecture:** Qualify only post-geometry coverage failures; publish distinct immutable
partial markers and propagate quality through independent verification and workflow.
**Tech Stack:** Existing Python, NumPy, TIFF/SQLite, native audit backends.
**Spec:** `docs/superpowers/specs/2026-10-31-partial-tifxyz-design.md`.

## Global Constraints

- Strict default; partial only by explicit `coverage_policy=allow_partial`.
- No relaxed geometry, units, provenance, resource or native identity gates.
- Source scan may stop at first failure; unexamined extent remains UNMEASURED.
- PARTIAL.json and READY.json mutually exclusive, marker-last publication.
- No commits/installs/deletion; keep paused r3b and original datasets unchanged.

## Review Focus

1. Coverage permission must not authorize geometry/root/resource failures.
2. A forged/truncated coverage report must not become complete or partially verified.
3. Mixed levels and interrupted markers must not falsely register COMPLETE.
4. Backend/source identity checks must run even after a nonfatal coverage failure.
5. Parent workflow and reader must preserve masks and distinguish partial quality.

### Task1: Audit and publication quality

Files: tifxyz_contract.py, tifxyz_audit.py, tifxyz.py; contract/audit/export tests.
Interfaces: `Settings.coverage_policy`; audit `geometry_complete`, `blocked_phase`,
structured `reasons[].region` (ROI-relative/native global cell, dyadic subcell index,
depth). `_audit_status(audit, policy)` returns READY/PARTIAL or raises. Partial keeps
source_scan_complete=false. Export result PARTIAL has coverage_complete=false.
- [ ] RED natural plane field3x8x8 with pitch4/tau3 gives a valid but incomplete grid;
  strict blocks, allow_partial writes TIFXYZ/PARTIAL.json and no READY.
- [ ] RED hard-error/cleanup/resource cases cannot publish partial; record region.
- [ ] Implement explicit policy and per-level/global quality/marker publication.
- [ ] Run focused tests: PASS.

### Task2: Independent verify and interruption semantics

Files: tifxyz.py, tifxyz_runtime.py, export/runtime tests.
- [ ] RED fresh partial verify returns TIFXYZ_PARTIAL_VERIFIED; mutated masks,
  quality, reasons, source/backend, or conflicting markers fail. Mixed levels tested.
- [ ] RED partial resume stays partial; interrupted writes never leave a final marker.
- [ ] Implement recomputed partial audits, quality checks and marker cleanup.
- [ ] Run focused tests: PASS.

### Task3: CLI/workflow and reader

Files: cli.py,workflow.py,test_tifxyz_cli.py,test_tifxyz_vc3d.py, README/FORMATS.
- [ ] RED PARTIAL and TIFXYZ_PARTIAL_VERIFIED exit2; parent registers snapshots and
  partial marker but RUN remains PARTIAL. Resume corruption fails; legacy unchanged.
- [ ] Extend native reader acceptance to partial files without claiming full coverage.
- [ ] Document opt-in JSON, statuses, known versus unmeasured extent and commands.
- [ ] Run focused tests: PASS.

### Task4: End-to-end delivery

- [ ] Full suite and independent review; Important fixes RED→GREEN.
- [ ] Linux export→PARTIAL→fresh partial verify→real VC3D reader; inspect masks/files.
- [ ] Installed wheel and source assets; record evidence and real-data disposition.

Execution: user approved end-to-end implementation now; use the established native
in-place method without another approval pause. Isolated CLI/workflow work may be
delegated under this contract; engine/quality integration stays with the main agent.
