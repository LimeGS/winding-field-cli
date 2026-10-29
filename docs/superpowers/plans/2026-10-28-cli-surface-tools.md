# CLI surface-tool ownership implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Leave winding-field with field construction and numerical field benchmarks; move downstream surface, export and visualization functionality into winding-field-cli.

**Architecture:** Move existing implementations and their tests into `winding_field`, preserving algorithms and validation. Core benchmarks produce numerical results independently; `winding-field ct-visuals` renders frozen CT overlays in a separate output directory. Existing prepared and CT workflows import the CLI-owned mesh modules.

**Tech stack:** Existing Python, NumPy/SciPy, Pillow, scikit-image and optional Matplotlib; installed winding-field numerical dependency.

**Spec:** User request: “move esto solo a cli, que el core quede solo con construccion y benchmark de campo, el resto todo a cli.”

## Global constraints

- Preserve the original dirty core checkout; use the existing clean linked worktrees, based on current published main trees.
- Preserve numerical solvers, physical coordinates, support, resource bounds, validation and integrity-bound resume.
- Move tests with implementation; core tests must not import CLI or image libraries.
- Package tools under `winding_field`; child processes use `python -m winding_field.…`.
- Retain the CLI's existing workflow stage defaults.
- Removing flat surface imports is an API change: advance core to `1.3.0.dev1` and match CLI dependency/provenance.

## Review focus

- Installed wheels, outside either checkout: all moved imports and subprocesses resolve.
- Corrupt/changed field, support and visual archives: reject without a success receipt.
- Numerical-only benchmarks: operate even when case JSON contains an unavailable visualization block.
- Existing run outputs: never rewrite receipts to bypass changed implementation identity.
- Mixed tests: keep solver/benchmark checks in core and mesh/render/export checks in CLI.

## Task 1: Move surface implementation and tests

Files: core `src/surface_mesh.py`, `src/surface_sources.py`, `tools/surface_{export,atlas,probe,geometry_check}.py`, `tools/run_watch.py`; matching CLI modules/tests; callers in `field.py`, `meshes.py`, `flatten.py`, `workflow.py`, `surface_workflow.py`.

- [x] Move implementations, update package imports, use module-based child launch, and preserve source hashes in provenance.
- [x] Move matching tests (mesh regressions to `test_surface_mesh_regressions.py` to avoid collision); make the dataset fixture independent of core test modules.
- [x] Update dependency identity assertions and add installed command routing/help checks.
- [x] Run migrated tests plus existing field/mesh/workflow tests.

## Task 2: Separate numerical evaluation from pictures

Files: core `benchmarks/winding/{evaluate,ct_visuals,test_evaluate,test_geometry}.py`, `src/patch_check.py`, `examples/benchmark_demo.py`, `tests/test_benchmark_demo.py`; CLI `ct_visuals.py`, `patch_visuals.py`, `cli.py`, tests and offline visual example.

- [x] Move CT rendering and visual tests; retain numerical evaluator tests and demo in core.
- [x] Expose `winding-field ct-visuals --case --field --metadata --output [--support …] [--check-only]`, with field/support hashes, geometry checks and output/input manifests.
- [x] Move patch rendering to a CLI module using core `evaluate_patch`; keep numeric patch evaluation in core.
- [x] Verify core ignores optional visual metadata and imports no plotting/CLI modules; verify CLI renders actual candidate samples and preserves missing support.

## Task 3: Packaging, documentation and verification

Files: both `pyproject.toml`, CI workflows and guides; CLI `provenance/core.json`, `THIRD_PARTY.md`, `docs/SURFACE_TOOLS.md`, verification receipt.

- [x] Remove surface modules/extras from core packaging, add CLI plot/test extras, update CLI routing, docs and migration instructions.
- [x] Run full core tests with coverage, lint/format/type checks and offline benchmark; run full CLI tests.
- [x] Build core wheel/sdist and CLI wheel; check inventories and run installed tests outside checkouts without core surface modules.
- [x] Review both changesets; record exact checks and remaining untested integrations.
- [x] Commit local core changes, pin the revision in CLI provenance, and commit the CLI migration.
