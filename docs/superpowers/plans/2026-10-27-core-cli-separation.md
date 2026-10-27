# Core / CLI separation implementation plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkboxes for tracking.

**Goal:** Move prepared workflows and rendering to winding-field-cli and replace its copied numerical core with the original library dependency.

**Architecture:** Preserve the original flat numerical modules and their performance implementation. The CLI owns `winding_field.field_workflow`, `surface_workflow`, `surface_render` and the sole `winding-field` executable; prepared commands use `winding-field prepared build|resume|mesh|render|surfaces`. Both repositories get local `feat/core-cli-separation` branches.

**Tech Stack:** Python 3.11/3.12, setuptools, pytest, existing NumPy/SciPy/Torch implementations.

**Spec:** User's two requests in this session: move field_workflow.py, surface_workflow.py and rendering out of the original; consume the original library rather than a copy; create a new original-repository branch.

## Global constraints

- Preserve the dirty original checkout, including unpublished performance work; snapshot it in an isolated worktree before migration.
- Keep solver math, support/ROI semantics, independent positive field pitch and retained-data integrity.
- Preserve prepared workflow capabilities, supervised workers, exit statuses and examples.
- Keep the existing distribution name `winding-field-experiments`, bump the unreleased library to `1.3.0.dev0`, require that exact version in the CLI. Record its original repository commit for installation/CI.
- Existing implementation-bound runs require the original packages; do not rewrite historical receipts to adopt them.
- Keep branches local until publication is requested.

## Review focus

- Installed wheels must share actual solver objects and have exactly one `winding-field` entrypoint.
- Prepared subprocesses must run from outside the checkout, including supervision/prefetch.
- Changes in external solver source must invalidate CT resume without modifying retained results.
- Mixed surface regressions must remain with their owner, with the core suite independent of the CLI.
- Existing partial/completed workflow resume and invalid-input exit statuses must survive namespace changes.

## Task 1: Preserve the baseline and establish isolated branches

- [x] Snapshot dirty tracked/untracked original files and their hashes; copy into a new original worktree on `feat/core-cli-separation`. Create the CLI branch from `0435f59` in its existing isolated worktree.
- [x] Run original baseline tests/checks, retain logs, and commit the exact original snapshot as a separate baseline commit on the new branch.

## Task 2: Move workflows and remove the original command

**Files:** original/CLI `src`, `tests/test_{field_workflow,surface_workflow,surface_render,block_example,surface_regressions}.py`, `examples/{block_field,surface_pipeline}.py`, JSON recipes, `tools/{run_field,run_surfaces,benchmark_blocks}.py`, packaging, CI and README.

**Interfaces:** Preserve `run_field(config_path, output, *, resume=False, prefetch=False, stop=None, admit=None)`, `run_surfaces(config_path, output, *, resume=False, render=True, stop=None)` and `render_bundle(mesh, output, *, settings=None)` under `winding_field`.

- [x] Move the three modules, their complete tests, wrappers and examples. Split mixed regressions so numerical tests still run without the CLI; give retained numerical tests independent fixture setup.
- [x] Adapt only package imports and supervised launch (`python -m winding_field.field_workflow`). Add prepared command dispatch and preserve its error codes.
- [x] Remove the original console script/module exports, relocate workflow docs and installed CI acceptance, and bump the library version.
- [x] Run migrated workflow/render tests, supervised examples, original tests, formatting/lint/type checks and build the original wheel. Commit the library separation.

## Task 3: Consume and fingerprint the original dependency

**Files:** CLI `pyproject.toml`, numerical consumers (`field`, `predict`, `meshes`, `flatten`, `workflow`), `tests/test_field.py`, `tests/test_core_dependency.py`, `tests/test_cli.py`, `tools/vendor.py`, `provenance/core.json`, `THIRD_PARTY.md`, README and CI.

**Interfaces:** Import `solve_blocks`, `pair_constraints`, `BlockStore`, `FieldSource`, `extract_mesh`, `MeshReader`, `density_normals` directly from installed flat modules. `implementation_sha()` includes the dependency's version and Python sources.

- [x] Add dependency-identity regression: replacing a numerical module's source hash changes `implementation_sha()` and rejects resume while preserving the checkpoint. Run it before implementation to observe failure.
- [x] Declare the exact library dependency, replace all `_core` imports, remove `_core` and its vendoring logic, and record the new original commit plus migrated-file provenance.
- [x] Fingerprint the installed numerical dependency. Test dispatch via actual installed command, dependency identity and shared numerical objects.
- [x] Update installation instructions and CI to install the original at its recorded revision alongside the CLI.

## Task 4: Verify the two distributions together

- [x] Build/install both wheels in a verification environment; run both suites outside their source checkouts, dependency checks and command ownership checks.
- [x] Run prepared build/resume/prefetch/supervision and surface extraction/render examples using installed packages; run the CT pipeline and confirm completed resume.
- [x] Run fresh bounded original-CT Qual acceptance at 17 µm; compare numerical products with the prior acceptance and record scientific limits without rewriting it.
- [x] Check diffs, imports, source/installed parity and original-checkout hashes. Record results, commit the CLI changes and report branch paths/commits and publication status.

## Review resolution

- Independent whole-change review found no critical/runtime issue. Its two
  findings are addressed: core manifest integrity coverage is restored directly
  in the library (three cases), and offline commands create their `runs` parent.
- Core review fix: `163ae9e8bd62a89af51d408d7dbb20302afdbc3c`; CLI review fix and
  updated dependency pin: `6b8557e36f1a0f794e2a4eb20dd8040a4d05178f`.
- Final source verification: 1227 core tests with 92.35% coverage, 273 CLI tests;
  core format/lint/mypy passed. Exact documented examples also pass from a fresh
  directory against installed runtime-only dependencies.
- Rebuilt installed wheels pass 1227 core / 273 CLI tests outside the checkouts;
  their runtime sources, sole executable owner and final provenance match.
- Review corrections change tests/documentation only. Installed runtime sources
  retain the accepted CT implementation identity; another CT solve is unnecessary.
- Legacy algorithms remain unchanged; the independent core suite guards them,
  but this relocation does not establish new scientific/CUDA/full-scroll results.
- Branches remain local as requested; remote CLI CI needs publication of the
  recorded original revision. The final verification documents are checked
  directly for artifact/product hashes, counts, links and command syntax.
