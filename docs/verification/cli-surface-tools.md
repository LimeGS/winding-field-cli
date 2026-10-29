# Field-only core / CLI surface tools verification

Local verification on October 28, 2026, macOS arm64 / Python 3.11.13.

## Ownership

- Core base: `a7246af3cbd43bd88eb529b31b9eb22daf3a6885`.
- CLI base: `40d39628ec4f40908a8a5ecb7448a8ee7fc38d08`.
- Coordinated numerical dependency: `winding-field-experiments==1.3.0.dev1`;
  core commit `f750fd11fcf8d3e5bc6a80290e90d125cfd02a36` and migrated-file
  hashes are pinned in `provenance/core.json`.
- Mesh extraction/readers, surface export/probe/atlas/geometry checks,
  supervision and CT/patch pictures now live under `winding_field`.
- Core benchmarks produce numerical metrics without rendering or CLI imports.
  See [commands and migration](../SURFACE_TOOLS.md).

## Preservation

All **22 other numerical modules are byte-identical** to core main. Every
numerical function in `patch_check` has the same AST; only its plotting command
moved. `surface_mesh.py` and `surface_sources.py` are byte-identical moves.
Remaining adaptations are package imports, command routing, module-based child
launch, independent CT visual preflight and packaging/test/documentation ownership.

The original dirty `winding-field-gh` checkout retained **all 115 files and its
exact Git status**, compared with the pre-separation snapshot. Work occurred in
the existing isolated worktrees, on `refactor/core-field-only` and
`refactor/cli-surface-tools`.

## Checks

| Check | Result |
| --- | --- |
| Baseline core | 1227 passed |
| Baseline CLI | 328 passed, 1 opt-in skipped |
| Core source, coverage | 976 passed, 1 optional codec skipped; 93.22% |
| Core installed wheel outside checkout, codec enabled | 977 passed |
| Core format / lint / mypy | Passed; 24 checked source files including typed caller |
| Core offline benchmark | 6/6 fibre pairs covered; separation 3/3; collapse controls 0/3 |
| Rebuilt core sdist | Minimal field solve passed |
| CLI source, official writer opt-in enabled | 598 passed (107.08s) |
| CLI installed outside checkout, official writer opt-in enabled | 598 passed (105.54s) |
| Review-fix CT/geometry/patch regressions | 37 passed, including nine RED→GREEN cases |
| Installed migrated command help and synthetic CT example | Passed; actual candidate samples checked |
| Independent whole-change review | Completed; all four findings addressed |
| Installed dependency consistency | All 66 CLI-environment packages compatible |

The isolated core environment contains no CLI, Pillow, scikit-image or
Matplotlib. The wheel has exactly the remaining numerical modules and no console
entrypoint. The sdist also carries numerical benchmarks, demos, runners and test
fixtures; its previous incomplete source inventory was reproduced and corrected
with a CI assertion. Core wheel sources matched the checkout byte-for-byte.
The CLI wheel contains 48 Python/JSON package files matching the checkout,
includes the exact core provenance, and owns the sole `winding-field` console
entrypoint.

Source tests used `PYTHONPATH=src`; installed suites use copies of tests/examples/
wrappers outside both checkouts, with no source `PYTHONPATH`. Legacy core test
paths named `src/` point to the actual installed modules. CLI tests use
`-o pythonpath=`. The full CLI source run enabled `WINDING_LASAGNA_PYTHON` for the
existing official Python 3.14 writer regression. This does not run fresh inference.

Regression checks observed failing before implementation and passing afterward:
CLI-owned mesh imports, numerical evaluation without CT archives, the CT visual
command and patch plotting. Migrated CT tests preserve candidate gauge, frozen
levels, unsupported pixels, physical axes, archive integrity and synthetic labels.
An intermediate full-suite failure selected an old installed CLI after upgrading
the core; installing both coordinated packages resolved all three failures.

## Review corrections and migration decisions

The independent review found no Critical issues. Four findings were addressed:
nonempty-frame and producer-success validation, explicit historical `--config`
geometry fallback, the original calibration tolerance, and prepared-workflow
ownership documentation. Nine new cases failed before correction and passed
afterward; the final full suites above include them. No findings were deferred.

- Include numerical benchmark/demo/runner sources and fixtures in the core sdist:
  artifact inspection showed they were missing. Cost: a slightly larger source
  archive, with the numerical wheel inventory preserved.
- Treat tolerance and ownership corrections as required migration work:
  exact equality rejected previously valid inputs, and extraction ownership is
  the central requirement. Cost: preserve the original acceptance semantics and
  correct the guide's ownership paragraph.
- Verify software/package behavior with local synthetic and installed checks:
  physical accuracy and live native integrations need separate evidence. Cost:
  this migration makes no new real-data or native-runtime acceptance claim.

## Local integration status

The changes are retained on `refactor/core-field-only` and
`refactor/cli-surface-tools`; neither branch was pushed or merged as part of this
migration. Publish the pinned core commit before publishing the CLI branch so
the documented GitHub-source dependency can resolve.

## Limits

These are local CPU software/package checks. Native Flatboi and external geometry
gate binaries were represented by existing test fixtures; no fresh GPU, real-CT
reconstruction or physical-sheet validation was performed. Old implementation-
bound runs require their original packages or a fresh output directory. Publication
of these local branches is separate from this verification.
