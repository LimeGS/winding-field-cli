# Review checklist

Read by reviewers (human or Claude) and by `python tools/verify.py` (`review` step), which
lists the items whose trigger matches the diff. CI reads this file from the **base
branch**, so a change cannot edit the rules that judge it.

Each item has a trigger readable from the diff, the check, and the incident it comes from.
An item triggers when a changed file matches **Paths** and, when a **Diff regex** is given,
one of that file's added or removed lines matches it. Add an item only when a miss happened in
at least two changes or caused an incident, and it is detectable from a diff. Prune items
that have not matched in a year.

| ID | Paths | Diff regex | Check | Origin |
|---|---|---|---|---|
| R1 | `src/winding_field/cli.py`, `src/winding_field/cli_settings.py`, `src/winding_field/field_workflow.py` | `^-\s.*(add_argument\|add_parser\|choices=)` | A public command, flag or choice is removed or renamed only with an entry in `tools/verify-approved.json` (reason + migration); `flags` step passes | 3556a4b dropped `--nonconvergence-policy` and `--reuse-from` (restored in #47) |
| R2 | `src/winding_field/workflow.py`, `src/winding_field/config.py`, `src/winding_field/reuse.py`, `src/winding_field/field_workflow.py` | `implementation_sha\|RUN\.json\|_code_sources\|identity` | Runs made by the previous release resume or are refused cleanly, never modified; new config keys have defaults so an old `RUN.json` validates; `cross-version` step passes | Resume pins (`implementation_sha`, identical RUN.json config); stage reuse loss in 3556a4b |
| R3 | `src/winding_field/surface_render.py` | — | Byte-identical to its provenance hash; never formatted (excluded from ruff) | Accidentally reformatted during the flattening refactor |
| R4 | `README.md`, `docs/*.md` | `^[-+](.*!\[\|## )` | Badges and sections asserted by `tests/test_installation.py` change together with the test | 0e6bdab turned main red (#57) |
| R5 | `src/winding_field/flattening/*.py`, `src/winding_field/native_*.py` | — | `golden` identical or approved; LSCM only through `flattening/lscm.py`, SLIM only through `flattening/flatboi.solve_batch` | SLIM unification (#55), flattening layout (#56) |
| R6 | `src/winding_field/field*.py`, `src/winding_field/config.py` | `cache_bytes\|cpu_workers\|buffer_bytes\|prefetch` | Performance settings never change results: `tests/test_perf_equivalence.py` covers every knob, all on vs all off | Caches and CPU pool added in #46 |
| R7 | `.github/workflows/*`, `Dockerfile`, `docker/*`, `docker-bake.hcl`, `MANIFEST.in`, `pyproject.toml` | — | The container and sdist suites only run on pushes to main: check mounts, packaged files and test-image dependencies before merging | #53, #54 failed only after merge |
| R8 | `tests/*.py` | `^\+.*(pytest\.mark\.skip\|xfail\|time\.sleep\|importorskip\|timeout=)` | No skip, xfail, sleep, retry or larger timeout to get green: find the root cause, or delete the test and write down why | Verification policy; xdist stop-test budget raised 1.5 → 20 s without a root cause |
| R9 | `src/winding_field/flattening/flatboi.py`, `provenance/*.json` | `SHA256\|sha256\|APPIMAGE` | Pins match `provenance/*.json`; `test_embedded_appimage_pin_matches_provenance` passes | Flatboi runtime pinning (#55) |
| R10 | `REVIEW.md`, `tools/verify.py`, `tools/verify-approved.json`, `.claude/*`, `CLAUDE.md` | — | Changes to the verifier, its approvals or these rules carry a reason in the PR body; approvals name the baseline they apply to and are removed once merged | Changes must not weaken their own checks |

## Severity

- **Important**: a bug this change adds or makes worse.
- **Pre-existing**: already broken on the base branch; not the author's to fix here.
- **Nit**: style or minor; never blocks.
