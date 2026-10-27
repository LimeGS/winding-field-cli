# Core / CLI separation verification

Verified locally on October 27, 2026, on macOS / Python 3.11.13.
Both repositories use local `feat/core-cli-separation` branches.

## Ownership and preservation

- Original library: `winding-field-experiments==1.3.0.dev0`, revision
  `163ae9e8bd62a89af51d408d7dbb20302afdbc3c`.
- The original's previously uncommitted performance work was preserved first in
  `769ed1b1587fc2a275a66059e15a36b8fa7acc93`. The existing
  `feat/performance-optimizations` checkout retained all 115 tracked/untracked
  source files byte-for-byte and its exact Git status.
- All **25 remaining library modules** have the original bytes. Its wheel has
  no workflow/render modules and no console entrypoint.
- CLI implementation commit: `b0155e0`. The CLI imports the original numerical
  objects directly. Its wheel has the three moved modules, no `_core`, and owns
  the sole installed `winding-field` entrypoint.
- Review corrections and final dependency pin: CLI `6b8557e`; its verification
  artifacts were built from that exact committed snapshot.
- Every test function removed from the original remains in one of the two test
  suites. Numerical surface regressions have an independent core fixture.
- Both wheels match their source runtime files. Both sdists contain the required
  recipes, examples, test helpers and documentation. Artifact hashes are in the
  [receipt](core-cli-separation-receipt.json); these are local verification builds,
  made after review corrections and before this final evidence documentation,
  not published releases.

## Tests and checks

| Check | Result |
| --- | --- |
| Original pre-migration baseline | 1380 passed, 147.81 s; 93.00% coverage |
| Core source after separation | **1227 passed**, 124.88 s; **92.35% coverage** |
| Core installed wheel | **1227 passed**, 99.26 s |
| CLI source | **273 passed**, 107.25 s |
| CLI installed wheel | **273 passed**, 106.15 s |
| Core format / lint / mypy | Passed |
| Installed dependency compatibility | All 66 packages compatible |

A second clean environment installed only the two wheels and their declared
runtime dependencies (**40 packages**, no pytest/Zarr/Matplotlib/dev extras).
Supervised prepared execution, prefetch, the surface example and the complete CT
demo also passed there, so the development extras do not mask runtime imports.

Installed suites ran outside both checkouts with `PYTHONPATH` removed. The core
tests' legacy `src/` references were symlinked to actual installed site-packages,
not a copied source tree. CLI wheel tests had pytest's `pythonpath` empty.

Three migration regressions failed before implementation and passed afterward:
shared original solver objects, rejection of changed external solver source on
CT resume without rewriting retained products, and prepared-command input-error
dispatch. Migrated tests retain prepared supervision/prefetch, strict validation,
partial/completed resume, corruption rejection and bounded rendering coverage.

The actual installed executable exercised all five prepared commands:
`build`, `resume`, `mesh`, `render`, `surfaces`, including supervised workers and
prefetch. Relocated examples stopped/resumed successfully; completed surface
resume retained mesh bytes. The complete-cycle probe confirmed numerical parity.

## Review corrections

The independent review found two issues, both addressed:

- Restored the two mesh-manifest integrity tests (three parametrized cases) to
  the core, using `extract_mesh` and `MeshReader` directly. They pass with the
  CLI absent from the test environment. The CLI retains workflow/render tests.
- Added `mkdir -p runs` before the offline examples. The exact documented block
  now passes from a new directory using only installed runtime dependencies.

These corrections change tests/documentation only. The final dependency pin
includes the restored core tests; all runtime Python/JSON sources and the CT
implementation identity remain unchanged.

The final installed sources were compared with both checkouts, and the installed
provenance matches the final original revision. The CT run therefore retains the
same implementation after review; it was not repeated for test/documentation-only
changes.

## Original-CT comparison

A fresh installed run used the same original 175A CT and ROI as the
[previous Qual acceptance](qual-v2-splines.md): native 8.64 µm observations,
ROI `1488 1536 4032 4080 6048 6096`, storage chunk 16, one solve sweep,
field chunk 16 and exact **17 µm** field pitch.

Results: **8 splines, 106 constraint rows, 24³ field nodes, 227 mesh faces,
12 charts and 100% extracted-face coverage**.

Compared with that prior run:

- All seven observation channels are **bit-identical**.
- Decoded field values and support are **bit-identical**.
- Mesh triangles and group IDs are **bit-identical**.
- AFV and atlas SHA-256 hashes are identical.
- Completed resume preserved all nine selected product hashes.

The new implementation identity is
`7e1e84f8aa31f4e9699fe28e784dedfbb55e7ef7b63b8f190844360386fcf7e5`.
Historical receipts were not rewritten. Local logs, built artifacts and the
fresh run are retained under `.runtime/separation/`.

## Operational limits

Prepared commands now require the `prepared` prefix, and workflow imports use
`winding_field.*`. Prior implementation-bound runs require their original
packages or a new output directory. The dependency version and original revision
are pinned; publish the original branch before running remote CLI CI. No merge,
push, PR or package release was performed for this separation.

This is CPU/ROI software equivalence, not new physical validation. Global
convergence and physical sheet accuracy remain unvalidated. CUDA and whole-scroll
performance were not exercised in this migration.
