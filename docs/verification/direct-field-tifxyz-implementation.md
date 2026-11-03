# Direct field → TIFXYZ: implementation checkpoint

## Partial publication update

Explicit `coverage_policy=allow_partial` is implemented end to end; strict remains
the default. Only source-uncovered/depth-limit failures after complete delivered
geometry qualify. Regions and unexamined coverage are recorded; masks stay-1 and
no points are fabricated. PARTIAL.json is exclusive with READY.json. Fresh verify
returns TIFXYZ_PARTIAL_VERIFIED and does not upgrade coverage. CLI uses exit2 for
usable partial results; integrated RUN remains PARTIAL, with bound product snapshots.

Verification:545local passes/17skips; actual Linux libigl/CUDA partial export,
fresh independent verify, realVC3D reader and readonlypartialresume all passed on
a natural incomplete synthetic control. Installed-wheel checks passed. Independent
review's weaker-parent-validation finding was fixed with four RED→GREEN regressions.
Evidence: `direct-field-tifxyz-evidence/PARTIAL-EXPORT-INTEGRATION.json`.

The new real candidate1 chain is running under
`gpu-1:/ssd/draft/partial-tifxyz-real-candidate1-20261031-01/CURRENT.json`.
It reuses the existing fine field with the same ROI/W/tolerance, in a new frozen
namespace, and automatically runs export→independent verify→VC3D. It does not
silently migrate old-code blocks or their quality claims. Real completion is
pending; prior runs remain preserved. Sections below retain earlier evidence.

## Hybrid audit implementation update

The approved CPU/CUDA integration is implemented. Default `audit_backend=numpy`
uses bounded filters; optional `cpu`/`cuda` use explicitly built, SHA-bound native
artifacts and retain exact Python fallbacks. CUDA additionally accelerates the
batched orientation/coverage filters; host compiled exact contact filters remain.
SQL preparation/transactions, read-only resource polling and audit progress were
also improved. CPU keeps AS; CUDA uses DATA plus the same aggregate RSS supervision.
No library/device path is taken from an untrusted export receipt.

Verification:461local passes/17skips;189Linux focused passes; real CUDA build and
supervised synthetic export/fresh verification passed. Installed wheel, native-source
assets and explicit native build from an external installed-package cwd passed.
One independent-review finding (late accelerator replacement) was reproduced in
four cases and fixed before acceptance testing.

Complete calibrated33×33 audit medians: frozenreference46.956s,NumPy27.712s,
compiledCPU17.851s,CUDA17.278s. Every audit receipt/counter/max agrees exactly.
This is **2.72× on that full-audit control**, not a real-W or full-export speedup.
CompiledCPU is nearly as fast in this small control; remaining traversal/Python
costs still matter. Detailed evidence: `direct-field-tifxyz-evidence/HYBRID-INTEGRATION.json`.

A separate real retained-tile audit finished **BLOCKED** after3h27 under
`gpu-1:/ssd/draft/hybrid-tifxyz-20261031-01/real-cuda/STATUS.json`.
The reason is COVERAGE_DEPTH_LIMIT at border cellZYX(511,62,0), depth8. Read-only
diagnosis now confirms an actual boundary coverage deficit: an exact source W=-4
point is24.984632777um from either delivered triangulation, above tau17.28um.
The guide contains that boundary point, but the UV raster omits the local strip
below its first valid row. Exact replays at depths9/10 still fail; increasing depth
cannot repair unchanged output. All4862732triangles were independently bbox-scanned
and libigl confirmed the distance. See `direct-field-tifxyz-evidence/COVERAGE-BORDER-DIAGNOSIS.json`.
It cannot publish READY and does not substitute for root reprojection. The user
approved stopping originalr3b: it is now PAUSED, its worker/controller exited, and
the acceptance follower exited withPILOT_NOT_ACCEPTED. All129work-file sizes/mtimes
were preserved; no READY exists. The frozen r3b runtime remains unchanged. The
historical checkpoints below describe earlier work and their observed states.

## Status

Core T1–T6 and the T7 experimental CLI branch are implemented and tested locally.
**The empirical quality gate has not passed. Full Linux/VC3D acceptance and
large-W execution remain pending.** No real-field direct TIFXYZ is claimed accepted.

Repository base: `6b84457`; changes are uncommitted. The independent Qual V2 plan
is preserved. The legacy mesh/flatten workflow remains the default; compatibility
tests cover the new branch selection. No core solver changes were made.

## Implemented modules

- `tifxyz_contract.py`: explicit requests, units, ROI and frame contracts.
- `tifxyz_field.py`: torch-free source binding and bounded physical field views;
  coarse support checks include intervening fine nodes.
- `tifxyz_guide.py`: bounded global guide, conservative cuts and global UV solve.
- `tifxyz_project.py`: exact physical ray DDA, certified interval fast path and
  exact rational fallback; no nearest-root selection or hidden root-failure masks.
- `tifxyz_store.py`: immutable bounded blocks, journal/DB quotas, recovery and
  streamed tiled TIFF publication primitives.
- `tifxyz_audit.py`: delivered-node normalization, disk spatial indices, both
  diagonals, support/orientation/intersections and whole-ROI coverage bounds.
- `tifxyz.py`: export/verify engines, retained attempts, READY-last publication,
  independent reprojection and scratch-only verification.
- `tifxyz_runtime.py`: resource observations and Linux worker supervision.

## Verification observed

Latest complete local suite: **394 passed, 16 skipped**, 147.18 seconds.
Skips include Linux-only supervision and unavailable local libigl paths.
The GPU1 geometry/root suite ran against installed libigl, including regressions
for false physical-ray absence/uniqueness and internal-boundary cusps.

Review findings were reproduced RED then fixed: physical versus rounded-grid
rays; cusp classification at search endpoints; truncated witness ray partitions;
controller-adjusted child memory admission. Source-bound witness claims require
a separate replay of original source values and the frozen guide pixel.

Integration fixes include R-tree reopen authorization, standard TIFXYZ metadata,
and inward-directed float32 ROI bounds with displacement charged to root error.
Caller-level verification no longer hashes all unrelated checkpoint payloads:
consumed ROI bytes still pass the unchanged FieldSource integrity checks; resume
revalidates the complete ROI. Receipts explicitly disclaim whole-checkpoint
payload verification.

Latest independent review: `ses_e710797afffeFeFd6OlP1JLUVU`. Two Important
findings reproduced RED→GREEN: canonicalize source/output before the standalone
containment check; charge verification scratch left by abrupt worker termination.
Verification now serializes the shared budget with the existing output lock and
counts retained `.OUT-verify-*` siblings, without deleting them or changing the
accepted output. Both path spellings with `..`, symlink rejection, abrupt child
exit, and competing-lock cases have regressions. The full suite above includes them.

An earlier full-suite invocation timed out at120seconds after displaying an
unidentified failure. It was not localized; two subsequent full runs passed
(357/15 before the review fixes,362/15 after). Do not describe its cause as fixed.

## Optimizations with evidence

- Ray batching: controlled projection initially made465field reads; after filling
  the existing16384-cell queue, the same test passes the <100read bound and tile
  invariance checks. This is not yet an end-to-end speedup measurement on W−4.
- Exact audit contact predicates: common-power-of-two integer fast paths retain
  rational fallback. On a33×33-node real-store synthetic plane, complete audit
  fell22.01→11.03s, with identical counters/bounds. No300k-cell extrapolation claimed.
- All source/guide identity, support, coverage and publication gates are retained.

## Real pilot evidence

Fixed pilot/source receipts and counterexamples are in
`direct-field-tifxyz-evidence/`; native source adapter matched12tiles/127cases.
Four candidate guide failures are independently source-replayed and certified
by SymPy root counting over the complete physical ray, not the production solver.

Candidate1/stride4 remains unproven. Its native baseline prerequisite was checked:
397384vertices,792198faces, exactly1component, edge/vertex manifold. Native census
took102.86s and peaked628350976bytes RSS; this is not the complete global-UV baseline.
Receipt on gpu-1:
`/ssd/draft/direct-field-tifxyz-20261029/native-census-candidate1-20261030-8cfd6a72/RESULT.json`
SHA `c04b31abcea462cc5d6187a0784f582ca68da8e6602600f8488ac4d9990ae80c`.

The earlier pre-fix full projection trial took841.45s before rejecting a boundary
ray and peaked344055808bytes. It is retained as diagnostic evidence, not current
performance or a completed conversion.

## Current blocker and continuation

The user restored SSD headroom. The native reference subsequently failed in SLIM
even under its final reference-only3.5GiB address-space envelope; see
`direct-field-tifxyz-evidence/BASELINE-NO-COMPARISON.json`. The user approved
quality-first experimental integration. Comparison remains NO_COMPARISON and
performance ratios UNMEASURED. No further baseline retry is scheduled.

The real direct run `/ssd/draft/direct-tifxyz-qf-candidate1-20261030-r2-4c61d9a0`
finished BLOCKED/SQLITE_WRITE_LIMIT, without READY, after10694.26worker seconds;
worker peakRSS543928320bytes. Read-only diagnosis found the delivered DB at
268419072bytes/65532pages,1218172nodes and1645184triangles committed, with no free
pages and92656246784bytes free on the SSD. The256MiB fixed DB quota caused the stop.
The corrected quota is scratch/8 (2GiB within16GiB), with unchanged page, journal,
cache, aggregate scratch and disk-reserve enforcement. A real257MiB write passes
with the larger approved scratch allowance and still blocks with2GiB scratch.

The refined1247×1247grid also had38conservative output-distance failures across
4862732triangle checks. All38are obtuse; their L/2 vertex-cover bounds plus root
error and margin have maximum15.901949um, below the unchanged17.28um tolerance.
Exact right/obtuse classification now uses that bound; acute triangles retain
L/sqrt(3). Source support, orientation, intersections and reverse coverage remain
mandatory and unproven for this output. No pixels or geometry were discarded.

Next: complete the new fixed candidate1 pilot under the same limits.
Require export READY, independent source-bound verification and the real
VC3D reader before accepting it. Installed-wheel checks below have passed.
Whole-source topology and physical papyrus
correctness remain UNMEASURED regardless of numerical completion.

### Current execution

The corrected frozen runtime also passed184Linux engine/audit/store/runtime tests
in59.59seconds, with CPU1/nice19 and a3GiB address-space limit. Remote CLI tests
could not collect because the existing geometry venv lacks numcodecs; local CLI
coverage passed. No dependencies were installed.

The next real pilot was launched under
`/ssd/draft/direct-tifxyz-qf-candidate1-20261030-r3b-31c85ad2` (controller955075).
Frozen manifest SHA:
`8d688fb1e57c61ff3f0ca39bdab4b8012b66474e28c8052feee834a373d949b7`.
Read CURRENT.json for live status; launch is not acceptance. The preceding r3
namespace is retained: its smoke failed immediately because the launcher omitted
the existing pytest dependency path. r3b fixes only that environment invocation,
using the same archive and implementation bytes. No real source work ran in r3.

At elapsed11289s, r3b was still AUDITING pitch8.64 (1247×1247), with aggregate
RSS443498496bytes and no READY. Its index had grown to567566336bytes: the former
256MiB ceiling was passed without disabling the quota. This is progress evidence,
not a completed geometric certificate. Always read CURRENT.json for current state.

## Installed wheel and real-reader controls

Built wheel193464bytes, SHA
`2d5949cf79479f7785eab934b5a6baa5aa5fd25a6a2fce49c3ff471b6084c8be`.
Installed with `--no-deps --no-index --target` into a fresh temporary directory;
all50packaged Python files match the checkout, and provenance/licenses are present.
From an external working directory: direct-command help, lazy torch/libigl imports,
legacy offline demo and resume passed. Mac direct verification correctly returned
RUNTIME_UNSUPPORTED/75. No dependencies or existing environments were modified.
The initial no-isolation build found setuptools65.5, below the declared>=77;
the successful build reused the existing release environment's setuptools84.2.1.
Receipt: `/var/folders/s_/nnzqmcgs1sdffgsfx68wj7wr0000gn/T/opencode/tifxyz-wheel-20261030-71f2c8a4/RESULT.json`.

`tests/test_tifxyz_vc3d.py` adds an independent pilot-only reader check, limited
to4194304pixels before decoding, with streamed OBJ faces and bounded numeric arrays.
It checks actual vertex/UV mapping, both diagonal choices, exactly two compatible
triangles per complete cell, every normalized node, scale, absolute coordinates,
mask holes, partial256tails and immutable source bytes/mtimes. Full-image decode
is restricted to this independent capped checker; exporter/verify remain tiled.
Local focused32pass/1optional skip; real Linux control33pass. Native VC3D control:
259×258,66821nodes,66302cells,132604faces, maximumXYZerror1.562498255e-7inputunits.
Observed reader+checker RSS120594432bytes; minimum RAMavailable5804302336bytes;
minimum disk92130725888bytes; source artifacts unchanged.
Proof on gpu-1:
`/ssd/draft/tifxyz-reader-control-20261030-t8-acceptance-01/control-03/test_real_vc3d_control0/reader/READER_VERIFIED.json`,
SHA `d476a7514bfce8c47d6924ed6c5b3412ab19a9af7b52a3e5d34b6d4e93b49192`.
The control is not acceptance of the real candidate; invoke the checker on that
candidate only after export READY and the separate source-bound verification pass.

### Automatic continuation of the active run

The read-only pilot follower is running on gpu-1 (PID4085021) under
`/ssd/draft/direct-tifxyz-acceptance-r3b-20261030-4ef592c0/`.
It waits for the frozen controller's successful smoke/export/fresh-verify receipts,
then invokes the pinned real-reader checker in its own new directory. Any missing
or failed gate stops the chain; it never writes inside the export. Only after all
checks pass will it write `ACCEPTANCE.json` with the numerical-ROI scope and the
existing UNMEASURED/physical-false flags. `STATUS.json` is the live progress record.
Follower SHA `3ba07ce5c9a1ed50be1b04ec352842c986b871f12a26b414bcd1a7eca2ebfbaf`;
checker SHA `f8776554c77390908610b9de48e147c4be417713562752c127c19c878904e878`.
Local and remote admission-receipt self-checks passed. Last observation: follower
alive and WAITING_FOR_INDEPENDENT_VERIFICATION; pilot AUDITING at11785.65s,
aggregateRSS443498496bytes; neither READY nor ACCEPTANCE exists yet.
