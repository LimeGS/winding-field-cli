# Business-logic audit fixes — October 27, 2026

Scope: the ten findings following v0.1.0. Numerical `_core` and pinned `_vendor`
implementations remain unchanged. New runs are required for this implementation;
historical release artifacts and runs retain their original identity.

## Regression evidence

Every new behavior was reproduced as a failing test before implementation.

| Finding | Regression coverage |
| --- | --- |
| Gzip catalog | Decoded response equality and decoded-size budget rejection |
| Independent field pitch | Exact 8.65/17/130/300 µm geometry; anisotropic source; physical analytical field; constraints, extraction and changed-pitch resume |
| Storage-dependent SD2 | Window-dependent model across chunk sizes and overlapping ROIs; actual pinned SD2 comparison below |
| Late validation | Invalid stride/chunk/solver/trace/chart/spacing options fail before source I/O or output creation |
| Wrong disk admission | Cache and output space checked separately; payload admission preserves cache reserve |
| Uninterruptible exports | Stop after one chunk/record, retained progress on resume, changed-input/corruption rejection, stage completion withheld |
| Missing acquisition validity | Explicit fill policy, complete 40-voxel observation-filter footprint, valid zero controls, identity-bound resume |
| Offline failures | Fresh local mirror metadata; cached conflicting metadata rejected; cached catalog with network forbidden |
| Concurrent checkpoints | Two synchronized downloads both succeed, use distinct verified temporary files and leave no temporary residue |
| Repeated map decompression | Joint channel reads and sparse Cartesian sampling load each touched tile exactly once |

Initial full source suite: **72 passed in 48.11s**. Rebuilt sdist/wheel; installed
wheel suite, copied outside the checkout with `pythonpath` unset:
**72 passed in 57.07s**. Python 3.11.13, macOS, CPU, four Torch/OMP threads.

Final acceptance after review and recovery fixes: **80 passed in 47.30s** from
source, **80 passed in 56.09s** from the rebuilt installed wheel outside the
checkout. Artifact and implementation hashes are in
[`audit-fixes-receipt.json`](audit-fixes-receipt.json).

Fresh-context review confirmed three additional cases, all reproduced RED→GREEN:
corrupted retained AFV data/inventory, lost boundary planes when explicitly asking
for native pitch (8.64 or 8.65 µm), and duplicate mesh levels accepted by preflight.
The final original-CT run also encountered actual low-disk admission failure.
That exposed an initial-checkpoint retry bug; atomic publication now prevents an
incomplete directory from blocking resume, with a separate regression.
The final run resumed successfully after space recovered, with its original
reserve and configuration. No review findings remain deferred.

## Original CT and SD2 consistency

Pinned checkpoint SHA: `2528a46e38844c06e591084f1dc1af9442ccb45a17aae9b2a68441e6028bbdd6`.

Original source:
`s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr`.

Native ROI: `1504 1536 4032 4064 6048 6080`. Preparation with storage chunks 16
versus 32 yielded **exactly zero differing values** in CT, presence, density,
normals, tangents and support over the entire 32³ ROI. This measures storage-layout
invariance of the canonical tiled estimator, not equivalence to whole-volume inference.

Live `plan --scroll PHerc0175A --field-spacing-um 300` decoded the public catalog
successfully. It reported native 8.64 µm observations and an exact 300 µm field
with shape ZYX `[368,270,270]` and bounded streaming policy.

## Installed end-to-end demonstrations

| Run | Observations | Field | Fragments | Mesh faces | Flat charts | Face coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Synthetic classical | 4 µm, 24×32×56 | same | 95 | 9,286 | 17 | 100% |
| Original 175A / SD2 | 8.64 µm, 32³ | **17 µm**, 16³ | 25 | 206 | 13 | 100% |

Both final acceptance runs completed all stages and resumed with identical field-checkpoint, mesh
manifest and flat-receipt hashes. No acquisition fill sentinel was asserted for
the real-data run; validity-mask behavior was tested separately with declared fill.

Exact-scale acceptance also ran four installed-package jobs over the same 48³
synthetic layered CT at **8.64 µm observation spacing**:

| Requested pitch | Field shape | Executed through |
| --- | --- | --- |
| 8.65 µm | 47³ | field |
| 17 µm | 24³ | field |
| 130 µm | 4³ | flatten |
| 300 µm | 2³ | flatten |

All four retained the **same prepared-map SHA**:
`00b03cb5cb9a63cf62d7e5d8e2587ea8682dda49938929cc498912275f9cb28f`.
All field nodes were supported and finite, and every field checkpoint was unchanged
after resume. Intermediate `--until field` jobs report `PAUSED` by the existing
workflow convention; their field receipt is `FIELD_READY`.

## Evidence locations and boundaries

Local reproduction scripts and logs: `.runtime/audit/{task1,task2,task3,task4}-*.log`,
`fixed-tiles.json`, `final-exact-scales.json`, `catalog-fixed.json`,
`catalog-offline-fixed.json`, `acceptance-suite.log`, `acceptance-installed-suite.log`.
Final run outputs are `acceptance-synthetic/` and `acceptance-real-17um/` there;
the latter retains the real reserve-failure/resume history. The offline catalog
check explicitly forbade network access after the successful live resolution.

No CUDA or native-resolution whole-scroll throughput test was performed. These
checks establish software execution, coordinates, coverage and reproducibility;
they do not establish physical sheet correctness or global numerical convergence.
Receipts continue to report `physical_acceptance=false`, `global_converged=false`.
