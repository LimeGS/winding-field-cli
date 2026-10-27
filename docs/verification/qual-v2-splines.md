# Qual V2 and official splines — October 27, 2026

The default pipeline now uses the pinned Qual AFV fibre V2 checkpoint and official
CPU spline generator. Raw V/H probabilities, physical curves and family-labelled
AFV schema v1 exports feed the existing field, mesh and flatten stages. The
numerical `_core` is unchanged. Pins and bundled upstream hashes are in
[`provenance/qual.json`](../../provenance/qual.json).

## Regression and installed-package checks

- Final source suite: **117 passed in 79.38s**.
- Built sdist/wheel; installed wheel tested outside the checkout with `PYTHONPATH`
  removed and pytest's `pythonpath` empty: **117 passed in 80.71s**.
- Installed Python/model JSON files match the source tree; upstream model and
  plans match their provenance hashes. The official generator's installed direct
  URL points to the pinned commit archive.
- Coverage includes channel order, normalization, incompatible CT rejection,
  fixed inference windows, input corruption, interrupted materialization/import,
  child termination, cross-block constraints, short curves, support gaps, source
  coordinate conversion, H/V exports and corruption of later AFV blocks.
- Classical demo/resume and explicit SD2 preparation/download regressions pass
  in both suites. Actual SD2 acceptance from the preceding audit is recorded
  [separately](audit-fixes.md).

Environment: macOS arm64, 16 GiB RAM, Python 3.11.13, four CPU threads. Exact
versions, implementation identity and artifact hashes are in the
[`receipt`](qual-v2-splines-receipt.json). The artifacts retain development version
`0.1.0`; this is verification evidence, not a new published release.

## Independent review and fixes

A fresh-context read-only review of the whole change found four issues. Each was
reproduced as a failing regression and fixed before the final suites and fresh
original-CT acceptance:

1. Crossing spline edges could disappear in small solve blocks. Clipping each
   supported edge and sampling at field pitch now supplies free-node constraints
   in **all 144 intersected blocks** of a 1,296-block, 3 µm-field schedule.
2. Retained upstream batches and phase arrays lacked content checks. A post-child
   SHA-256 inventory now rejects changed batches, arrays, progress and missing seals
   before reuse; tests stop the actual generator after committed batches.
3. Missing import-ledger rows or extra curves/bounds could survive resume. The
   importer now rejects all unaccounted inventory before inserting new curves.
4. Completed workflow resume omitted probability integrity. The final fibre
   receipt now binds that input and both resume entrypoints check it.

No review findings are deferred. Work seals cover STOP, Ctrl-C and handled
failures; a hard kill of the parent before sealing requires a new output directory.
That limitation is explicit rather than silently trusting unverifiable scratch.

## CPU workspace adjustment

Profiling selected PyTorch's `slow_conv3d` on this machine without oneDNN. A
64-channel 3³ convolution's measured im2col allocation at 16³ scales to **13.5 GiB**
at the required 128³ input. The adapter therefore evaluates CPU convolutions in
depth slabs with exact halos, targeting 256 MiB im2col scratch or one output plane.
Whole-window instance normalization, weights and topology are retained.

Two regression cases compare full versus slabbed convolutions, including stride,
dilation and groups, at `atol=rtol=2e-6`. Original-CT preparation then completed
in **20.88s**, with process maximum RSS **3,525,083,136 bytes** (~3.28 GiB).
This measurement is for this small ROI and backend, not a whole-scroll RAM bound.

## Original-CT acceptance

Installed-wheel command, using the SHA-verified checkpoint and shared CT cache:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
winding-field models --path .runtime/qual-v2.pt
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 \
  --field-spacing-um 17 --sweeps 1 --chunk 16 --field-chunk 16 \
  --ct-mode stream --cache /path/to/ct-cache --cache-gib .25 --reserve-gib .25 \
  --threads 4 --model-path .runtime/qual-v2.pt \
  --until prepare --output .runtime/qual-reviewed-17um
winding-field fibres .runtime/qual-reviewed-17um
winding-field resume .runtime/qual-reviewed-17um
```

The tracer was deliberately stopped after its work manifest appeared, before
publishing upstream AFV. The CLI returned `PAUSED` with only preparation committed
and nine scratch files sealed by their content hashes.
Resume completed the entire pipeline, with these results:

| Product | Result |
| --- | --- |
| Observations | 48³, native 8.64 µm, support fraction 0.222177 |
| Imported curves | **8 splines: 5 V, 3 H** |
| Field | **17 µm**, 24³ nodes, **106 fibre-constraint rows** |
| Mesh | **227 faces** |
| Flattening | **12 charts**, all 227 extracted faces covered |

No acquisition fill sentinel was asserted for this dataset. Explicit fill and
unsupported-gap behavior are exercised separately by regressions.

### Resume and storage-layout checks

The materialized probability input's SHA-256 and modification time were identical
before stopping and after resuming the actual tracer. A subsequent completed-run
resume preserved field checkpoint, mesh manifest, flat receipt, fibre receipt,
curve database, AFV, probability-input and upstream-work-seal hashes.

A second installed Qual preparation used storage chunks 32 instead of 16 and
requested a 300 µm field. **All seven map channels were bit-identical** over the
same 48³ ROI: CT, presence, V/H probabilities, density, normals, tangents and
support. This checks storage-layout and field-pitch independence of the canonical
tiled estimator; it does not establish whole-volume inference equivalence.

Local outputs are `.runtime/qual-reviewed-17um/`, `.runtime/qual-reviewed-chunk32/`
and `.runtime/qual-reviewed-checks.json`; logs are retained in
`.runtime/qual-evidence/`. The receipt retains the final acceptance hashes.

## Scientific and scale limits

These checks establish software execution, coordinate handling and reproducible
products. They do not establish correct physical sheet reconstruction or global
convergence: `physical_acceptance=false` and `global_converged=false` remain set.
CUDA and native-resolution whole-scroll throughput were not tested. Upstream seed
sorting needs RAM proportional to seed count; scratch estimates and resource
contracts are documented in [FORMATS.md](../FORMATS.md).
