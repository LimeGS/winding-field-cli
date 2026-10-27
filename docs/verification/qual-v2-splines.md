# Qual V2 and official splines — October 27, 2026

The default pipeline now uses the pinned Qual AFV fibre V2 checkpoint and official
CPU spline generator. Raw V/H probabilities, physical curves and family-labelled
AFV schema v1 exports feed the existing field, mesh and flatten stages. The
numerical `_core` is unchanged. Pins and bundled upstream hashes are in
[`provenance/qual.json`](../../provenance/qual.json).

## Regression and installed-package checks

- Source suite: **107 passed in 66.62s**.
- Built sdist/wheel; installed wheel tested outside the checkout with `PYTHONPATH`
  removed and pytest's `pythonpath` empty: **107 passed in 66.87s**.
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

## CPU workspace adjustment

Profiling selected PyTorch's `slow_conv3d` on this machine without oneDNN. A
64-channel 3³ convolution's measured im2col allocation at 16³ scales to **13.5 GiB**
at the required 128³ input. The adapter therefore evaluates CPU convolutions in
depth slabs with exact halos, targeting 256 MiB im2col scratch or one output plane.
Whole-window instance normalization, weights and topology are retained.

Two regression cases compare full versus slabbed convolutions, including stride,
dilation and groups, at `atol=rtol=2e-6`. Original-CT preparation then completed
in **17.78s**, with process maximum RSS **3,271,458,816 bytes** (~3.05 GiB).
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
  --until prepare --output .runtime/qual-real-17um
winding-field fibres .runtime/qual-real-17um
winding-field resume .runtime/qual-real-17um
```

The tracer was deliberately stopped after its work manifest appeared, before
publishing upstream AFV. The CLI returned `PAUSED` with only preparation committed.
Resume completed the entire pipeline, with these results:

| Product | Result |
| --- | --- |
| Observations | 48³, native 8.64 µm, support fraction 0.222177 |
| Imported curves | **8 splines: 5 V, 3 H** |
| Field | **17 µm**, 24³ nodes, **26 fibre-constraint rows** |
| Mesh | **222 faces** |
| Flattening | **13 charts**, all 222 extracted faces covered |

No acquisition fill sentinel was asserted for this dataset. Explicit fill and
unsupported-gap behavior are exercised separately by regressions.

### Resume and storage-layout checks

The materialized probability input's SHA-256 and modification time were identical
before stopping and after resuming the actual tracer. A subsequent completed-run
resume preserved field checkpoint, mesh manifest, flat receipt, fibre receipt,
curve database, AFV and probability-input hashes.

A second installed Qual preparation used storage chunks 32 instead of 16 and
requested a 300 µm field. **All seven map channels were bit-identical** over the
same 48³ ROI: CT, presence, V/H probabilities, density, normals, tangents and
support. This checks storage-layout and field-pitch independence of the canonical
tiled estimator; it does not establish whole-volume inference equivalence.

Local outputs are `.runtime/qual-real-17um/`, `.runtime/qual-real-chunk32/` and
`.runtime/qual-real-checks.json`. The receipt retains their acceptance hashes.

## Scientific and scale limits

These checks establish software execution, coordinate handling and reproducible
products. They do not establish correct physical sheet reconstruction or global
convergence: `physical_acceptance=false` and `global_converged=false` remain set.
CUDA and native-resolution whole-scroll throughput were not tested. Upstream seed
sorting needs RAM proportional to seed count; scratch estimates and resource
contracts are documented in [FORMATS.md](../FORMATS.md).
