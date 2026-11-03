# Qual V2 and spline integration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the default real-CT pipeline use Qual's latest public fibre model and official spline tracer, with those splines constraining the field and exported to AFV.

**Architecture:** Retain the numerical core and orchestration. Add a pinned Qual model adapter, retain V/H probabilities in prepared tiles, and run the pinned upstream spline CLI in an interruptible child process. Import its curves into physical-coordinate SQLite storage, index their extents for solver queries, and reuse the existing resumable AFV exporter with family labels.

**Tech stack:** Python 3.11+, PyTorch >=2.6, NumPy, SciPy, SQLite; dynamic-network-architectures 0.4.4; pinned avf-spline-generator (Numba/filelock).

**Spec:** Approved chat decision on 2026-10-27: “Sí, integración completa” — CT → Qual V2 → Qual splines → field → mesh, including AFV export. The concrete adapter decisions below are implementation choices.

## Global constraints

- Independent repository only; preserve the source/experiment checkouts and `_core` numerical implementations.
- Model: `Qualzz20/afv_fiber_9um_v2`, revision `2a8a26a70754a5d90aeca91f9ef8e7130e6e4f81`; checkpoint SHA-256 `b8519a4ce4044c93ed98af0c278628fdcdfdd8808275b3c8becd7211f38b7967` (566,154,112 bytes).
- Generator: `Qualzz/AVF_SPLINE_GENERATOR`, commit `3c8b73628ca56cb28042b9b2e79439b2e1c693d2`. Reuse upstream code, do not reproduce its tracing algorithm.
- Qual input: uint8 CT, isotropic observation spacing 8.64–9.6 µm. Reject incompatible input before downloading weights or preparing data; report how level/stride affect observation spacing.
- Fixed, source-anchored 128³ inference windows: 48³ owned cores and 40-voxel halo, independent of storage chunk and ROI. Use upstream normalization (sample standard deviation, floor 10), softmax segmentation, normalized axial XYZ directions.
- Preserve raw float32 V/H probabilities in that order. Presence is `1 - P(background)`; intersection remains distinct from V/H.
- `--field-spacing-um` remains independent of observation/inference pitch. Existing explicit SD2/classical modes remain selectable; synthetic demo remains classical.
- Upstream traces at twice observation pitch. Its AFV coordinates are `2 * tracing_XYZ + 0.5`, in ROI-local observation voxels. Convert to world micrometres with prepared origin/spacing, then export in declared selected-source voxel coordinates.
- V2 is the model/data generation, not an AFV schema upgrade: the published generator writes AFV SQLite schema v1.
- Preserve strict output ROI, support gaps, stop/resume, bounded reads and cache/output disk admission. New implementation requires new output directories.
- Keep the practical README concise. Execution evidence is distinct from physical accuracy/global convergence.

## Review focus

1. Long splines must affect every intersected field block, even when their starting point lies elsewhere; exercise spatial queries and coarse-field constraints.
2. Nonzero ROI origins, selected levels/stride and half-voxel pooling offsets must round-trip between AFV and physical coordinates.
3. STOP/Ctrl-C during input materialization, upstream tracing or import must retain recoverable committed work and terminate the child.
4. Invalid/support gaps must split constraints and exports; upstream's short low-probability tolerance must not bridge unsupported observations.
5. Immutable prediction inputs, resumed outputs and external code identity must reject changes; disk estimates must include dense generator scratch.

## Task 1: Pinned model and prepared probabilities

**Files:** `models.py`, `predict.py`, `config.py`, `cli.py`, `workflow.py`, bundled Qual model/plans under `_vendor/`, `pyproject.toml`, `provenance/qual.json`, `THIRD_PARTY.md`, `licenses/qual-Apache-2.0.txt`; tests in `tests/test_qual.py` and `tests/test_prepare.py`.

**Interfaces:** Add `QualV2(path=None, device='cpu', offline=False).predict(ct)` returning `(presence_zyx, tangent_zyx_xyz, probabilities_zyx_vh)`. Extend checkpoint selection with an explicit prediction keyword while preserving existing SD2 callers. `prepare` persists `probabilities` beside existing map channels for `qual-v2`.

- [ ] Add failing tests for four-logit softmax/channel order, direction normalization, std floor, wrong checkpoint/dtype, and isotropic pitch/range admission.
- [ ] Add a window-sensitive fake-model regression asserting equal Qual outputs for different storage chunks and overlapping ROIs; retain explicit-invalid-value and independent-field-spacing coverage.
- [ ] Run `python -m pytest tests/test_qual.py tests/test_prepare.py -q`; confirm the new cases fail for missing Qual behavior.
- [ ] Bundle the exact upstream model definition/plans with attribution; add pinned architecture dependency and verified checkpoint download/loading using `weights_only=True`.
- [ ] Implement the adapter, fixed window geometry, probability persistence, CLI default/model selection and shared geometry validation. Keep model identity and normalization in preparation receipts.
- [ ] Run the focused tests; require all pass. Check concurrent checkpoint acquisition for both backends using the existing shared download path.
- [ ] Commit this independently testable model/preparation change after reviewing its diff.

## Task 2: Official spline execution and durable physical curves

**Files:** new `src/winding_field/splines.py`; `fibres.py`, `workflow.py`, `pyproject.toml`, `provenance/qual.json`; tests in `tests/test_splines.py`.

**Interfaces:** `trace_qual(run, config, stop=None)` returns a FIBRES receipt or `{'status':'PAUSED'}`. Dispatch from `fibres.trace` for prepared Qual maps. Curves remain NPY blobs of XYZ µm in `fibres/fibres.sqlite`; new Qual rows include H/V family and an R-tree of physical curve bounds.

- [ ] Add failing tests for tiled `(2,Z,Y,X)` float32 NPY materialization, changed retained input detection, stop/resume without rewriting committed input, and exact coordinate conversion at a nonzero origin.
- [ ] Add failing tests for ordered multi-block AFV import, retained family labels, resumable import and splitting at an unsupported gap (including a segment whose endpoints are supported).
- [ ] Run `python -m pytest tests/test_splines.py -q`; confirm the new tests fail for missing behavior.
- [ ] Depend on the generator via its immutable commit archive URL. Materialize immutable predictions with bounded tile writes, a durable cursor/hash ledger and reserve checks.
- [ ] Run `python -m avf_spline_generator` as a child using CPU tracing, bounded threads, ROI-local observation geometry, and upstream `--resume` when its manifest exists. Capture logs; forward stop via SIGINT and wait/reap the child before returning PAUSED. Validate reuse of a completed intermediate AFV.
- [ ] Stream AFV blocks, validate continuity/geometry, transform points to XYZ µm, split unsupported segments using fine-grid support, and transactionally store curves/families/spatial extents with an import cursor. Publish FIBRES only on completion.
- [ ] Include model assets and external implementation fingerprints in resumability identities. Include native float32 materialization and upstream's approximately 18 bytes/tracing-voxel scratch in `plan` output and disk admission.
- [ ] Run focused tests and a real upstream CPU trace on a small synthetic H/V probability volume. Require nonempty family-labelled curves and unchanged results after resume.
- [ ] Commit the verified adapter.

## Task 3: Long-curve constraints and AFV interchange

**Files:** `field.py`, `exports.py`, `docs/FORMATS.md`; tests in `tests/test_field.py`, `tests/test_exports.py`, `tests/test_splines.py`.

**Interfaces:** Qual constraints query curve extents intersecting the physical solver block, rather than a single preparation-tile tag. Existing legacy curves keep their indexed tile query. AFV export uses the same validated curves as the field, including H/V labels and coordinate provenance.

- [ ] Add a failing regression with a spline crossing multiple observation and field tiles; assert constraints in a block away from its origin, at exact 17 µm and coarse field pitches.
- [ ] Add AFV round-trip tests for H/V family, source-coordinate conversion, long-curve block boundaries, support-gap splitting and interrupted/corrupted exports.
- [ ] Run `python -m pytest tests/test_field.py tests/test_exports.py tests/test_splines.py -q`; confirm the new behavior fails first.
- [ ] Select each overlapping curve once using its spatial index; retain bounded point batches and avoid edges across separated runs. Reuse the numerical constraint builder.
- [ ] Extend the existing resumable AFV exporter for family labels and bounded point blocks, preserving inventory/content validation and declaring selected-source geometry accurately. Keep AFV schema v1.
- [ ] Run the focused suite; require passing spatial-query, coarse-grid and export regressions. Update the format contract and commit.

## Task 4: Installed acceptance and practical documentation

**Files:** `README.md`, `THIRD_PARTY.md`, `docs/verification/qual-v2-splines.md`, compact verification receipt; package metadata as needed.

- [ ] Run the complete source test suite sequentially with four CPU threads.
- [ ] Build/install the wheel in the existing release verification environment; run the suite outside the checkout and verify packaged model assets/dependencies.
- [ ] Download and SHA-verify the actual Qual checkpoint. Run original 175A CT at 8.64 µm observations and exact 17 µm field through splines, field, mesh and flatten; select a bounded ROI large enough to produce actual splines.
- [ ] Stop/resume an actual spline run and verify final input/curve/field/export hashes are stable. Compare actual Qual predictions across two storage layouts on the same small ROI.
- [ ] Verify that the classical demo and explicit SD2 path still work through their relevant focused tests; avoid repeating historical full demonstrations without a new concern.
- [ ] Update quick-start model wording, supported input pitch, spline outputs, checkpoint size, tracing disk cost and direct CLI examples. Record exact evidence and scientific limits.
- [ ] Request whole-change review, fix verified findings, rerun affected checks, and run `git diff --check`.
- [ ] Report implementation, test counts and actual-CT results; leave integration/publication choice explicit for the new feature work.

## Plan self-review

All approved pipeline stages are covered. Task 1 owns model and observation contracts; task 2 owns tracing, immutable inputs and interruption; task 3 owns cross-block field constraints and interchange; task 4 owns installed/original-CT evidence. Review-focus cases map to tasks 2–3. No training, GUI or numerical-core redesign is required.
