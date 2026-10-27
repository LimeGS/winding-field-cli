# Lasagna Observations and External AFV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run official Lasagna for surface normals/density and Qual for fibres, with an optional validated external AFV replacing Qual.

**Architecture:** Keep the CT stage pipeline and numerical core. A bounded, hash-bound bridge runs official Lasagna in a separate Python 3.14 environment; a streaming AFV importer normalizes external curves into the existing world-coordinate spatial store. Preparation keeps surface evidence distinct from fibre evidence, and retained source/model/product identities govern recovery.

**Tech Stack:** Existing Python 3.11/3.12 CLI, NumPy, Torch, SQLite, portable Zarr-v2 reader; pinned official Villa/Lasagna subprocess under Python 3.14.

**Spec:** `docs/superpowers/specs/2026-10-27-lasagna-afv-inputs-design.md`

## Global Constraints

- Lasagna supplies surface normals and target winding density; Qual V2 supplies fibres/splines unless an external AFV is selected.
- Use `scrollprize/lasagna/model_current.pt`, revision `de45f01147adea9f4ad0f621b1ccd885a1f03760`, SHA-256 `4cc0c40a846b296769b9f7982470559078662a1a0b0e1f74ff41ed9341c0d600`, 558,287,587 bytes.
- Official Villa revision: `b29f310957c7b29e7d332f9876e0e487819cfaa3`; its Lasagna distribution requires Python 3.14. Preserve the CLI's Python 3.11/3.12 support.
- Preserve strict output ROI, unsupported acquisition gaps, local-first bounded I/O, disk reserve, independent field spacing and integrity-bound resume.
- Source/model/geometry changes must not inherit previous receipts. Imported AFVs remain read-only.
- Keep classical offline demo and explicit SD2 selection. Do not replace numerical solver algorithms or add dependencies merely to implement JSON/SQLite/process handling.
- No new universal-resolution or scientific-accuracy claims; real upstream execution is a required integration check.

## Review Focus

- An AFV has plausible dimensions but refers to another source level/origin: require matching embedded geometry or an explicit transform (Task 2).
- A sparse AFV edge enters/exits the ROI with endpoints outside or crosses a support gap: retain supported portions without inventing bridges (Task 2).
- Missing Qual predictions or an external AFV must not suppress valid Lasagna surface support or trigger a Qual download (Task 3).
- Lasagna output grid spacing differs from model input spacing: apply density decoding and physical scaling exactly once (Task 1).
- A stopped/completed run retains changed upstream files or a changed external AFV: reject reuse before skipping its stage (Tasks 1–3).

## File Map

- `src/winding_field/lasagna.py` (new): upstream runtime preflight, bounded CT staging, official inference, product decoding and retained-work validation.
- `src/winding_field/afv.py` (new): AFV input identity, coordinate metadata and streaming import into the existing fibre store.
- `models.py`: Lasagna pinned weights through the existing acquisition helper.
- `cli.py`, `config.py`, `workflow.py`: input options, preflight/estimates, stage dispatch and integrity.
- `predict.py`, `fibres.py`, `splines.py`, `field.py`, `meshes.py`, `exports.py`: source-specific observation/fibre semantics and existing downstream consumers.
- `tests/test_lasagna.py`, `tests/test_afv_import.py` (new), connected preparation/export/CLI/workflow tests: behavioral regressions.
- `provenance/lasagna.json`, `README.md`, `docs/LASAGNA.md`, `docs/FORMATS.md`, packaging metadata: installation, pinned provenance and supported contracts.

### Task 1: Official Lasagna Runtime and Observation Bridge

**Files:** Create `src/winding_field/lasagna.py`, `tests/test_lasagna.py`, `provenance/lasagna.json`; modify `src/winding_field/models.py`, `pyproject.toml`, `MANIFEST.in`, `THIRD_PARTY.md`.

**Interfaces:**
- `models.model_sha('lasagna')` and `models.checkpoint(path=None, offline=False, prediction='lasagna')` use the pinned public file.
- `lasagna.runtime_identity(python=None) -> dict`: default interpreter `~/.cache/winding-field/lasagna-venv/bin/python`; report validated upstream code and package identity. Missing/incompatible runtimes fail before a CT output is created.
- `lasagna.prepare(volume, run, config, geometry, stop=None) -> dict`: return `LASAGNA_READY` or `PAUSED`; persist `LASAGNA.json` and bounded staging/output under `run/lasagna-source/`.
- `lasagna.decode_observations(grad_mag, nx, ny, cos, *, input_spacing_xyz, encode_scale, grad_mag_factor, validity) -> dict`: arrays `density`, `normals`, `support`, `surface_phase`, with physical XYZ normals and density.
- `lasagna.read_observations(run, geometry, lo, hi) -> dict`: read a requested observation-grid crop from frozen outputs; honor actual source/output transforms.
- `lasagna.validate_retained(run, config) -> None`: validate source/model/runtime/product bindings, including completed-stage reuse.

- [ ] **1. Write `test_pinned_checkpoint`, `test_missing_runtime_preflight` and `test_density_uses_input_pitch_once`.** Assert the exact model hash; missing offline weights fail; a nonexistent runtime leaves no output. Decode an X-normal with raw magnitude 100, encode scale 1000 and isotropic 10 µm input to density 0.01 µm⁻¹; changing only the stored output-grid pitch must not rescale it again. Nonfinite/unsupported values cannot become valid observations. The numerical assertion is `np.testing.assert_allclose(result['density'][result['support']], 0.01)`.
- [ ] **2. Run RED:** `python -m pytest tests/test_lasagna.py -q`; failures must identify the missing bridge/Lasagna model path rather than a fixture/setup failure.
- [ ] **3. Implement acquisition and runtime preflight.** Reuse atomic download/hash verification. Invoke the official module with the selected external Python, verify pinned upstream provenance and fingerprint its effective implementation/dependencies. Do not import Python-3.14 Lasagna into the CLI interpreter.
- [ ] **4. Implement bounded staging and official execution.** Stage only declared input context via `Volume` reads into a local Zarr-v2 input with explicit geometry; commit verified chunks. Call the official predict3d entrypoint using the real checkpoint metadata and documented options. Include staged CT, output and upstream scratch in estimates. Use bounded process supervision and reap interrupted workers.
- [ ] **5. Add and implement recovery tests.** STOP during staging and upstream execution must resume committed work; changed staged/output chunks must fail, including after `LASAGNA_READY`. Reuse existing atomic writes/ledgers rather than introducing a general job framework.
- [ ] **6. Implement product sampling and decoding.** Read actual upstream manifests, reject inconsistent paths/grids, preserve normal encoding and density factors. Convert index-space gradient to physical space using inverse input spacing and renormalize normals. Record raw products and transforms. Define `surface_phase` from the actual Lasagna cos output, with its documented interpretation rather than calling it a fibre probability.
- [ ] **7. Run GREEN:** `python -m pytest tests/test_lasagna.py tests/test_core_dependency.py -q`.
- [ ] **8. Verify the real upstream boundary.** In external scratch, install the pinned official runtime with Python 3.14, acquire the verified public checkpoint, and run a bounded CT crop. Compare bridge decoding with official exported values. Record actual compatible tile/context settings; if the public checkpoint/runtime do not work together, stop and resolve that blocker before integrating a mock-only path.
- [ ] **9. Commit:** `git add src/winding_field/lasagna.py src/winding_field/models.py tests/test_lasagna.py provenance/lasagna.json pyproject.toml MANIFEST.in THIRD_PARTY.md` then `git commit -m "feat: integrate pinned official Lasagna observations"`.

### Task 2: Streaming External AFV Import

**Files:** Create `src/winding_field/afv.py`, `tests/test_afv_import.py`; modify `src/winding_field/splines.py` only where a shared primitive is genuinely reused.

**Interfaces:**
- `afv.source_identity(path, geometry_path, source) -> dict`: read-only input/descriptor hashes plus resolved XYZ-to-world transform; `source` is the CT `Volume.identity`.
- Explicit descriptor for ambiguous files: `{"source": <exact CT source>, "origin_xyz_um": [x,y,z], "spacing_xyz_um": [sx,sy,sz]}`. The transform declares `world = origin + AFV_XYZ * spacing`; use `[1,1,1]` and zero origin for already-world-µm coordinates. Source identity and descriptor bytes are receipt-bound.
- `afv.import_fibres(run, config, stop=None) -> dict`: use `config['fibres']` and optional `config['fibres_geometry']`; publish `FIBRES_READY`/`PAUSED`, the existing world-coordinate `fibres.sqlite`, a spatial index and import ledger. Receipt includes original input hash, transform, source IDs, counts and exclusions.
- `afv.validate_retained(run, config) -> None`: recheck frozen external input/descriptor and normalized products.

- [ ] **1. Write `test_import_preserves_physical_frame` and parametrized `test_invalid_afv_input`.** A point `[1,2,3]` with origin `[100,200,300]` and spacing `[9,9,9]` becomes `[109,218,327]` exactly once: `np.testing.assert_allclose(imported_point, [109,218,327])`. Test current CLI-export frame metadata, selected nonzero source level, Qual ROI-local data with explicit geometry, unknown family, missing geometry and conflicting metadata. Reject source mismatch, incomplete databases, nonfinite points, discontinuous blocks and live uncheckpointed journals whose contents are absent from the input hash.
- [ ] **2. Run RED:** `python -m pytest tests/test_afv_import.py -q`.
- [ ] **3. Implement geometry resolution and bounded validation.** Prefer complete embedded frame metadata. A Qual header with only shape/voxel size is insufficient to guess a cropped origin; require the explicit descriptor. Validate SQLite application/schema/completion and stream blocks rather than loading the archive. Keep the source read-only.
- [ ] **4. Add edge/support tests.** A segment whose endpoints both lie outside opposite ROI faces must retain its intersection. A long sparse segment crossing an unsupported interval must split into supported pieces; a boundary or missing region must never become an anchor. Include a multi-block curve exceeding one AFV payload.
- [ ] **5. Implement bounded normalization and import.** Transform, clip and resample only as needed for support checks. Keep source lineage and family labels, including unknown. Reuse existing world-coordinate storage and R-tree conventions; commit bounded batches with enough source progress to resume without duplicating constraints. Do not retrace/refit external fibres.
- [ ] **6. Add and implement `test_import_resume_preserves_inventory` and `test_import_rejects_changed_inputs`.** Stop after one committed batch, resume without duplicates, and verify byte-preserving completed resume. Mutation of the source (including during import), descriptor, imported payload or unaccounted ledger inventory must fail; compare source identity before and after reading.
- [ ] **7. Run GREEN:** `python -m pytest tests/test_afv_import.py tests/test_splines.py -q`.
- [ ] **8. Commit:** `git add src/winding_field/afv.py src/winding_field/splines.py tests/test_afv_import.py` then `git commit -m "feat: import calibrated external AFV fibres"`.

### Task 3: Wire Independent Surface and Fibre Sources Through the CT Pipeline

**Files:** Modify `src/winding_field/cli.py`, `config.py`, `workflow.py`, `predict.py`, `fibres.py`, `field.py`, `meshes.py`, `exports.py`; create `tests/test_ct_sources.py`; extend `tests/test_prepare.py`, `tests/test_cli.py`, `tests/test_exports.py`, and Tasks 1/2 tests.

**Interfaces:**
- `run`/`plan`: `--fibres PATH`, `--fibres-geometry JSON`, `--lasagna-python PATH`, `--lasagna-model-path PATH`; normalize path options like existing model/cache paths. `models --prediction lasagna` prefetches the pinned checkpoint.
- Effective sources: default Qual fibres + Lasagna surfaces; explicit SD2 fibres + Lasagna surfaces; explicit classical remains the offline/local baseline. External AFV + Lasagna bypasses all fibre model inference/tracing. Reject an orphan geometry descriptor or an explicitly conflicting fibre model path.
- Maps keep `density`, `normals`, `support` for surfaces; store `surface_phase` separately from fibre `presence`, `tangents` and V/H `probabilities`. External-AFV maps do not invent measured fibre probabilities/tangents.
- `PREPARED.json` records effective surface/fibre methods, geometries and Lasagna receipt identity. `FIBRES.json` records whether curves were traced or imported. Spatial constraint selection uses the normalized-store capability for both Qual and external AFV.

- [ ] **1. Write `test_surface_support_independent_of_fibres`, `test_external_afv_bypasses_qual` and `test_external_plan_storage`.** Stub only the expensive official inference boundary. Assert Lasagna supplies the solver target despite deliberately contradictory Qual presence. Make Qual acquisition/construction/tracing raise if invoked with an AFV. Verify `assert plan['estimated_spline_dense_bytes'] == 0` for external inputs and assert positive Lasagna staging/workspace estimates.
- [ ] **2. Run RED:** `python -m pytest tests/test_ct_sources.py tests/test_exports.py tests/test_cli.py -q`; confirm the new source-selection assertions fail for the expected reason.
- [ ] **3. Wire preflight and preparation.** Check Lasagna runtime/model/source settings before output creation; run Task 1 preparation and combine its products with independently computed fibre predictions only when needed. Apply acquisition validity separately and retain Lasagna surface support where Qual presence is zero. Apply Qual-specific sampling validation only when Qual is actually used.
- [ ] **4. Wire fibre dispatch and field consumers.** Route external AFV through Task 2 before Qual dispatch. Use the global spatial curve query/clipping path for both sources. Keep classical/SD2 streamline behavior and scalar solver numerics intact.
- [ ] **5. Add and implement mesh/export tests.** Automatic mesh phase uses Lasagna `surface_phase` for the Lasagna route; classical keeps its existing presence rule. Export a real surface `grad_mag` array with encoding/units consistent with the manifest; verify anisotropic index/world conversion instead of multiplying by an arbitrary output pitch. Imported AFV exports preserve source lineage and never claim `automatically_traced=True`.
- [ ] **6. Extend recovery.** Revalidate Lasagna and AFV source identities before skipping completed stages. Include relevant products in snapshot ledgers. Tests mutate each retained input after completion and verify rejection. Confirm old receipts remain untouched.
- [ ] **7. Run connected GREEN tests:** `python -m pytest tests/test_ct_sources.py tests/test_prepare.py tests/test_afv_import.py tests/test_lasagna.py tests/test_splines.py tests/test_field.py tests/test_exports.py tests/test_cli.py -q`. Exercise default, external, explicit SD2 and classical/demo selection through public CLI parsing, including invalid option combinations.
- [ ] **8. Commit:** stage only the modified source/tests, then `git commit -m "feat: use Lasagna surfaces with Qual or imported fibres"`.

### Task 4: Public Installation, Documentation and End-to-End Acceptance

**Files:** Create `docs/LASAGNA.md`, `docs/verification/lasagna-afv-inputs.md`; update `README.md`, `docs/FORMATS.md`, `THIRD_PARTY.md`, packaging/provenance as needed; retain large run evidence outside tracked source.

**Interfaces:** Document the exact public flags and one-time isolated runtime setup used in the verified integration. Existing `winding-field run`, `plan`, stage and resume commands remain the user-facing entrypoints.

- [ ] **1. Document reproducible setup.** Provide Python-3.14 venv creation at the default runtime location, installation from the exact Villa archive/subdirectory, checkpoint prefetch, `doctor`/preflight, offline use and local interpreter override. Describe the public checkpoint as a default candidate, not a guarantee for every scroll.
- [ ] **2. Document both source paths.** Include one normal CT command and one `--fibres` example with explicit geometry where needed. Explain density units, unsigned normals versus global orientation, raw products, support, imported lineage and immutable resume. Ensure every referenced file/flag exists.
- [ ] **3. Run the complete CPU suite:** `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m pytest -q`. All tests must pass; add no network/model downloads to ordinary tests.
- [ ] **4. Build and test installation.** Run `python -m build`, install the wheel and pinned core into a clean environment, invoke CLI help/demo and exercise installed tests outside the checkout with `-o pythonpath=`. Validate package resources and provenance are included.
- [ ] **5. Run real-CT acceptance.** Use a bounded 175A ROI with official Lasagna weights and Qual, then import its AFV in a separate run with the same Lasagna settings. Verify geometry, source/family preservation and field/mesh availability; do not require byte-identical AFV container metadata across imports. Exercise STOP/resume and mutation rejection. Record commands, exact revisions, platform, shapes, counts, hashes and measured disk/runtime.
- [ ] **6. Review the whole change.** Request a fresh review focused on units/transforms, unsupported gaps, real official-model use, boundedness, runtime compatibility and both resume paths. Resolve findings with regression checks.
- [ ] **7. Commit documentation/evidence and summarize verified behavior.** No push, merge or package release is implied by implementation approval.

## Plan Self-Review

- [x] Every approved spec section maps to one of the four tasks.
- [x] Cross-task function names, flags and product responsibilities are defined above.
- [x] Each Review Focus condition has a behavioral test in its owning task.
- [x] Runtime/checkpoint compatibility is checked before full pipeline integration.
- [x] Scientific comparison and general cross-run caching are not implementation claims.
- [x] User reviewed the plan and selected native execution.
