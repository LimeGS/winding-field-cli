# Original global CT recipe implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose the exact global mapping, CT raster and TIFXYZ recipe used in the November 3 fleet through the installed CLI, then compare coarse and corrected-fine meshes at matching nominal W levels.

**Architecture:** Select the recipe with `--flattening-method original`, independently of coarse/fine field resolution. Share the approved numerical implementation between CLI and reference comparison rather than invoking campaign scripts or SSH from the package. Keep mesh extraction, mapping and rendering provenance separate.

**Tech Stack:** Existing Python/NumPy/SciPy/Pillow, pinned libigl and tifffile optional dependencies.

**Spec:** User approval in this session: the recipe that produced the 120 PNG/mask/TIFXYZ bundles, not the experimental replacement pins. This document records the implementation choices below for review.

## Global constraints

- Preserve all existing uncommitted work and historical outputs. No commits/pushes or global installs requested.
- Preserve largest-area source-component selection, curvature cuts -0.1/-0.05/-0.02 until genus zero, original boundary-loop two-pin LSCM, global edge-length calibration and Z orientation.
- Preserve anisotropy <=1.25, metric scale 0.5..2, raster overlap/ambiguity semantics and source-face denominators.
- Preserve CT frame conversion and metadata binding, normal offset zero, window 0..180 and existing 4096-square canvas policy. Never hardcode 175A coordinates into general CLI input handling.
- PNG mask: 0 unmapped, 1 mapped/CT missing, 2 CT observed, 3 ambiguous. TIFXYZ validity is separate, with explicit exclusions and existing float32/sentinel/scale contract.
- New output must reproduce both image and mask; a matching map or hash count alone is insufficient.
- W26 remains a reported degenerate-geometry failure; no implicit face removal or experimental anchor substitutions.
- Native meshes require a provenance audit before reuse; receipt completion is not proof of physical correctness.

## Review focus

- L0/physical-micron/pyramid transforms must not silently change CT sample locations.
- Missing CT and ambiguous UV overlap must retain distinct mask states.
- Existing run configuration must not silently switch recipe on resume.
- Global fine meshes may exceed solver RAM; admission must precede allocation, without hidden decimation or fallback to packed charts.
- Reference mismatches and partial outputs must fail visibly and preserve evidence.

## Task 1: Audit reference and native provenance

**Files:** `docs/verification/original-global-recipe.json` (new evidence).

- [ ] Verify hashes and dependency closure of frozen whole-code-05 mapping/raster and the fleet TIFXYZ adapter.
- [ ] Inspect native job config, frozen extraction source, field descriptor, gain and support policy; compare with coarse coefficient provenance.
- [ ] Check existence/sizes of all 121 native artifacts against INVENTORY.json, then verify pilot artifact hashes. Record exactly which checks were performed.
- [ ] Select W-30 and W+31 pilots; record their original/reference input hashes and resource estimates. Do not reextract all levels before the audit.

## Task 2: Package the original numerical recipe

**Files:** `src/winding_field/global_recipe.py`, `tests/test_global_recipe.py`, `provenance/original-global-recipe.json`, `THIRD_PARTY.md`.

**Interfaces:** `make_map(vertices, faces)` returns original world/faces/UV/display and source mappings/report. `raster_sheet(vertices, faces, uv, eligible, sample, canvas, *, window, guard, capture)` preserves original results; capture exposes XYZ ownership before release for TIFXYZ export.

- [ ] Add failing tests `test_global_recipe_reference_map`, `test_global_recipe_observed_missing_ambiguous`, `test_global_recipe_rejects_degenerate_domain`, `test_capture_preserves_image_and_mask`.
- [ ] Run `.venv/bin/python -m pytest tests/test_global_recipe.py -q` and establish failure before implementation.
- [ ] Port only the required approved functions and dependencies, preserving numerical operation order and attribution; reuse existing compatible package primitives only when equality is demonstrated.
- [ ] Run the tests and a frozen historical map/image/mask comparison on a worker with pinned dependencies.

## Task 3: CLI integration and resumable outputs

**Files:** `src/winding_field/cli.py`, `config.py`, `workflow.py`, new `global_pipeline.py`, `tests/test_global_recipe_cli.py`, `README.md`, `docs/FORMATS.md`.

**Interface:** `--flattening-method original` routes run/mesh data into original global mapping and CT rendering; `--export-tifxyz` enables matching mesh-derived TIFXYZ. `export-sheets` accepts supported field descriptors or immutable mesh bundles. Existing saved configurations retain their branch.

- [ ] Add failing tests for global branch selection, CT coordinate conversion, immutable resume recipe, early resource rejection, and failure when reference image or mask differs.
- [ ] Integrate source component reads, map receipt, CT adapter, rendering, TIFXYZ, atomic output publication and hashes using existing storage/runtime primitives.
- [ ] Include algorithm identity, source field and mesh identity, coverage/exclusion counts and physical_acceptance=false in receipts.
- [ ] Document the runnable CLI invocation and optional dependencies; make no assertion that global fixes the observed negative-level distortions.
- [ ] Run focused new tests plus existing CLI/config/workflow tests affected by branch handling.

## Task 4: Coarse/fine comparison

**Files:** `docs/verification/original-global-recipe.json` and isolated remote experiment outputs.

- [ ] Prove installed CLI exact image AND mask reproduction on the coarse control, with independently read-back TIFXYZ.
- [ ] Run the approved mapping/render recipe on same-W native pilots with measured RAM/runtime admission. Record any failure rather than substituting another algorithm.
- [ ] Compare source area, selected/mapped area, distortion, UV inversions, CT coverage, geometry differences and viewable CT images at declared sampling/pitch.
- [ ] If pilots fit and provenance matches, process the remaining matching levels with unique ownership and verified resume; reextract only artifacts whose extraction is shown incompatible.
- [ ] Deliver side-by-side named coarse/fine PNGs, receipts and explicit failures. Keep heavy native geometry remote; local disk is nearly full after ZIP delivery.

## Progress

- [x] User confirmed the November 3 fleet recipe.
- [x] Inspected CLI/config/workflow and existing uncommitted changes.
- [x] Read native completion receipt and 121-entry inventory; full payload validation pending.
- [ ] Plan reviewed and implementation started.
