# Full CLI Docker Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver reproducible CPU and CUDA x86-64 images covering the existing CLI workflows.

**Architecture:** One shared Dockerfile and Bake targets package two isolated Python environments and a verified official Villa runtime. Existing command dispatch and checkpoint contracts remain authoritative; the only CLI extension is explicit selection of the newly qualified native recipe.

**Tech Stack:** Docker BuildKit/Bake, Ubuntu 24.04, uv 0.9.17, Python 3.12.12/3.14.0, PyTorch 2.10.0 CPU/cu128, CUDA 12.8.1, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-docker-runtime.md`

## Global Constraints

- Platform: `linux/amd64`; CLI Python 3.12.12 and Lasagna Python 3.14.0.
- Preserve official archive provenance, existing source/model pins and calibration.
- Preserve cgroup v2 `memory.max <=21GiB`, `memory.swap.max <=64GiB`, and disk reserves.
- Native candidate `historical` (official July archive fc25b4d) uses the three exact SHA256 values in the spec.
- Old defaults and binary-bound checkpoints retain their meaning.
- No claim of CUDA execution without a real GPU run.

## Review Focus

- A mounted model directory must not obscure the separate Lasagna environment.
- An arbitrary host UID must write mounted outputs without runtime root setup.
- A replacement binary or modified source must fail before checkpoint reuse.
- Container stop/replacement must retain durable outputs and resume identity.
- CPU/CUDA locks must not accidentally resolve Torch from the other index.

---

### Task 1: Qualify and expose the official native recipe

**Files:**
- Create: `provenance/flatboi.json`, `tests/test_official_flatboi.py`.
- Modify: `src/winding_field/{patch_pipeline,flattening_matrix,matrix_pipeline,constant_sheet,sheet_finish}.py`.
- Modify: `tests/test_flattening_matrix.py`, `tests/test_constant_sheet.py`.
- Modify: `docs/FLATTENING.md`, `docs/CONSTANT-SHEET.md`, `pyproject.toml` provenance data files.

**Interfaces:**
- Consumes: existing `check_flatboi(app_run, recipe)`, checkpoint runner and CLI.
- Produces: checkpoint qualification of existing recipe `historical`; optional matrix `native_recipe`; constant-sheet `--native-recipe` on flatten/import. Defaults remain `installed-2f89bfc9`.

- [x] Add failing tests for explicit recipe selection, unknown/wrong-binary rejection, matrix forwarding, constant-sheet import forwarding, and checkpoint-bound validation. Historical recipes/defaults must retain their exact hashes.
- [x] Run those tests and confirm missing selection/recipe causes failure.
- [x] Preserve the existing recipe hashes and pass the selected recipe through existing callers; restrict checkpointed routes to the installed and qualified historical recipes.
- [x] Extract the verified official AppImage with all bundled libraries; run a real positive-UV mesh with the unchanged checkpoint runner under bounded Linux x86-64. Check warm-start energy, source preservation, checkpoint continuation and changed-binary rejection.
- [x] Run real SLIM patches/sheet qualification using the existing parametrized matrix tests, then constant-sheet prepare/flatten/validate. Only retain the new recipe if these pass.
- [x] Run affected tests, record qualification evidence, update command examples and packaged provenance, and commit this deliverable.

### Task 2: Lock and build both installed runtimes

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-bake.hcl`.
- Create: `docker/{cli,lasagna}.in`, `docker/{cli,lasagna}-{cpu,cu128}.txt`, `docker/build-constraints.txt`.
- Create: `docker/verify-runtime.py`.

**Interfaces:**
- Consumes: repository source pins and `provenance/flatboi.json` from Task 1.
- Produces: Bake targets `cpu` and `cuda`, images `winding-field:cpu` and `winding-field:cuda`, entrypoint `winding-field`, paths from the spec.

- [x] Add a runtime self-check that imports installed optional dependencies, verifies official native hashes and Lasagna identity, checks Torch build flavor in both environments, and asserts no distribution comes from an editable checkout. Run it in a bare base to confirm absent dependencies fail.
- [x] Resolve four platform-specific exact dependency locks with hashes; pin source archives and native-reader revision, retaining the original Lasagna direct URL. Freeze build dependencies separately.
- [x] Implement one Dockerfile with digest-pinned base/uv stages, a frozen Ubuntu snapshot, verified AppImage extraction, locked isolated Python installs, installed wheel and provenance inventories.
- [x] Configure non-root execution and persistent mounts; link the default Lasagna path outside the models mount. Build explicit TIFXYZ accelerators using the existing build module and numerical flags.
- [x] Add Bake targets selecting CPU/cu128 locks and matching CPU/CUDA bases; reject unsupported architectures.
- [x] Build `docker buildx bake cpu --load` and `docker buildx bake cuda --load`; run self-checks and dependency checks in each image without source mounts.
- [x] Run the CPU image with a host UID override and mounted models/output/scratch; verify Lasagna is still discoverable and outputs are writable. Commit the passing packaging deliverable.

### Task 3: Verify workflows and document operation

**Files:**
- Create: `docker/smoke.py`, `docs/DOCKER.md`, `.github/workflows/docker.yml`.
- Modify: `README.md` install entry and appropriate existing native integration tests.

**Interfaces:**
- Consumes: installed images, native recipe, paths and inventories from Tasks 1–2.
- Produces: executable small-fixture smoke checks and documented reproducible build/run commands.

- [x] Add a small-fixture smoke script for complete classical demo, ROI field production and checkpoint-preserving resume. Reuse the installed integration suite for custom fibres/Lasagna inputs, reused fields and level/range extraction rather than duplicating those fixtures in the smoke script.
- [x] Exercise all four solver/scope combinations, original/patch routes and constant-sheet/atlas/TIFXYZ commands; verify source IDs, UV validity, calibration and retained checkpoint identity. Reuse existing integration checks instead of duplicating algorithms.
- [x] Run model-backed Lasagna/Qual inference on a small supported fixture; record weights and runtime identities separately from offline checks.
- [x] Stop a bounded job, replace the container and resume using the same mounts/image. Assert durable checkpoint hashes remain valid; changed native binary must fail reuse.
- [x] Add CI CPU image build and offline/native smoke checks with explicit cgroup limits; provide a CUDA build job and a separately invoked GPU smoke check. Pin third-party actions to immutable revisions.
- [x] Document build targets, model acquisition, UID ownership, mounts, cgroups, CPU/CUDA device selection, all workflow routes, new recipe selection, resume and exact verification limits. Link from README.
- [x] Run the installed affected suite and container smoke checks, inspect `git diff --check`, and record remaining hardware-dependent checks. Commit the verified deliverable.

## Execution record

- Existing isolated worktree: `feat/docker-runtime`, based on CI fix `3ec2f44` (PR #11).
- Hindsight unavailable (`ECONNREFUSED`); no initiative page was created.
- Five official Flatboi artifacts inspected; user approved qualification of a new official recipe.
- x86-64 Ubuntu executes under local emulation; official AppRun additionally needs host `libGLX.so.0` in a minimal Ubuntu image.
- GitHub PR #11: all four push/PR Linux/macOS checks passed; rechecked after final Docker builds.
- Task 2 packaging committed as `f9847e2`; final CPU/CUDA installed suites each passed 1785 tests with 7 skips, plus the opt-in official Lasagna writer passed separately in both images.
- Final offline CPU/GPU model inference and SIGTERM/container replacement passed; exact image IDs, qualification coverage and limitations are recorded in `docs/DOCKER.md`.
- The optional external July `vc_tifxyz2obj` reader control failed (66,305/66,821 valid nodes retained). Its geometric checks remain intact; this external converter is not qualified. Four external W50 controls were unavailable; the macOS-only case does not apply to these images. Container CI has not yet run on GitHub.
