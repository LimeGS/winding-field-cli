# Private field-flattening / CT benchmark

This is an opt-in experiment, **not a production flattening replacement**.
Its protocol compares identical source domains first, then finalists end to end,
with separate per-field holdout and local fine-mesh confirmation.
Both Paris4 and 175A are required. Passing software tests is not a completed
experiment; a final recommendation also needs actual CT review, holdout and
local fine Paris4 confirmation.

## Isolation

- Keep original meshes, UVs, models, inference services and viewer collections unchanged.
- Use a new output directory and a separate environment/cache. Do not run benchmark
  scratch or downloads on the inference GPU host.
- Linux heavy jobs: one cell at a time, `CPUQuota=200%`, `CPUWeight=10`, `Nice=10`,
  `MemoryMax=20G`, `MemorySwapMax=0`; require at least 24 GiB available host RAM.
- Keep 20 GiB free on **both** result and cache filesystems. Initial cache: 8 GiB
  encoded, 256 MiB decoded; cell budget: 60 active minutes and 20 GiB scratch.
- The operational directory for this study is
  `138.201.58.245:/mnt/field-flattening-benchmark-20261115/`.
  Frozen r9b v3 environments are not benchmark environments.

## Inputs

`PROFILES.json` lists explicit `sources`, `methods`, `scopes`, `backends`,
`resources` and the local `protected_inference` status root.
Each source binds `field_id`, `resolution_id`, nominal `W`, `raw_phi`,
`frame`, mesh path, `mesh_manifest_sha256`, stage and its own `ct_profile`.
175A legacy sources additionally bind the saved map, its SHA and `legacy_unit_um`.
See `tests/test_flattening_benchmark.py` for small executable input examples.

Mesh XYZ is converted once to micrometres using the mesh's `voxel_um`:
8.64 for 175A, 9.6 for Paris4. **Do not use field spacing as that multiplier
or reapply an origin already present in the mesh.** Native CT is 8.64 µm
for 175A and 2.4 µm for Paris4. Paris4's mesh frame suffix `:L2` does not
select CT level 2.

CT profiles pin the source URL, level, metadata hashes, spacing, shape and origin;
they also fix overview/ROI pitch and physical depth offsets. Both fields use
the same trilinear sampler and `[0,255]` window. Missing chunks are not observed
zeroes. A transport failure is an error, not missing tissue.

## Commands

Use the isolated environment's Python, with this checkout's `src/` importable:

```sh
python tools/bench_flattening.py inventory --config PROFILES.json --output NEW_RUN
python tools/bench_flattening.py run --run NEW_RUN --stage pilot --resume
python tools/bench_flattening.py verify --run NEW_RUN
python tools/flattening_review.py build --run NEW_RUN --output NEW_RUN/review
python tools/flattening_review.py summarize --run NEW_RUN --votes CT-votes.json --output NEW_RUN/report
```

Run heavy commands in the cgroup above, not an unrestricted SSH shell.
The `run` command requires a `field-flattening-*.slice` whose **aggregate**
CPU/memory/swap limits match that envelope, plus Nice ≥10 and host RAM headroom.
The study uses `field-flattening-benchmark20261115.slice`, linked at runtime to
its unit file in the owned `ops/` directory. Each cell gets its own service in
that slice, so a failed worker cannot leave detached native solvers running.
Only the selected numerical-thread, Python-path and GPU-disable environment
variables are passed into those services; systemd does not inherit them implicitly.
The supervisor enforces the remaining active-time budget even if a worker is
blocked inside one CT tile or native call; each worker also has a systemd runtime
limit as backup. Spent time survives a forced stop, and expired cells remain
`RESOURCE_LIMIT` rather than getting a fresh budget on resume.
Do not change code, profiles or backend versions in an existing frozen run.
Use a new run identity instead.

Real systemd cleanup tests are opt-in: on a privileged Linux qualification host,
set `FLATTENING_SYSTEMD_TESTS=1` and run the `cgroup_kills` and `systemd_deadline`
tests in `tests/test_flattening_benchmark.py`. Ordinary CI skips these two checks;
that skip is not evidence of real cgroup qualification.

The frozen-v4 pilot exposed why cooperative checks alone are insufficient:
one Paris4 W−100 tile took almost three hours before returning to its next check.
Its retained `TIME.json` records about 14,202 active seconds, **not** a successful
one-hour run. Preserve that baseline and its partial CT as resource-limit evidence.
Remaining solver trials move to a new bound run with the external deadline;
reports must retain the original baselines and costs, not silently drop them.
The study's `run-pilot-v8` covers those 64 remaining non-legacy cells and binds
all four v4 baseline outcomes in `RUN.json`'s `config.baseline_reference`.
Its launcher checks that the consumed numerical modules and library versions
still match v4; the newer tools also use the already-versioned seam/ROI corrections.
Keep v4's retirement STOP. Its worker `TIME.json` is not the complete operational
cost: revalidation of already-completed receipts can add time visible only in
the retained service logs.

Spatial patch cuts can create a pinched vertex even when the original component
is manifold. The separately versioned v10 domain policy bisects such patches
further, using geometry only, until they are admissible; every original face is
still owned exactly once and component-wide calibration remains unchanged.
`DOMAINS.json` records the policy and refinement count. It never silently
repartitions a requested whole sheet. Frozen-v8 patch-preparation failures remain
visible as implementation failures, not evidence that a solver's texture is poor.
Corrected patches require a new bound evaluation, not overwriting the active run.

`RUN.json`, `INVENTORY.json`, `SELECTION.json`, `BACKENDS.json` and `QUEUE.json`
record the experiment. Sources, domains, solver outcomes, maps, metrics and CT
tiles have hash-bound receipts. Raw float32 CT stacks, masks, XYZ and source-face
rows remain available; PNGs alone are not the retained scientific product.

Create `RUN/STOP` to pause benchmark work. Cell-local `STOP` also pauses a cell.
Remove only the benchmark-owned stop marker before resuming. Do not delete partial
products or reset failed inference work. Completed tiles/checkpoints are reused
only after verification. A stale protected inference status pauses the benchmark;
it does not restart or modify inference.

### Native CT reference controls

Use a **new versioned evaluation directory** and its pinned code snapshot, not
the active frozen pilot's code:

```sh
python tools/flattening_ct.py references --run FROZEN_RUN --output NEW_CONTROLS
```

First create an operation-owned `FROZEN_RUN/STOP`, then wait for the benchmark
parent and all cell services to exit. Do not stop inference. The reference
command refuses `RUNNING` queue rows and holds the source queue lock, keeping
the two eight-worker download pools from running together. Run it under the
same aggregate resource slice and disk reserve; keep its partial results on error.
Remove only your own unchanged STOP marker and relaunch the original frozen
pilot/environment after controls exit, including on a failed reference attempt.

Each source-fixed ROI gets tangent, XY, XZ and YZ planes at native pitch,
central depth and the same `[0,255]` window. Undefined normals leave the tangent
plane explicitly missing. Retain raw CT, support masks, physical XYZ and
source/code/metadata-bound receipts. These are local diagnostic planes, not
flattened-surface coverage, human preferences or anatomical acceptance.
New ROI policies and corrected seam metrics do not amend old frozen receipts;
reconcile them in explicit evaluation sidecars before final reporting.

Add completed controls to a **new** private gallery snapshot:

```sh
python tools/flattening_review.py build --run FROZEN_RUN --output NEW_GALLERY --controls NEW_CONTROLS
```

The gallery checks the source-run binding and copies only public CT assets and
physical metadata. Reference planes are labelled diagnostics and cannot receive
candidate votes; their central plane stays at zero depth when candidate depth
changes. Save the previous private review key before making another gallery:
the current operator API keeps that key at `FROZEN_RUN/PRIVATE-review-key.json`.
Vote import rechecks the controls receipt too.

When the current run pins `config.baseline_reference`, gallery build also includes
those historical baseline panels, including their resource-limit gaps. It checks
the old run, inventory, queue and specified artifact hashes, plus matching source
and CT-profile identities. Controls from that explicitly pinned baseline run are
allowed only after those checks. The historical runs and method keys stay outside
the served directory, even when their panels are hidden by the active stage.

For the corrected-patch cohort, `config.trial_references` is a list of pinned
origin runs using the same `run`, `run_sha256`, `inventory_sha256`,
`queue_sha256` and `rows` layout as `baseline_reference`. Each explicit trial row
must match its origin's `output`, `case_id`, `field_id`, `resolution_id`, `method`,
`scope`, `track`, `stage` and terminal `status`. Completed trials additionally
require `cell_sha256` and `ct_sha256`; supplied map/metrics hashes are checked too.
Build and vote import revalidate these bindings. Duplicate case/track/scope/method
candidates are rejected rather than silently choosing an attempt or counting it
twice. Imported holdout panels require both the cohort and source freezes.

Keep the corrected cohort's four-source inventory, execute only the three
changed patch cases, and explicitly import v8 sheets and input-equivalent W−100
patches plus the four v4 baselines. Pin v8 only after its queue is stable; do not
alter its active queue to make a reference pass. Domain equivalence and the
patch-policy amendment must be retained as separate hashed evidence: matching
source geometry alone does **not** prove matching solver inputs or recipes.
No automatic transitive imports occur, and old preparation failures remain in
their origin run as diagnostics, not extra competing solver candidates.

## Review and interpretation

Serve only the `review/` directory, bound to localhost or through a private tunnel.
Never serve the run root: `PRIVATE-review-key.json` contains solver identities.
The gallery is separate from the operational ink viewer and never edits annotations.
Download the actual human votes; do not manufacture preferences from numerical metrics.

Coverage uses **all original 3D area**, counting each source face once.
Mapped, geometrically valid/nonambiguous and CT-supported areas are different
quantities. Distortion quantiles are weighted by original 3D area. An overlapping
face is excluded conservatively in full. Report seams and largest continuous piece:
many tiny charts are not equivalent to one readable sheet.

Compare candidates within each mesh and field. Report Paris4 and 175A separately;
an optional coarse aggregate weights the two fields equally. Keep local fine
validation separate from whole-sheet coarse coverage. Geometry remains
`physical_acceptance=false`; a scalar isosurface is not a physically verified sheet.

`COMPLETE` means retained cell outputs exist, **not** good texture or an accepted winner.
`UNAVAILABLE`, `INCOMPATIBLE_GEOMETRY`, `FAILED`, `RESOURCE_LIMIT`, `PAUSED` and
missing inputs must remain visible. No fallback solver may impersonate a missing
method. In particular, do not describe unavailable ABF/BFF/particles as compared.

The current SLIM seed policy requires positive triangles and a simple convex
boundary. ARAP can violate either requirement before SLIM starts. Record such
failures as seed-admission outcomes, not native SLIM convergence or CT-quality
results; nonconvex-boundary rejection is a conservative policy of this recipe,
not proof that SLIM cannot accept a nonconvex start. Do not silently substitute
an initializer or relax the policy within a frozen run.

Before scaling beyond pilots, publish measured costs for each field, inspect the
browser and obtain the protocol's human review. Holdout requires a run/decision-bound
`FREEZE.json`. Track B, local fine confirmation and 16-point coverage sensitivity
are additional gates; the presence of a holdout CLI alone does not satisfy them.
Changing the production default or reprocessing the 120 meshes needs later approval.

## Current evidence

Run-specific execution ledgers, source snapshots, logs, private review keys and
products remain outside the source distribution. Retain them with each run's
hash-bound inputs, backend qualifications, resource decisions and remaining gates.
No final comparative CT conclusion is recorded here.
