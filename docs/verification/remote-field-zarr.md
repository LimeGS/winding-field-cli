# Remote field Zarr v2 verification — November 10, 2026

## Scope

Read-only consumption of `winding-field-zarr-v1` exports through `FieldSource`;
existing prepared mesh/surface, flatten/export and UVW field consumers share the
reader. No numerical-core, dependency, Studio or published-data changes.

The source example is `examples/paris4-remote.json`, pinned to HF revision
`6753c9812ec8c455401da69b4e758769e565e39f`. Scientific acceptance remains false.

## Evidence collected

- Final pre-commit focused run: **257 passed, 2 opt-in skipped** in 138.05 s
  (`test_remote_field`, `test_surface_sources`, `test_surface_workflow`,
  `test_surface_parallel`) against the source tree being committed.
- Existing reader/source baseline: **149 passed**, Python 3.12, Linux.
- Reader/config/mesh integration: **246 passed** before subsequent network cases.
- Expanded remote tests including the actual public HF crop: **53 passed**.
  The first actual read exposed HF's percent-encoded nested-path redirects;
  a regression was observed RED, then passed with validated redirect decoding.
- A 2×2×2 crop at ZYX `[64:66,64:66,64:66]` exactly matched the preserved NPY
  (source SHA256 `ab3d3dc19d16eaac3ff7512aaa0ddfcc85ac3728c9cc594e03e07c8f7b2dd505`).
  Its support matched finite source support. Beyond admission metadata, it fetched
  only `receipts/0/0/0.json`, `field.zarr/phi/0/0/0` and
  `field.zarr/valid/0/0/0`: **5,843,073 encoded bytes**, not the whole volume.
- Missing/corrupt payloads, completion/manifest pins, duplicate/path-traversing
  inventories, mask sentinels, geometry/codec/layout mismatch, cache mutation,
  unsafe destinations, DNS-bound connections and token-scoped redirects have
  network-independent regressions using real Blosc/Zstd chunks/native decoding.
- After parallel admission integration: **251 passed, 2 opt-in skipped** across
  remote/source/workflow/parallel suites. A first full-mesh attempt exposed a
  parent-supervisor scope error after child extraction; it was reproduced and
  corrected, but that attempt is not mesh acceptance.
- Independent review identified signed-query error redaction and remote resume
  gaps in UVW and continuous ranges. Regressions reproduced all three findings;
  after corrections the remote suite passed **59 tests, 2 opt-in skipped**.
  UVW now verifies remote payloads before retained-output mutation; ranges bind
  remote metadata/runtime identity and prevalidate source payloads on resume.

- An isolated build produced a wheel and installed it in a new owned runtime;
  imports were asserted outside the checkout. Installed remote tests reported
  **60 passed, 1 failed**: fresh W=-60 collection completed and its OBJ exactly
  matched a local-NPY extraction, but subsequent completed-resume verification
  stopped on an HTTPS transport error. No full installed-suite GREEN is claimed.

- Repeated completed-resume verification using the same installed wheel succeeded
  in **20.9914 s**, with **all collection file hashes and mtimes unchanged**.
  The complete local/remote OBJ is 22,605,619 bytes, SHA256
  `f50af75b5058741fd0d1f5db3730ed4a61b33eedf3d921fb8e6514ec77a3a2d0`.
  This establishes matched-tile128 full-mesh parity and completed-resume integrity;
  the preceding HTTPS failure remains a recorded failed invocation, not a green
  installed-suite run.

Final whole-suite results are pending. Broader focused invocations that reached
tool time limits are not counted as passing runs.

## Reproduce

```bash
python -m pytest tests/test_remote_field.py tests/test_surface_sources.py tests/test_surface_workflow.py tests/test_surface_parallel.py -q

# Requires the preserved original NPY for independent value/geometry comparison:
WF_REMOTE_ZARR_LIVE=1 WF_PARIS4_ORIGINAL_NPY=/path/to/reference.npy \
  python -m pytest tests/test_remote_field.py::test_published_paris4_crop_matches_original_npy -q

WF_REMOTE_ZARR_MESH=1 WF_PARIS4_ORIGINAL_NPY=/path/to/reference.npy \
  python -m pytest tests/test_remote_field.py::test_published_paris4_complete_mesh_and_completed_resume -q
```

The mesh check uses matched tile128 settings for local and remote input. It
preserves sampling resolution, coordinates and every component. It runs in a new
temporary output, compares complete OBJ bytes and verifies completed collection
resume preserves all file hashes and mtimes.
