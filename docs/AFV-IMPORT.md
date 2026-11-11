# AFV import: bounded commits, progress and measurement

External AFV import retains the existing source/geometry/map checks, physical
coordinates, classification/annotations, ROI/support clipping and curve checksums.
It still streams one source block at a time. No scientific parameter or SQLite
durability setting is changed.

## Transactions and recovery

- New blocks share a transaction until **1,000 blocks or 2 seconds** at a
  complete-block boundary, whichever occurs first. The time limit is checked
  between blocks: a slow individual block or commit can exceed it.
- A block's curves, R-tree bounds and `done` checksum are atomic, including a
  `done` row for an excluded block with no retained curves.
- Normal completion commits the final partial batch. Stop or exception rolls
  back **the entire unconfirmed batch**, preserving previous committed batches.
  Hard process death relies on SQLite's unchanged durable journal recovery.
- Resume uses the database, not telemetry. It revalidates already committed
  blocks with the original clipping/checksum/inventory checks, without inserting
  them again; only uncommitted writes must be imported again. This validation
  still costs time and is deliberately not skipped for speed.
- `FIBRES.json`, its field meanings and final database hash retain their existing
  contract. Different commit boundaries can change SQLite bytes; logical curves,
  annotations, clipping, spatial indexes and inventories must match instead.

No list of batches/fibres/results is accumulated in Python. SQLite retains its
existing bounded page cache and normal journal/spill behavior. These limits do
not make an individual oversized source block acceptable: existing block-size
validation is unchanged.

## Progress

`fibres/STATUS.json` uses the existing `atomic_json` mechanism. The optional
`afv_progress` object in `winding-field status RUN` exposes it without replacing
existing RUN fields or `native_progress`. Studio is unchanged.

Updates are limited to once every **2 seconds**, plus start, final validation,
pause/failure and completion. A fast batch commit does not force a progress write.

| Field | Meaning |
|---|---|
| `schema` | `afv-import-progress-v1` |
| `status` | `IMPORTING`, `VALIDATING`, `PAUSED`, `FAILED`, or `FIBRES_READY` |
| `blocks_total` | Source AFV block inventory, not fibre or curve count |
| `blocks_processed` | Fully processed/revalidated blocks in this invocation |
| `blocks_committed` | Unique durable `done` rows, including earlier invocations |
| `blocks_pending` | Completed new blocks in the current uncommitted transaction |
| `commits_this_invocation` | Successful new-block batch commits in this invocation |
| `blocks_per_second` | Observed processed blocks / invocation elapsed seconds |
| `elapsed_s` | Invocation elapsed seconds, including validation |
| `last_batch_committed_utc` | Epoch seconds recorded after a successful commit; null if unknown |
| `updated_utc` | Snapshot wall-clock epoch seconds |
| `identity_sha256` | Import identity associated with this advisory snapshot |

On resume, committed blocks can exceed processed blocks while retained data is
being revalidated. After rollback, processed work can exceed durable work while
pending is zero. None of these counters means retained fibres, convergence or
scientific acceptance. The previous commit timestamp is reused only when its
snapshot identity and committed count match the database; it may be unknown
after a crash between a commit and its status update.

**The sidecar is not a checkpoint.** It can lag or disappear on a crash. Missing,
malformed or stale telemetry never supplies recovery counts. If an ordinary
status write fails, import stops safely; confirmed batches remain durable and
the original failure is not replaced by another telemetry write failure.
Completion authority remains the verified `FIBRES.json` receipt.

## Reproduce the benchmark

Freeze the old importer before testing, then install the candidate only in an
isolated test environment. Never update or restart an active run/environment.

```bash
git show b8ed5eb:src/winding_field/afv.py > /new-owned-directory/baseline-afv.py

# Same generated AFV/maps/config for old/new; independent worker processes/outputs.
PYTHONPATH=src python tools/bench_afv_import.py \
  --baseline-source /new-owned-directory/baseline-afv.py \
  --output /new-owned-directory/excluded --blocks 2000 --retained-every 0 \
  --repeats 3 --profile

PYTHONPATH=src python tools/bench_afv_import.py \
  --baseline-source /new-owned-directory/baseline-afv.py \
  --output /new-owned-directory/clipped --blocks 2000 --retained-every 1 \
  --repeats 3 --profile

# An explicitly supplied immutable real reference, not an active run:
PYTHONPATH=src python tools/bench_afv_import.py \
  --baseline-source /new-owned-directory/baseline-afv.py \
  --output /new-owned-directory/real \
  --prepared /immutable-reference/prepared --config /immutable-reference/import.json \
  --repeats 3 --profile
```

The real-reference directory supplies `PREPARED.json` and `maps/`; they are copied
to each independent worker output. Config paths are resolved relative to that
config file. Original AFV/config/maps remain unchanged. Output directories must
be new and separate from input. Run benchmark workloads sequentially to avoid
measuring their own I/O contention.

`RESULT.json` records individual/median time, blocks/s, commit calls/time,
non-commit time, peak process RSS, SQLite synchronous/journal settings,
input/config/code hashes and streaming logical database digests. Non-commit time
includes validation, geometry, queries, maps and progress; it is not pure CPU
time. Peak RSS includes interpreter/dependency overhead. Commit count is not an
fsync syscall count: SQLite may spill pages during a transaction.

Order alternates between repetitions; OS caches are not dropped. `--profile`
runs a separate extra candidate invocation, excluded from timing medians, and
records top cumulative functions plus `PROFILE.pstats`. Cumulative call times
can overlap; inspect the call graph before adding another optimization.

## Measured results and limits

Final measurements ran sequentially, with three alternating-order repetitions
per workload and a separate profiled candidate invocation. Each comparison used
the same immutable generated input/config/maps and independent process outputs.
The medians below exclude profiling and earlier concurrent exploratory runs.

| Workload (2,000 blocks) | Baseline | Batched | Speedup | Block commits | Blocks/s, old → new | Peak RSS MiB, old → new |
|---|---:|---:|---:|---:|---:|---:|
| All excluded by ROI | 51.893 s | 0.304 s | 170.47× | 2,000 → 2 | 38.54 → 6,570.09 | 41.61 → 42.09 |
| Retained/support-clipped (4,000 curves) | 62.724 s | 4.749 s | 13.21× | 2,000 → 3 | 31.89 → 421.15 | 43.11 → 45.12 |

Median time inside block commits fell from **50.642 to 0.040 s** for excluded
blocks and **49.486 to 0.075 s** for retained blocks. All six logical database
digests per workload matched, including curves, classification, annotations,
spatial indexes, processed-block checksums and receipt inventories. All workers
reported `synchronous=2`, `journal_mode=delete`; no durability relaxation was used.

The retained profile now places geometric/support clipping first: `_pieces`
6.165 s cumulative, `supported` 4.594 s, `Tiles.load` 2.250 s, versus SQLite
`execute` 0.222 s. These times overlap and profiling adds overhead. No further
geometry, query or cache optimization was applied without separate evidence.

Runtime: Python 3.12.15, NumPy 2.4.6, SQLite 3.54.1, Linux x86_64/glibc 2.39,
overlayfs with 4,096-byte filesystem blocks. The host is shared and OS caches
were warm. These are synthetic workloads, **not** the observed 2.58-million-block
AFV or a verified reproduction of its `xlog_wait_on_iclog` filesystem. A production
speedup claim requires that immutable input/config and a matched benchmark.

Raw receipts are retained in the local task ledger directory
`.superpowers/sdd/2026-11-10-afv-batches/`:

- `benchmark-final-excluded.json`, SHA256 `f578ea744755238fb11e28a1895d6c331e51f8389cbcae3b76b0c203ae62b6fd`.
- `benchmark-final-retained.json`, SHA256 `65c7c2ce6855ea54d7618fcebb4ffad115d76c7720beeb53315f496f11081c45`.

Baseline importer SHA256:
`bcbb646b3d25c315de1e8a6431f2ed223957c623e8846013126e29e0fc123713`.
Candidate importer SHA256:
`f93f53781a673eca38a7c91f1501688c815b57b7d05be3b81d5a1baf3f29fe42`.

Verification recorded 141 relevant source tests passing and 75 tests passing
against an isolated installed wheel. The memory regression compared 250 with
2,500 excluded blocks, allowing less than 2 MiB additional traced Python peak;
the latter committed at 1,000, 2,000 and 2,500 blocks. Initial whole-suite
qualification recorded 2,055 passed, 27 skipped and two failures in 3,223.69 s.
Both failures traced to the shared memory observer misreading a hybrid,
container-mounted cgroup hierarchy, rather than an AFV result mismatch. After
correcting controller/mount resolution, both original integration tests passed
unchanged against a rebuilt isolated wheel, with 225 related tests passing.
Full-suite requalification after that correction is pending; this is not a
whole-suite GREEN claim.

```bash
python -m pytest tests/test_afv_import.py tests/test_afv_batching.py \
  tests/test_afv_benchmark.py tests/test_splines.py tests/test_exports.py -q
```
