# AFV import: bounded commits, progress and measurement

External AFV import retains the existing source/geometry/map checks, physical
coordinates, classification/annotations, ROI/support clipping and curve checksums.
The serial route streams one source block at a time; optional local workers use
bounded in-flight blocks and one ordered parent writer. No scientific parameter or SQLite
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

There is no unbounded list of batches/fibres/results. SQLite retains its
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
| `execution` | Requested/admitted workers, limiting resource, worker PIDs and support loads; advisory only |

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

## Local prepare and import throughput

- `--import-workers 1` is the default serial reference. `0` requests automatic
  admission; positive `N` is a maximum, not a promise to start N processes.
- Admission respects CPU affinity/quota, writer capacity, effective memory and
  shared disk headroom. It preserves 2 GiB host and 128 MiB effective-cgroup RAM
  headroom and the configured disk reserve. Unknown memory selects serial mode.
  Decoded source chunks, interpolation scratch, in-flight results and parent
  replay are charged, not just output tile size.
- Parallel tile computation with purely local CT and prediction channels uses private
  worker caches. With a remote CT or remote predictions it runs only when
  `--import-workers` is set explicitly; workers then share the run's encoded-chunk cache
  so no chunk is downloaded once per worker, and the parent still records and
  re-verifies every consumed chunk. External AFV clipping
  reads only the run's local maps and the AFV, so it uses the admitted workers even
  when the CT or predictions are remote. Direct
  `predict.prepare` callers without a bound CT ledger also stay serial.
  Without external `--fibres`, surface reads stay serial alongside parent fibre
  inference; its `--threads` setting is unchanged.
- Imported observations use at most 256 MiB aggregate decoded cache across
  parent/workers; support caches are separately bounded. Exact vectorized fibre
  clipping retains the scalar streaming reference for pathological blocks.
  Results/pins exceeding the 16 MiB worker envelope are recomputed by the parent,
  never truncated or downsampled.
- Calibrated local predictions are sampled directly into main maps. The legacy
  standalone normalized-map importer remains available; saved identities are not
  silently relabeled or migrated.
- Source-ledger writes flush at 128 new chunks or 2 seconds and before the
  consuming tile is published. AFV export commits up to 1,000 complete fibres or
  2 seconds with its bounds/cursor atomically; a graceful export stop flushes its
  complete records. Lasagna export records up to 128 atomically written files or
  2 seconds. Exceptions roll back only pending database writes.
- Worker errors are not silently replayed. STOP cancels admission and reaps owned
  processes; after 5 seconds uncooperative workers are terminated. Graceful
  partial/completed resume preserves committed tiles. A hard-killed parent
  leaving an unsealed source inventory still rejects recovery rather than
  guessing; use a new output for that existing failure mode.

`lasagna-import/PERFORMANCE.json` (base-grid) or
`lasagna-source/PERFORMANCE.json` (calibrated) reports tile admission/counters.
AFV execution is in `fibres/STATUS.json`. Neither is a completion receipt.
Hashes, geometry, missingness and authoritative ledgers remain the acceptance
boundary. Incompatible implementation/config identities still require a new run.

To measure **prepare + fibres + mandatory exports**, use a local-input JSON
containing `source`, `fibres`, `lasagna_input` and any required geometry/ROI fields:

```bash
python tools/bench_local_imports.py --config /immutable-inputs/import.json \
  --output /new-owned-directory/import-benchmark --repeats 3 \
  --workers 1 2 4 0 --profile
```

The output must be new and cannot contain or be inside an input. Every trial uses
a separate process/output. `RESULT.json` records requested/effective workers,
overlapping stage/validation/map-compression times, parent/worker decode/hash/
commit counters, sampled aggregate peak RSS, output bytes, implementation/input
hashes and logical map/curve parity. Commit counts are not fsync counts; sampled
RSS is not a hard memory ceiling. Header hashing is not separately instrumented.
Input hashing warms the OS cache: these are warm-cache trials, not cold-I/O
claims. Profiling and separate partial-resume trials are excluded from medians;
completed-resume time and logical/retained-tile preservation are checked too.

Use matched baseline/candidate environments with the **published pinned core**;
`tests/test_core_distribution.py` checks its version and recorded source hashes.
Run repetitions sequentially on the same filesystem, without competing
benchmarks or cache drops. Default serial execution remains available. Real-data
throughput/scaling is unqualified until representative immutable inputs and
resource budgets are supplied; synthetic wins do not qualify production speedups.

### Synthetic qualification (November 17, 2026)

CLI v1.5.6 (`af4bb05c`) was compared with this branch using the same frozen
task-1 benchmark instrumenter (`1734296`), published core 1.5.2, immutable tiny
fixtures with 200 fibres, three sequential repetitions and warm OS caches.
Times include prepare, fibre import and mandatory exports:

| Input route | Baseline median | Candidate median | Reduction |
|---|---:|---:|---:|
| Local base-grid | 0.7516 s | 0.4695 s | 37.5% |
| Local calibrated | 0.8389 s | 0.4580 s | 45.4% |

All logical map/curve digests matched. Base-grid support loads fell 600→3,
decodes 135→38 and commits 406→73. Calibrated support loads fell 800→4 and
commits 328→48; its 25 decodes were unchanged.

Separate extended trials requested workers 1/2/4/0 for both routes and checked
partial/completed resume, retained-tile bytes/mtime, source immutability and
profiling outside medians. **Every request admitted one worker** on this Mac
because memory accounting was unknown. Timing differences between those requests
are noise, not parallel scaling. Real 2/4-process correctness, order, cancellation
and mutation checks passed under controlled admission on tiny test fixtures.

The separate 1,000-block AFV benchmark also checked retained, excluded and
scattered inputs against the frozen importer, with exact logical parity:

| AFV case | Baseline median | Candidate median |
|---|---:|---:|
| Retained, fragmented support | 1.4637 s | 0.4672 s |
| All excluded | 0.0497 s | 0.1988 s |
| Scattered (one retained in twenty) | 0.1216 s | 0.2161 s |

**Tiny standalone exclusion-heavy imports regress:** their new admission-module
cold imports dominate (0.285 s of 0.351 s in a separate instrumented profile).
This fixed startup cost is included, not hidden by warming imports. The measured
whole-pipeline improvements do not promise faster execution for every workload.

Software qualification: 2,466 tests passed, 61 skipped, with one existing
multithreaded-fork deprecation warning. The installed macOS/Python 3.12 wheel
passed 129 tests with 5 skips outside the checkout; package contents and pinned
core source hashes were checked. Linux/container and Python 3.11 execution were
not qualified locally. No production data, active job, release or deployment was
changed.

## Reproduce the serial AFV benchmark

Freeze the old importer before testing, then install the candidate only in an
isolated test environment. Never update or restart an active run/environment.

```bash
git show 36e5840:src/winding_field/afv.py > /new-owned-directory/baseline-afv.py

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

## Historical durable-batching results and limits (November 11)

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
geometry, query or cache optimization was applied in that historical comparison;
the local pipeline optimizations above are separate.

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
