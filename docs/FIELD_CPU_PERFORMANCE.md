# CPU field preparation

These optimizations preserve ordered GPU solves, current halo reads and a single
checkpoint writer. They do not change tolerances or certify physical accuracy.

## Stage 1: existing static constraint cache

`--constraint-cache-bytes N` already controls the ordinary provider's first-fit
CPU cache. Zero keeps the existing no-retention default. This is distinct from
`--prepared-cache-bytes`, which may retain CPU/device solver workspaces.

Field receipts expose `constraint_cache_stats`: budget, hits, misses, builds,
peak bytes/entries and denied insertion attempts. Entries and retained bytes are
zero after cleanup; peak values describe the maximum retained state. Empty
constraint blocks cost4096 bytes. The budget does not cap transient allocations,
parent/worker RSS or GPU memory. Never size it from host RAM.

### Measured local stage-1 result

Five paired repetitions, alternating baseline/candidate order, against main
`4a53a59`; frozen synthetic64-curve,256-point,4-block field,4 sweeps, one numerical
thread and published core1.5.2. The CLI field stage includes validation,
initialization, solves and checkpoints; fixture creation and interpreter import
are excluded. Raw records also retain full child-process wall time.

| Cache budget | Baseline median | Cached median | Field-stage speedup |
| --- | ---: | ---: | ---: |
|128 MiB|1.523 s|0.698 s|2.18×|
|512 MiB|1.518 s|0.696 s|2.18×|
|2 GiB|1.492 s|0.695 s|2.15×|

All repeats produced identical field/support digests. This small field fits the
smallest tested cache; it does not determine the memory needed for637 real blocks.
Real provider replay previously found approximately31–60 MB per sampled entry;
neither those hits nor this synthetic speedup establishes whole-scroll acceleration.

The developer harness is `benchmarks/field_cpu.py`. Give each invocation a fresh
output directory. `--against-root` selects an immutable baseline source checkout;
it never resumes production checkpoints or flushes global OS caches.

## Stage 2: batched short-edge pairs

The ordinary provider now batches clipping and two-endpoint interpolation per
curve. Potentially long, near-step and underflow-sensitive edges retain the exact
scalar norm/linspace calculation. Curve/segment/pair order and dtypes are
preserved; the bounded provider's existing generator is unchanged.

Five alternating paired repetitions of the same synthetic field:

| Comparison | Before | After | Field-stage speedup |
| --- | ---: | ---: | ---: |
|No cache: scalar versus batched|1.557 s|0.581 s|2.68×|
|Same128MiB cache: stage1 versus stage2|0.708 s|0.467 s|1.52×|

All field/support digests matched. Endpoint regressions compare exact bytes,
including signed zero and mixed/degenerate segment cases. These CPU results are
not a CUDA or full-volume measurement.

## Stage 3: experimental CPU lookahead (off by default)

Ordinary `run`/`plan` configurations accept `--field-cpu-workers N` and an
explicit positive `--field-cpu-buffer-bytes B` (or the corresponding underscore
JSON/YAML keys). Both default to zero. Bounded and automatic-convergence
providers reject pooling: their exterior-dependent constraints remain serial.
This does not add pooling to the separate `prepared build` workflow.

Fresh spawned processes have hidden CUDA devices, one numerical thread and
worker-local read-only SQLite connections. Only immutable observations and local
constraint rows travel back. The coordinator retains the existing constraint
cache, requests blocks in solver order, reads live halos in the unchanged core
and remains the sole writer. Cache hits do not rebuild rows in workers.

Buffer admission reserves four payload copies per slot and an extra previous
block; source decompression, worker runtimes, conservative scratch, parent
numerical threads and future cache retention are charged separately. Oversized
curves/interpolation reject rather than exceed a slot. This is conservative
admission, **not an OS-enforced total-RSS cap**. Unknown memory rejects pooling;
there is no silent fallback on malformed payloads, crashes or resource failure.
`field_cpu_stats` reports admitted/requested workers, builds, completed blocks,
buffer reservations, coordinator wait and worker peak RSS.

### Measurements and decision

Same 128 MiB constraint cache; five alternating paired repetitions per stage.
Ranges are observed min–max, not confidence intervals.

| Variant | Field median (range), s | Full subprocess median (range), s |
| --- | ---: | ---: |
|B0, scalar/no cache|1.523 (1.492–1.556)|2.355 (2.337–2.518)|
|B1, scalar/cache|0.698 (0.692–0.699)|1.526 (1.500–1.553)|
|B2, vector/cache|0.467 (0.460–0.470)|1.354 (1.322–1.374)|
|B3, one worker/cache, 64 MiB buffer|1.384 (1.377–1.446)|2.252 (2.215–2.299)|

B1/B2 improved field-stage time approximately 2.18×/3.26× cumulatively against
B0, across separate paired panels. B2's incremental improvement was 1.52×.
B3's contemporaneous B2 control was 0.463 s: pooling was **2.99× slower** in
the field stage and **1.71× slower** in full subprocess time. All field/support
digests matched. B3 built four constraint blocks for sixteen solver visits;
its maximum parent/worker peak RSS was approximately 239/230 MiB, separately,
not summed simultaneous peak memory.

**Do not enable or recommend the pool for this workload.** Cache/vectorization
removed enough preparation cost that fresh-worker startup and IPC dominated.
The B3 timing predates final curve-validation hardening and benchmark
cold/later-sweep instrumentation; it is not a final-source timing claim.
Later 128/64 MiB buffer attempts were rejected as local RAM headroom fell.
Even a 32 MiB preflight had no admitted slot. A resource snapshot with twelve
CPUs and about 1.2 GiB safe headroom admitted only one worker; counts
2/4/8/16/24 were not timed or forced. Functional two-worker tests use controlled
admission and do not establish real resource availability or speedup.

The benchmark now separates initialization, first sweep and later sweep closure
durations, but a complete final-source paired run was blocked by RAM; those
new metrics remain unmeasured. No real-volume, CUDA, production, all-core scaling,
or global scientific-convergence result is established.

### Verification limits

Focused tests cover exact provider/field parity, reversed worker completion,
cache reuse/empty blocks, STOP/resume, malformed payloads, oversized inputs,
CPU/RAM admission, timeout/crash and owned-worker-only cleanup. A full suite and
Docker are outside this work's authorization. The local environment additionally
lacks the pinned `vc_reader` runtime for six import-flag tests (also reproduced
against the frozen stage-2 source); published core 1.5.2 cannot enable the
pre-existing seven advice-integration tests. Neither limitation was hidden by
changing dependencies or weakening assertions.

The source-focused field/spline/boundary/convergence selection passed 283 tests.
A privately built CLI wheel, installed into an isolated target and tested
outside the checkout against the same published core 1.5.2, passed 107
field/pool/spline tests. The four changed module files matched the wheel bytes,
and the unpacked sdist passed 18 benchmark tests (the archive-building test was
deselected inside the snapshot to avoid a nested rebuild). This is CPU-only
artifact evidence, not deployment or CUDA qualification.

Independent review of the earlier CPU stages found two Important issues: the
preparation wait did not observe `field/STOP`, and the source distribution
omitted the benchmark. Both were reproduced and corrected. Callback/`field/STOP`
cancellation now publishes the advisory `CHECKPOINTED_STOP` state, preserves
checkpoint/payload bytes and reaps owned workers.

Benchmark aggregation now rejects missing or mixed source, harness or
configuration provenance within a variant. Aggregated records carry the harness
digest and repetition index; single-run records carry the harness digest.
`paired_comparison` applies those provenance checks itself and rejects absent or
duplicate repetition pairs, so a selection cannot silently compare different
implementations or inputs. `summarize` remains a display aggregator;
completeness is enforced by the paired comparison used for decisions.

## Follow-up: actual multiworker measurements on Hetzner

The user subsequently authorized isolated remote benchmarking. This supersedes
the local RAM block **for these CPU measurements only**, not the lack of
CUDA/real-volume/full-suite qualification.

The shared server had 16 logical CPUs, about 128 GiB RAM and other active jobs.
An owned low-priority systemd unit admitted five CPUs (parent plus up to four
workers), with a 32 GiB memory ceiling, no swap and a 30-minute watchdog.
It finished in 20 min 16 s, peak cgroup memory 1.6 GiB, swap 0 B. A private
Python 3.12.3/core 1.5.2 environment reused dependencies read-only:
NumPy 2.4.6, SciPy 1.18.1 and CPU execution with Torch 2.10.0+xpu.
The existing server core module's hash was unchanged.

There were **70 uninstrumented executions**, five alternating serial/pool
pairs per row below. All used the same frozen candidate source; cache and pool
buffer were each 512 MiB except the explicitly cache-disabled diagnostic.
The original four-block panel was supplemented by a 32-block field with
128 curves of 4096 points each, still entirely synthetic. Each worker count has
its own contemporaneous serial control; host load was not constant across rows.

| Panel | Cache | Workers | Serial field median, s | Pool field median, s | Pool wins / 5 pairs |
| --- | --- | ---: | ---: | ---: | ---: |
|Small, 4 blocks|512 MiB|1|2.103|4.958|0|
|Small, 4 blocks|512 MiB|2|2.148|5.143|0|
|Small, 4 blocks|512 MiB|4|2.179|6.249|0|
|Dense, 32 blocks|512 MiB|1|19.707|18.746|2|
|Dense, 32 blocks|512 MiB|2|19.048|18.257|2|
|Dense, 32 blocks|512 MiB|4|17.873|18.606|1|
|Dense, 32 blocks|Disabled|4|24.964|18.642|5|

**When all static rows fit in cache, there is no reproducible incremental pool
gain.** Ratios of separate medians suggest 4–5% gains for one/two workers, but
the medians of the five *paired* speedup ratios are 1.000×, 0.982× and 0.910×
for one/two/four workers respectively. Those small aggregate apparent gains
must not be promoted as consistent acceleration.

**Without constraint-cache retention, four workers do help this denser panel:**
field time falls about 25% (1.34× ratio of medians; 1.32× median paired speedup),
with all five pairs winning and paired speedups 1.288×–1.358×.
Observed field ranges were 24.082–25.440 s serial and 18.308–19.382 s pooled.
Full subprocess medians were 29.142→23.046 s, about 21% less time (1.26×).
This is evidence for the tested no-cache workload, not proof for arbitrary
partially fitting caches, real volumes or more than four workers.

### Diagnostic evidence

- All requested worker counts actually ran: one/two/four distinct worker PIDs
  in every corresponding repetition. A separate instrumented dense/four-worker
  run confirmed **four simultaneous preparations**, not just spawned processes.
- The dense cached case built 32 blocks once and hit the parent cache on 96
  later visits. Cache-disabled runs built 128 blocks. Exact full-field/support
  hashes and every checkpoint payload hash matched across each panel, including
  cached versus cache-disabled dense runs.
- In the separate cached diagnostic, each worker spent 2.28–2.64 s initializing
  its runtime/source handles. Across all workers, preparation of 128 tasks
  summed to about 2.50 s: observations 0.38 s, constraints 2.09 s. These are
  **overlapping worker durations, not additive wall-time components**.
- An extra diagnostic pickle pass summed to about 0.027 s for about 60 MB of
  payloads. That does **not** measure all IPC: queue scheduling, transfer,
  coordinator validation and waits remain combined. Blaming serialization
  alone is unsupported.
- Later cached sweeps were roughly 3.4–3.8 s, without meaningful worker scaling.
  In the cache-disabled four-worker comparison, later-sweep median fell from
  5.98 to 3.51 s, enough to amortize startup. The remaining ordered solve/commit
  path was unchanged and was not individually profiled.

**Decision:** prefer the cache/vectorized serial path when static rows fit.
Keep the pool opt-in: it has a measured conditional benefit when static
constraints must be rebuilt, rather than being universally ineffective.
No automatic activation or worker-count optimum is justified.
Eight/sixteen-worker scaling was not attempted while unrelated jobs were active.

### Direct cache-versus-pool comparison

A second isolated matrix compared six explicit profiles on the same frozen
32-block synthetic panel, five alternating-order rounds (30 executions total).
It requested the same owned low-priority unit properties: five CPUs admitted
(parent plus four workers), a declared 32 GiB memory ceiling, swap disabled and
one numerical thread per process.
Input, source, runner and driver digests were identical in all 30 runs; the full
field/support hash and all 32 checkpoint payload hashes matched too. Requested
pools admitted exactly two or four distinct worker PIDs and no worker initialized
CUDA. Workers rebuilt constraints in the cache-disabled and partial-cache
profiles; the full-cache pool reused the coordinator's rows instead. Shared-host
load was about 10–15 during the matrix.

| Profile | Cache | Workers | Field median, s | Total subprocess median, s | First sweep, s | Later sweeps, s | Paired ratio vs full-cache serial | Wins / 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|cache64-serial|64 MiB|0|17.681|22.606|5.823|3.476|—|—|
|pool2-cache0|off|2|19.204|23.792|6.421|3.635|0.950|0|
|pool4-cache0|off|4|19.515|23.915|6.313|3.678|0.948|0|
|cache64-pool4|64 MiB|4|18.837|23.586|6.515|3.376|0.929|0|
|cache16-serial|16 MiB|0|22.533|27.021|5.819|5.158|0.837|0|
|cache16-pool4|16 MiB|4|19.543|23.952|6.496|3.534|0.942|0|

The paired ratio is median total-subprocess baseline/candidate; above 1 means
the candidate was faster. **Full-cache serial was the direct winner: every other
profile lost all five total-time pairs.** Adding four workers to a fitting cache
was about 7% slower in total time. Two/four workers without cache were about 5%
slower than full-cache serial, consistent with the earlier finding that pooling
pays only when static-constraint reconstruction dominates.

The 16 MiB partial budget retained 9 of 32 blocks (101 builds, 27 hits,
92 denied entries). **Partial cache plus four workers beat its partial serial
control in all five pairs**: median paired ratio 1.150× (range 1.062–1.178×),
total median 27.021→23.952 s. That is the measured conditional pool benefit.
It still did not beat full-cache serial (0.942×).

| Profile | Cache peak entries / bytes | Builds / hits / denied | Parent peak RSS, MiB | Max worker RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
|cache64-serial|32 / 56.6 MiB|32 / 96 / 0|769.5|—|
|cache64-pool4|32 / 56.6 MiB|32 / 96 / 0|768.2|697.5|
|pool2-cache0|0 / 0|128 / 0 / 128|717.7|698.3|
|pool4-cache0|0 / 0|128 / 0 / 128|727.0|698.0|
|cache16-serial|9 / 15.8 MiB|101 / 27 / 92|731.3|—|
|cache16-pool4|9 / 15.8 MiB|101 / 27 / 92|742.0|697.7|

Cache bytes are retained charges, not total RSS. Parent and worker RSS are
reported separately; worker figures are the maximum single worker, not a sum.
This was a shared, synthetic, CPU-only panel with no OS cache flush, real
volume, CUDA, or eight/sixteen-worker scaling. Treat the choice as
workload-specific evidence, not statistical significance or a universal
optimum.

## Choosing cache and pool settings

All three settings default to zero (disabled). Choose them explicitly:

- **Cache:** reserve memory for the solver, coordinator and prepared inputs
  first. `constraint_cache_bytes` limits retained static restriction bytes, not
  total process RSS. Read `constraint_cache_stats`: `builds`/`misses` count
  reconstructions, `hits` count reuse, and `denied_entries` counts blocks/entries
  that did not fit. With a partial budget, `peak_bytes` is only what was retained,
  not the full demand, so it cannot by itself prove the cache is large enough.
  Measure with a generous budget once, then size deliberately. If demand is
  unknown, state that it is unknown instead of enabling an unbounded cache.
- **Pool:** `field_cpu_workers` prepares immutable ordinary-provider CPU work
  only. Solves and checkpoint writes remain ordered and serial. Halo-expanded
  observation/constraint preparation can run in workers; live halo values are
  still read at solve time. Set `field_cpu_buffer_bytes` explicitly; it is a
  preparation/IPC budget, not a total RAM cap. Bounded and
  automatic-convergence providers reject pooling.
- **Tested explicit profiles (synthetic panel only):** full cache serial uses a
  budget at least the observed working set (64 MiB fit 59.4 MB here) with
  `field_cpu_workers=0`. Partial cache plus pool uses an explicit partial budget
  (16 MiB retained 9 of 32 blocks here) with `field_cpu_workers=4` and
  `field_cpu_buffer_bytes=512 MiB`; it beat partial serial 5/5 but remained
  slower than full cache serial. Do not hardcode these sizes for production.
- **Do not extrapolate** the synthetic 32-block dense result (about 59 MB
  retained, roughly 56.6 MiB) to a real 637-block run. Fibre distribution,
  geometry and block occupancy change the working set. Size from that run's
  own `constraint_cache_stats`.

Field-stage time excludes fixture construction and parent imports, but includes
worker startup, validation, initialization, solves and checkpoints. Full
subprocess time additionally includes fixture construction/imports and
result/provenance checks. Raw records preserve initialization, first/later sweep
closures, source/script/input/output digests, configuration, cache counts,
waits and memory. Instrumented diagnostics are excluded from timing medians;
OS caches were not flushed. Shared-host scheduling remains a limitation.
