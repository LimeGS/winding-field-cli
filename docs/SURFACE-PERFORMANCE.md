# Prepared surface extraction performance

## Measured case

Paris4 coarse W−60, raw phi `-71.54987931251526`, 307.2 µm grid,
tile 64, one CPU, 3 GiB RAM, no GPU. The input is the original NPY
with SHA256 `ab3d3dc19d16eaac3ff7512aaa0ddfcc85ac3728c9cc594e03e07c8f7b2dd505`.
These measurements do not exercise remote Zarr reading.

| Run | Serial | Batched prepared workflow |
| --- | ---: | ---: |
| SSD, unprofiled container wall time | 88.080 s | 49.278 s |
| SSD, cProfile function time | 147.002 s | 63.464 s |
| HDD, cProfile function time | 701.429 s | Not measured |

The unprofiled comparison saved 44% of elapsed time (1.79×). This is one
sequential serial-then-candidate comparison on 2026-11-10, without flushing
caches, not a cold-cache or repeated statistical benchmark. Both unprofiled runs
use the same published image, source-mounted baseline `eccd904` or candidate,
input, SSD, configuration and resource limits. Profiler overhead is substantial:
do not compare a profiled baseline against an unprofiled candidate.

## What changed

The HDD profile spent about 77% of its time in SQLite transaction exits and
`fsync`. Moving the output to SSD improved the serial profile by 4.77× with
an identical SQLite file hash; this is a storage-placement gain, not a code gain.
No durability settings were relaxed.

On SSD, serial mesh insertion became the main cost. Fresh prepared fields now
reuse the existing batched `native_mesh` extractor instead of the serial store.
The numerical tile extraction kernel is unchanged. Legacy prepared resumes still
use the serial extractor selected by their saved algorithm identity; the
extractor continues to validate the complete identity. Imported meshes are
unchanged. There are no new dependencies or configuration options.

The candidate profile reduced SQLite `execute` calls from 11,233,106 to 3,061.
Batched insertion, SQL, durable transactions and canonical row hashing remain
significant costs; this does not make extraction independent of storage speed.

## Geometry and limitations

Both real comparisons preserve 301,420 vertices, 602,824 faces and all six
components, with zero boundary, nonmanifold or inconsistent edges and duplicate
faces. Area summaries also match exactly. The exported OBJ is byte-identical:
`55f451addccf93107b5044e7214d3209b4857778c5b6b215a3d206c09a1c6eb8`.
Implementation-specific SQLite hashes are not a geometry acceptance criterion.

The original front and oblique 1600×1600 PNGs also match byte for byte after
rendering the candidate through the CLI with the same settings.

This is an uncertified geometry diagnostic, not CT/ink or physical sheet
acceptance. Self-intersection remains unmeasured. Focused regression validation
passed 219 tests. Fresh review found no confirmed production bug and requested
additional sensitive-geometry and corruption-preservation coverage. The expanded
prepared workflow file passed all 44 tests, including those cases. Complete-suite
validation is blocked: the baseline global run reported 1,935 passed, 117 failed,
42 errors and 18 skipped. Native loading fails on the test container's default
no-exec `/tmp`; a representative native build test passes with executable `/tmp`.
Other failures include Numba cache and disk-reserve errors, but not all failures
have been classified. The candidate global result is still pending; this is not
whole-suite approval. Inherited OBJ mismatch handling preserves the changed
OBJ but can leave a `.pending` staging file; that cleanup behavior is unchanged.

Benchmark receipts are on gpu-1 under
`/mnt/fast2/prepared-surface-performance-20261110/`, in
`baseline-ssd-01/evidence/`, `candidate-ssd-profile-01/evidence/` and
`matched-unprofiled-01/evidence/`. The HDD reference is under
`/mnt/bulk/paris4-hf-coarse-20261110/perf-W-60-01/evidence/`.

## Independent parallel levels

The opt-in `prepared mesh --parallel-levels` route was built as a wheel, installed
offline into a separate target, and run outside the source checkout on two original
Paris4 raw levels: `-71.54987931251526` and `-70.35738132397334`.
Both runs used two assigned CPUs, 3 GiB RAM, tile64, the same input/runtime/SSD,
and no profiler. The serial multi-level bundle took 79.317 s; independent parallel
bundles took 58.570 s with two automatically admitted workers: 1.35× observed
speedup (26% less elapsed time).

This is a single sequential, uncontrolled-cache observation on a shared host with
other verification running, not an isolated or statistically repeatable scaling
benchmark. An earlier pre-review diagnostic gave a different ratio; it is not the
final-build result. Additional field reads and competing disk I/O limit scaling.

For both levels the OBJ bytes and every geometry/topology/area metric match the
serial result exactly (ignoring the intentionally child-local level ID zero).
The second level retains 305,336 vertices, 610,652 faces and eight components,
with no boundary/nonmanifold/inconsistent edges or duplicate faces. Its OBJ SHA256
is `67c7770a6ba45fcc440982269cb957f91babce8d001d36e9112b773bbd12623a`.
Installed completed resume preserved product bytes and modification times.

Final package/build, timings, parity and resume logs are retained under
`gpu-1:/mnt/fast2/prepared-surface-performance-20261110/parallel-final-e2e-01/`.
The focused post-review suite passed 272 tests, with an additional real CLI
SIGTERM/checkpoint/resume test passing separately. Final global-suite verification
is still pending; no merge, deployment or publication is implied.
