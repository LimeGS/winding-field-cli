# Solver work reuse

The numerical core reuses PCG residual norms and scratch buffers without changing
the configured precision, tolerances, recurrence order or final true-residual check.
Output fields own their storage independently of prepared workspaces.

## Optional caches

For a new CT run, add `--prepared-cache-bytes 67108864` and
`--constraint-cache-bytes 67108864` to the existing `winding-field run` command.
Both defaults are **zero**. Prepared workflows use the same names in `execution`.

- Prepared cache: static gradient/constraint operators and solver buffers, counted
  conservatively across CPU and device allocations.
- Constraint cache: static pair selection, indices and coefficients. The bounded
  pair catalog refreshes exterior targets on every visit. The legacy local-pair
  route retains only rows whose targets are independent of exterior field values.
- First-fit retention avoids repeated eviction when only a subset fits. Caches
  are process-local, excluded from checkpoint products and released on exit.
- These are additional retained-storage budgets, **not peak-memory limits**. Keep
  room for input arrays, checkpoint reads and transient assembly. Small problems
  can be slower with caching; choose budgets from measurements.

The existing prepared `--prefetch` option remains compatible. It overlaps immutable
observation loading in a CPU subprocess; GPU transfers and live boundary reads are
not double-buffered.

## Reproducible measurements

`benchmarks/performance/solver.py` takes explicit core/CLI checkouts, writes hashes
of loaded sources, environment details, timings, solver receipts and complete field
arrays, and rejects existing output directories. Use the **same Python environment**
for baseline and candidate. For example:

```sh
python benchmarks/performance/solver.py --core /path/to/baseline-core \
  --cli /path/to/baseline-cli --case blocks --size 16 --repeats 3 \
  --output /path/to/new-baseline
python benchmarks/performance/solver.py --core /path/to/candidate-core \
  --cli /path/to/candidate-cli --case blocks --size 16 --repeats 3 \
  --cache-bytes 67108864 --output /path/to/new-candidate
```

Alternate baseline/candidate order, include a candidate with caches disabled, and
compare `fields.npz`, iteration counts, failure reasons and true residuals before
interpreting speed. `--case solve` measures repeated local solves with cold/warm
prepared state; `--case blocks` measures three sweeps, checkpoint writing and live
cross-block constraints. Every block repetition starts with a fresh output/cache;
its warm label refers to the process/runtime, not to a reused previous run's field.
Imports/device discovery occur before timing. Block global verification is outside
the measured interval; a small fixed sweep budget is not global convergence.

Use `--device cuda:0` only in an explicitly allocated GPU environment. For diagnostic
phase costs, run separately with `--profile`: this synchronizes phase boundaries,
records exclusive host wall times and exports a PyTorch CPU/CUDA Chrome trace.
Phases cover constraint preparation, input validation/transfers, gradient and sparse
operators, reductions, output transfers and checkpoints. GPU copies/kernels appear
in the trace. Synchronizing profile timings must not be mixed with throughput runs.
The profile also adds NVTX ranges for `nsys profile --trace=cuda,nvtx`; use Nsight
Compute on kernels selected from that trace, separately from timed measurements.

All benchmark data is synthetic FP64. It measures implementation throughput, not
production CT scale, physical reconstruction accuracy or general hardware speedups.

## CPU measurement (2026-11-07)

Apple M2 Pro, macOS arm64, PyTorch 2.10.0, one Torch/BLAS thread. Baseline core `f750fd11` /
CLI `2cee51a`; candidate core `f6387ec8` with the CLI integration described above.
Two fresh processes per variant, reversed order in round two, three executions
per process. Cold medians use the first execution; warm medians use the other four.
The cached variant allows 64 MiB for each cache. Times below exclude profiling.

| Workload | Base cold → cached cold | Base warm → cached warm | Warm time reduction |
|---|---:|---:|---:|
| Local solve, 24³ | 44.799 → 46.193 ms | 42.955 → 42.501 ms | 1.1% |
| Local solve, 48³ | 414.299 → 394.491 ms | 405.859 → 398.763 ms | 1.7% |
| Block run, 16³, three sweeps | 268.078 → 254.667 ms | 243.794 → 220.610 ms | 9.5% |

These few repetitions are descriptive, not confidence intervals. The local-solve
gains are small. With caches disabled, the candidate block warm median was
247.595 ms (**1.6% slower**); the 48³ local median improved to 395.861 ms (2.5%).
Cold setup can regress, which is why cache budgets remain opt-in.

All 36 candidate field arrays matched their corresponding baseline arrays exactly;
iteration counts, reasons, true local residuals and stored-field global verification
also matched, as did all 96 compared stored chunk payload hashes (field and support).
The synthetic three-sweep block workload is not globally converged.
Its verification result is compared, not promoted to physical acceptance.

## CUDA measurement: numerical parity, timing inconclusive

On GPU 1 of `gpu-1-remoto` (GTX 1660, PyTorch 2.10.0+cu128), all 36 candidate
arrays, solver residuals/iterations/reasons and 96 stored payload hashes matched
baseline exactly in the final campaign. The solver ran on CPU 6; a separate
monitor on CPU 7 sampled GPU process residency every 200 ms.

That campaign recorded 761 GPU process records, including 98 explicitly named
`llama-server` records and 51 other unrecognized/vanished-process records. It
correctly failed the contention check after numerical comparisons passed.
**No CUDA speedup is established.** The earlier campaign shared GPU 1 with
`lasagna-rust`; another checked availability only at launch. None provides a
reliable uncontended timing comparison. A reserved GPU window is needed; merely
waiting for one process to exit is insufficient when new clients keep starting.

## Profiling and qualification

The separate 24³ CPU profile reduced norm/dot calls from 828 to 623, with unchanged
210 gradient and constraint operator calls. Timings from these synchronized,
instrumented runs are diagnostic only.

Nsight Systems captured the CUDA candidate: `cusparse::csrmv_v3_kernel` was the
largest individual kernel category (29.2% of kernel time in that trace). This
diagnostic capture had foreign GPU clients at launch and is not speedup evidence.
Nsight Compute could attach but returned `ERR_NVGPUCTRPERM`; hardware-counter
metrics were unavailable under the existing host permissions.

Validation of the published core: **987 tests**, **93.30% coverage**, Ruff and
mypy clean. The installed CLI macOS suite passed **1786 tests**, with 36 skips;
four targeted benchmark/prefetch/cache/resume checks passed after test expansion.
The published core archive passed hash-checked installation and all nine recorded
source hashes matched. The CLI wheel matched all 99 Python sources and provenance;
the source distribution includes the benchmark.

Installed Linux qualification passed **83 CLI tests** and **214 core tests**,
using the published archive and CLI wheel in an ephemeral container. A test-only
follow-up (`dd36513` on the core feature branch) permits CPython's shared-key
dictionary accounting to shrink by 8 bytes when another operator is constructed;
buffer addresses and numerical outputs remain checked exactly. The CLI stays
pinned to `f6387ec8`: the follow-up changes no runtime code. The adjusted ten-test
work-reuse suite also passed on macOS.

Raw reports, arrays, checkpoints and traces are retained under ignored
`.runtime/solver-measurements-*`; the remote Nsight report and SQLite export are
under `/mnt/fast2/winding-field-docker-20261106/solver-perf/.runtime/`.
