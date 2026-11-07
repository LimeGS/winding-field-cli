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

macOS arm64, PyTorch 2.10.0, one Torch/BLAS thread. Baseline core `f750fd11` /
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
also matched. The synthetic three-sweep block workload is not globally converged.
Its verification result is compared, not promoted to physical acceptance.

The first CUDA campaign also had exact numerical parity, but another `lasagna-rust`
process occupied GPU 1 and timings varied more than 10×. Those timings are excluded
from speedup claims; a clean GPU measurement is pending.
