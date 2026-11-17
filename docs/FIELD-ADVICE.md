# Consultative field progress

This development feature is **explicitly opt-in** and has no scientific control
authority. Published core 1.5.2 lacks its observer hooks: enabling it fails with
a clear capability error rather than approximating pre-encoding measurements.
Normal installation requires a separately qualified core/dependency transition.
No installed runtime is upgraded automatically.

## Requests and read-only reporting

Add `--field-advice` to a **new** `run` or `field-converge` request.
JSON/YAML CLI settings use `field_advice: true`; defaults omit the key entirely.
For `prepared build`, the native recipe key is `execution.field_advice: true`,
or use the same flag. Resume retains saved settings and implementation guards;
this is not a migration mechanism for old fields.

`winding-field field-advice RUN` reads the sibling diagnostics. A direct prepared
field root is accepted too. The command does not run a solver, load observations,
verify a field, write STOP, change a budget or publish FIELD.

The ordinary legacy route has **no equivalent automatic global verifier**.
Bounded runs retain their budget-end verification; automatic convergence retains
its initialization/per-sweep cadence; prepared verification stays explicit.
Only calls those routes already make are timed and recorded.

## What the evidence means

- **Measured:** owned observed-node updates before encoding (maximum, sum of
  squares, count and RMS); fixed stored-field values, raw and reference-relative
  changes, encoding-resolution intervals, gradient stencils and read costs.
  Halo nodes and unsupported nodes do not enter observed counters.
- **Estimated:** local first-order normal displacement only where both closures
  have complete observed stencils, known gauge and gradients resolved against
  h/2h, temporal and encoding variation. Outside that local stencil domain,
  unsupported or weak-gradient estimates are null, not zero.
- **Unknown:** remaining equation/geometric error, whole-volume stability and
  physical sufficiency. Every recommendation retains **“precisión útil no calibrada”**.

`measured.comparisons[].points[]` shows changes and uncertainty;
`estimated.local_normal_displacement` contains conditional estimates separately.
The report includes the
fixed plan, required/common counts, exclusions, closure hashes and rationale.
The fixed sample is not proof that the whole volume is representative.

Three adjacent complete closures and the same complete required observed set
(at least two points) are necessary. A multiblock plan also needs finite planned
interface evidence. Recommendation precedence is:

1. Current global verification passed: `DATOS_INSUFICIENTES` about *further
   geometric usefulness*, while displaying measured numerical convergence.
2. Comparable measured global residual did not decrease: `POSIBLE_ESTANCAMIENTO`.
3. Both sample-change intervals are unresolved at encoding resolution:
   `CONVIENE_VERIFICAR` only if an equivalent verifier exists and this closure
   is unverified; otherwise `POSIBLE_ESTANCAMIENTO`.
4. Resolved sample contraction: `SEGUIR_CALCULANDO`.
5. Resolved noncontraction: `POSIBLE_ESTANCAMIENTO`.
6. Missing/incompatible evidence or overlapping intervals: `DATOS_INSUFICIENTES`.

These are exploratory labels, never actions, physical thresholds, contraction
rates or remaining-error bounds. Existing automatic-mode stopping rules remain
independent and unchanged.

## Storage, recovery and timing

Diagnostics live in `FIELD_ROOT-advice`, outside the scientific store and its
admission cap. The 4 MiB quota includes temporary replacement bytes. Previous
records are not deleted when full. Diagnostic writes reserve pending scientific
bytes and the existing filesystem reserve first.

A two-slot journal binds counters to exact proposed/current checkpoint hashes.
Normal interruption/resume restores committed counters without replay. A sink
outage can leave a diagnostic gap: no counters or retired field payloads are
reconstructed. The next full sweep starts a fresh complete segment. Malformed or
incompatible diagnostic history is preserved and disables advice, not science.
Actual scientific read/hash/encoding/admission errors still fail normally.

The deterministic plan freezes after the first complete instrumented sweep,
with at most 128 points and 64 actual payload chunks including neighbours.
Optional global region statistics cover at most the first 64 owned cores in
the existing pass; missing entries are unavailable, never a global proxy.

Active sweep timing includes commits and the closure-record fsync, excluding
stop callbacks (which may perform existing global checks). A separate timing
receipt persists the known complete interval; its own final fsync is outside
that sweep interval. Interrupted intervals are lower bounds, exclude downtime,
and cannot supply a precise cost ratio. The report shows historical verifier
cost / recent complete measured sweep median only when both are available.
External paired measurements include **all** diagnostics and persistence.
No GPU, real-volume overhead or calibrated physical accuracy is claimed.

## Tiny CPU measurement (2026-11-17)

Five paired fresh-store executions after warming both paths, alternating pair
order: 441 nodes (`7×7×9`), 12 blocks, halo 2, three sweeps, float64, local
`rtol=1e-7`, one CPU thread. Both paths used identical settings and produced
identical scientific payload hashes, sweep records and 36 observation loads.
This ordinary core route has no global verification; none was added.

| Pair | Off (ms) | On (ms) | Added (ms) |
|---|---:|---:|---:|
| 1 | 202.475 | 297.875 | 95.400 |
| 2 | 206.954 | 291.639 | 84.684 |
| 3 | 209.161 | 280.963 | 71.802 |
| 4 | 190.025 | 288.918 | 98.893 |
| 5 | 189.545 | 277.384 | 87.838 |

Off/on medians: **202.475 / 288.918 ms**. Median *paired* overhead:
**87.838 ms / 46.34%**, range **71.802–98.893 ms**. This is substantial relative
overhead on a very small workload, not evidence of negligible cost.

The frozen plan had 39 points. Across all three closures it decoded 36 payload
chunks/reads, 5,754 decoded payload bytes and 26,897 encoded payload bytes
(manifest reads/inventory checks are additional and included in elapsed time).
Diagnostic atomic-publication time was 27.1–29.3 ms per execution; serialization,
reads and all journal/sample/timing-receipt fsyncs are included in total elapsed
time. All measured sweep intervals were complete, with no pause/restart.

Environment: macOS arm64, Python 3.12.10, NumPy 2.4.6, PyTorch 2.6.0.
Isolated development core `1.5.3.dev0`, core commit `bc88881`;
`block_solver.py` SHA256:
`6beefb0f70ddafbec6e4dc313ef53b7a9c80e50911acae4fc35ea1f0aa7db406`.
Local plan workspace retains `measure_overhead.py`, `overhead.json` and the
uninstalled development wheel. No real-volume, cold-process, CUDA or physical
accuracy conclusion follows from this experiment.
