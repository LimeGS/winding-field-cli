# Automatic field convergence (opt-in)

This command is separate from `run`, `field`, `resume` and `prepared`.
Those legacy routes keep their current operators, fixed sweep budgets and guards.
No new release or production/CUDA qualification is implied.

## Start and resume

Prepare observations and fibres once using the existing workflow, with the same
planning/resource checks as usual. Use `--until fibres`, not `--until field`:

```bash
winding-field plan --source /data/ct.zarr --voxel-um 8.64 \
  --roi 0 48 0 48 0 48 --device cpu --until fibres --output runs/inputs
winding-field run --source /data/ct.zarr --voxel-um 8.64 \
  --roi 0 48 0 48 0 48 --device cpu --until fibres --output runs/inputs
winding-field field-converge runs/inputs --output runs/auto \
  --max-sweeps 32 --global-rtol 1e-6
```

Replace paths/geometry/ROI with the actual data; this is not a full-scroll sizing
recommendation. The command requires completed `PREPARED.json`/`FIBRES.json`
receipts bound by a workflow `RUN.json`, and no existing legacy `field/`.
Both bounded and ordinary prepared inputs are supported. It does not open CT,
run inference or trace fibres again.

`SOURCE` is always read-only. `--output` must be fresh, outside SOURCE (not its
ancestor either), on a local POSIX filesystem with flock/rename/fsync semantics.
Inputs and any inherited checkpoint are **private copies**, not writable
hardlinks; plan for that extra disk space. The default reserve is inherited from
SOURCE, otherwise 1 GiB. Copying checks headroom per file, solves retain the core's
resource guards, and pair construction retains its existing quota checks.

Ctrl-C, SIGTERM or `touch runs/auto/STOP` pauses cooperatively. Remove a STOP file
yourself before resuming:

```bash
rm runs/auto/STOP  # only if you created this stop marker
winding-field field-converge runs/auto --resume
```

Resume requires the same implementation, inputs, operator and saved policy.
It keeps completed sweeps, a partially completed sweep, global metrics and limits.
It does not clear STOP or reset the budget. If copying was interrupted, resume
copies only missing, hash-pinned files before solving; the original source must
still be available for those files. A killed copy may leave a `.copy-*` scratch
file at the generation root; it is never treated as an observation or fibre.
Complete generations are self-contained.

An **automatic** field that stopped with `SWEEP_LIMIT` can continue in a fresh
generation, with a larger absolute total:

```bash
winding-field field-converge runs/auto --output runs/auto-next --max-sweeps 64
```

If the first generation completed 32 sweeps, at most 32 additional sweeps remain.
Its source bytes, metrics, receipts and result remain unchanged; the new
checkpoint records the audited budget extension. The original global tolerance,
local settings and stagnation history stay in force. A paused generation instead
uses `--resume`; converged/stagnated generations are terminal.

## Exact meaning of converged

Four different claims must not be confused:

1. **Local solve convergence:** conditional block solve meets its local `rtol`,
   before saving/encoding.
2. **Sweep stability:** small `max_core_update`, diagnostic only, in field units.
3. **Global numerical convergence:** the stored field passes the criterion below.
4. **Physical-sheet correctness:** not established by this command.

The new mode uses immutable, global arclength-sampled fibre pairs. For ordinary
inputs this is an explicitly different, versioned operator from legacy
box-dependent clipping/resampling; selecting this command opts into it **before
the first field solve**. Bounded inputs use their existing global pair operator.
Solver and verifier restrict exactly the same catalogue with live exterior node
values. There is no per-block resampling or summing overlapping local residuals.

Let D contain each positive-axis nearest-neighbour edge of the global ZYX grid
once. For edge u→v along WORLD axis k:

```
w_e = (w_u + w_v)/2
g_e = spacing_k * (target_u,k + target_v,k)/2
E(phi) = 1/2 Σ_e w_e ((D phi)_e - g_e)^2
       + 1/2 Σ_i lambda_i ((C phi)_i - d_i)^2
H = Dᵀ W D + Cᵀ Lambda C
b = Dᵀ W g + Cᵀ Lambda d
```

C contains each global trilinear pair-difference row once, with fixed weight
`lambda_i=fibre_weight` and `d_i=0`. Observations retain the existing calibrated
prior (including unsupported-node regularization weights). No robust reweighting
or physical acceptance is added.

F are all free nodes. K contains the origin node with mathematical gauge
`c_K=0`; global outer faces are natural boundaries, not artificial fixed faces.
Local solve box faces condition a block update, not the global verifier.

```
R = ||(H phi_stored - b)_F||₂
B = ||b_F - H_FK c_K||₂
global_converged iff R <= global_rtol * B AND the codec gauge check passes
```

For B=0, require R=0; no substituted denominator or hidden absolute tolerance.
Norms accumulate in float64. Every free node is counted once, including halo
crossings and incomplete final cores. Constraint restrictions include exterior
contributions to both residual and fixed-node RHS elimination.

`phi_stored` is decoded from hash-verified float32 deltas plus float64 chunk
offsets, not the pre-encoding local tensor. The origin must equal the exact
codec representation `float32(-offset) + offset` of zero; logical and storage-aware
gauge errors are both reported. Residual tolerance is not relaxed for storage
precision. A tiny tolerance may be unattainable and end in stagnation; no
unproved “precision floor” is asserted.

The first complete initialized field is also verified (journal sweep 0), so an
already valid initialization needs no sweeps. Each subsequent completed sweep is
verified before another starts. The final budget sweep is explicitly verified,
even though the core has no final stop callback. Verification reads a pinned
checkpoint under the writer lease; interruption leaves verification pending,
never a falsely certified sweep or a repeated committed solve. Publication holds
the same writer lease and pins the exact verified checkpoint hash. If interrupted
after FIELD.json but before its result summary, resume validates that publication
against the journal and finishes the summary without another solve.

## Controls

| Flag | Meaning/default |
|---|---|
| `SOURCE` | Prepared workflow without a field, or an automatic generation |
| `--output` | Fresh private generation; forbidden with `--resume` |
| `--resume` | Resume SOURCE with its frozen settings/limits |
| `--max-sweeps` | Required positive integer for new generations; **absolute total**, including inherited sweeps |
| `--global-rtol` | Separate positive finite global tolerance; new default 1e-6 |
| `--field-chunk` | Block edge, 4..256 nodes; inherited, otherwise 64 |
| `--field-spacing-um` | Positive field pitch in µm for ordinary inputs; bounded inputs retain their prepared grid |
| `--rtol` | Positive local tolerance, not global; inherited, otherwise 1e-6 |
| `--maxiter` | Positive local iteration limit; inherited, otherwise 2000 |
| `--fibre-weight` | Nonnegative global pair weight; inherited, otherwise 1 |
| `--device` | Saved PyTorch device; inherited, otherwise cpu; explicit auto resolves once |
| `--threads` | Positive PyTorch CPU threads; inherited, otherwise 4 |
| `--reserve-bytes` | Nonnegative disk reserve; inherited, otherwise 1 GiB |
| `--bounded-bytes` | Inherited allowance, otherwise 12 GiB; catalogue quota is 1/8 of it, also used for ordinary automatic pairs; bounded field store uses 1/8 too |
| `--prepared-cache-bytes` | Retained workspaces, nonnegative; inherited, otherwise 0 |
| `--constraint-cache-bytes` | Retained static restrictions, nonnegative; inherited, otherwise 0 |

Resume cannot change these values. Generation continuation changes only the
explicit total sweep budget. Dtype float64, halo 2 and natural-zero-gauge match
the existing field adapter. Stagnation is fixed and persisted: after 5 verified
sweeps, stop if the best residual improved by less than 1e-3 relative to the best
residual before that window. Updates never enter this test. Convergence takes
precedence over budget exhaustion, which takes precedence over stagnation.

JSON/YAML defaults use `--cli-config FILE` (`--config` alias), explicit flags >
file > inherited settings/defaults; file paths are relative to that file:

```yaml
source: runs/inputs
output: runs/auto
max_sweeps: 32
global_rtol: 1.0e-6
```

## Results and downstream work

| Reason | Meaning | Exit |
|---|---|---|
| `VERIFIED_CONVERGED` | Stored global equations and codec gauge passed | 0 |
| `SWEEP_LIMIT` | Finite budget exhausted, not converged | 2 |
| `STAGNATED` | Insufficient residual progress, not converged | 2 |
| `PAUSED` | Cooperative stop/interruption; resumable | 130 |
| `NUMERICAL_ERROR` / `INTEGRITY_ERROR` / `RESOURCE_ERROR` | Failure during execution, never accepted | 1 |

Admission errors also exit 1 before solving. Numerical/integrity/resource errors
during solving retain a `CONVERGENCE_ERROR.json` diagnostic and propagate as
failure. Correct the underlying issue; resume is still subject to all integrity
checks. Input hashes are checked before/after verification, not just on admission;
this adds I/O and is not a throughput-optimization claim.

Files:

- `CONVERGENCE.json`: checksum-bound immutable policy/config, implementation,
  source provenance and copied-input inventory. No fabricated workflow RUN.
- `INPUTS_READY.json`: atomic copy/admission commit.
- `source/`: original source receipt/request bytes, not rewritten identities.
- `convergence/STATE.json`: checksum-bound ordered sweep metrics, exact verified
  checkpoint/operator/pair hashes, numerator, denominator, tolerance and reason.
- `CONVERGENCE_RESULT.json`: atomic terminal/pause result bound to request,
  journal and any field publication.
- `field/CHECKPOINT.json`: core-owned cursor, chunks and local sweep diagnostics.
- `FIELD.json`: **only** globally verified automatic fields. Always
  `physical_acceptance=false`.

The field-only command never dispatches meshes or flattening, regardless of the
stop reason. Automatic generations are not workflow RUNs: do not pass them to
legacy `resume`, `mesh RUN` or `export-sheets --run`. After success, downstream
tools may explicitly consume the SHA-bound checkpoint plus the geometry from
FIELD.json through their existing field-descriptor interfaces. That numerical
admission still does not prove physical sheet validity.

## Legacy and PHerc0483A

Ordinary checkpoints created **today** with the legacy provider are still
legacy; this distinction is about the operator, not age. They keep their current
usage/resume limits. Do not delete FIELD.json, rewrite RUN/implementation hashes,
graft a recovery RUN or treat a new catalogue as an equivalent continuation.

PHerc0483A's recovered field is outside the approved migration scope. This
command rejects it read-only rather than continuing it, restarting it or
certifying its four existing sweeps. No admission procedure for that recovery
format or further real sweeps is claimed. The original output/backups remain
untouched. New fields can opt into automatic convergence from their inception.

## Verification boundary

Acceptance uses tiny CPU synthetic fixtures, independently assembled dense
edge/trilinear normal equations, real multiblock solves, stored-field residuals,
crossings/final cores, interruption/crash recovery, immutable sources, corruption,
locks, legacy regressions and the installed wheel outside the checkout.
This does not establish CUDA behavior, production-scale feasibility, whole-scroll
throughput, physical-sheet correctness or validation on PHerc0483A.
