# Hybrid TIFXYZ audit

Approved direction: user said “Perfecto, implementalo ahora” after the validated
CPU/CUDA predicate experiment. Implement natively in the existing authorized
checkout; retain all existing work and frozen remote runs. No commits or installs.

## Goal and scope

Reduce complete audit time, not just arithmetic microbenchmarks. Preserve the
17.28um pilot tolerance, fine source, both reader diagonals, support/orientation,
intersections, exhaustive coverage, independent verification and READY-last rules.
The exact Python path remains the oracle/fallback. No physical/topological claim.

## Design

- `audit_backend`: `numpy` (default), `cpu`, or `cuda`. NumPy needs no new dependency.
- Native artifacts are explicitly built with `python -m winding_field.tifxyz_build`;
  runtime never compiles/downloads/installs. Sources ship in the ordinary wheel.
- Native manifest path comes ONLY from operator environment
  `WINDING_TIFXYZ_ACCELERATOR`, never from an export/field receipt. Pin source, ABI,
  platform, manifest and binary hashes. CUDA additionally requires an explicit
  `WINDING_TIFXYZ_CUDA_DEVICE=GPU-...` UUID. No automatic device selection.
- Modes0/1 provide outward-rounded FP64 filters for coverage/orientation. A result
  of -1 means unknown and invokes the existing exact predicate; 0/1 are certified.
  Reject unsupported numerical envelopes into the exact path, never guess a sign.
- CPU mode2 accelerates existing exact contact filters with bounded `__int128`;
  unsupported ranges and unresolved contacts use the original arbitrary-precision
  implementation. CUDA artifacts also expose this host CPU function.
- Native ABI1 uses contiguous double[N,24] input, int8[N] output, N<=65536.
  Audit queues<=4096 rows. CPU handles small batches; CUDA handles larger modes0/1.
- Coverage inputs retain exact center/half-diagonal bounds instead of rounding the
  original rational center into another point. Existing exact fallbacks keep the
  original rational arguments. Do not infer distance from a field residual.
- Batch orientation stencils and pair tests; preserve cross-block contacts and
  candidate order. Keep all source/stencil checks and deterministic acceptance
  counters. Timing/backend fallback diagnostics must not decide acceptance.
- During read-only audit work, poll STOP at least every0.1s and disk accounting
  every1s at checkpoints. Writes retain all pre-write quota/headroom enforcement.
  Batch INSERTs remain<=128 rows; group<=2048 rows per transaction and prepare/check
  each executemany statement once instead of once per row. No unsafe SQL-plan cache.
- Report audit subphase and counters through the existing STATUS callback. Verify
  remains read-only with respect to accepted artifacts.

## Resource decisions

Keep aggregate real RAM<=3GiB, RAM reserve2GiB, scratch16GiB and disk reserve12.25GiB.
CPU workers retain RLIMIT_AS. CUDA workers use RLIMIT_DATA for their child allowance
plus the existing aggregate RSS supervisor: CUDA reserves virtual addresses that
are not resident RAM. A real GTX1660 initialization passed under2.5GiB DATA with
VmSize~4.8GiB and RSS~112MiB. Record which limit was enforced; do not call DATA an AS
limit. Native GPU buffers are fixed/bounded and require256MiB free VRAM reserve.
Keep CPU affinity bounded and BLAS1; choose CPU1 for new remote trials while r3b
uses CPU0. No change to r3b or its automatic acceptance follower.

## Acceptance

Adversarial mathematical comparisons, real native builds, resource/failure tests,
full legacy suite, installed-wheel assets/lazy imports, and complete-audit timing
on identical fixtures. Validate CUDA on the approved GTX1660. Then run a separately
bound audit of retained real tiles without claiming export READY or skipping root
recertification. A real export/verify acceptance remains distinct from this audit.

## Explicit limits

This does not promise a whole-W speedup from the microbenchmark, change the source
solver, introduce a native-mesh fallback, or automatically integrate a new runtime
into an existing export. Native code is an operator-trusted optional dependency.
