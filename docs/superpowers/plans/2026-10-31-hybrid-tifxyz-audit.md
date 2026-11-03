# Hybrid TIFXYZ Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the measured CPU/CUDA acceleration into the complete direct auditor.
**Architecture:** Bounded optional native filters, exact fallbacks, batched existing
audit traversals, unchanged publication gates and explicitly admitted GPU resources.
**Tech Stack:** Existing Python/NumPy/SQLite, optional C++17/CUDA compiler at build time.
**Spec:** `docs/superpowers/specs/2026-10-31-hybrid-tifxyz-audit-design.md`.

## Global Constraints

- Existing in-place checkout approved; no commit, install, cleanup or active-run edit.
- Preserve fine source, tolerance, support, both diagonals and independent verify.
- Aggregate RAM3GiB/reserve2GiB; scratch16GiB/reserve12.25GiB; GPU free reserve256MiB.
- Native rows<=65536; audit queues4096; SQL batches128/transactions2048.
- Explicit native artifact/device operator environment; no library path from receipts.

## Review Focus

1. Floating uncertainty/overflow/cancellation must fall back, never falsely certify.
2. Export metadata must not become a native-code loading or device-selection channel.
3. Batched contacts must retain boundary/seam contacts and the exact acceptance counters.
4. Throttled read checks must not weaken pre-write quota enforcement or STOP handling.
5. GPU virtual mappings must not bypass real RAM accounting or cause false AS failures.

## Interfaces

`tifxyz_predicates.Backend(kind)` is a context manager; `classify(mode, rows)` returns
int8[-1,0,1]. `identity(kind)` reads only trusted environment/native metadata.
Native ABI1: `wf_accel_abi`, `wf_accel_kind`, `wf_cpu_filter`, and (CUDA only)
`wf_cuda_init(capacity,reserve_bytes)`, `wf_cuda_filter`, `wf_cuda_close`,
`wf_cuda_memory(free,total)`; statuses0 success/negative argument or budget failure.
Row0 coverage: pointXYZ[0:3],output_origin[3:6],unit[6],center_lo[7:10],
center_hi[10:13],half2_lo[13],half2_hi[14],tau[15]. Remaining columns zero.
Row1 orientation: triangleXYZ[0:9],unit[9],spacingXYZ[10:13],Q1cornersZYX[13:21].
Row2 contact: firstXYZ[0:9],secondXYZ[9:18],first-triangle allowed-vertex mask[18:21].

### Task1: Native filters and explicit build

Files: `src/winding_field/native/{predicates.h,predicates.cpp,predicates.cu}`,
`tifxyz_build.py`; `tests/test_tifxyz_native.py`; package-data declaration.
- [ ] RED: compiled CPU matches exact coverage/orientation/contact oracles; cap,
  nonfinite/overflow and near-boundary cases do not falsely pass.
- [ ] Implement shared interval predicates, bounded exact integer contact filter,
  CPU/CUDA entrypoints, runtime numeric self-checks and error-safe cleanup.
- [ ] Explicit builder creates a NEW destination and a SHA/source/ABI/platform
  manifest; no runtime compilation. CUDA disables FMA/FTZ/fast-math and treats
  cross-execution-space warnings as errors.
- [ ] Run `pytest tests/test_tifxyz_native.py -q`: PASS (CUDA optional locally).

### Task2: Python backend and source binding

Files: `tifxyz_predicates.py`, `tifxyz_contract.py`, `tifxyz.py`, tests.
- [ ] RED: all three backend selections, portable defaults, trusted-env-only load,
  bad manifest/hash/ABI/platform, altered artifact and no torch/numpy eager import.
- [ ] Implement bounded NumPy filters/native wrapper, exact-unknown envelope,
  artifact/device binding and settings identity. Invalid explicit backends BLOCK.
- [ ] Run focused predicate/contract/export tests: PASS.

### Task3: Batch complete audits and eliminate repeated work

Files: `tifxyz_audit.py`, `tifxyz_store.py`, `tifxyz.py`, audit/store/export tests.
- [ ] RED: queues bound to4096, both diagonals, seams/cross-block pairs, rational
  centers and all near-boundary/negative controls agree with reference behavior.
- [ ] Batch orientation/contact work and coverage candidates, retaining exact
  fallbacks and original geometric checks. Cache only bounded immutable data.
- [ ] RED: read-only checkpoints avoid per-query directory walks, but STOP is
  responsive and write/journal limits still block and preserve prior commits.
- [ ] Batch guarded SQL preparation/transactions; expose audit phase/counters.
- [ ] Run focused audit/store/export tests and compare complete small audits: PASS.

### Task4: CUDA supervision, packaging and measured acceptance

Files: `tifxyz_runtime.py`, runtime/CLI tests, README/FORMATS and evidence.
- [ ] RED: explicit GPU UUID, CPU stays CUDA-disabled, CUDA DATA ceiling precedes
  imports, oversized allocation fails, aggregate RSS/STOP/reap still enforced.
- [ ] Implement resource mode; verification uses operator GPU opt-in, never a
  device or library path supplied by untrusted export metadata.
- [ ] Full local suite, native Linux CPU/CUDA tests, installed assets/lazy imports,
  complete old/new audit timing and separate real retained-tile audit.
- [ ] Independent whole-change review; Important findings RED→GREEN, final suite.

Execution ruling: the user explicitly requested implementation now and previously
selected native execution. This plan records that approved direction; no additional
approval pause or automatic commit. Isolated native-kernel work may run in parallel
under the frozen ABI; Python integration and final verification remain sequential.
