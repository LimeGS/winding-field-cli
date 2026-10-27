"""Opt-in, single-writer block solves with durable progress and immutable prefetch."""

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike

from .constraint_templates import StencilConstraints
from .field_constraints import QuadraticConstraints
from .field_prefetch import ObservationPrefetch
from .field_runtime import ResourceUnavailable, check_resources
from .field_solver import solve_field
from .field_store import BlockStore, atomic_json


def _integer(value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("integer run setting outside bounds")
    return int(value)


def _vector(value, name, positive=False):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real XYZ")
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError("invalid " + name)
    return result


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _observations(value, shape):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("observation loader must return target, weights, observed, receipts")
    for array, expected, scalar in zip(value[:3], ((*shape, 3), shape, shape), ("float32", "float32", "bool")):
        if isinstance(array, torch.Tensor):
            valid = tuple(array.shape) == tuple(expected) and array.dtype == getattr(torch, scalar)
            finite = bool(torch.isfinite(array).all())
        elif isinstance(array, np.ndarray):
            valid = array.shape == tuple(expected) and array.dtype == np.dtype(scalar)
            finite = bool(np.isfinite(array).all())
        else:
            valid = finite = False
        if not valid or not finite:
            raise ValueError("observation shape/dtype/finite values differ")
    if not bool((value[1] > 0).all()):
        raise ValueError("observation weights must be strictly positive")
    receipts = value[3]
    if not isinstance(receipts, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
        for k, v in receipts.items()
    ):
        raise ValueError("observation receipts must contain named SHA256 digests")
    return value


def _record_sources(seen, receipts):
    for name, digest in receipts.items():
        if name in seen and seen[name] != digest:
            raise ValueError("observation source changed: " + name)
        seen[name] = digest


def _block_order(order, shape, chunk):
    raw = np.asarray(order, dtype=object)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("order must be Nx3 integer chunk keys")
    try:
        ordered = np.array([_integer(n) for n in raw.flat], np.int64).reshape(raw.shape)
    except OverflowError as error:
        raise ValueError("order keys exceed integer indexing capacity") from error
    grid_shape = (np.asarray(shape) + chunk - 1) // chunk
    if (
        len(ordered) != math.prod(map(int, grid_shape))
        or (ordered >= grid_shape).any()
        or len(np.unique(ordered, axis=0)) != len(ordered)
    ):
        raise ValueError("order must contain each owned chunk exactly once")
    return ordered


def solve_blocks(
    root: str | Path,
    shape: Sequence[int],
    chunk: int,
    identity: dict[str, Any],
    load_observations: Callable[..., Any],
    *,
    origin_xyz: ArrayLike,
    spacing_xyz: ArrayLike,
    order: Sequence[Sequence[int]],
    constraints: StencilConstraints | Callable[..., QuadraticConstraints | None] | None = None,
    fixed: Callable[..., ArrayLike] | None = None,
    halo: int = 1,
    sweeps: int = 1,
    rtol: float = 1e-5,
    maxiter: int = 4000,
    dtype: Literal["float32", "float64"] = "float64",
    device: str = "cpu",
    cache_bytes: int = 0,
    read_workers: int = 1,
    max_bytes: int | None = None,
    reserve_bytes: int = 0,
    resume: bool = False,
    stop: Callable[[], bool] | None = None,
    admit: Callable[[], Any] | None = None,
    prefetch: ObservationPrefetch | None = None,
) -> dict[str, Any]:
    """Solve each explicit owned core, persisting its exact next-block cursor.

    The caller's nonempty JSON identity must pin source/provider/code identity,
    coordinate frame, calibration and boundary policy. Geometry, traversal,
    stencil digest and numerical settings are additionally bound automatically.
    Readers are immutable; constraint targets/fixed/exterior values are never prefetched.
    Only immutable gradient observations may be prepared on a secondary device.

    Internal solve faces read the current shared field; global faces are natural
    except caller-supplied exact fixed values. With no supplied fixed provider,
    the global first node is the zero gauge. ``order`` lists every chunk once.
    Stop/admission are checked at block boundaries and immediately before solve
    and commit. A long solver call is not a hard real-time interruption point.

    Local convergence applies before core quantization, not to the decoded core
    or assembled field. Budget exhaustion is explicitly not global convergence.
    Cache/workers/storage budgets may change on resume; the numerical problem may
    not. The caller budgets observations, solver workspace and optional prefetch
    in ``admit``. This function closes a supplied prefetch client on exit.
    Large fibre archives can supply ``constraints(lo, hi, read_nodes)`` selecting
    complete groups per block. Pin that provider/geometry in ``identity``; it must
    return local QuadraticConstraints and read current exterior values each time.
    """
    try:
        shape = tuple(_integer(n, 2) for n in shape)
        if len(shape) != 3 or math.prod(shape) > np.iinfo(np.intp).max:
            raise ValueError("invalid ZYX run shape")
        chunk, halo, sweeps, maxiter = _integer(chunk, 1), _integer(halo), _integer(sweeps, 1), _integer(maxiter)
        origin, spacing = _vector(origin_xyz, "origin"), _vector(spacing_xyz, "spacing", True)
        if (
            dtype not in ("float32", "float64")
            or not isinstance(rtol, (int, float))
            or not math.isfinite(rtol)
            or rtol <= 0
        ):
            raise ValueError("invalid dtype or positive solver tolerance")
        if not isinstance(identity, dict) or not identity or not callable(load_observations):
            raise ValueError("nonempty source identity and observation loader required")
        if fixed is not None and not callable(fixed):
            raise ValueError("fixed must be a bounds callable")
        for callback in (stop, admit):
            if callback is not None and not callable(callback):
                raise ValueError("stop/admit must be callable")
        ordered = _block_order(order, shape, chunk)
        if halo == 0 and len(ordered) > 1:
            raise ValueError("a positive halo is required between multiple owned cores")
        constraint_identity = None
        if isinstance(constraints, StencilConstraints):
            if constraints.shape != shape:
                raise ValueError("constraint/grid identity differs")
            constraint_identity = constraints.digest
        elif constraints is not None:
            if not callable(constraints):
                raise ValueError("constraints must be static stencils or a block provider")
            constraint_identity = "caller-bound-provider"
        if prefetch is not None and tuple(prefetch.shape) != shape:
            raise ValueError("prefetch/grid identity differs")
        problem = dict(
            schema="winding-block-solve-v1",
            source_identity=deepcopy(identity),
            shape=list(shape),
            chunk=chunk,
            origin_xyz=origin.tolist(),
            spacing_xyz=spacing.tolist(),
            order_sha256=hashlib.sha256(ordered.astype("<i8").tobytes()).hexdigest(),
            constraints=constraint_identity,
            boundary="caller-fixed" if fixed is not None else "natural-zero-gauge",
            halo=halo,
            sweeps=sweeps,
            rtol=float(rtol),
            maxiter=maxiter,
            dtype=dtype,
            device=str(device),
        )
        json.dumps(problem, allow_nan=False)
        root, started = Path(root), time.perf_counter()
        prefetched, parity = 0, False
        with BlockStore(
            root,
            shape,
            chunk,
            problem,
            resume=resume,
            cache_bytes=cache_bytes,
            workers=read_workers,
            max_bytes=max_bytes,
            reserve_bytes=reserve_bytes,
        ) as store:
            manifest = store.manifest
            progress = manifest["progress"]
            stored_bytes = sum(v["bytes"] for v in manifest["chunks"].values())
            if not progress:
                progress = dict(sweep=1, next_block=0, sweep_max_update=0.0, completed_sweeps=[], source_receipts={})
                store.save_progress(progress)
            first_sweep = _integer(progress.get("sweep"), 1)
            first_block = _integer(progress.get("next_block"))
            if first_sweep > sweeps or first_block > len(ordered):
                raise ValueError("checkpoint cursor exceeds run budget")
            maximum = progress.get("sweep_max_update")
            completed, seen = progress.get("completed_sweeps"), progress.get("source_receipts")
            if (
                type(maximum) not in (int, float)
                or not math.isfinite(maximum)
                or maximum < 0
                or not isinstance(completed, list)
                or not isinstance(seen, dict)
                or len(completed) != first_sweep - int(first_block < len(ordered))
            ):
                raise ValueError("invalid persisted sweep statistics/source receipts")
            for i, value in enumerate(completed, 1):
                if (
                    not isinstance(value, dict)
                    or value.get("sweep") != i
                    or value.get("cores") != len(ordered)
                    or type(value.get("max_core_update")) not in (int, float)
                    or not math.isfinite(value["max_core_update"])
                    or value["max_core_update"] < 0
                ):
                    raise ValueError("invalid completed sweep statistics")
            if any(
                not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
                for k, v in seen.items()
            ):
                raise ValueError("invalid persisted source receipts")

            def emit(status, **extra):
                result = dict(
                    status=status,
                    pid=os.getpid(),
                    updated_epoch=time.time(),
                    checkpoint=str(root / "CHECKPOINT.json"),
                    sweep=progress["sweep"],
                    next_block=progress["next_block"],
                    total_blocks=len(ordered),
                    completed_sweeps=completed,
                    global_converged=False,
                    physical_acceptance=False,
                    stored_local_residual=None,
                    stored_bytes=stored_bytes,
                    cache_stats=dict(store.stats),
                    prefetched_blocks=prefetched,
                    prefetch_parity=parity,
                    elapsed_s=time.perf_counter() - started,
                    **extra,
                )
                atomic_json(root / "STATUS.json", result)
                return result

            def stopped():
                return (root / "STOP").exists() or (stop is not None and bool(stop()))

            def admitted():
                check_resources(root, reserve_bytes=reserve_bytes)
                if admit is not None:
                    admit()

            def bounds(key):
                low = key * chunk
                high = np.minimum(low + chunk, shape)
                return low, high, np.maximum(low - halo, 0), np.minimum(high + halo, shape)

            try:
                if stopped():
                    return emit("CHECKPOINTED_STOP")
                if first_block == len(ordered):
                    first_sweep, first_block, maximum = first_sweep + 1, 0, 0.0
                for sweep in range(first_sweep, sweeps + 1):
                    if sweep != first_sweep:
                        maximum = 0.0
                    for block in range(first_block if sweep == first_sweep else 0, len(ordered)):
                        if stopped():
                            return emit("CHECKPOINTED_STOP")
                        admitted()
                        key = ordered[block]
                        low, high, lo, hi = bounds(key)
                        crop_shape = tuple(map(int, hi - lo))
                        emit("RUNNING", phase="observations", active_sweep=sweep, active_block=block)
                        fetched = None
                        if prefetch is not None and prefetch.available:
                            if prefetch.pending is None:
                                prefetch.submit(lo, hi)
                            fetched = prefetch.receive(lo, hi)
                        if fetched is None:
                            values = _observations(load_observations(lo, hi), crop_shape)
                        else:
                            values = _observations(fetched, crop_shape)
                            if not parity:
                                primary = _observations(load_observations(lo, hi), crop_shape)
                                if (
                                    any(
                                        not np.array_equal(_numpy(a), _numpy(b))
                                        for a, b in zip(values[:3], primary[:3])
                                    )
                                    or values[3] != primary[3]
                                ):
                                    raise ValueError("primary/secondary observation parity failed")
                                parity = True
                                del primary
                            prefetched += 1
                        target, weights, observed, receipts = values
                        _record_sources(seen, receipts)
                        if prefetch is not None and prefetch.available and block + 1 < len(ordered):
                            _, _, next_lo, next_hi = bounds(ordered[block + 1])
                            prefetch.submit(next_lo, next_hi)
                        current = store.read(lo, hi)
                        if constraints is None:
                            rows = None
                        elif isinstance(constraints, StencilConstraints):
                            rows = constraints.restrict(lo, hi, store.nodes)
                        else:
                            rows = constraints(lo, hi, store.nodes)
                        if rows is not None and not isinstance(rows, QuadraticConstraints):
                            raise ValueError("constraint provider must return local QuadraticConstraints")
                        fixed_values = np.full(crop_shape, np.nan)
                        for axis in range(3):
                            for side, internal in ((0, lo[axis] > 0), (-1, hi[axis] < shape[axis])):
                                if internal:
                                    plane: list[slice | int] = [slice(None)] * 3
                                    plane[axis] = side
                                    fixed_values[tuple(plane)] = current[tuple(plane)]
                        if fixed is None:
                            if not lo.any():
                                fixed_values[0, 0, 0] = 0.0
                        else:
                            declared = np.asarray(fixed(lo, hi))
                            if (
                                declared.shape != crop_shape
                                or declared.dtype.kind not in "iuf"
                                or np.isinf(declared).any()
                            ):
                                raise ValueError("invalid fixed-value observation")
                            declared = declared.astype(np.float64)
                            label = "fixed:" + ",".join(map(str, (*lo, *hi)))
                            _record_sources(seen, {label: hashlib.sha256(declared.tobytes()).hexdigest()})
                            valid = np.isfinite(declared)
                            fixed_values[valid] = declared[valid]
                        if stopped():
                            return emit("CHECKPOINTED_STOP")
                        admitted()
                        emit("RUNNING", phase="solving", active_sweep=sweep, active_block=block)
                        result, solver = solve_field(
                            target,
                            weights,
                            spacing,
                            constraints=rows,
                            fixed=fixed_values,
                            initial=current,
                            device=device,
                            dtype=dtype,
                            rtol=rtol,
                            maxiter=maxiter,
                            return_tensor=True,
                        )
                        if not solver["converged"]:
                            raise RuntimeError("conditional block did not converge")
                        owned = tuple(slice(int(a), int(b)) for a, b in zip(low - lo, high - lo))
                        core = _numpy(result[owned])
                        change = float(np.max(abs(core - current[owned])))
                        maximum = max(maximum, change)
                        next_completed = list(completed)
                        if block + 1 == len(ordered):
                            next_completed.append(dict(sweep=sweep, cores=len(ordered), max_core_update=maximum))
                        # Report precision loss separately from the pre-encoding local gate.
                        encoded_values = core.astype(np.float64, copy=False)
                        offset = encoded_values.mean()
                        restored = (encoded_values - offset).astype(np.float32).astype(np.float64) + offset
                        anchor = np.isfinite(fixed_values[owned])
                        fixed_error = float(np.max(abs(restored[anchor] - fixed_values[owned][anchor]), initial=0))
                        pending = dict(
                            sweep=sweep,
                            next_block=block + 1,
                            sweep_max_update=maximum,
                            completed_sweeps=next_completed,
                            source_receipts=seen,
                            solver=dict(solver, scope="pre_encoding_local_solve"),
                            core_max_update=change,
                            decoded_fixed_value_error_max=fixed_error,
                            global_converged=False,
                            physical_acceptance=False,
                        )
                        admitted()
                        commit = store.commit(key, core, pending, _numpy(observed[owned]))
                        stored_bytes = commit["stored_bytes"]
                        progress, completed = pending, next_completed
                        emit("RUNNING", phase="checkpointed")
                        del (
                            target,
                            weights,
                            observed,
                            values,
                            fetched,
                            current,
                            rows,
                            result,
                            core,
                            fixed_values,
                            restored,
                            encoded_values,
                        )
                return emit("SWEEP_BUDGET_REACHED")
            except Exception as error:
                emit("RESOURCE_UNAVAILABLE" if isinstance(error, ResourceUnavailable) else "FAILED", error=str(error))
                raise
    finally:
        if prefetch is not None:
            prefetch.close()
