"""Matrix-free weighted gradient/fibre least squares on a nonperiodic ZYX grid."""

import math
import time
from functools import partial
from typing import Any, Literal, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray


class _FieldOperator:
    """Unanchored graph Laplacian on validated tensors; shared by solve and tests."""

    def __init__(self, weights, pairs, lam):
        self.edges = []
        self.diagonal = torch.zeros_like(weights)
        for axis in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            w = 0.5 * weights[lo] + 0.5 * weights[hi]
            self.edges.append((lo, hi, w))
            self.diagonal[lo] += w
            self.diagonal[hi] += w
        self.pairs = None
        self.lam = lam
        if pairs is not None and pairs.numel() and lam:
            # Self-pairs contribute zero to both the operator and its diagonal.
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            self.pairs = pairs[:, 0], pairs[:, 1]
            counts = torch.full((len(pairs),), lam, dtype=weights.dtype, device=weights.device)
            for endpoint in self.pairs:
                self.diagonal.view(-1).index_add_(0, endpoint, counts)

    def matvec(self, field):
        out = torch.zeros_like(field)
        for lo, hi, weight in self.edges:
            flux = (field[hi] - field[lo]) * weight
            out[lo] -= flux
            out[hi] += flux
        if self.pairs is not None:
            p, q = self.pairs
            delta = self.lam * (field.view(-1)[p] - field.view(-1)[q])
            out.view(-1).index_add_(0, p, delta)
            out.view(-1).index_add_(0, q, -delta)
        return out

    def rhs(self, target_xyz, spacing_xyz):
        out = torch.zeros_like(self.diagonal)
        for axis, (lo, hi, weight) in enumerate(self.edges):
            g = target_xyz[..., 2 - axis]
            flux = weight * ((0.5 * g[lo] + 0.5 * g[hi]) * spacing_xyz[2 - axis])
            out[lo] -= flux
            out[hi] += flux
        return out


def _scalar(value, name, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _array(value, name, device, shape=None, allow_nan=False, *, dtype=torch.float32):
    is_complex = value.is_complex() if isinstance(value, torch.Tensor) else np.iscomplexobj(value)
    if is_complex:
        raise ValueError(f"{name} must be real")
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    try:
        result = torch.as_tensor(value, dtype=dtype, device=device).detach()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real array") from exc
    if shape is not None and tuple(result.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    invalid = torch.isinf(result) if allow_nan else ~torch.isfinite(result)
    if invalid.any():
        raise ValueError(f"{name} contains nonfinite values (or exceeds {dtype} range)")
    return result.contiguous()


def _norm(value):
    # Squaring small/large float32 residuals can falsely report zero/infinity.
    return float(torch.linalg.vector_norm(value, dtype=torch.float64))


def _constraint_operator(constraints, device, dtype):
    if constraints is None:
        return None
    if dtype == torch.float32:
        return constraints.operator(device)  # Preserve the existing device-only protocol.
    return constraints.operator(device, np.float64, layout="csr")


class PreparedField:
    """Own static operators for one block; anchors and constraint targets stay live.

    Retains one target/weight snapshot and static constraint arrays. Reuse compares
    values as well as geometry/dtype/device, including mutations through NumPy
    aliases that bypass Torch version counters. Retention is not a peak-RAM bound.
    """

    def __init__(self) -> None:
        self.signature: tuple[Any, ...] | None = None

    def bind(self, target, weights, spacing, constraints):
        arrays = () if constraints is None else constraints._structural_arrays()
        rows = None if constraints is None else constraints.matrix.shape
        signature = (tuple(target.shape), target.dtype, target.device, tuple(spacing), rows)
        if self.signature is None:
            self.target, self.weights = target.clone(), weights.clone()
            self.structure = tuple(a.copy() for a in arrays)
            self.operator = _FieldOperator(self.weights, None, 0)
            self.extra = _constraint_operator(constraints, weights.device, weights.dtype)
            self.gradient_rhs = self.operator.rhs(self.target, spacing)
            self.signature = signature
        elif (
            self.signature != signature
            or not torch.equal(target, self.target)
            or not torch.equal(weights, self.weights)
            or len(arrays) != len(self.structure)
            or any(not np.array_equal(a, b) for a, b in zip(arrays, self.structure))
        ):
            raise ValueError("prepared field inputs changed; create a new workspace")
        if self.extra is not None:
            self.extra.target = _array(
                constraints.targets, "targets", weights.device, (constraints.matrix.shape[0],), dtype=weights.dtype
            ).clone()
        return self.operator, self.extra, self.gradient_rhs.clone()


@overload
def solve_field(
    target_xyz: ArrayLike | torch.Tensor,
    weights: ArrayLike | torch.Tensor,
    spacing_xyz: ArrayLike | torch.Tensor = (1, 1, 1),
    pairs: ArrayLike | torch.Tensor | None = None,
    lam: float = 3.0,
    fixed: ArrayLike | torch.Tensor | None = None,
    reference: ArrayLike | torch.Tensor | None = None,
    reference_weight: ArrayLike | torch.Tensor = 0.0,
    initial: ArrayLike | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    rtol: float = 1e-5,
    maxiter: int | np.integer[Any] = 4000,
    constraints: Any = None,
    *,
    dtype: Literal["float32"] = "float32",
    prepared: PreparedField | None = None,
    return_tensor: Literal[False] = False,
) -> tuple[NDArray[np.float32], dict[str, Any]]: ...


@overload
def solve_field(
    target_xyz: ArrayLike | torch.Tensor,
    weights: ArrayLike | torch.Tensor,
    spacing_xyz: ArrayLike | torch.Tensor = (1, 1, 1),
    pairs: ArrayLike | torch.Tensor | None = None,
    lam: float = 3.0,
    fixed: ArrayLike | torch.Tensor | None = None,
    reference: ArrayLike | torch.Tensor | None = None,
    reference_weight: ArrayLike | torch.Tensor = 0.0,
    initial: ArrayLike | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    rtol: float = 1e-5,
    maxiter: int | np.integer[Any] = 4000,
    constraints: Any = None,
    *,
    dtype: Literal["float64"],
    prepared: PreparedField | None = None,
    return_tensor: Literal[False] = False,
) -> tuple[NDArray[np.float64], dict[str, Any]]: ...


@overload
def solve_field(
    target_xyz: ArrayLike | torch.Tensor,
    weights: ArrayLike | torch.Tensor,
    spacing_xyz: ArrayLike | torch.Tensor = (1, 1, 1),
    pairs: ArrayLike | torch.Tensor | None = None,
    lam: float = 3.0,
    fixed: ArrayLike | torch.Tensor | None = None,
    reference: ArrayLike | torch.Tensor | None = None,
    reference_weight: ArrayLike | torch.Tensor = 0.0,
    initial: ArrayLike | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    rtol: float = 1e-5,
    maxiter: int | np.integer[Any] = 4000,
    constraints: Any = None,
    *,
    dtype: Literal["float32", "float64"] = "float32",
    prepared: PreparedField | None = None,
    return_tensor: Literal[True],
) -> tuple[torch.Tensor, dict[str, Any]]: ...


@overload
def solve_field(
    target_xyz: ArrayLike | torch.Tensor,
    weights: ArrayLike | torch.Tensor,
    spacing_xyz: ArrayLike | torch.Tensor = (1, 1, 1),
    pairs: ArrayLike | torch.Tensor | None = None,
    lam: float = 3.0,
    fixed: ArrayLike | torch.Tensor | None = None,
    reference: ArrayLike | torch.Tensor | None = None,
    reference_weight: ArrayLike | torch.Tensor = 0.0,
    initial: ArrayLike | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    rtol: float = 1e-5,
    maxiter: int | np.integer[Any] = 4000,
    constraints: Any = None,
    *,
    dtype: Literal["float32", "float64"] = "float32",
    prepared: PreparedField | None = None,
    return_tensor: bool = False,
) -> tuple[NDArray[np.float32] | NDArray[np.float64] | torch.Tensor, dict[str, Any]]: ...


@torch.no_grad()
def solve_field(
    target_xyz: ArrayLike | torch.Tensor,
    weights: ArrayLike | torch.Tensor,
    spacing_xyz: ArrayLike | torch.Tensor = (1, 1, 1),
    pairs: ArrayLike | torch.Tensor | None = None,
    lam: float = 3.0,
    fixed: ArrayLike | torch.Tensor | None = None,
    reference: ArrayLike | torch.Tensor | None = None,
    reference_weight: ArrayLike | torch.Tensor = 0.0,
    initial: ArrayLike | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    rtol: float = 1e-5,
    maxiter: int | np.integer[Any] = 4000,
    constraints: Any = None,
    *,
    dtype: Literal["float32", "float64"] = "float32",
    prepared: PreparedField | None = None,
    return_tensor: bool = False,
) -> tuple[NDArray[np.float32] | NDArray[np.float64] | torch.Tensor, dict[str, Any]]:
    """Return a ZYX field and a JSON-compatible solver receipt (float32 by default).

    Minimize sum_edges w_e*(phi[v]-phi[u]-h_e*mean(g_e))**2 +
    lam*sum_pairs (phi[p]-phi[q])**2 + sum(reference_weight*(phi-reference)**2).
    Edge weights are arithmetic means of strictly positive nodal weights.
    Gradients have XYZ components in field units per physical coordinate unit;
    spacing_xyz contains physical distances between grid samples. Pairs are an
    integer (M, 2) array of flat ZYX indices; duplicates retain their multiplicity.

    fixed is ZYX, NaN at free nodes, finite at exact constraints (represented in
    the selected dtype). reference, when supplied, is finite everywhere; reference_weight
    is a finite nonnegative scalar or a ZYX array of nodal weights (zero disables
    a node's penalty). Without effective anchors, node zero is pinned to zero.
    No recentering is applied. Inputs are not mutated; tensors need no gradients.
    Array inputs accept NumPy arrays, nested sequences or PyTorch tensors;
    shapes, real values and finite ranges are validated at runtime.

    PCG uses exact fixed-node elimination and a cached Jacobi diagonal. Residuals
    are measured on free equations against the reduced RHS (absolute residual
    if that RHS is zero), and recomputed before reporting convergence. Numerical
    breakdown or exhaustion returns converged=False with a reason; callers MUST
    gate publication on converged. Invalid inputs/overflow during assembly raise
    ValueError. An unrepresentable residual is reported as None, never JSON NaN.
    elapsed_s includes transfers and setup; no files are written.

    Explicit dtype='float64' preserves float64 inputs/output and fixed values.
    return_tensor=True avoids the final device-to-host copy. A PreparedField
    reuses one block's static gradient and continuous-row operators; discrete
    pairs and reference penalties are unsupported in that optional path. Current
    fixed values and constraint targets are assembled anew on every call.
    """
    start = time.perf_counter()
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if type(return_tensor) is not bool:
        raise ValueError("return_tensor must be boolean")
    scalar = torch.float64 if dtype == "float64" else torch.float32
    array = partial(_array, dtype=scalar)
    lam = _scalar(lam, "lam")
    rtol = _scalar(rtol, "rtol", positive=True)
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer)) or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer")
    target = array(target_xyz, "target_xyz", device)
    if target.ndim != 4 or target.shape[-1] != 3 or min(target.shape[:3]) < 1:
        raise ValueError("target_xyz must have nonempty shape (Z, Y, X, 3)")
    shape = tuple(target.shape[:3])
    nodal = (reference_weight.ndim if isinstance(reference_weight, torch.Tensor) else np.ndim(reference_weight)) != 0
    rw = (
        array(reference_weight, "reference_weight", device, shape)
        if nodal
        else _scalar(reference_weight, "reference_weight")
    )
    if nodal and (rw < 0).any():
        raise ValueError("reference_weight must be nonnegative")
    has_reference = bool((rw > 0).any()) if nodal else rw > 0
    weight = array(weights, "weights", device, shape)
    if (weight <= 0).any():
        raise ValueError("weights must be strictly positive")
    spacing = array(spacing_xyz, "spacing_xyz", "cpu", (3,))
    if (spacing <= 0).any():
        raise ValueError("spacing_xyz must be strictly positive")
    pairs_t: torch.Tensor | None = None
    if pairs is not None:
        try:
            pairs_t = torch.as_tensor(pairs, device=device)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pairs must be an integer (M, 2) array") from exc
        if pairs_t.numel() == 0 and pairs_t.ndim == 1:
            pairs_t = pairs_t.reshape(0, 2).long()
        elif pairs_t.shape == (0, 2):
            pairs_t = pairs_t.long()
        if (
            pairs_t.ndim != 2
            or pairs_t.shape[1] != 2
            or pairs_t.is_floating_point()
            or pairs_t.is_complex()
            or pairs_t.dtype == torch.bool
        ):
            raise ValueError("pairs must be an integer (M, 2) array")
        pairs_t = pairs_t.long()
        if ((pairs_t < 0) | (pairs_t >= weight.numel())).any():
            raise ValueError("pair indices are outside the flat ZYX field")
    if has_reference and reference is None:
        raise ValueError("positive reference_weight requires reference")
    ref = None if reference is None else array(reference, "reference", device, shape)
    fixed_values = torch.zeros_like(weight)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if fixed is not None:
        f = array(fixed, "fixed", device, shape, allow_nan=True)
        mask = torch.isfinite(f)
        fixed_values[mask] = f[mask]
    if mask.any():
        gauge = dict(kind="fixed", count=int(mask.sum()))
    elif has_reference:
        gauge = (
            dict(kind="reference", weighting="nodal", count=int((rw > 0).sum()))
            if nodal
            else dict(kind="reference", weight=rw)
        )
    else:
        mask.view(-1)[0] = True
        gauge = dict(kind="zero", flat_index=0, value=0.0)
    x = torch.zeros_like(weight) if initial is None else array(initial, "initial", device, shape).clone()
    x[mask] = 0
    if prepared is None:
        operator = _FieldOperator(weight, pairs_t, lam)
        extra = _constraint_operator(constraints, device, scalar)
        b = operator.rhs(target, spacing.tolist())
    else:
        if not isinstance(prepared, PreparedField) or pairs is not None or has_reference:
            raise ValueError("prepared blocks require continuous rows and no reference penalty")
        operator, extra, b = prepared.bind(target, weight, spacing.tolist(), constraints)
    if extra is not None and len(extra.diagonal) != weight.numel():
        raise ValueError("continuous constraints do not match grid")
    if extra is not None:
        b += extra.rhs().reshape(shape)
    if has_reference:
        if nodal:
            b.addcmul_(ref, rw)
        else:
            b.add_(ref, alpha=rw)
    # Eliminate fixed columns; fixed rows are omitted, not softly penalized.
    b -= operator.matvec(fixed_values)
    if extra is not None:
        b -= extra.matvec(fixed_values)
    b[mask] = 0
    diagonal = operator.diagonal + rw
    if extra is not None:
        diagonal += extra.diagonal.reshape(shape)
    diagonal[mask] = 1
    if not torch.isfinite(b).all() or not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise ValueError("assembled system is not finite with a positive free diagonal")
    inverse = diagonal.reciprocal()
    del target, weight, ref, diagonal

    def matvec(p):
        out = operator.matvec(p)
        if extra is not None:
            out += extra.matvec(p)
        if has_reference:
            if nodal:
                out.addcmul_(p, rw)
            else:
                out.add_(p, alpha=rw)
        out[mask] = 0
        return out

    scale = _norm(b)
    if not math.isfinite(scale):
        raise ValueError("assembled RHS norm is not finite")
    scale = scale or 1.0
    r = b - matvec(x)
    z = r * inverse
    p = z.clone()
    rz = float(torch.dot(r.view(-1), z.view(-1)))
    iterations = 0
    reason = "maxiter"
    for _ in range(maxiter):
        norm = _norm(r)
        if not math.isfinite(norm):
            reason = "nonfinite_residual"
            break
        if norm <= rtol * scale:
            break
        ap = matvec(p)
        curvature = float(torch.dot(p.view(-1), ap.view(-1)))
        if not math.isfinite(curvature) or not math.isfinite(rz):
            reason = "nonfinite_curvature"
            break
        if curvature <= 0 or rz <= 0:
            reason = "nonpositive_curvature"
            break
        alpha = rz / curvature
        x.add_(p, alpha=alpha)
        r.add_(ap, alpha=-alpha)
        iterations += 1
        # Recurrence residuals drift in float32: verify, and restart if needed.
        restart = _norm(r) <= rtol * scale
        if restart:
            r = b - matvec(x)
        z = r * inverse
        next_rz = float(torch.dot(r.view(-1), z.view(-1)))
        if restart:
            p = z.clone()
        else:
            p.mul_(next_rz / rz).add_(z)
        rz = next_rz
    residual = _norm(b - matvec(x)) / scale
    converged = math.isfinite(residual) and residual <= rtol
    if not math.isfinite(residual):
        residual, reason = None, "nonfinite_residual"
    x[mask] = fixed_values[mask]
    field = x if return_tensor else x.cpu().numpy()
    info = dict(
        converged=converged,
        iterations=iterations,
        relative_residual=residual,
        elapsed_s=time.perf_counter() - start,
        gauge=gauge,
        reason="converged" if converged else reason,
        dtype=dtype,
    )
    return field, info
