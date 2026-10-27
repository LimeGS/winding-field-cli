"""Fractional sheet density in input voxels and unsigned XYZ surface normals."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def _sum_correlation(padded: torch.Tensor, weights: list[float], size: int) -> torch.Tensor:
    result = padded.new_zeros((*padded.shape[:-1], size))
    for index, weight in enumerate(weights):
        result.add_(padded[..., index : index + size], alpha=weight)
    return result


def _gaussian(value: torch.Tensor, sigma: float, order: tuple[int, int, int] = (0, 0, 0)) -> torch.Tensor:
    """Separable SciPy-default (truncate=4, half-sample reflect) filtering."""
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    for axis, derivative in enumerate(order):
        # conv1d is correlation, so the first-derivative kernel has +x sign.
        weight = kernel * x / (sigma * sigma) if derivative else kernel
        lines = value.movedim(axis, -1)
        shape = lines.shape
        size = shape[-1]
        index = torch.arange(-radius, size + radius, device=value.device) % (2 * size)
        index = torch.where(index < size, index, 2 * size - 1 - index)
        padded = lines.index_select(-1, index).reshape(-1, 1, size + 2 * radius)
        if value.device.type == "cuda":
            # Avoid cuDNN's costly large-batch single-channel convolution path.
            filtered = _sum_correlation(padded, weight.tolist(), size)
        else:
            filtered = F.conv1d(padded, weight.to(device=value.device, dtype=value.dtype).reshape(1, 1, -1))
        value = filtered.reshape(shape).movedim(-1, axis)
    return value


def _eigh(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ascending eigenvalues and principal director of packed PSD 3x3 tensors.

    Packing is (00, 01, 02, 11, 12, 22). Analytic, trace-normalized batches
    bound temporary memory and avoid per-voxel LAPACK/cuSOLVER calls.
    A zero/isotropic tensor returns a zero director, not an invented axis.
    """
    flat = tensor.reshape(-1, 6)
    values = tensor.new_empty((len(flat), 3))
    normals = torch.empty_like(values)
    tiny = torch.finfo(tensor.dtype).tiny
    for start in range(0, len(flat), 262144):
        stop = min(start + 262144, len(flat))
        block = flat[start:stop]
        scale = (block[:, 0] + block[:, 3] + block[:, 5]).clamp_min(tiny)
        a, b, c, d, e, f = (block / scale[:, None]).unbind(-1)
        q = (a + d + f) / 3
        aa, dd, ff = a - q, d - q, f - q
        p = ((aa * aa + dd * dd + ff * ff + 2 * (b * b + c * c + e * e)) / 6).sqrt()
        denominator = p.clamp_min(tiny)
        ba, bb, bc, bd, be, bf = [v / denominator for v in (aa, b, c, dd, e, ff)]
        determinant = ba * bd * bf + 2 * bb * bc * be - ba * be * be - bd * bc * bc - bf * bb * bb
        angle = ((determinant / 2).clamp(-1, 1)).acos() / 3
        largest = q + 2 * p * angle.cos()
        smallest = q + 2 * p * (angle + 2 * math.pi / 3).cos()
        middle = 3 * q - largest - smallest
        values[start:stop] = torch.stack((smallest, middle, largest), -1) * scale[:, None]

        rows = (
            torch.stack((a - largest, b, c), -1),
            torch.stack((b, d - largest, e), -1),
            torch.stack((c, e, f - largest), -1),
        )
        normal = torch.linalg.cross(rows[0], rows[1])
        norm2 = normal.square().sum(-1)
        for i, j in ((0, 2), (1, 2)):
            candidate = torch.linalg.cross(rows[i], rows[j])
            candidate_norm2 = candidate.square().sum(-1)
            use = candidate_norm2 > norm2
            normal = torch.where(use[:, None], candidate, normal)
            norm2 = torch.maximum(norm2, candidate_norm2)
        normals[start:stop] = normal / norm2.sqrt().clamp_min(tiny)[:, None]
    shape = (*tensor.shape[:-1], 3)
    return values.reshape(shape), normals.reshape(shape)


@torch.no_grad()
def density_normals(
    volume: np.ndarray | torch.Tensor, *, halo: int = 40, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (density, normals_xyz, supported) for the halo-trimmed ZYX core.

    Input is a NumPy or Torch uint8 ZYX volume. Outputs are float32 ZYX,
    float32 ZYXx3, and boolean ZYX tensors on ``device`` (CPU or CUDA).
    Density is sheets per INPUT voxel; physical conversion and orientation
    belong to the caller. Directors are unsigned surface normals, zero where
    locally incoherent or unsupported. Support is Gaussian(coherent, 6) > .2.

    The recipe thresholds material at >127, uses Gaussian derivatives at 1,
    a structure tensor smoothed at 2, occupancy >.02 and anisotropy >.3,
    then symmetric fractional crossings /4 and density smoothing at 6.
    Halo 40 covers the complete filter support. Smaller nonnegative integer
    halos are accepted, but their core can depend on reflected boundaries.
    The caller owns tiling and supplying the halo; this processes one volume.
    """
    if not isinstance(volume, (np.ndarray, torch.Tensor)):
        raise TypeError("volume must be a NumPy or Torch uint8 ZYX array")
    if volume.dtype not in (np.dtype("uint8"), torch.uint8):
        raise TypeError("volume must have uint8 dtype")
    if volume.ndim != 3:
        raise ValueError("volume must be ZYX (three dimensions)")
    if isinstance(halo, (bool, np.bool_)) or not isinstance(halo, (int, np.integer)) or halo < 0:
        raise ValueError("halo must be a nonnegative integer")
    if any(size <= 2 * halo for size in volume.shape):
        raise ValueError("volume must have a nonempty core beyond the halo")
    material = (torch.as_tensor(volume, device=device) > 127).to(torch.float32)
    # Preserve FP32 filtering even inside a caller's mixed-precision context.
    with torch.autocast(device_type=material.device.type, enabled=False), torch.backends.cudnn.flags(allow_tf32=False):
        gradients = [_gaussian(material, 1.0, order) for order in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        tensor = material.new_empty((*material.shape, 6))
        for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            tensor[..., k] = _gaussian(gradients[i] * gradients[j], 2.0)
        del gradients
        eigenvalues, normals = _eigh(tensor)
        del tensor
        anisotropy = (eigenvalues[..., 2] - eigenvalues[..., 1]) / eigenvalues[..., 2].clamp_min(1e-12)
        coherent = (_gaussian(material, 2.0) > 0.02) & (anisotropy > 0.3)
        del eigenvalues, anisotropy
        normals.masked_fill_(~coherent[..., None], 0)

        crossings = torch.zeros_like(material)
        grid = torch.empty_like(normals)
        for sign in (-1.0, 1.0):
            for xyz, axis in enumerate((2, 1, 0)):
                size = material.shape[axis]
                shape = [1, 1, 1]
                shape[axis] = size
                coordinate = torch.arange(size, device=material.device, dtype=material.dtype).reshape(shape)
                grid[..., xyz] = (coordinate + sign * normals[..., axis] + 0.5) * (2 / size) - 1
            sample = F.grid_sample(
                material[None, None], grid[None], mode="bilinear", padding_mode="border", align_corners=False
            )[0, 0]
            crossings.add_((material - sample).abs())
        del grid, sample
        crossings.mul_(0.25).mul_(coherent)
        density = _gaussian(crossings, 6.0)
        supported = _gaussian(coherent.to(torch.float32), 6.0) > 0.2
        density.masked_fill_(~supported, 0)
        normals.masked_fill_(~supported[..., None], 0)
        core = tuple(slice(halo, size - halo) for size in material.shape)
        return density[core].clone(), normals[core].flip(-1).contiguous(), supported[core].clone()
