"""Sheet density counted on the Lasagna sheet channel, written as Lasagna channels.

Lasagna's ``grad_mag`` is a learned estimate of the sheet frequency. On PHerc0175A it
reads 1.4-1.7 times the sheets that are really there (measured against ``cos`` and CT),
so a field solved with it counts too many windings and its level surfaces cross sheets.
The October 2026 fields counted the sheets instead, and this module brings that recipe
back unchanged: threshold ``cos`` at mid-grey, take the structure-tensor normal of the
sheet mask, count mask crossings one voxel along the normal on both sides (each sheet
has two faces, each crossing is seen from both sides: x 1/4), and smooth over 6 voxels.
On planar sheets of period 16 voxels it returns 1/16 within 2 %.

The result is written as three uint8 Zarr v2 arrays on the ``cos`` grid (``grad_mag``,
``nx``, ``ny``, in the encoding the Lasagna importer reads) and a copy of the input
manifest whose ``grad_mag``/``nx``/``ny`` groups point at them; ``cos`` is kept. Every
existing command takes the new manifest as ``--lasagna-input``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .util import atomic_json, file_sha

HALO = 40
CHUNK = 64
RECIPE = "october-2026-counted-crossings-v1"


# --- the October filters (torch port of gradmag_fractional.py), unchanged -------------


def _gaussian(value, sigma, order=(0, 0, 0)):
    """Separable SciPy-default (truncate=4, half-sample reflect) filtering.

    Each axis is a weighted sum of shifted slices of the reflect-padded volume, in tap
    order. On CPU this is 2.4-2.6x faster than ``conv1d`` over single-channel lines
    (which also needed a gather on a non-contiguous ``movedim`` view) and gives the
    same bits; on CUDA it avoids cuDNN's costly large-batch single-channel path.
    """
    import torch

    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    for axis, derivative in enumerate(order):
        # Correlation, so the first-derivative kernel has +x sign.
        weights = (kernel * x / (sigma * sigma) if derivative else kernel).tolist()
        size = value.shape[axis]
        index = torch.arange(-radius, size + radius, device=value.device) % (2 * size)
        index = torch.where(index < size, index, 2 * size - 1 - index)
        padded = value.index_select(axis, index)
        filtered = torch.zeros_like(value)
        for tap, weight in enumerate(weights):
            filtered.add_(padded.narrow(axis, tap, size), alpha=weight)
        value = filtered
    return value


def _eigh(tensor):
    """Ascending eigenvalues and principal director of packed PSD 3x3 tensors.

    Packing is (00, 01, 02, 11, 12, 22). A zero/isotropic tensor returns a zero
    director, not an invented axis.
    """
    import torch

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


def density_normals(volume, *, halo=HALO, device="cpu"):
    """(density, normals_xyz, supported) of the halo-trimmed ZYX core of a uint8 volume.

    Density is sheets per input voxel; normals are unsigned unit sheet normals (XYZ),
    zero where the sheet mask is incoherent; support is Gaussian(coherent, 6) > 0.2.
    A halo of 40 voxels covers the filters' complete support.
    """
    import torch
    from torch.nn import functional as F

    if not isinstance(volume, (np.ndarray, torch.Tensor)) or volume.dtype not in (
        np.dtype("uint8"),
        torch.uint8,
    ):
        raise TypeError("volume must be a uint8 ZYX array")
    if volume.ndim != 3 or any(size <= 2 * halo for size in volume.shape):
        raise ValueError("volume must be ZYX with a nonempty core beyond the halo")
    with torch.no_grad():
        material = (torch.as_tensor(volume, device=device) > 127).to(torch.float32)
        with (
            torch.autocast(device_type=material.device.type, enabled=False),
            torch.backends.cudnn.flags(allow_tf32=False),
        ):
            gradients = [_gaussian(material, 1.0, o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
            tensor = material.new_empty((*material.shape, 6))
            pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
            for k, (i, j) in enumerate(pairs):
                tensor[..., k] = _gaussian(gradients[i] * gradients[j], 2.0)
            del gradients
            eigenvalues, normals = _eigh(tensor)
            del tensor
            anisotropy = (eigenvalues[..., 2] - eigenvalues[..., 1]) / eigenvalues[
                ..., 2
            ].clamp_min(1e-12)
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
                    coordinate = torch.arange(
                        size, device=material.device, dtype=material.dtype
                    ).reshape(shape)
                    grid[..., xyz] = (coordinate + sign * normals[..., axis] + 0.5) * (2 / size) - 1
                sample = F.grid_sample(
                    material[None, None],
                    grid[None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )[0, 0]
                crossings.add_((material - sample).abs())
            del grid, sample
            crossings.mul_(0.25).mul_(coherent)
            density = _gaussian(crossings, 6.0)
            supported = _gaussian(coherent.to(torch.float32), 6.0) > 0.2
            density.masked_fill_(~supported, 0)
            normals.masked_fill_(~supported[..., None], 0)
            core = tuple(slice(halo, size - halo) for size in material.shape)
            return (
                density[core].cpu().numpy(),
                normals[core].flip(-1).cpu().numpy(),
                supported[core].cpu().numpy(),
            )


# --- Zarr v2 chunks ---------------------------------------------------------------------


class _Array:
    """Minimal Zarr v2 reader/writer for uint8 C-order chunked arrays (missing = fill)."""

    def __init__(self, path, meta=None):
        import numcodecs

        self.path = Path(path)
        if meta is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            atomic_json(self.path / ".zarray", meta)
        self.meta = json.loads((self.path / ".zarray").read_text())
        if (
            self.meta.get("zarr_format") != 2
            or np.dtype(self.meta["dtype"]) != np.uint8
            or self.meta.get("order", "C") != "C"
            or self.meta.get("filters")
        ):
            raise ValueError(
                f"{self.path}: a uint8 C-order Zarr v2 array without filters is required"
            )
        self.shape = [int(n) for n in self.meta["shape"]]
        self.chunks = [int(n) for n in self.meta["chunks"]]
        self.separator = self.meta.get("dimension_separator", ".")
        self.fill = int(self.meta.get("fill_value") or 0)
        compressor = self.meta.get("compressor")
        self.codec = numcodecs.get_codec(compressor) if compressor else None

    def _file(self, key):
        return self.path / self.separator.join(map(str, key))

    def read(self, lo, hi):
        """Values in [lo, hi) (ZYX); outside the array and missing chunks read as fill."""
        out = np.full([b - a for a, b in zip(lo, hi)], self.fill, np.uint8)
        first = [max(0, a) for a in lo]
        last = [min(n, b) for n, b in zip(self.shape, hi)]
        if any(b <= a for a, b in zip(first, last)):
            return out
        ranges = [range(a // c, (b - 1) // c + 1) for a, b, c in zip(first, last, self.chunks)]
        for key in np.ndindex(*[len(r) for r in ranges]):
            key = [r[k] for r, k in zip(ranges, key)]
            path = self._file(key)
            if not path.is_file():
                continue
            raw = path.read_bytes()
            chunk = np.frombuffer(self.codec.decode(raw) if self.codec else raw, np.uint8)
            chunk = chunk.reshape(self.chunks)
            origin = [k * c for k, c in zip(key, self.chunks)]
            a = [max(f, o) for f, o in zip(first, origin)]
            b = [min(hi, o + c) for hi, o, c in zip(last, origin, self.chunks)]
            out[tuple(slice(p - s, q - s) for p, q, s in zip(a, b, lo))] = chunk[
                tuple(slice(p - o, q - o) for p, q, o in zip(a, b, origin))
            ]
        return out

    def write(self, lo, values):
        """Write whole chunks starting at chunk-aligned ``lo``; all-fill chunks are skipped."""
        for offset in np.ndindex(*[-(-n // c) for n, c in zip(values.shape, self.chunks)]):
            start = [o * c for o, c in zip(offset, self.chunks)]
            part = values[tuple(slice(s, s + c) for s, c in zip(start, self.chunks))]
            if not (part != self.fill).any():
                continue
            if part.shape != tuple(self.chunks):
                full = np.full(self.chunks, self.fill, np.uint8)
                full[tuple(slice(0, n) for n in part.shape)] = part
                part = full
            key = [(lo_ + s) // c for lo_, s, c in zip(lo, start, self.chunks)]
            path = self._file(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = np.ascontiguousarray(part).tobytes()
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(self.codec.encode(raw) if self.codec else raw)
            os.replace(temporary, path)


# --- the command ------------------------------------------------------------------------


def _group(manifest, name):
    matches = [g for g in manifest.get("groups", {}).values() if g.get("channels") == [name]]
    if len(matches) != 1:
        raise ValueError(f"the Lasagna manifest needs exactly one group for {name}")
    return matches[0]


def _local(value, base):
    path = Path(value)
    return path if path.is_absolute() else base / path


def encode(density, normals, supported, *, scaledown, encode_scale, grad_mag_factor):
    """uint8 grad_mag/nx/ny the Lasagna importer decodes back to this density and normal.

    The importer reads ``grad_mag * factor / encode_scale`` cycles per base voxel; the
    counted density is sheets per voxel of the ``cos`` grid, 2**scaledown base voxels.
    Normals are stored hemisphere-encoded (z >= 0) as ``127 n + 128``.
    """
    scale = encode_scale / (grad_mag_factor * (1 << scaledown))
    grad_mag = np.where(supported, np.clip(np.rint(density * scale), 1, 255), 0).astype(np.uint8)
    flip = np.where(normals[..., 2:3] < 0, -1.0, 1.0)
    unit = normals * flip
    channels = [
        np.where(supported, np.clip(np.rint(unit[..., k] * 127 + 128), 0, 255), 0).astype(np.uint8)
        for k in (0, 1)
    ]
    return grad_mag, channels[0], channels[1]


def build(args):
    manifest_path = Path(args.lasagna_input).resolve()
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    cos_group = _group(manifest, "cos")
    scaledown = int(cos_group["scaledown"])
    cos = _Array(_local(cos_group["zarr"], base))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(
        shape=cos.shape,
        chunks=[CHUNK] * 3,
        dtype="|u1",
        fill_value=0,
        order="C",
        filters=None,
        dimension_separator="/",
        compressor=dict(id="blosc", cname="zstd", clevel=3, shuffle=1, blocksize=0),
        zarr_format=2,
    )
    arrays = {
        name: _Array(output / f"{name}.zarr" / str(scaledown), meta)
        for name in ("grad_mag", "nx", "ny")
    }
    tile = int(args.tile)
    if tile <= 0 or tile % CHUNK:
        raise ValueError(f"--tile must be a positive multiple of {CHUNK}")
    if args.bounds:
        factor = 1 << scaledown
        lo = [args.bounds[2 * i] // factor for i in range(3)]
        hi = [-(-args.bounds[2 * i + 1] // factor) for i in range(3)]
    else:
        lo, hi = [0, 0, 0], list(cos.shape)
    lo = [max(0, a) // CHUNK * CHUNK for a in lo]
    hi = [min(n, b) for n, b in zip(cos.shape, hi)]
    corners = [corner for corner in np.ndindex(*[-(-(b - a) // tile) for a, b in zip(lo, hi)])]
    part, parts = args.part
    if not 0 <= part < parts:
        raise ValueError("--part K N needs 0 <= K < N")
    done_dir = output / "done"
    done_dir.mkdir(exist_ok=True)
    identity = dict(
        recipe=RECIPE,
        cos=str(_local(cos_group["zarr"], base)),
        cos_scaledown=scaledown,
        manifest_sha256=file_sha(manifest_path),
        tile=tile,
        halo=HALO,
        min_fill=args.min_fill,
        encode_scale=manifest.get("grad_mag_encode_scale", 1000.0),
        grad_mag_factor=manifest.get("grad_mag_factor", 1.0),
    )
    identity_path = output / "COUNTED.json"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text()).get("identity")
        if previous != identity:
            raise ValueError(
                "output holds counted density from different inputs; use a new --output"
            )
    else:
        atomic_json(identity_path, dict(identity=identity, status="RUNNING"))
    started = time.time()
    computed = skipped = reused = 0
    total, weight = 0.0, 0
    for number, corner in enumerate(corners):
        if number % parts != part:
            continue
        a = [lo_ + c * tile for lo_, c in zip(lo, corner)]
        b = [min(h, s + tile) for h, s in zip(hi, a)]
        marker = done_dir / "-".join(map(str, a))
        if marker.is_file():
            reused += 1
            continue
        volume = cos.read([s - HALO for s in a], [e + HALO for e in b])
        core = volume[(slice(HALO, -HALO),) * 3]
        if (core > 0).mean() < args.min_fill:
            skipped += 1
            marker.write_text("empty\n")
            continue
        density, normals, supported = density_normals(volume, halo=HALO, device=args.device)
        channels = encode(
            density,
            normals,
            supported,
            scaledown=scaledown,
            encode_scale=identity["encode_scale"],
            grad_mag_factor=identity["grad_mag_factor"],
        )
        for name, values in zip(("grad_mag", "nx", "ny"), channels):
            arrays[name].write(a, values)
        computed += 1
        if supported.any():
            total += float(density[supported].sum())
            weight += int(supported.sum())
        marker.write_text(json.dumps(dict(supported=int(supported.sum()))) + "\n")
        if computed % 25 == 0:
            print(
                f"[counted-density] {number + 1}/{len(corners)} tiles, {computed} computed, "
                f"{skipped} empty, {time.time() - started:.0f}s",
                flush=True,
            )
    remaining = sum(
        1
        for corner in corners
        if not (done_dir / "-".join(str(lo_ + c * tile) for lo_, c in zip(lo, corner))).is_file()
    )
    counted = dict(manifest)
    counted["groups"] = {
        key: group
        for key, group in manifest["groups"].items()
        if group.get("channels") not in (["grad_mag"], ["nx"], ["ny"])
    }
    for name, array in arrays.items():
        counted["groups"][name] = dict(channels=[name], scaledown=scaledown, zarr=str(array.path))
    if counted.get("provenance"):
        counted["provenance"] = str(_local(counted["provenance"], base))
    cos_entry = _group(counted, "cos")
    cos_entry["zarr"] = str(_local(cos_entry["zarr"], base))
    manifest_out = output / "counted.lasagna.json"
    status = "COUNTED_READY" if remaining == 0 else "INCOMPLETE"
    if remaining == 0:
        atomic_json(manifest_out, counted)
    atomic_json(
        identity_path,
        dict(
            identity=identity,
            status=status,
            tiles=len(corners),
            remaining_tiles=remaining,
            manifest=str(manifest_out) if remaining == 0 else None,
        ),
    )
    mean = total / weight if weight else None
    return dict(
        status=status,
        output=str(output),
        manifest=str(manifest_out) if remaining == 0 else None,
        tiles=len(corners),
        computed=computed,
        empty=skipped,
        reused=reused,
        remaining_tiles=remaining,
        mean_sheet_spacing_um=None
        if not mean
        else float(
            (1 << scaledown)
            * float(manifest.get("ct_source", {}).get("spacing_xyz", [1.0])[0])
            / mean
        ),
        seconds=time.time() - started,
    )
