import json
from argparse import Namespace
import platform
import sys

import numpy as np
import pytest

from winding_field import counted_density as cd

PERIOD = 16
# The shifted-slice filter adds taps in conv1d's order. On x86-64 Linux (the production
# hosts) torch's CPU conv1d gives the same bits; other backends (macOS ARM) accumulate
# differently, about 1 ulp of float32.
SAME_BITS = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")


def sheets(shape):
    """Planar sheets normal to z: material on half of every PERIOD voxels."""
    z = np.arange(shape[0])[:, None, None]
    return np.broadcast_to(np.where((z % PERIOD) < PERIOD // 2, 255, 0), shape).astype(np.uint8)


def test_planar_sheets_count_one_per_period_with_normals_along_z():
    volume = sheets((96, 96, 96))
    density, normals, supported = cd.density_normals(volume, halo=cd.HALO, device="cpu")
    assert supported.all()
    assert np.median(density) == pytest.approx(1 / PERIOD, rel=0.02)
    assert np.abs(normals[..., 2]).min() > 0.99


def _conv1d_gaussian(value, sigma, order=(0, 0, 0)):
    """The pre-2.0 CPU filter (conv1d over movedim lines): the bits the recipe pins."""
    import torch
    from torch.nn import functional as F

    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    for axis, derivative in enumerate(order):
        weight = kernel * x / (sigma * sigma) if derivative else kernel
        lines = value.movedim(axis, -1)
        shape = lines.shape
        size = shape[-1]
        index = torch.arange(-radius, size + radius) % (2 * size)
        index = torch.where(index < size, index, 2 * size - 1 - index)
        padded = lines.index_select(-1, index).reshape(-1, 1, size + 2 * radius)
        filtered = F.conv1d(padded, weight.to(value.dtype).reshape(1, 1, -1))
        value = filtered.reshape(shape).movedim(-1, axis)
    return value


@pytest.mark.parametrize(
    "sigma, order", [(1.0, (1, 0, 0)), (1.0, (0, 0, 1)), (2.0, (0, 0, 0)), (6.0, (0, 0, 0))]
)
def test_gaussian_gives_the_conv1d_bits(sigma, order):
    import torch

    value = torch.rand((23, 31, 57), generator=torch.Generator().manual_seed(0))
    new, old = cd._gaussian(value, sigma, order), _conv1d_gaussian(value, sigma, order)
    if SAME_BITS:
        assert torch.equal(new, old)
    else:
        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)


def test_density_normals_unchanged_by_the_filter_rewrite(monkeypatch):
    rng = np.random.default_rng(1)
    z, y, x = np.meshgrid(*[np.arange(112)] * 3, indexing="ij")
    radius = np.hypot(y - 50.3, x - 61.7) + 0.2 * z + rng.normal(0, 0.5, z.shape)
    volume = ((np.cos(2 * np.pi * radius / 14) * 0.5 + 0.5) * 255).astype(np.uint8)
    new = cd.density_normals(volume, halo=cd.HALO, device="cpu")
    monkeypatch.setattr(cd, "_gaussian", _conv1d_gaussian)
    old = cd.density_normals(volume, halo=cd.HALO, device="cpu")
    if SAME_BITS:
        for a, b in zip(new, old):
            np.testing.assert_array_equal(a, b)
        return
    # Elsewhere float32 rounding may move a voxel across the coherence or support
    # threshold; allow a few such voxels and compare the rest closely.
    (density, normals, supported), (density0, normals0, supported0) = new, old
    assert (supported != supported0).mean() < 1e-3
    both = supported & supported0
    np.testing.assert_allclose(density[both], density0[both], rtol=1e-5, atol=1e-7)
    assert (np.abs(normals[both] - normals0[both]) > 1e-4).any(-1).mean() < 1e-3


def test_encoding_is_what_the_lasagna_importer_decodes():
    density = np.array([0.0625, 0.11, 0.0])
    normals = np.array([[0.0, 0.6, -0.8], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    supported = np.array([True, True, False])
    grad_mag, nx, ny = cd.encode(
        density, normals, supported, scaledown=1, encode_scale=1000.0, grad_mag_factor=1.0
    )
    # importer: grad_mag * factor / encode_scale cycles per base voxel; cos voxels are 2.
    np.testing.assert_allclose(grad_mag[:2] / 1000.0 * 2, density[:2], atol=1e-3)
    assert grad_mag[2] == 0 and nx[2] == 0 and ny[2] == 0
    # hemisphere encoding: z >= 0, so (0, .6, -.8) is stored as (0, -.6, .8)
    assert (nx[0], ny[0]) == (128, round(-0.6 * 127 + 128))
    assert (nx[1], ny[1]) == (255, 128)


def write_manifest(tmp_path, shape):
    cos = cd._Array(
        tmp_path / "lasagna/cos.zarr/1",
        dict(
            shape=list(shape),
            chunks=[64, 64, 64],
            dtype="|u1",
            fill_value=0,
            order="C",
            filters=None,
            dimension_separator="/",
            compressor=dict(id="blosc", cname="zstd", clevel=3, shuffle=1, blocksize=0),
            zarr_format=2,
        ),
    )
    cos.write([0, 0, 0], sheets(shape))
    manifest = dict(
        version=2,
        base_shape_zyx=[2 * n for n in shape],
        grad_mag_encode_scale=1000.0,
        grad_mag_factor=1.0,
        ct_source=dict(spacing_xyz=[8.64] * 3),
        groups={
            "cos": dict(channels=["cos"], scaledown=1, zarr="cos.zarr/1"),
            "grad_mag": dict(channels=["grad_mag"], scaledown=2, zarr="grad_mag.zarr/2"),
            "nx": dict(channels=["nx"], scaledown=2, zarr="nx.zarr/2"),
            "ny": dict(channels=["ny"], scaledown=2, zarr="ny.zarr/2"),
        },
    )
    path = tmp_path / "lasagna/input.lasagna.json"
    path.write_text(json.dumps(manifest))
    return path


def args(manifest, output, **extra):
    values = dict(
        lasagna_input=manifest,
        output=output,
        bounds=None,
        tile=64,
        device="cpu",
        part=(0, 1),
        min_fill=0.01,
    )
    values.update(extra)
    return Namespace(**values)


def test_build_writes_counted_channels_and_a_manifest_and_resumes(tmp_path):
    manifest = write_manifest(tmp_path, (64, 64, 128))
    result = cd.build(args(manifest, tmp_path / "counted"))
    assert result["status"] == "COUNTED_READY" and result["computed"] == 2
    # One sheet every 16 cos voxels of 17.28 um (fewer counted at the array faces).
    assert result["mean_sheet_spacing_um"] == pytest.approx(16 * 17.28, rel=0.2)
    written = json.loads((tmp_path / "counted/counted.lasagna.json").read_text())
    for name in ("grad_mag", "nx", "ny"):
        group = written["groups"][name]
        assert group["scaledown"] == 1
        assert group["zarr"] == str(tmp_path / f"counted/{name}.zarr/1")
    assert written["groups"]["cos"]["zarr"] == str(tmp_path / "lasagna/cos.zarr/1")
    grad_mag = cd._Array(tmp_path / "counted/grad_mag.zarr/1").read([24, 24, 24], [40, 40, 104])
    assert np.median(grad_mag) / 500.0 == pytest.approx(1 / PERIOD, rel=0.03)

    again = cd.build(args(manifest, tmp_path / "counted"))
    assert again["computed"] == 0 and again["reused"] == 2
    with pytest.raises(ValueError, match="different inputs"):
        cd.build(args(manifest, tmp_path / "counted", tile=128))


def test_parts_split_tiles_and_the_manifest_waits_for_all_of_them(tmp_path):
    manifest = write_manifest(tmp_path, (64, 64, 128))
    first = cd.build(args(manifest, tmp_path / "counted", part=(0, 2)))
    assert first["status"] == "INCOMPLETE" and first["remaining_tiles"] == 1
    assert not (tmp_path / "counted/counted.lasagna.json").exists()
    second = cd.build(args(manifest, tmp_path / "counted", part=(1, 2)))
    assert second["status"] == "COUNTED_READY" and second["computed"] == 1


def test_cuda_request_without_cuda_falls_back_to_cpu_with_a_warning(monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cd.resolve_device("cuda") == "cpu"
    assert cd.resolve_device("cuda:1") == "cpu"
    assert "no CUDA device" in capsys.readouterr().err
    assert cd.resolve_device("cpu") == "cpu"
    assert capsys.readouterr().err == ""


def test_build_reports_the_device_it_used(tmp_path, monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manifest = write_manifest(tmp_path, (64, 64, 128))
    result = cd.build(args(manifest, tmp_path / "counted", device="cuda"))
    assert result["device"] == "cpu" and result["status"] == "COUNTED_READY"
    assert "WARNING" in capsys.readouterr().err
