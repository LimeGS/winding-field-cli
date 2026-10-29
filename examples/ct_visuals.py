#!/usr/bin/env python3
"""Render the former benchmark's synthetic CT-like layers using the installed CLI."""

import argparse
from pathlib import Path

import numpy as np

from winding_field.ct_visuals import main as render
from winding_field.util import atomic_json, file_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.output / "inputs"
    folder.mkdir()
    shape, step = (32, 32, 128), 2.0
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    field = np.broadcast_to(z / 4, shape).copy()
    field_path = folder / "field.npy"
    np.save(field_path, field)
    geometry = dict(frame="synthetic-example/world", voxel_um=9.6, step=step,
                    origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1])
    atomic_json(folder / "field.npy.json", dict(geometry, shape_zyx=list(shape),
                field_sha256=file_sha(field_path), synthetic=True))
    # Smooth bright bands illustrate analytic layers; these are not scan data.
    intensities = (20 + 210 * (0.5 + 0.5 * np.cos(2 * np.pi * z / 8)) ** 8).astype(np.uint8)
    ct = np.broadcast_to(intensities, shape)
    visual = folder / "ct"
    visual.mkdir()
    planes = []
    for name, axis in (("XY", 2), ("XZ", 1), ("YZ", 0)):
        index = shape[2 - axis] // 2
        pixels = np.take(ct, index, axis=2 - axis)
        np.savez_compressed(visual / (name + "-samples.npz"), ct=pixels)
        planes.append(dict(name=name, fixed_axis_xyz=axis, fixed_position_world=index, ct_pixels=pixels.size))
    atomic_json(visual / "RECEIPT.json", dict(geometry, synthetic=True, size_xyz=list(shape[::-1]),
        ct_window=[0, 255], contour_levels=[0, 1, 2, 3, 4], field_sha256={}, planes=planes,
        ct_url="synthetic://analytic-parallel-layers", input_ct={}))
    atomic_json(visual / "MANIFEST.json", {p.name: file_sha(p) for p in sorted(visual.iterdir())})
    atomic_json(folder / "case.json", dict(geometry, visualization=dict(schema="ct-winding-overlays-v2",
        archive_manifest=dict(path="ct/MANIFEST.json", sha256=file_sha(visual / "MANIFEST.json")))))
    status = render(["--case", str(folder / "case.json"), "--field", str(field_path),
                   "--metadata", str(folder / "field.npy.json"), "--output", str(args.output / "views")])
    if status == 0:
        with np.load(args.output / "views/ct-winding-samples.npz") as samples:
            np.testing.assert_allclose(samples["XY_candidate"], 2)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
