"""Evaluate human TIFXYZ patches and render field residuals in sheet units."""

import argparse
import json
from pathlib import Path

import numpy as np

from patch_check import evaluate_patch, load_patch, sha256


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--patch", type=Path, nargs="+", required=True)
    p.add_argument("--patch-frame", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=float, help="fixed baseline step for a paired comparison")
    p.add_argument("--margin", type=int, default=8)
    args = p.parse_args(argv)
    meta = json.loads(args.metadata.read_text())
    if meta["frame"] != args.patch_frame:
        raise ValueError("patch/field coordinate frames differ")
    field = np.load(args.field, mmap_mode="r").astype(np.float32)
    if list(field.shape) != meta["shape_zyx"] or not np.isfinite(field).all():
        raise ValueError("field shape or finite-value check failed")
    step = args.step if args.step is not None else meta["step"]
    output = dict(
        field=str(args.field), field_sha256=sha256(args.field), metadata=meta,
        step_used=step, patches=[],
        limitation="Same-sheet constancy and local jumps, not proof against collapse or merges.",
    )
    for patch in args.patch:
        pm, xyz = load_patch(patch)
        result, residual = evaluate_patch(
            field, xyz, meta["origin_xyz"], meta["spacing_xyz"], step, args.margin, min_component=30
        )
        result.update(
            uuid=pm["uuid"], source=str(patch),
            sha256={f: sha256(patch / f) for f in ("meta.json", "x.tif", "y.tif", "z.tif")},
        )
        output["patches"].append(result)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        try:
            im = ax.imshow(residual, cmap="coolwarm", vmin=-3, vmax=3, interpolation="nearest")
            ax.set_title(f"{pm['uuid']} — deviation from each connected piece (sheets)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(args.output.with_name(args.output.stem + "_" + pm["uuid"] + ".png"), dpi=140)
        finally:
            plt.close(fig)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
