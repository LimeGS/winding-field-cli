"""Installed-library CPU demo: tiled field -> connected OBJ meshes -> actual3D PNGs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from winding_field.surface_workflow import run_surfaces


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tiles", type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.resume:
        root.mkdir(exist_ok=False)
        z, y, x = np.indices((9, 9, 21))
        values = np.minimum(
            (x - 5) ** 2 + (y - 4) ** 2 + (z - 4) ** 2, (x - 15) ** 2 + (y - 4) ** 2 + (z - 4) ** 2
        ).astype(np.float64)
        np.save(root / "field.npy", values)
        config = dict(
            schema="winding-surfaces-v1",
            field=dict(kind="npy", path="field.npy", sha256=sha(root / "field.npy")),
            geometry=dict(
                frame="synthetic-two-objects/world",
                voxel_um=7.5,
                sheet_step=1.0,
                origin_xyz=[10.0, 20.0, 30.0],
                spacing_xyz=[2.0, 3.0, 4.0],
                shape_zyx=list(values.shape),
            ),
            levels=[3.5, 7.5],
            execution=dict(tile=4, level_batch=2),
            render=dict(width=256, height=256),
        )
        (root / "surfaces.json").write_text(json.dumps(config, indent=2) + "\n")

    def stop():
        path = root / "mesh/STATUS.json"
        return bool(
            args.stop_after_tiles is not None
            and path.exists()
            and json.loads(path.read_bytes())["next_tile"] >= args.stop_after_tiles
        )

    result = run_surfaces(root / "surfaces.json", root / "mesh", resume=args.resume, stop=stop)
    if result["status"] == "SURFACES_READY":
        assert [v["components"] for v in result["mesh"]["levels"]] == [2, 2]
        assert all(v["nonempty_pixels"] > 0 for v in result["renders"]["outputs"])
        mesh = root / "mesh/level-0000.obj"
        imported = dict(
            schema="winding-surfaces-v1",
            mesh=dict(path="mesh/level-0000.obj", sha256=sha(mesh)),
            geometry=dict(frame="synthetic-two-objects/world", voxel_um=7.5),
            render=dict(width=256, height=256),
        )
        (root / "import.json").write_text(json.dumps(imported, indent=2) + "\n")
        second = run_surfaces(
            root / "import.json", root / "imported", resume=args.resume and (root / "imported").exists()
        )
        assert second["mesh"]["levels"][0]["components"] == 2
        assert all(v["nonempty_pixels"] > 0 for v in second["renders"]["outputs"])
    (root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(root), physical_acceptance=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
