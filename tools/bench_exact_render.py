"""Time a real saved UV map with deterministic CT; compare receipts across checkouts."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from winding_field.flattening.patches import render_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path)
    parser.add_argument("--pitch-um", type=float, default=19.2)
    parser.add_argument("--voxel-um", type=float, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tifxyz", action="store_true")
    args = parser.parse_args()
    with np.load(args.map, allow_pickle=False) as data:
        mapped = {k: data[k] for k in ("world", "faces", "uv", "display")}
        if "source_face_rows" not in data:
            parser.error("map must include source_face_rows; database IDs are not row offsets")
        mapped["source_face_rows"] = data["source_face_rows"]

    def sample(xyz):
        return xyz.sum(axis=1) % 256, (xyz[:, 0].astype(int) % 17) != 0

    with tempfile.TemporaryDirectory() as scratch:
        start = perf_counter()
        record = render_map(
            mapped,
            Path(scratch) / "tiles",
            sample,
            {"voxel_um": args.voxel_um},
            pitch_um=args.pitch_um,
            tile_size=args.tile_size,
            raster_workers=args.workers,
            export_tifxyz=args.tifxyz,
        )
        seconds = perf_counter() - start
    # The render record binds every tile's PNG, mapping, TIFF and metadata file hash.
    signature = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    print(
        json.dumps(
            {
                "seconds": seconds,
                "faces": len(mapped["faces"]),
                "shape": record["shape_yx"],
                "tiles": len(record["tiles"]),
                "record_sha256": signature,
            }
        )
    )


if __name__ == "__main__":
    main()
