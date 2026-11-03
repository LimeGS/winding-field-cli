# Field-to-TIFXYZ setup

Start with the [commands in README section 5](../README.md). This page explains
the two JSON files used by `export-sheets`.

## Configuration files

### `field.json`: the input field

Use this template for a NumPy field. Replace the path, hash and geometry with
your field's actual values; the numbers below are examples.

```json
{
  "field": {
    "kind": "npy",
    "path": "/data/field.npy",
    "sha256": "REPLACE_WITH_FILE_SHA256"
  },
  "geometry": {
    "frame": "my-scroll/my-scan",
    "voxel_um": 1.0,
    "origin_xyz": [0.0, 0.0, 0.0],
    "spacing_xyz": [17.28, 17.28, 17.28],
    "shape_zyx": [100, 200, 200],
    "sheet_step": 1.0
  }
}
```

- Get the hash with `shasum -a 256 /data/field.npy`.
- Use absolute paths. The array is ZYX; origin and spacing are XYZ.
- With `voxel_um: 1`, origin and spacing are in micrometres.
- If the field has a support mask, add `"mask": {"path": "/data/mask.npy",
  "sha256": "MASK_SHA256"}` inside `field`. It must be a boolean array of the same shape.

For a CLI block checkpoint, replace the `field` object with:

```json
{
  "kind": "chunks",
  "path": "/data/run/field/CHECKPOINT.json",
  "sha256": "REPLACE_WITH_CHECKPOINT_SHA256",
  "support_policy": "observed"
}
```

Copy `geometry` from that run's `FIELD.json`. Keep the checkpoint and its chunk
files together. The existing field determines coarse/fine resolution.

### `flattening.json`: the output coordinates

```json
{
  "output_frame": {
    "id": "my-scroll/my-scan",
    "origin_xyz_um": [0.0, 0.0, 0.0],
    "voxel_um": 8.64
  },
  "memory_gib": 4.0
}
```

Set the ID, origin and voxel size to the destination scan's coordinate frame.
This example writes XYZ in 8.64 µm voxels. It does not change field resolution.
`memory_gib` is a mapping admission estimate, not a hard RAM cap; large fine
fields need more memory.

The command's `--source` must be the matching CT. `--voxel-um` supplies the CT
scale when its metadata lacks units; it is separate from the output frame above.

## Reuse an existing mesh

Replace `--field-config field.json --levels -31` with
`--mesh-bundle /data/run/meshes`. The bundle must contain `MESH.json`, `RUN.json`
and `mesh.sqlite`; it already records its W levels.

## Resume and inspect results

Repeat the same export command with `--resume`. Changed inputs or settings need
a new output directory.

Each requested level produces an `original/level-NNNN/` directory. Open
`image.png` to inspect the flattened CT. Keep the complete `tifxyz/` directory,
including its validity mask and metadata, when copying the surface elsewhere.
`level-0000` means the first requested level, not W=0.

## Technical reference

- [Original mapping, masks and coordinate conventions](FORMATS.md#original-recipe-products)
- [Native output and time/coverage settings](FORMATS.md#native-global-surfaces)
- [Experimental direct-field TIFXYZ](FORMATS.md#direct-tifxyz-experimental)
- [Historical W+31 verification](verification/original-field-e2e-W31.json)
