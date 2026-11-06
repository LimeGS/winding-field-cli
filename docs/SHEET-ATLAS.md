# Constant-W visual sheet atlas

Package independently flattened constant-W sheets side by side, at a common
physical pixel pitch, with a local-file pan/zoom viewer and stored-pixel XYZ lookup:

```bash
winding-field sheet-atlas build --sheets W-59/SHEET.json W-60/SHEET.json W-61/SHEET.json \
  --output atlas
winding-field sheet-atlas query atlas 1200 4500
```

Open `atlas/index.html` directly. Drag to pan, scroll or use the buttons to zoom;
keyboard arrows, `+`, `-`, and `Home` also work. It loads previews at low zoom and
visible full-resolution tiles at high zoom. There are no network dependencies.
`atlas.png` is the full-resolution grayscale image; `mask.png` retains the four
original mask values: **0** unmapped, **1** missing CT, **2** observed, **3** ambiguous.
Queries return XYZ and source face ID only for mask 2; other pixels return null.

## Input contract

Each `SHEET.json` uses schema `winding-atlas-sheet-v1` and contains:

- `kind: "constant_w"`, finite normalized `w`, and `level_index` matching the
  hash-bound `MESH.json` (`w = level / metadata.sheet_step`).
- `frame` with `id`, `voxel_um`, and `origin_xyz_um`; common `ct` identity.
- Common `reference_z`, per-sheet `anchor_uv`, and `z_increases_up: true`.
- `files`: relative paths to SHA256 values, including `map.npz`, `RENDER.json`,
  `MESH.json`, plus any quality, disconnected-component and provenance receipts.
  A bound `CT-IDENTITY.json`, when present, must agree with `ct` and `frame`.
- `map.npz` contains exact `world`, oriented `faces`, per-vertex `uv`, preserved
  `source_vertex_ids`, and per-face `source_face_rows` and `source_face_ids`.
- `RENDER.json` and `render/tile-*/TILE.json` come from `patch_pipeline.render_map`:
  window 0–255, rows decreasing in UV V, PNG/mask/XYZ payload hashes and global
  lattice coordinates. Every expected tile and matching halo must be present.

The producer owns flattening quality, constant-W geometry, Z orientation and
the physical reference anchor. The atlas verifies receipts, normalized labels,
frame compatibility, map dimensions, source-row ownership and complete tile
coverage; it does not recompute the winding field or certify physical continuity.

## Coordinates and preservation

Sheets are ordered by ascending W. Layout uses integer pixel translations only,
with 32-pixel black separators. Anchors align to within half a pixel, recorded in
`ATLAS.json`; equal atlas rows elsewhere need not correspond to equal physical Z.
There is no welding, rotation, resampling, or deformation during packaging.

For a sheet with atlas offset `(ox, oy)` and UV origin `(u0, v_top)`:

```text
column = atlas_x - ox       row = atlas_y - oy
u = u0 + column * pitch    v = v_top - row * pitch
```

`query` returns the original stored XYZ, not a newly fitted inverse. Source IDs
resolve through the preserved map. Coordinates are in the declared global frame;
layout pixels are not a volumetric UVW field. Disconnected geometry and receipts
can accompany each sheet without inventing a visual placement for those pieces.

Inputs are copied and rehashed, never modified. Output must be fresh and disjoint
from inputs. The default disk reserve is 8 GiB; admission includes copies and
canvas scratch. Maps are bounded to 512 MiB uncompressed, mapping tiles to 128 MiB,
and the atlas to 512 million pixels. Run large packages in an appropriate memory
limit (the Paris4 trial uses 6 GiB). `MANIFEST.json` binds payload hashes and layout;
`ATLAS.json` is published last. A failed output without it is incomplete.
