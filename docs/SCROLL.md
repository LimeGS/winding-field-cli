# Whole-scroll fields from tiles

`winding-field scroll` builds one fine field for a whole scroll by splitting it into
overlapping tiles that run independently (on one machine or many), then blends them
into a single field that the existing mesh and render commands read. Every tile is an
ordinary `winding-field run`; nothing here changes how a tile is solved.

```text
coarse run ──► scroll plan ──► scroll tiles ──► scroll merge ──► prepared mesh ──► scroll render-ct
 (prior)        TILES.json      tiles/<id>/       MERGED/          per W level       CT at full pitch
                fibres/<id>.afv                   field/, LEVELS.json
```

## 1. Coarse field (the prior)

Tiles need a completed coarse field of the same scroll **at the same CT level**: the
field prior is refused when the frames differ (for example a level-0 coarse field for
level-2 tiles). A 138.24 µm coarse field at level 2 is stride 4:

```sh
wf run --source "$CT" --voxel-um 8.64 --level 2 --stride 4 --field-spacing-um 138.24 \
  --fibres /data/scroll.afv --lasagna-input /data/lasagna/input.lasagna.json \
  --sweeps 32 --until field --output /outputs/coarse
```

## 2. Plan

```sh
wf scroll plan --source "$CT" --voxel-um 8.64 --level 2 --field-spacing-um 34.56 \
  --coarse /outputs/coarse --fibres /data/scroll.afv \
  --lasagna-input /data/lasagna/input.lasagna.json \
  --tile-nodes 512 --overlap-nodes 64 --output /outputs/plan
```

- **One global grid.** The field spacing must be a whole multiple of the selected CT
  level's spacing (34.56 µm is 4 × 8.64 at level 0 and 1 × 34.56 at level 2). Tile ROIs
  start on that grid, so every tile's nodes coincide with the global nodes.
- **Tiles.** Cores of `--tile-nodes` partition each axis; each tile adds
  `--overlap-nodes` on interior sides. Both are multiples of 64 (the field chunk), so a
  tile chunk is exactly one merged chunk.
- **Skipped tiles.** Tiles whose box has no observed support in the coarse field are
  marked `supported: false` and never run.
- **Per-tile fibres.** By default the plan writes `fibres/<tile>.afv` with every whole
  fibre whose block points touch the tile box plus two field spacings. `run` still clips
  fibres to its ROI exactly; the split only avoids scanning the whole scroll AFV in every
  tile (about 49 minutes per tile for PHerc0175A's 966k blocks, versus seconds). The copy
  keeps the original metadata and adds a `scroll_split` provenance key.
  `--no-split-fibres` passes the full AFV instead.
- **Tile runs** use `--field-prior <coarse> --field-scope supported --until field`,
  the plan's level/stride/spacing, `--sweeps` (default 8) and any `--run-arg` flags.

### Level 2 or stride 4

At a 34.56 µm field, `--level 2 --stride 1` and `--level 0 --stride 4` read the same
Lasagna samples: on the PHerc0175A pilot tile their density, normals, support and
surface phase were bit-identical. Level 2 reads 64× less CT (17 MB instead of 315 MB for
that tile) because preparation only uses CT for validity and previews. Lasagna's normals
and density are stored every 34.56 µm, so neither route drops them; the surface phase
(`cos`, stored every 17.28 µm) is read at every other sample and only places the mesh
levels. Rendering is independent of both (see `render-ct`).

## 3. Run the tiles

```sh
wf scroll tiles /outputs/plan --jobs 16
```

Runs each supported tile (`run`, or `field` to continue an existing one) with at most
`--jobs` at a time, logs to `logs/<tile>.log` and records `status/<tile>.json`. Ctrl-C or
SIGTERM writes `STOP` into every running tile and waits. Rerun the same command to
continue. `--ids` limits the run to some tiles, so different machines can take different
tiles of the same plan.

## 4. Merge

```sh
wf scroll merge /outputs/plan --output /outputs/merged --workers 16
```

Writes `merged/field/CHECKPOINT.json` in the ordinary chunk-checkpoint format
(`open_field`, `prepared mesh`/`surfaces` read it), plus:

- `FIELD.json`: geometry, checkpoint hash, `global_converged: false`.
- `SEAMS.json`: for every pair of overlapping tiles, the median and 95th percentile of
  |ΔW| on nodes both solved, and how many differ by more than half a winding. A merge is
  a blend, not a global solve: read this before using the field.
- `LEVELS.json`: one phase offset for the whole scroll from the Lasagna surface phase
  (each node counted once, by the tile whose core owns it) and the resulting W levels.
- `surface-run.json`: a ready `prepared mesh`/`surfaces` request for those levels.

**Coarse-grid correction (default on).** A tile is gauged by a single prior node and
relaxed for a few sweeps, so its low frequencies drift from the coarse field, differently
in every tile. On the PHerc0175A pilot two neighbouring tiles disagreed in their overlap
by a median 0.21 and a 95th percentile of 1.0 winding (21% of nodes by more than half a
winding). Before blending, the merge subtracts from every tile a Gaussian low-pass of
`(tile - coarse)` (`--coarse-sigma-mm`, default 0.5 mm, computed on a 4x reduced grid):
detail below that scale comes from the tile, everything coarser from the one coarse
solve that spans all tiles. On the pilot this gave a median 0.018 and a 95th percentile
of 0.135 (0.26% of nodes over half a winding). Smaller sigma agrees better but keeps less
fine detail (sigma 0.25 mm: 0.043; 1 mm: 0.31). The global winding structure therefore
comes from the coarse field. `--coarse-sigma-mm 0` blends the raw tiles.

Blending is a linear cross-fade over the whole overlap on interior tile faces. Nodes
inside a solved chunk keep their value; observed support is the union of the tiles'
observed support. Missing tiles are refused unless `--allow-missing`. The merge holds the
checkpoint's `.writer.lock` exclusively while it writes, as the core writer does.

## 5. Meshes and full-resolution CT

Mesh the merged field per level with the existing command, then flatten and render each
level against the original CT at its own pitch:

```sh
wf prepared mesh --config /outputs/merged/surface-run.json --output /outputs/meshes --parallel-levels
wf scroll render-ct --mesh /outputs/meshes/level-0000 --source "$CT" --voxel-um 8.64 \
  --level 0 --render-pitch-um 8.64 --output /outputs/render/level-0000
```

`render-ct` uses the same native flatten/render path as `run`, with these differences:

- **Pitch and level.** Pixels are `--render-pitch-um` (a `run` renders at the field
  spacing, 34.56 µm here) and the CT level is chosen independently of the level the
  field was built from.
- **CT prefetch, then offline.** An online render re-verifies every CT chunk against the
  source at the start and end of the render, one HTTPS connection per chunk. On a
  PHerc0175A winding that was 53 of 77 s with every chunk already cached. `render-ct`
  lists the chunks the mesh can touch (triangle bounds plus one voxel for trilinear
  sampling), downloads them into `--cache` over `--prefetch-workers` (default 16)
  parallel connections, sizes the cache so nothing is evicted, and renders `--offline`
  from the ledger-pinned cache. `--prefetch-workers 0` restores the online render.
- **`--flattening-scope sheet` (opt-in).** The default (`charts`) flattens ~1,000-face
  charts and welds each to one parent; every other seam is a cut, and drift between
  charts opens those cuts into gaps or overlapping (blanked, `ambiguous_xyz`) pixels.
  `sheet` flattens each edge-connected piece as **one map**: it is cut to a disk
  (a closed winding is opened along one seam), seeded with a global LSCM and optimized
  by one Flatboi SLIM solve, then stored as raster charts that already agree on every
  shared vertex. There is nothing to weld; the only seams are the disk cuts, and +v
  follows world +Z on every level. A piece that cannot be flattened whole (a failed
  or folded solve) uses the chart route under its own component and is listed in
  `LEVEL.json` → `sheet_flatten.fallback`; per-piece iterations and edge-stretch/area
  percentiles are in `sheet_flatten.pieces`. It needs `--flattening-method slim` and,
  for multi-million-face meshes, a larger `--surface-time-limit`.

  On a synthetic 20 × 14 mm curved, non-developable sheet (radius 8 mm, 80 µm bumps),
  decimated 1/9 to 51,800 faces and rasterized at 8.64 µm:

  | Scope | Cut seams (gap median / max) | Ambiguous + missing pixels | Long-range distance error (p95) | Flatten |
  |---|---|---|---|---|
  | `charts` | 1,222 edges (90 µm / 1.26 mm) | 5.7% + 5.1% | 8.8% | 7 s |
  | `sheet` | 0 | 0% + 0.001% | 0.41% | 22 s |

  The same SLIM solve on 466,216 faces took 90 s (25 steps). These are synthetic
  single runs, not a 175A measurement: check `ambiguous_xyz` in each `RENDER.json` and
  the `sheet_flatten` receipt on real levels before a sweep.

  Real meshes are not clean, so two things happen before the solve and are counted in
  the receipt. `repair`: faces stay joined only across edges with exactly two faces of
  opposite direction, and a vertex is copied once per fan of such faces; a pinched
  vertex, a third face on an edge, a flipped or a duplicated face come apart instead of
  sending the whole sheet to the chart route. `holes` (per piece): a hole whose rim
  turns by about -2 pi is left as a hole; only closed wraps and handles are cut. On a
  synthetic 73,527-face sheet with 12 holes, cutting every hole to the boundary cost
  642 seam vertices and 0.88% ambiguous + 0.90% missing pixels; keeping them costs none.
- **`--decimate N`** collapses the mesh to 1/N of its faces (libigl qslim) before
  flattening; `DECIMATION.json` records the deviation of a vertex sample from the new
  surface. Triangles only need to follow the surface: CT is still sampled at every
  pixel. On PHerc0175A, 1/9 (95 µm edges) moved the surface by at most 1.05-1.35 µm (p99
  0.5 µm; one CT voxel is 8.64 µm) and cut flatten+assemble from 17.6 to 1.7 s; 1/16
  started to raise flattening stretch.
- **Per-tile PNGs.** Each 256-pixel tile is written as `component-*/png/Y-X.png` (the
  same gray + observed-alpha pixels as `sheet.png`) with the usual `overview.png`; there
  is no whole-canvas PNG and no 100M-pixel canvas bound (an outer winding at 8.64 µm is
  about 290M pixels). `--whole-png` restores `sheet.png`/`mask.png`. `sheet-atlas` reads
  `sheet.png` and needs `--whole-png`.
- **TIFXYZ by default.** Per-pixel XYZ is kept and TIFXYZ is written
  (`component-*/tifxyz/{x,y,z}.tif`, `meta.json`), always. 2.0 removed `--export-tifxyz`
  and `--no-export-tifxyz`.
- **`--no-ct-texture` (opt-in).** Flattens and writes the mesh-derived TIFXYZ without
  sampling CT, for consumers that only need the geometry (e.g. ink-sdk). It
  skips the CT prefetch; the CT metadata is still opened for the
  output frame. The TIFXYZ is byte-identical to a textured render; every mapped pixel is
  `unsupported_ct` (mask 1), the PNGs carry no observed alpha, and `LEVEL.json` reports
  `usable_render=false` with reason `ct-texture-not-requested` while the level can still
  be COMPLETE. `wf run --no-ct-texture` does the same for the native surfaces stage
  (the field stage still reads CT). Textured and untextured checkpoints do not mix.

Measured on a merged two-tile winding (98,656 faces, empty cache): 187 s online and
98.98% mapped before; 39 s with prefetch, `--decimate 9` and tiles (5 s decimation, 9 s
for 70 CT chunks, ~13 s flatten and render), 100% mapped, 3.4 MB instead of 21 MB.
