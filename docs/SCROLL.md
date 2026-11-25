# Whole-scroll fields from tiles

`winding-field scroll` builds one fine field for a whole scroll by splitting it into
overlapping tiles that run independently (on one machine or many), then blends them
into a single field that the existing mesh and render commands read. Every tile is an
ordinary `winding-field run`; nothing here changes how a tile is solved.

```text
scroll counted-density ──► coarse run ──► scroll plan ──► scroll tiles ──► scroll merge ──► prepared mesh ──► scroll render-ct
 counted.lasagna.json       (prior)        TILES.json      tiles/<id>/       MERGED/          per W level       CT at full pitch
                                           fibres/<id>.afv                   field/, LEVELS.json
```

## 0. Sheet density counted on Lasagna `cos`

```sh
wf scroll counted-density --lasagna-input /data/lasagna/input.lasagna.json \
  --bounds 1488 11488 2144 8608 992 8704 --output /outputs/counted
```

Every later stage takes `/outputs/counted/counted.lasagna.json` as `--lasagna-input`.

**Why.** Lasagna's `grad_mag` is a learned sheet frequency. On PHerc0175A it reads
1.4-1.7 times the sheets that are there (median 1.57 against the `cos` channel; `cos`
itself matches CT in 23 of 23 checked regions), and fields solved with it count that
many windings: their level surfaces cross a sheet every few millimetres. The October 2026
PHerc0175A fields counted sheets on the sheet mask instead, and their meshes stay on one
sheet where the sheets are ordered. This command is that recipe, unchanged: threshold
`cos` at mid-grey, structure-tensor normals of the mask, mask crossings one voxel along
the normal on both sides (x 1/4), Gaussian 6 voxels; planar sheets of period 16 return
1/16 within 2 %. It writes `grad_mag`/`nx`/`ny` on the `cos` grid (17.28 µm for
PHerc0175A) and a copy of the manifest pointing at them; `cos` is kept.

On a 5 mm PHerc0175A pilot (34.56 µm, one block): 0.99 field units per sheet instead of
1.70. `--bounds` are level-0 voxels; tiles of `--tile` cos voxels (default 192) are
computed on the GPU with a 40-voxel halo, tiles with less than `--min-fill` nonzero `cos`
are skipped, and a rerun continues. `--part K N` splits the tiles between processes
(one per GPU); the last part to finish writes the manifest.

## 1. Coarse field (the prior)

Tiles need a completed coarse field of the same scroll **at the same CT level**: the
field prior is refused when the frames differ (for example a level-0 coarse field for
level-2 tiles). A 138.24 µm coarse field at level 2 is stride 4:

```sh
wf run --source "$CT" --voxel-um 8.64 --level 2 --stride 4 --field-spacing-um 138.24 \
  --fibres /data/scroll.afv --lasagna-input /data/lasagna/input.lasagna.json \
  --field-chunk 640 --sweeps 1 --rtol 1e-7 --maxiter 50000 --until field \
  --output /outputs/coarse
```

**Solve the coarse as one block** (`--field-chunk` at least the largest grid dimension,
one sweep): the merge takes every scale above `--coarse-sigma-mm` from it, so an
unconverged coarse puts the wrong winding count into the whole scroll. Blocks exchange
information only through their overlap, and with the default 2-node halo a whole-scroll
coarse does not converge. Measured on PHerc0175A (625 × 404 × 482 nodes) against the
one-block solution, after removing the constant offset:

| Coarse | Median \|dW\| | p99 | Nodes over half a winding |
|---|---|---|---|
| 12 blocks of 256, halo 2, 8 sweeps | 10.98 | 65.9 | 97 % |
| halo 32 (`--field-halo 32`), 8 sweeps | 1.57 | 3.04 | 86 % |
| halo 32, 16 sweeps | 0.38 | 0.62 | 16 % |
| one block | 0 (reference) | | |

The per-sweep change does not show this (halo 32, sweep 16: 0.148). One block needs GPU
memory for the whole system in float64, about 121 bytes per node plus about 419 bytes per
fibre constraint row (~50 GB here; 19 min on a 96 GB GPU). Operational scripts and the
other measured pitfalls of a whole-scroll run are in
[tools/scroll](../tools/scroll/README.md).

## 2. Plan

```sh
wf scroll plan --source "$CT" --voxel-um 8.64 --level 2 --field-spacing-um 34.56 \
  --coarse /outputs/coarse --fibres /data/scroll.afv \
  --lasagna-input /data/lasagna/input.lasagna.json \
  --gpu-memory-gib 80 --output /outputs/plan
```

- **One global grid.** The field spacing must be a whole multiple of the selected CT
  level's spacing (34.56 µm is 4 × 8.64 at level 0 and 1 × 34.56 at level 2). Tile ROIs
  start on that grid, so every tile's nodes coincide with the global nodes.
- **Tiles.** Cores of `--tile-nodes` partition each axis; each tile adds
  `--overlap-nodes` (default 32, a multiple of 16) on interior sides. Cores are
  multiples of 64 (the field chunk); the merge reads tiles at any node offset. 32 nodes
  per side (1.1 mm) is twice the overlap the 22 mm PHerc0175A box was joined with
  (0.14 sheets median); 64 needs ~1.7x the node solves.
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

### Strong fibres, one block per tile

With counted density, fibres decide whether a level stays on its sheet. Fibres mostly
run along one sheet (72 % stay within half a sheet spacing), but at the default weight 1
the field does not follow them. Measured on a 22 mm PHerc0175A box at 34.56 µm, solved as
one block, against the Lasagna sheet channel along the October meshes' own lines:

| Fine field | Units per mm | Net drift off the sheet over ~20 mm | Back and forth per cm |
|---|---|---|---|
| Lasagna grad_mag, fibres 1 | 12.1 | 0.7 sheets | 2.2 |
| counted density, `--fibre-weight 1000` | 7.6 | 0.2 sheets | 0.9 |

Strong fibres need the tile solved as **one block** (`--run-arg=--field-chunk=N` with N at
least the tile's nodes per axis, `--sweeps 1`, `--rtol 1e-7 --maxiter 50000`): the same
tiles solved as 64-node blocks for 8 sweeps counted 8.7-10.1 units per mm and differed
from the one-block solution by a 95th percentile of 0.75-0.93 sheets (halo 16; 1.5 with
the default halo 2). Keep weight 1 for the 138 µm coarse: there strong fibres compress
the count (weight 1000: 2.2 units per mm where 6.3 sheets per mm are).

### Tile size from GPU memory (`--tile-nodes auto`, the default)

A one-block tile holds its whole system on one GPU, and every tile also solves its
overlap: with 64 overlap nodes per side PHerc0175A needs 49.6 G node solves with 128-node
cores, 22.5 G with 256 and 13.5 G with 512. `scroll plan` picks the largest core (a
multiple of 64) whose tile fits the GPU: `--gpu-memory-gib` when given (plan for the machine that will run the tiles), else
the smallest GPU visible here; `--jobs-per-gpu N` divides it between N tiles. Auto tiles
run as one block (`--field-chunk` = the tile edge, one sweep, `--rtol 1e-7
--maxiter 50000`); `--run-arg` flags still come last and win. `TILES.json` →
`tile_memory` records the choice.

The memory model is 121 bytes per node and 419 per fibre constraint row in float64, 25 %
allocator overhead, 0.5 GB for the CUDA context, at most 85 % of the GPU, and 1.5x the
densest measured fibre rows (0.045 per node at 34.56 µm, scaled with the square of the
spacing). Checked on a 256³ PHerc0175A tile: 3.1 GB measured, 3.4 GB modelled.

| GPU memory | 6 GB | 16 GB | 24 GB | 32 GB | 48 GB | 80 GB | 96 GB | 192 GB |
|---|---|---|---|---|---|---|---|---|
| Core nodes per axis, AMG (default) | 128 | 192 | 256 | 256 | 320 | 384 | 448 | 576 |
| Core nodes per axis, Jacobi (`--tile-preconditioner jacobi`) | 192 | 320 | 384 | 448 | 512 | 640 | 704 | 896 |

(34.56 µm, overlap 32.)

**Preconditioner (`--tile-preconditioner auto`, the default).** One-block tiles use AMG
(`--preconditioner amg`) on the largest tile that fits AMG's memory model, down to a floor
of 192 nodes per axis; only below that floor does a tile use Jacobi, sized with Jacobi's
smaller footprint. Fixed `--tile-nodes` solve 64-node blocks, where the AMG setup does not
pay off, so they keep Jacobi. On a real 192³ PHerc0175A tile on the GTX 1660 AMG needed
41 CG iterations instead of 2,888 (7.8 s with a 3.0 s setup, against 18.3 s), and Jacobi's
iterations grow with the tile (6,972 at 256³) while AMG's barely do. AMG's hierarchy adds
350 bytes per node to the model: that tile peaked at 4.37 GB on the GPU, and the model
gives 4.9 GB. `TILES.json` → `tile_memory.preconditioner` records the choice; the solver
records it in each tile's block identity, so a tile keeps it when resumed.

**Larger is not cheaper per node.** Measured alone on a GTX 1660, the solve took 24 s for
128³ nodes, 142 s for 192³ and 538 s for 256³: 11, 20 and 32 µs per node, about the tile
edge to the power 1.5 (iterations grow with the tile), with the memory controller busy
100 % of the solve (bandwidth-bound). Extrapolated to PHerc0175A with 64-node overlaps,
cores of 256-320 need the least GPU time and 576 about 40 % more; beyond 256³ this is not
measured. Measure a mid-size and a large tile on the target GPU before a whole run.
`scroll tiles` runs one tile per visible GPU at a time (times the plan's jobs per GPU),
each process seeing only its GPU (`CUDA_VISIBLE_DEVICES`); `--gpus 0,2` picks GPUs and
`--jobs` overrides the count.

Measured cost of that 256³ tile (16.8 M nodes, 752,370 fibre rows) on a GTX 1660
(192 GB/s): 12.8 min in all (prepare 39 s, fibres 9 s,
field 12 min of which 9 min solving); with fibre weight 1 the solve took 5 min.
`tools/scroll/run-scroll.sh` sets all of this (`FIBRE_WEIGHT`, `TILE_NODES=auto`,
`OVERLAP_NODES`, `JOBS_PER_GPU`, `GPU_MEMORY_GIB`).

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

**Overlap join (default, `--join overlap`).** Tiles are solved independently, so each
has its own level and a gentle slope (set by its single gauge node and its boundaries),
and neighbours disagree on the nodes they share. The merge gives every tile one constant
and one tilt (four numbers) and solves them together from all overlaps (every 4th shared
node), choosing the smallest corrections; tilts are charged (0.1 per shared node), so a
step between two tiles becomes constants, not ramps. The tiles' sheet-following stays
untouched. On the 22 mm PHerc0175A box split into 4 tiles of the October recipe, against
the same box solved as one block: median |ΔW| 0.14 sheets, 95th percentile 0.52; the
joined level followed the sheet like the one-block one (net drift 0.6 sheets in 20 mm).
`FIELD.json` → `join` reports the overlap RMS before and after (there 4.7 → 0.21).

**Coarse-grid correction (`--join coarse`).** The previous default. Pulling each tile to
the coarse field also removes its sheet-following wherever the coarse grid is wider than
the sheet spacing: on the same box the joined level drifted 14 sheets in 16 mm. Kept for
fields whose tiles do not follow sheets and for comparison. A tile is gauged by a single prior node and
relaxed for a few sweeps, so its low frequencies drift from the coarse field, differently
in every tile. On the PHerc0175A pilot two neighbouring tiles disagreed in their overlap
by a median 0.21 and a 95th percentile of 1.0 winding (21% of nodes by more than half a
winding). Before blending, the merge subtracts from every tile a Gaussian low-pass of
`(tile - coarse)` (`--coarse-sigma-mm`, default 0.5 mm, computed on a 4x reduced grid):
detail below that scale comes from the tile, everything coarser from the one coarse
solve that spans all tiles. On the pilot this gave a median 0.018 and a 95th percentile
of 0.135 (0.26% of nodes over half a winding). Smaller sigma agrees better but keeps less
fine detail (sigma 0.25 mm: 0.043; 1 mm: 0.31). The global winding structure therefore
comes from the coarse field. `--join none` (or `--coarse-sigma-mm 0`) blends the raw
tiles.

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

**What one level costs.** `--parallel-levels` runs one process per level, and each walks
every 64-node tile of the field in order (32,240 for PHerc0175A at 34.56 µm): levels fill
the cores, one level never uses more than one. A PHerc0175A level of 12.07 M faces took
230 bytes per face in `mesh.sqlite` (2.77 GB) and under 0.6 GB of RAM. Leave free at least
the size of that database again for the closing step, which rewrites the component of
nearly every face in one transaction and then writes the `.obj` (about 50 bytes per face):
a run with 0.66 GB free was still growing its rollback journal at 0.28 GB when it was
stopped. Write meshes to local NVMe; on a rotational disk the commits were 24 % of the
time, on NVMe 4 %. The field reader keeps up to 256 MiB of decoded chunks per level;
`--cache-bytes` large enough for two Z planes of chunks (1.7 GB for that field) removes
the remaining repeated decodes (25.0 s against 33.8 s on the first 1,612 tiles).

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
- **A slab, not one layer.** A level of the field runs parallel to the sheets but does not
  stay on one: on PHerc0175A five levels were on a Lasagna sheet over 48-58% of their area
  (52.6% of the papyrus volume is sheet) and swung between sheet and gap 2.7-5.0 times per
  centimetre. `render-ct` therefore samples the CT every `--render-pitch-um` along the
  mesh normal over `--thickness-um` and writes their `--composite` (`max`, default, or
  `mean`); `--thickness-um 0` is the single surface layer and `--keep-layers` also stores
  every layer per tile. The default is 70 µm (9 layers at 8.64 µm), about half the sheet
  spacing: a thinner slab leaves gaps, a thicker one reaches the neighbouring sheet.
  PHerc0175A sheets are 142 µm apart (median of 141 zones of the Lasagna `cos` channel;
  13% of the zones under 120 µm). On a 6 x 7 mm patch the single layer showed dark patches
  where the surface left the sheet, 70 µm and 120 µm a continuous fibre grid (a few darker
  spots left at 70), and 240 µm started to overlay the next sheet. CT sampling took about
  eight times longer with 15 layers. The image still mixes stretches of neighbouring
  sheets wherever the level crosses from one to the next.
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
