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

`render-ct` uses the same native flatten/render path as `run`, with two differences: the
pixel pitch is `--render-pitch-um` (a `run` renders at the field spacing, 34.56 µm here)
and the CT level is chosen independently of the level the field was built from.
