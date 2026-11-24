# Whole-scroll runs: scripts and measured lessons

Operational scripts for building a whole-scroll field with `winding-field scroll`
(coarse → tiles → merge → surfaces → viewer) on rented machines, and the numbers behind
their defaults. They were written while running PHerc0175A (8.64 µm CT, 34.56 µm field,
138.24 µm coarse) in November 2026. See [docs/SCROLL.md](../../docs/SCROLL.md) for the
method.

## Recipe

1. **Inputs, straight from their origins, before any run**
   - Fibres: the AFV from its dataset (`fetch_ranged.py` downloads one large file over
     parallel range requests; `HF_TOKEN` is read from the environment). Keep fibres longer
     than 1 mm with `afv_filter.py` (PHerc0257: 11.05 M → 1.34 M fibres, 7.3 GiB, 22 s);
     an unfiltered import takes many hours.
   - Lasagna: the levels named by the manifest groups (`cos/1`, `grad_mag/2`, `nx/2`,
     `ny/2`). `mirror_s3_prefix.py` mirrors a public S3 prefix (1.39 M objects, 40 GiB in
     ~25 min). Hugging Face limits unauthenticated clients to 3,000 files per 5 minutes
     (5,000 with a token), so a 180,000-file Lasagna takes hours there;
     `fetch_lasagna_ledger.py` fetches and verifies exactly the chunks a prepared run pinned.
   - CT: `prefetch_ct.py` fills a local cache for level 2 (tiles) and level 0 (renders).
2. **Counted density** (`run-scroll.sh counted`, on by default): sheet density counted
   on Lasagna `cos` instead of `grad_mag` (see docs/SCROLL.md, step 0). Every field stage
   then reads `$ROOT/counted/counted.lasagna.json`. `COUNTED_JOBS`/`COUNTED_GPUS` split it
   between GPUs.
3. **Bounds**: the 99.9 % extent of fibre length plus a margin, in level-2 voxels
   (`BOUNDS_L2` in `scroll.env`). The absolute extent includes thin tails of stray fibres.
4. **Coarse, as ONE block** on a GPU that holds it (`--field-chunk` ≥ the grid,
   `--sweeps 1`). Memory in float64: about 121 bytes per node plus about 419 bytes per fibre
   constraint row (PHerc0175A: 122 M nodes, 58.4 M rows, ~50 GB; 10,457 PCG iterations,
   19 min on an RTX PRO 6000). Prepare (Lasagna and fibre import) on a cheap many-core
   machine and move only the prepared run to the GPU host.
5. **`run-scroll.sh plan tiles merge gate surfaces`** with `COARSE_DIR` pointing at the
   coarse, `TILE_CACHE` at the prefetched level-2 CT, and everything `--offline`. Tiles
   use `--fibre-weight 1000` and are solved as one block each (`TILE_NODES=128`,
   `OVERLAP_NODES=64`); the merge joins them on their overlaps, not to the coarse.
6. **`winding-field scroll site`** for the Deep Zoom viewer.

`field_status.py RUN` (under `watch`) shows phase, progress, ETA and per-sweep changes.

## What went wrong, with numbers

| Finding | Evidence (PHerc0175A) | Consequence |
|---|---|---|
| A blocked coarse with the 2-node halo does not converge | vs the one-block solution: median 11.0 windings (p99 66) after 8 sweeps; the per-sweep change looked small | Solve the coarse as one block. Blocked: `--field-halo 32` reaches median 0.38 after 16 sweeps (0.148 per-sweep change), still 16 % of nodes over half a winding |
| Per-sweep change underestimates the error | halo 32, sweep 8: change 1.30, error up to 3.7 | Judge a blocked field against a one-block reference (`compare_fields.py`) |
| Tiles calibrated orientation from their own ROI | 49.8 % of seam nodes over half a winding, 12.6 % over four | `scroll plan` passes `--field-prior-calibration inherit`; `seam_gate.py` stops before surfaces |
| Online CT re-verifies every cached chunk against the source | 13–30 min per resume at ~6 MB/s; a 2 GiB `cache_bytes` evicts and re-downloads in a loop | Prefetch, then `--offline` with `--cache-gib` above the cache size, for every run, resume, tile and render |
| A resume re-hashes the Lasagna chunks it imported | manifest-only copy failed with `source chunk changed` | The Lasagna levels in use (40 GiB) must be local on the host that resumes |
| GPU sharing is not the tile bottleneck | 1, 4, 8, 12 concurrent solvers: 4.7, 4.6, 4.6, 5.4 block solves/s in total | NVIDIA MPS does not help; float64 throughput does (consumer GPUs run fp64 at 1/64 rate) |
| Whole-scroll meshes are not collapsible as they come | 68 of 821,728 vertices shared by two fans made libigl's edge collapse return nothing; a three-face edge makes it hang | `decimate_bundle` splits fans, drops degenerate faces and refuses instead of hanging |
| Mesh extraction re-decoded a one-chunk field per tile | one level of the one-block coarse: 42 min, 77 s with the chunk cached | The mesh reader keeps one stored chunk |
| Mesh extraction scans every tile of the grid | 34.56 µm field: 32,240 tiles per level at ~10 tiles/s, ~50 min per level on one core | Run many levels in parallel (`BATCH`); a per-chunk W range index would let a level skip most tiles |
| Lasagna grad_mag overcounts sheets | 1.4-1.7 field units per sheet (median grad_mag/cos frequency 1.57); levels crossed 5-6 sheets in 12 mm where the sheets are clean | `scroll counted-density` (October recipe): 0.99 units per sheet on the pilot |
| Fibres at weight 1 do not hold a level on its sheet | along fibres the field changed by a median 1.14 units | `--fibre-weight 1000` for 34.56 µm tiles, solved as one block (64-node blocks drift ~1 sheet) |
| The coarse-grid merge correction undid the tiles' sheet-following | same 4 tiles: level drifted 14 sheets in 16 mm after the coarse correction, 0.6 after the overlap join | `scroll merge --join overlap` (default) |
| Seam statistics could not be written | p95 in the open last bin was `inf` | Fixed; `SEAMS.json` reports `over_four_windings` |

## Scripts

| Script | Purpose |
|---|---|
| `run-scroll.sh`, `scroll.env.example` | One scroll on one machine, resumable by stage |
| `bootstrap-native.sh` | Native install on a stock Ubuntu 22.04/24.04 + NVIDIA host (no Docker) |
| `prefetch_ct.py` | Parallel CT prefetch: whole level, voxel bounds, or a run's ledger |
| `mirror_s3_prefix.py`, `fetch_ranged.py`, `fetch_lasagna_ledger.py` | Input downloads |
| `afv_filter.py` | Copy an AFV keeping fibres longer than a length (geometry unchanged) |
| `field_status.py` | Live status of a field solve |
| `compare_fields.py` | Distance of fields from a reference field on common observed nodes |
| `tile_vs_coarse.py` | A tile's difference from the coarse, split above and below 0.5 mm |
| `seam_gate.py` | Fail when too many seam nodes differ by more than half a winding |
| `check_windings.py` | Field winding count against CT layers along rays from the scroll axis |

The Python tools import `winding_field`; run them with the CLI's interpreter.
