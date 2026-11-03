# Experimental global native export

`tools/bench_global_native.py` runs an explicitly selected component of a genuine
indexed native mesh through global cutting, mean-value initialization, direct
libigl ARAP, geometric qualification and the shared native renderer. It produces
CT images and optional mesh-derived TIFXYZ with bounded workers and resumable
phase checkpoints. The public `native` workflow default is still the local-chart
path; this command is the explicit experimental global path.

## Run

Use an existing Linux environment containing the project's dependencies and
optional libigl/tifffile. No dependency is installed by the command.

```bash
python tools/bench_global_native.py \
  --mesh /path/to/indexed-mesh \
  --component 1 \
  --ct https://example.org/original-volume.zarr \
  --frame frame.json \
  --output /path/to/new-global-export \
  --pitch-um 8.64 \
  --coverage-target 0.90 \
  --seconds 1800 \
  --export-tifxyz
```

Component is an explicit `mesh.sqlite` component root, not an implicit largest
selection. Input must contain one level and coordinates in micrometres
(`metadata.voxel_um=1`). Frame ID must match the mesh metadata; the CT reader
independently checks the declared dimensions, origin and spacing. For historical
175A, `frame.json` is:

```json
{
  "id": "PHerc0175A/20250521115057",
  "origin_xyz_um": [0, 0, 0],
  "spacing_xyz_um": [8.64, 8.64, 8.64],
  "shape_zyx": [12748, 9363, 9363],
  "voxel_um": 8.64
}
```

Resume with identical arguments and `--resume`. A `STOP` file in the output or
Ctrl+C requests a supervised pause. Remove an intentional STOP before resuming.
Active time remains cumulative; clean pauses do not consume time. Interrupted
controller lifetimes are conservatively charged, and an exhausted budget does
not authorize unbounded verification.

`STATUS.json` reports progress and failure. Final `RESULT.json` is accompanied by
`PUBLICATION-RECORD.json` and the last-written `PUBLICATION.json` seal. Resume
compares claims with the verified phase records and supervises source, artifact
and CT verification. A partial result uses exit2; it is not a direct-export READY.

## Meaning and limits

- The denominator is the **whole input mesh**. Unselected components, invalid
  faces and both participants of every detected UV overlap are explicitly recorded.
- Qualified faces are stored in internal chunks with UVs from **one global map**;
  these chunks are not independently positioned charts or separate deliverables.
- Source-triangle area coverage precedes TIF cell masking. It is not a claim of
  retained-grid source-area coverage or full winding-field coverage.
- TIFXYZ keeps only locally qualified reader cells, with separate unsafe-cell,
  removed-node, collateral-cell and orphan counts. Retained XYZ is unchanged.
- Any omitted source or grid geometry produces PARTIAL. `target_met`, image
  contrast, retained-grid connectivity and physical correctness are separate facts.
- `physical_acceptance=false` and `field_coverage_certified=false` remain explicit.
- Component limits are1,250,000faces and750,000source/cut vertices. Numeric units
  are≤64MiB; overlap qualification is bounded to20Mnarrow tests. Exhaustion fails
  without relabeling an incomplete audit as success.
- Default resources:3GiBaggregate RSS,2GiBRAM reserve,16GiBscratch and12.25GiBdisk
  reserve. Separate phase workers release sparse factors; DATA admission uses
  current headroom. Numerical libraries use one thread and no CUDA.
- Timing starts from an **existing indexed mesh**. Mesh generation, the original
  field solve, physical-sheet validation and the whole-field W−4 speed goal are
  separate measurements.

## Evidence before the fresh-command acceptance run

The earlier source-bound W−50 global control qualified584001faces,95.8598%of the
whole610448-face input's area. Its1071×1083TIFXYZ retained692868nodes and686230cells;
99.212%of those cells belonged to one connected patch. Actual VC3D recovered
692868vertices and1372460triangles, with matching topology and≤5.000002e−7L0 text
error. This control reused earlier UV computation, so its14m17s operational wall
time is **not** a fresh-command end-to-end benchmark.

Local delivery:
`.superpowers/sdd/2026-10-31-fast-native-surfaces/delivery/W-50-global-control/`.
The image contains CT layering, cuts and gaps; it does not establish a physically
isolated papyrus sheet or readable text.

Independent review found source-capture races, an unbound resume summary, and
controller-side verification outside supervision. All three were reproduced and
fixed, including typed resealing attacks and real verification cancellation.
Latest local run:756passed32skipped1deselected in310.90s. The deselected existing
large-SQLite test lacked local disk headroom; both cases then passed remotely.
Frozen Linux tests for the new math/controller/renderer and that SQLite test:
122passed2opt-in skips in188.66s.

## Fresh-command acceptance

The frozen command completed from the genuine indexed W−50 mesh in **445.725s
(7m26s)**, including its final supervised verification, without saved UV reuse.
It qualified **584001/610448faces and95.8598%of whole-mesh source area**. The
1071×1083TIFXYZ retained692868nodes/686230cells; the largest connected retained
patch contains680824cells (**99.2122%**). Status remains **PARTIAL**, with
`target_met=true`; mapped source-area coverage and retained-grid connectivity
are different quantities.

| Phase | Supervised seconds |
| --- | ---: |
| Source |9.58|
| Cut |19.77|
| MVC |65.03|
| ARAP |30.93|
| Qualification |66.58|
| Register/render |210.89|
| Initial verification |41.60|

Peak aggregate RSS was **1034539008bytes (~987MiB)**, minimum RAM headroom
3242954752bytes, peak scratch234042500bytes, and minimum free disk81647763456bytes.
Original CT was fetched into a cold144703488-byte cache. Existing mesh generation
and import are outside this fresh mesh-to-render timing.

Actual VC3D exited0 in6.93s, with child peakRSS101359616bytes. Independent checking
recovered692868vertices/1372460faces, exact cell topology and XYZ text error
≤5.000002e−7L0. Mask reconstruction, original/final gate replay and retained-XYZ
bit parity passed. Reader/checking are separate acceptance costs, not included
silently in the445.725s command time.

The same command with `--resume` spent42.20active seconds on supervised verification.
Its cumulative clock became487.937s;262source/phase/publication files retained
identical bytes and mtimes, including the original result/timing. All owned jobs
exited. The local7.03MiB package passed23checksum checks and contains PNG/TIFXYZ,
source/exclusion receipts and reproduction commands:

`.superpowers/sdd/2026-10-31-fast-native-surfaces/delivery/W-50-global-command/`

Fresh W−50 command acceptance is retained in
`gpu-1:/ssd/draft/native-global-command-20261102-01/`, frozen manifest
`f227a03ef386e622ae52cc6b98e70cec3721adaf2f5e15c446615fceb6f6bbc2`.
The complete acceptance record is `acceptance-01/RESULT.json`; the reproducible
commands are in `acceptance-01/REPRODUCE.json` and the local delivery.
This establishes an experimental coherent native export for this case, not a
physical papyrus-sheet identification or whole-field W−4 throughput.
