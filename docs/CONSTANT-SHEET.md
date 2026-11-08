# Constant-W sheets, from exact mesh to CT atlas

Use `winding-field constant-sheet` to prepare, flatten and render one constant-W
mesh. Input: a verified `MESH.json`/`mesh.sqlite` bundle and a centerline.
Output: a CT sheet that can be added to a [sheet atlas](SHEET-ATLAS.md).

W is `raw field level / sheet_step`, not the CT pyramid level. Follow the
[installation guide](../README.md) and install the `tifxyz` extra.
These stages require Linux; native Flatboi also needs its pinned AppRun and a
cgroup v2 memory/swap limit. Docker supplies the [supported runtime](DOCKER.md).

`flatten` and `import-native` default to `--native-recipe installed-2f89bfc9`.
For the official Villa stable release in [native provenance](../provenance/flatboi.json),
pass `--native-recipe historical` explicitly. The selected launcher and binary
must match their recorded SHA256 values. Resume and import preserve binary
identity: they cannot transfer a checkpoint to a different Flatboi release.

## Stages

```sh
# centerline.json is an array [[x,y,z], ...] in the mesh's world coordinates,
# strictly increasing in Z. Use the same reference for neighboring sheets.
winding-field constant-sheet prepare --mesh /data/extraction --w -61 \
  --centerline /data/centerline.json --output /data/W-61-prepared

systemd-run --user --wait --pipe -p MemoryMax=21G -p MemorySwapMax=64G \
  winding-field constant-sheet flatten --prepared /data/W-61-prepared \
  --app-run /opt/vc3d/AppRun --native-recipe historical --output /data/W-61-native

watch -n 2 winding-field constant-sheet status --native /data/W-61-native
# --json gives machine-readable status. Calculadas includes the active batch;
# Guardadas is durable progress. Repeat flatten to resume the last checkpoint.

winding-field constant-sheet validate --prepared /data/W-61-prepared \
  --native /data/W-61-native --reference-z 9500 --output /data/W-61-validated

systemd-run --user --wait --pipe -p MemoryMax=6G -p MemorySwapMax=0 \
  winding-field constant-sheet render --validated /data/W-61-validated \
  --config /data/render.json --output /data/W-61-sheet

winding-field sheet-atlas build --sheets /data/W-61-sheet/SHEET.json --output /data/atlas
winding-field sheet-atlas verify --atlas /data/atlas
winding-field sheet-atlas query /data/atlas 120 300
```

Replace `/data/...`, W and `--reference-z` with your calibrated inputs. The
examples select the official `historical` binary shipped in Docker. To package
several sheets, pass all their `SHEET.json` paths after `--sheets`. On systems
without user systemd, use a host/container that provides the required cgroup
limits; do not omit them to bypass admission.

Preparation selects the component with greatest physical area, retains every
other component separately, and cuts existing edges near +X. An input already
forming a disk needs no further seam; cutting across it can disconnect it. Handles are cut
to a disk; positive mean-value coordinates initialize Flatboi. No XYZ is moved
and every oriented source face is accounted for. Native defaults are cap200,
batches10, two threads, symmetric Dirichlet and tolerance1e−6. Warm starts
reconstruct SLIM state; they are not bit-identical uninterrupted native runs.

Validation reopens the original extraction, checks exact XYZ/topology/source
IDs (including disconnected pieces), audits UV overlaps, scales isotropically
and rotates toward increasing Z. The common reference anchor crosses the
explicit reference Z nearest +X, breaking ties with leftmost UV. An optional
`validate --mesh` locates an unchanged extraction moved to another directory.

## Explicit CT configuration

```json
{
  "ct": {"source": "/data/scan.zarr/2", "voxel_um": 9.6, "level": 0},
  "settings": {
    "output_frame": {"id": "scan:L2", "voxel_um": 9.6, "origin_xyz_um": [0,0,0]},
    "app_run": "/opt/vc3d/AppRun",
    "memory_gib": 6,
    "process_tree_gib": 6,
    "cache_dir": "/data/shared-ct-cache",
    "cache_mib": 2048,
    "cache_gib": 64,
    "raster_workers": 2,
    "ct_workers": 4,
    "tile_size": 1024,
    "disk_reserve_gib": 8
  }
}
```

The frame ID and voxel size must match the mesh; this unchanged global-XYZ
recipe requires zero `origin_xyz_um`. `ct.voxel_um` is the CT
sampling scale; `settings.output_frame.voxel_um` is the world-coordinate scale.
Rendering always uses19.2µm/pixel and window0–255 with the native-float64
sampler, checks scalar CT/XYZ samples and all tile halos, then publishes
`SHEET.json` last, binding the CT chunk ledger as a delivered payload. Masks
retain unmapped0, missing CT1, observed2, ambiguous3. Tile sizes whose mapping
would exceed the128MiB readback bound are rejected before output creation.

Optional `prior_identity`, `prior_ledger`, `prior_ledger_sha256` in this JSON
identify a preserved CT identity receipt and its chunk-hash ledger. Base CT
identity must match before reusing hashes; paths and the ledger digest are
explicit, and the original files remain read-only. The cache is shared through
the normal PatchCT settings, independent of this ledger.

## Reuse existing prepared/native results

Prepared directories produced by the operational recipe can be used directly
after payload verification. Legacy preparation did not embed the centerline;
provide `validate --centerline /data/centerline.json` explicitly. New
preparations embed it and reject a contradictory replacement.

```sh
winding-field constant-sheet import-native --prepared /data/W-59-prepared \
  --native /data/old-W-59/native --runner-sha256 ORIGINAL_RUNNER_SHA256 \
  --app-run /opt/vc3d/AppRun --native-recipe historical --output /data/new-W-59-native
```

The original runner hash is an explicit provenance pin, not a bypass for
source, launcher, binary, recipe or UV validation. `IMPORTED.json` preserves
the original checkpoint and identity, and the new identity binds that receipt.
A converged or cap200 import can go directly to validation; validation requires
the current installed runner hash and the exact preparation receipt bound at
native-run creation/import. Historical runs cannot bypass import. Otherwise resume
`flatten` in the imported directory with the original batch size. Historical
directories are never modified by import. Do not substitute the installed
runner hash for an old one.

Preparation, validation, rendering, import and atlas build require fresh
output directories. Native flatten resumes only an identical identity.
`STOP` or SIGINT/SIGTERM during flatten preserves the last published batch.
Interrupted rendering retains completed tiles but does not publish a sheet;
use a fresh render directory with the already-validated map and shared cache.
All stages keep an8GiB disk reserve. Resource limits are admission/runtime
bounds, not a performance guarantee.

These products are diagnostics. Exact XYZ, pixel checks and finished Flatboi
do not establish physical sheet identity, global injectivity or the2% metric
acceptance gate. Independent sheets keep their own UV; atlas placement is
reversible integer translation and never welding.
