# Bounded raster and CT verification — 2026-11-05

The changes retain scalar raster/interpolation mathematics, original face order
and ownership, ambiguity blockers, missing CT, and two-node tile halos. Native
PointPlan sampling was evaluated but not adopted: 57 of 200,000 rounded pixels
changed (maximum float difference 0.0059814453125), despite exact validity.

## Verification

Experiments ran on gpu-1 in fresh namespaces, independently of the frozen Paris4
three-turn exporter. Relevant suites: 129 passed, 2 skipped, across the raster,
patch pipeline/parallel, original recipe, mapping, range and patch-visuals tests.
The visual test used isolated matplotlib dependencies with original NumPy 2.4.6.
These results do not constitute a whole-repository test pass.

Independent review identified omitted per-worker sampling workspace. Two
regressions failed before the fix; admission now reserves up to 100,000 points
at 1024 bytes each per worker for overlapping corner jobs and sorting arrays.
Additional tests exercise concurrent shared PatchCT/SQLite access and running
worker cancellation without publishing later receipts.

## Measurements and limits

- Real-map geometry, fake CT: exact XYZ/mask/owner/image hashes on two full tile
  canvases. Populated tile: 115.17 → 14.32 seconds; empty corner: 134.46 → 2.34.
  Other validation shared the allocated CPU, so these timings are not isolated.
- 200,000 real CT points: first read 34.25 s, RAM-warm 0.386 s; identical results.
  Separate SSD-warm/RAM-cold read 1.644 s versus RAM-warm 0.397 s.
- Initial two-full-tile concurrency comparison was **not fully RAM-cached**:
  839 chunks require 1.64 GiB, exceeding its 1 GiB LRU. Its 36.87 s serial versus
  48.27 s parallel cannot establish equal-resident-cache performance.
- Controlled 128×128 subwindows: same source, reader, points and 30 chunks
  (~60 MiB); zero native loads in all measured passes and exact output hashes.
  Serial 5.304/4.117 s; two workers 3.212/2.959 s. Mean speedup 1.53× applies
  only to these subwindows, excluding compression and publication.

- Full 1024×1024 tiles, including PNG, compressed mapping NPZ, hashes, ordered
  receipts and file fsync: serial 5.569/5.430 s; two workers 4.059/3.919 s.
  Means: 5.500 versus 3.989 s, **1.38× throughput (27.5% less elapsed time)**.
  Same source, reader and coordinates; all 111 chunks (232,840,704 accounted
  bytes) stayed in RAM, with zero native loads in every measured pass. Exact
  XYZ/mask/source-owner/image hash assertions passed. Both modes had CPU6–7;
  the process had a 1152 MiB hard cap and zero swap. These complete tiles have
  partial surface coverage (76,389 and 98,040 observed pixels). The benchmark
  maps saved source face IDs, rather than transient source face row indices.
  TIFXYZ and dense-tile throughput were not measured.

The serial default remains appropriate until full-workload gains are established.
No measurement here establishes a complete extraction/flattening/export speedup
or physical-sheet continuity. Frozen active jobs are not upgraded in place.

Remote evidence directories:

- `/ssd/draft/paris4-raster-parity-20261105`
- `/ssd/draft/paris4-native-sampling-20261105-02`
- `/ssd/draft/paris4-raster-threads-20261105`
- `/ssd/draft/paris4-raster-threads-cached-20261105`
- `/ssd/draft/paris4-full-tile-benchmark-20261105-02`
