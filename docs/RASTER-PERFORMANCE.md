# Bounded raster and CT verification — 2026-11-05

The initial changes retained scalar interpolation, original face order
and ownership, ambiguity blockers, missing CT, and two-node tile halos. The earlier
native PointPlan sampler was evaluated but not adopted: 57 of 200,000 rounded pixels
changed (maximum float difference 0.0059814453125), despite exact validity.

## Verified float64 native sampler

New patch exports default to `"ct_sampler": "native-float64"`; `"scalar"`
selects the original NumPy accumulation. Both return float32 samples and retain
exact positive-weight support: missing chunks remain invalid, including at chunk
boundaries, while observed zero-valued voxels remain valid. The native computed
reader consumes only PatchCT's hash-verified chunks, with native persistent RAM
caching disabled. The existing shared Python LRU owns retained decoded memory.

The CT extra pins vc_reader `baf5e605bb42a3d55a62b7c5c728e0908137225c`.
That release removed raw chunk/publication bindings. PatchCT reads its declared
lossless decoded cache, including explicit `.missing` markers; uncached remote
chunks are loaded through native indexing and flushed before use. Publication
failure is an error, never evidence of absent CT. Local whole chunks are read and
decoded directly. Every demand load retains the decoded-byte SHA ledger check.

Continuation inherits its saved sampler; receipts predating this setting use
scalar. Native continuation requires matching sampler source, extension and
native-library hashes, preventing a mixture of implementations in saved tiles.
New exports record these identities; existing completed exports stay immutable.

### Integrated benchmark (gpu-1, 2026-11-05)

Actual PatchCT on both paths, including per-load hashing, SQLite receipts,
four-worker prefetch and a 512 MiB verified LRU. Three alternating repetitions,
200,000 fixed points and two complete 1024×1024 tiles; one native thread, CPUs 6–7.

| Workload | Scalar median | Native float64 median | Speedup |
| --- | ---: | ---: | ---: |
| Points, RAM cold | 0.9041 s | 0.6917 s | 1.31× |
| Points, RAM warm | 0.2816 s | 0.1039 s | 2.71× |
| Two tiles, RAM cold | 5.2423 s | 5.1427 s | 1.019× |
| Two tiles, RAM warm | 4.5831 s | 4.5026 s | 1.018× |

All compared public sample values, masks, rounded pixels, XYZ and source-face IDs
matched exactly. Tile timings include geometry, halos, PNG/NPZ compression,
hashes and fsync. Peak process RSS was 677,380 KiB. These are local raw fixtures,
with the OS cache warmed by input verification; neither network-cold, CUDA nor
whole-map performance was measured. The smaller end-to-end gain supersedes any
interpretation of the isolated sampler's speedup as a complete-render speedup.

Evidence: `/mnt/fast2/paris4-vcreader2440-20261105/cli-benchmark-01/benchmark02/RESULTS.json`.

### Shared encoded transport migration

Shared Volume reads now use `DiskCache.open_file`. Its new API trusts existing
disk bytes, so the CLI owns source revalidation using its existing bounded HTTP
fetch and a separate strong-ETag table. Weak validators and Last-Modified cannot
establish identity. A receipt without a validator needs one full payload fetch
on its next validation epoch; later epochs can use conditional requests. Native
replacement downloads invalidate old validators. Offline verified hits and
authoritative 404 receipts remain distinct from cache misses and corruption.
Sized and streaming unsized transfers retain the 256 MiB reservation cap and
the disk reserve; a small unsized response may now succeed within those bounds.
The native streaming allocator reserves ahead in increments, so a near-cap
unsized response can be refused before reaching 256 MiB of payload.
Old encoded-cache ownership is migrated before native reconciliation, preserving
disk accounting and eviction. For Delta3D decoded caches, D3D1 v1/v2 payload
quantization is checked independently of metadata; unrecognized wire formats fail
closed. Existing payloads take precedence over stale missing markers.

Final integration regression run on gpu-1: **145 passed, 2 skipped** across
native sampling, shared transport, prefetch, pipeline/mapping/parallelism,
continuation, CT sources, IO, visual geometry and the original recipe. The run
reported 50 dependency deprecation warnings from scikit-image with NumPy 2.5.
This is an isolated Python 3.13 source-test runtime, not an installed-package
qualification for the advertised Python 3.11/3.12 environments.

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

## Saved full-map overlap audit

The Paris4 Flatboi 88-iteration checkpoint was audited without repeating extraction
or flattening. Radius-bucket trees replace one global-radius tree; the candidate
cap covers all buckets together, with the same exact intersection predicate and
ordered face pairs. Tests: 21 passed, 9 optional-runtime skips; exact original/new
pair and count parity on 16,041 real faces.

The full Flatboi audit completed in 143.57 s at 685 MiB peak RSS under a 6 GiB,
no-swap limit: 22,774,809 narrow tests, 6,634 overlap pairs and 2,141 affected faces.
The retained LSCM audit has 7,705 pairs and 2,732 affected faces. Both maps have
zero inverted faces; these are diagnostic results, with overlap ambiguity masked
during rasterization, not physical acceptance or global injectivity.

Evidence: `gpu-1:/mnt/fast2/paris4-vcreader2440-20261105/final-map-validation-02/`.
The original one-hour timeout remains recorded in `final-map-validation-01/`.
