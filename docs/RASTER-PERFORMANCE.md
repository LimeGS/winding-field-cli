# CT and raster performance

These measurements were recorded on **2026-11-05 and 2026-11-06**. They describe
specific fixtures and runtimes, not the speed of every later release or a full
scroll. Current usage and cache settings are in [streaming](STREAMING.md) and
[patch export](PATCHES.md).

**Practical defaults:** keep rasterization serial until your workload shows a
gain; size decoded RAM caches separately from disk caches; compare identical
values, masks and source IDs before trusting a speedup. Cold/warm application
caches are not the same as cold/warm operating-system or network caches.

## Verified float64 native sampler

All CLI CT samplers use native float64 interpolation and return float32 samples, retaining
exact positive-weight support: missing chunks remain invalid, including at chunk
boundaries, while observed zero-valued voxels remain valid. The native computed
reader consumes only PatchCT's hash-verified chunks, with native persistent RAM
caching disabled. The existing shared Python LRU owns retained decoded memory.

The required reader pins LimeGS/villa `84207424c3a988e6cafe6b50fe7d4d7e5f839929`.
PatchCT shares Volume's checked encoded transport and native snapshot decoder,
retaining its separate decoded-byte SHA ledger and bounded RAM cache. It no longer
creates a second native decoded disk cache. Publication failure is an error, never
evidence of absent CT. Zero disk-cache budgets support streaming.

Continuation requires matching sampler source, extension and native-library
hashes. Older reader-bound checkpoints reject before modifying retained products;
use their original v1.5.0 runtime. There is no selectable scalar production path.

### Native-only cutover (macOS arm64, 2026-11-06)

Installed v1.5.0 baseline versus the candidate, run sequentially with
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`. Fixture: deterministic 64³ uint8 CT,
16³ Blosc/LZ4 chunks, 25,000 seeded points and a complete 125×125 planar render.
Three repetitions; medians below. Each repetition starts a new disk/RAM cache,
then repeats the read in the same Volume. OS pages are warm; HTTP is loopback,
not a storage/network throughput measurement. Both readers retain their ledgers.

| Workload | v1.5.0 ms | Native-only ms |
| --- | ---: | ---: |
| Local crop, cold application cache | 7.54 | 8.61 |
| Local crop, warm | 1.30 | 1.73 |
| Local points after crop | 39.24 | 21.95 |
| Local points, warm | 30.84 | 13.67 |
| Complete render, warm | 34.80 | 31.59 |
| HTTP crop, cold application cache | 44.13 | 48.17 |
| HTTP crop, warm | 8.28 | 5.43 |
| HTTP points after crop | 97.69 | 83.81 |
| HTTP points, warm | 45.15 | 27.12 |

Crop/sample/mask arrays and PNG hashes match exactly, including all repetitions.
Peak process RSS was 132.7 MiB versus 106.8 MiB. Local crops regressed while
warm sampling improved; these small fixtures do not establish a universal speedup.

#### Ledger-bound cache correction

The first local-crop regression came from redundant file-signature checks on
ledger-bound decoded cache hits. Current entries use the encoded-byte SHA check
already performed
on every hit, without redundant file signatures. Binding a ledger clears existing
decoded entries, so retained arrays come from that ledger's verified snapshots.
Bounded reads, SHA checks, cache locking/eviction and fresh source validation remain
in place; entries without a ledger retain their file-signature checks.

Four alternating before/after blocks, 20 fresh Volume instances each (80 cold and
80 warm crops per size/runtime), checked every voxel against an independent formula.
Each crop spans 27 Blosc/LZ4 chunks, with enough decoded RAM to retain them all.
These are OS-warm synthetic uint8 arrays on macOS arm64; comparison uses the retained
installed candidate and the corrected source with that same native Release runtime.

| Decoded chunk size | Warm before / after, median ms | Reduction | Cold before / after, median ms |
| --- | ---: | ---: | ---: |
| 4 KiB (16³) | 1.853 / 1.361 | 26.6% | 9.212 / 8.884 |
| 256 KiB (64³) | 2.493 / 2.022 | 18.9% | 10.680 / 10.043 |
| 2 MiB (128³) | 7.945 / 7.467 | 6.0% | 21.425 / 20.573 |

Python-level file-stat calls returned to 27 per crop. Mutation, corruption,
missing chunks, cache eviction and concurrent charging remain checked. See
[end-to-end scope and its TIFXYZ limitation](DOCKER.md#ct-cache-fix-end-to-end-qualification).

### Integrated benchmark (gpu-1, 2026-11-05)

Historical pre-cutover results, not measurements of the current migration:
actual PatchCT on both paths, including per-load hashing, SQLite receipts,
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


### Shared encoded transport migration

Volume uses bounded `ObjectStore.read`, `DiskCache.write` and `decode_chunk`.
Native decoding consumes the exact verified snapshot without reopening its source.
CLI source revalidation uses bounded HTTP and strong ETags; weak validators and
Last-Modified cannot establish identity. Receipts without validators need a full
verification download. Offline verified payloads and authoritative 404 receipts
remain distinct from cache misses and corruption. Encoded reads and decoding are
capped at 256 MiB; disk publication preserves the configured reserve.

The archived pre-cutover source-test experiment also used Python 3.13. It is
not qualification for the supported Python 3.11/3.12 installed environments.
Current raster admission includes per-worker sampling workspace: up to 100,000
points at 1024 bytes each for corner jobs and sorting, in addition to tile/cache
memory. Worker cancellation must not publish unfinished tiles.

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

Historical raw reports are development artifacts, not installed inputs or a
current CI result. For new measurements, retain source versions, cache state,
thread/CPU limits, output hashes and complete elapsed times.

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
