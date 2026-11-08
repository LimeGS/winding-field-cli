# CT input and streaming

The CLI reads local or public HTTP/S3 **3D Zarr v2** data without loading the
whole scan into RAM. Start with a small ROI, inspect `plan`, then run with the
same options. Zarr v3 and unsupported codecs are rejected.

## Choose access and inspect usage

```bash
winding-field plan --source /data/scroll.zarr --voxel-um 8.64 \
  --ct-mode stream --cache-gib 2 --reserve-gib 1 --until field --output runs/field
```

Replace the source and scale with your scan, add `--roi` for a bounded region,
then use `run` with the same arguments. `plan` does not create the output.
No `--roi` means the entire selected CT level, which may exceed available disk.
For a remote source, add `--local /data/mirror.zarr` to reuse matching local
chunks. `--offline` forbids CT network access; model weights and the Lasagna
runtime must be provisioned separately.

Inspect `winding-field status runs/field` and `RUN.json.io` for actual reads and
cache use. Resume with `winding-field field runs/field` to stop at the field, or
`winding-field resume runs/field` to finish the selected pipeline. Preserve the
original source, configuration and run path.

## Shipped backend

The installed CLI uses `vc_reader-zarr-v2-checked-v1`: numeric 3D Zarr v2,
local files or public HTTP/S3. Native code decodes verified chunk snapshots,
assembles crops and interpolates CT; CLI checks retain NGFF geometry, consumed-byte
hashes and resource bounds. It does not load the entire CT into memory.
Supported inputs use native compressors and numeric Delta/Shuffle filters with
C/F storage order. Other codecs and Zarr v3/sharded layouts fail explicitly;
use the original v1.5.0 runtime for unsupported inputs and older checkpoints.

| Setting | Controls | Does not control |
| --- | --- | --- |
| `--ct-mode stream` | Fetch chunks when needed | Model or field memory |
| `--ct-mode download` | Prefetch the selected input domain | Whole-scroll feasibility |
| `--cache-gib` | Retained compressed CT payloads on disk | Total RAM or output size |
| `--reserve-gib` | Free disk headroom | A reservation against other jobs |
| `--chunk` / `--field-chunk` | Observation tiles / solver blocks | CT's stored chunk layout |

Lookup order is matching local CT, bounded compressed-payload cache, then the
original remote source. Missing local chunks are fetched remotely. Only an
authoritative remote 404 establishes sparse Zarr fill; local-only missing chunks
fail. Source metadata must match the local mirror. Remote content is revalidated
when accessed in a new session; offline mode can only use already available data.
Online local mirrors are verified against authoritative remote content as well.
Single-part S3 ETags permit conditional validation without a payload transfer;
other origins or ETags may require a verification download.

`--ct-mode auto` compares the input domain including preparation halo, cache cap,
estimated output size and available disk space. If it fits, it prefetches that
domain; otherwise it streams. Estimates use uncompressed CT chunk sizes and a
coarse product allowance, not exact compressed transfer or final mesh size.
Resume subtracts persisted products from that allowance and does not charge a
completed run for building its products again. Individual numerical/mesh writes
also enforce their own reserve and transient-space checks.
Cache index/metadata, in-flight reads and decode RAM are separate from the payload
cap. `--chunk`, `--field-chunk` and `--chart-faces` bound different operations;
there is no universal hard RSS/VRAM cap.

`--lasagna-input` reads existing local prediction channels through the same bounded
Zarr v2 reader, with per-channel consumed-chunk ledgers. Prediction files must be
present locally; CT streaming remains independent. The plan adds normalized
imported-tile storage and removes native Lasagna staging/inference workspace.
With external AFV too, preparation reads only the CT observation ROI, without an
inference halo. See [custom inputs](LASAGNA.md#external-lasagna-predictions).

`RUN.json.io` reports downloaded bytes, payload downloads, cache hits, local hits
and cached payload bytes. Include cache state when comparing timings.

### Bounded Qual CPU execution

`--bounded-preparation` trades retained arrays for repeated model windows. It
requires Qual V2, stride 1, explicit `--field-spacing-um`, no imported inputs,
and ROI origins divisible by 4. Its field grid includes the ROI endpoints, so
actual spacing can differ slightly from the requested maximum. Inspect the plan.
This bounds retained/scratch work, not execution time: CPU inference can be slow.

Bounded Qual tracing and coverage replay use the pinned Numba kernels with
globally addressed tile readers. Cached scalar reads stay compiled; a tile miss
calls the existing Python loader. Each field keeps one bounded LRU cache shared
by Python and compiled access. Crossing a tile boundary does not restart or clip
a fibre, and seeds still commit sequentially before their coverage becomes visible.
Paged spline fitting is compiled serially because the tile caches are mutable.

The numerical reference for new bounded Qual runs is the pinned original
compiled Numba implementation (accepted November 5, 2026), not the former
Python rebound kernels. NumPy/Python scalar promotion can round intermediate
values to float32 where the original Numba kernels use float64. On four saved
real seeds, paged tracing matched original dense Numba bit for bit; legacy
Python differed by up to 4.96e-5 in coordinates. The former 1e-5 comparison
against Python is therefore not a compatibility guarantee or an acceptance
gate for new runs; it remains a diagnostic, not a relaxed tolerance.
Regression comparisons use the original compiled kernels. This scoped parity
does not establish whole-volume equivalence or physical-sheet correctness.

The first use of each compiled signature includes JIT startup; measure warm
kernel time separately. This accelerates CPU kernels, not model inference or CT
transport, and does not establish whole-volume feasibility. Existing runs retain
their implementation fingerprint: use their frozen package to resume, or a new
output directory for the new implementation.

The pinned native reader is required by normal installation (a compatible C++
build environment is needed). The `ct` extra remains an empty installation alias.
Bounded native object reads also support zero/tiny disk caches; an object larger
than the cache budget is consumed without retaining its payload. Missing native
APIs fail explicitly. `RUN.json.io` identifies the reader and transport.
If the native reader was built with CUDA, its shared CUDA runtime must match
PyTorch's: place the matching runtime library directory first in
`LD_LIBRARY_PATH` before starting the CLI. Verify importing both the native
extension and PyTorch in that environment before starting a run.

Native payloads live in the CLI-owned `vc-reader/` cache subdirectory. Legacy
payloads count against the same disk budget and are reclaimed as needed.
Portable-cache payloads are not imported: populate the new native cache online
before working offline, or use v1.5.0 to read the old portable cache.
Per-object locks coordinate payloads and receipts; native publication owns payload
eviction. Do not point unrelated writers at this private subdirectory. The adapter
relies on the pinned reader's SQLite cache schema.

Encoded-byte SHA256 receipts and the run's source ledger remain authoritative.
Fresh validation starts a new remote epoch; only strong ETags can avoid payload
downloads. Last-Modified and weak ETags are insufficient for byte identity.
Only authoritative HTTP 404 establishes sparse fill; its marker persists offline,
where an unknown or evicted block remains an error. Native cache corruption is
rejected. Object reads and chunk decoding are bounded at 256 MiB, including
unsized and HTTP-content-compressed responses; assembled crops are capped at
512 MiB. Disk publication retains the configured free-space reserve. Read errors
remain explicit errors, never sparse fill. The adapter accesses the native index
through the reader's SQLite library, avoiding conflicting WAL locks when Python
bundles another SQLite version.
Native speedup and end-to-end throughput must be measured for the actual workload.
