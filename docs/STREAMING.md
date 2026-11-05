# CT input and streaming

## Choose access and inspect usage

```bash
winding-field plan --source /data/scroll.zarr --voxel-um 8.64 \
  --ct-mode stream --cache-gib 2 --reserve-gib 1 --until field --output runs/field
```

Replace the source and scale with your scan, add `--roi` for a bounded region,
then use `run` with the same arguments. `plan` does not create the output.
For a remote source, add `--local /data/mirror.zarr` to reuse matching local
chunks. `--offline` forbids CT network access; model weights and the Lasagna
runtime must be provisioned separately.

Inspect `winding-field status runs/field` and `RUN.json.io` for actual reads and
cache use. Resume with `winding-field field runs/field` to stop at the field, or
`winding-field resume runs/field` to finish the selected pipeline. Preserve the
original source, configuration and run path.

## Shipped backend

The installed CLI uses `portable-zarr-v2`: numeric 3D Zarr v2, local files or
public HTTP/S3, with numcodecs decompression. It reads intersecting chunks,
respects NGFF geometry and records hashes of consumed content. It does not load
the entire CT into memory. Zarr v3/sharded layouts are not currently supported.

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

`RUN.json.io` reports downloaded bytes, payload downloads, cache hits, local hits
and cached payload bytes. Include cache state when comparing timings.

Install the pinned native transport with `pip install '.[ct]'` (requires a
compatible C++ build environment). For remote volumes with a cache budget of
at least 256 MiB, an available native extension enables
`vc_reader.reader.encoded_files.EncodedFiles`. `RUN.json.io.transport` identifies
the selected transport; `backend` still identifies the NumPy/numcodecs Zarr-v2
decoder. Native decoded-array assembly is not enabled by this integration.
Without the optional extension, or with smaller streaming budgets, the portable
transport remains available. A broken installed native runtime fails explicitly.
If the native reader was built with CUDA, its shared CUDA runtime must match
PyTorch's: place the matching runtime library directory first in
`LD_LIBRARY_PATH` before starting the CLI. Verify importing both the native
extension and PyTorch in that environment before starting a run.

Native payloads live in the CLI-owned `vc-reader/` cache subdirectory. Existing
legacy payloads are reused and revalidated in place; their bytes reduce the native
budget, so these are not two independent cache allowances. The parent cache lock
coordinates CLI writers. Do not point unrelated native writers at this private
subdirectory. The adapter relies on the pinned reader's SQLite cache schema.

Encoded-byte SHA256 receipts and the run's source ledger remain authoritative.
Fresh validation starts a new remote epoch; only strong ETags can avoid payload
downloads. Last-Modified and weak ETags are insufficient for byte identity.
Only authoritative HTTP 404 establishes sparse fill; its marker persists offline,
where an unknown or evicted block remains an error. Native cache corruption is
rejected. The disk reserve accounts for the old inode pinned during replacement.
The native admission index refuses any single transfer reservation above 256 MiB
before payload streaming. Unsized or HTTP-content-compressed responses reserve
the entire native allowance and are therefore refused when that allowance exceeds
256 MiB; ordinary S3 chunk responses have a declared encoded length. These errors
remain explicit errors, never sparse fill. The adapter accesses the native index
through the reader's SQLite library, avoiding conflicting WAL locks when Python
bundles another SQLite version.
Native speedup and end-to-end throughput must be measured for the actual workload.
