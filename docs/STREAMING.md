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
and cached payload bytes. Include cache state when comparing timings. The CLI
does not integrate Villa's native `vc_reader`; these behaviors describe its
portable Python reader.
