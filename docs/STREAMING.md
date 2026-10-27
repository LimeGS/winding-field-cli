# Original-CT streaming and native reader research

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

`RUN.json.io` reports downloaded bytes, payload downloads, cache hits, local hits and cached
payload bytes. Report cache state with timings: the initial real-CT demonstration
used a warm cache and is not a cold S3 throughput benchmark.

## vc_reader

The newer Villa reader and its source-reading contract were researched:

- [Upstream source-reading documentation](https://github.com/ScrollPrize/villa/blob/main/vc_reader/docs/source-reading.md)
- [Discord announcement](https://discord.com/channels/1079907749569237093/1561853714845671446/1561853714845671446)
- [Cache/locality discussion](https://discord.com/channels/1079907749569237093/1534253403591676014/1534770125708984440)

The reported native speedups concern local SSD workloads; they do not imply an
equivalent increase in S3/network bandwidth. A native build was attempted on the
development Mac but failed at Apple's libc++ `std::atomic<std::shared_ptr<...>>`
specialization. Consequently this release has **no integrated or verified native
vc_reader backend**. The portable reader is the exercised path, including the
original PHerc0175A S3 demonstration. No native performance benefit is claimed.
