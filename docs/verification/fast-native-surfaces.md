# Native surfaces: implementation and measured limits

Status: native implementation and review fixes are tested; an experimental global
mesh-to-render command now has a fresh coherent W−50 acceptance result in7m26s.
**The30minute whole-field W−4 performance goal has not been met.** Mesh-to-render
timing does not substitute for that goal.

## Architecture

`native_mesh.py` owns batched indexed storage and3D stitching; it calls the
unchanged vendored numerical tile extractor. `native_geometry.py` owns UV charts
and assembly. `native_render.py` owns shared XYZ/CT rasterization and optional
mesh-derived TIFXYZ. `native_pipeline.py` supervises per-level work; `cli.py`
dispatches only. `_core/` and `provenance/core.json` remain unchanged.

`native_global.py` and `tools/bench_global_native.py` provide the explicit
experimental global route. See [command and fresh acceptance](native-global-export.md).

Tests compare byte-identical OBJ geometry and component/edge counts against the
original mesh implementation, including multi-level/tile input, duplicates,
nonmanifold edges, vertex-only contacts, interruption and corrupted checkpoints.

## Measured real runs

All are on gpu-1, using bound historical175A input and original CT L0. Frozen
runtimes/outputs remain separate under `/ssd/draft/fast-native-surfaces-20261101-0N/`.

| Case | Result | Measurement |
| --- | --- | --- |
| Runtime01, tile7084 |3208native faces,100%mapped, PNG/TIFXYZ |4.538s mesh;10.863s render;80.749s including source startup |
| Runtime01, dense tile7423 |76021faces extracted; time stop during UV |120.24s; checkpoint retained |
| Runtime03, same dense tile |76021faces,64.3747%mapped; PNG; TIFXYZ declined folded cell |9.768s mesh,35.459s rendering,120.698s including source startup |
| Runtime03, full W-4,90%target |**PAUSED/TIME_LIMIT during mesh** |1802.14s;196of2527volume tiles committed; peakRSS1.356GiB |
| Runtime06, full W-4 with bound pruning/affine reconstruction |**PAUSED/TIME_LIMIT during mesh** |1802.18s;445of2527tiles; peakRSS928698368bytes |
| Runtime07, saved W-50, local charts |81.441%area mapped; mainTIF reader-valid but fragmented |largest retained grid patch8.69%; coherence goal not met |
| Fresh global W-50 command, existing indexed mesh |**PARTIAL, target met, coherent grid verified** |445.725s;95.8598%whole-mesh area qualified;99.2122%of retained cells in one patch;~987MiBRSS |

The whole-W case uses the entire declared field `[1594,4682,4682]`, not the entire
CT scroll height. It retained more than2GiB available RAM and the12.25GiB disk
reserve. Tile progress is not face/area progress and must not be extrapolated as
a linear ETA. Field traversal/reconstruction is still a throughput blocker.

The sampled CT ranges in the two real small renders were **0..0**. Some requests
were authoritative Zarr fill; other original chunks contain nonzero data, but the
sampled isosurface points remain dark. These runs establish geometry/format and
resource behavior, not useful papyrus texture or physical surface alignment.

## Cancellation and resume evidence

- Actual installed CLI SIGINT and STOP-file tests preserve committed chart bytes
  and mtimes, then resume to published products.
- An intentionally unresponsive worker ignores SIGTERM; the supervisor kills
  only its process group, and the committed SQLite/payload checkpoint reopens.
- Cumulative time is retained; explicit extra allowance does not relabel the
  original speed goal as met.
- Image receipts precede optional TIFF work; partial channels stay under
  `.tifxyz.pending/` until atomic promotion.
- Final review added regressions for interruption between `LEVEL.json` rename and
  ledger registration, exact-once validation timing, native-only CT source changes,
  and small/off-center/grid-aligned source holes inside otherwise valid TIFXYZ cells.

## Verification record so far

- Local full run before final fixes:602passed,18skipped,1resource failure in the
  existing large-SQLite test. Mac free space was1.7GiB, below its2GiB journal
  admission. Both parametrizations passed on the server without weakening limits.
- Runtime01 installed-wheel native suite:42passed.
- Runtime03 Linux mesh/geometry/render subset:31passed, including actual libigl.
- Remote full attempt stopped after296passes: original public-direct integration
  blocked by RAM admission (available~4.3GiB, required5GiB), then the overall900s
  timeout interrupted a direct-partial CLI case. That CLI case passed isolated.
- Final review identified4Important findings; each is reproduced and being
  verified RED→GREEN. Whole-change review was independent and read-only.

The entries above describe historical checkpoints. Latest applicable local run:
756passed32skipped1deselected in310.90s; the existing large-SQLite positive was
excluded for local disk admission and passed on Linux. Frozen Linux global math,
controller, renderer and both SQLite cases:122passed2opt-in skips in188.66s.
The global command's three Important review findings (source capture races,
unbound completed-resume summary, unsupervised verification) were reproduced and
fixed with regressions. Actual fresh W−50 VC3D and readonly-resume acceptance are
recorded in [native-global-export.md](native-global-export.md).
