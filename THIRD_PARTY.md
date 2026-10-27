# Attribution and source provenance

## Winding-field numerical core

The original [winding-field](https://github.com/LimeGS/winding-field) library is
installed as `winding-field-experiments==1.3.0.dev0`. Numerical modules are imported
directly from that distribution; no numerical core is bundled here. The original
revision and numerical source hashes are recorded in `provenance/core.json`.

Prepared workflows (`field_workflow.py`, `surface_workflow.py`), CPU geometry
rendering (`surface_render.py`), their tests and supporting recipes/examples/tools
were moved from the original repository to this CLI. Their source revision and
hashes are also recorded there. Adaptations are package imports, prepared command
routing and module-based supervised launch; solver mathematics are unchanged.

License: MIT, copyright (c) 2026. Full notice: `licenses/winding-field-MIT.txt`.

## Villa neural network modules and interchange formats

Source: <https://github.com/ScrollPrize/villa>, revision
`b29f310957c7b29e7d332f9876e0e487819cfaa3`.

Five required SD2 model/direction/utility files are bundled under
`winding_field._vendor.vesuvius`. Namespace imports and empty package initializers
are the only adaptations. Exact hashes: `provenance/villa.json`.
License: MIT, copyright (c) 2024 Vesuvius Challenge, `licenses/villa-MIT.txt`.

Lasagna v2 and AFV v1 interchange schemas follow this same pinned Villa source.
Our exporter records its ROI/source coordinate frames explicitly.

## SD2 weights

Downloaded separately from `scrollprize/lasagna-fiber`, Hugging Face revision
`a9aaf89cc504206f4dda0f0398a4f3e263fee25f`, file
`s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt`.
SHA-256: `2528a46e38844c06e591084f1dc1af9442ccb45a17aae9b2a68441e6028bbdd6`.

Weights are not included or relicensed by this repository. Their upstream terms
and those of each original CT dataset apply independently. The PHerc0175A catalog
records its CT as CC BY-NC 4.0. This repository's MIT license does not replace that.

## Qual AFV fibre V2

Model: <https://huggingface.co/Qualzz20/afv_fiber_9um_v2>, revision
`2a8a26a70754a5d90aeca91f9ef8e7130e6e4f81`. The upstream `model.py` and
`plans.json` are bundled unchanged under `winding_field._vendor`; exact hashes
are in `provenance/qual.json`. The adapter uses the upstream normalization and
architecture (`dynamic-network-architectures==0.4.4`).
On CPUs without oneDNN, the application adapter evaluates convolutions in depth
slabs to bound temporary allocations; upstream source files remain unchanged.

The separately downloaded checkpoint SHA-256 is
`b8519a4ce4044c93ed98af0c278628fdcdfdd8808275b3c8becd7211f38b7967`.
Model code and weights are published under Apache-2.0; see
`licenses/qual-Apache-2.0.txt`. Weights are not bundled.

Spline generator: <https://github.com/Qualzz/AVF_SPLINE_GENERATOR>, pinned to
`3c8b73628ca56cb28042b9b2e79439b2e1c693d2`, Apache-2.0. Its V2 model outputs
use the AFV schema v1 container; V2 does not designate a different SQLite schema.
