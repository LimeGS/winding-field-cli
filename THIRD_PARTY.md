# Attribution and source provenance

## Winding-field numerical core

Selected files from `vesubius-challenge/release/winding-field-gh/src` are bundled
under `winding_field._core`. The source checkout included current uncommitted
block/mesh modules; exact source-file SHA-256 values, rather than a misleading
clean-commit claim, are recorded in `provenance/core.json`. Only module imports
were namespaced. The solver's mathematical behavior is retained.

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
