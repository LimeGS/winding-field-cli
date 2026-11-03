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

## Direct TIFXYZ coarse-guide helpers

`winding_field.tifxyz_guide` adapts the bounded guide validation, conservative
cuts, harmonic parameterization and SLIM relaxation helpers from
`vesubius-challenge/tools/mesh_to_tifxyz/geometry.py` (MIT). Exact source SHA-256,
adapted function names and changes are recorded in `provenance/tifxyz-guide.json`,
also packaged under `share/winding-field-cli/provenance/`.

Source SHA-256: `26b17a649860cdc45f63cb6abd95068d2638fe3abd68921e375ca9a9b54b2d3d`.
The adaptation uses only a bounded coarse guide, preserves seam normals and adds
interruptible SLIM iterations; it does not include largest-component selection,
OBJ reading or native-mesh raster conversion. The shared project's MIT notice is
in `licenses/winding-field-MIT.txt`.

The optional `tifxyz` extra uses separately installed libigl 2.6.3 and tifffile
(>=2023.7); these distributions are not vendored or installed by runtime commands.

## Native shared-edge assembly

`winding_field.native_geometry.weld` adapts the bounded cotangent-displacement
seam welding from this project's Paris4 experiment:
`vesubius-challenge/experiments/winding-field-exp/six-surfaces-opt-20261016/global_sheet.py`.
The native implementation keeps a disk-backed frontier and explicit cycle cuts;
it does not copy Paris4-specific checkpoints, units or component selection.
Streaming raster/CT rendering follows the same barycentric-coordinate approach
as `global_sheet_render.py`, with the CLI's existing source/frame/cache reader.
The shared project's MIT notice is in `licenses/winding-field-MIT.txt`.

## Original global CT recipe

`original_recipe.py` and `original_tifxyz.py` port this project's frozen
`whole-code-05` global LSCM, robust barycentric raster, positive-weight CT sampling,
and November 3 mesh-derived TIFXYZ adapter. Provenance and the frozen entry-point
SHA256 are in `provenance/original-recipe.json`, included in installed distributions.
The project MIT notice is in `licenses/winding-field-MIT.txt`. libigl is installed
separately through the `tifxyz` extra. The port contains no campaign SSH/runtime
dependencies or scroll-specific geometry constants.

## SD2 weights

Downloaded separately from `scrollprize/lasagna-fiber`, Hugging Face revision
`a9aaf89cc504206f4dda0f0398a4f3e263fee25f`, file
`s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt`.
SHA-256: `2528a46e38844c06e591084f1dc1af9442ccb45a17aae9b2a68441e6028bbdd6`.

Weights are not included or relicensed by this repository. Their upstream terms
and those of each original CT dataset apply independently. The PHerc0175A catalog
records its CT as CC BY-NC 4.0. This repository's MIT license does not replace that.
