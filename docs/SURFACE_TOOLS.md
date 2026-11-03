# Surface and visualization tools

The CLI owns mesh extraction, surface export, flattening, atlases and pictures.
The numerical library owns field construction and numerical benchmarks.

## Migrating from winding-field

| Former core path/import | CLI command or import |
| --- | --- |
| `tools/surface_export.py` | `winding-field surface-export` |
| `tools/surface_probe.py` | `winding-field surface-probe` |
| `tools/surface_atlas.py` | `winding-field surface-atlas` |
| `tools/surface_geometry_check.py` | `winding-field surface-geometry-check` |
| `tools/run_watch.py` | `winding-field run-watch` |
| `benchmarks/winding/ct_visuals.py` | `winding-field ct-visuals` / `winding_field.ct_visuals` |
| `src/patch_check.py` command | `winding-field patch-visuals` |
| `surface_mesh` | `winding_field.surface_mesh` |
| `surface_sources` | `winding_field.surface_sources` |

Each command also works as `python -m winding_field.<module_name>`; use underscores
instead of hyphens. Existing export/probe/atlas flags and artifact formats are
preserved. `patch_check.evaluate_patch`, `sample_field` and `load_patch` remain
numerical-library APIs. The old flat surface imports and core `surface`/`plot`
extras are removed in `winding-field-experiments==1.3.0.dev1`.

Implementation-bound resumes made with the old packages require those packages;
otherwise use a new output directory. This migration does not rewrite old receipts.

## Export and atlas

```bash
winding-field surface-export --field phi.npy --metadata phi.npy.json \
  --level 1.25 --seed-xyz 10 20 30 --pitch .25 --output new-surface \
  --ct ct.npy --ct-metadata ct.json --depth-offsets -2 -1 0 1 2
winding-field surface-probe --help
winding-field surface-atlas --help
winding-field surface-geometry-check --help
```

`surface-export` uses PCA unless `--app-run PATH` selects an external Flatboi
launcher. The launcher and geometry-gate programs are external inputs, not bundled
binaries. Existing UV, distortion, source-topology and raster gates still apply.

`surface-probe --config ...` reads declared CT/prior grids and freezes its seed
before rendering. It accepts an optional human `--patch` for evaluation.
`context.surface` may specify `seed_margin`, `ct_padding_samples`, `pitch_world`,
`depth_offsets_world`, `mesh_step` and `ct_window`. CT may be uint8, uint16 or
another finite numeric volume; its source spacing is applied explicitly.

`surface-atlas` consumes the probe's `inputs.json`; without `--patch` it reuses
that frozen prior seed. Supplying `--patch` selects human-seeded development mode.
Raster pitch and offsets come from the frozen inputs. Missing human references
are not evaluated; missing tile ownership has no seam claim.

Human patch frame and physical units must agree with the field. For legacy TIFXYZ
metadata missing either declaration, supply `--patch-frame` and/or
`--patch-voxel-um` to the probe/atlas. Conflicting declarations are rejected and
the original patch is not rewritten. TIFXYZ reserves negative coordinates as
invalid sentinels; its exporter requires nonnegative valid XYZ. Other field and
benchmark formats can use translated world frames.

## Frozen CT overlays

Install the optional plotting dependency after the numerical library:

```bash
python -m pip install '.[plot]'
winding-field ct-visuals --case case.json --field field.npy \
  --metadata field.npy.json --output new-views
```

Numerical scoring still uses the core's `benchmarks/winding/evaluate.py` and
produces scores without image dependencies. It ignores the optional
`visualization` block. Run `ct-visuals` separately with that same case, candidate
and metadata, using a **new output directory**. CT visualization does not score
or validate numerical benchmark populations, AFVs or separation controls.

The case declares `frame`, positive `step`, `voxel_um`, and:

```json
{
  "visualization": {
    "schema": "ct-winding-overlays-v2",
    "archive_manifest": {
      "path": "ct-archive/MANIFEST.json",
      "sha256": "<SHA-256 of the frozen archive manifest>"
    },
    "reference_key": "coarse",
    "reference_label": "Reference field"
  }
}
```

Paths resolve relative to the case. The candidate sidecar provides its file hash,
frame, `origin_xyz`, positive `spacing_xyz`, `shape_zyx` and calibration `step`.
For masked fields, pass `--support support.npy` and declare at least one expected
hash. Metadata `support_sha256` and `--support-sha256` hash the complete NPY file;
metadata `support_array_sha256` hashes `np.ascontiguousarray(support).tobytes()`.
Each supplied hash must match its respective bytes. A declared mask cannot be omitted.

For historical cases without `voxel_um`, pass `--config dataset.json` with the
matching explicit frame and units. That configuration is hashed into preflight;
the original case is not rewritten. Empty frames, declared nonconverged solvers
and failed pilot gates are rejected. Calibration comparison retains the numerical
evaluator's `rtol=1e-12`, `atol=0` tolerance.

The frozen archive contains a hash-bound `RECEIPT.json` and ordered
`XY-samples.npz`, `XZ-samples.npz`, `YZ-samples.npz`. Each NPZ supplies finite
numeric `ct`. A selected reference additionally needs matching values and a
`<key>_valid` boolean mask. Omit `reference_key` for CT + candidate columns only.

The receipt declares matching `frame`, `step`, `voxel_um`, world `origin_xyz`,
integer pixel counts `size_xyz`, positive world `spacing_xyz`, `ct_window`,
consecutive integer `contour_levels`, `field_sha256`, `input_ct` and `ct_url`.
Each plane declares `name`, `fixed_axis_xyz`, `fixed_position_world` and
`ct_pixels`. Positions identify actual slices. Physical pixel sizes and plot
axes follow the declared XYZ spacing. CT is read locally from the frozen archive.

Legacy `ct-winding-overlays-v1` archives with `fixed_position_l2` remain readable.
Without explicit spacing, their mip/native-scale relationship must imply unit
world sampling; nonunit sampling requires `spacing_xyz`.

Outputs:

- `ct-winding.png` and `ct-winding-{XY,XZ,YZ}.png`: matched overview and views.
- `ct-winding-samples.npz`: original CT, actual candidate values, valid masks and
  optional frozen reference samples.
- `VISUALS.json`: coordinates, frozen display settings, provenance and coverage.
- `PREFLIGHT.json`, `SUMMARY.json`, `MANIFEST.json`: input/source hashes, visual
  completion and output identities. `ERROR.json` records a failed attempt.

Exit 0 means `PASS_VISUALIZED` or `PASS_PREFLIGHT`; exit 2 means an error.
`--check-only` validates inputs/dependencies without images. Existing output
directories are rejected. Inputs are rehashed before successful completion.

Views preserve inherited gauge and fixed `phi / step` levels. Unsupported samples
stay missing; fully unsupported or out-of-range views are labeled. Visual sampling
uses physical support without the numerical table's outer-node margin. These are
QC pictures, not physical-sheet certificates. Synthetic archives declare
`synthetic: true` so their pictures are visibly labeled.

Offline example, using the installed package:

```bash
python -I examples/ct_visuals.py --output runs/synthetic-views
```
