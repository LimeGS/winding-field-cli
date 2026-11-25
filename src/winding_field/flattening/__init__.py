"""Flattening: every UV parameterization path of the CLI.

Modules:

- ``flatboi`` / ``flatboi_checkpoint``: Flatboi SLIM runtime, batched solves and
  durable checkpointed batches (the only SLIM backend).
- ``lscm``: the shared libigl LSCM primitive (callers keep their own pins).
- ``native_charts`` (was ``native_geometry``): native-run local charts, seam welding
  and assembly; ``native_sheet`` (was ``native_global``): cut-to-disk, mean-value and
  ARAP whole-sheet maps.
- ``matrix`` / ``matrix_pipeline`` (were ``flattening_matrix`` / ``matrix_pipeline``):
  ``export-sheets --flattening-solver {lscm,slim} --scope {patches,sheet}``.
- ``patches``, ``patch_mapping``, ``patch_continue``, ``patch_visuals`` (were
  ``patch_pipeline`` …): historical Flatboi patches.
- ``original`` / ``original_pipeline`` / ``original_tifxyz`` (were ``original_recipe`` …):
  the original global LSCM recipe.
- ``sheet_prepare``, ``sheet_finish``, ``sheet_atlas*``, ``constant_sheet``,
  ``run_sheet_export``: whole-sheet preparation, publication and atlases.
- ``legacy_charts`` (was ``flatten``): legacy linked-chart LSCM for ``--surface-output atlas``.
"""

from pathlib import Path


# Receipts and benchmark identities key implementation hashes by the module names used
# before this package existed; the keys stay stable and resolve to the module that now
# holds that code.
KERNEL_MODULES = {
    "constant_sheet": "flattening.constant_sheet",
    "flatboi_checkpoint": "flattening.flatboi_checkpoint",
    "flatten": "flattening.legacy_charts",
    "flattening_matrix": "flattening.matrix",
    "matrix_pipeline": "flattening.matrix_pipeline",
    "native_geometry": "flattening.native_charts",
    "native_global": "flattening.native_sheet",
    "original_pipeline": "flattening.original_pipeline",
    "original_recipe": "flattening.original",
    "original_tifxyz": "flattening.original_tifxyz",
    "patch_continue": "flattening.patch_continue",
    "patch_mapping": "flattening.patch_mapping",
    "patch_pipeline": "flattening.patches",
    "patch_visuals": "flattening.patch_visuals",
    "run_sheet_export": "flattening.run_sheet_export",
    "sheet_atlas": "flattening.sheet_atlas",
    "sheet_atlas_verify": "flattening.sheet_atlas_verify",
    "sheet_atlas_view": "flattening.sheet_atlas_view",
    "sheet_finish": "flattening.sheet_finish",
    "sheet_prepare": "flattening.sheet_prepare",
}


def module_file(name):
    """Source file of ``winding_field.<name>``, or of the module a receipt key names."""
    from importlib import import_module

    return Path(import_module(f"winding_field.{KERNEL_MODULES.get(name, name)}").__file__)


def package_sources():
    """Top-level and flattening sources keyed by package-relative path (code identities)."""
    root = Path(__file__).resolve().parent.parent
    return sorted(
        [*root.glob("*.py"), *(root / "flattening").glob("*.py")],
        key=lambda path: path.relative_to(root).as_posix(),
    )
