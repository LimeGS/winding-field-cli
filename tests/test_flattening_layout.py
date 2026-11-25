"""The flattening package layout; the pre-2.0 top-level module names are gone."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest

ALIASES = {
    "flatten": "legacy_charts",
    "native_geometry": "native_charts",
    "native_global": "native_sheet",
    "original_recipe": "original",
    "original_pipeline": "original_pipeline",
    "original_tifxyz": "original_tifxyz",
    "patch_pipeline": "patches",
    "patch_mapping": "patch_mapping",
    "patch_continue": "patch_continue",
    "patch_visuals": "patch_visuals",
    "flattening_matrix": "matrix",
    "matrix_pipeline": "matrix_pipeline",
    "flatboi_checkpoint": "flatboi_checkpoint",
    "constant_sheet": "constant_sheet",
    "sheet_prepare": "sheet_prepare",
    "sheet_finish": "sheet_finish",
    "sheet_atlas": "sheet_atlas",
    "sheet_atlas_verify": "sheet_atlas_verify",
    "sheet_atlas_view": "sheet_atlas_view",
    "run_sheet_export": "run_sheet_export",
}


@pytest.mark.parametrize("old,new", sorted(ALIASES.items()))
def test_old_module_names_are_gone(old, new):
    moved = importlib.import_module(f"winding_field.flattening.{new}")
    assert Path(moved.__file__).parent.name == "flattening"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"winding_field.{old}")


def test_flattening_module_entrypoint_runs():
    result = subprocess.run(
        [sys.executable, "-m", "winding_field.flattening.patch_continue", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_every_lscm_solve_uses_the_shared_primitive():
    root = Path(importlib.import_module("winding_field").__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in [*root.glob("*.py"), *(root / "flattening").glob("*.py")]
        if path.name != "lscm.py" and "igl.lscm(" in path.read_text()
    ]
    assert offenders == []


def test_kernel_hashes_resolve_moved_sources():
    from winding_field.flattening import module_file

    assert module_file("patch_pipeline").name == "patches.py"
    assert module_file("native_ct_sampling").name == "native_ct_sampling.py"
    assert all(path.is_file() for path in map(module_file, ("patch_mapping", "native_patches")))
