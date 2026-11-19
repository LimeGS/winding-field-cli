"""The flattening package layout and its compatibility aliases."""

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
def test_old_module_names_are_the_moved_module(old, new, monkeypatch):
    moved = importlib.import_module(f"winding_field.flattening.{new}")
    alias = importlib.import_module(f"winding_field.{old}")
    assert alias is moved
    assert Path(moved.__file__).parent.name == "flattening"
    # Monkeypatching through the old name patches the module the code actually uses.
    monkeypatch.setattr(alias, "_layout_probe", object(), raising=False)
    assert moved._layout_probe is alias._layout_probe


def test_old_module_entrypoint_still_runs():
    result = subprocess.run(
        [sys.executable, "-m", "winding_field.patch_continue", "--help"],
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
