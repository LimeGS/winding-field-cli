"""Flatboi exports run without --app-run: the pinned runtime is resolved at run time."""

import json


from test_flattening_matrix import request, source_bundle
from test_native_slim import flatboi_app  # noqa: F401  (fixture)
from winding_field.util import atomic_json


def test_patch_export_resolves_pinned_flatboi_without_app_run(tmp_path, flatboi_app, monkeypatch):  # noqa: F811
    from winding_field.cli import main

    monkeypatch.setenv("WINDING_FLATBOI_APP_RUN", flatboi_app)
    mesh, ct = source_bundle(tmp_path)
    req = request(tmp_path)
    assert "app_run" not in req
    config = tmp_path / "config.json"
    atomic_json(config, req)
    out = tmp_path / "out"
    args = [
        "export-sheets",
        "--mesh-bundle",
        str(mesh),
        "--source",
        str(ct),
        "--voxel-um",
        "9.6",
        "--flattening-method",
        "patches",
        "--flattening-config",
        str(config),
        "--output",
        str(out),
    ]
    assert main(args) == 0
    receipt = json.loads((out / "PATCHES.json").read_text())
    assert receipt["status"].startswith("PATCHES") or receipt.get("levels")


def test_matrix_slim_does_not_index_a_missing_app_run(monkeypatch):
    """The CLI omits app_run; every Flatboi entry must read it with .get()."""
    import inspect

    from winding_field.flattening import matrix, matrix_pipeline, patches

    for module in (matrix, matrix_pipeline, patches):
        assert 'cfg["app_run"]' not in inspect.getsource(module).replace(
            'isinstance(cfg["app_run"], str) or not cfg["app_run"]', ""
        ), module.__name__
