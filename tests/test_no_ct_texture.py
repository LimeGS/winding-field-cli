"""--no-ct-texture: flatten and write mesh-derived TIFXYZ without sampling CT."""

import json
import numpy as np
import pytest

from test_native_geometry import plane_bundle
from test_native_render import analytic_ct


def _xyz(folder):
    import tifffile

    return np.stack(
        [tifffile.imread(folder / "tifxyz" / n) for n in ("x.tif", "y.tif", "z.tif")], axis=-1
    )


def test_untextured_render_writes_identical_tifxyz_without_reading_ct(tmp_path, monkeypatch):
    from winding_field.surface_mesh import MeshReader
    from winding_field import native_render
    from winding_field.native_pipeline import render_mesh

    config = {"chart_faces": 8, "render_pitch_um": 0.5, "export_tifxyz": True}
    bundle = plane_bundle(tmp_path)
    with MeshReader(bundle) as reader, analytic_ct(tmp_path) as volume:
        textured = render_mesh(reader, tmp_path / "textured", volume, config)
    assert textured["status"] == "COMPLETE"

    def refuse(*args, **kwargs):
        raise AssertionError("CT sampled with ct_texture=False")

    monkeypatch.setattr(native_render, "sample_ct", refuse)
    out = tmp_path / "bare"
    bare_config = dict(config, ct_texture=False)
    with MeshReader(bundle) as reader, analytic_ct(tmp_path / "second") as volume:
        result = render_mesh(reader, out, volume, bare_config)
        assert render_mesh(reader, out, volume, bare_config) == result
    record = result["components"][0]
    assert result["status"] == "COMPLETE"
    assert result["requested_tifxyz_complete"] and result["coverage_fraction"] == 1.0
    assert result["usable_render"] is False
    assert result["usable_render_reason"] == "ct-texture-not-requested"
    assert record["counts"]["observed_ct"] == 0 and record["counts"]["unsupported_ct"] > 20
    assert record["ct_value_range"] is None
    assert record["tifxyz"]["status"] == "WRITTEN"
    base = textured["components"][0]
    assert record["counts"]["background"] == base["counts"]["background"]
    np.testing.assert_array_equal(
        _xyz(out / record["path"]).view(np.uint32),
        _xyz(tmp_path / "textured" / base["path"]).view(np.uint32),
    )
    identity = json.loads((out / "IDENTITY.json").read_bytes())
    assert identity["settings"]["ct_texture"] is False
    old = json.loads((tmp_path / "textured" / "IDENTITY.json").read_bytes())
    assert "ct_texture" not in old["settings"]


def test_texture_setting_is_opt_in_and_needs_tifxyz():
    from winding_field.native_pipeline import _settings

    assert "ct_texture" not in _settings({})
    assert "ct_texture" not in _settings({"ct_texture": True})
    with pytest.raises(ValueError, match="requires export_tifxyz"):
        _settings({"ct_texture": False})
    with pytest.raises(ValueError, match="boolean"):
        _settings({"ct_texture": 0, "export_tifxyz": True})


def test_textured_and_untextured_render_checkpoints_do_not_mix(tmp_path):
    from test_native_render import ready_plane
    from winding_field.native_render import render

    with ready_plane(tmp_path) as store, analytic_ct(tmp_path) as volume:
        render(store, volume, 0.5, ct_texture=False)
        with pytest.raises(ValueError, match="render identity changed"):
            render(store, volume, 0.5)


@pytest.mark.parametrize("branch", ["atlas", "tifxyz"])
def test_config_rejects_untextured_nonnative_runs(branch):
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="no-ct-texture requires --surface-output native"):
        validate_config({"surface_output": branch, "ct_texture": False, "export_tifxyz": True})
    with pytest.raises(ValueError, match="requires --export-tifxyz"):
        validate_config({"surface_output": "native", "ct_texture": False})


def test_run_flag_implies_tifxyz_and_is_absent_by_default(tmp_path, monkeypatch, capsys):
    from winding_field.cli import main
    from winding_field.util import atomic_json
    import winding_field.workflow as workflow

    def capture(config, **kwargs):
        atomic_json(tmp_path / "captured.json", config)
        return {"status": "PAUSED"}

    monkeypatch.setattr(workflow, "run", capture)
    base = ["run", "--source", "unused", "--output", str(tmp_path / "out")]
    assert main(base) == 0, capsys.readouterr().err
    config = json.loads((tmp_path / "captured.json").read_bytes())
    assert "ct_texture" not in config and config["export_tifxyz"] is True
    # 2.0 removed the opt-out: TIFXYZ is always written.
    for removed in ("--no-export-tifxyz", "--export-tifxyz"):
        with pytest.raises(SystemExit):
            main([*base, removed])
    assert main([*base, "--no-ct-texture"]) == 0, capsys.readouterr().err
    config = json.loads((tmp_path / "captured.json").read_bytes())
    assert config["ct_texture"] is False and config["export_tifxyz"] is True


def test_render_ct_flag_skips_prefetch_and_sampling(tmp_path, monkeypatch):
    from winding_field import native_render, scroll
    import winding_field.workflow as workflow

    bundle = plane_bundle(tmp_path)
    volume = analytic_ct(tmp_path)
    monkeypatch.setattr(workflow, "open_volume", lambda config: volume.thing)
    monkeypatch.setattr(volume.thing, "remote", True, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("CT read with --no-ct-texture")

    monkeypatch.setattr(scroll, "prefetch_ct", refuse)
    monkeypatch.setattr(native_render, "sample_ct", refuse)
    args = scroll._parser().parse_args(
        [
            "render-ct",
            "--mesh",
            str(bundle),
            "--source",
            "unused",
            "--render-pitch-um",
            "0.5",
            "--chart-faces",
            "8",
            "--flattening-method",
            "default",
            "--no-ct-texture",
            "--output",
            str(tmp_path / "out"),
        ]  # fmt: skip
    )
    result = scroll.render_ct(args)
    assert result["status"] == "COMPLETE"
    assert result["components"][0]["tifxyz"]["status"] == "WRITTEN"
    identity = json.loads((tmp_path / "out" / "IDENTITY.json").read_text())
    assert identity["settings"]["ct_texture"] is False
    assert identity["settings"]["export_tifxyz"] is True


def test_render_ct_defaults_to_tifxyz_without_a_texture_key(tmp_path, monkeypatch):
    from winding_field import native_pipeline, scroll
    import winding_field.workflow as workflow

    bundle = plane_bundle(tmp_path)
    volume = analytic_ct(tmp_path)
    monkeypatch.setattr(workflow, "open_volume", lambda config: volume.thing)
    seen = {}

    def capture(reader, output, volume, config):
        seen.update(config)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(native_pipeline, "render_mesh", capture)
    args = scroll._parser().parse_args(
        [
            "render-ct",
            "--mesh",
            str(bundle),
            "--source",
            "unused",
            "--render-pitch-um",
            "0.5",
            "--prefetch-workers",
            "0",
            "--output",
            str(tmp_path / "out"),
        ]  # fmt: skip
    )
    scroll.render_ct(args)
    assert "ct_texture" not in seen
    assert seen["export_tifxyz"] is True and seen["render_keep_xyz"] is True
