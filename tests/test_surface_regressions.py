import json

import pytest
from test_surface_workflow import prepared

from field_runtime import ResourceUnavailable
from surface_mesh import MeshReader
from winding_field.surface_workflow import render_bundle, run_surfaces


def test_completed_workflow_resume_is_idempotent(tmp_path):
    path, _ = prepared(tmp_path)
    root = tmp_path / "output"
    run_surfaces(path, root)
    before = (root / "mesh.sqlite").read_bytes()
    assert run_surfaces(path, root, resume=True)["status"] == "SURFACES_READY"
    assert (root / "mesh.sqlite").read_bytes() == before


def test_workflow_recovers_after_one_view_was_published(tmp_path, monkeypatch):
    from winding_field import surface_workflow

    path, _ = prepared(tmp_path)
    root = tmp_path / "output"
    publish = surface_workflow._publish_png
    calls = [0]

    def interrupted(*args):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("interrupted second view")
        return publish(*args)

    with monkeypatch.context() as patch:
        patch.setattr(surface_workflow, "_publish_png", interrupted)
        with pytest.raises(RuntimeError):
            run_surfaces(path, root)
    before = (root / "renders/front.png").read_bytes()
    assert run_surfaces(path, root, resume=True)["status"] == "SURFACES_READY"
    assert (root / "renders/front.png").read_bytes() == before


@pytest.mark.parametrize("member", ["metadata", "levels"])
def test_manifest_cannot_relabel_committed_mesh(tmp_path, member):
    path, _ = prepared(tmp_path)
    root = tmp_path / "mesh"
    run_surfaces(path, root, render=False)
    manifest = json.loads((root / "MESH.json").read_bytes())
    if member == "metadata":
        manifest["metadata"]["frame"] = "unrelated-world"
    else:
        manifest["levels"][0]["level"] = 999
    (root / "MESH.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        MeshReader(root)


def test_completed_resume_preserves_changed_manifest(tmp_path):
    path, _ = prepared(tmp_path)
    root = tmp_path / "mesh"
    run_surfaces(path, root, render=False)
    file = root / "MESH.json"
    manifest = json.loads(file.read_bytes())
    manifest["metadata"]["frame"] = "foreign-frame"
    file.write_text(json.dumps(manifest))
    before = file.read_bytes()
    with pytest.raises(ValueError):
        run_surfaces(path, root, resume=True, render=False)
    assert file.read_bytes() == before


def test_render_has_its_own_disk_admission(tmp_path):
    path, _ = prepared(tmp_path)
    root = tmp_path / "mesh"
    run_surfaces(path, root, render=False)
    with pytest.raises(ResourceUnavailable):
        render_bundle(root, tmp_path / "render", settings=dict(reserve_bytes=2**63))
    assert not (tmp_path / "render").exists()


@pytest.mark.parametrize("filename", ["front.json", "RENDER.json"])
def test_render_retry_preserves_foreign_json_receipts(tmp_path, filename):
    path, _ = prepared(tmp_path)
    root = tmp_path / "mesh"
    run_surfaces(path, root, render=False)
    output = tmp_path / "render"
    settings = dict(width=32, height=32)
    render_bundle(root, output, settings=settings)
    (output / filename).write_text('{"foreign":true}')
    with pytest.raises(ValueError):
        render_bundle(root, output, settings=settings)
    assert (output / filename).read_text() == '{"foreign":true}'


@pytest.mark.parametrize("name", ["PLAN", "plan", "RENDER"])
def test_reserved_view_names_fail_before_output(tmp_path, name):
    path, config = prepared(tmp_path)
    config["render"]["views"] = [dict(name=name)]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_surfaces(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
