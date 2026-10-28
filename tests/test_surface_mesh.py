import hashlib
import importlib.util
import sqlite3

import numpy as np
import pytest

from winding_field.surface_sources import FieldSource


def module():
    assert importlib.util.find_spec("winding_field.surface_mesh"), "tiled mesh engine missing"
    from winding_field import surface_mesh

    return surface_mesh


def source(tmp_path, values, mask=None, origin=(11.0, -4.0, 19.0), spacing=(2.0, 3.0, 5.0)):
    path = tmp_path / "field.npy"
    np.save(path, np.asarray(values, np.float64))
    descriptor = dict(kind="npy", path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if mask is not None:
        path = tmp_path / "support.npy"
        np.save(path, mask)
        descriptor["mask"] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    geometry = dict(
        frame="synthetic/world",
        voxel_um=8.64,
        sheet_step=1.0,
        origin_xyz=list(origin),
        spacing_xyz=list(spacing),
        shape_zyx=list(values.shape),
    )
    return FieldSource(descriptor, geometry)


def geometry(root, level=0):
    with sqlite3.connect(root / "mesh.sqlite") as db:
        vertices = dict(
            (row[0], row[1:]) for row in db.execute("SELECT id,x,y,z FROM vertices WHERE level=?", (level,))
        )
        faces = list(db.execute("SELECT a,b,c FROM faces WHERE level=? ORDER BY id", (level,)))
    return np.array([[vertices[i] for i in face] for face in faces], float).reshape(-1, 3, 3)


@pytest.mark.parametrize("tile", [1, 2, 4, 16])
@pytest.mark.parametrize("level", [1.5, 2.0])
def test_plane_global_edges_and_native_area_have_no_tile_cracks(tmp_path, tile, level):
    values = np.broadcast_to(np.arange(7.0), (5, 6, 7)).copy()
    with source(tmp_path, values) as field:
        result = module().extract_mesh(field, tmp_path / "mesh", [level], tile=tile)
    tri = geometry(tmp_path / "mesh")
    assert result["status"] == "MESH_READY"
    np.testing.assert_allclose(tri[:, :, 0], 11 + level * 2, atol=1e-10)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    assert area == pytest.approx(300.0)
    assert len(tri) == 40 and result["levels"][0]["components"] == 1
    assert result["levels"][0]["nonmanifold_edges"] == 0
    assert len({tuple(sorted(map(tuple, face))) for face in tri}) == len(tri)
    assert (tmp_path / "mesh/level-0000.obj").is_file()


def test_masked_cell_does_not_invent_a_bridge(tmp_path):
    values = np.broadcast_to(np.arange(7.0), (5, 6, 7)).copy()
    mask = np.ones(values.shape, bool)
    mask[2, 2, 2] = False
    with source(tmp_path, values, mask) as field:
        module().extract_mesh(field, tmp_path / "mesh", [2.5], tile=2)
    tri = geometry(tmp_path / "mesh")
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    assert area == pytest.approx(240.0)


def test_multiple_components_and_empty_levels_are_preserved(tmp_path):
    z, y, x = np.indices((9, 9, 16))
    values = np.minimum((x - 4) ** 2 + (y - 4) ** 2 + (z - 4) ** 2, (x - 11) ** 2 + (y - 4) ** 2 + (z - 4) ** 2).astype(
        float
    )
    with source(tmp_path, values, origin=(0, 0, 0), spacing=(1, 1, 1)) as field:
        result = module().extract_mesh(field, tmp_path / "mesh", [3.5, -100.0], tile=3, level_batch=1)
    assert result["levels"][0]["components"] == 2
    assert result["levels"][1]["faces"] == 0 and result["levels"][1]["components"] == 0
    assert result["levels"][0]["nonmanifold_edges"] == 0


def test_stopped_resume_matches_full_and_rejects_changed_problem(tmp_path):
    values = np.broadcast_to(np.arange(7.0), (5, 6, 7)).copy()
    with source(tmp_path, values) as field:
        api = module()
        api.extract_mesh(field, tmp_path / "full", [1.5, 3.5], tile=2)
        calls = [0]

        def stop():
            calls[0] += 1
            return calls[0] > 3

        result = api.extract_mesh(field, tmp_path / "split", [1.5, 3.5], tile=2, stop=stop)
        assert result["status"] == "CHECKPOINTED_STOP" and result["next_tile"] > 0
        with pytest.raises(ValueError):
            api.extract_mesh(field, tmp_path / "split", [2.0], tile=2, resume=True)
        api.extract_mesh(field, tmp_path / "split", [1.5, 3.5], tile=2, resume=True)
    for level in range(2):
        np.testing.assert_array_equal(geometry(tmp_path / "full", level), geometry(tmp_path / "split", level))
    assert (tmp_path / "full/level-0000.obj").read_bytes() == (tmp_path / "split/level-0000.obj").read_bytes()


@pytest.mark.parametrize(
    "options",
    [
        dict(tile=0),
        dict(tile=True),
        dict(level_batch=0),
        dict(levels=[float("nan")]),
        dict(levels=[]),
        dict(levels=[1.0, 1.0]),
    ],
)
def test_invalid_controls_leave_no_new_output(tmp_path, options):
    values = np.broadcast_to(np.arange(4.0), (3, 3, 4)).copy()
    with source(tmp_path, values) as field:
        args = dict(levels=[1.5])
        args.update(options)
        with pytest.raises(ValueError):
            module().extract_mesh(field, tmp_path / "bad", **args)
    assert not (tmp_path / "bad").exists()


def test_foreign_output_and_failed_disk_admission_preserve_source(tmp_path):
    values = np.broadcast_to(np.arange(4.0), (3, 3, 4)).copy()
    root = tmp_path / "foreign"
    root.mkdir()
    (root / "keep").write_text("user")
    with source(tmp_path, values) as field:
        with pytest.raises((ValueError, FileExistsError)):
            module().extract_mesh(field, root, [1.5])
        with pytest.raises(RuntimeError):
            module().extract_mesh(field, tmp_path / "disk", [1.5], reserve_bytes=2**63)
    assert (root / "keep").read_text() == "user"


def test_reader_streams_real_world_triangles_and_detects_tampering(tmp_path):
    values = np.broadcast_to(np.arange(4.0), (3, 3, 4)).copy()
    with source(tmp_path, values) as field:
        module().extract_mesh(field, tmp_path / "mesh", [1.5], tile=1)
    with module().MeshReader(tmp_path / "mesh") as reader:
        batches = list(reader.triangles(batch_size=3))
        assert all(len(t) <= 3 for t, labels in batches)
        assert sum(len(t) for t, labels in batches) == 8
        assert reader.metadata["frame"] == "synthetic/world"
        assert np.asarray(reader.bounds).shape == (2, 3)
    with sqlite3.connect(tmp_path / "mesh/mesh.sqlite") as db:
        db.execute("UPDATE vertices SET x=x+1")
    with pytest.raises(ValueError):
        module().MeshReader(tmp_path / "mesh")


@pytest.mark.parametrize("seed", [2, 9, 17])
def test_ambiguous_cells_have_same_global_topology_tiled_or_monolithic(tmp_path, seed):
    values = np.random.default_rng(seed).normal(size=(6, 7, 8))
    with source(tmp_path, values, origin=(0, 0, 0), spacing=(1, 1, 1)) as field:
        large = module().extract_mesh(field, tmp_path / "large", [0.13], tile=32)["levels"][0]
        tiled = module().extract_mesh(field, tmp_path / "small", [0.13], tile=2)["levels"][0]
    for key in ("faces", "components", "boundary_edges", "nonmanifold_edges", "duplicate_faces"):
        assert tiled[key] == large[key], (key, tiled, large)
    assert tiled["area_world2"] == pytest.approx(large["area_world2"], rel=2e-6)


def test_obj_export_resume_preserves_modified_user_output(tmp_path):
    values = np.broadcast_to(np.arange(4.0), (3, 3, 4)).copy()
    with source(tmp_path, values) as field:
        module().extract_mesh(field, tmp_path / "mesh", [1.5], tile=1)
        target = tmp_path / "mesh/level-0000.obj"
        target.write_text("user changed it")
        with pytest.raises(ValueError, match="OBJ"):
            module().extract_mesh(field, tmp_path / "mesh", [1.5], tile=1, resume=True)
    assert target.read_text() == "user changed it"
