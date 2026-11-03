import hashlib
import io
import json
import math
import shutil
import sqlite3
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from test_surface_mesh import source

import field_store
from field_runtime import ResourceUnavailable
from field_store import BlockStore
from winding_field.surface_mesh import MeshReader, extract_mesh, import_mesh
from winding_field.surface_sources import FieldSource


@pytest.mark.parametrize("tile", [1, 2, 4])
def test_exact_lattice_sphere_is_closed_and_connected_for_every_tile(tmp_path, tile):
    z, y, x = np.indices((5, 5, 5))
    values = ((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2).astype(float)
    with source(tmp_path, values, origin=(0, 0, 0), spacing=(1, 1, 1)) as field:
        info = extract_mesh(field, tmp_path / "mesh", [2.0], tile=tile)["levels"][0]
    assert info["faces"] == 32 and info["components"] == 1 and info["boundary_edges"] == 0


@pytest.mark.parametrize("corruption", ["cursor", "vertices", "faces"])
def test_partial_sqlite_logical_corruption_is_not_adopted(tmp_path, corruption):
    values = np.broadcast_to(np.arange(4.0), (3, 3, 4)).copy()
    with source(tmp_path, values) as field:
        root = tmp_path / "mesh"
        calls = [0]

        def stop():
            calls[0] += 1
            return calls[0] > 11

        extract_mesh(field, root, [1.5], tile=1, stop=stop)
        with sqlite3.connect(root / "mesh.sqlite") as db:
            if corruption == "cursor":
                db.execute("UPDATE meta SET value='12' WHERE key='next_tile'")
            elif corruption == "vertices":
                db.execute("UPDATE vertices SET x=x+100")
            else:
                db.execute("UPDATE faces SET area=area+1")
        with pytest.raises(ValueError):
            extract_mesh(field, root, [1.5], tile=1, resume=True)


def chunks(tmp_path):
    root = tmp_path / "chunks"
    values = np.broadcast_to(np.arange(7.0) + 1.0, (5, 6, 7)).copy()
    geometry = dict(
        frame="other-scroll/world",
        voxel_um=3.0,
        sheet_step=2.0,
        origin_xyz=[17.0, -4.0, 29.0],
        spacing_xyz=[2.0, 3.0, 5.0],
        shape_zyx=list(values.shape),
    )
    with BlockStore(root, values.shape, 7, dict(geometry=geometry)) as store:
        store.commit((0, 0, 0), values, {}, np.ones(values.shape, bool))
    cp = root / "CHECKPOINT.json"
    descriptor = dict(
        kind="chunks", path=str(cp), sha256=hashlib.sha256(cp.read_bytes()).hexdigest(), support_policy="observed"
    )
    return descriptor, geometry


def test_resume_hashes_consumed_payloads_even_with_no_tiles_left(tmp_path):
    descriptor, geometry = chunks(tmp_path)
    root = tmp_path / "mesh"
    with FieldSource(descriptor, geometry) as field:
        extract_mesh(field, root, [2.5], tile=2)
    cp = json.loads(Path(descriptor["path"]).read_bytes())
    payload = Path(descriptor["path"]).parent / next(iter(cp["chunks"].values()))["path"]
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 1
    payload.write_bytes(raw)
    with FieldSource(descriptor, geometry) as field:
        with pytest.raises(ValueError, match="hash"):
            extract_mesh(field, root, [2.5], tile=2, resume=True)


def test_decoder_implementation_is_part_of_resume_identity(tmp_path, monkeypatch):
    descriptor, geometry = chunks(tmp_path)
    root = tmp_path / "mesh"
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 1

    with FieldSource(descriptor, geometry) as field:
        extract_mesh(field, root, [2.5], tile=1, stop=stop)
    original = field_store._decode

    def changed(*args):
        delta, offset, mask = original(*args)
        return delta, offset - 4, mask

    monkeypatch.setattr(field_store, "_decode", changed)
    with FieldSource(descriptor, geometry) as field:
        with pytest.raises(ValueError):
            extract_mesh(field, root, [2.5], tile=1, resume=True)


@pytest.mark.parametrize("member", ["metadata", "levels"])
def test_manifest_cannot_relabel_committed_mesh(tmp_path, member):
    root = tmp_path / "mesh"
    values = np.broadcast_to(np.arange(7.0), (5, 6, 7)).copy()
    with source(tmp_path, values) as field:
        extract_mesh(field, root, [1.5, 3.5], tile=2)
    manifest = json.loads((root / "MESH.json").read_bytes())
    if member == "metadata":
        manifest["metadata"]["frame"] = "unrelated-world"
    else:
        manifest["levels"][0]["level"] = 999
    (root / "MESH.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        with MeshReader(root):
            pass


def test_completed_resume_preserves_changed_manifest(tmp_path):
    root = tmp_path / "mesh"
    values = np.broadcast_to(np.arange(7.0), (5, 6, 7)).copy()
    with source(tmp_path, values) as field:
        extract_mesh(field, root, [1.5, 3.5], tile=2)
        file = root / "MESH.json"
        manifest = json.loads(file.read_bytes())
        manifest["metadata"]["frame"] = "foreign-frame"
        file.write_text(json.dumps(manifest))
        before = file.read_bytes()
        with pytest.raises(ValueError):
            extract_mesh(field, root, [1.5, 3.5], tile=2, resume=True)
        assert file.read_bytes() == before


def test_npz_declared_shape_is_bounded_before_numpy_allocation(tmp_path, monkeypatch):
    path = tmp_path / "bad.npz"
    header = io.BytesIO()
    faces = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr="<f8", fortran_order=False, shape=(100_000_000, 3)))
    np.save(faces, np.empty((0, 3), np.int64))
    with ZipFile(path, "w") as archive:
        archive.writestr("vertices_xyz.npy", header.getvalue())
        archive.writestr("faces.npy", faces.getvalue())
    original = np.ndarray

    def allocation(shape, *args, **kwargs):
        count = int(shape) if isinstance(shape, (int, np.integer)) else math.prod(shape)
        if count > 1_000_000:
            raise AssertionError("unbounded allocation attempted")
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "ndarray", allocation)
    with pytest.raises(ValueError):
        import_mesh(
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            tmp_path / "output",
            dict(frame="world", voxel_um=1.0),
            max_bytes=1024,
        )


def test_import_reserve_covers_initial_database_growth(tmp_path, monkeypatch):
    path = tmp_path / "input.obj"
    path.write_text("v 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
    monkeypatch.setattr(
        "field_runtime.shutil.disk_usage", lambda path: shutil._ntuple_diskusage(1_000_000, 899_999, 100_001)
    )
    with pytest.raises(ResourceUnavailable):
        import_mesh(
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            tmp_path / "output",
            dict(frame="world", voxel_um=1.0),
            reserve_bytes=100_000,
        )
    assert not (tmp_path / "output").exists()


def test_near_node_plane_retains_float64_crossing_after_float32_rounding(tmp_path):
    from winding_field.surface_mesh import _tile_mesh

    level = 100.0 - 2e-7
    values = np.broadcast_to(np.arange(129.0), (3, 3, 129)).copy()
    keys, vertices, faces = _tile_mesh(
        values, np.ones(values.shape, bool), np.zeros(3, int), level, np.zeros(3), np.ones(3)
    )
    assert len(faces) == 8 and len(keys) == len(vertices)
    np.testing.assert_allclose(vertices[:, 0], level, rtol=0, atol=1e-12)
    triangles = vertices[faces]
    area = (
        np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
    )
    assert area == pytest.approx(4.0)


@pytest.mark.parametrize("tile", [1, 2, 128])
def test_tiny_near_node_component_is_not_erased_or_cracked_across_tiles(tmp_path, tile):
    values = np.ones((3, 3, 130))
    values[1, 1, 64] = -1e-8
    with source(tmp_path, values, origin=(0, 0, 0), spacing=(1, 1, 1)) as field:
        info = extract_mesh(field, tmp_path / "mesh", [0.0], tile=tile)["levels"][0]
    assert info["vertices"] == 6 and info["faces"] == 8
    assert info["components"] == 1 and info["boundary_edges"] == 0
    assert info["nonmanifold_edges"] == info["inconsistent_edges"] == info["duplicate_faces"] == 0


def test_crossing_whose_sign_underflows_float32_is_not_silently_discarded(tmp_path):
    values = np.ones((2, 2, 2))
    values[0, 0, 0] = -1e-50
    with source(tmp_path, values, origin=(0, 0, 0), spacing=(1, 1, 1)) as field:
        with pytest.raises(ValueError, match="float32"):
            extract_mesh(field, tmp_path / "mesh", [0.0], tile=1)
