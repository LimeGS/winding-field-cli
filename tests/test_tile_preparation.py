import sqlite3

import numpy as np
import pytest
from test_surface_mesh import source

from winding_field import surface_mesh as mesh


def test_extract_prepares_each_tile_once_for_all_levels(tmp_path, monkeypatch):
    values = np.broadcast_to(np.arange(5.0), (5, 5, 5)).copy()
    original = mesh._prepare_tile
    calls = []

    def counted(values, valid):
        calls.append(values.shape)
        return original(values, valid)

    monkeypatch.setattr(mesh, "_prepare_tile", counted)
    with source(tmp_path, values) as field:
        info = mesh.extract_mesh(field, tmp_path / "mesh", [-1, 0, 1.5, 2, 3.5, 4, 5], tile=2, level_batch=2)
    assert len(calls) == info["total_tiles"] == 8
    assert [item["level"] for item in info["levels"]] == [-1, 0, 1.5, 2, 3.5, 4, 5]
    assert info["levels"][0]["faces"] == info["levels"][-1]["faces"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_prepared_geometry_matches_each_level_separately(partial):
    values = np.random.default_rng(42).normal(size=(4, 5, 6))
    valid = np.ones(values.shape, bool)
    if partial:
        valid[1, 1, 1] = False
    prepared = mesh._prepare_tile(values, valid)
    low, origin, spacing = np.array([4, 5, 6]), np.array([11.0, 17.0, 19.0]), np.array([2.0, 3.0, 5.0])
    for level in (-10, values.min(), 0.0, values[2, 2, 2], values.max(), 10):
        expected = mesh._tile_mesh(values, valid, low, level, origin, spacing)
        actual = mesh._tile_mesh(values, valid, low, level, origin, spacing, prepared=prepared)
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])


def test_preparation_never_downcasts_before_level_subtraction():
    values = np.broadcast_to(1e10 + np.arange(3.0), (3, 3, 3)).copy()
    valid = np.ones(values.shape, bool)
    prepared = mesh._prepare_tile(values, valid)
    for level in (1e10 + 0.25, 1e10 + 1.5):
        keys, world, faces = mesh._tile_mesh(
            values, valid, np.zeros(3, int), level, np.zeros(3), np.ones(3), prepared=prepared
        )
        assert len(faces) == 8 and all(key.startswith("e:") for key in keys)
        np.testing.assert_array_equal(world[:, 0], np.full(len(world), level - 1e10))


def test_recursive_subcrops_prepare_their_own_masks(monkeypatch):
    values = np.broadcast_to(np.arange(3.0), (3, 3, 3)).copy()
    valid = np.ones(values.shape, bool)
    original, prepare = mesh._tile_mesh_once, mesh._prepare_tile
    shapes = []

    def forced(values, *args, **kwargs):
        if values.shape == (3, 3, 3):
            raise mesh._LatticePrecisionError("exercise subdivision")
        return original(values, *args, **kwargs)

    def counted(values, valid):
        shapes.append(values.shape)
        return prepare(values, valid)

    monkeypatch.setattr(mesh, "_tile_mesh_once", forced)
    monkeypatch.setattr(mesh, "_prepare_tile", counted)
    _, _, faces = mesh._tile_mesh(values, valid, np.zeros(3, int), 0.5, np.zeros(3), np.ones(3))
    assert len(faces) == 8
    assert shapes == [(3, 3, 3)] + [(2, 2, 2)] * 8


def test_multilevel_geometry_and_topology_match_single_level_runs(tmp_path):
    values = np.random.default_rng(17).normal(size=(4, 4, 4))
    with source(tmp_path, values) as field:
        levels = [-10.0, 0.13, 0.0, 10.0]
        multi = mesh.extract_mesh(field, tmp_path / "multi", levels, tile=2)
        for i, level in enumerate(levels):
            single = mesh.extract_mesh(field, tmp_path / f"single-{i}", [level], tile=2)
            assert dict(multi["levels"][i], id=0) == single["levels"][0]
            with (
                sqlite3.connect(tmp_path / "multi/mesh.sqlite") as a,
                sqlite3.connect(tmp_path / f"single-{i}/mesh.sqlite") as b,
            ):
                query = "SELECT key,x,y,z,local FROM vertices WHERE level=? ORDER BY local"
                assert list(a.execute(query, (i,))) == list(b.execute(query, (0,)))
