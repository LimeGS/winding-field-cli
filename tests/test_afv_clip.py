from contextlib import closing

import numpy as np
import pytest

from test_afv_benchmark import tool
from winding_field import afv
from winding_field.artifacts import Tiles


def test_support_cache_survives_adjacent_source_blocks(tmp_path, monkeypatch):
    config = tool().generated_input(tmp_path / "run", 1000, 1)
    loads = []
    original = Tiles.load

    def load(self, key, names=None):
        if names == ("support",):
            loads.append(key)
        return original(self, key, names)

    monkeypatch.setattr(Tiles, "load", load)
    result = afv.import_fibres(tmp_path / "run", config)
    assert result["count"] == 2000
    assert len(loads) <= 3


def test_support_cache_eviction_and_ownership(tmp_path):
    from winding_field.afv_clip import SupportTiles

    tool().generated_input(tmp_path / "run", 1, 1)
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        cache = SupportTiles(maps, cache_bytes=512)
        first = cache.load((0, 0, 0))["support"]
        assert not first.flags.writeable
        cache.load((0, 0, 1))
        assert cache.bytes <= 512
        np.testing.assert_array_equal(cache.load((0, 0, 0))["support"], first)
        cache.close()
        assert maps.load((0, 0, 0))["support"].all()
        tiny = SupportTiles(maps, cache_bytes=1)
        tiny.load((0, 0, 0))
        assert tiny.bytes == 0


@pytest.mark.parametrize("backend", ["reference", "numpy"])
def test_cached_support_cannot_hide_mid_import_map_mutation(tmp_path, monkeypatch, backend):
    config = dict(tool().generated_input(tmp_path / "run", 4, 1), afv_clip_backend=backend)
    original = afv._pieces
    changed = False

    def pieces(*args, **kwargs):
        nonlocal changed
        yield from original(*args, **kwargs)
        if not changed:
            changed = True
            with closing(Tiles(tmp_path / "run/maps")) as maps:
                data = maps.load((0, 0, 0))
                data["support"][0, 0, 0] = False
                maps.write((0, 0, 0), data)

    monkeypatch.setattr(afv, "_pieces", pieces)
    with pytest.raises(ValueError, match="maps changed"):
        afv.import_fibres(tmp_path / "run", config)
    assert not (tmp_path / "run/FIBRES.json").exists()


@pytest.mark.parametrize(
    "points",
    [
        [[0, 2, 3], [23, 2, 3]],
        [[23, 2, 3], [0, 2, 3]],
        [[2, 2, 3], [2, 2, 3], [20, 2, 3]],
        [[-2, 2, 3], [8, 2, 3], [25, 2, 3]],
        [[0, 0, 0], [7, 7, 7]],
        [[0, 0, 0], [0, 0, 7]],
        [[2, 2, 3], [7, 2, 3], [12, 2, 3], [20, 2, 3]],
        [[12, 2, 3], [12, 6, 3]],
    ],
)
def test_vector_clipping_preserves_scalar_fragments_exactly(tmp_path, points):
    from winding_field.afv_clip import pieces, SupportTiles

    tool().generated_input(tmp_path / "run", 1, 1)
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        cache = SupportTiles(maps, cache_bytes=1 << 20)
        expected = list(afv._scalar_pieces(np.asarray(points, float), maps))
        actual = list(pieces(np.asarray(points, float), cache))
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_random_vector_clipping_keeps_order_vertices_and_subvoxel_gaps(tmp_path):
    from winding_field.afv_clip import pieces, SupportTiles

    tool().generated_input(tmp_path / "run", 1, 1)
    rng = np.random.default_rng(52)
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        for key in maps.keys():
            support = rng.random((8, 8, 8)) > 0.1
            maps.write(key, dict(support=support))
        cache = SupportTiles(maps, cache_bytes=1 << 20)
        for _ in range(40):
            points = rng.uniform([-2, -2, -2], [26, 10, 10], (12, 3))
            expected = list(afv._scalar_pieces(points, maps))
            actual = list(pieces(points, cache))
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)


def test_vector_clipping_streams_pathological_intervals_and_cancels(tmp_path, monkeypatch):
    from winding_field import afv_clip

    tool().generated_input(tmp_path / "run", 1, 1)
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        points = np.array([[0.0, 2.0, 3.0], [23.0, 2.0, 3.0]])
        expected = list(afv._scalar_pieces(points, maps))
        monkeypatch.setattr(afv_clip, "MAX_INTERVALS", 2)
        actual = list(afv_clip.pieces(points, maps))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        with pytest.raises(InterruptedError):
            list(afv_clip.pieces(points, maps, stop=lambda: True))
