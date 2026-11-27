"""Tiles stored by several threads and published in order equal a serial write."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from winding_field.artifacts import Tiles
from winding_field.util import file_sha


def _tiles(rng, tiles):
    return {
        key: dict(
            presence=rng.random((8, 8, 8), dtype=np.float32),
            tangents=rng.integers(0, 5, (8, 8, 8, 3)).astype(np.int16),
            support=rng.random((8, 8, 8)) > 0.5,
        )
        for key in tiles.keys()
    }


def test_threaded_store_then_ordered_publish_matches_write(tmp_path):
    serial = Tiles(tmp_path / "serial", shape=(8, 16, 24), chunk=8, identity="x")
    threaded = Tiles(tmp_path / "threaded", shape=(8, 16, 24), chunk=8, identity="x")
    data = _tiles(np.random.default_rng(0), serial)
    for key, arrays in data.items():
        serial.write(key, arrays, dict(ct_sha256="a"))
    with ThreadPoolExecutor(4) as pool:
        stored = [(key, pool.submit(threaded.store, key, arrays)) for key, arrays in data.items()]
        for key, sha in stored:
            threaded.publish(key, sha.result(), dict(ct_sha256="a"))
    for key in data:
        assert serial.path(key).read_bytes() == threaded.path(key).read_bytes()
        assert threaded.info(key) == serial.info(key)
    assert threaded.identity() == serial.identity()


def test_stored_tile_exists_only_once_published(tmp_path):
    tiles = Tiles(tmp_path / "maps", shape=(8, 8, 8), chunk=8, identity="x")
    (arrays,) = _tiles(np.random.default_rng(1), tiles).values()
    sha = tiles.store((0, 0, 0), arrays)
    assert tiles.path((0, 0, 0)).exists() and not tiles.has((0, 0, 0))
    tiles.publish((0, 0, 0), sha, dict(ct_sha256="a"))
    assert tiles.has((0, 0, 0)) and file_sha(tiles.path((0, 0, 0))) == sha
