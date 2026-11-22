"""Native/snapshot AFV import: exact parity with the frozen deployed importer.

``afv_frozen``/``afv_clip_frozen`` are byte copies of the importer at f5f682c
(deployed v1.6.5). Fragments are compared by their float64 bytes (signed zero,
retained bend vertices, order); imports are compared by logical database rows.
"""

from contextlib import closing
import io
import json
import shutil
import sqlite3

import numpy as np
import pytest

import afv_clip_frozen
import afv_frozen
from test_afv_benchmark import tool
from test_afv_import import fixture
from test_splines import upstream_curves
from winding_field import afv, afv_clip, afv_native, afv_support
from winding_field.artifacts import Tiles
from winding_field.util import atomic_json


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    import os

    os.environ.setdefault("WINDING_FIELD_NATIVE_CACHE", str(tmp_path_factory.mktemp("native")))
    return afv_native.load(afv_native.build())


def _random_support(maps, rng, keep=0.85):
    for key in maps.keys():
        lo, hi = maps.box(key)
        support = rng.random(tuple(hi - lo)) < keep
        # Narrow one-voxel gaps on exact planes.
        support[..., np.arange(lo[2], hi[2]) == 12] = False
        maps.write(key, dict(support=support))


def _bits(fragments):
    return [(f.shape, f.tobytes()) for f in fragments]


EDGE_CASES = [
    [[0, 2, 3], [23, 2, 3]],
    [[23, 2, 3], [0, 2, 3]],
    [[2, 2, 3], [2, 2, 3], [20, 2, 3]],  # repeated point / zero-length edge
    [[-2, 2, 3], [8, 2, 3], [25, 2, 3]],  # ROI entry and exit
    [[0, 0, 0], [7, 7, 7]],  # grid nodes
    [[0, 0, 0], [0, 0, 7]],
    [[2, 2, 3], [7, 2, 3], [12, 2, 3], [20, 2, 3]],  # bends on the gap plane
    [[12, 2, 3], [12, 6, 3]],  # along the gap plane
    [[23, 7, 7], [23, 0, 7], [0, 0, 7]],  # upper faces
    [[3.5, 3.5, 3.5], [3.5, 3.5, 3.5]],  # degenerate curve
    [[-5, -5, -5], [-1, -1, -1]],  # fully outside
    [[0.0, -0.0, 3.0], [5.0, -0.0, 3.0]],  # signed zero on a lower face
    [[-0.0, 2.0, 3.0], [-0.0, 2.0, 9.0]],
    [[11.999999999, 2, 3], [12.000000001, 2, 3]],  # sub-voxel crossing of the gap
    [[1e-12, 1e-12, 1e-12], [23, 7, 7]],
]


def _cases(rng, n=300):
    cases = [np.asarray(c, float) for c in EDGE_CASES]
    for i in range(n):
        size = int(rng.integers(2, 14))
        points = rng.uniform([-3, -3, -3], [27, 11, 11], (size, 3))
        if i % 3 == 0:
            points = np.round(points * 2) / 2  # faces, edges and nodes
        if i % 7 == 0:
            points[1::2] = points[::2][: len(points[1::2])]  # repeats
        cases.append(points)
    return cases


def test_native_and_batched_numpy_reproduce_frozen_fragments_bit_for_bit(tmp_path, library):
    tool().generated_input(tmp_path / "run", 1, 1)
    rng = np.random.default_rng(20261121)
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        _random_support(maps, rng)
        dense, info = afv_support.open_or_build(
            maps, tmp_path / "snap", maps.identity(), limit_bytes=1 << 20, reserve_bytes=0
        )
        assert info["snapshot"] == "dense"
        cases = _cases(rng)
        frozen_cache = afv_clip_frozen.SupportTiles(maps, cache_bytes=1 << 20)
        expected = [list(afv_clip_frozen.pieces(points, frozen_cache)) for points in cases]
        scalar = [list(afv_frozen._scalar_pieces(points, maps)) for points in cases]
        native = afv_native.clip_blocks(library, cases, dense.dense, afv_clip.MAX_INTERVALS)
        handled = 0
        for points, want, slow, got in zip(cases, expected, scalar, native):
            assert _bits(want) == _bits(slow)
            # Batched NumPy, over the dense snapshot and over bounded tiles.
            assert _bits(list(afv_clip.pieces(points, dense))) == _bits(want)
            assert _bits(list(afv_clip.pieces(points, maps))) == _bits(want)
            if got is not None:
                handled += 1
                assert _bits(got) == _bits(want), points
        assert handled > 0.9 * len(cases)
        dense.close()


def test_native_defers_ambiguous_and_streaming_blocks_to_the_reference(tmp_path, library):
    tool().generated_input(tmp_path / "run", 1, 1)
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        dense, _ = afv_support.open_or_build(
            maps, tmp_path / "snap", maps.identity(), limit_bytes=1 << 20, reserve_bytes=0
        )
        signed = np.array([[-0.0, 2.0, 3.0], [-0.0, 2.0, 9.0]])
        long = np.array([[0.0, 2.0, 3.0], [23.0, 2.0, 3.0]])
        result = afv_native.clip_blocks(library, [signed, long, long], dense.dense, 4)
        # -0.0 clip inputs and over-budget intervals are the reference's to decide.
        assert result[0] is None and result[1] is None and result[2] is None
        ok = afv_native.clip_blocks(library, [long], dense.dense, afv_clip.MAX_INTERVALS)
        assert _bits(ok[0]) == _bits(list(afv_clip_frozen.pieces(long, maps)))
        # Bounded output: a tiny first capacity grows instead of truncating.
        many = [long] * 50
        grown = afv_native.clip_blocks(library, many, dense.dense, afv_clip.MAX_INTERVALS)
        assert all(_bits(g) == _bits(ok[0]) for g in grown)
        with pytest.raises(ValueError, match="finite"):
            afv_native.clip_blocks(library, [long * np.inf], dense.dense, 10)
        dense.close()


def test_support_snapshot_reuse_mutation_and_corrupt_tiles(tmp_path):
    tool().generated_input(tmp_path / "run", 1, 1)
    snap = tmp_path / "snap"
    snap.mkdir()
    with closing(Tiles(tmp_path / "run/maps")) as maps:
        _random_support(maps, np.random.default_rng(3))
        identity = maps.identity()
        dense, info = afv_support.open_or_build(
            maps, snap, identity, limit_bytes=1 << 20, reserve_bytes=0
        )
        for key in maps.keys():
            np.testing.assert_array_equal(
                dense.load(key)["support"], maps.load(key, ("support",))["support"]
            )
        assert not info["reused"]
        dense.close()
        again, info = afv_support.open_or_build(
            maps, snap, identity, limit_bytes=1 << 20, reserve_bytes=0
        )
        assert info["reused"]
        # Mutating the published snapshot is detected by the final verification.
        with open(again.path, "r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
        with pytest.raises(ValueError, match="snapshot changed"):
            again.verify()
        again.close()
        rebuilt, info = afv_support.open_or_build(
            maps, snap, identity, limit_bytes=1 << 20, reserve_bytes=0
        )
        assert not info["reused"]
        rebuilt.close()
        assert (
            afv_support.open_or_build(maps, snap, identity, limit_bytes=1, reserve_bytes=0)[0]
            is None
        )
        maps.path((0, 0, 0)).write_bytes(maps.path((0, 0, 0)).read_bytes() + b"x")
        with pytest.raises(ValueError, match="corrupt tile"):
            afv_support.open_or_build(
                maps, tmp_path / "other", identity, limit_bytes=1 << 20, reserve_bytes=0
            )


def _random_import(root, seed, n=60):
    _, config = fixture(root)
    rng = np.random.default_rng(seed)
    with closing(Tiles(root / "maps")) as maps:
        _random_support(maps, rng)
        identity = maps.identity()
    meta = json.loads((root / "PREPARED.json").read_bytes())
    meta["tiles_sha256"] = identity
    atomic_json(root / "PREPARED.json", meta)
    path = root / "external.afv"
    path.unlink()
    curves = []
    # Grid XYZ inside the AFV volume (8, 8, 64): beyond x = 24 lies outside the maps.
    upper = np.array([63.5, 7.5, 7.5])
    for i in range(n):
        size = int(rng.integers(2, 30))
        points = rng.uniform([0, 0, 0], upper, (size, 3))
        if i % 3 == 0:
            points = np.minimum(np.round(points * 2) / 2, upper)
        curves.append((int(i % 2 + 1), points))
    upstream_curves(path, (8, 8, 64), curves)
    with sqlite3.connect(path) as db:
        for fid in range(1, n + 1):
            note = json.dumps(dict(original_annotation=fid))
            db.execute("UPDATE fibers SET annotation=? WHERE id=?", (note, fid))
    return config


def _logical(root):
    with closing(sqlite3.connect(root / "fibres/fibres.sqlite")) as db:
        return [
            list(db.execute(f"SELECT * FROM {table} ORDER BY {order}"))
            for table, order in (("curves", "id"), ("curve_bounds", "id"), ("done", "fid,block"))
        ]


def _receipt(root):
    receipt = json.loads((root / "FIBRES.json").read_bytes())
    receipt.pop("sha256")
    return receipt


@pytest.mark.parametrize(
    "backend,workers",
    [("auto", 1), ("native", 1), ("numpy", 1), ("reference", 1), ("native", 3), ("numpy", 2)],
)
def test_import_matches_frozen_importer_logically(tmp_path, monkeypatch, library, backend, workers):
    from test_import_parallel import resources

    config = _random_import(tmp_path / "frozen", 7)
    assert afv_frozen.import_fibres(tmp_path / "frozen", config)["status"] == "FIBRES_READY"
    shutil.copytree(tmp_path / "frozen", tmp_path / "new")
    shutil.rmtree(tmp_path / "new/fibres")
    (tmp_path / "new/FIBRES.json").unlink()
    if workers > 1:
        resources(monkeypatch)
    monkeypatch.setattr(afv, "_JOB_BLOCKS", 7)  # several work units, partial last unit
    receipt = afv.import_fibres(
        tmp_path / "new", dict(config, afv_clip_backend=backend, import_workers=workers)
    )
    assert receipt["status"] == "FIBRES_READY"
    assert _logical(tmp_path / "new") == _logical(tmp_path / "frozen")
    assert _receipt(tmp_path / "new") == _receipt(tmp_path / "frozen")
    status = json.loads((tmp_path / "new/fibres/STATUS.json").read_bytes())["execution"]
    if backend in ("auto", "native"):
        assert status["clipper"] == "native" and status["native_blocks"] > 0
    if backend == "reference":
        assert status["support"]["snapshot"] == "bounded-tiles"
    assert not list((tmp_path / "new/fibres").glob("support-*"))  # snapshot removed


def test_stop_and_fast_resume_verify_retained_rows_without_reclipping(
    tmp_path, monkeypatch, library
):
    config = _random_import(tmp_path / "frozen", 11)
    afv_frozen.import_fibres(tmp_path / "frozen", config)
    run = tmp_path / "run"
    shutil.copytree(tmp_path / "frozen", run)
    shutil.rmtree(run / "fibres")
    (run / "FIBRES.json").unlink()
    monkeypatch.setattr(afv, "_BATCH_BLOCKS", 5)
    monkeypatch.setattr(afv, "_JOB_BLOCKS", 4)

    def stop():
        path = run / "fibres/fibres.sqlite"
        if not path.exists():
            return False
        with closing(sqlite3.connect(path)) as db:
            return db.execute("SELECT count(*) FROM done").fetchone()[0] >= 10

    assert afv.import_fibres(run, config, stop)["status"] == "PAUSED"
    with closing(sqlite3.connect(run / "fibres/fibres.sqlite")) as db:
        assert db.execute("SELECT count(*) FROM done").fetchone()[0] == 10
    clipped = []
    original = afv._pieces

    def counting(points, *args, **kwargs):
        clipped.append(len(points))
        yield from original(points, *args, **kwargs)

    monkeypatch.setattr(afv, "_pieces", counting)
    assert afv.import_fibres(run, config)["status"] == "FIBRES_READY"
    status = json.loads((run / "fibres/STATUS.json").read_bytes())["execution"]
    assert status["resume"] == "fast" and status["retained_verified"] == 10
    with closing(sqlite3.connect(run / "fibres/fibres.sqlite")) as db:
        total = db.execute("SELECT count(*) FROM done").fetchone()[0]
    assert len(clipped) == total - 10  # retained blocks were not re-clipped
    assert _logical(run) == _logical(tmp_path / "frozen")


@pytest.mark.parametrize("damage", ["blob", "delete-curve", "bounds", "done"])
def test_fast_resume_rejects_corrupt_or_inconsistent_retained_rows(tmp_path, monkeypatch, damage):
    config = _random_import(tmp_path, 13)
    monkeypatch.setattr(afv, "_BATCH_BLOCKS", 5)

    def stop():
        path = tmp_path / "fibres/fibres.sqlite"
        if not path.exists():
            return False
        with closing(sqlite3.connect(path)) as db:
            return db.execute("SELECT count(*) FROM curves").fetchone()[0] >= 6

    assert afv.import_fibres(tmp_path, config, stop)["status"] == "PAUSED"
    with sqlite3.connect(tmp_path / "fibres/fibres.sqlite") as db:
        cid, blob = db.execute("SELECT id,points FROM curves ORDER BY id LIMIT 1").fetchone()
        if damage == "blob":
            world = np.load(io.BytesIO(blob))
            world[0, 0] = np.nextafter(world[0, 0], np.inf)
            buffer = io.BytesIO()
            np.save(buffer, world)
            db.execute("UPDATE curves SET points=? WHERE id=?", (buffer.getvalue(), cid))
        elif damage == "delete-curve":
            db.execute("DELETE FROM curves WHERE id=?", (cid,))
            db.execute("DELETE FROM curve_bounds WHERE id=?", (cid,))
        elif damage == "bounds":
            db.execute("UPDATE curve_bounds SET max_x=max_x+100 WHERE id=?", (cid,))
        else:
            db.execute("UPDATE done SET sha=? WHERE rowid=1", ("0" * 64,))
    with pytest.raises(ValueError, match="payload changed|inventory changed"):
        afv.import_fibres(tmp_path, config)


def test_legacy_partial_run_keeps_reclipping_verification(tmp_path, monkeypatch, library):
    config = _random_import(tmp_path / "run", 17)
    monkeypatch.setattr(afv_frozen, "_BATCH_BLOCKS", 5)

    def stop():
        path = tmp_path / "run/fibres/fibres.sqlite"
        if not path.exists():
            return False
        with closing(sqlite3.connect(path)) as db:
            return db.execute("SELECT count(*) FROM done").fetchone()[0] >= 10

    assert afv_frozen.import_fibres(tmp_path / "run", config, stop)["status"] == "PAUSED"
    tables = lambda: {  # noqa: E731
        r[0]
        for r in sqlite3.connect(tmp_path / "run/fibres/fibres.sqlite").execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "import_meta" not in tables()
    assert afv.import_fibres(tmp_path / "run", config)["status"] == "FIBRES_READY"
    status = json.loads((tmp_path / "run/fibres/STATUS.json").read_bytes())["execution"]
    assert status["resume"] == "legacy" and status["retained_reclipped"] == 10
    assert "import_meta" not in tables()
    reference = tmp_path / "reference"
    reference.mkdir()
    shutil.copytree(tmp_path / "run/maps", reference / "maps")
    for name in ("PREPARED.json", "external.afv", "geometry.json"):
        shutil.copy(tmp_path / "run" / name, reference / name)
    reference_config = dict(config, fibres=str(reference / "external.afv"))
    reference_config["fibres_geometry"] = str(reference / "geometry.json")
    afv_frozen.import_fibres(reference, reference_config)
    assert _logical(tmp_path / "run") == _logical(reference)


def test_blob_matches_numpy_save_exactly():
    for world in (
        np.array([[1.0, -0.0, 3.0], [np.nextafter(1.0, 2.0), 2.0, -1e-300]]),
        np.zeros((7, 3)),
        np.arange(3 * 1000, dtype=float).reshape(1000, 3),
    ):
        buffer = io.BytesIO()
        np.save(buffer, world, allow_pickle=False)
        assert afv._blob(world) == buffer.getvalue()


def test_native_backend_requires_a_compiler_only_when_requested(tmp_path, monkeypatch):
    config = _random_import(tmp_path / "auto", 19, n=8)
    shutil.copytree(tmp_path / "auto", tmp_path / "native")

    def missing():
        raise afv_native.NativeUnavailable("no compiler")

    monkeypatch.setattr(afv_native, "build", missing)
    receipt = afv.import_fibres(tmp_path / "auto", config)
    assert receipt["status"] == "FIBRES_READY"
    status = json.loads((tmp_path / "auto/fibres/STATUS.json").read_bytes())["execution"]
    assert status["clipper"] == "numpy"
    with pytest.raises(afv_native.NativeUnavailable):
        afv.import_fibres(tmp_path / "native", dict(config, afv_clip_backend="native"))
    with pytest.raises(ValueError, match="afv_clip_backend"):
        afv_native.backend(dict(afv_clip_backend="gpu"))


def test_downstream_spline_reader_consumes_native_fragments(tmp_path, library):
    from winding_field import splines

    config = _random_import(tmp_path, 23)
    afv.import_fibres(tmp_path, dict(config, afv_clip_backend="native"))
    with closing(sqlite3.connect(tmp_path / "fibres/fibres.sqlite")) as db:
        rows = db.execute("SELECT points FROM curves ORDER BY id").fetchall()
    assert rows
    for (blob,) in rows:
        world = np.load(io.BytesIO(blob), allow_pickle=False)
        assert world.dtype == np.float64 and world.shape[1] == 3 and len(world) >= 2
    assert hasattr(splines, "bind")
