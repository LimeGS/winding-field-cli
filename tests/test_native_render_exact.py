"""Bit and journal regressions for the native raster and its three PNGs."""

import sqlite3
from collections import Counter
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from winding_field import native_render as native
from winding_field.native_geometry import _schema
from winding_field.native_store import Journal, Paused


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raster_bits_across_batches_and_sticky_conflicts(dtype):
    origin = np.array([-0.25, 0.75])
    pitch = 0.5
    triangle = np.array([[0, 0], [256, 0], [0, 256]])
    uv = (triangle * pitch + origin).astype(dtype)
    v = np.array([[0.125, -2.5, 1.25], [128.125, -2.5, -62.75], [0.125, 125.5, -94.75]], dtype)
    face = np.array([[0, 1, 2]])
    # Repeated faces cross the 131072-candidate batch boundary. A smaller triangle
    # at a different Z conflicts; a final copy of the first cannot reclaim its pixels.
    cut_uv = (triangle / 4 * pitch + origin).astype(dtype)
    charts = [(v, np.tile(face, (6, 1)), uv), (v[::-1], face, cut_uv), (v, face, uv)]
    shape = (239, 231)
    xyz, normal, mask = native.rasterize(charts, origin, pitch, shape)

    y, x = np.indices(shape)
    covered = x + y <= 256
    conflict = x + y <= 64
    expected_xyz = np.stack([0.125 + x / 2, -2.5 + y / 2, 1.25 - x / 4 - 3 * y / 8], axis=-1)
    expected_xyz[~covered | conflict] = np.nan
    expected_normal = np.zeros((*shape, 3))
    expected_normal[covered] = np.array([2, 3, 4], dtype) / np.sqrt(dtype(29))
    np.testing.assert_array_equal(mask, np.where(conflict, 3, covered))
    np.testing.assert_array_equal(xyz.view(np.uint64), expected_xyz.view(np.uint64))
    np.testing.assert_array_equal(normal.view(np.uint64), expected_normal.view(np.uint64))


def test_normals_only_for_faces_that_cover_a_pixel(monkeypatch):
    # The second face has a candidate at (1,1) but no hits and collapsed XYZ.
    uv = np.array([[0, 0], [61, 0], [0, 61], [0, 0.6], [1.1, 1.7], [1.1, 2.0]])
    vertices = np.column_stack([uv, np.zeros(len(uv))])
    vertices[3:] = 0
    faces = np.arange(6).reshape(2, 3)
    rows = []
    cross = np.cross

    def measured(a, b):
        rows.append(len(a))
        return cross(a, b)

    monkeypatch.setattr(np, "cross", measured)
    _, normals, mask = native.rasterize([(vertices, faces, uv)], [0, 0], 1, (2, 62))
    assert mask[1, 1] == 1 and normals[1, 1, 2] == 1
    assert sum(rows) == 1
    uv[3:] = [[0, 0], [1, 0], [0, 1]]
    with pytest.raises(ValueError, match="degenerate raster geometry"):
        native.rasterize([(vertices, faces, uv)], [0, 0], 1, (2, 62))


def add_charts(store, width, height):
    _schema(store)
    uv = np.array([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0]])
    for i, points in enumerate((uv, uv * 0.3)):
        name = f"chart-{i}"
        uv_path = f"uv/{name}.npz"
        v = np.column_stack([points, np.full(3, i + 0.25)])
        store.save_arrays(f"charts/{name}.npz", vertices=v, faces=np.array([[0, 1, 2]]))
        store.save_arrays(uv_path, uv=points)
        with store.db:
            store.db.execute(
                "INSERT INTO charts VALUES (?,0,1.0,1,'MAPPED',?,0,?,0,?,NULL,'{}')",
                (name, uv_path, points[:, 0].max(), points[:, 1].max()),
            )
            for a, b, sign in ((0, 1, 1), (1, 2, 1), (0, 2, -1)):
                store.db.execute(
                    "INSERT INTO edges VALUES (?,?,?,?,?,?)",
                    (3 * i + a, 3 * i + b, name, a, b, sign),
                )


def samples(_, xyz):
    x, y = np.rint(xyz[:, :2]).astype(int).T
    return ((x + 11 * y) % 256).astype(np.float32), (x + y) % 7 != 0


def volume(ledger):
    return SimpleNamespace(
        identity={"ct": "analytic"},
        dtype=np.dtype("float32"),
        _ledger=ledger,
        validate_used=lambda: None,
        spacing_xyz=[1.0] * 3,
        origin_xyz=[0.0] * 3,
        shape=[8, 4102, 4102],
    )


@pytest.mark.parametrize(
    "width,height,pause_after_mask,tifxyz",
    [
        (4101, 17, False, False),
        (4101, 17, True, False),
        (17, 4101, False, False),
        (17, 4101, True, False),
        (269, 37, False, True),
    ],
)
def test_render_matches_analytic_pixels(
    tmp_path, monkeypatch, width, height, pause_after_mask, tifxyz
):
    if tifxyz:
        tifffile = pytest.importorskip("tifffile")
    monkeypatch.setattr(native, "sample_ct", samples)
    with Journal(tmp_path, {}) as store, closing(sqlite3.connect(":memory:")) as ledger:
        add_charts(store, width, height)
        if pause_after_mask:
            store.stop = lambda: any(tmp_path.glob("component-*/mask.png"))
            with pytest.raises(Paused, match="STOP"):
                native.render(store, volume(ledger), 1, tifxyz)
            store.stop = None
        record = native.render(store, volume(ledger), 1, tifxyz)["components"][0]

    # Both charts start at (0,0); the second has 3/10 of the first's side lengths.
    y, x = np.indices((height, width))
    line = x * (height - 1) + y * (width - 1)
    area = (width - 1) * (height - 1)
    conflict = 10 * line <= 3 * area
    valid = (line <= area) & ~conflict
    observed = valid & ((x + y) % 7 != 0)
    mask = np.where(conflict, 3, valid.astype(np.uint8) + observed)
    sheet = np.stack([np.where(valid, (x + 11 * y) % 256, 0), observed * 255], axis=-1)
    assert record["gray_window"] == [0.0, 255.0]
    keys = ("background", "unsupported_ct", "observed_ct", "ambiguous_xyz")
    assert record["counts"] == dict(zip(keys, np.bincount(mask.ravel(), minlength=4), strict=True))
    folder = tmp_path / record["path"]
    stride = (max(width, height) + 2047) // 2048
    for name, expected in (
        ("mask", mask),
        ("sheet", sheet),
        ("overview", sheet[::stride, ::stride]),
    ):
        with Image.open(folder / f"{name}.png") as image:
            np.testing.assert_array_equal(np.asarray(image), expected)
    if tifxyz:
        assert record["tifxyz"]["status"] == "WRITTEN"
        cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        used = np.zeros_like(valid)
        for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):
            used[dy : dy + height - 1, dx : dx + width - 1] |= cells
        for axis, coordinate in (("x", x), ("y", y), ("z", 0.25)):
            actual = tifffile.imread(folder / "tifxyz" / f"{axis}.tif")
            expected = np.where(used, coordinate, -1).astype(np.float32)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_png_publication_skips_geometry_decode_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "sample_ct", samples)
    with Journal(tmp_path, {}) as store, closing(sqlite3.connect(":memory:")) as ledger:
        add_charts(store, 549, 27)
        vol = volume(ledger)
        reads = Counter()
        cls = np.lib.npyio.NpzFile
        get = cls.__getitem__

        def measured(archive, key):
            if store.phase == "publish":
                reads[key] += 1
            return get(archive, key)

        monkeypatch.setattr(cls, "__getitem__", measured)
        for name in ("sheet.png", "mask.png"):
            store.stop = lambda name=name: any(tmp_path.glob("component-*/" + name))
            with pytest.raises(Paused, match="STOP"):
                native.render(store, vol, 1)
            published = next(tmp_path.glob("component-*/" + name))
            assert store.validate_file(str(published.relative_to(tmp_path))) == published
            assert not list(tmp_path.glob("component-*/overview.png"))
        sheet = next(tmp_path.glob("component-*/sheet.png"))
        mtime = sheet.stat().st_mtime_ns
        store.stop = None
        report = native.render(store, vol, 1)
        assert all(report["components"][0]["counts"].values())
        assert sheet.stat().st_mtime_ns == mtime
        assert set(reads) == {"ct", "mask"}


def test_pngs_stream_one_full_width_band_and_decode_ct_once(tmp_path, monkeypatch):
    with Journal(tmp_path, {}) as store:
        shape = (512, 769)
        mask = np.full((256, 256), 2, np.uint8)
        for y in range(0, shape[0], 256):
            for x in range(0, shape[1], 256):
                pixels = mask[:, : shape[1] - x]
                store.save_arrays(native._tile_name("c", y, x), mask=pixels, ct=pixels.astype("f4"))
        reads = []
        bands = []
        blocks = native._blocks
        get = np.lib.npyio.NpzFile.__getitem__

        def measured_blocks(*args, **kwargs):
            previous_band = None
            for y, band in blocks(*args, **kwargs):
                if previous_band is not None:
                    bands.append(np.shares_memory(previous_band, band))
                previous_band = band
                yield y, band

        def measured_get(archive, key):
            reads.append(key)
            return get(archive, key)

        monkeypatch.setattr(native, "_blocks", measured_blocks)
        monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", measured_get)
        native._publish_images(store, "c", shape, [0, 255])
        assert Counter(reads) == {"ct": 8, "mask": 16}
        assert bands and all(bands)


def test_publication_rechecks_entire_archive_between_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "sample_ct", samples)
    with Journal(tmp_path, {}) as store, closing(sqlite3.connect(":memory:")) as ledger:
        add_charts(store, 39, 29)
        register = store.register

        def corrupt_after_mask(name):
            register(name)
            if name.endswith("/mask.png"):
                tile = next(tmp_path.glob("component-*/tiles/*.npz"))
                with tile.open("r+b") as f:
                    f.seek(80)
                    byte = f.read(1)
                    f.seek(80)
                    f.write(bytes([byte[0] ^ 1]))

        monkeypatch.setattr(store, "register", corrupt_after_mask)
        with pytest.raises(ValueError, match="committed native payload changed"):
            native.render(store, volume(ledger), 1)
        assert not list(tmp_path.glob("component-*/overview.png"))


def test_resume_still_checks_readability_of_committed_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "sample_ct", samples)
    with Journal(tmp_path, {}) as store, closing(sqlite3.connect(":memory:")) as ledger:
        add_charts(store, 39, 29)
        store.stop = lambda: store.phase == "publish"
        with pytest.raises(Paused):
            native.render(store, volume(ledger), 1)
        store.stop = None
        (name,) = store.db.execute("SELECT name FROM rendered").fetchone()
        tile = store.load_arrays(name)
        # Even with a matching file receipt, a historical tile must still be readable
        # with allow_pickle=False before its statistics or pixels can be reused.
        tile["normal"] = np.array([None], dtype=object)
        np.savez_compressed(store.path(name), **tile)
        with store.db:
            store.db.execute("DELETE FROM files WHERE name=?", (name,))
        store.register(name)
        with pytest.raises(ValueError, match="allow_pickle=False"):
            native.render(store, volume(ledger), 1)
