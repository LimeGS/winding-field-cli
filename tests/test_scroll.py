import json
from argparse import Namespace

import numpy as np
import pytest

from winding_field import scroll


def test_tile_cores_partition_the_axis_and_tiles_add_overlap():
    tiles = scroll.tile_ranges(300, 128, 64)
    assert [t["core"] for t in tiles] == [(0, 63), (64, 191), (192, 299)]
    assert [t["nodes"] for t in tiles] == [(0, 127), (0, 255), (128, 299)]
    with pytest.raises(ValueError, match="multiples of 64"):
        scroll.tile_ranges(300, 100, 64)
    with pytest.raises(ValueError, match="multiples of 16"):
        scroll.tile_ranges(300, 128, 20)
    narrow = scroll.tile_ranges(300, 128, 32)
    assert [t["nodes"] for t in narrow] == [(0, 95), (32, 223), (160, 299)]


def test_tile_cores_share_the_axis_evenly_instead_of_leaving_a_narrow_remainder():
    # The dry-run box: 192-node cores used to give tiles of 224 and 96 nodes.
    tiles = scroll.tile_ranges(256, 192, 32)
    assert [t["core"] for t in tiles] == [(0, 127), (128, 255)]
    assert [t["nodes"] for t in tiles] == [(0, 159), (96, 255)]
    # Whole multiples of the core are unchanged.
    assert [t["core"] for t in scroll.tile_ranges(384, 128, 32)] == [
        (0, 127),
        (128, 255),
        (256, 383),
    ]
    # No core exceeds the requested one, and cores stay on 64-node chunks.
    for nodes in range(2, 1200, 37):
        for core in (64, 128, 192, 320):
            cores = [t["core"] for t in scroll.tile_ranges(nodes, core, 32)]
            assert cores[0][0] == 0 and cores[-1][1] == nodes - 1
            assert len(cores) == -(-nodes // core)
            assert all(b - a + 1 <= core for a, b in cores)
            assert all(a % 64 == 0 for a, _ in cores)
            assert all(c[1] + 1 == d[0] for c, d in zip(cores, cores[1:]))


def test_global_grid_counts_nodes_on_the_field_pitch():
    grid = scroll.global_grid([4, 4 + 4 * 10 + 1, 0, 9, 8, 100], 4, 8.64, [1.0, 2.0, 3.0])
    assert grid["shape_zyx"] == [11, 3, 23]
    assert grid["spacing_um"] == pytest.approx(34.56)
    assert grid["origin_zyx_um"] == pytest.approx([3.0 + 4 * 8.64, 2.0, 1.0 + 8 * 8.64])
    with pytest.raises(ValueError, match="whole multiple"):
        scroll._integer_ratio(34.56, 9.36)
    assert scroll._integer_ratio(34.56, 8.64) == 4


GRID = dict(
    voxel_lo_zyx=[0, 0, 0],
    ratio=1,
    shape_zyx=[2, 3, 320],
    origin_zyx_um=[0.0, 0.0, 0.0],
    spacing_um=2.0,
)


def truth(x):
    return x / 160


def write_tile(root, tile, bias):
    from field_store import BlockStore
    from winding_field.artifacts import Tiles
    from winding_field.util import atomic_json, file_sha

    run = root / "tiles" / tile["id"]
    run.mkdir(parents=True)
    lo, hi = np.asarray(tile["nodes_lo"]), np.asarray(tile["nodes_hi"])
    shape = (hi - lo + 1).tolist()
    geometry = dict(
        frame="test-frame",
        voxel_um=1.0,
        origin_xyz=(lo[::-1] * GRID["spacing_um"]).astype(float).tolist(),
        spacing_xyz=[GRID["spacing_um"]] * 3,
        shape_zyx=shape,
        sheet_step=1.0,
    )
    store = BlockStore(run / "field", shape, 64, dict(geometry=geometry))
    with store:
        for key in np.ndindex(*[(n - 1) // 64 + 1 for n in shape]):
            first = np.asarray(key) * 64
            last = np.minimum(first + 64, shape)
            x = np.arange(first[2], last[2]) + lo[2]
            values = np.broadcast_to(truth(x) + bias, tuple(last - first)).copy()
            store.commit(key, values, {}, observed=np.ones(values.shape, bool))
    atomic_json(
        run / "FIELD.json",
        dict(geometry=geometry, checkpoint_sha256=file_sha(run / "field/CHECKPOINT.json")),
    )
    atomic_json(
        run / "PREPARED.json",
        dict(
            surface_method="lasagna",
            origin_xyz=geometry["origin_xyz"],
            spacing_xyz=geometry["spacing_xyz"],
            shape_zyx=shape,
        ),
    )
    maps = Tiles(run / "maps", shape, 64, identity="test")
    for key in maps.keys():
        a, b = maps.box(key)
        x = np.arange(a[2], b[2]) + lo[2]
        # Phase peaks where truth + offset 0.25 is an integer: cos(2π(φ - 0.25)).
        phase = (np.cos(2 * np.pi * (truth(x) - 0.25)) + 1) / 2
        maps.write(key, dict(surface_phase=np.broadcast_to(phase, tuple(b - a)).astype(np.float32)))
    maps.close()
    atomic_json(run / "RUN.json", dict(stages=dict(field={})))


def plan_two_biased_tiles(tmp_path, bias=(0.0, 0.004, 0.0), overlap=64):
    from winding_field.util import atomic_json

    tiles = []
    # Fixed 128-node cores (the last one 64): these merge tests need tiles that only the
    # first tile covers, which the balanced tile_ranges split no longer gives.
    cores = [(0, 127), (128, 255), (256, 319)]
    ranges = [dict(core=c, nodes=(max(0, c[0] - overlap), min(319, c[1] + overlap))) for c in cores]
    for k, t in enumerate(ranges):
        tiles.append(
            dict(
                id=f"000-000-{k:03d}",
                key=[0, 0, k],
                nodes_lo=[0, 0, t["nodes"][0]],
                nodes_hi=[1, 2, t["nodes"][1]],
                core_lo=[0, 0, t["core"][0]],
                core_hi=[1, 2, t["core"][1]],
                roi=[],
                supported=True,
            )
        )
    root = tmp_path / "plan"
    root.mkdir()
    coarse = write_coarse(tmp_path / "coarse")
    atomic_json(
        root / "TILES.json",
        dict(
            schema=scroll.PLAN_SCHEMA,
            grid=GRID,
            overlap_nodes=overlap,
            tiles=tiles,
            run_args=[],
            coarse=coarse,
        ),
    )
    for tile, b in zip(tiles, bias):
        write_tile(root, tile, b)
    return root, tiles


def write_coarse(run):
    """The exact field on a 2x coarser grid, as a completed field run."""
    from field_store import BlockStore
    from winding_field.util import atomic_json, file_sha

    shape = [2, 2, 161]
    geometry = dict(
        frame="test-frame",
        voxel_um=1.0,
        origin_xyz=[0.0, 0.0, 0.0],
        spacing_xyz=[4.0, 4.0, 4.0],
        shape_zyx=shape,
        sheet_step=1.0,
    )
    run.mkdir()
    with BlockStore(run / "field", shape, 64, dict(geometry=geometry)) as store:
        for key in np.ndindex(*[(n - 1) // 64 + 1 for n in shape]):
            first = np.asarray(key) * 64
            last = np.minimum(first + 64, shape)
            x_um = np.arange(first[2], last[2]) * 4.0
            values = np.broadcast_to(truth(x_um / GRID["spacing_um"]), tuple(last - first))
            store.commit(key, values.copy(), {}, observed=np.ones(tuple(last - first), bool))
    sha = file_sha(run / "field/CHECKPOINT.json")
    atomic_json(run / "FIELD.json", dict(geometry=geometry, checkpoint_sha256=sha))
    return dict(path=str(run), checkpoint_sha256=sha)


def test_coarse_correction_removes_each_tiles_own_drift(tmp_path):
    from winding_field.field import open_field

    # Each tile is offset from the truth by its own constant, as a weakly gauged tile is.
    root, tiles = plan_two_biased_tiles(tmp_path, bias=(-0.2, 0.3, 0.1))
    raw = scroll.merge(
        Namespace(
            plan=root,
            output=tmp_path / "raw",
            workers=1,
            allow_missing=False,
            coarse_sigma_mm=0,
            join="none",
        )
    )
    assert raw["worst_seam_p95_abs_dW"] >= 0.5
    result = scroll.merge(
        Namespace(
            plan=root,
            output=tmp_path / "merged",
            workers=1,
            allow_missing=False,
            coarse_sigma_mm=0.5,
            join="coarse",
        )
    )
    with open_field(tmp_path / "merged") as source:
        values, valid = source.read([0, 0, 0], GRID["shape_zyx"])
    assert valid.all()
    np.testing.assert_allclose(
        values, np.broadcast_to(truth(np.arange(320)), values.shape), atol=1e-5
    )
    assert result["worst_seam_p95_abs_dW"] <= 0.002
    meta = json.loads((tmp_path / "merged/FIELD.json").read_text())
    assert meta["coarse_correction"]["p95_abs_correction"] == pytest.approx(
        {"000-000-000": 0.2, "000-000-001": 0.3, "000-000-002": 0.1}, abs=1e-6
    )


def test_overlap_join_levels_tiles_against_each_other_without_the_coarse(tmp_path):
    from winding_field.field import open_field

    # Each tile is offset by its own constant; the coarse is not consulted.
    root, tiles = plan_two_biased_tiles(tmp_path, bias=(-0.2, 0.3, 0.1))
    result = scroll.merge(
        Namespace(plan=root, output=tmp_path / "merged", workers=1, allow_missing=False)
    )
    with open_field(tmp_path / "merged") as source:
        values, valid = source.read([0, 0, 0], GRID["shape_zyx"])
    assert valid.all()
    error = values - truth(np.arange(320))
    # Agreement everywhere; the common level is the tiles' mean offset.
    assert np.ptp(error) <= 1e-5
    assert error.mean() == pytest.approx(np.mean([-0.2, 0.3, 0.1]), abs=1e-5)
    assert result["worst_seam_p95_abs_dW"] <= 0.002
    meta = json.loads((tmp_path / "merged/FIELD.json").read_text())
    assert meta["join"]["method"] == "overlap" and meta["join"]["pairs"] == 2
    assert meta["join"]["rms_before"]["worst"] == pytest.approx(0.5, abs=1e-6)
    assert meta["join"]["rms_after"]["worst"] <= 1e-6
    assert meta["coarse_correction"] is None


def test_overlap_join_with_tiles_off_the_chunk_grid(tmp_path):
    from winding_field.field import open_field

    # 32-node overlaps: tiles start between 64-node chunks of the merged grid.
    root, tiles = plan_two_biased_tiles(tmp_path, bias=(-0.2, 0.3, 0.1), overlap=32)
    assert tiles[1]["nodes_lo"][2] % 64
    result = scroll.merge(
        Namespace(plan=root, output=tmp_path / "merged", workers=1, allow_missing=False)
    )
    with open_field(tmp_path / "merged") as source:
        values, valid = source.read([0, 0, 0], GRID["shape_zyx"])
    assert valid.all()
    error = values - truth(np.arange(320))
    assert np.ptp(error) <= 1e-5
    assert result["worst_seam_p95_abs_dW"] <= 0.002


def test_unknown_join_is_refused(tmp_path):
    root, tiles = plan_two_biased_tiles(tmp_path)
    with pytest.raises(ValueError, match="--join"):
        scroll.merge(
            Namespace(plan=root, output=tmp_path / "m", workers=1, allow_missing=False, join="best")
        )


def test_merge_blends_tiles_into_one_readable_field_and_reports_seams(tmp_path):
    from winding_field.field import open_field

    root, tiles = plan_two_biased_tiles(tmp_path)
    out = tmp_path / "merged"
    result = scroll.merge(
        Namespace(
            plan=root, output=out, workers=1, allow_missing=False, coarse_sigma_mm=0, join="none"
        )
    )
    assert result["status"] == "MERGED" and result["tiles"] == 3

    with open_field(out) as source:
        values, valid = source.read([0, 0, 0], GRID["shape_zyx"])
    assert valid.all()
    x = np.arange(320)
    error = values - truth(x)
    # Nodes only the unbiased first tile covers are exact; elsewhere the biased middle
    # tile is cross-faded in, never beyond its own bias.
    np.testing.assert_allclose(error[..., :64], 0, atol=1e-6)
    assert error.min() >= -1e-6 and error.max() <= 0.004 + 1e-6
    assert error.max() > 0.003

    seams = json.loads((out / "SEAMS.json").read_text())
    pair = seams["pairs"]["000-000-000|000-000-001"]
    assert pair["nodes"] > 0 and pair["over_half_winding"] == 0
    assert pair["median_abs_dW"] == pytest.approx(0.004, abs=0.0011)

    levels = json.loads((out / "LEVELS.json").read_text())
    assert levels["phase_offset"] == pytest.approx(0.25, abs=0.01)
    assert levels["levels"] == pytest.approx([0.25, 1.25], abs=0.01)

    surface = json.loads((out / "surface-run.json").read_text())
    assert surface["field"]["kind"] == "chunks" and surface["levels"] == levels["levels"]


def test_merge_refuses_tiles_off_the_plan_grid(tmp_path):
    root, tiles = plan_two_biased_tiles(tmp_path)
    field = root / "tiles" / tiles[1]["id"] / "FIELD.json"
    meta = json.loads(field.read_text())
    meta["geometry"]["origin_xyz"][0] += 1.0
    field.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not on the plan grid"):
        scroll.merge(
            Namespace(
                plan=root,
                output=tmp_path / "merged",
                workers=1,
                allow_missing=False,
                coarse_sigma_mm=0,
                join="none",
            )
        )


def test_merge_requires_every_supported_tile_unless_allowed(tmp_path):
    root, tiles = plan_two_biased_tiles(tmp_path)
    (root / "tiles" / tiles[2]["id"] / "RUN.json").write_text("{}")
    args = Namespace(
        plan=root,
        output=tmp_path / "merged",
        workers=1,
        allow_missing=False,
        coarse_sigma_mm=0,
        join="none",
    )
    with pytest.raises(ValueError, match="no completed field"):
        scroll.merge(args)
    args.allow_missing = True
    result = scroll.merge(args)
    assert result["missing_tiles"] == 1


def test_render_ct_uses_the_requested_pitch_not_the_mesh_spacing(tmp_path, monkeypatch):
    from test_native_geometry import plane_bundle
    from test_native_render import analytic_ct
    import winding_field.workflow as workflow

    bundle = plane_bundle(tmp_path)
    volume = analytic_ct(tmp_path)
    monkeypatch.setattr(workflow, "open_volume", lambda config: volume.thing)
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
            "--output",
            str(tmp_path / "out"),
        ]  # fmt: skip
    )
    result = scroll.render_ct(args)
    assert result["status"] == "COMPLETE"
    level = json.loads((tmp_path / "out" / "LEVEL.json").read_text())
    assert level["status"] == "COMPLETE"
    identity = json.loads((tmp_path / "out" / "IDENTITY.json").read_text())
    assert identity["pitch_um"] == 0.5


def test_split_afv_keeps_whole_fibres_touching_each_box_and_stays_importable(tmp_path):
    import sqlite3

    from test_splines import prepared, upstream_curves
    from winding_field import afv

    maps, meta = prepared(tmp_path, shape=(8, 8, 24), chunk=8)
    maps.close()
    path = tmp_path / "scroll.afv"
    # (family, points): fibres 1, 2, 3 span x = 1-3, 5-7 and 9-11 voxels.
    fibres = [
        (1, np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]])),
        (2, np.array([[5.0, 2.0, 3.0], [7.0, 2.0, 3.0]])),
        (1, np.array([[9.0, 2.0, 3.0], [11.0, 2.0, 3.0]])),
    ]
    upstream_curves(path, (8, 8, 24), fibres)
    frame = dict(
        source=meta["source"],
        coordinate_order="XYZ",
        units="selected-source voxels",
        level=0,
        origin_xyz_um=meta["origin_xyz"],
        spacing_xyz_um=meta["spacing_xyz"],
    )
    with sqlite3.connect(path) as db:
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("frame", json.dumps(frame)))
    origin = np.asarray(meta["origin_xyz"])
    spacing = np.asarray(meta["spacing_xyz"])
    boxes = {
        "left": (origin + spacing * [0, 0, 0], origin + spacing * [8, 7, 7]),
        "right": (origin + spacing * [8.5, 0, 0], origin + spacing * [23, 7, 7]),
    }
    before = path.read_bytes()
    out = scroll.split_afv(path, boxes, tmp_path / "split", margin_um=0.0)
    assert path.read_bytes() == before
    assert {k: v["fibres"] for k, v in out.items()} == {"left": 2, "right": 1}
    for name, ids in (("left", [1, 2]), ("right", [3])):
        with sqlite3.connect(out[name]["path"]) as db:
            assert [r[0] for r in db.execute("SELECT id FROM fibers ORDER BY id")] == ids
            split = json.loads(
                db.execute("SELECT value FROM metadata WHERE key='scroll_split'").fetchone()[0]
            )
        assert split["tile"] == name
        identity = afv.source_identity(out[name]["path"], None, meta["source"])
        assert identity["input_count"] == len(ids)


def test_tiled_render_writes_the_same_pixels_as_whole_png_and_drops_xyz(tmp_path):
    from PIL import Image
    from test_native_geometry import plane_bundle
    from test_native_render import analytic_ct
    from winding_field.native_pipeline import render_mesh
    from winding_field.surface_mesh import MeshReader

    bundle = plane_bundle(tmp_path, nx=40, ny=30)
    config = {"chart_faces": 64, "render_pitch_um": 0.1}
    with MeshReader(bundle) as reader, analytic_ct(tmp_path) as volume:
        whole = render_mesh(reader, tmp_path / "whole", volume, config)
        tiled = render_mesh(
            reader,
            tmp_path / "tiled",
            volume,
            dict(config, render_tiled=True, render_keep_xyz=False),
        )
    assert whole["status"] == tiled["status"] == "COMPLETE"
    component = whole["components"][0]["path"]
    sheet = np.asarray(Image.open(tmp_path / "whole" / component / "sheet.png"))
    assert sheet.shape[0] > 256 and sheet.shape[1] > 256  # several tiles per axis
    assert not (tmp_path / "tiled" / component / "sheet.png").exists()
    rebuilt = np.zeros_like(sheet)
    for png in (tmp_path / "tiled" / component / "png").iterdir():
        y, x = (int(v) for v in png.stem.split("-"))
        tile = np.asarray(Image.open(png))
        rebuilt[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    np.testing.assert_array_equal(rebuilt, sheet)
    for name in ("overview.png",):
        assert (tmp_path / "tiled" / component / name).read_bytes() == (
            tmp_path / "whole" / component / name
        ).read_bytes()
    stored = next((tmp_path / "tiled" / component / "tiles").glob("*.npz"))
    with np.load(stored) as data:
        assert set(data.files) == {"mask", "ct"}


def test_decimate_bundle_reports_deviation_and_keeps_a_plane_exact(tmp_path):
    from test_native_geometry import plane_bundle

    bundle = plane_bundle(tmp_path, nx=20, ny=20)
    mesh, receipt = scroll.decimate_bundle(bundle, 4, tmp_path / "decimated")
    assert receipt["faces"] <= receipt["source_faces"] // 4 + 2
    assert receipt["deviation_um"]["max"] < 1e-9
    assert (mesh / "MESH.json").is_file()


def test_mesh_ct_chunks_cover_every_sampled_voxel(tmp_path):
    from types import SimpleNamespace

    from test_native_geometry import plane_bundle
    from winding_field.surface_mesh import MeshReader

    bundle = plane_bundle(tmp_path, nx=9, ny=7)
    volume = SimpleNamespace(
        origin_xyz=[0.0, 0.0, 0.0],
        spacing_xyz=[1.0, 1.0, 1.0],
        chunks=(4, 4, 4),
        shape=(12, 12, 16),
    )
    with MeshReader(bundle) as reader:
        keys = set(scroll.mesh_ct_chunks(reader, volume))
    # The plane spans x 2..10, y 3..9 at z = 4; with one voxel of margin that is
    # z 3..5, y 2..10, x 1..11 -> chunks z 0..1, y 0..2, x 0..2.
    expected = {(z, y, x) for z in range(2) for y in range(3) for x in range(3)}
    assert keys == expected


def test_build_pyramid_keeps_full_resolution_tiles_and_halves_each_level(tmp_path):
    from PIL import Image

    component = tmp_path / "component"
    (component / "png").mkdir(parents=True)
    height, width = 300, 600
    (component / "RENDER.json").write_text(
        json.dumps(dict(grid=dict(shape_yx=[height, width], tile_pixels=256)))
    )
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (height, width, 2), dtype=np.uint8)
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            Image.fromarray(image[y : y + 256, x : x + 256], "LA").save(
                component / "png" / f"{y:09d}-{x:09d}.png"
            )
    result = scroll.build_pyramid(component, tmp_path / "out", "w")
    assert result == dict(dzi="w.dzi", width=width, height=height, tile=256)
    files = tmp_path / "out" / "w_files"
    top = 10  # ceil(log2(600))
    assert sorted(int(p.name) for p in files.iterdir()) == list(range(top + 1))
    for name in (files / str(top)).iterdir():
        col, row = (int(v) for v in name.stem.split("_"))
        np.testing.assert_array_equal(
            np.asarray(Image.open(name)),
            image[row * 256 : row * 256 + 256, col * 256 : col * 256 + 256],
        )
    w, h = width, height
    for level in range(top - 1, -1, -1):
        w, h = -(-w // 2), -(-h // 2)
        with Image.open(files / str(level) / "0_0.png") as tile:
            assert tile.size == (min(256, w), min(256, h))
    assert 'Width="600" Height="300"' in (tmp_path / "out" / "w.dzi").read_text()


def test_tiles_use_a_shared_cache_when_given(tmp_path):
    import sys

    from winding_field.util import atomic_json

    root = tmp_path / "plan"
    root.mkdir()
    tiles = [
        dict(id=f"000-000-00{k}", supported=True, roi=[0, 2, 0, 2, k, k + 2]) for k in range(2)
    ]
    atomic_json(
        root / "TILES.json",
        dict(schema=scroll.PLAN_SCHEMA, tiles=tiles, run_args=["--fibres", "full.afv"]),
    )
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import json, sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.mkdir(parents=True)\n"
        "(out / 'ARGS.json').write_text(json.dumps(sys.argv[1:]))\n"
        "(out / 'RUN.json').write_text(json.dumps({'stages': {'field': {}}}))\n"
        "(out / 'FIELD.json').write_text('{}')\n"
    )
    shared = tmp_path / "shared-cache"
    result = scroll.run_tiles(
        scroll._parser().parse_args(
            [
                "tiles",
                str(root),
                "--jobs",
                "2",
                "--cache",
                str(shared),
                "--executable",
                sys.executable,
                str(fake),
            ]
        )  # fmt: skip
    )
    assert result["status"] == "TILES_DONE" and result["DONE"] == 2
    for tile in tiles:
        args = json.loads((root / "tiles" / tile["id"] / "ARGS.json").read_text())
        assert args[args.index("--cache") + 1] == str(shared.resolve())


def test_seam_percentile_in_the_open_last_bin_is_finite_json():
    import json

    import numpy as np

    from winding_field import scroll

    hist = np.zeros(len(scroll.SEAM_BINS) - 1, np.int64)
    hist[0], hist[-1] = 10, 90  # most seam nodes differ by more than 4 windings
    p95 = scroll._percentile(hist, 0.95)
    assert p95 == 4.0
    json.dumps({"p95_abs_dW": p95}, allow_nan=False)
    assert scroll._percentile(hist, 0.05) == float(scroll.SEAM_BINS[1])


def _npz_bundle(tmp_path, vertices, faces):
    from winding_field.surface_mesh import import_mesh
    from winding_field.util import file_sha

    path = tmp_path / "source.npz"
    np.savez(path, vertices_xyz=np.asarray(vertices, float), faces=np.asarray(faces))
    meta = dict(frame="test", origin_xyz=[0.0, 0.0, 0.0], spacing_xyz=[1.0, 1.0, 1.0],
                shape_zyx=[40, 40, 40], voxel_um=1.0, sheet_step=1.0)  # fmt: skip
    import_mesh({"path": str(path), "sha256": file_sha(path)}, tmp_path / "mesh", meta)
    return tmp_path / "mesh"


def _grid(n, origin, first=0):
    x, y = np.meshgrid(np.arange(float(n)), np.arange(float(n)))
    vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]) + origin
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = first + row * n + col
            faces.extend([(a, a + 1, a + n), (a + 1, a + n + 1, a + n)])
    return vertices, np.array(faces)


def test_decimation_splits_a_vertex_shared_by_two_sheets_and_reuses_its_output(tmp_path):
    import igl

    n = 12
    va, fa = _grid(n, [2.0, 2.0, 5.0])
    vb, fb = _grid(n, [2.0 + n - 1, 2.0 + n - 1, 5.0], first=n * n)
    # The two sheets touch at one corner: make them share that vertex (a bow-tie).
    fb = np.where(fb == n * n, n * n - 1, fb)
    vertices, faces = np.concatenate([va, vb]), np.concatenate([fa, fb])
    refused = igl.qslim(np.asfortranarray(vertices), np.asfortranarray(faces.astype(np.int32)), 60)
    assert len(refused[1]) == 0  # libigl refuses the raw mesh
    bundle = _npz_bundle(tmp_path, vertices, faces)
    mesh, receipt = scroll.decimate_bundle(bundle, 4, tmp_path / "decimated")
    assert 0 < receipt["faces"] <= len(faces) // 4 + 2
    assert receipt["split_vertex_fans"] == 1 and receipt["dropped_degenerate_faces"] == 0
    assert receipt["deviation_um"]["max"] < 1e-9
    again = scroll.decimate_bundle(bundle, 4, tmp_path / "decimated")
    assert again == (mesh, receipt)
    (tmp_path / "decimated/DECIMATION.json").unlink()  # an interrupted attempt is redone
    assert scroll.decimate_bundle(bundle, 4, tmp_path / "decimated")[1]["faces"] == receipt["faces"]


def test_decimation_refuses_a_mesh_with_a_non_manifold_edge(tmp_path):
    vertices, faces = _grid(8, [2.0, 2.0, 5.0])
    fin = len(vertices)
    vertices = np.concatenate([vertices, [[2.5, 2.5, 9.0]]])
    faces = np.concatenate([faces, [[0, 1, fin]]])  # a third face on the edge 0-1
    bundle = _npz_bundle(tmp_path, vertices, faces)
    with pytest.raises(scroll.DecimationRefused):
        scroll.decimate_bundle(bundle, 4, tmp_path / "decimated")
    assert not (tmp_path / "decimated").exists()


def test_auto_tile_size_grows_with_gpu_memory_and_fits_the_measured_tile():
    sizes = {}
    for gib in (6, 16, 24, 32, 48, 80, 96, 192):
        core, sizing = scroll.auto_tile_nodes(gib * 2**30, 34.56, 64)
        assert core % 64 == 0 and sizing["tile_nodes_per_axis"] == core + 128
        assert sizing["estimated_peak_bytes"] <= scroll.GPU_USABLE * gib * 2**30
        sizes[gib] = core
    assert list(sizes.values()) == sorted(sizes.values())
    # A 6 GB GTX 1660 ran the 256^3 PHerc0175A tile (3.1 GB peak); the model keeps it.
    assert sizes[6] == 128
    _, sizing = scroll.auto_tile_nodes(6 * 2**30, 34.56, 64)
    assert 3.1e9 <= sizing["estimated_peak_bytes"] <= 6 * 2**30
    assert sizes[80] >= 512
    # Two tiles per GPU get half the memory each.
    assert scroll.auto_tile_nodes(80 * 2**30 / 2, 34.56, 64)[0] < sizes[80]
    with pytest.raises(ValueError, match="cannot hold"):
        scroll.auto_tile_nodes(2**30, 34.56, 64)


def test_auto_tile_size_prefers_amg_down_to_its_floor_then_jacobi():
    # A 6 GB GTX 1660 holds a 192^3 AMG tile (the real PHerc0175A tile peaked at 4.37 GB).
    core, sizing = scroll.auto_tile_size(6 * 2**30, 34.56, 32)
    assert sizing["preconditioner"] == "amg" and core == 128
    assert sizing["tile_nodes_per_axis"] == scroll.AMG_MIN_TILE_NODES
    assert 4.37e9 <= sizing["estimated_peak_bytes"] <= scroll.GPU_USABLE * 6 * 2**30
    # Below the floor AMG is not worth its setup: Jacobi with its own, larger tile.
    core, sizing = scroll.auto_tile_size(4 * 2**30, 34.56, 32)
    assert sizing["preconditioner"] == "jacobi"
    assert (core, sizing) == scroll.auto_tile_nodes(4 * 2**30, 34.56, 32)
    # AMG tiles are smaller than Jacobi's on the same GPU (its hierarchy needs memory).
    amg, sizing = scroll.auto_tile_size(80 * 2**30, 34.56, 32)
    assert sizing["preconditioner"] == "amg"
    assert amg < scroll.auto_tile_nodes(80 * 2**30, 34.56, 32)[0]
    assert sizing["estimated_peak_bytes"] <= scroll.GPU_USABLE * 80 * 2**30
    # Forcing one preconditioner sizes with its model only.
    assert scroll.auto_tile_size(80 * 2**30, 34.56, 32, "jacobi") == scroll.auto_tile_nodes(
        80 * 2**30, 34.56, 32
    )
    with pytest.raises(ValueError, match="cannot hold"):
        scroll.auto_tile_size(2**30, 34.56, 32, "amg")
    with pytest.raises(ValueError, match="preconditioner"):
        scroll.auto_tile_size(6 * 2**30, 34.56, 32, "spectral")


class _FakeVolume:
    spacing_xyz = (1.0, 1.0, 1.0)
    shape = (4, 6, 640)
    origin_xyz = (0.0, 0.0, 0.0)
    identity = dict(test="volume")

    def close(self):
        pass


def test_auto_plan_sizes_tiles_for_the_gpu_and_solves_each_as_one_block(tmp_path, monkeypatch):
    from winding_field import workflow

    monkeypatch.setattr(workflow, "open_volume", lambda config: _FakeVolume())
    coarse = write_coarse(tmp_path / "coarse")
    args = scroll._parser().parse_args(
        [
            "plan", "--source", "ct.zarr", "--voxel-um", "1", "--field-spacing-um", "2",
            "--coarse", coarse["path"], "--fibres", "f.afv", "--lasagna-input", "l.json",
            "--gpu-memory-gib", "2", "--no-split-fibres", "--run-arg=--fibre-weight=1000",
            "--output", str(tmp_path / "plan"),
        ]
    )  # fmt: skip
    result = scroll.plan(args)
    document = json.loads((tmp_path / "plan/TILES.json").read_text())
    memory = document["tile_memory"]
    assert memory["mode"] == "auto" and memory["core_nodes"] == 128
    assert document["tile_nodes"] == 128 and document["overlap_nodes"] == 32
    assert result["tiles"] == 3
    run_args = document["run_args"]
    # The largest tile (128 core + 2 x 32 overlap along x) is one solver block.
    assert run_args[run_args.index("--field-chunk") + 1] == "192"
    assert run_args[run_args.index("--sweeps") + 1] == "1"
    assert run_args[run_args.index("--maxiter") + 1] == "50000"
    # 2 GiB is below the AMG floor: Jacobi, the solver's default, so no flag.
    assert memory["preconditioner"] == "jacobi" and "--preconditioner" not in run_args
    # Extra run flags come last, so they override the defaults above.
    assert run_args[-1] == "--fibre-weight=1000"


def _plan_args(tmp_path, coarse, *extra):
    return scroll._parser().parse_args(
        [
            "plan", "--source", "ct.zarr", "--voxel-um", "1", "--field-spacing-um", "2",
            "--coarse", coarse["path"], "--fibres", "f.afv", "--lasagna-input", "l.json",
            "--no-split-fibres", *extra, "--output", str(tmp_path / "plan"),
        ]
    )  # fmt: skip


def test_auto_preconditioner_solves_tiles_above_the_floor_with_amg(tmp_path, monkeypatch):
    from winding_field import workflow

    monkeypatch.setattr(workflow, "open_volume", lambda config: _FakeVolume())
    coarse = write_coarse(tmp_path / "coarse")
    scroll.plan(
        _plan_args(tmp_path, coarse, "--gpu-memory-gib", "8", "--tile-preconditioner", "auto")
    )
    document = json.loads((tmp_path / "plan/TILES.json").read_text())
    assert document["tile_memory"]["preconditioner"] == "amg"
    run_args = document["run_args"]
    assert run_args[run_args.index("--preconditioner") + 1] == "amg"
    assert run_args[run_args.index("--field-chunk") + 1] == "192"


def test_fixed_tiles_use_jacobi_unless_amg_is_forced(tmp_path, monkeypatch):
    from winding_field import workflow

    monkeypatch.setattr(workflow, "open_volume", lambda config: _FakeVolume())
    coarse = write_coarse(tmp_path / "coarse")
    scroll.plan(
        _plan_args(tmp_path, coarse, "--tile-nodes", "128", "--tile-preconditioner", "auto")
    )
    document = json.loads((tmp_path / "plan/TILES.json").read_text())
    assert document["tile_memory"] is None and "--preconditioner" not in document["run_args"]
    forced = tmp_path / "forced"
    forced.mkdir()
    scroll.plan(_plan_args(forced, coarse, "--tile-nodes", "128", "--tile-preconditioner", "amg"))
    run_args = json.loads((forced / "plan/TILES.json").read_text())["run_args"]
    assert run_args[run_args.index("--preconditioner") + 1] == "amg"


def test_auto_plan_without_gpu_or_memory_is_refused(tmp_path, monkeypatch):
    from winding_field import workflow

    monkeypatch.setattr(workflow, "open_volume", lambda config: _FakeVolume())
    monkeypatch.setattr(scroll, "visible_gpus", lambda: [])
    coarse = write_coarse(tmp_path / "coarse")
    args = scroll._parser().parse_args(
        [
            "plan", "--source", "ct.zarr", "--voxel-um", "1", "--field-spacing-um", "2",
            "--coarse", coarse["path"], "--fibres", "f.afv", "--lasagna-input", "l.json",
            "--no-split-fibres", "--output", str(tmp_path / "plan"),
        ]
    )  # fmt: skip
    with pytest.raises(ValueError, match="gpu-memory-gib"):
        scroll.plan(args)


def test_tiles_run_one_process_per_gpu_each_seeing_only_its_gpu(tmp_path, monkeypatch):
    import sys

    from winding_field.util import atomic_json

    monkeypatch.setattr(scroll, "visible_gpus", lambda: [("3", 8 << 30), ("5", 8 << 30)])
    root = tmp_path / "plan"
    root.mkdir()
    tiles = [
        dict(id=f"000-000-00{k}", supported=True, roi=[0, 2, 0, 2, k, k + 2]) for k in range(4)
    ]
    atomic_json(
        root / "TILES.json",
        dict(
            schema=scroll.PLAN_SCHEMA,
            tiles=tiles,
            run_args=["--device", "cuda"],
            tile_memory=dict(mode="auto", jobs_per_gpu=1),
        ),
    )
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import json, os, sys, pathlib, time\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.mkdir(parents=True)\n"
        "start = time.time(); time.sleep(1.5)\n"
        "(out / 'GPU.json').write_text(json.dumps(\n"
        "    [os.environ.get('CUDA_VISIBLE_DEVICES'), start, time.time()]))\n"
        "(out / 'RUN.json').write_text(json.dumps({'stages': {'field': {}}}))\n"
        "(out / 'FIELD.json').write_text('{}')\n"
    )
    result = scroll.run_tiles(
        scroll._parser().parse_args(["tiles", str(root), "--executable", sys.executable, str(fake)])
    )
    assert result["status"] == "TILES_DONE" and result["jobs"] == 2
    assert result["gpus"] == ["3", "5"]
    runs = [json.loads((root / "tiles" / t["id"] / "GPU.json").read_text()) for t in tiles]
    assert {gpu for gpu, _, _ in runs} == {"3", "5"}
    for gpu in ("3", "5"):
        spans = sorted((a, b) for g, a, b in runs if g == gpu)
        assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_tiles_stopped_by_the_plan_until_are_prepared_and_solve_on_the_next_run(tmp_path):
    import sys

    from winding_field.util import atomic_json

    root = tmp_path / "plan"
    root.mkdir()
    tiles = [dict(id=f"000-000-00{k}", supported=True, roi=[0, 2, 0, 2, k, k + 2]) for k in (0, 1)]
    atomic_json(
        root / "TILES.json",
        dict(
            schema=scroll.PLAN_SCHEMA,
            tiles=tiles,
            # The plan's own --until field, then --run-arg=--until=fibres: the last one wins.
            run_args=["--device", "cpu", "--until", "field", "--until=fibres"],
        ),
    )
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import json, sys, pathlib\n"
        "if sys.argv[1] == 'field':\n"
        "    out = pathlib.Path(sys.argv[2])\n"
        "    (out / 'RUN.json').write_text(json.dumps({'stages': {'fibres': {}, 'field': {}}}))\n"
        "    (out / 'FIELD.json').write_text('{}')\n"
        "else:\n"
        "    out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "    out.mkdir(parents=True)\n"
        "    (out / 'RUN.json').write_text(json.dumps({'stages': {'fibres': {}}}))\n"
    )
    args = scroll._parser().parse_args(
        ["tiles", str(root), "--executable", sys.executable, str(fake)]
    )
    first = scroll.run_tiles(args)
    assert first["status"] == "TILES_PREPARED" and first["PREPARED"] == 2 and first["FAILED"] == 0
    assert json.loads((root / "status/000-000-000.json").read_text())["status"] == "PREPARED"
    second = scroll.run_tiles(args)
    assert second["status"] == "TILES_DONE" and second["DONE"] == 2
