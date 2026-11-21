import json
from argparse import Namespace

import numpy as np
import pytest

from winding_field import scroll


def test_tile_cores_partition_the_axis_and_tiles_add_overlap():
    tiles = scroll.tile_ranges(300, 128, 64)
    assert [t["core"] for t in tiles] == [(0, 127), (128, 255), (256, 299)]
    assert [t["nodes"] for t in tiles] == [(0, 191), (64, 299), (192, 299)]
    with pytest.raises(ValueError, match="multiples of 64"):
        scroll.tile_ranges(300, 100, 64)


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


def plan_two_biased_tiles(tmp_path, bias=(0.0, 0.004, 0.0)):
    from winding_field.util import atomic_json

    tiles = []
    for k, t in enumerate(scroll.tile_ranges(320, 128, 64)):
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
            overlap_nodes=64,
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


def test_merge_blends_tiles_into_one_readable_field_and_reports_seams(tmp_path):
    from winding_field.field import open_field

    root, tiles = plan_two_biased_tiles(tmp_path)
    out = tmp_path / "merged"
    result = scroll.merge(
        Namespace(plan=root, output=out, workers=1, allow_missing=False, coarse_sigma_mm=0)
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
