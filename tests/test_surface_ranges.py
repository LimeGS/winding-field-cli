import hashlib
import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

SHAPE = (13, 14, 15)
CHUNK = 4


def scroll_like():
    """Concentric shells (radius ~ W) with an unobserved wedge and one absent chunk."""
    z, y, x = np.indices(SHAPE, dtype=float)
    values = np.hypot(y - 6.3, x - 7.1) * 0.9 + 0.05 * z
    observed = ~((y > 9) & (x > 10))
    return values, observed


def chunk_field(tmp_path, values=None, observed=None, name="chunks"):
    from field_store import BlockStore

    if values is None:
        values, observed = scroll_like()
    root = tmp_path / name
    with BlockStore(root, SHAPE, CHUNK, dict(frame="synthetic/world", spacing_xyz=[2.0] * 3)) as s:
        for key in itertools.product(*(range(-(-n // CHUNK)) for n in SHAPE)):
            if key == (0, 3, 3):  # absent chunk: unsupported, never a measured zero
                continue
            low = np.asarray(key) * CHUNK
            region = tuple(slice(a, min(a + CHUNK, n)) for a, n in zip(low, SHAPE))
            s.commit(key, values[region], dict(sweep=1, next_tile=1), observed[region])
    cp = root / "CHECKPOINT.json"
    return dict(
        kind="chunks",
        path=f"{name}/CHECKPOINT.json",
        sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        support_policy="observed",
    )


def config(tmp_path, field, levels, tile=3, name="surface.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            dict(
                schema="winding-surfaces-v1",
                field=field,
                geometry=dict(
                    frame="synthetic/world",
                    voxel_um=1.0,
                    sheet_step=1.0,
                    origin_xyz=[0.0, 0.0, 0.0],
                    spacing_xyz=[2.0, 2.0, 2.0],
                    shape_zyx=list(SHAPE),
                ),
                levels=levels,
                execution=dict(tile=tile, level_batch=1),
            )
        )
    )
    return path


def build(path, output, workers=2):
    from winding_field.surface_ranges import build_ranges
    from winding_field.surface_workflow import load_surface_config

    loaded = load_surface_config(path)
    return build_ranges(loaded["field"], loaded["geometry"], output, workers=workers)


def products(root):
    mesh = json.loads((root / "MESH.json").read_bytes())
    objs = {p.name: p.read_bytes() for p in sorted(root.glob("level-*.obj"))}
    return mesh["levels"], mesh["topology"] if "topology" in mesh else None, objs


def decoded(tmp_path, field):
    from winding_field.surface_workflow import load_surface_config
    from winding_field.surface_sources import FieldSource

    loaded = load_surface_config(config(tmp_path, field, [0.0], name="decode.json"))
    with FieldSource(loaded["field"], loaded["geometry"], cache_bytes=0) as source:
        return source.read((0, 0, 0), SHAPE)


def levels_for(values, observed):
    """Stored (decoded) values and admissible mask, as the extractor sees them."""
    admissible = values[observed]
    low, high = float(admissible.min()), float(admissible.max())
    # interior levels, a level equal to a stored node value (lattice crossing),
    # the exact global extremes, and levels outside the field
    node = float(values[6, 6, 6])
    return [low - 1.0, low, 1.37, 2.5, node, 4.81, 6.0, high, high + 1.0]


@pytest.mark.parametrize("tile", [2, 3, 5])
def test_ranges_skip_tiles_but_mesh_is_identical(tmp_path, monkeypatch, tile):
    from winding_field import surface_sources
    from winding_field.surface_workflow import run_surfaces

    field = chunk_field(tmp_path)
    path = config(tmp_path, field, levels_for(*decoded(tmp_path, field)), tile=tile)
    ranges = tmp_path / "ranges.json"
    build(path, ranges)

    reads = []
    real = surface_sources.FieldSource.read
    monkeypatch.setattr(
        surface_sources.FieldSource, "read", lambda self, *a: reads.append(1) or real(self, *a)
    )
    plain = run_surfaces(path, tmp_path / "plain", render=False)
    without = len(reads)
    reads.clear()
    ranged = run_surfaces(path, tmp_path / "ranged", render=False, tile_ranges=ranges)
    assert plain["status"] == ranged["status"] == "MESH_READY"
    assert sum(level["faces"] for level in plain["levels"]) > 0
    a, b = products(tmp_path / "plain"), products(tmp_path / "ranged")
    assert a == b  # MESH.json level records and OBJ bytes
    assert len(reads) <= without


def test_single_level_reads_only_tiles_whose_chunks_hold_it(tmp_path, monkeypatch):
    from winding_field import surface_sources
    from winding_field.surface_workflow import run_surfaces

    field = chunk_field(tmp_path)
    path = config(tmp_path, field, [1.37], tile=2)
    ranges = tmp_path / "ranges.json"
    build(path, ranges)
    reads = []
    real = surface_sources.FieldSource.read
    monkeypatch.setattr(
        surface_sources.FieldSource, "read", lambda self, *a: reads.append(1) or real(self, *a)
    )
    run_surfaces(path, tmp_path / "plain", render=False)
    without = len(reads)
    reads.clear()
    run_surfaces(path, tmp_path / "ranged", render=False, tile_ranges=ranges)
    assert len(reads) < without / 2
    assert products(tmp_path / "plain") == products(tmp_path / "ranged")


def test_ranges_record_admissible_extremes_and_skip_absent_chunks(tmp_path):
    field = chunk_field(tmp_path)
    result = build(config(tmp_path, field, [2.5]), tmp_path / "ranges.json")
    document = json.loads((tmp_path / "ranges.json").read_bytes())
    values, valid = decoded(tmp_path, field)
    assert not valid[0:4, 12:14, 12:15].any()  # the absent chunk is unsupported
    assert result["raw_min"] == float(values[valid].min())
    assert result["raw_max"] == float(values[valid].max())
    assert "0-3-3" not in document["ranges"] and result["chunks"] == 63
    assert document["binding"]["checkpoint_sha256"] == field["sha256"]


def test_ranges_for_another_field_or_policy_are_refused(tmp_path):
    from winding_field.surface_workflow import run_surfaces

    values, observed = scroll_like()
    field = chunk_field(tmp_path, values, observed)
    other = chunk_field(tmp_path, values + 1.0, observed, name="other")
    build(config(tmp_path, other, [2.5], name="other.json"), tmp_path / "other-ranges.json")
    path = config(tmp_path, field, [2.5])
    with pytest.raises(ValueError, match="another field"):
        run_surfaces(
            path, tmp_path / "out", render=False, tile_ranges=tmp_path / "other-ranges.json"
        )
    build(path, tmp_path / "ranges.json")
    stored = config(tmp_path, dict(field, support_policy="stored"), [2.5], name="stored.json")
    with pytest.raises(ValueError, match="another field"):
        run_surfaces(stored, tmp_path / "out2", render=False, tile_ranges=tmp_path / "ranges.json")
    with pytest.raises(ValueError, match="exists"):
        build(path, tmp_path / "ranges.json")


def test_empty_tile_bounds_exclude_levels_safely():
    from winding_field.surface_ranges import EMPTY, RangedSource

    source = RangedSource(object(), 4, {"0-0-0": (1.0, 2.0), "1-0-0": (5.0, 6.0)})
    assert source.tile_bounds((0, 0, 0), (3, 3, 3)) == (1.0, 2.0)
    assert source.tile_bounds((2, 0, 0), (6, 3, 3)) == (1.0, 6.0)  # spans two chunks
    assert source.tile_bounds((0, 4, 0), (3, 7, 3)) == EMPTY  # no admissible chunk


def cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "winding_field", "prepared", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_cli_ranges_then_serial_and_parallel_meshes_match(tmp_path, monkeypatch):
    values, observed = scroll_like()
    field = chunk_field(tmp_path, values, observed)
    path = config(tmp_path, field, [1.37, 2.5, 4.81], tile=3)
    made = cli(
        "ranges", "--config", path, "--output", "r.json", "--ranges-workers", 2, cwd=tmp_path
    )
    assert made.returncode == 0, made.stderr
    assert json.loads(made.stdout)["admissible_chunks"] > 0
    plain = cli("mesh", "--config", path, "--output", "plain", cwd=tmp_path)
    ranged = cli(
        "mesh", "--config", path, "--output", "ranged", "--tile-ranges", "r.json", cwd=tmp_path
    )
    assert plain.returncode == ranged.returncode == 0, ranged.stderr
    assert products(tmp_path / "plain") == products(tmp_path / "ranged")
    parallel = cli(
        "mesh",
        "--config",
        path,
        "--output",
        "parallel",
        "--parallel-levels",
        "--tile-ranges",
        "r.json",
        cwd=tmp_path,
    )
    assert parallel.returncode == 0, parallel.stderr
    serial_objs = products(tmp_path / "plain")[2]
    for i in range(3):
        child = tmp_path / "parallel" / f"level-{i:04d}" / "level-0000.obj"
        assert child.read_bytes() == serial_objs[f"level-{i:04d}.obj"]
    bad = cli(
        "render", "--mesh", "plain", "--output", "views", "--tile-ranges", "r.json", cwd=tmp_path
    )
    assert bad.returncode == 2 and "mesh/surfaces" in bad.stderr
