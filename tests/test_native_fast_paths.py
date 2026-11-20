"""Fast paths must decide exactly as the exhaustive code they short-circuit."""

import numpy as np
import pytest

from test_meshes import curved_strip


def _signed(uv, faces):
    from winding_field.flattening.legacy_charts import cross2

    t = uv[faces]
    return cross2(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])


def _wrapped_strip():
    """Every triangle positive, yet the strip wraps twice and overlaps itself."""
    t, r = np.meshgrid(np.linspace(0, 4 * np.pi, 60), np.linspace(0, 1, 4), indexing="ij")
    uv = np.stack([(2 + r) * np.cos(t), (2 + r) * np.sin(t)], -1).reshape(-1, 2)
    k = (np.arange(59)[:, None] * 4 + np.arange(3)[None, :]).ravel()
    faces = np.concatenate([np.stack([k, k + 4, k + 1], 1), np.stack([k + 1, k + 4, k + 5], 1)])
    if _signed(uv, faces).sum() < 0:
        faces = faces[:, [0, 2, 1]]
    return np.column_stack([uv, np.zeros(len(uv))]), faces, uv


def test_simple_disk_accepts_valid_chart_and_matches_exhaustive_test():
    from winding_field.flattening.legacy_charts import overlap_pairs
    from winding_field.flattening.native_charts import _lscm, _simple_disk

    v, f = curved_strip()
    uv, _ = _lscm(v, f)
    assert _simple_disk(uv, f, _signed(uv, f)) and overlap_pairs(uv, f) == 0


def test_overlapping_positive_chart_falls_back_and_is_rejected():
    from winding_field.flattening.legacy_charts import overlap_pairs
    from winding_field.flattening.native_charts import _simple_disk, _validate

    v, f, uv = _wrapped_strip()
    assert (_signed(uv, f) > 0).all() and overlap_pairs(uv, f) > 0
    assert not _simple_disk(uv, f, _signed(uv, f))
    with pytest.raises(ValueError, match="overlapping"):
        _validate(v, f, uv)


def _obj(tmp_path, v, f, uv, *, comment=False):
    from winding_field.flattening.flatboi import _write_obj

    path = tmp_path / ("c.obj" if comment else "plain.obj")
    _write_obj(path, v, uv, f)
    if comment:
        path.write_text("# comment\n" + path.read_text())
    return path


@pytest.mark.parametrize("comment", [False, True])
def test_fast_obj_parse_matches_general_parser(tmp_path, monkeypatch, comment):
    from winding_field import surface_export as se

    v, f = curved_strip()
    uv = v[:, :2] * [1.5, 0.5]
    path = _obj(tmp_path, v, f, uv, comment=comment)
    assert (se._simple_obj(path.read_text()) is None) == comment
    fast = se.read_flatboi_uv(path, v, f, vertex_atol=0)
    monkeypatch.setattr(se, "_simple_obj", lambda text: None)
    general = se.read_flatboi_uv(path, v, f, vertex_atol=0)
    np.testing.assert_array_equal(fast[0], general[0])
    assert fast[1] == general[1]


def test_vectorized_binding_rejects_uv_seams_and_counts_aliases(tmp_path):
    from winding_field import surface_export as se

    v, f = curved_strip()
    path = tmp_path / "seam.obj"
    lines = [f"v {x} {y} {z}" for x, y, z in v] + [f"vt {x} {y}" for x, y in v[:, :2]]
    lines += ["vt 99 99"]  # a second, different UV for vertex 1
    faces = [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in f]
    shared = int(np.flatnonzero(np.bincount(f[:, 0]) > 1)[0])  # used by several faces
    row = int(np.flatnonzero(f[:, 0] == shared)[0])
    a, b, c = f[row]
    faces[row] = f"f {a + 1}/{len(v) + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}"
    path.write_text("\n".join(lines + faces) + "\n")
    with pytest.raises(ValueError, match="UV seam"):
        se.read_flatboi_uv(path, v, f, vertex_atol=0)


def test_scratch_accounting_tolerates_a_sibling_journal_still_being_created(tmp_path):
    """Parallel levels read each other's journals; a sibling may exist before its schema.

    sqlite3.connect creates work.sqlite before Journal creates its tables, so a level that
    checks its siblings at that moment must count the newcomer as zero scratch, not fail
    with "no such table: state" (seen on macOS CI with spawned level workers).
    """
    import json
    import sqlite3
    from contextlib import closing

    from winding_field.native_store import Journal

    (tmp_path / "NATIVE.json").write_text("{}")
    sibling = tmp_path / "level-0001"
    sibling.mkdir()
    (sibling / "IDENTITY.json").write_text(json.dumps({"level": 1}))  # Journal's order
    sqlite3.connect(sibling / "work.sqlite").close()
    with Journal(tmp_path / "level-0000", {"level": 0}) as store:
        store.check()
        assert store.other_scratch == 0
    with Journal(sibling, {"level": 1}):  # once its tables exist, its scratch counts again
        pass
    with closing(sqlite3.connect(sibling / "work.sqlite")) as db, db:
        db.execute("INSERT OR REPLACE INTO state VALUES ('scratch_bytes', '7')")
    with Journal(tmp_path / "level-0000", {"level": 0}) as store:
        store.check()
        assert store.other_scratch == 7


def test_parallel_levels_publish_the_serial_result(tmp_path):
    import hashlib
    import json

    from test_native_runtime import field_case
    from winding_field.native_pipeline import work

    results = {}
    for workers in (1, 2):
        (tmp_path / str(workers)).mkdir()
        run, config = field_case(tmp_path / str(workers), levels=(0.5, 1.5, 2.5))
        config.update(flattening_method="default", level_workers=workers)
        result = work(run, config)
        assert result["status"] == "COMPLETE", result  # reason/active level if it pauses
        root = run / "surfaces"
        files = sorted(p for p in root.rglob("*") if p.suffix in (".npz", ".png"))
        assert json.loads((run / "SURFACES.json").read_bytes())["status"] == "COMPLETE"
        # Run-specific paths enter identities; the produced arrays and images must not differ.
        results[workers] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        }
    assert results[1] == results[2] and len(results[1]) > 3


@pytest.mark.parametrize("value", [0, 65, 1.5, True])
def test_level_workers_are_validated(value):
    from winding_field.config import validate_config

    with pytest.raises(ValueError, match="level_workers"):
        validate_config({"level_workers": value})
