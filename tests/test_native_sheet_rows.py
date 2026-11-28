"""Whole-sheet flattening records each finished piece in its own small state row.

The SLIM solve is replaced by its own seed (always a valid map), so these tests run
without Flatboi and exercise only the bookkeeping: the receipt, resume after an
interruption, and stores written before per-piece rows.
"""

import json

import numpy as np
import pytest

from test_native_sheet_scope import bundle, grid_faces


def pieces(count=12, size=5):
    vertices, faces = [], []
    for index in range(count):
        s, h = np.meshgrid(np.arange(size) * 1.0, np.arange(size) * 1.0)
        offset = np.array([index % 4, index // 4, 0]) * (size + 3) + 5
        vertices.append(
            np.column_stack([s.ravel(), 0.3 * np.sin(s + h).ravel(), h.ravel()]) + offset
        )
        faces.append(grid_faces(size, size) + index * size * size)
    return np.concatenate(vertices), np.concatenate(faces)


@pytest.fixture
def seeded_solve(monkeypatch):
    from winding_field.flattening import flatboi, patches

    monkeypatch.setattr(flatboi, "resolve_app_run", lambda app_run, offline=False: "seeded")
    monkeypatch.setattr(patches, "check_flatboi", lambda app_run, mode: {"app_run": app_run})
    monkeypatch.setattr(
        flatboi,
        "solve_batch",
        lambda items, app_run, **kw: (
            [seed for _, _, seed in items],
            dict(steps=1, converged=True),
        ),
    )


def run(tmp_path, root, name, interrupt_after=None):
    from winding_field.flattening import native_global
    from winding_field.native_store import Journal, Paused
    from winding_field.surface_mesh import MeshReader

    with MeshReader(root) as reader, Journal(tmp_path / name, {"source": name}) as store:
        if interrupt_after is not None:
            real, rows = store.set, []

            def set_and_stop(key, value):
                if key.startswith(native_global.PIECE_ROW):
                    rows.append(key)
                    if len(rows) > interrupt_after:
                        raise Paused("STOP")
                real(key, value)

            store.set = set_and_stop
            with pytest.raises(Paused):
                native_global.flatten_sheet(reader, store, 64)
            store.set = real
            return None
        receipt = native_global.flatten_sheet(reader, store, 64)
        charts = store.db.execute(
            "SELECT name,component,area,faces,status,minx,maxx,miny,maxy FROM charts ORDER BY name"
        ).fetchall()
        edges = store.db.execute("SELECT * FROM edges ORDER BY a,b,chart").fetchall()
        rows = store.db.execute(
            "SELECT count(*) FROM state WHERE name GLOB ?", (native_global.PIECE_ROW + "*",)
        ).fetchone()[0]
    return dict(receipt, seconds=None), charts, edges, rows


def test_every_piece_gets_one_row_and_the_receipt_lists_them_in_order(tmp_path, seeded_solve):
    root = bundle(tmp_path, *pieces())
    receipt, charts, _, rows = run(tmp_path, root, "out")
    assert len(receipt["pieces"]) == rows == 12 and receipt["fallback"] == []
    components = [piece["component"] for piece in receipt["pieces"]]
    assert components == sorted(components)
    assert {c[1] for c in charts} == set(components)


def test_resume_after_an_interruption_matches_an_uninterrupted_run(tmp_path, seeded_solve):
    root = bundle(tmp_path, *pieces())
    whole = run(tmp_path, root, "whole")
    run(tmp_path, root, "resumed", interrupt_after=5)
    assert run(tmp_path, root, "resumed") == whole


def test_a_store_with_the_old_whole_receipt_resumes_to_the_same_result(tmp_path, seeded_solve):
    from winding_field.flattening.native_global import PIECE_ROW

    root = bundle(tmp_path, *pieces())
    whole = run(tmp_path, root, "whole")
    run(tmp_path, root, "old", interrupt_after=5)

    def as_old_format(store):
        # Rewrite the finished rows as the single receipt value older versions kept.
        values = [
            json.loads(v)
            for (v,) in store.db.execute(
                "SELECT value FROM state WHERE name GLOB ? ORDER BY name", (PIECE_ROW + "*",)
            )
        ]
        receipt = store.get("sheet_receipt") or None
        assert receipt is None and len(values) == 5
        with store.db:
            store.db.execute("DELETE FROM state WHERE name GLOB ?", (PIECE_ROW + "*",))
            store.set("sheet_pieces", sorted(v["component"] for v in values))
            store.set(
                "sheet_receipt",
                dict(
                    schema="native-sheet-flatten-v1",
                    repair=whole[0]["repair"],
                    pieces=[v["record"] for v in values if v["kind"] == "pieces"],
                    fallback=[v["record"] for v in values if v["kind"] == "fallback"],
                ),
            )

    from winding_field.native_store import Journal

    with Journal(tmp_path / "old", {"source": "old"}) as store:
        as_old_format(store)
    result = run(tmp_path, root, "old")
    assert result[0] == whole[0] and result[1:3] == whole[1:3]
    assert result[3] == 7  # only the pieces finished after the switch have rows


@pytest.fixture
def no_sheet_seed(monkeypatch):
    """Every piece fails its whole-sheet seed, so all of them take the chart route."""
    from winding_field.flattening import native_global, native_sheet

    def refuse(*args):
        raise ValueError("no seed (test)")

    monkeypatch.setattr(native_global, "_minimal_cut", lambda *args: None)
    monkeypatch.setattr(native_sheet, "cut_component", refuse)
    monkeypatch.setattr(native_global, "SLIM_BATCH_FACES", 100)  # groups of 4 pieces


def test_seedless_pieces_share_flatboi_calls_and_resume_to_the_same_groups(
    tmp_path, monkeypatch, seeded_solve, no_sheet_seed
):
    from winding_field.flattening import flatboi

    calls = []
    solve = flatboi.solve_batch
    monkeypatch.setattr(
        flatboi,
        "solve_batch",
        lambda items, *a, **k: calls.append(len(items)) or solve(items, *a, **k),
    )
    root = bundle(tmp_path, *pieces())
    receipt, charts, edges, rows = run(tmp_path, root, "whole")
    assert receipt["pieces"] == [] and rows == 12
    assert [f["reason"] for f in receipt["fallback"]] == ["no seed (test)"] * 12
    assert len(calls) == 3 and sum(calls) == len(charts)  # 12 pieces of 32 faces, 4 per call
    assert {c[4] for c in charts} == {"LOCAL"}
    run(tmp_path, root, "resumed", interrupt_after=5)  # inside the second group
    assert run(tmp_path, root, "resumed") == (receipt, charts, edges, rows)
