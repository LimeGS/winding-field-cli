import json
import shutil
import sqlite3

import numpy as np
import pytest

from test_lasagna_input import source_case
from test_splines import upstream_curves
from winding_field import workflow
from winding_field.util import atomic_json


def test_chunk_ledger_quota_rolls_back_and_preserves_committed_chunks(tmp_path):
    from winding_field.io import Volume

    _, _, config = source_case(tmp_path)
    volume = Volume(config["source"])
    ledger = tmp_path / "ledger.sqlite"
    try:
        volume.bind_ledger(ledger, quota_bytes=100352)
        volume._record("first", b"original")
        with pytest.raises(sqlite3.DatabaseError, match="full"):
            for i in range(1000):
                volume._record(str(i) * 100, b"value")
        assert ledger.stat().st_size <= 16384
        with sqlite3.connect(ledger) as db:
            assert db.execute("SELECT count(*) FROM chunks WHERE key='first'").fetchone() == (1,)
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        volume.close()


@pytest.mark.parametrize("odd_roi", [False, True])
def test_imported_assets_prepare_without_retaining_native_maps(tmp_path, monkeypatch, odd_roi):
    volume, _, config = source_case(tmp_path, embedded=True)
    volume.close()
    afv = tmp_path / "external.afv"
    upstream_curves(afv, (16, 16, 40), [(1, np.array([[4.0, 2.0, 6.0], [4.0, 8.0, 6.0]]))])
    geometry = tmp_path / "fibres-geometry.json"
    atomic_json(
        geometry,
        dict(
            source=config["source"],
            origin_xyz_um=[100.0, 200.0, 300.0],
            spacing_xyz_um=[5.0, 10.0, 20.0],
        ),
    )
    root = tmp_path / "prepared"
    config.update(
        output=str(root),
        fibres=str(afv),
        fibres_geometry=str(geometry),
        prediction="qual-v2",
        bounded_preparation=True,
        bounded_bytes=64 << 20,
        field_spacing_um=40.0,
        field_chunk=4,
        chunk=4,
        sweeps=1,
        maxiter=200,
        threads=1,
        previews=False,
    )
    if odd_roi:
        config.update(roi_lo=[1, 1, 1], roi_hi=[8, 9, 10])
    result = workflow.run(config, until="fibres")
    assert set(result["stages"]) == {"prepare", "fibres"}
    assert not (root / "maps").exists()
    assert not (root / "lasagna").exists()
    meta = json.loads((root / "PREPARED.json").read_text())
    assert meta["fibre_method"] == "external-afv"
    assert (root / "coarse/COARSE.json").is_file()
    assert json.loads((root / "FIBRES.json").read_text())["count"] > 0
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) < config["bounded_bytes"]
    legacy = tmp_path / "legacy"
    workflow.run(dict(config, output=str(legacy), bounded_preparation=False), until="fibres")

    def curves(path):
        with sqlite3.connect(path / "fibres/fibres.sqlite") as db:
            return db.execute(
                "SELECT points,family,source_id,source_block,annotation FROM curves ORDER BY id"
            ).fetchall()

    assert curves(root) == curves(legacy)
    from winding_field.exports import export_afv

    # An export must fail before publication if its database cannot fit.
    exported = tmp_path / "export-quota"
    shutil.copytree(root, exported)
    (exported / "fibres/fibres.afv").unlink()
    (exported / "exports.sqlite").unlink()
    with pytest.raises((OSError, sqlite3.DatabaseError), match="quota|full"):
        export_afv(exported, quota_bytes=81920)
    assert not (exported / "fibres/fibres.afv").exists()
    assert workflow.run(config, resume=True, until="fibres")["stages"] == result["stages"]
    from winding_field.cli import main

    assert main(["resume", str(root)]) == 0
    assert set(json.loads((root / "RUN.json").read_bytes())["stages"]) == {"prepare", "fibres"}
