"""scroll refine: tiles that disagree are solved again with their faces held at the merge."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_field import prepared_run
from winding_field import scroll, scroll_refine

SPACING = 2.0
SHAPE = (6, 6, 96)
# Two tiles along x, 32 shared nodes: the synthetic scroll of prepared_run, W = x / 8.
BOXES = {"000-000-000": (0, 63, 0, 47), "000-000-001": (32, 95, 48, 95)}
RUN_CONFIG = dict(sweeps=1, field_chunk=64, device="cpu", threads=1)


def truth(x):
    return x / 8


def write_plan(root, errors):
    """A plan whose tiles are prepared and hold a first field: the truth plus
    ``errors[id](x_local, z)`` (windings), as an earlier ``scroll tiles`` would leave."""
    from field_store import BlockStore
    from winding_field.config import field_geometry
    from winding_field.util import atomic_json, file_sha

    tiles = []
    (root / "tiles").mkdir(parents=True)
    for k, (tid, (lo, hi, core_lo, core_hi)) in enumerate(BOXES.items()):
        shape = (SHAPE[0], SHAPE[1], hi - lo + 1)
        run = prepared_run(
            root / "tiles" / tid,
            shape=shape,
            chunk=64,
            spacing=(SPACING,) * 3,
            origin=(lo * SPACING, 0.0, 0.0),
        )
        geometry = field_geometry(json.loads((run / "PREPARED.json").read_bytes()))
        geometry.update(voxel_um=1.0, sheet_step=1.0)
        z, _, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        values = truth(x + lo) + errors.get(tid, lambda x, z: 0.0)(x, z)
        with BlockStore(run / "field", shape, 64, dict(geometry=geometry)) as store:
            store.commit((0, 0, 0), values.astype(float), {}, observed=np.ones(shape, bool))
        atomic_json(
            run / "FIELD.json",
            dict(
                status="FIELD_READY",
                geometry=geometry,
                checkpoint_sha256=file_sha(run / "field/CHECKPOINT.json"),
            ),
        )
        atomic_json(run / "RUN.json", dict(stages=dict(field={}), config=RUN_CONFIG))
        tiles.append(
            dict(
                id=tid,
                key=[0, 0, k],
                nodes_lo=[0, 0, lo],
                nodes_hi=[SHAPE[0] - 1, SHAPE[1] - 1, hi],
                core_lo=[0, 0, core_lo],
                core_hi=[SHAPE[0] - 1, SHAPE[1] - 1, core_hi],
                supported=True,
            )
        )
    atomic_json(
        root / "TILES.json",
        dict(
            schema=scroll.PLAN_SCHEMA,
            grid=dict(
                voxel_lo_zyx=[0, 0, 0],
                ratio=1,
                shape_zyx=list(SHAPE),
                origin_zyx_um=[0.0, 0.0, 0.0],
                spacing_um=SPACING,
            ),
            overlap_nodes=16,
            coarse=dict(path="unused", checkpoint_sha256="unused"),
            run_args=["--device", "cpu"],
            tiles=tiles,
        ),
    )
    return root


def arguments(root, output, **changes):
    values = dict(
        plan=str(root),
        output=str(output),
        max_passes=3,
        layers=2,
        trigger_share=0.02,
        trigger_p95=0.5,
        bound=0.25,
        max_kink_ratio=2.0,
        jobs=2,
        gpus="",
        workers=1,
        allow_missing=False,
        prune=False,
    )
    return SimpleNamespace(**dict(values, **changes))


def merged(run):
    from winding_field import field

    with field.open_field(run) as source:
        values, seen = source.read([0, 0, 0], list(SHAPE))
    assert seen.all()
    return values


def off_truth(values):
    """Largest distance to the truth once the merge's free constant and tilts are taken out."""
    z, y, x = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
    design = np.column_stack([np.ones(values.size), z.ravel(), y.ravel(), x.ravel()])
    error = (values - truth(x)).ravel()
    return float(np.abs(error - design @ np.linalg.lstsq(design, error, rcond=None)[0]).max())


def bent(x, z):
    """The second tile curves away from the sheets: what a constant and a tilt cannot level."""
    return 3.0 * (x / 63) ** 2 * np.where(z >= 3, 1.0, -1.0)


def tree(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_mtime_ns) for p in folder.rglob("*"))


def test_refine_solves_the_disagreeing_tiles_until_the_merge_stops_moving(tmp_path):
    root = write_plan(tmp_path / "plan", {"000-000-001": bent})
    before = tree(root / "tiles")
    result = scroll_refine.refine(arguments(root, tmp_path / "refined"))
    assert result["status"] == "REFINED" and result["reasons"] == []
    assert result["passes"] == 3 and result["tiles_solved"] == [0, 2, 2, 2]
    assert result["pairs_over_trigger"] == [] and result["seam_over_half_share"] == 0
    assert result["orientation"] == {"radial": 2}
    # The plan's tiles keep their first fields; passes live beside them.
    assert tree(root / "tiles") == before
    receipt = json.loads((tmp_path / "refined/FIELD.json").read_bytes())
    refine = receipt["refine"]
    assert refine["status"] == "REFINED" and refine["final_pass"] == 3
    assert refine["orientation"] == result["orientation"]
    first, second, third, last = refine["passes"]
    assert first["seams"]["pairs_over_trigger"] == ["000-000-000|000-000-001"]
    assert first["seams"]["over_half_share"] > 0.02 and first["change"] is None
    assert first["selected"] == ["000-000-000", "000-000-001"]
    # Pass 1 moves the merge by windings. Each tile was held at a merge that still carried
    # the other's error, so they bend at those faces and the seam stays open.
    assert second["tiles"] == first["selected"] and second["change"]["p95"] > 0.5
    assert second["boundary_checkpoint_sha256"] == first["merge_checkpoint_sha256"]
    assert second["rate"] is None and second["bound"] is None and second["kink"]["max_ratio"] > 2
    assert second["seams"]["pairs_over_trigger"] == first["seams"]["pairs_over_trigger"]
    # Pass 2 closes the seam and the change collapses; a bend at the held faces remains.
    assert third["seams"]["pairs_over_trigger"] == [] and third["bound"] < 0.01
    assert third["rate"] < 0.1 and third["change"]["p95"] < 0.1 * second["change"]["p95"]
    assert third["tiles"] == second["tiles"] and "bends" in third["reasons"][0]
    assert last["kink"]["max_ratio"] < 2 and last["bound"] < 0.01 and last["reasons"] == []
    assert receipt["tile_fields"] == {
        tid: f"passes/3/tiles/{tid}" for tid in ("000-000-000", "000-000-001")
    }
    errors = [off_truth(merged(root / f"passes/{k}/merge")) for k in range(4)]
    assert errors[0] > 1 and errors[3] < 2e-3 and errors == sorted(errors, reverse=True)
    np.testing.assert_array_equal(merged(tmp_path / "refined"), merged(root / "passes/3/merge"))
    held = json.loads((root / "passes/1/tiles/000-000-001/FIELD.json").read_bytes())
    assert held["field_boundary"]["checkpoint_sha256"] == first["merge_checkpoint_sha256"]
    assert held["field_boundary"]["fixed_nodes"] == {"x-": 72}
    surface = json.loads((tmp_path / "refined/surface-run.json").read_bytes())
    assert surface["field"]["path"] == str(tmp_path / "refined/field/CHECKPOINT.json")
    assert surface["field"]["sha256"] == receipt["checkpoint_sha256"]

    # Rerunning returns the published result without touching a pass.
    passes = tree(root / "passes")
    again = scroll_refine.refine(arguments(root, tmp_path / "refined"))
    assert again == dict(result, resumed=True) and tree(root / "passes") == passes
    # Other thresholds are another refinement: never mixed with these passes.
    with pytest.raises(ValueError, match="other settings \\(bound=0.25\\)"):
        scroll_refine.refine(arguments(root, tmp_path / "other", bound=0.1))


def test_refine_continues_after_an_interrupted_pass(tmp_path):
    """A pass cut after one of its tiles, and a merge cut before its receipt, are picked
    up: the finished tile is kept and the result equals an uninterrupted run."""
    whole = write_plan(tmp_path / "whole", {"000-000-001": bent})
    scroll_refine.refine(arguments(whole, tmp_path / "whole-refined"))

    root = write_plan(tmp_path / "plan", {"000-000-001": bent})
    capped = scroll_refine.refine(arguments(root, tmp_path / "capped", max_passes=1))
    assert capped["status"] == "REFINE_NOT_CONVERGED" and capped["passes"] == 1
    assert capped["pairs_over_trigger"] == ["000-000-000|000-000-001"]
    receipt = json.loads((tmp_path / "capped/FIELD.json").read_bytes())
    assert receipt["refine"]["status"] == "REFINE_NOT_CONVERGED"
    assert "1 tile pairs over the seam trigger" in receipt["refine"]["reasons"][0]
    # Pass 2 as a stopped run leaves it: one tile done, a merge without its receipt.
    boundary = root / "passes/1/merge"
    out = root / "passes/2/tiles/000-000-000"
    out.mkdir(parents=True)
    spec = dict(run=str(root / "tiles/000-000-000"), output=str(out), boundary=str(boundary))
    (out / "REFINE_TILE.json").write_text(json.dumps(dict(spec, layers=2)))
    worker = [sys.executable, "-m", "winding_field.scroll_refine", str(out / "REFINE_TILE.json")]
    done = subprocess.run(worker, capture_output=True, text=True)
    assert done.returncode == 0 and json.loads(done.stdout) == {"status": "FIELD_READY"}
    solved = (out / "FIELD.json").stat().st_mtime_ns
    (root / "passes/2/merge/field").mkdir(parents=True)
    with pytest.raises(ValueError, match="new directory to continue past the published pass"):
        scroll_refine.refine(arguments(root, tmp_path / "capped"))
    result = scroll_refine.refine(arguments(root, tmp_path / "refined"))
    assert result["status"] == "REFINED" and result["passes"] == 3
    assert (out / "FIELD.json").stat().st_mtime_ns == solved
    np.testing.assert_allclose(
        merged(tmp_path / "refined"), merged(tmp_path / "whole-refined"), atol=1e-9
    )
    # A pass tile solved against another merge is refused, not reused.
    sha = json.loads((boundary / "FIELD.json").read_bytes())["checkpoint_sha256"]
    assert scroll_refine._pass_tile_done(out, sha, 2)
    with pytest.raises(ValueError, match="solved against another merge or with other layers"):
        scroll_refine._pass_tile_done(out, "0" * 64, 2)
    with pytest.raises(ValueError, match="solved against another merge or with other layers"):
        scroll_refine._pass_tile_done(out, sha, 3)
    assert not scroll_refine._pass_tile_done(root / "passes/9/tiles/none", sha, 2)


def test_pruned_passes_can_still_be_continued(tmp_path):
    """--prune removes the merges no later pass reads; a rerun continues without them and
    refuses only a newest merge that is gone."""
    root = write_plan(tmp_path / "plan", {"000-000-001": bent})
    capped = scroll_refine.refine(arguments(root, tmp_path / "capped", max_passes=2, prune=True))
    assert capped["status"] == "REFINE_NOT_CONVERGED" and capped["passes"] == 2
    assert not (root / "passes/0/merge").exists() and (root / "passes/1/merge").is_dir()
    result = scroll_refine.refine(arguments(root, tmp_path / "refined", prune=True))
    assert result["status"] == "REFINED" and result["passes"] == 3
    assert not (root / "passes/1/merge").exists()
    # The published field does not depend on the pruned or the remaining pass directories.
    assert np.isfinite(merged(tmp_path / "capped")).all()
    (root / "passes/3/merge/FIELD.json").unlink()
    with pytest.raises(ValueError, match="missing or changed since it was measured"):
        scroll_refine.refine(arguments(root, tmp_path / "again", prune=True))


def test_tiles_that_agree_need_no_pass(tmp_path):
    root = write_plan(tmp_path / "plan", {"000-000-001": lambda x, z: 0.3 + 0.0 * x})
    result = scroll_refine.refine(arguments(root, tmp_path / "refined"))
    assert result["status"] == "REFINED" and result["passes"] == 0
    assert result["tiles_solved"] == [0] and not (root / "passes/1").exists()
    refine = json.loads((tmp_path / "refined/FIELD.json").read_bytes())["refine"]
    assert refine["reasons"] == ["no tile pair over the seam trigger"]
    assert off_truth(merged(tmp_path / "refined")) < 1e-6


def test_refine_command_reports_its_result_and_exit_code(tmp_path, capsys):
    root = write_plan(tmp_path / "plan", {"000-000-001": bent})
    common = ["--gpus", "", "--workers", "1"]
    code = scroll.main(
        ["refine", str(root), "--output", str(tmp_path / "capped"), "--max-passes", "0", *common]
    )
    printed = json.loads(capsys.readouterr().out)
    # The merge is still published, with what kept it from being accepted.
    assert code == 3 and printed["status"] == "REFINE_NOT_CONVERGED" and printed["passes"] == 0
    assert printed["pairs_over_trigger"] == ["000-000-000|000-000-001"]
    assert (tmp_path / "capped/FIELD.json").is_file()
    code = scroll.main(["refine", str(root), "--output", str(tmp_path / "bad"), "--layers", "0"])
    assert code == 2 and "--layers must be a positive integer" in capsys.readouterr().err
    level = write_plan(tmp_path / "level", {})
    code = scroll.main(["refine", str(level), "--output", str(tmp_path / "level-refined"), *common])
    assert code == 0 and json.loads(capsys.readouterr().out)["status"] == "REFINED"


def test_orientation_is_the_rule_each_tile_recorded(tmp_path):
    """A pass solves a tile with its recorded settings: the receipt says which rule that is."""
    rules = {"a": {}, "b": {"field_orientation": "prior-gradient"}, "c": {}}
    for tid, config in rules.items():
        (tmp_path / "tiles" / tid).mkdir(parents=True)
        (tmp_path / "tiles" / tid / "RUN.json").write_text(json.dumps(dict(config=config)))
    tiles = [dict(id=tid) for tid in rules]
    assert scroll_refine.tile_orientation(tmp_path, tiles) == {"prior-gradient": 1, "radial": 2}


def _tiles(*keys):
    return [dict(id="-".join(f"{k:03d}" for k in key), key=list(key)) for key in keys]


def test_selection_takes_the_pair_and_the_tiles_sharing_a_face_with_it():
    tiles = _tiles((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (1, 1, 2), (0, 0, 3))
    chosen = scroll_refine.select(tiles, ["000-000-000|000-000-001", "000-000-000|missing"])
    # (0,1,2) and (0,0,3) touch the pair only across an edge or not at all.
    assert chosen == ["000-000-000", "000-000-001", "000-000-002", "000-001-001"]
    assert scroll_refine.select(tiles, []) == []


def test_seam_gate_counts_every_pair_and_triggers_on_share_or_p95():
    seams = dict(
        worst_p95_abs_dW=0.7,
        pairs={
            "a|b": dict(nodes=1000, over_half_winding=30, p95_abs_dW=0.2),
            "a|c": dict(nodes=1000, over_half_winding=10, p95_abs_dW=0.7),
            "b|c": dict(nodes=2000, over_half_winding=40, p95_abs_dW=0.1),
            "c|d": dict(nodes=0, over_half_winding=0, p95_abs_dW=None),
        },
    )
    summary = scroll_refine.seam_summary(seams, 0.02, 0.5)
    assert summary["over_half_share"] == 80 / 4000 and summary["pairs"] == 4
    assert summary["pairs_over_trigger"] == ["a|b", "a|c"]
    assert summary["worst_pair"]["pair"] == "a|b"


def test_rate_bound_needs_a_shrinking_change_above_the_resolution():
    change = lambda p95: dict(p95=p95)  # noqa: E731
    assert scroll_refine.rate_bound(None, None) == (None, None)
    assert scroll_refine.rate_bound(change(0.498), None) == (None, None)
    rate, bound = scroll_refine.rate_bound(change(0.134), change(0.498))
    # The 175A 384 box (8g): rate 0.27, bound 0.049.
    assert rate == pytest.approx(0.269, abs=1e-3) and bound == pytest.approx(0.0493, abs=1e-3)
    assert scroll_refine.rate_bound(change(0.6), change(0.5)) == (pytest.approx(1.2), None)
    assert scroll_refine.rate_bound(change(0.001), change(0.001)) == (None, 0.001)
    assert scroll_refine.rate_bound(change(0.001), None) == (None, 0.001)


def _record(number, over=(), rate=None, bound=None, kink=None):
    return {
        "pass": number,
        "seams": dict(pairs_over_trigger=list(over)),
        "rate": rate,
        "bound": bound,
        "kink": kink,
    }


def test_decision_accepts_only_closed_seams_a_small_bound_and_no_kink():
    settings = dict(bound=0.25, max_kink_ratio=2.0)
    tiles = _tiles((0, 0, 0), (0, 0, 1), (0, 0, 2))
    pair = ["000-000-000|000-000-001"]
    decide = scroll_refine.decide
    assert decide(_record(0), settings, 3, tiles, [])[0] == "REFINED"
    status, reasons, chosen = decide(_record(0, pair), settings, 3, tiles, [])
    assert status is None and chosen == ["000-000-000", "000-000-001", "000-000-002"]
    assert decide(_record(0, pair), settings, 0, tiles, [])[0] == "REFINE_NOT_CONVERGED"
    # Closed seams after one pass: the same tiles again, to measure the rate.
    status, reasons, chosen = decide(_record(1), settings, 3, tiles, ["000-000-001"])
    assert status is None and chosen == ["000-000-001"] and "two passes" in reasons[0]
    assert decide(_record(2, rate=0.27, bound=0.049), settings, 3, tiles, [])[0] == "REFINED"
    assert decide(_record(1, bound=0.001), settings, 3, tiles, [])[0] == "REFINED"
    status, reasons, _ = decide(_record(3, rate=0.8, bound=0.4), settings, 3, tiles, ["x"])
    assert status == "REFINE_NOT_CONVERGED" and "0.4 over 0.25" in reasons[0]
    status, reasons, _ = decide(_record(3, rate=1.2), settings, 3, tiles, ["x"])
    assert status == "REFINE_NOT_CONVERGED" and "not shrinking" in reasons[0]
    kink = dict(max_ratio=3.1, worst_tile="000-000-001")
    status, reasons, _ = decide(_record(3, rate=0.2, bound=0.01, kink=kink), settings, 3, tiles, [])
    assert status == "REFINE_NOT_CONVERGED" and "000-000-001 bends 3.1 times" in reasons[0]
