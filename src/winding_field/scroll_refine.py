"""Close the seams of a tiled field with Dirichlet passes between the tiles.

``scroll tiles`` solves every tile alone, with its internal faces free. Where a face cuts
a region the data determine poorly, the tile drifts whole windings from its neighbours
and ``scroll merge`` cannot level it with one constant and one tilt. ``scroll refine``
repeats, after that first merge (pass 0):

    measure   seams, the change against the previous merge, slope kinks at held faces
    select    the tiles of every pair over the seam trigger, and the tiles next to them
    solve     the selected tiles with ``run --field-boundary`` on the current merge
    merge     the plan again, each tile with its newest field

It stops when no pair is over the trigger, the bound of the remaining error is small and
no solved tile bends at its held faces, or at ``--max-passes``. The plan's tiles are never
written: pass k keeps its fields in PLAN/passes/k/tiles/ID and its merge in
PLAN/passes/k/merge, and the final merge is published in ``--output`` with the receipt of
every pass (FIELD.json ``refine``). Rerunning the command continues where it stopped.

With the faces of two neighbours held at one merge their overlap agrees partly by
construction, so the seams alone would flatter the result: the change between
consecutive merges and its rate are what say the iteration has settled. Settled is not
the same as right: a merge can stop moving away from the single-block solution of the
same box, and only a comparison with one has shown it.

``--worst-first`` (off by default, experimental) narrows the first pass to the tiles the
open seams are attributed to. The pairs alone do not identify the tiles that drifted: two
tiles that drifted the same way agree with each other and the blame falls on their
neighbours.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .field_boundary import FACES, RULE as BOUNDARY_RULE
from .scroll import PLAN_SCHEMA, _json, _Tile, _tile_done, merge, visible_gpus
from .util import atomic_json, file_sha

REFINE_SCHEMA = "winding-scroll-refine-v2"
# |dW| between merges in windings: 1/1000 up to 4, then one open bin.
CHANGE_BINS = np.concatenate([np.linspace(0.0, 4.0, 4001), [np.inf]])
# The mid-plane reference of the kink ratio is never taken below a bend of this share of
# the local sheet slope per node: a field flatter than that (a synthetic one) would turn
# any bend at a face into a large ratio.
KINK_FLOOR = 0.05
SETTINGS = ("layers", "trigger_share", "trigger_p95", "bound", "max_kink_ratio", "worst_first")


def seam_summary(seams, trigger_share, trigger_p95):
    """The seam gate of a merge's SEAMS.json and the pairs over the trigger.

    The share counts every pair's shared nodes over half a winding apart, after the join:
    a node that three tiles share counts in its three pairs.
    """
    pairs = seams["pairs"]
    nodes = sum(p["nodes"] for p in pairs.values())
    half = sum(p["over_half_winding"] for p in pairs.values())

    def share(p):
        return p["over_half_winding"] / p["nodes"] if p["nodes"] else 0.0

    over = sorted(
        name
        for name, p in pairs.items()
        if share(p) > trigger_share or (p["p95_abs_dW"] or 0.0) > trigger_p95
    )
    worst = max(pairs, key=lambda name: (share(pairs[name]), name), default=None)
    return dict(
        pairs=len(pairs),
        nodes=nodes,
        over_half_winding=half,
        over_half_share=half / nodes if nodes else 0.0,
        worst_p95_abs_dW=seams.get("worst_p95_abs_dW"),
        worst_pair=None
        if worst is None
        else dict(
            pair=worst, over_half_share=share(pairs[worst]), p95_abs_dW=pairs[worst]["p95_abs_dW"]
        ),
        pairs_over_trigger=over,
    )


def select(tiles, pairs):
    """Ids of the tiles in ``pairs`` ("a|b") and of the tiles that share a face with one."""
    by_id = {t["id"]: t for t in tiles}
    chosen = {tid for pair in pairs for tid in pair.split("|") if tid in by_id}
    keys = [by_id[tid]["key"] for tid in chosen]
    for tile in tiles:
        if any(sum(abs(a - b) for a, b in zip(tile["key"], key)) == 1 for key in keys):
            chosen.add(tile["id"])
    return sorted(chosen)


def tile_blame(pairs):
    """{tile: b}: the share of seam nodes over half a winding that each tile accounts for.

    A pair's share is modelled as the sum of its two tiles' b >= 0 and fitted by least
    squares over all pairs, each weighted by its shared nodes. A tile at odds with every
    neighbour gets a large b; a tile whose only open seams are with such a tile gets
    little, because its other pairs are closed. Two tiles with nothing but their common
    pair to tell them apart get the same b.
    """
    from scipy.optimize import lsq_linear
    from scipy.sparse import coo_matrix

    names = sorted(name for name, p in pairs.items() if p["nodes"])
    ids = sorted({tid for name in names for tid in name.split("|")})
    if not ids:
        return {}
    column = {tid: k for k, tid in enumerate(ids)}
    rows, columns, entries, target = [], [], [], []
    for row, name in enumerate(names):
        pair = pairs[name]
        weight = float(np.sqrt(pair["nodes"]))
        for tid in name.split("|"):
            rows.append(row)
            columns.append(column[tid])
            entries.append(weight)
        target.append(weight * pair["over_half_winding"] / pair["nodes"])
    # What the pairs leave open is settled by two faint pulls. On the sum of b: of the
    # fits that explain a row of tiles equally well, the one that blames fewest (the tile
    # in the middle, not its two neighbours). On each b, fainter: two tiles with one
    # common pair share it evenly.
    heaviest = max(entries)
    for k in range(len(ids)):
        rows.append(len(names))
        columns.append(k)
        entries.append(1e-3 * heaviest)
    target.append(0.0)
    for k in range(len(ids)):
        rows.append(len(names) + 1 + k)
        columns.append(k)
        entries.append(1e-6 * heaviest)
        target.append(0.0)
    matrix = coo_matrix((entries, (rows, columns)), shape=(len(target), len(ids))).tocsr()
    fit = lsq_linear(matrix, np.asarray(target), bounds=(0.0, np.inf), tol=1e-12)
    return dict(zip(ids, (float(b) for b in fit.x)))


def worst_tiles(pairs, over, share):
    """The tiles of the pairs ``over`` the trigger whose blame is at least ``share`` of
    the largest one, and that largest blame; ([], 0.0) when nothing stands out."""
    blame = tile_blame(pairs)
    candidates = {tid for name in over for tid in name.split("|") if tid in blame}
    top = max((blame[tid] for tid in candidates), default=0.0)
    if top <= 0:
        return [], 0.0
    # Ties are kept together: 1e-6 of the top is far under what tells tiles apart.
    return sorted(tid for tid in candidates if blame[tid] >= share * top - 1e-6 * top), top


def merge_change(current, previous, boxes):
    """How far a merge moved from the previous one, in windings.

    The overlap join leaves each merge one free constant and tilt, so their difference is
    first cleared of the constant and tilt that fit it best over every node both merges
    observed. The statistics are then taken on the nodes of ``boxes`` (inclusive
    ``(lo, hi)`` node boxes: the tiles solved in this pass); the nodes of the other tiles
    did not move and would only dilute them.
    """
    from .field import open_field

    meta = _json(current / "FIELD.json")
    shape = np.asarray(meta["geometry"]["shape_zyx"])
    step = float(meta["geometry"].get("sheet_step", 1.0))
    manifest = _json(current / "field/CHECKPOINT.json")
    chunk = int(manifest["chunk"])
    earlier = _json(previous / "field/CHECKPOINT.json")["chunks"]
    keys = sorted(tuple(map(int, k.split("-"))) for k in manifest["chunks"] if k in earlier)
    centre = (shape - 1) / 2
    scale = np.maximum(centre, 1.0)
    with open_field(current) as new, open_field(previous) as old:

        def differences(inside):
            for key in keys:
                lo = np.asarray(key) * chunk
                hi = np.minimum(lo + chunk, shape)
                if inside is not None and not any(
                    (a <= hi - 1).all() and (b >= lo).all() for a, b in inside
                ):
                    continue
                values, seen = new.read(lo.tolist(), hi.tolist())
                before, also = old.read(lo.tolist(), hi.tolist())
                both = seen & also
                if inside is not None:
                    within = np.zeros(both.shape, bool)
                    for a, b in inside:
                        first, last = np.maximum(a, lo) - lo, np.minimum(b + 1, hi) - lo
                        if (last > first).all():
                            within[tuple(slice(x, y) for x, y in zip(first, last))] = True
                    both &= within
                if both.any():
                    nodes = np.argwhere(both) + lo
                    design = np.column_stack([np.ones(len(nodes)), (nodes - centre) / scale])
                    yield (values[both] - before[both]) / step, design

        gram, vector, common = np.zeros((4, 4)), np.zeros(4), 0
        for delta, design in differences(None):
            gram += design.T @ design
            vector += design.T @ delta
            common += len(delta)
        if not common:
            return None
        # lstsq: a grid one node thick has no tilt along that axis.
        fit = np.linalg.lstsq(gram, vector, rcond=None)[0]
        boxes = [(np.asarray(a), np.asarray(b)) for a, b in boxes]
        hist, count, over = np.zeros(len(CHANGE_BINS) - 1, np.int64), 0, 0
        for delta, design in differences(boxes):
            left = np.abs(delta - design @ fit)
            hist += np.histogram(left, CHANGE_BINS)[0]
            count += len(left)
            over += int((left > 0.5).sum())
    if not count:
        return None
    index = int(np.searchsorted(np.cumsum(hist), 0.95 * count))
    return dict(
        nodes=count,
        common_nodes=common,
        p95=float(CHANGE_BINS[min(index + 1, len(CHANGE_BINS) - 2)]),
        over_half_share=over / count,
        removed=dict(constant=float(fit[0]), tilt=fit[1:].tolist()),
    )


def rate_bound(change, previous):
    """(r, bound): the ratio of two consecutive changes and the bound it puts on what is
    left, change x r / (1 - r) for a geometric decay. No bound unless the change shrinks.
    ``previous`` is the change of the pass before on the same tiles, or None: changes
    taken on different tiles have no ratio.

    A change within the resolution of its histogram has no measurable rate: the merge has
    stopped moving, and the change itself is the bound.
    """
    if change is None:
        return None, None
    if change["p95"] <= CHANGE_BINS[1]:
        return None, change["p95"]
    if previous is None:
        return None, None
    rate = change["p95"] / previous["p95"]
    return rate, (change["p95"] * rate / (1 - rate) if rate < 1 else None)


def tile_kink(plan_root, tile, grid, layers):
    """A solved tile's bend at its held faces against its own mid-planes.

    The second difference of W across three consecutive planes, over the local sheet
    slope (density x spacing, windings per node), on observed nodes: centred on the inner
    held layer of every face the solve held (``tile["field_run"]``'s receipt), and on the
    mid-plane of each axis. A face that was pulled away from what the tile's data ask for
    shows as a ratio well over 1.
    """
    from .field import map_axes

    if layers < 2:
        return None
    held = _held_faces(_json(Path(tile["field_run"]) / "FIELD.json"))
    opened = _Tile(plan_root, tile, grid)
    try:
        shape = np.asarray(opened.geometry["shape_zyx"])
        spacing = float(grid["spacing_um"])

        def bends(axis, first):
            lo, hi = np.zeros(3, np.int64), shape.copy()
            lo[axis], hi[axis] = first, first + 3
            if first < 0 or hi[axis] > shape[axis]:
                return np.zeros(0)
            values, seen = opened.observed.read(lo.tolist(), hi.tolist())
            planes = [np.take(values, k, axis) for k in range(3)]
            valid = np.all([np.take(seen, k, axis) for k in range(3)], axis=0)
            middle = lo.copy(), hi.copy()
            middle[0][axis], middle[1][axis] = first + 1, first + 2
            maps = opened.maps.sample(
                ("density", "support"), map_axes(opened.prepared, opened.geometry, *middle)
            )
            density = np.take(maps["density"], 0, axis).astype(float)
            valid &= np.take(maps["support"], 0, axis).astype(bool) & (density > 0)
            bend = np.abs(planes[0] - 2 * planes[1] + planes[2])
            return bend[valid] / (density[valid] * spacing)

        faces = []
        for axis in range(3):
            if FACES[2 * axis] in held:
                faces.append(bends(axis, layers - 2))
            if FACES[2 * axis + 1] in held:
                faces.append(bends(axis, shape[axis] - layers - 1))
        middles = [bends(axis, shape[axis] // 2 - 1) for axis in range(3)]
    finally:
        opened.close()
    face = np.concatenate(faces) if faces else np.zeros(0)
    middle = np.concatenate(middles)
    if not len(face) or not len(middle):
        return None
    face_p95, middle_p95 = (float(np.percentile(v, 95)) for v in (face, middle))
    return dict(face_p95=face_p95, mid_p95=middle_p95, ratio=face_p95 / max(middle_p95, KINK_FLOOR))


def _held_faces(receipt):
    """Faces of a pass tile that held nodes of another tile: its ``field_boundary``."""
    boundary = receipt["field_boundary"]
    return sorted(face for face, tiles in boundary["source_tiles"].items() if tiles)


def held_summary(fields, selection):
    """What the tiles of a pass held: nodes in all, and the internal faces left free
    because no merged tile continues beyond them ("id face")."""
    nodes, faces, open_faces = 0, 0, []
    for tid in selection:
        boundary = _json(fields[tid] / "FIELD.json")["field_boundary"]
        nodes += sum(boundary["fixed_nodes"].values())
        for face, tiles in boundary["source_tiles"].items():
            if tiles:
                faces += 1
            else:
                open_faces.append(f"{tid} {face}")
    return dict(nodes=nodes, faces=faces, open_faces=len(open_faces), e_g=open_faces[:12])


def kink_summary(kinks):
    ratios = {tid: k["ratio"] for tid, k in kinks.items() if k is not None}
    if not ratios:
        return None
    worst = max(ratios, key=lambda tid: (ratios[tid], tid))
    return dict(
        tiles=len(ratios),
        median_ratio=float(np.median(list(ratios.values()))),
        max_ratio=ratios[worst],
        worst_tile=worst,
    )


def decide(record, settings, max_passes, tiles, previous_selection):
    """(status, reasons, selection): accept the merge of this pass, give up, or name the
    tiles of the next pass (status None). At ``max_passes`` the selection is what a
    further pass would solve."""
    number = record["pass"]
    over = record["seams"]["pairs_over_trigger"]
    reasons = []
    if over:
        reasons.append(f"{len(over)} tile pairs over the seam trigger, e.g. {over[:3]}")
    if number >= 1:
        if record["bound"] is None and record["rate"] is None:
            reasons.append(
                "the rate of change between merges needs two passes that solve these "
                "tiles and move the merge"
            )
        elif record["bound"] is None:
            reasons.append(
                f"the change between merges is not shrinking (rate {record['rate']:.3g})"
            )
        elif record["bound"] > settings["bound"]:
            reasons.append(
                f"bound of the remaining error {record['bound']:.3g} over {settings['bound']}"
            )
        kink = record.get("kink")
        if kink and kink["max_ratio"] > settings["max_kink_ratio"]:
            reasons.append(
                f"tile {kink['worst_tile']} bends {kink['max_ratio']:.3g} times more at its "
                "held faces than at its mid-planes"
            )
    if not reasons:
        return "REFINED", ["no tile pair over the seam trigger"] if not number else [], []
    worst = (record.get("worst_first") or {}).get("tiles")
    if not number and worst:
        # The first pass: the drifted tiles alone, against neighbours that have not moved.
        selection = list(worst)
    else:
        # Seams closed but the rate is not known or not small yet: the same tiles again.
        selection = select(tiles, over) if over else list(previous_selection)
    if number >= max_passes or not selection:
        return "REFINE_NOT_CONVERGED", reasons, selection
    return None, reasons, selection


def _earlier_change(passes, records, number, selection, boxes):
    """The change of pass ``number - 1`` on the tiles of this pass, for the rate; None when
    that pass did not solve them all. A pass that solved more tiles is measured again on
    these ones, from its merge and the one before it."""
    before = records[number - 1]
    if before["tiles"] == selection:
        return before["change"]
    older = passes / str(number - 2) / "merge"
    if number >= 2 and set(selection) < set(before["tiles"]) and _merge_ready(older):
        return merge_change(passes / str(number - 1) / "merge", older, boxes)
    return None


def tile_orientation(root, tiles):
    """{rule: tiles}: the orientation rule each tile's recorded run settings name. A pass
    solves a tile with those settings, whatever made the field the plan holds."""
    counts = {}
    for tile in tiles:
        config = _json(root / "tiles" / tile["id"] / "RUN.json")["config"]
        rule = config.get("field_orientation", "radial")
        counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items()))


# --- one tile of a pass -----------------------------------------------------------------


def _pass_tile_done(out, boundary_sha, layers):
    """True for a completed pass tile; refuses one solved against another merge."""
    path = out / "FIELD.json"
    if not path.is_file():
        return False
    receipt = _json(path)
    held = receipt.get("field_boundary") or {}
    if (
        held.get("checkpoint_sha256") != boundary_sha
        or held.get("layers") != layers
        or held.get("rule") != BOUNDARY_RULE
    ):
        raise ValueError(
            f"{out} was solved against another merge, with other layers or by another "
            "boundary rule; move it away to solve this pass again"
        )
    return receipt.get("status") == "FIELD_READY"


def solve_tile(spec):
    """Worker: solve one tile's field into the pass directory, from its prepared run."""
    import torch

    from .field import solve

    run, out = Path(spec["run"]), Path(spec["output"])
    config = dict(
        _json(run / "RUN.json")["config"],
        field_boundary=spec["boundary"],
        field_boundary_layers=spec["layers"],
    )
    out.mkdir(parents=True, exist_ok=True)
    scope = run / "FIELD_SCOPE.json"
    if scope.is_file() and not (out / "FIELD_SCOPE.json").exists():
        # The owned chunks do not depend on the boundary; the solve checks the key.
        shutil.copy2(scope, out / "FIELD_SCOPE.json")
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: interrupted.append(True))
    torch.set_num_threads(config.get("threads", min(4, os.cpu_count() or 1)))
    result = solve(run, config, lambda: bool(interrupted) or (out / "STOP").exists(), output=out)
    return result["status"]


def _solve_pass(root, number, selection, boundary, settings, args, document):
    """Solve ``selection`` against ``boundary`` (a merge) with bounded concurrency."""
    folder = root / "passes" / str(number)
    for name in ("tiles", "logs", "status"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    boundary_sha = _json(boundary / "FIELD.json")["checkpoint_sha256"]
    layers = settings["layers"]
    pending = [
        tid
        for tid in selection
        if not _pass_tile_done(folder / "tiles" / tid, boundary_sha, layers)
    ]
    run_args = list(document.get("run_args", []))
    on_cpu = "--device" in run_args and run_args[run_args.index("--device") + 1].startswith("cpu")
    if args.gpus is not None:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        gpus = [] if on_cpu else [g for g, _ in visible_gpus()]
    per_gpu = int((document.get("tile_memory") or {}).get("jobs_per_gpu", 1))
    jobs = args.jobs or (len(gpus) * per_gpu if gpus else 1)
    load = {g: 0 for g in gpus}
    running, results, stopping = {}, {}, False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for tile_id in running:
            (folder / "tiles" / tile_id / "STOP").touch()

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.time()
    try:
        while pending or running:
            while pending and len(running) < jobs and not stopping:
                tile_id = pending.pop(0)
                out = folder / "tiles" / tile_id
                out.mkdir(exist_ok=True)
                (out / "STOP").unlink(missing_ok=True)
                spec = dict(
                    run=str(root / "tiles" / tile_id),
                    output=str(out),
                    boundary=str(boundary),
                    layers=layers,
                )
                atomic_json(out / "REFINE_TILE.json", spec)
                command = [
                    sys.executable, "-m", "winding_field.scroll_refine", str(out / "REFINE_TILE.json")
                ]  # fmt: skip
                log = open(folder / "logs" / f"{tile_id}.log", "ab")
                log.write(f"\n# {time.strftime('%FT%T')} {' '.join(command)}\n".encode())
                gpu, env = None, None
                if gpus:
                    gpu = min(gpus, key=lambda g: load[g])
                    load[gpu] += 1
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                    log.write(f"# GPU {gpu}\n".encode())
                log.flush()
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[tile_id] = (process, log, time.time(), gpu)
            if not running:
                break
            time.sleep(0.2)
            for tile_id, (process, log, begun, gpu) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[tile_id]
                if gpu is not None:
                    load[gpu] -= 1
                done = code == 0 and _pass_tile_done(
                    folder / "tiles" / tile_id, boundary_sha, layers
                )
                results[tile_id] = "DONE" if done else "STOPPED" if stopping else "FAILED"
                atomic_json(
                    folder / "status" / f"{tile_id}.json",
                    dict(status=results[tile_id], exit_code=code, seconds=time.time() - begun),
                )
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    unfinished = sorted(tid for tid, state in results.items() if state != "DONE")
    return unfinished + sorted(pending), time.time() - started, jobs, gpus


# --- the command ------------------------------------------------------------------------


def _merge_ready(folder):
    return (folder / "FIELD.json").is_file() and (folder / "SEAMS.json").is_file()


def _publish(merged, output, block):
    """The final merge in ``output``: its files linked (copied across filesystems), its
    FIELD.json with the refine receipt."""
    field = output / "field"
    field.mkdir(parents=True)
    manifest = _json(merged / "field/CHECKPOINT.json")
    names = [c["path"] for c in manifest["chunks"].values()] + ["CHECKPOINT.json"]
    for name in names:
        try:
            os.link(merged / "field" / name, field / name)
        except OSError:
            shutil.copy2(merged / "field" / name, field / name)
    for name in ("SEAMS.json", "LEVELS.json"):
        shutil.copy2(merged / name, output / name)
    surface = _json(merged / "surface-run.json")
    surface["field"]["path"] = str(field / "CHECKPOINT.json")
    atomic_json(output / "surface-run.json", surface)
    atomic_json(output / "FIELD.json", dict(_json(merged / "FIELD.json"), refine=block))


def refine(args):
    root = Path(args.plan).resolve()
    document = _json(root / "TILES.json")
    if document.get("schema") != PLAN_SCHEMA:
        raise ValueError("not a scroll tile plan")
    output = Path(args.output).resolve()
    settings = dict(
        layers=args.layers,
        trigger_share=args.trigger_share,
        trigger_p95=args.trigger_p95,
        bound=args.bound,
        max_kink_ratio=args.max_kink_ratio,
        worst_first=args.worst_first,
    )
    if not 0 <= args.worst_first <= 1:
        raise ValueError("--worst-first must be from 0 to 1")
    if type(args.layers) is not int or args.layers < 1:
        raise ValueError("--layers must be a positive integer")
    if type(args.max_passes) is not int or args.max_passes < 0:
        raise ValueError("--max-passes must be a nonnegative integer")
    if not all(
        np.isfinite(settings[k]) and settings[k] > 0
        for k in ("trigger_share", "trigger_p95", "bound", "max_kink_ratio")
    ):
        raise ValueError("refine thresholds must be positive numbers")
    passes = root / "passes"
    passes.mkdir(exist_ok=True)
    state_path = passes / "REFINE.json"
    plan_sha = file_sha(root / "TILES.json")
    state = dict(schema=REFINE_SCHEMA, plan_sha256=plan_sha, settings=settings, passes=[])
    if state_path.is_file():
        saved = _json(state_path)
        if saved.get("schema") != REFINE_SCHEMA or saved.get("plan_sha256") != plan_sha:
            raise ValueError(
                "PLAN/passes belongs to another plan or another version of scroll refine; "
                "move it away to refine again"
            )
        changed = sorted(k for k in SETTINGS if saved["settings"].get(k) != settings[k])
        if changed:
            raise ValueError(
                "PLAN/passes was refined with other settings ("
                + ", ".join(f"{k}={saved['settings'].get(k)}" for k in changed)
                + "); passes of different settings are not mixed: move it away to start again"
            )
        state = saved
    published = state.get("published")
    if published and published["output"] == str(output) and _merge_ready(output):
        if published["status"] == "REFINED" or published["pass"] >= args.max_passes:
            return dict(published["result"], resumed=True)
    if output.exists():
        raise ValueError(
            "refine output must be a new directory"
            + (" to continue past the published pass" if published else "")
        )
    grid = document["grid"]
    tiles = [
        t for t in document["tiles"] if t["supported"] and _tile_done(root / "tiles" / t["id"])
    ]
    by_id = {t["id"]: t for t in tiles}
    orientation = tile_orientation(root, tiles)
    workers = SimpleNamespace(
        plan=str(root),
        workers=args.workers,
        allow_missing=args.allow_missing,
        join="overlap",
        coarse_sigma_mm=0.0,
    )
    fields = {}
    selection = []
    number = 0
    while True:
        folder = passes / str(number)
        merged = folder / "merge"
        record = state["passes"][number] if number < len(state["passes"]) else None
        if record is not None:
            # Only the newest measured merge is read again (the next pass holds its tiles
            # at it and the result is published from it); --prune removes older ones.
            newest = number == len(state["passes"]) - 1
            if (newest or _merge_ready(merged)) and (
                not _merge_ready(merged)
                or record["merge_checkpoint_sha256"]
                != (_json(merged / "FIELD.json")["checkpoint_sha256"])
            ):
                raise ValueError(
                    f"{merged} is missing or changed since it was measured; move PLAN/passes "
                    "away to refine again"
                )
            for tid in record["tiles"]:
                fields[tid] = folder / "tiles" / tid
        else:
            solve_seconds = jobs = gpus = None
            if number:
                boundary = passes / str(number - 1) / "merge"
                unfinished, solve_seconds, jobs, gpus = _solve_pass(
                    root, number, selection, boundary, settings, args, document
                )
                if unfinished:
                    atomic_json(state_path, state)
                    return dict(
                        status="INCOMPLETE",
                        reason=f"pass {number}: {len(unfinished)} tiles did not finish; logs in "
                        + str(folder / "logs"),
                        tiles=unfinished,
                    )
                for tid in selection:
                    fields[tid] = folder / "tiles" / tid
            if merged.exists():
                # A merge that did not finish: its receipt is written last.
                shutil.rmtree(merged)
            merge_result = merge(SimpleNamespace(**vars(workers), output=str(merged)), dict(fields))
            field_meta = _json(merged / "FIELD.json")
            record = dict(
                **{"pass": number},
                tiles=list(selection),
                boundary_checkpoint_sha256=None
                if not number
                else _json(passes / str(number - 1) / "merge/FIELD.json")["checkpoint_sha256"],
                solve_seconds=solve_seconds,
                jobs=jobs,
                gpus=gpus,
                merge_seconds=field_meta["seconds"],
                merge_checkpoint_sha256=field_meta["checkpoint_sha256"],
                merged_tiles=merge_result["tiles"],
                seams=seam_summary(
                    _json(merged / "SEAMS.json"), settings["trigger_share"], settings["trigger_p95"]
                ),
                join=field_meta.get("join"),
                worst_first=None,
                held=None,
                change=None,
                rate=None,
                bound=None,
                kink=None,
            )
            if not number and settings["worst_first"] > 0:
                worst, top = worst_tiles(
                    _json(merged / "SEAMS.json")["pairs"],
                    record["seams"]["pairs_over_trigger"],
                    settings["worst_first"],
                )
                record["worst_first"] = dict(share=settings["worst_first"], blame=top, tiles=worst)
            if number:
                record["held"] = held_summary(fields, selection)
                boxes = [(by_id[tid]["nodes_lo"], by_id[tid]["nodes_hi"]) for tid in selection]
                record["change"] = merge_change(merged, passes / str(number - 1) / "merge", boxes)
                record["rate"], record["bound"] = rate_bound(
                    record["change"],
                    _earlier_change(passes, state["passes"], number, selection, boxes),
                )
                record["kink"] = kink_summary(
                    {
                        tid: tile_kink(
                            root,
                            dict(by_id[tid], field_run=str(fields[tid])),
                            grid,
                            settings["layers"],
                        )
                        for tid in selection
                    }
                )
            status, reasons, chosen = decide(record, settings, args.max_passes, tiles, selection)
            record.update(reasons=reasons, selected=chosen)
            state["passes"].append(record)
            state.pop("published", None)
            atomic_json(state_path, state)
        if args.prune and number >= 2:
            shutil.rmtree(passes / str(number - 2) / "merge", ignore_errors=True)
        # A saved pass is decided again: --max-passes may have been raised since.
        status, reasons, chosen = decide(
            record, settings, args.max_passes, tiles, state["passes"][number]["tiles"]
        )
        if status is not None:
            break
        selection = chosen
        number += 1
    final = state["passes"][number]
    block = dict(
        schema=REFINE_SCHEMA,
        status=status,
        reasons=reasons,
        settings=settings,
        orientation=orientation,
        max_passes=args.max_passes,
        final_pass=number,
        passes=state["passes"][: number + 1],
        plan_passes=str(passes),
    )
    _publish(passes / str(number) / "merge", output, block)
    result = dict(
        status=status,
        output=str(output),
        passes=number,
        reasons=reasons,
        seam_over_half_share=final["seams"]["over_half_share"],
        pairs_over_trigger=final["seams"]["pairs_over_trigger"],
        change_p95=None if final["change"] is None else final["change"]["p95"],
        rate=final["rate"],
        bound=final["bound"],
        tiles_solved=[len(p["tiles"]) for p in state["passes"][: number + 1]],
        orientation=orientation,
    )
    state["published"] = dict(output=str(output), status=status, **{"pass": number}, result=result)
    atomic_json(state_path, state)
    return result


def main(argv=None):
    """Internal entry of one pass tile: ``python -m winding_field.scroll_refine SPEC.json``."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m winding_field.scroll_refine REFINE_TILE.json", file=sys.stderr)
        return 2
    try:
        status = solve_tile(_json(Path(argv[0])))
    except (ValueError, OSError) as error:
        print("scroll refine tile: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(dict(status=status)))
    return 0 if status == "FIELD_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
