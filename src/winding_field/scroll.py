"""Whole-scroll fields from overlapping field-prior tiles, and full-resolution CT renders.

``winding-field scroll`` splits one scroll into overlapping ``run`` tiles on a single
global field grid, runs them, merges them into one chunk checkpoint that the existing
readers (``open_field``, ``prepared mesh``/``surfaces``) accept, and renders single
meshes against CT at a chosen pitch and level:

    plan      choose tile ROIs on one global grid and write TILES.json
    tiles     run/continue the planned tiles with bounded local concurrency
    merge     blend tiles into MERGED/field/CHECKPOINT.json, FIELD.json, LEVELS.json
    render-ct flatten and render one mesh bundle against CT at --render-pitch-um

Every tile is an ordinary ``winding-field run`` (imported fibres and Lasagna,
``--field-prior`` coarse field, ``--field-scope supported``); nothing here changes
how a tile is solved. The merge is a weighted blend, not a global solve: seams are
measured and reported, never certified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np

from .util import atomic_json, file_sha

PLAN_SCHEMA = "winding-scroll-plan-v1"
MERGE_SCHEMA = "winding-scroll-merge-v1"
FIELD_CHUNK = 64
SEAM_BINS = np.concatenate([np.linspace(0.0, 1.0, 1001), [2.0, 4.0, np.inf]])


def _json(path):
    return json.loads(Path(path).read_bytes())


def _integer_ratio(field_um, level_um):
    ratio = field_um / level_um
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > 1e-9 * ratio:
        raise ValueError(
            f"field spacing {field_um} µm must be a whole multiple of the CT level spacing "
            f"{level_um} µm so tiles share one grid"
        )
    return int(nearest)


# --- plan -------------------------------------------------------------------------------


def tile_ranges(nodes, core, overlap):
    """Inclusive node ranges per axis: cores partition [0, nodes); tiles add overlap."""
    if core % FIELD_CHUNK or overlap % FIELD_CHUNK or core <= 0 or overlap < 0:
        raise ValueError(f"tile and overlap nodes must be positive multiples of {FIELD_CHUNK}")
    out = []
    for start in range(0, nodes, core):
        end = min(nodes, start + core) - 1
        out.append(
            dict(
                core=(start, end),
                nodes=(max(0, start - overlap), min(nodes - 1, end + overlap)),
            )
        )
    if len(out) > 1 and out[-1]["nodes"][1] - out[-1]["nodes"][0] < 1:
        raise ValueError("last tile has fewer than two nodes; change --tile-nodes")
    return out


def global_grid(bounds, ratio, level_spacing_um, origin_xyz):
    """Global node grid over selected-level voxel bounds (ZYX, exclusive upper)."""
    lo = np.asarray(bounds[0::2], np.int64)
    hi = np.asarray(bounds[1::2], np.int64)
    if (hi <= lo).any():
        raise ValueError("bounds must be increasing z0 z1 y0 y1 x0 x1")
    shape = (hi - 1 - lo) // ratio + 1
    if (shape < 2).any():
        raise ValueError("bounds give fewer than two field nodes on an axis")
    spacing = float(level_spacing_um) * ratio
    origin = [float(origin_xyz[2 - i]) + int(lo[i]) * float(level_spacing_um) for i in range(3)]
    return dict(
        voxel_lo_zyx=lo.tolist(),
        ratio=int(ratio),
        shape_zyx=shape.tolist(),
        origin_zyx_um=origin,
        spacing_um=spacing,
    )


def _coarse_supported(coarse, grid, nodes_lo, nodes_hi):
    """True when the coarse field's observed support touches this tile's box."""
    if coarse is None:
        return True
    source, geometry = coarse
    spacing = np.asarray(geometry["spacing_xyz"], float)[::-1]
    origin = np.asarray(geometry["origin_xyz"], float)[::-1]
    lo_um = np.asarray(grid["origin_zyx_um"]) + np.asarray(nodes_lo) * grid["spacing_um"]
    hi_um = np.asarray(grid["origin_zyx_um"]) + np.asarray(nodes_hi) * grid["spacing_um"]
    first = np.maximum(0, np.floor((lo_um - origin) / spacing).astype(np.int64))
    last = np.minimum(
        np.asarray(source.shape) - 1, np.ceil((hi_um - origin) / spacing).astype(np.int64)
    )
    if (last < first).any():
        return False
    _, valid = source.read(first.tolist(), (last + 1).tolist())
    return bool(valid.any())


AFV_APPLICATION_ID = 1447249475


def split_afv(afv, boxes, out_dir, margin_um):
    """Write one AFV per tile with every fibre whose stored bounds touch its µm box.

    ``boxes`` maps tile id to (lo_xyz_um, hi_xyz_um). Fibres are copied whole with their
    blocks, bounds index and original metadata, so ``run`` still clips them exactly to
    its ROI; only the per-tile scan of the full scroll AFV is avoided. Counts in the
    copied header are updated and a ``scroll_split`` provenance key is added.
    """
    import sqlite3

    afv = Path(afv).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha(afv)
    uri = afv.as_uri() + "?mode=ro&immutable=1"
    src = sqlite3.connect(uri, uri=True)
    try:
        if src.execute("PRAGMA application_id").fetchone() != (AFV_APPLICATION_ID,):
            raise ValueError("not an AFV v1 file")
        header = {k: v for k, v in src.execute("SELECT key,value FROM metadata")}
        frame = json.loads(header.get("frame", "{}"))
        origin = np.asarray(frame.get("origin_xyz_um", [np.nan] * 3), float)
        spacing = np.asarray(frame.get("spacing_xyz_um", [np.nan] * 3), float)
        if not (np.isfinite(origin).all() and np.isfinite(spacing).all() and (spacing > 0).all()):
            raise ValueError("splitting needs an AFV with an embedded µm frame")
        schema = [
            sql
            for (name, sql) in src.execute(
                "SELECT name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid"
            )
            if not name.startswith("block_bounds_")
        ]
        has_bounds = any("block_bounds" in s for s in schema)
        # Bounds come from the block points themselves, not the stored fibre columns.
        extent = {}
        for fid, raw in src.execute("SELECT fiber_id,points FROM blocks"):
            points = np.frombuffer(raw, dtype="<f8").reshape(-1, 3)
            low, high = points.min(0), points.max(0)
            if fid in extent:
                extent[fid] = (np.minimum(extent[fid][0], low), np.maximum(extent[fid][1], high))
            else:
                extent[fid] = (low, high)
        bounds = np.array(
            [
                (fid, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
                for fid, (lo, hi) in sorted(extent.items())
            ],
            dtype=np.float64,
        ).reshape(-1, 7)
        if not np.isfinite(bounds).all():
            raise ValueError("nonfinite AFV block points")
        page_size = src.execute("PRAGMA page_size").fetchone()[0]
    finally:
        src.close()
    counts = {}
    for tile_id, (lo_um, hi_um) in boxes.items():
        lo = (np.asarray(lo_um) - margin_um - origin) / spacing
        hi = (np.asarray(hi_um) + margin_um - origin) / spacing
        pick = (
            (bounds[:, 2] >= lo[0]) & (bounds[:, 1] <= hi[0])
            & (bounds[:, 4] >= lo[1]) & (bounds[:, 3] <= hi[1])
            & (bounds[:, 6] >= lo[2]) & (bounds[:, 5] <= hi[2])
        )  # fmt: skip
        ids = bounds[pick, 0].astype(np.int64)
        target = out_dir / f"{tile_id}.afv"
        partial = out_dir / f".{tile_id}.afv.partial"
        partial.unlink(missing_ok=True)
        dst = sqlite3.connect(str(partial), uri=True)
        try:
            dst.execute(f"PRAGMA page_size={int(page_size)}")
            dst.execute("PRAGMA journal_mode=OFF")
            dst.execute("ATTACH DATABASE ? AS src", (uri,))
            for sql in schema:
                dst.execute(sql)
            dst.execute("CREATE TEMP TABLE pick(id INTEGER PRIMARY KEY)")
            dst.executemany("INSERT INTO pick VALUES (?)", ((int(i),) for i in ids))
            dst.execute("INSERT INTO metadata SELECT * FROM src.metadata")
            dst.execute("INSERT INTO fibers SELECT f.* FROM src.fibers f JOIN pick USING(id)")
            dst.execute(
                "INSERT INTO blocks SELECT b.* FROM src.blocks b "
                "JOIN pick p ON p.id=b.fiber_id ORDER BY b.id"
            )
            if has_bounds:
                dst.execute(
                    "INSERT INTO block_bounds SELECT bb.* FROM src.block_bounds bb "
                    "JOIN blocks b ON b.id=bb.id"
                )
            fibres, points = dst.execute(
                "SELECT count(*),coalesce(sum(point_count),0) FROM fibers"
            ).fetchone()
            blocks = dst.execute("SELECT count(*) FROM blocks").fetchone()[0]
            for key, value in (
                ("fiber_count", fibres),
                ("point_count", points),
                ("block_count", blocks),
                (
                    "scroll_split",
                    dict(
                        parent_sha256=source_sha,
                        tile=tile_id,
                        box_xyz_um=[list(map(float, lo_um)), list(map(float, hi_um))],
                        margin_um=margin_um,
                        rule="whole fibres whose stored bounds intersect the box",
                    ),
                ),
            ):
                dst.execute(
                    "INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, json.dumps(value))
                )
            dst.commit()
            dst.execute("DETACH DATABASE src")
            dst.execute(f"PRAGMA application_id={AFV_APPLICATION_ID}")
            dst.execute("PRAGMA user_version=1")
            dst.commit()
        finally:
            dst.close()
        os.replace(partial, target)
        counts[tile_id] = dict(path=str(target), fibres=int(fibres), blocks=int(blocks))
    return counts


def plan(args):
    from .surface_sources import FieldSource
    from .workflow import open_volume

    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("plan output must be a new directory")
    volume = open_volume(
        dict(
            source=args.source,
            voxel_um=args.voxel_um,
            level=args.level,
            cache=str(output.parent / (output.name + ".ct-cache")),
        )
    )
    try:
        spacing = np.asarray(volume.spacing_xyz, float)
        if not np.allclose(spacing, spacing[0], rtol=1e-9, atol=0):
            raise ValueError("scroll tiles require isotropic CT spacing")
        ratio = _integer_ratio(args.field_spacing_um, float(spacing[0]))
        stride = args.stride if args.stride is not None else ratio
        if ratio % stride:
            raise ValueError("--stride must divide the field/CT spacing ratio")
        shape = list(volume.shape)
        bounds = args.bounds or [0, shape[0], 0, shape[1], 0, shape[2]]
        for i in range(3):
            if not 0 <= bounds[2 * i] < bounds[2 * i + 1] <= shape[i]:
                raise ValueError("bounds outside the selected CT level")
        grid = global_grid(bounds, ratio, float(spacing[0]), volume.origin_xyz)
        ct = dict(
            source=str(args.source),
            voxel_um=args.voxel_um,
            level=args.level,
            level_spacing_um=float(spacing[0]),
            shape_zyx=shape,
            identity_sha256=hashlib.sha256(
                json.dumps(volume.identity, sort_keys=True, default=str).encode()
            ).hexdigest(),
        )
    finally:
        close = getattr(volume, "close", None)
        if close:
            close()
    coarse_dir = Path(args.coarse).resolve()
    coarse_meta = _json(coarse_dir / "FIELD.json")
    coarse = FieldSource(
        dict(
            kind="chunks",
            path=str(coarse_dir / "field/CHECKPOINT.json"),
            sha256=coarse_meta["checkpoint_sha256"],
            support_policy="observed",
        ),
        coarse_meta["geometry"],
        cache_bytes=256 << 20,
    )
    axes = [tile_ranges(n, args.tile_nodes, args.overlap_nodes) for n in grid["shape_zyx"]]
    tiles = []
    with coarse:
        for key in itertools.product(*(range(len(a)) for a in axes)):
            nodes_lo = [axes[i][k]["nodes"][0] for i, k in enumerate(key)]
            nodes_hi = [axes[i][k]["nodes"][1] for i, k in enumerate(key)]
            core_lo = [axes[i][k]["core"][0] for i, k in enumerate(key)]
            core_hi = [axes[i][k]["core"][1] for i, k in enumerate(key)]
            roi = []
            for i in range(3):
                start = grid["voxel_lo_zyx"][i] + nodes_lo[i] * ratio
                roi += [start, grid["voxel_lo_zyx"][i] + nodes_hi[i] * ratio + 1]
            tiles.append(
                dict(
                    id="-".join(f"{k:03d}" for k in key),
                    key=list(key),
                    nodes_lo=nodes_lo,
                    nodes_hi=nodes_hi,
                    core_lo=core_lo,
                    core_hi=core_hi,
                    roi=roi,
                    supported=_coarse_supported(
                        (coarse, coarse_meta["geometry"]), grid, nodes_lo, nodes_hi
                    ),
                )
            )
    run_args = [
        "--source", str(args.source),
        "--voxel-um", repr(args.voxel_um),
        "--level", str(args.level),
        "--stride", str(stride),
        "--field-spacing-um", repr(args.field_spacing_um),
        "--fibres", str(Path(args.fibres).resolve()),
        "--lasagna-input", str(Path(args.lasagna_input).resolve()),
        "--field-prior", str(coarse_dir),
        "--field-scope", "supported",
        "--sweeps", str(args.sweeps),
        "--device", args.device,
        "--threads", str(args.threads),
        "--import-workers", str(args.import_workers),
        "--until", "field",
    ] + list(args.run_arg or [])  # fmt: skip
    document = dict(
        schema=PLAN_SCHEMA,
        created_unix=time.time(),
        ct=ct,
        grid=grid,
        field_chunk=FIELD_CHUNK,
        tile_nodes=args.tile_nodes,
        overlap_nodes=args.overlap_nodes,
        coarse=dict(path=str(coarse_dir), checkpoint_sha256=coarse_meta["checkpoint_sha256"]),
        run_args=run_args,
        tiles=tiles,
    )
    output.mkdir(parents=True)
    supported_tiles = [t for t in tiles if t["supported"]]
    if args.split_fibres and supported_tiles:
        origin = np.asarray(grid["origin_zyx_um"])[::-1]
        boxes = {
            t["id"]: (
                origin + np.asarray(t["nodes_lo"])[::-1] * grid["spacing_um"],
                origin + np.asarray(t["nodes_hi"])[::-1] * grid["spacing_um"],
            )
            for t in supported_tiles
        }
        split = split_afv(args.fibres, boxes, output / "fibres", 2 * grid["spacing_um"])
        for tile in supported_tiles:
            tile["fibres"] = split[tile["id"]]
    atomic_json(output / "TILES.json", document)
    supported = len(supported_tiles)
    return dict(
        status="PLANNED",
        plan=str(output / "TILES.json"),
        grid_shape_zyx=grid["shape_zyx"],
        spacing_um=grid["spacing_um"],
        tiles=len(tiles),
        supported_tiles=supported,
    )


# --- tiles ------------------------------------------------------------------------------


def _tile_done(run):
    try:
        state = _json(run / "RUN.json")
    except (OSError, ValueError):
        return False
    return "field" in state.get("stages", {}) and (run / "FIELD.json").is_file()


def run_tiles(args):
    root = Path(args.plan).resolve()
    document = _json(root / "TILES.json")
    if document.get("schema") != PLAN_SCHEMA:
        raise ValueError("not a scroll tile plan")
    wanted = set(args.ids or [])
    queue = [t for t in document["tiles"] if t["supported"] and (not wanted or t["id"] in wanted)]
    for name in ("tiles", "logs", "status", "cache"):
        (root / name).mkdir(exist_ok=True)
    executable = (
        list(args.executable) if args.executable else [sys.executable, "-m", "winding_field"]
    )
    running = {}
    results = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for tile_id in running:
            (root / "tiles" / tile_id / "STOP").touch()

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        pending = list(queue)
        while pending or running:
            while pending and len(running) < args.jobs and not stopping:
                tile = pending.pop(0)
                run = root / "tiles" / tile["id"]
                if _tile_done(run):
                    results[tile["id"]] = "DONE"
                    continue
                if (run / "RUN.json").exists():
                    command = executable + ["field", str(run)]
                else:
                    roi = [str(v) for v in tile["roi"]]
                    run_args = list(document["run_args"])
                    if "fibres" in tile:
                        run_args[run_args.index("--fibres") + 1] = tile["fibres"]["path"]
                    command = executable + [
                        "run",
                        *run_args,
                        "--roi",
                        *roi,
                        "--cache",
                        str(
                            Path(args.cache).resolve()
                            if args.cache
                            else root / "cache" / tile["id"]
                        ),
                        "--output",
                        str(run),
                    ]
                log = open(root / "logs" / f"{tile['id']}.log", "ab")
                log.write(f"\n# {time.strftime('%FT%T')} {' '.join(command)}\n".encode())
                log.flush()
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                running[tile["id"]] = (process, log, time.time())
            if not running:
                break
            time.sleep(1.0)
            for tile_id, (process, log, started) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[tile_id]
                done = _tile_done(root / "tiles" / tile_id)
                state = "DONE" if done else ("STOPPED" if stopping else "FAILED")
                results[tile_id] = state
                atomic_json(
                    root / "status" / f"{tile_id}.json",
                    dict(status=state, exit_code=code, seconds=time.time() - started),
                )
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    counts = {s: sum(v == s for v in results.values()) for s in ("DONE", "FAILED", "STOPPED")}
    return dict(status="TILES_DONE" if counts["DONE"] == len(queue) else "INCOMPLETE", **counts)


# --- merge ------------------------------------------------------------------------------


def _ramp(index, lo, hi, last, width):
    """Linear cross-fade weight over ``width`` nodes on interior tile faces only."""
    w = np.ones(index.shape, np.float64)
    if lo > 0:
        w = np.minimum(w, (index - lo + 0.5) / width)
    if hi < last:
        w = np.minimum(w, (hi - index + 0.5) / width)
    return np.clip(w, 0.0, 1.0)


def _encode(values, observed):
    """The core BlockStore payload codec: float32 delta + float64 offset + packed bits."""
    offset = np.float64(values.mean())
    delta = (values - offset).astype(np.float32)
    restored = delta.astype(np.float64) + offset
    error = float(np.max(np.abs(restored - values)))
    if not (np.isfinite(delta).all() and np.isfinite(restored).all() and math.isfinite(error)):
        raise ValueError("nonfinite merged chunk encoding")
    stream = io.BytesIO()
    np.savez_compressed(stream, delta=delta, offset=offset, observed=np.packbits(observed))
    return stream.getvalue(), error


class _Tile:
    def __init__(self, plan_root, tile, grid):
        from .artifacts import Tiles
        from .surface_sources import FieldSource

        self.tile = tile
        self.run = Path(plan_root) / "tiles" / tile["id"]
        meta = _json(self.run / "FIELD.json")
        self.geometry = meta["geometry"]
        self.prepared = _json(self.run / "PREPARED.json")
        expected = [tile["nodes_hi"][i] - tile["nodes_lo"][i] + 1 for i in range(3)]
        if list(self.geometry["shape_zyx"]) != expected:
            raise ValueError(f"tile {tile['id']} field shape differs from its plan")
        offset_um = np.asarray(self.geometry["origin_xyz"], float)[::-1] - grid["origin_zyx_um"]
        offset = offset_um / grid["spacing_um"]
        if not np.allclose(offset, tile["nodes_lo"], atol=1e-6) or not np.allclose(
            self.geometry["spacing_xyz"], grid["spacing_um"], rtol=1e-12, atol=0
        ):
            raise ValueError(f"tile {tile['id']} is not on the plan grid")
        descriptor = dict(
            kind="chunks",
            path=str(self.run / "field/CHECKPOINT.json"),
            sha256=meta["checkpoint_sha256"],
        )
        self.stored = FieldSource(
            dict(descriptor, support_policy="stored"), self.geometry, cache_bytes=64 << 20
        )
        self.observed = FieldSource(
            dict(descriptor, support_policy="observed"), self.geometry, cache_bytes=64 << 20
        )
        self.maps = Tiles(self.run / "maps", readonly=True)
        self.lo = np.asarray(tile["nodes_lo"])
        self.hi = np.asarray(tile["nodes_hi"])
        self.core_lo = np.asarray(tile["core_lo"])
        self.core_hi = np.asarray(tile["core_hi"])
        manifest = _json(self.run / "field/CHECKPOINT.json")
        if manifest["chunk"] != FIELD_CHUNK:
            raise ValueError("tile field chunk must be 64")
        self.chunks = [tuple(map(int, k.split("-"))) for k in manifest["chunks"]]

    def close(self):
        self.stored.close()
        self.observed.close()
        self.maps.close()


CORRECTION_STEP = 4


def tile_correction(task):
    """Low-pass of (tile - coarse) on a CORRECTION_STEP-reduced grid.

    A tile is gauged by one prior node and relaxed for a few sweeps, so its low
    frequencies drift from the coarse field differently in every tile; neighbours then
    disagree by up to a winding in their overlap. Subtracting this smooth difference
    keeps the tile's detail below ``sigma`` and takes everything coarser from the one
    coarse solve that spans all tiles (a coarse-grid correction).
    """
    from scipy.ndimage import gaussian_filter

    from .field_prior import FieldPrior

    plan_root, tile_info, grid, coarse_dir, sigma_mm, out = task
    tile = _Tile(plan_root, tile_info, grid)
    step = CORRECTION_STEP
    try:
        prior = FieldPrior(coarse_dir, tile.geometry, tile.prepared.get("source"))
        shape = np.asarray(tile.geometry["shape_zyx"])
        reduced = (shape + step - 1) // step
        sums = np.zeros(reduced)
        counts = np.zeros(reduced)
        spacing = np.asarray(tile.geometry["spacing_xyz"], float)
        origin = np.asarray(tile.geometry["origin_xyz"], float)
        for key in tile.chunks:
            lo = np.asarray(key) * FIELD_CHUNK
            hi = np.minimum(lo + FIELD_CHUNK, shape)
            values, stored = tile.stored.read(lo.tolist(), hi.tolist())
            if not stored.any():
                continue
            index = np.stack(
                np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1
            )
            xyz = index[..., ::-1].reshape(-1, 3) * spacing + origin
            coarse, valid = prior.sample(xyz)
            coarse = np.asarray(coarse, float).reshape(values.shape)
            good = stored & np.asarray(valid, bool).reshape(values.shape) & np.isfinite(coarse)
            cells = tuple((index[..., i] // step)[good] for i in range(3))
            np.add.at(sums, cells, (values - coarse)[good])
            np.add.at(counts, cells, 1.0)
        prior.close()
        sigma = sigma_mm * 1000.0 / float(grid["spacing_um"]) / step
        weight = gaussian_filter(counts, sigma, mode="nearest")
        low = gaussian_filter(sums, sigma, mode="nearest") / np.maximum(weight, 1e-12)
        low[weight < 1e-9] = 0.0
        path = Path(out) / f"{tile_info['id']}.npz"
        np.savez(path, low=low, step=step)
        used = counts > 0
        return tile_info["id"], dict(
            path=str(path),
            sha256=file_sha(path),
            p95_abs_correction=float(np.percentile(np.abs(low[used]), 95)) if used.any() else 0.0,
        )
    finally:
        tile.close()


_WORKER = {}


def _spawn():
    # Fresh interpreters: forking a parent that already holds threads/SQLite can deadlock.
    import multiprocessing

    return multiprocessing.get_context("spawn")


def _worker_init(plan_root, tiles, grid, overlap, out, corrections=None):
    _WORKER.update(plan_root=plan_root, grid=grid, overlap=overlap, out=Path(out), open={})
    _WORKER["tiles"] = {t["id"]: t for t in tiles}
    _WORKER["corrections"] = corrections or {}
    _WORKER["low"] = {}


def _worker_tile(tile_id):
    opened = _WORKER["open"]
    if tile_id not in opened:
        opened[tile_id] = _Tile(_WORKER["plan_root"], _WORKER["tiles"][tile_id], _WORKER["grid"])
    return opened[tile_id]


def _correction(tile_id, local_lo, local_hi):
    """Trilinear value of a tile's reduced low-pass correction on local nodes."""
    record = _WORKER["corrections"].get(tile_id)
    if record is None:
        return 0.0
    if tile_id not in _WORKER["low"]:
        if file_sha(record["path"]) != record["sha256"]:
            raise ValueError("tile correction changed: " + tile_id)
        with np.load(record["path"], allow_pickle=False) as data:
            _WORKER["low"][tile_id] = (data["low"], int(data["step"]))
    from scipy.ndimage import map_coordinates

    low, step = _WORKER["low"][tile_id]
    axes = [(np.arange(a, b) - (step - 1) / 2) / step for a, b in zip(local_lo, local_hi)]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), 0)
    return map_coordinates(low, coords, order=1, mode="nearest")


def _merge_chunk(task):
    key, tile_ids = task
    grid, overlap = _WORKER["grid"], _WORKER["overlap"]
    shape = np.asarray(grid["shape_zyx"])
    lo = np.asarray(key) * FIELD_CHUNK
    hi = np.minimum(lo + FIELD_CHUNK, shape)
    size = tuple((hi - lo).tolist())
    total = np.zeros(size)
    weight = np.zeros(size)
    observed = np.zeros(size, bool)
    layers = []
    owners = []
    phase, phase_weight = 0j, 0.0
    low, high = math.inf, -math.inf
    width = 2 * overlap + 1
    for tile_id in tile_ids:
        tile = _worker_tile(tile_id)
        first = np.maximum(lo, tile.lo)
        last = np.minimum(hi, tile.hi + 1)
        if (last <= first).any():
            continue
        local_lo, local_hi = first - tile.lo, last - tile.lo
        values, stored = tile.stored.read(local_lo.tolist(), local_hi.tolist())
        _, seen = tile.observed.read(local_lo.tolist(), local_hi.tolist())
        if not stored.any():
            continue
        values = values - _correction(tile_id, local_lo, local_hi)
        axes = [np.arange(first[i], last[i]) for i in range(3)]
        w = (
            _ramp(axes[0], tile.lo[0], tile.hi[0], shape[0] - 1, width)[:, None, None]
            * _ramp(axes[1], tile.lo[1], tile.hi[1], shape[1] - 1, width)[None, :, None]
            * _ramp(axes[2], tile.lo[2], tile.hi[2], shape[2] - 1, width)[None, None, :]
        )
        w = np.where(stored, np.maximum(w, 1e-12), 0.0)
        region = tuple(slice(a - b, c - b) for a, b, c in zip(first, lo, last))
        total[region] += w * values
        weight[region] += w
        observed[region] |= seen & stored
        layer = np.full(size, np.nan)
        layer[region] = np.where(stored, values, np.nan)
        layers.append((tile_id, layer))
        # Sheet phase comes from the tile that owns these nodes (cores partition the grid).
        own_lo = np.maximum(first, tile.core_lo)
        own_hi = np.minimum(last, tile.core_hi + 1)
        if (own_hi > own_lo).all():
            from .field import map_axes

            meta = tile.prepared
            channel = "surface_phase" if meta.get("surface_method") == "lasagna" else "presence"
            presence = tile.maps.sample(
                (channel,), map_axes(meta, tile.geometry, own_lo - tile.lo, own_hi - tile.lo)
            )[channel]
            own = tuple(slice(a - b, c - b) for a, b, c in zip(own_lo, lo, own_hi))
            own_tile = tuple(slice(a - b, c - b) for a, b, c in zip(own_lo, first, own_hi))
            owners.append((own, presence, seen[own_tile] & stored[own_tile]))
    seams = {}
    for (a, la), (b, lb) in itertools.combinations(layers, 2):
        both = np.isfinite(la) & np.isfinite(lb)
        if both.any():
            diff = np.abs(la[both] - lb[both])
            seams[f"{a}|{b}"] = np.histogram(diff, SEAM_BINS)[0]
    if not weight.any():
        return key, None, seams, (phase, phase_weight, low, high)
    values = np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), 0.0)
    for own, presence, valid in owners:
        if valid.any():
            field = values[own][valid]
            wts = presence[valid]
            phase += complex(np.sum(wts * np.exp(2j * np.pi * field)))
            phase_weight += float(wts.sum())
            low, high = min(low, float(field.min())), max(high, float(field.max()))
    raw, error = _encode(values, observed)
    label = "-".join(map(str, key))
    name = f"{label}.g{uuid.uuid4().hex}.npz"
    path = _WORKER["out"] / name
    with open(path, "xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    record = dict(
        path=name,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        quantization_max=error,
    )
    return key, record, seams, (phase, phase_weight, low, high)


def _percentile(hist, q):
    total = hist.sum()
    if not total:
        return None
    index = int(np.searchsorted(np.cumsum(hist), q * total))
    return float(SEAM_BINS[min(index + 1, len(SEAM_BINS) - 1)])


def merge(args):
    root = Path(args.plan).resolve()
    document = _json(root / "TILES.json")
    if document.get("schema") != PLAN_SCHEMA:
        raise ValueError("not a scroll tile plan")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("merge output must be a new directory")
    grid = document["grid"]
    tiles = [t for t in document["tiles"] if t["supported"]]
    missing = [t["id"] for t in tiles if not _tile_done(root / "tiles" / t["id"])]
    if missing and not args.allow_missing:
        raise ValueError(f"{len(missing)} tiles have no completed field, e.g. {missing[:5]}")
    tiles = [t for t in tiles if t["id"] not in missing]
    geometries = {}
    tasks = {}
    for tile in tiles:
        probe = _Tile(root, tile, grid)
        try:
            geometries[tile["id"]] = probe.geometry
            for key in probe.chunks:
                node_lo = probe.lo + np.asarray(key) * FIELD_CHUNK
                node_hi = np.minimum(node_lo + FIELD_CHUNK, probe.hi + 1) - 1
                for merged in itertools.product(
                    *(
                        range(a // FIELD_CHUNK, b // FIELD_CHUNK + 1)
                        for a, b in zip(node_lo, node_hi)
                    )
                ):
                    tasks.setdefault(merged, set()).add(tile["id"])
        finally:
            probe.close()
    frames = {g["frame"] for g in geometries.values()}
    steps = {g["sheet_step"] for g in geometries.values()}
    units = {g["voxel_um"] for g in geometries.values()}
    if len(frames) != 1 or len(steps) != 1 or len(units) != 1:
        raise ValueError("tiles differ in frame, sheet step or units")
    # Every stored tile chunk also contributes to neighbours' overlaps.
    for merged in list(tasks):
        lo = np.asarray(merged) * FIELD_CHUNK
        hi = lo + FIELD_CHUNK - 1
        for tile in tiles:
            if (np.asarray(tile["nodes_lo"]) <= hi).all() and (
                np.asarray(tile["nodes_hi"]) >= lo
            ).all():
                tasks[merged].add(tile["id"])
    field_dir = output / "field"
    field_dir.mkdir(parents=True)
    # The same writer lease as the core BlockStore: readers (prepared mesh) take it shared.
    import fcntl

    lease = os.open(
        field_dir / ".writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    corrections = {}
    if args.coarse_sigma_mm > 0:
        correction_dir = output / "corrections"
        correction_dir.mkdir()
        coarse = document["coarse"]
        if file_sha(Path(coarse["path"]) / "field/CHECKPOINT.json") != coarse["checkpoint_sha256"]:
            raise ValueError("coarse field changed since the plan")
        jobs = [
            (str(root), t, grid, coarse["path"], args.coarse_sigma_mm, str(correction_dir))
            for t in tiles
        ]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=_spawn()) as pool:
            corrections = dict(pool.map(tile_correction, jobs))
    chunks = {}
    seams = {}
    phase, phase_weight, low, high = 0j, 0.0, math.inf, -math.inf
    ordered = sorted(tasks.items())
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=_spawn(),
        initializer=_worker_init,
        initargs=(
            str(root),
            tiles,
            grid,
            document["overlap_nodes"],
            str(field_dir),
            corrections,
        ),
    ) as pool:
        for key, record, chunk_seams, stats in pool.map(
            _merge_chunk, [(k, sorted(v)) for k, v in ordered], chunksize=4
        ):
            if record is not None:
                chunks["-".join(map(str, key))] = record
            for pair, hist in chunk_seams.items():
                seams[pair] = seams.get(pair, 0) + hist
            phase += stats[0]
            phase_weight += stats[1]
            low, high = min(low, stats[2]), max(high, stats[3])
    frame = frames.pop()
    sheet_step = steps.pop()
    geometry = dict(
        frame=frame,
        voxel_um=units.pop(),
        origin_xyz=grid["origin_zyx_um"][::-1],
        spacing_xyz=[grid["spacing_um"]] * 3,
        shape_zyx=grid["shape_zyx"],
        sheet_step=sheet_step,
    )
    identity = dict(
        schema=MERGE_SCHEMA,
        geometry=geometry,
        plan_sha256=file_sha(root / "TILES.json"),
        tiles={
            tid: _json(root / "tiles" / tid / "FIELD.json")["checkpoint_sha256"]
            for tid in geometries
        },
        blend="linear cross-fade over 2*overlap+1 nodes on interior tile faces",
        coarse_correction=dict(
            sigma_mm=args.coarse_sigma_mm,
            reduced_step=CORRECTION_STEP,
            coarse_checkpoint_sha256=document["coarse"]["checkpoint_sha256"],
            tiles={tid: c["sha256"] for tid, c in corrections.items()},
        )
        if corrections
        else None,
    )
    manifest = dict(
        identity=identity,
        shape=grid["shape_zyx"],
        chunk=FIELD_CHUNK,
        chunks=chunks,
        progress=dict(merged_chunks=len(chunks), physical_acceptance=False),
        initialization="weighted blend of field-prior tiles; not a global solve",
    )
    atomic_json(field_dir / "CHECKPOINT.json", manifest)
    os.close(lease)
    seam_report = {}
    for pair, hist in sorted(seams.items()):
        total = int(hist.sum())
        seam_report[pair] = dict(
            nodes=total,
            median_abs_dW=_percentile(hist, 0.5),
            p95_abs_dW=_percentile(hist, 0.95),
            over_half_winding=int(hist[SEAM_BINS[1:] > 0.5].sum()),
        )
    worst = max((v["p95_abs_dW"] or 0.0 for v in seam_report.values()), default=0.0)
    atomic_json(output / "SEAMS.json", dict(pairs=seam_report, worst_p95_abs_dW=worst))
    levels = []
    offset = None
    if phase_weight > 0 and math.isfinite(low) and high - low > 1e-6:
        offset = float(np.angle(phase) / (2 * np.pi) % 1)
        levels = (
            np.arange(math.ceil(low - offset + 1e-6), math.floor(high - offset - 1e-6) + 1) + offset
        ).tolist()
    atomic_json(
        output / "LEVELS.json",
        dict(
            levels=levels,
            phase_offset=offset,
            concentration=abs(phase) / max(phase_weight, 1e-12),
            raw_low=low if math.isfinite(low) else None,
            raw_high=high if math.isfinite(high) else None,
            sheet_step=sheet_step,
        ),
    )
    field_meta = dict(
        schema=MERGE_SCHEMA,
        status="FIELD_READY",
        geometry=geometry,
        checkpoint_sha256=file_sha(field_dir / "CHECKPOINT.json"),
        tiles=len(geometries),
        missing_tiles=missing,
        merged_chunks=len(chunks),
        worst_seam_p95_abs_dW=worst,
        coarse_correction=dict(
            sigma_mm=args.coarse_sigma_mm,
            p95_abs_correction={tid: c["p95_abs_correction"] for tid, c in corrections.items()},
        )
        if corrections
        else None,
        global_converged=False,
        physical_acceptance=False,
        numerical_status="blend of independently solved field-prior tiles; seams in SEAMS.json",
        seconds=time.time() - started,
    )
    atomic_json(output / "FIELD.json", field_meta)
    surface = dict(
        schema="winding-surfaces-v1",
        field=dict(
            kind="chunks",
            path=str(field_dir / "CHECKPOINT.json"),
            sha256=field_meta["checkpoint_sha256"],
            support_policy="observed",
        ),
        geometry=dict(
            geometry,
            calibration=f"raw field phi; W = phi / {sheet_step}; levels from Lasagna phase",
        ),
        levels=levels[:4096],
        execution=dict(tile=FIELD_CHUNK, level_batch=1, mesh_bytes=8 << 30),
    )
    atomic_json(output / "surface-run.json", surface)
    return dict(
        status="MERGED",
        output=str(output),
        merged_chunks=len(chunks),
        tiles=len(geometries),
        missing_tiles=len(missing),
        levels=len(levels),
        worst_seam_p95_abs_dW=worst,
    )


# --- render-ct --------------------------------------------------------------------------


def _indexed_mesh(reader):
    """(vertices_xyz, faces) of a single-level bundle, faces as 0-based vertex rows."""
    reader.validate()
    ids, vertices = [], []
    for row in reader.db.execute("SELECT id,x,y,z FROM vertices ORDER BY id"):
        ids.append(row[0])
        vertices.append(row[1:])
    ids = np.asarray(ids, np.int64)
    faces = np.asarray(
        reader.db.execute("SELECT a,b,c FROM faces ORDER BY id").fetchall(), np.int64
    )
    reader.validate()
    return np.asarray(vertices, np.float64), np.searchsorted(ids, faces)


def decimate_bundle(mesh, factor, output, *, sample=200_000, seed=0):
    """Edge-collapse (libigl qslim) a single-level bundle to 1/factor of its faces.

    Rendering samples CT at every pixel of the flattened surface, so triangles only need
    to follow the surface, not the field grid: on PHerc0175A 1/9 (95 um edges) moved the
    surface by at most 1.05 um while cutting flatten+assemble ~10x. The receipt reports
    the measured deviation of a sample of original vertices from the new surface.
    """
    import igl

    from .surface_mesh import MeshReader, import_mesh

    with MeshReader(mesh) as reader:
        meta = reader.manifest["metadata"]
        levels = reader.manifest["levels"]
        vertices, faces = _indexed_mesh(reader)
    if len(levels) != 1:
        raise ValueError("decimation needs a single-level bundle")
    target = max(1, len(faces) // factor)
    out_v, out_f, _, _ = igl.qslim(
        np.asfortranarray(vertices), np.asfortranarray(faces.astype(np.int32)), target
    )
    rng = np.random.default_rng(seed)
    probe = vertices[rng.choice(len(vertices), min(sample, len(vertices)), replace=False)]
    distance = np.sqrt(
        igl.point_mesh_squared_distance(probe, out_v, np.asarray(out_f, np.int32))[0]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "decimated.npz"
    np.savez(source, vertices_xyz=out_v, faces=np.asarray(out_f, np.int64))
    geometry = {
        k: meta[k]
        for k in ("frame", "voxel_um", "origin_xyz", "spacing_xyz", "shape_zyx", "sheet_step")
    }
    import_mesh({"path": str(source), "sha256": file_sha(source)}, output / "mesh", geometry)
    receipt = dict(
        factor=factor,
        source_level=levels[0]["level"],
        source_faces=len(faces),
        faces=int(len(out_f)),
        sampled_vertices=len(probe),
        deviation_um=dict(
            p50=float(np.median(distance)),
            p99=float(np.percentile(distance, 99)),
            max=float(distance.max()),
        ),
    )
    atomic_json(output / "DECIMATION.json", receipt)
    return output / "mesh", receipt


def mesh_ct_chunks(reader, volume, margin=1):
    """CT chunk indices (ZYX) that rasterizing and trilinear sampling the mesh can touch."""
    origin = np.asarray(volume.origin_xyz, float)
    spacing = np.asarray(volume.spacing_xyz, float)
    chunks = np.asarray(volume.chunks, np.int64)
    last = np.asarray(volume.shape, np.int64) - 1
    keys = set()
    for triangles, _ in reader.triangles(batch_size=65536):
        grid = (triangles - origin) / spacing
        lo = np.clip(np.floor(grid.min(1)).astype(np.int64)[:, ::-1] - margin, 0, last)
        hi = np.clip(np.ceil(grid.max(1)).astype(np.int64)[:, ::-1] + margin, 0, last)
        klo, khi = lo // chunks, hi // chunks
        single = (klo == khi).all(1)
        keys.update(map(tuple, np.unique(klo[single], axis=0).tolist()))
        for a, b in zip(klo[~single].tolist(), khi[~single].tolist()):
            keys.update(itertools.product(*(range(x, y + 1) for x, y in zip(a, b))))
    return sorted(keys)


def prefetch_ct(volume, keys, workers):
    """Fill the CT cache with ``keys`` using parallel connections; returns bytes fetched."""
    from concurrent.futures import ThreadPoolExecutor

    separator = volume.metadata.get("dimension_separator", ".")
    prefix = volume.array_path + "/" if volume.array_path else ""

    def fetch(key):
        raw = volume._native_bytes(prefix + separator.join(map(str, key)))
        return 0 if raw is None else len(raw)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(fetch, keys))


def render_ct(args):
    import shutil

    from .native_pipeline import render_mesh
    from .surface_mesh import MeshReader
    from .workflow import open_volume

    config = dict(
        source=args.source,
        voxel_um=args.voxel_um,
        level=args.level,
        local=args.local,
        cache=args.cache,
        offline=args.offline,
        render_pitch_um=args.render_pitch_um,
        chart_faces=args.chart_faces,
        flattening_method=args.flattening_method,
        surface_time_limit_s=args.surface_time_limit,
        coverage_target=args.coverage_target,
        export_tifxyz=args.export_tifxyz,
        render_keep_xyz=args.export_tifxyz,
        render_tiled=not args.whole_png,
    )
    if args.cache_gib is not None:
        config["cache_bytes"] = int(args.cache_gib * (1 << 30))
    output = Path(args.output)
    mesh = Path(args.mesh)
    receipts = {}
    started = time.time()
    source_levels = _json(mesh / "MESH.json")["levels"]
    source_faces = sum(level["faces"] for level in source_levels)
    if args.decimate > 1 and source_faces < args.decimate_min_faces:
        # Small meshes are cheap to flatten as they are, and collapsing a few thousand
        # faces to a few hundred left PHerc0175A edge levels with no mappable chart.
        receipts["decimation"] = dict(
            factor=1, requested_factor=args.decimate, source_faces=source_faces,
            source_level=source_levels[0]["level"],
            reason=f"fewer than {args.decimate_min_faces} faces",
        )  # fmt: skip
    elif args.decimate > 1:
        mesh, receipts["decimation"] = decimate_bundle(
            mesh, args.decimate, output.parent / (output.name + ".decimated")
        )
        receipts["decimation"]["seconds"] = time.time() - started
    with MeshReader(mesh) as reader:
        volume = open_volume({k: v for k, v in config.items() if v is not None})
        if volume.remote and not args.offline and args.prefetch_workers > 0:
            # Prefetch every touched chunk over parallel connections, then render offline:
            # an online render re-verifies each chunk against the source per epoch, one
            # HTTPS connection at a time (53 of 77 s on a PHerc0175A winding).
            t = time.time()
            keys = mesh_ct_chunks(reader, volume)
            need = len(keys) * math.prod(volume.chunks) * volume.dtype.itemsize
            config["cache_bytes"] = max(config.get("cache_bytes", 2 << 30), int(need * 1.25))
            volume.close()
            volume = open_volume({k: v for k, v in config.items() if v is not None})
            free = shutil.disk_usage(volume.cache).free
            if need > free:
                raise ValueError(f"CT prefetch needs {need >> 30} GiB, {free >> 30} GiB free")
            fetched = prefetch_ct(volume, keys, args.prefetch_workers)
            volume.close()
            config["offline"] = True
            volume = open_volume({k: v for k, v in config.items() if v is not None})
            receipts["prefetch"] = dict(
                chunks=len(keys), fetched_bytes=fetched, seconds=time.time() - t
            )
        try:
            result = render_mesh(reader, output, volume, config)
        finally:
            volume.close()
    if receipts:
        result["scroll_render"] = receipts
        atomic_json(output / "SCROLL-RENDER.json", receipts)
    return result


# --- site -------------------------------------------------------------------------------


def build_pyramid(component, output, name):
    """Deep Zoom (DZI) pyramid from a tiled render component's per-tile PNGs."""
    from PIL import Image

    component, output = Path(component), Path(output)
    grid = _json(component / "RENDER.json")["grid"]
    height, width = grid["shape_yx"]
    size = grid["tile_pixels"]
    top = max(0, math.ceil(math.log2(max(width, height))))
    files = output / f"{name}_files"
    level = files / str(top)
    level.mkdir(parents=True, exist_ok=True)
    for png in (component / "png").glob("*.png"):
        y, x = (int(v) for v in png.stem.split("-"))
        target = level / f"{x // size}_{y // size}.png"
        if not target.exists():
            target.write_bytes(png.read_bytes())
    w, h = width, height
    for index in range(top - 1, -1, -1):
        source, target_dir = files / str(index + 1), files / str(index)
        target_dir.mkdir(exist_ok=True)
        w, h = math.ceil(w / 2), math.ceil(h / 2)
        for col in range(math.ceil(w / size)):
            for row in range(math.ceil(h / size)):
                target = target_dir / f"{col}_{row}.png"
                if target.exists():
                    continue
                canvas = Image.new("LA", (2 * size, 2 * size))
                found = False
                for dc in (0, 1):
                    for dr in (0, 1):
                        child = source / f"{2 * col + dc}_{2 * row + dr}.png"
                        if child.exists():
                            with Image.open(child) as image:
                                canvas.paste(image, (dc * size, dr * size))
                            found = True
                if found:
                    tile = canvas.resize((size, size), Image.BOX).crop(
                        (0, 0, min(size, w - col * size), min(size, h - row * size))
                    )
                    tile.save(target)
    (output / f"{name}.dzi").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{size}" '
        f'Overlap="0" Format="png"><Size Width="{width}" Height="{height}"/></Image>\n'
    )
    return dict(dzi=f"{name}.dzi", width=width, height=height, tile=size)


SITE_HTML = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{margin:0;font:14px system-ui,sans-serif;display:flex;height:100vh;background:#111;color:#ddd}}
#list{{width:220px;overflow:auto;border-right:1px solid #333}}
#list a{{display:block;padding:6px 10px;color:#ddd;text-decoration:none;font-variant-numeric:tabular-nums}}
#list a.on{{background:#2a4a7a}} #list a small{{color:#999}} #view{{flex:1}}
h1{{font-size:15px;margin:10px}}
</style>
<div id="list"><h1>{title}</h1></div><div id="view"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
<script>
const levels = {levels};
const viewer = OpenSeadragon({{id:"view", prefixUrl:
  "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/", showNavigator:true,
  maxZoomPixelRatio:4}});
const list = document.getElementById("list");
function show(i){{
  document.querySelectorAll("#list a").forEach((a,j)=>a.classList.toggle("on",j===i));
  const l = levels[i];
  // Inline Deep Zoom description: no XHR for the .dzi (works from file:// too).
  viewer.open({{Image:{{xmlns:"http://schemas.microsoft.com/deepzoom/2008",
    Url:l.dzi.replace(/\\.dzi$/,"_files/"), Format:"png", Overlap:"0", TileSize:String(l.tile),
    Size:{{Width:String(l.width), Height:String(l.height)}}}}}});
  location.hash = i;
}}
levels.forEach((l,i)=>{{const a=document.createElement("a"); a.href="#"+i;
  a.innerHTML="W "+l.w.toFixed(2)+" <small>"+l.status+" "+(100*l.coverage).toFixed(1)+"%</small>";
  a.onclick=e=>{{e.preventDefault(); show(i);}}; list.appendChild(a);}});
if (levels.length) show(Math.min(levels.length-1, parseInt(location.hash.slice(1))||0));
</script>
"""


def site(args):
    """Pyramids for every rendered level/component and one viewer page for the scroll."""
    renders, output = Path(args.renders), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    levels = []
    for level_dir in sorted(p for p in renders.iterdir() if (p / "LEVEL.json").is_file()):
        record = _json(level_dir / "LEVEL.json")
        extra = level_dir / "SCROLL-RENDER.json"
        w = (
            _json(extra)["decimation"]["source_level"]
            if extra.is_file() and "decimation" in _json(extra)
            else record["components"][0]["grid"]["level"]
        )
        for component in record["components"]:
            folder = level_dir / component["path"]
            if not (folder / "png").is_dir():
                continue
            name = f"{level_dir.name}-{component['path']}"
            pyramid = build_pyramid(folder, output / "dzi", name)
            levels.append(
                dict(
                    w=float(w),
                    status=record["status"],
                    coverage=float(record.get("coverage_fraction", 0.0)),
                    dzi="dzi/" + pyramid["dzi"],
                    width=pyramid["width"],
                    height=pyramid["height"],
                    tile=pyramid["tile"],
                    source=str(level_dir),
                )
            )
    levels.sort(key=lambda item: item["w"])
    atomic_json(output / "levels.json", levels)
    (output / "index.html").write_text(
        SITE_HTML.format(title=args.title, levels=json.dumps(levels, allow_nan=False))
    )
    return dict(status="SITE_READY", output=str(output), levels=len(levels))


# --- CLI --------------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser(prog="winding-field scroll", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("plan", help="choose overlapping tile ROIs on one global field grid")
    p.add_argument("--source", required=True)
    p.add_argument("--voxel-um", type=float, required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--stride", type=int, help="observation stride (default: field/CT ratio)")
    p.add_argument("--field-spacing-um", type=float, required=True)
    p.add_argument("--bounds", type=int, nargs=6, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
    p.add_argument("--tile-nodes", type=int, default=512, help="core nodes per axis (×64)")
    p.add_argument("--overlap-nodes", type=int, default=64, help="extra nodes per side (×64)")
    p.add_argument("--coarse", required=True, help="completed coarse field run (the prior)")
    p.add_argument("--fibres", required=True)
    p.add_argument("--lasagna-input", required=True)
    p.add_argument("--sweeps", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--import-workers", type=int, default=0)
    p.add_argument("--run-arg", action="append", help="extra flag passed to every tile run")
    p.add_argument(
        "--no-split-fibres",
        dest="split_fibres",
        action="store_false",
        help="give every tile the full AFV instead of a per-tile copy of its fibres",
    )
    p.add_argument("--output", required=True)

    p = sub.add_parser("tiles", help="run or continue the planned tiles")
    p.add_argument("plan")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--executable", nargs="+", help="command prefix (default: this Python)")
    p.add_argument(
        "--cache",
        help="one CT cache shared by every tile (e.g. a prefetched level, with --run-arg "
        "--offline at plan time); default: a private cache per tile",
    )

    p = sub.add_parser("merge", help="blend completed tiles into one field checkpoint")
    p.add_argument("plan")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument(
        "--coarse-sigma-mm",
        type=float,
        default=0.5,
        help="replace each tile's (tile - coarse) content above this Gaussian scale by the "
        "coarse field before blending (0 disables)",
    )

    p = sub.add_parser("render-ct", help="flatten and render one mesh bundle against CT")
    p.add_argument("--mesh", required=True, help="single-level mesh bundle directory")
    p.add_argument("--source", required=True)
    p.add_argument("--voxel-um", type=float)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--local")
    p.add_argument("--cache")
    p.add_argument("--cache-gib", type=float)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--render-pitch-um", type=float, required=True)
    p.add_argument("--chart-faces", type=int, default=1024)
    p.add_argument("--flattening-method", default="slim", choices=("slim", "default"))
    p.add_argument("--surface-time-limit", type=float, default=1800.0)
    p.add_argument("--coverage-target", type=float, default=1.0)
    p.add_argument(
        "--export-tifxyz",
        action="store_true",
        help="also write TIFXYZ; keeps per-pixel XYZ, otherwise only CT and mask are stored",
    )
    p.add_argument(
        "--whole-png",
        action="store_true",
        help="write whole-canvas sheet/mask PNGs (100M-pixel bound) instead of per-tile PNGs",
    )
    p.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="collapse the mesh to 1/N of its faces before flattening (9 is ~1 um on 175A)",
    )
    p.add_argument(
        "--decimate-min-faces",
        type=int,
        default=50000,
        help="render meshes with fewer faces without decimation",
    )
    p.add_argument(
        "--prefetch-workers",
        type=int,
        default=16,
        help="parallel CT downloads before an offline render (0: render online as before)",
    )
    p.add_argument("--output", required=True)

    p = sub.add_parser("site", help="Deep Zoom pyramids and one viewer page for a scroll")
    p.add_argument("renders", help="directory of render-ct outputs (one per level)")
    p.add_argument("--title", default="scroll")
    p.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.action == "plan":
            result = plan(args)
        elif args.action == "tiles":
            result = run_tiles(args)
        elif args.action == "merge":
            result = merge(args)
        elif args.action == "site":
            result = site(args)
        else:
            result = render_ct(args)
    except (ValueError, OSError) as error:
        print("scroll: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, default=float))
    return 0 if result.get("status") not in ("INCOMPLETE",) else 3


if __name__ == "__main__":
    raise SystemExit(main())
