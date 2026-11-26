"""Whole-scroll fields from overlapping field-prior tiles, and full-resolution CT renders.

``winding-field scroll`` splits one scroll into overlapping ``run`` tiles on a single
global field grid, runs them, merges them into one chunk checkpoint that the existing
readers (``open_field``, ``prepared mesh``/``surfaces``) accept, and renders single
meshes against CT at a chosen pitch and level:

    counted-density  sheet density counted on Lasagna cos (October recipe) as a manifest
    plan      choose tile ROIs on one global grid and write TILES.json
    tiles     run/continue the planned tiles with bounded local concurrency
    merge     blend tiles into MERGED/field/CHECKPOINT.json, FIELD.json, LEVELS.json
    refine    solve again the tiles whose seams stay open, their faces held at the merge
    render-ct flatten and render one mesh bundle against CT at --render-pitch-um

Every tile is an ordinary ``winding-field run`` (imported fibres and Lasagna,
``--field-prior`` coarse field, ``--field-scope supported``); nothing here changes
how a tile is solved. The merge is not a global solve: tiles are first levelled against
each other on the nodes they share (one constant and one tilt per tile, all overlaps
solved together), then blended. Seams are measured and reported, never certified.
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
JOINS = ("overlap", "coarse", "none")
FIELD_CHUNK = 64
# Overlaps need not fill whole chunks: the merge reads tiles at any node offset. 32 nodes
# per side (1.1 mm at 34.56 um) joined a 22 mm PHerc0175A box to 0.14 sheets (median)
# with only 32 shared nodes in all; 64 costs ~1.7x the node solves of 32.
OVERLAP_STEP = 16
DEFAULT_OVERLAP = 32
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
    """Inclusive node ranges per axis: cores partition [0, nodes); tiles add overlap.

    ``core`` is the largest core. The axis takes as few tiles as that allows, and their
    cores share its 64-node chunks evenly, the larger ones last (the last chunk may be
    partial). A plain ``core`` + remainder split left a narrow tile beside a wide one:
    on the 2026-11-25 PHerc0175A dry-run box (256 nodes, overlap 32), cores 192 + 64
    (tile edges 224 / 96) put 9.31% of seam nodes over half a winding apart, and cores
    128 + 128 (edges 160 / 160) 1.77%.
    """
    if core % FIELD_CHUNK or core <= 0:
        raise ValueError(f"tile nodes must be positive multiples of {FIELD_CHUNK}")
    if overlap % OVERLAP_STEP or overlap < 0:
        raise ValueError(f"overlap nodes must be nonnegative multiples of {OVERLAP_STEP}")
    count = -(-nodes // core)
    base, extra = divmod(-(-nodes // FIELD_CHUNK), count)
    starts = [0]
    for k in range(count - 1):
        starts.append(starts[-1] + (base + (k >= count - extra)) * FIELD_CHUNK)
    out = []
    for k, start in enumerate(starts):
        end = (starts[k + 1] if k + 1 < count else nodes) - 1
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


# --- tile size from GPU memory ------------------------------------------------------------

# A tile solved as one block holds its whole system on one GPU. Model, in float64: 121 bytes
# per node and 419 per fibre constraint row (the one-block PHerc0175A coarse), 25 %
# allocator overhead, plus the CUDA context. Fibre rows per node grow with the square of
# the field spacing (rows ~ fibre length / spacing, nodes ~ volume / spacing^3); at 34.56 um
# the densest measured PHerc0175A tile had 0.045 (752,370 rows on 256^3 nodes, 3.1 GB peak
# on the GPU), taken here with 1.5x margin. The model predicts 3.6 GB for that tile.
BYTES_PER_NODE = 121.0
BYTES_PER_FIBRE_ROW = 419.0
FIBRE_ROWS_PER_NODE_34UM = 0.045 * 1.5
ALLOCATOR_OVERHEAD = 1.25
GPU_RESERVE_BYTES = 0.5e9
GPU_USABLE = 0.85
# One block per tile: the solver settings the 34.56 um tiles were measured with.
ONE_BLOCK_RUN_ARGS = ("--rtol", "1e-7", "--maxiter", "50000", "--nonconvergence-policy", "continue")
# AMG (smoothed aggregation, built on the solve device) adds its hierarchy to the Jacobi
# footprint. Measured on a real 192^3 PHerc0175A tile (fibre weight 1000, one block, rtol
# 1e-7, GTX 1660): AMG peaked at 620 bytes per node on the GPU and 1,386 on the host
# (Jacobi: 183 and 415), and needed 618 CG iterations against Jacobi's 5,586, but each
# V-cycle iteration costs ~12x a Jacobi one: 196 s of solve against 142 s. Strong fibre
# pairs weaken the aggregation (41 iterations on a 128^3 tile with weak fibres), so Jacobi
# is the default and AMG stays opt-in (--tile-preconditioner auto or amg).
AMG_HIERARCHY_BYTES_PER_NODE = 350.0
# Below this tile edge the AMG setup is not paid back: auto uses Jacobi for smaller tiles.
AMG_MIN_TILE_NODES = 192
TILE_PRECONDITIONERS = ("auto", "amg", "jacobi")
TILE_KERNELS = ("eager", "fused")


def tile_bytes_per_node(spacing_um, preconditioner="jacobi"):
    rows = FIBRE_ROWS_PER_NODE_34UM * (float(spacing_um) / 34.56) ** 2
    extra = AMG_HIERARCHY_BYTES_PER_NODE if preconditioner == "amg" else 0.0
    return ALLOCATOR_OVERHEAD * (BYTES_PER_NODE + extra + BYTES_PER_FIBRE_ROW * rows)


def auto_tile_nodes(memory_bytes, spacing_um, overlap, preconditioner="jacobi"):
    """Largest core (a multiple of 64 nodes) whose tile, core + 2*overlap nodes per axis,
    fits ``memory_bytes`` of GPU memory when solved as one block with ``preconditioner``."""
    per_node = tile_bytes_per_node(spacing_um, preconditioner)
    usable = GPU_USABLE * float(memory_bytes) - GPU_RESERVE_BYTES
    edge = int((max(usable, 0.0) / per_node) ** (1.0 / 3.0) + 1e-9)
    core = (edge - 2 * overlap) // FIELD_CHUNK * FIELD_CHUNK
    if core < FIELD_CHUNK:
        raise ValueError(
            f"{memory_bytes / 2**30:.1f} GiB per tile cannot hold a {FIELD_CHUNK}-node core "
            f"plus {overlap} overlap nodes per side as one block; use a larger GPU, fewer "
            "--jobs-per-gpu, a smaller --overlap-nodes or an explicit --tile-nodes"
        )
    tile = core + 2 * overlap
    return core, dict(
        preconditioner=preconditioner,
        bytes_per_node=per_node,
        tile_nodes_per_axis=tile,
        estimated_peak_bytes=tile**3 * per_node + GPU_RESERVE_BYTES,
    )


def auto_tile_size(memory_bytes, spacing_um, overlap, preconditioner="auto"):
    """Tile core and sizing for one GPU share. ``auto``: AMG with the largest tile that fits
    its memory model, as long as the tile is at least AMG_MIN_TILE_NODES per axis; below
    that, Jacobi with the largest tile that fits Jacobi's (smaller) footprint."""
    if preconditioner not in TILE_PRECONDITIONERS:
        raise ValueError(f"tile preconditioner must be one of {', '.join(TILE_PRECONDITIONERS)}")
    if preconditioner != "auto":
        return auto_tile_nodes(memory_bytes, spacing_um, overlap, preconditioner)
    try:
        core, sizing = auto_tile_nodes(memory_bytes, spacing_um, overlap, "amg")
    except ValueError:
        core, sizing = None, None
    if core is not None and sizing["tile_nodes_per_axis"] >= AMG_MIN_TILE_NODES:
        return core, sizing
    return auto_tile_nodes(memory_bytes, spacing_um, overlap, "jacobi")


def visible_gpus():
    """[(id, total memory bytes)] of the GPUs a child process may use.

    Queried in a short-lived interpreter so this process never holds a CUDA context.
    Ids follow CUDA_VISIBLE_DEVICES when it is set.
    """
    code = (
        "import json, torch\n"
        "n = torch.cuda.device_count() if torch.cuda.is_available() else 0\n"
        "print(json.dumps([torch.cuda.get_device_properties(i).total_memory for i in range(n)]))"
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True
        )
        memory = [int(v) for v in json.loads(done.stdout.strip().splitlines()[-1])]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = (
        [v.strip() for v in visible.split(",") if v.strip()]
        if visible
        else [str(i) for i in range(len(memory))]
    )
    return list(zip(ids, memory))


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
        # Hold the decoded coarse in memory across tiles: a one-block coarse is a single
        # chunk (PHerc0175A: 625 x 404 x 482 nodes, ~1.1 GB decoded); with a smaller budget
        # every tile decoded it again (336 tiles: 48 min of planning instead of seconds).
        cache_bytes=4 << 30,
    )
    tile_memory = None
    tile_nodes = args.tile_nodes
    if str(tile_nodes) == "auto":
        if args.gpu_memory_gib:
            memory = float(args.gpu_memory_gib) * 2**30
            source = "--gpu-memory-gib"
        else:
            gpus = visible_gpus()
            if not gpus:
                raise ValueError(
                    "--tile-nodes auto needs a visible GPU or --gpu-memory-gib (the memory of "
                    "each GPU that will run the tiles)"
                )
            memory = min(m for _, m in gpus)
            source = f"smallest of {len(gpus)} visible GPUs"
        if args.jobs_per_gpu < 1:
            raise ValueError("--jobs-per-gpu must be at least 1")
        tile_nodes, sizing = auto_tile_size(
            memory / args.jobs_per_gpu,
            grid["spacing_um"],
            args.overlap_nodes,
            args.tile_preconditioner,
        )
        tile_memory = dict(
            mode="auto",
            gpu_memory_bytes=int(memory),
            source=source,
            jobs_per_gpu=int(args.jobs_per_gpu),
            core_nodes=int(tile_nodes),
            **sizing,
        )
    else:
        tile_nodes = int(tile_nodes)
    # Fixed-size tiles are solved in 64-node blocks, where the AMG setup does not pay off.
    preconditioner = (
        tile_memory["preconditioner"]
        if tile_memory
        else ("jacobi" if args.tile_preconditioner == "auto" else args.tile_preconditioner)
    )
    axes = [tile_ranges(n, tile_nodes, args.overlap_nodes) for n in grid["shape_zyx"]]
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
        # Tiles orient sheets from the scroll centre and use the coarse's spacing.
        "--field-prior-calibration", "inherit",
        "--sweeps", str(1 if tile_memory else args.sweeps),
        "--device", args.device,
        "--threads", str(args.threads),
        "--import-workers", str(args.import_workers),
        "--until", "field",
    ]  # fmt: skip
    if tile_memory:
        # One block per tile: the largest tile edge is the solver block.
        edge = max(t["nodes_hi"][i] - t["nodes_lo"][i] + 1 for t in tiles for i in range(3))
        run_args += ["--field-chunk", str(edge), *ONE_BLOCK_RUN_ARGS]
    if preconditioner == "amg":
        run_args += ["--preconditioner", "amg"]
    if args.tile_kernel == "fused":
        run_args += ["--field-kernel", "fused"]
    run_args += list(args.run_arg or [])
    document = dict(
        schema=PLAN_SCHEMA,
        created_unix=time.time(),
        ct=ct,
        grid=grid,
        field_chunk=FIELD_CHUNK,
        tile_nodes=tile_nodes,
        overlap_nodes=args.overlap_nodes,
        tile_memory=tile_memory,
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


def _run_until(run_args):
    """The stage a tile run stops after: the last --until wins, as in argparse."""
    until = "field"
    for k, arg in enumerate(run_args):
        if arg == "--until" and k + 1 < len(run_args):
            until = run_args[k + 1]
        elif arg.startswith("--until="):
            until = arg.split("=", 1)[1]
    return until


def _tile_prepared(run, until):
    """True when a plan that stops before the field (e.g. --run-arg=--until=fibres) left
    this tile at that stage: its inputs are ready for `scroll tiles` to solve later."""
    if until == "field":
        return False
    try:
        state = _json(run / "RUN.json")
    except (OSError, ValueError):
        return False
    return until in state.get("stages", {})


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
    run_args = list(document.get("run_args", []))
    on_cpu = "--device" in run_args and run_args[run_args.index("--device") + 1].startswith("cpu")
    if args.gpus is not None:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        gpus = [] if on_cpu else [g for g, _ in visible_gpus()]
    until = _run_until(run_args)
    per_gpu = int((document.get("tile_memory") or {}).get("jobs_per_gpu", 1))
    jobs = args.jobs or (len(gpus) * per_gpu if gpus else 1)
    # Each tile process sees one GPU; tiles go to the least loaded one.
    load = {g: 0 for g in gpus}
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
            while pending and len(running) < jobs and not stopping:
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
                gpu, env = None, None
                if gpus:
                    gpu = min(gpus, key=lambda g: load[g])
                    load[gpu] += 1
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                    log.write(f"# GPU {gpu}\n".encode())
                    log.flush()
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[tile["id"]] = (process, log, time.time(), gpu)
            if not running:
                break
            time.sleep(1.0)
            for tile_id, (process, log, started, gpu) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[tile_id]
                if gpu is not None:
                    load[gpu] -= 1
                run = root / "tiles" / tile_id
                if _tile_done(run):
                    state = "DONE"
                elif code == 0 and _tile_prepared(run, until):
                    state = "PREPARED"
                else:
                    state = "STOPPED" if stopping else "FAILED"
                results[tile_id] = state
                atomic_json(
                    root / "status" / f"{tile_id}.json",
                    dict(status=state, exit_code=code, seconds=time.time() - started),
                )
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    counts = {
        s: sum(v == s for v in results.values()) for s in ("DONE", "PREPARED", "FAILED", "STOPPED")
    }
    if counts["DONE"] == len(queue):
        status = "TILES_DONE"
    elif counts["PREPARED"] and counts["DONE"] + counts["PREPARED"] == len(queue):
        # Every tile stopped where the plan asked (--until): rerun to solve them.
        status = "TILES_PREPARED"
    else:
        status = "INCOMPLETE"
    return dict(
        status=status,
        jobs=jobs,
        gpus=gpus,
        **counts,
    )


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
    """A solved tile, read on its plan box (``nodes_lo``..``nodes_hi``).

    A tile may be solved on a larger box than its plan gives it, to move a free face away
    from the nodes it contributes: every reader here indexes the plan box, and the nodes
    outside it take no part in the join, the blend or the seam statistics.

    ``tile["field_run"]`` names the directory of a field solved apart from the prepared
    tile (a ``scroll refine`` pass): the maps still come from the plan's tile.
    """

    def __init__(self, plan_root, tile, grid):
        from .artifacts import Tiles
        from .surface_sources import FieldSource

        self.tile = tile
        self.run = Path(plan_root) / "tiles" / tile["id"]
        self.field_run = Path(tile["field_run"]) if tile.get("field_run") else self.run
        meta = _json(self.field_run / "FIELD.json")
        self.geometry = meta["geometry"]
        self.prepared = _json(self.run / "PREPARED.json")
        self.lo = np.asarray(tile["nodes_lo"])
        self.hi = np.asarray(tile["nodes_hi"])
        offset_um = np.asarray(self.geometry["origin_xyz"], float)[::-1] - grid["origin_zyx_um"]
        offset = offset_um / grid["spacing_um"]
        self.solved_lo = np.rint(offset).astype(np.int64)
        self.solved_hi = self.solved_lo + np.asarray(self.geometry["shape_zyx"]) - 1
        if not np.allclose(offset, self.solved_lo, atol=1e-6) or not np.allclose(
            self.geometry["spacing_xyz"], grid["spacing_um"], rtol=1e-12, atol=0
        ):
            raise ValueError(f"tile {tile['id']} is not on the plan grid")
        if (self.solved_lo > self.lo).any() or (self.solved_hi < self.hi).any():
            raise ValueError(f"tile {tile['id']} field does not cover its plan box")
        # Plan-box node -> node of the stored field.
        self.shift = self.lo - self.solved_lo
        self.grown = bool(self.shift.any() or (self.solved_hi > self.hi).any())
        descriptor = dict(
            kind="chunks",
            path=str(self.field_run / "field/CHECKPOINT.json"),
            sha256=meta["checkpoint_sha256"],
        )
        self.stored = FieldSource(
            dict(descriptor, support_policy="stored"), self.geometry, cache_bytes=64 << 20
        )
        self.observed = FieldSource(
            dict(descriptor, support_policy="observed"), self.geometry, cache_bytes=64 << 20
        )
        self.maps = Tiles(self.run / "maps", readonly=True)
        self.core_lo = np.asarray(tile["core_lo"])
        self.core_hi = np.asarray(tile["core_hi"])
        manifest = _json(self.field_run / "field/CHECKPOINT.json")
        self.chunk = int(manifest["chunk"])
        self.chunks = [tuple(map(int, k.split("-"))) for k in manifest["chunks"]]

    def read(self, source, lo, hi):
        """``source`` (stored or observed) on plan-box nodes ``lo``..``hi``."""
        return source.read((lo + self.shift).tolist(), (hi + self.shift).tolist())

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
        if tile.chunk != FIELD_CHUNK:
            raise ValueError("--join coarse needs tiles solved with --field-chunk 64")
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


JOIN_STEP = 4
JOIN_MINIMUM = 16
JOIN_TILT_WEIGHT = 0.1


def pair_moments(task):
    """Normal equations of one tile pair for the overlap join.

    On the nodes both tiles observed, their fields may differ only by what the two
    corrections explain: ``C_a - C_b = W_a - W_b`` with ``C = constant + tilt · position``.
    Nodes are taken every JOIN_STEP on the global grid; positions are measured from each
    tile's centre in half-tiles, so the four unknowns of a tile have comparable scale.
    """
    plan_root, first, second, grid = task
    a, b = _Tile(plan_root, first, grid), _Tile(plan_root, second, grid)
    gram, rhs, count, square = np.zeros((8, 8)), np.zeros(8), 0, 0.0
    try:
        lo, hi = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi) + 1
        blocks = [
            range(int(lo_) // FIELD_CHUNK * FIELD_CHUNK, int(h), FIELD_CHUNK)
            for lo_, h in zip(lo, hi)
        ]
        for corner in itertools.product(*blocks) if (hi > lo).all() else ():
            first_node = np.maximum(lo, corner)
            first_node += -first_node % JOIN_STEP
            last_node = np.minimum(hi, np.asarray(corner) + FIELD_CHUNK)
            if (last_node <= first_node).any():
                continue
            pick = (slice(None, None, JOIN_STEP),) * 3
            va, seen = a.read(a.observed, first_node - a.lo, last_node - a.lo)
            va, seen = va[pick], seen[pick]
            if not seen.any():
                continue
            vb, other = b.read(b.observed, first_node - b.lo, last_node - b.lo)
            seen = seen & other[pick]
            if not seen.any():
                continue
            nodes = np.stack(
                np.meshgrid(
                    *[np.arange(f, end, JOIN_STEP) for f, end in zip(first_node, last_node)],
                    indexing="ij",
                ),
                -1,
            )[seen].astype(np.float64)
            design = np.empty((len(nodes), 8))
            for column, tile, sign in ((0, a, 1.0), (4, b, -1.0)):
                centre = (tile.lo + tile.hi) / 2
                design[:, column] = sign
                design[:, column + 1 : column + 4] = sign * (
                    (nodes - centre) / np.maximum((tile.hi - tile.lo) / 2, 1.0)
                )
            difference = (va - vb[pick])[seen]
            gram += design.T @ design
            rhs += design.T @ difference
            count += len(difference)
            square += float(difference @ difference)
    finally:
        a.close()
        b.close()
    return first["id"], second["id"], gram, rhs, count, square


def join_tiles(plan_root, tiles, grid, workers, tilt_weight=JOIN_TILT_WEIGHT):
    """One constant and one tilt per tile, from every overlap at once.

    A tile solved on its own follows the sheets inside it, but its level and a gentle
    slope are set by its single gauge node and its boundaries, so neighbours disagree on
    the nodes they share. Pulling each tile to the coarse field removes that disagreement
    and also the tile's sheet-following, because the coarse grid is wider than the sheet
    spacing. Here the tiles are corrected against each other instead: the least-squares
    corrections that make all overlaps agree, the smallest ones among those (the mean
    level and slope the tiles took from their prior stay).
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    index = {t["id"]: k for k, t in enumerate(tiles)}
    jobs = [
        (str(plan_root), a, b, grid)
        for a, b in itertools.combinations(tiles, 2)
        if (np.asarray(a["nodes_lo"]) <= b["nodes_hi"]).all()
        and (np.asarray(b["nodes_lo"]) <= a["nodes_hi"]).all()
    ]
    size = 4 * len(tiles)
    rows, cols, vals = [], [], []
    rhs = np.zeros(size)
    pairs = []
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_spawn()) as pool:
            for a, b, gram, vector, count, square in pool.map(pair_moments, jobs):
                if count < JOIN_MINIMUM:
                    continue
                where = np.concatenate([np.arange(4) + 4 * index[a], np.arange(4) + 4 * index[b]])
                rows.append(np.repeat(where, 8))
                cols.append(np.tile(where, 8))
                vals.append(gram.ravel())
                rhs[where] += vector
                pairs.append((a, b, where, gram, vector, count, square))
    solution = np.zeros(size)
    if pairs:
        matrix = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (size, size)
        ).tocsc()
        # The same constant and slope added to every tile changes no overlap, and inside a
        # narrow overlap a tilt across it is almost a constant. Tilts are therefore charged
        # (tilt_weight per shared node and half-tile unit), so a step between tiles becomes
        # constants, not ramps; the tiny ridge on constants leaves the mean level alone.
        shared = matrix.diagonal()[0::4]
        penalty = np.repeat(1e-9 * shared.max(), size)
        for axis in (1, 2, 3):
            penalty[axis::4] += tilt_weight * shared
        solution = np.asarray(spsolve(matrix + diags(penalty, format="csc"), rhs))
    before, after = [], []
    for a, b, where, gram, vector, count, square in pairs:
        p = solution[where]
        before.append(math.sqrt(square / count))
        after.append(math.sqrt(max(0.0, square - 2 * p @ vector + p @ gram @ p) / count))
    corrections = {}
    for tile in tiles:
        constant, *tilt = solution[4 * index[tile["id"]] :][:4].tolist()
        extent = np.asarray(tile["nodes_hi"]) - tile["nodes_lo"]
        corrections[tile["id"]] = dict(
            affine=dict(
                constant=constant,
                tilt=tilt,
                centre=(extent / 2).tolist(),
                scale=np.maximum(extent / 2, 1.0).tolist(),
            )
        )
    summary = dict(
        method="overlap",
        step=JOIN_STEP,
        pairs=len(pairs),
        pairs_without_shared_nodes=len(jobs) - len(pairs),
        rms_before=dict(median=float(np.median(before)), worst=float(np.max(before)))
        if pairs
        else None,
        rms_after=dict(median=float(np.median(after)), worst=float(np.max(after)))
        if pairs
        else None,
    )
    return corrections, summary


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
    """A tile's correction on local nodes: its join constant and tilt, or the trilinear
    value of its reduced low-pass against the coarse field."""
    record = _WORKER["corrections"].get(tile_id)
    if record is None:
        return 0.0
    if "affine" in record:
        affine = record["affine"]
        value = np.full([b - a for a, b in zip(local_lo, local_hi)], affine["constant"])
        for axis in range(3):
            along = (np.arange(local_lo[axis], local_hi[axis]) - affine["centre"][axis]) / affine[
                "scale"
            ][axis]
            value += (affine["tilt"][axis] * along).reshape(
                [-1 if i == axis else 1 for i in range(3)]
            )
        return value
    if tile_id not in _WORKER["low"]:
        if file_sha(record["path"]) != record["sha256"]:
            raise ValueError("tile correction changed: " + tile_id)
        with np.load(record["path"], allow_pickle=False) as data:
            _WORKER["low"][tile_id] = (data["low"], int(data["step"]))
    from scipy.ndimage import map_coordinates

    low, step = _WORKER["low"][tile_id]
    # The low-pass lives on the tile's stored field, which may start before its plan box.
    shift = _worker_tile(tile_id).shift
    axes = [
        (np.arange(a, b) + s - (step - 1) / 2) / step for a, b, s in zip(local_lo, local_hi, shift)
    ]
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
        values, stored = tile.read(tile.stored, local_lo, local_hi)
        _, seen = tile.read(tile.observed, local_lo, local_hi)
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
                (channel,),
                map_axes(meta, tile.geometry, own_lo - tile.solved_lo, own_hi - tile.solved_lo),
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
    edge = SEAM_BINS[min(index + 1, len(SEAM_BINS) - 1)]
    # The last bin is open-ended (>= 4 windings): report its finite lower edge, so receipts
    # stay valid JSON; over_four_windings counts those nodes.
    return float(edge if np.isfinite(edge) else SEAM_BINS[-2])


def _merged_frame(geometries, sources):
    """The merged field's frame, and a receipt when the tiles carry more than one.

    A frame digests the CT identity including the reader that read it, and every host
    builds its own native reader, so tiles solved on two hosts carry two frames for one CT.
    They merge when their CT identities agree without the reader, the rule of
    ``--field-prior``; the merged field takes the frame most tiles carry (on a tie, the one
    met first in plan order).
    """
    from .field_prior import ct_differences, frame_match

    tiles_by_frame = {}
    for tile_id, geometry in geometries.items():
        tiles_by_frame.setdefault(geometry["frame"], []).append(tile_id)
    frame = max(tiles_by_frame, key=lambda f: len(tiles_by_frame[f]))
    if len(tiles_by_frame) == 1:
        return frame, None
    for tile_id, source in sources.items():
        if not isinstance(source, dict):
            raise ValueError(f"tiles differ in frame and tile {tile_id} records no CT identity")
    reference = tiles_by_frame[frame][0]
    for tile_id, source in sources.items():
        keys = ct_differences(sources[reference], source)
        if keys:
            raise ValueError(
                f"tiles differ in frame: the CT identity of tile {tile_id} differs from tile "
                f"{reference} in " + ", ".join(keys)
            )
    return frame, frame_match(frames=tiles_by_frame)


def merge(args, fields=None):
    """Blend the plan's solved tiles. ``fields`` maps a tile id to the directory of a field
    that replaces the one in the plan's tile (``scroll refine``)."""
    root = Path(args.plan).resolve()
    fields = {tid: Path(path) for tid, path in (fields or {}).items()}
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
    tiles = [
        dict(t, field_run=str(fields[t["id"]])) if t["id"] in fields else t
        for t in tiles
        if t["id"] not in missing
    ]
    geometries = {}
    sources = {}
    grown = {}
    tasks = {}
    for tile in tiles:
        probe = _Tile(root, tile, grid)
        try:
            geometries[tile["id"]] = probe.geometry
            sources[tile["id"]] = probe.prepared.get("source")
            if probe.grown:
                grown[tile["id"]] = dict(
                    nodes_lo=probe.solved_lo.tolist(), nodes_hi=probe.solved_hi.tolist()
                )
            for key in probe.chunks:
                node_lo = probe.solved_lo + np.asarray(key) * probe.chunk
                node_hi = np.minimum(node_lo + probe.chunk, probe.hi + 1) - 1
                node_lo = np.maximum(node_lo, probe.lo)
                if (node_hi < node_lo).any():
                    continue
                for merged in itertools.product(
                    *(
                        range(a // FIELD_CHUNK, b // FIELD_CHUNK + 1)
                        for a, b in zip(node_lo, node_hi)
                    )
                ):
                    tasks.setdefault(merged, set()).add(tile["id"])
        finally:
            probe.close()
    steps = {g["sheet_step"] for g in geometries.values()}
    units = {g["voxel_um"] for g in geometries.values()}
    if len(steps) != 1 or len(units) != 1:
        raise ValueError("tiles differ in sheet step or units")
    frame, frame_match = _merged_frame(geometries, sources)
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
    join = None
    method = getattr(args, "join", "overlap")
    if method not in JOINS:
        raise ValueError("--join must be one of " + ", ".join(JOINS))
    if method == "overlap":
        corrections, join = join_tiles(root, tiles, grid, args.workers)
    elif method == "coarse" and args.coarse_sigma_mm > 0:
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
            tid: _json(fields.get(tid, root / "tiles" / tid) / "FIELD.json")["checkpoint_sha256"]
            for tid in geometries
        },
        blend="linear cross-fade over 2*overlap+1 nodes on interior tile faces",
        join=dict(method="overlap", step=JOIN_STEP, tiles=corrections) if join else None,
        coarse_correction=dict(
            sigma_mm=args.coarse_sigma_mm,
            reduced_step=CORRECTION_STEP,
            coarse_checkpoint_sha256=document["coarse"]["checkpoint_sha256"],
            tiles={tid: c["sha256"] for tid, c in corrections.items()},
        )
        if corrections and not join
        else None,
    )
    manifest = dict(
        identity=identity,
        shape=grid["shape_zyx"],
        chunk=FIELD_CHUNK,
        chunks=chunks,
        progress=dict(merged_chunks=len(chunks), physical_acceptance=False),
        initialization="field-prior tiles joined on their overlaps and blended; not a global solve"
        if join
        else "weighted blend of field-prior tiles; not a global solve",
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
            over_four_windings=int(hist[-1]),
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
        join=join,
        coarse_correction=dict(
            sigma_mm=args.coarse_sigma_mm,
            p95_abs_correction={tid: c["p95_abs_correction"] for tid, c in corrections.items()},
        )
        if corrections and not join
        else None,
        global_converged=False,
        physical_acceptance=False,
        numerical_status="independently solved field-prior tiles, joined on their overlaps "
        "and blended; seams in SEAMS.json"
        if join
        else "blend of independently solved field-prior tiles; seams in SEAMS.json",
        seconds=time.time() - started,
    )
    if frame_match:
        field_meta["frame_match"] = frame_match
    if grown:
        # Solved on a larger box than planned; only the plan box was joined and blended.
        field_meta["grown_tiles"] = grown
    if fields:
        # Fields solved apart from their prepared tile, relative to the plan.
        field_meta["tile_fields"] = {
            tid: os.path.relpath(path, root)
            for tid, path in sorted(fields.items())
            if tid in geometries
        }
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


class DecimationRefused(ValueError):
    """The edge collapse returned no mesh (e.g. non-manifold edges); render the source mesh."""


def _collapsible(vertices, faces):
    """Drop degenerate faces and give every fan of a vertex its own copy of that vertex.

    libigl's edge collapse closes the boundary with a vertex at infinity and refuses a mesh
    that is then not edge-manifold. One vertex where two sheets or two boundary loops touch
    is enough, and whole-scroll isosurfaces have a few (68 of 821,728 on a PHerc0175A
    level): the collapse then returns an empty mesh. Geometry is unchanged: copies share the
    position, and only faces of (near-)zero area are dropped.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    first, second = (
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    area = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
    keep = area > 1e-6 * np.median(area) if len(area) else np.zeros(0, bool)
    faces = faces[keep]
    count = len(faces)
    # Corners (face, slot) of faces that share an edge belong to one fan at both endpoints.
    corner = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    corner_a, corner_b = corner.ravel(), corner[:, [1, 2, 0]].ravel()
    low = np.where(a < b, corner_a, corner_b)
    high = np.where(a < b, corner_b, corner_a)
    key = np.minimum(a, b) * np.int64(len(vertices)) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    i, j = order[:-1][same], order[1:][same]
    # libigl does not return on an edge with three or more faces, and needs one orientation.
    if (len(same) > 1 and (same[1:] & same[:-1]).any()) or ((a[i] < b[i]) == (a[j] < b[j])).any():
        raise DecimationRefused("mesh has non-manifold or inconsistently oriented edges")
    rows, cols = np.concatenate([low[i], high[i]]), np.concatenate([low[j], high[j]])
    fans, label = connected_components(
        coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(3 * count, 3 * count)),
        directed=False,
    )
    pair, inverse = np.unique(faces.ravel() * np.int64(max(fans, 1)) + label, return_inverse=True)
    owner = pair // max(fans, 1)
    extra = np.flatnonzero(np.r_[False, owner[1:] == owner[:-1]])
    identifier = owner.copy()
    identifier[extra] = len(vertices) + np.arange(len(extra))
    vertices = np.concatenate([vertices, vertices[owner[extra]]])
    return vertices, identifier[inverse].reshape(-1, 3), int((~keep).sum()), len(extra)


def decimate_bundle(mesh, factor, output, *, sample=200_000, seed=0):
    """Edge-collapse (libigl qslim) a single-level bundle to 1/factor of its faces.

    Rendering samples CT at every pixel of the flattened surface, so triangles only need
    to follow the surface, not the field grid: on PHerc0175A 1/9 (95 um edges) moved the
    surface by at most 1.05 um while cutting flatten+assemble ~10x. The receipt reports
    the measured deviation of a sample of original vertices from the new surface.

    Raises DecimationRefused for a mesh the collapse cannot take (an edge with three or
    more faces, mixed orientation, or an empty result); an interrupted earlier attempt in
    ``output`` is replaced, a completed one for the same request is reused.
    """
    import shutil

    import igl

    from .surface_mesh import MeshReader, import_mesh

    with MeshReader(mesh) as reader:
        meta = reader.manifest["metadata"]
        levels = reader.manifest["levels"]
        vertices, faces = _indexed_mesh(reader)
    if len(levels) != 1:
        raise ValueError("decimation needs a single-level bundle")
    output = Path(output)
    if output.exists():
        done = output / "DECIMATION.json"
        if done.is_file() and (output / "mesh/MESH.json").is_file():
            receipt = _json(done)
            if (receipt.get("factor"), receipt.get("source_faces")) == (factor, len(faces)):
                return output / "mesh", receipt
        shutil.rmtree(output)
    target = max(1, len(faces) // factor)
    clean_v, clean_f, degenerate, copies = _collapsible(vertices, faces)
    out_v, out_f, _, _ = igl.qslim(
        np.asfortranarray(clean_v), np.asfortranarray(clean_f.astype(np.int32)), target
    )
    if not len(out_f):
        raise DecimationRefused("edge collapse returned no faces; the mesh is not collapsible")
    rng = np.random.default_rng(seed)
    probe = vertices[rng.choice(len(vertices), min(sample, len(vertices)), replace=False)]
    distance = np.sqrt(
        igl.point_mesh_squared_distance(probe, out_v, np.asarray(out_f, np.int32))[0]
    )
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
        dropped_degenerate_faces=degenerate,
        split_vertex_fans=copies,
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
        flattening_scope=args.flattening_scope if args.flattening_scope != "charts" else None,
        surface_time_limit_s=args.surface_time_limit,
        coverage_target=args.coverage_target,
        # 2.0: TIFXYZ and per-pixel XYZ are always written.
        export_tifxyz=True,
        render_keep_xyz=True,
        render_tiled=not args.whole_png,
    )
    if args.no_ct_texture:
        # Opt-in key: absent by default so existing render identities are unchanged.
        config["ct_texture"] = False
    if args.keep_layers and (args.no_ct_texture or not args.thickness_um):
        raise ValueError("--keep-layers needs CT texture and --thickness-um above 0")
    if args.thickness_um and not args.no_ct_texture:
        # Without CT texture there is nothing to sample, so the default slab does not apply.
        config.update(render_thickness_um=float(args.thickness_um), render_composite=args.composite)
        if args.keep_layers:
            config["render_keep_layers"] = True
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
        try:
            mesh, receipts["decimation"] = decimate_bundle(
                mesh, args.decimate, output.parent / (output.name + ".decimated")
            )
        except DecimationRefused as error:
            # Render the level from its full mesh rather than lose it.
            receipts["decimation"] = dict(
                factor=1, requested_factor=args.decimate, source_faces=source_faces,
                source_level=source_levels[0]["level"], reason=str(error),
            )  # fmt: skip
        receipts["decimation"]["seconds"] = time.time() - started
    with MeshReader(mesh) as reader:
        volume = open_volume({k: v for k, v in config.items() if v is not None})
        if (
            volume.remote
            and not args.offline
            and args.prefetch_workers > 0
            and not args.no_ct_texture
        ):
            # Prefetch every touched chunk over parallel connections, then render offline:
            # an online render re-verifies each chunk against the source per epoch, one
            # HTTPS connection at a time (53 of 77 s on a PHerc0175A winding).
            t = time.time()
            # The slab reaches half its thickness to each side of the mesh.
            reach = math.ceil(args.thickness_um / 2 / min(volume.spacing_xyz))
            keys = mesh_ct_chunks(reader, volume, margin=1 + reach)
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

    p = sub.add_parser(
        "counted-density",
        help="count sheets on the Lasagna cos channel (October recipe) and write a manifest",
    )
    p.add_argument("--lasagna-input", required=True, help="Lasagna manifest with a cos group")
    p.add_argument("--output", required=True, help="directory (rerun to continue)")
    p.add_argument(
        "--bounds",
        type=int,
        nargs=6,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="level-0 (base grid) voxel bounds; default the whole cos array",
    )
    p.add_argument("--tile", type=int, default=192, help="cos voxels per tile edge (x64)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--part",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("K", "N"),
        help="compute every N-th tile starting at K (one process per GPU)",
    )
    p.add_argument(
        "--min-fill",
        type=float,
        default=0.01,
        help="skip tiles whose cos is nonzero on less than this fraction",
    )

    p = sub.add_parser("plan", help="choose overlapping tile ROIs on one global field grid")
    p.add_argument("--source", required=True)
    p.add_argument("--voxel-um", type=float, required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--stride", type=int, help="observation stride (default: field/CT ratio)")
    p.add_argument("--field-spacing-um", type=float, required=True)
    p.add_argument("--bounds", type=int, nargs=6, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
    p.add_argument(
        "--tile-nodes",
        default="auto",
        help="largest core nodes per axis (x64; an axis's cores share its 64-node chunks "
        "evenly), or auto (default): the largest tile that one GPU "
        "solves as one block (--gpu-memory-gib, else the smallest visible GPU); auto tiles "
        "run as one block with one sweep",
    )
    p.add_argument(
        "--gpu-memory-gib",
        type=float,
        help="with --tile-nodes auto: memory of each GPU that will run the tiles (default: "
        "detect on this machine)",
    )
    p.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="with --tile-nodes auto: tiles solved at once on one GPU (memory is divided)",
    )
    p.add_argument(
        "--tile-preconditioner",
        choices=TILE_PRECONDITIONERS,
        default="jacobi",
        help="jacobi (default); auto: with --tile-nodes auto, AMG on the largest tile that "
        f"fits the GPU down to {AMG_MIN_TILE_NODES} nodes per axis, else Jacobi; amg forces "
        "AMG. AMG needs ~3.4x the GPU and host memory of Jacobi and was slower with strong "
        "fibres (docs/SCROLL.md)",
    )
    p.add_argument(
        "--tile-kernel",
        choices=TILE_KERNELS,
        default="eager",
        help="eager (default); fused: solve tiles with the fused Triton GPU kernels "
        "(run --field-kernel fused; CUDA only, same system and stopping rules)",
    )
    p.add_argument(
        "--overlap-nodes",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"extra nodes per side (x{OVERLAP_STEP}, default {DEFAULT_OVERLAP})",
    )
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
    p.add_argument(
        "--jobs",
        type=int,
        help="tiles at once (default: GPUs x the plan's jobs per GPU, or 1 without GPUs)",
    )
    p.add_argument(
        "--gpus",
        help="comma-separated GPU ids for the tiles, one per tile process (default: all "
        "visible GPUs; empty string: none)",
    )
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
        "--join",
        choices=JOINS,
        default="overlap",
        help="how tiles are made to agree before blending: overlap (default) gives each "
        "tile one constant and one tilt from all the nodes tiles share and keeps their "
        "sheet-following; coarse pulls each tile to the coarse field above "
        "--coarse-sigma-mm (crosses sheets where the coarse grid is wider than the sheet "
        "spacing); none blends the raw tiles",
    )
    p.add_argument(
        "--coarse-sigma-mm",
        type=float,
        default=0.5,
        help="with --join coarse: replace each tile's (tile - coarse) content above this "
        "Gaussian scale by the coarse field before blending (0 disables)",
    )

    p = sub.add_parser(
        "refine",
        help="close the seams of the merged tiles: solve the tiles that disagree again with "
        "their internal faces held at the merge, until the merge stops changing",
    )
    p.add_argument("plan")
    p.add_argument("--output", required=True, help="the refined merged field (a new directory)")
    p.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="most passes of tile solves after the first merge (default: 3)",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=2,
        help="node layers held on each internal tile face (run --field-boundary-layers; "
        "default: 2)",
    )
    p.add_argument(
        "--trigger-share",
        type=float,
        default=0.02,
        help="solve again the tiles of a pair with more than this share of its shared nodes "
        "over half a winding apart, and the tiles next to them (default: 0.02)",
    )
    p.add_argument(
        "--trigger-p95",
        type=float,
        default=0.5,
        help="or whose 95th percentile of |dW| on shared nodes is over this (default: 0.5)",
    )
    p.add_argument(
        "--bound",
        type=float,
        default=0.25,
        help="accept when the bound of the remaining error, change x r / (1 - r) with r the "
        "ratio of the last two changes between merges (95th percentile, windings), is at "
        "most this (default: 0.25)",
    )
    p.add_argument(
        "--max-kink-ratio",
        type=float,
        default=2.0,
        help="accept when no solved tile bends more at its held faces than this many times "
        "its own mid-planes (95th percentile of the second difference of W; default: 2)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        help="tiles at once (default: GPUs x the plan's jobs per GPU, or 1 without GPUs)",
    )
    p.add_argument(
        "--gpus",
        help="comma-separated GPU ids for the tiles, one per tile process (default: all "
        "visible GPUs; empty string: none)",
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument(
        "--prune",
        action="store_true",
        help="delete each pass's merge once the next two passes no longer need it",
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
    p.add_argument(
        "--thickness-um",
        type=float,
        default=70.0,
        help="render a slab this thick around the surface, one layer per render pitch along "
        "the normal (default 70: 9 layers at 8.64 um, about half the sheet spacing); "
        "0 renders the single surface layer",
    )
    p.add_argument(
        "--composite",
        default="max",
        choices=("max", "mean"),
        help="how the layers of the slab become the image (default: max)",
    )
    p.add_argument(
        "--keep-layers",
        action="store_true",
        help="also store every layer per tile (ct_layers); large",
    )
    p.add_argument("--chart-faces", type=int, default=1024)
    p.add_argument("--flattening-method", default="slim", choices=("slim", "default"))
    p.add_argument(
        "--flattening-scope",
        default="charts",
        choices=("charts", "sheet"),
        help="charts: weld ~1,000-face charts (default); sheet: one global LSCM+SLIM map per "
        "connected piece, no welded seams (needs --flattening-method slim; raise "
        "--surface-time-limit for multi-million-face meshes)",
    )
    p.add_argument("--surface-time-limit", type=float, default=1800.0)
    p.add_argument("--coverage-target", type=float, default=1.0)
    p.add_argument(
        "--no-ct-texture",
        action="store_true",
        help="flatten and write TIFXYZ without sampling CT for the texture (skips the CT prefetch)",
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
    from .cli_settings import parse_options

    args = parse_options(
        _parser(),
        argv,
        paths=(
            "source",
            "coarse",
            "fibres",
            "lasagna_input",
            "output",
            "plan",
            "mesh",
            "local",
            "cache",
            "renders",
        ),
    )
    try:
        if args.action == "counted-density":
            from .counted_density import build

            result = build(args)
        elif args.action == "plan":
            result = plan(args)
        elif args.action == "tiles":
            result = run_tiles(args)
        elif args.action == "merge":
            result = merge(args)
        elif args.action == "refine":
            from .scroll_refine import refine

            result = refine(args)
        elif args.action == "site":
            result = site(args)
        else:
            result = render_ct(args)
    except (ValueError, OSError) as error:
        print("scroll: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, default=float))
    return 0 if result.get("status") not in ("INCOMPLETE", "REFINE_NOT_CONVERGED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
