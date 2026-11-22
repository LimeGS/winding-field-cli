"""Read-only, calibrated AFV import into bounded world-coordinate curve blocks."""

from contextlib import closing, suppress
from functools import lru_cache
import hashlib
import heapq
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from .artifacts import Tiles
from .io import http_url
from .splines import bind
from .util import atomic_json, digest, file_sha

_BATCH_BLOCKS = 1000
_BATCH_SECONDS = 2.0
_PROGRESS_SECONDS = 2.0


def _open(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("AFV not found: " + str(path))
    for suffix in ("-wal", "-journal"):
        journal = Path(str(path) + suffix)
        if journal.exists() and journal.stat().st_size:
            raise ValueError("AFV has an uncheckpointed journal; close/checkpoint its writer first")
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


def _header(db):
    if db.execute("PRAGMA application_id").fetchone() != (1447249475,) or db.execute(
        "PRAGMA user_version"
    ).fetchone() != (1,):
        raise ValueError("AFV schema v1 required")
    for table, columns in [
        ("metadata", {"key", "value"}),
        ("fibers", {"id", "point_count", "annotation"}),
        ("blocks", {"id", "fiber_id", "first_segment", "points"}),
    ]:
        if not columns <= {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
            raise ValueError("invalid AFV schema: " + table)
    if (
        db.execute("SELECT count(*) FROM metadata").fetchone()[0] > 64
        or db.execute("SELECT 1 FROM metadata WHERE length(value)>1048576 LIMIT 1").fetchone()
    ):
        raise ValueError("AFV metadata exceeds bounded header size")
    header = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}
    if header.get("complete") is not True:
        raise ValueError("incomplete AFV")
    root = header.get("root", {})
    if root.get("coordinate_order") != "XYZ":
        raise ValueError("AFV requires XYZ coordinate order")
    counts = db.execute("SELECT count(*),coalesce(sum(point_count),0) FROM fibers").fetchone()
    blocks = db.execute("SELECT count(*) FROM blocks").fetchone()[0]
    if (
        counts != (header.get("fiber_count"), header.get("point_count"))
        or blocks != header.get("block_count")
        or db.execute(
            "SELECT 1 FROM blocks b LEFT JOIN fibers f ON f.id=b.fiber_id WHERE f.id IS NULL LIMIT 1"
        ).fetchone()
    ):
        raise ValueError("AFV block/fibre inventory mismatch")
    if (
        db.execute(
            "SELECT 1 FROM blocks WHERE length(points)<48 OR length(points)>6168 OR length(points)%24!=0 LIMIT 1"
        ).fetchone()
        or db.execute("SELECT 1 FROM fibers WHERE length(annotation)>1048576 LIMIT 1").fetchone()
    ):
        raise ValueError("invalid AFV block or annotation size")
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("corrupt AFV")
    return header


def source_identity(path, geometry_path, source):
    path = Path(path).expanduser().resolve()
    sha = file_sha(path)
    try:
        with closing(_open(path)) as db:
            header = _header(db)
    except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid AFV metadata/schema") from error
    frame = header.get("frame", {})
    embedded = None
    if not isinstance(frame, dict):
        raise ValueError("invalid AFV frame")
    if "source" in frame:
        original = frame["source"]
        if not isinstance(original, dict):
            raise ValueError("invalid AFV source frame")
        root = header["root"]
        if (
            frame.get("coordinate_order") != "XYZ"
            or frame.get("units") != "selected-source voxels"
            or root.get("coordinate_units") != "voxel"
            or frame.get("level") != original.get("level")
            or frame.get("origin_xyz_um") != original.get("origin_xyz")
            or frame.get("spacing_xyz_um") != original.get("spacing_xyz")
            or root.get("coordinate_base_shape_zyx") != original.get("shape")
            or root.get("vc_open_data_source_coordinate_level") != original.get("level")
            or ("source" in root and http_url(root["source"]) != http_url(original.get("source")))
        ):
            raise ValueError("conflicting AFV embedded coordinate frame")
        if original.get("level") == source.get("level"):
            for name in ("shape", "spacing_xyz", "origin_xyz"):
                if original.get(name) != source.get(name):
                    raise ValueError("AFV selected-source frame mismatch")
        embedded = dict(
            source=original["source"],
            origin_xyz_um=frame["origin_xyz_um"],
            spacing_xyz_um=frame["spacing_xyz_um"],
        )
    explicit = None
    descriptor_sha = None
    if geometry_path:
        from .input_settings import declaration, declaration_location

        explicit, descriptor_sha, _ = declaration(geometry_path, limit=65536)
        geometry_path = declaration_location(geometry_path)
        if not isinstance(explicit, dict) or set(explicit) != {
            "source",
            "origin_xyz_um",
            "spacing_xyz_um",
        }:
            raise ValueError("AFV geometry requires source, origin_xyz_um and spacing_xyz_um")
    geometry = embedded or explicit
    if geometry is None:
        raise ValueError("ambiguous AFV geometry; supply --fibres-geometry")
    if embedded and explicit and embedded != explicit:
        raise ValueError("explicit AFV geometry conflicts with embedded frame")
    if http_url(geometry["source"]) != source["source"]:
        raise ValueError("AFV source mismatch")
    for name in ("origin_xyz_um", "spacing_xyz_um"):
        value = np.asarray(geometry[name], float)
        if (
            value.shape != (3,)
            or not np.isfinite(value).all()
            or (name == "spacing_xyz_um" and (value <= 0).any())
        ):
            raise ValueError("invalid AFV geometry " + name)
    if file_sha(path) != sha:
        raise ValueError("AFV source changed during validation")
    return dict(
        path=str(path),
        sha256=sha,
        geometry_path=geometry_path if geometry_path else None,
        geometry_sha256=descriptor_sha,
        **geometry,
        input_count=header["fiber_count"],
        block_count=header["block_count"],
    )


def _pieces(points, maps, stop=None):
    """Fragments of one block: native when the support carries the kernel, else NumPy."""
    native = getattr(maps, "native", None)
    if native is not None:
        from .afv_clip import MAX_INTERVALS
        from .afv_native import clip_blocks

        fragments = clip_blocks(native, [points], maps.dense, MAX_INTERVALS)[0]
        counts = maps.native_counts
        if fragments is not None:
            counts["native_blocks"] += 1
            yield from fragments
            return
        counts["reference_blocks"] += 1
    from .afv_clip import pieces

    yield from pieces(points, maps, stop)


def _scalar_pieces(points, maps, stop=None):
    """Split at exact voxel-cell boundaries: a sparse edge cannot jump a narrow gap."""

    @lru_cache(maxsize=8)
    def tile(key):
        return maps.load(key, ("support",))["support"]

    def supported(xyz):
        q = xyz[::-1]
        low = np.floor(q).astype(int)
        fraction = q - low
        for corner in itertools.product((0, 1), repeat=3):
            c = np.asarray(corner)
            if not np.all(np.where(c, fraction, 1 - fraction) > 0):
                continue
            ij = np.minimum(low + c, np.asarray(maps.shape) - 1)
            if not tile(tuple(ij // maps.chunk))[tuple(ij % maps.chunk)]:
                return False
        return True

    high = np.asarray(maps.shape)[::-1] - 1
    piece = []
    for a, b in zip(points[:-1], points[1:]):
        delta = b - a
        near = np.divide(-a, delta, out=np.full(3, -np.inf), where=delta != 0)
        far = np.divide(high - a, delta, out=np.full(3, np.inf), where=delta != 0)
        enter = max(0.0, np.minimum(near, far).max())
        leave = min(1.0, np.maximum(near, far).min())
        if leave <= enter or np.any((delta == 0) & ((a < 0) | (a > high))) or not np.any(delta):
            if len(piece) > 1:
                yield np.asarray(piece)
            piece = []
            continue
        start = np.clip(a + enter * delta, 0, high)
        end = np.clip(a + leave * delta, 0, high)
        d = end - start
        axes = []
        for i in range(3):
            if not d[i]:
                continue
            values = range(math.floor(min(start[i], end[i])) + 1, math.ceil(max(start[i], end[i])))
            if d[i] < 0:
                values = reversed(values)
            # Bind each axis before the lazy merge advances the generators.
            axes.append(map(lambda v, o=start[i], s=d[i]: (v - o) / s, values))
        previous = 0.0
        continuing = False
        for t in itertools.chain(heapq.merge(*axes), [1.0]):
            if stop and stop():
                raise InterruptedError("AFV import paused")
            if t <= previous:
                continue
            left = np.clip(start + previous * d, 0, high)
            right = np.clip(start + t * d, 0, high)
            if supported((left + right) / 2):
                if piece and not np.allclose(piece[-1], left, rtol=0, atol=1e-9):
                    if len(piece) > 1:
                        yield np.asarray(piece)
                    piece = []
                    continuing = False
                if not piece:
                    piece = [left, right]
                elif continuing:
                    piece[-1] = right
                else:
                    piece.append(right)
                continuing = True
            else:
                if len(piece) > 1:
                    yield np.asarray(piece)
                piece = []
                continuing = False
            previous = t
    if len(piece) > 1:
        yield np.asarray(piece)


def _checked_source(config, source, verified_source):
    """Only reuse a freshly checked result supplied by this invocation's caller."""
    from .input_settings import check_recipe_sources

    check_recipe_sources(config)
    if verified_source is None:
        return source_identity(config["fibres"], config.get("fibres_geometry"), source)
    from .input_settings import declaration_location

    geometry = declaration_location(config.get("fibres_geometry"))
    if (
        not isinstance(verified_source, dict)
        or verified_source.get("path") != str(Path(config["fibres"]).expanduser().resolve())
        or verified_source.get("geometry_path") != geometry
        or http_url(verified_source.get("source", "")) != source["source"]
    ):
        raise ValueError("verified AFV input context changed")
    return verified_source


def validate_retained(run, config, *, verified_source=None):
    run = Path(run)
    root = run / "fibres"
    binding = root / "IMPORT.json"
    if not binding.exists():
        return
    meta = json.loads((run / "PREPARED.json").read_bytes())
    identity = json.loads(binding.read_bytes())
    if identity["input"] != _checked_source(config, meta["source"], verified_source):
        raise ValueError("external AFV input identity changed")
    if identity["maps"] != meta["tiles_sha256"]:
        raise ValueError("AFV import maps changed")
    receipt = run / "FIBRES.json"
    if receipt.exists():
        saved = json.loads(receipt.read_bytes())
        if saved["identity"] != identity or file_sha(root / "fibres.sqlite") != saved["sha256"]:
            raise ValueError("imported AFV payload or inventory changed")


def _source_blocks(src, source, meta, stop):
    family_column = (
        "family"
        if "family" in {r[1] for r in src.execute("PRAGMA table_info(fibers)")}
        else "'unknown'"
    )
    total = 0
    for fid, family, count, annotation in src.execute(
        f"SELECT id,{family_column},point_count,annotation FROM fibers ORDER BY id"
    ):
        if (
            type(count) is not int
            or count < 2
            or not isinstance(annotation, str)
            or len(annotation) > 1048576
        ):
            raise ValueError("invalid AFV fibre metadata")
        note = json.loads(annotation)
        family = family or "unknown"
        cursor = 0
        previous = None
        classification = note.get("hv_classification") if isinstance(note, dict) else None
        tag = classification.get("manual_tag") if isinstance(classification, dict) else None
        if family == "unknown" and tag in ("V", "H"):
            family = tag
        for first, raw in src.execute(
            "SELECT first_segment,points FROM blocks WHERE fiber_id=? ORDER BY first_segment",
            (fid,),
        ):
            if stop and stop():
                raise InterruptedError("AFV import paused")
            if first != cursor or len(raw) % 24 or not 2 <= len(raw) // 24 <= 257:
                raise ValueError("invalid or discontinuous AFV block")
            points = np.frombuffer(raw, dtype="<f8").reshape(-1, 3)
            if not np.isfinite(points).all() or (
                previous is not None and not np.array_equal(previous, points[0])
            ):
                raise ValueError("nonfinite or discontinuous AFV block points")
            previous = points[-1]
            cursor += len(points) - 1
            total += 1
            world = points * source["spacing_xyz_um"] + source["origin_xyz_um"]
            if not np.isfinite(world).all():
                raise ValueError("nonfinite AFV world coordinates")
            yield fid, family, first, annotation, (world - meta["origin_xyz"]) / meta["spacing_xyz"]
        if cursor + 1 != count:
            raise ValueError("AFV block point inventory mismatch")
    if total != source["block_count"]:
        raise ValueError("AFV block inventory changed")


# Work units: consecutive source blocks, bounded by count, points and annotation bytes.
_JOB_BLOCKS = 512
_JOB_POINTS = 1 << 18
_JOB_ANNOTATION_BYTES = 4 << 20
# Native calls per worker step, so STOP is observed between bounded slices.
_NATIVE_SLICE_BLOCKS = 64
_SNAPSHOT_BYTES = 4 << 30
_RESUME = dict(
    schema="afv-import-resume-v2",
    clip="afv-clip-pieces-v1",
    serializer="npy-v1-float64-xyz",
    checksum="sha256(digest([family,annotation])) chained with sha256(blob) in curve order",
)
_HEADERS = {}

_WORKER_SUPPORT = None
_WORKER_LIBRARY = None


def _clip_initialize(path, cache_bytes, library=None, snapshot=None):
    """Worker state: the shared dense snapshot when given, else a bounded tile cache."""
    from .afv_clip import SupportTiles
    from .afv_support import DenseSupport

    global _WORKER_SUPPORT, _WORKER_LIBRARY
    if snapshot is not None:
        _WORKER_SUPPORT = DenseSupport(*snapshot)
    else:
        _WORKER_SUPPORT = SupportTiles(Tiles(path, readonly=True), cache_bytes=cache_bytes)
    _WORKER_LIBRARY = None
    if library is not None:
        from .afv_native import load

        _WORKER_LIBRARY = load(library)


def _clip(blocks, support, library, stop, counters):
    """Fragments for each block (None entries are skipped: retained blocks)."""
    from .afv_clip import MAX_INTERVALS

    results = [None] * len(blocks)
    pending = [i for i, points in enumerate(blocks) if points is not None]
    if library is not None and getattr(support, "dense", None) is not None:
        from .afv_native import clip_blocks

        for start in range(0, len(pending), _NATIVE_SLICE_BLOCKS):
            if stop and stop():
                raise InterruptedError("AFV import paused")
            chosen = pending[start : start + _NATIVE_SLICE_BLOCKS]
            native = clip_blocks(library, [blocks[i] for i in chosen], support.dense, MAX_INTERVALS)
            for index, fragments in zip(chosen, native):
                if fragments is not None:
                    results[index] = fragments
                    counters["native_blocks"] += 1
        pending = [i for i in pending if results[i] is None]
        counters["reference_blocks"] += len(pending)
    for index in pending:
        if stop and stop():
            raise InterruptedError("AFV import paused")
        results[index] = list(_pieces(blocks[index], support, stop))
    return results


def _clip_worker(blocks):
    """Clip one work unit; return results only, never echo the input back."""
    from .import_parallel import RESULT_BYTES, worker_stopped

    counters = dict(native_blocks=0, reference_blocks=0)
    before = _WORKER_SUPPORT.loads
    results = _clip(blocks, _WORKER_SUPPORT, _WORKER_LIBRARY, worker_stopped, counters)
    charge = 4096
    for index, fragments in enumerate(results):
        if fragments is None:
            continue
        charge += 256 + sum(piece.nbytes + 256 for piece in fragments)
        if charge > RESULT_BYTES:
            # The parent recomputes the rest; results are never truncated.
            for later in range(index, len(results)):
                results[later] = None
            break
    stats = dict(pid=os.getpid(), support_loads=_WORKER_SUPPORT.loads - before, **counters)
    return results, stats


def _blob(world):
    """Exactly ``np.save(world)`` for a native float64 (n, 3) array, header cached."""
    if world.dtype != np.float64 or world.ndim != 2 or not world.flags.c_contiguous:
        buffer = io.BytesIO()
        np.save(buffer, world, allow_pickle=False)
        return buffer.getvalue()
    n = world.shape[0]
    header = _HEADERS.get((n, world.shape[1]))
    if header is None:
        buffer = io.BytesIO()
        np.save(buffer, np.zeros(world.shape, np.float64), allow_pickle=False)
        raw = buffer.getvalue()
        header = raw[: len(raw) - world.nbytes]
        if len(_HEADERS) >= 4096:
            _HEADERS.clear()
        _HEADERS[(n, world.shape[1])] = header
    return header + world.tobytes()


def _batches(jobs, retained):
    """Bounded consecutive work units; retained blocks carry no clipping work."""
    batch, points, notes = [], 0, 0
    for job in jobs:
        fid, family, first, annotation, local = job
        old = retained(fid, first)
        batch.append((job, old))
        points += 0 if old else len(local)
        notes += len(annotation)
        if len(batch) >= _JOB_BLOCKS or points >= _JOB_POINTS or notes >= _JOB_ANNOTATION_BYTES:
            yield batch
            batch, points, notes = [], 0, 0
    if batch:
        yield batch


def _resume_mode(db):
    """``fast`` for runs that recorded the v2 resume contract, ``legacy`` otherwise.

    The contract row is written in the transaction of the first committed block.
    Earlier partial runs have committed blocks and no contract: they keep the
    re-clipping verification and their database is not altered to claim otherwise.
    """
    known = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='import_meta'"
    ).fetchone()
    row = (
        db.execute("SELECT value FROM import_meta WHERE key='resume'").fetchone() if known else None
    )
    if row is not None:
        if json.loads(row[0]) != _RESUME:
            raise ValueError("AFV import resume contract differs; use a new output")
        return "fast"
    if (
        db.execute("SELECT 1 FROM done LIMIT 1").fetchone()
        or db.execute("SELECT 1 FROM curves LIMIT 1").fetchone()
    ):
        return "legacy"
    return "new"  # the contract is written with the first committed block


def _record_contract(db):
    db.execute("CREATE TABLE IF NOT EXISTS import_meta(key TEXT PRIMARY KEY,value TEXT)")
    db.execute(
        "INSERT INTO import_meta VALUES ('resume',?)", (json.dumps(_RESUME, sort_keys=True),)
    )


def import_fibres(run, config, stop=None, *, verified_source=None, maps=None):
    started = time.monotonic()
    run = Path(run)
    meta = json.loads((run / "PREPARED.json").read_bytes())
    supplied = maps is not None
    if supplied:
        meta = dict(meta, **meta["native_geometry"])
    source = _checked_source(config, meta["source"], verified_source)
    identity = dict(
        input=source,
        maps=meta["tiles_sha256"],
        origin=meta["origin_xyz"],
        spacing=meta["spacing_xyz"],
    )
    from .afv_native import NativeUnavailable, backend as clip_backend
    from .import_parallel import limits, ordered_map

    chosen = clip_backend(config)
    snapshot_limit = config.get("afv_support_snapshot_bytes", _SNAPSHOT_BYTES)
    if type(snapshot_limit) is not int or snapshot_limit < 0:
        raise ValueError("afv_support_snapshot_bytes must be a nonnegative integer")
    if supplied:
        shape = tuple(maps.shape)
    else:
        tiles = run / "maps/TILES.json"
        shape = tuple(json.loads(tiles.read_bytes())["shape"]) if tiles.is_file() else ()
    snapshot_bytes = (
        math.prod(shape)
        if shape and not supplied and chosen != "reference" and math.prod(shape) <= snapshot_limit
        else 0
    )
    # Clip workers read only the run's local maps and the local AFV's block points, never
    # CT or prediction sources, so a remote CT source does not force a serial import.
    execution = limits(
        config.get("import_workers", 1) if not supplied else 1,
        output=run,
        reserve_bytes=config.get("reserve_bytes", 0) + snapshot_bytes,
        worker_bytes=368 << 20,
        shared_bytes=snapshot_bytes,
    )
    execution.update(
        worker_pids=[],
        worker_support_loads=0,
        streaming_replays=0,
        backend=chosen,
        native_blocks=0,
        reference_blocks=0,
        retained_verified=0,
        retained_reclipped=0,
    )
    root = run / "fibres"
    root.mkdir(exist_ok=True)
    bind(root / "IMPORT.json", identity)
    validate_retained(run, config, verified_source=source)
    if (run / "FIBRES.json").exists():
        if source != _checked_source(config, meta["source"], None):
            raise ValueError("external AFV source changed during import")
        return json.loads((run / "FIBRES.json").read_bytes())
    maps = maps if supplied else Tiles(run / "maps")
    src = _open(config["fibres"])
    db = sqlite3.connect(root / "fibres.sqlite")
    from .afv_clip import SupportTiles

    support = SupportTiles(maps, cache_bytes=64 << 20)
    dense = None
    library = None
    pending = processed = commits = 0
    committed = None
    batch_started = last_commit_utc = last_update = None
    counters = dict(native_blocks=0, reference_blocks=0)

    def progress(status, force=False):
        nonlocal last_update
        if committed is None:
            return
        now = time.monotonic()
        if not force and last_update is not None and now - last_update < _PROGRESS_SECONDS:
            return
        elapsed = max(0.0, now - started)
        execution.update(
            native_blocks=counters["native_blocks"], reference_blocks=counters["reference_blocks"]
        )
        atomic_json(
            root / "STATUS.json",
            dict(
                schema="afv-import-progress-v1",
                status=status,
                identity_sha256=digest(identity),
                blocks_total=source["block_count"],
                blocks_processed=processed,
                blocks_committed=committed,
                blocks_pending=pending,
                commits_this_invocation=commits,
                elapsed_s=elapsed,
                blocks_per_second=processed / elapsed if elapsed else 0.0,
                last_batch_committed_utc=last_commit_utc,
                updated_utc=time.time(),
                execution=execution,
            ),
        )
        last_update = now

    def commit_batch():
        nonlocal pending, batch_started, committed, commits, last_commit_utc
        if pending:
            db.commit()
            committed += pending
            commits += 1
            last_commit_utc = time.time()
            pending = 0
            batch_started = None

    try:
        if config.get("bounded_preparation"):
            from .util import sqlite_quota

            sqlite_quota(db, config.get("bounded_bytes", 12 << 30) // 4)
        if not supplied and maps.identity() != meta["tiles_sha256"]:
            raise ValueError("AFV import maps changed")
        db.executescript(
            "CREATE TABLE IF NOT EXISTS done(fid INTEGER,block INTEGER,sha TEXT,PRIMARY KEY(fid,block));"
            "CREATE TABLE IF NOT EXISTS curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT,source_id INTEGER,source_block INTEGER,annotation TEXT);"
            "CREATE INDEX IF NOT EXISTS curves_source ON curves(source_id,source_block);"
            "CREATE VIRTUAL TABLE IF NOT EXISTS curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);"
        )
        if (
            db.execute(
                "SELECT 1 FROM curves c LEFT JOIN done d ON d.fid=c.source_id AND d.block=c.source_block "
                "LEFT JOIN curve_bounds b ON b.id=c.id WHERE d.fid IS NULL OR b.id IS NULL LIMIT 1"
            ).fetchone()
            or db.execute(
                "SELECT 1 FROM curve_bounds b LEFT JOIN curves c ON c.id=b.id WHERE c.id IS NULL LIMIT 1"
            ).fetchone()
        ):
            raise ValueError("imported AFV inventory changed")
        resume = _resume_mode(db)
        execution["resume"] = resume
        committed = db.execute("SELECT count(*) FROM done").fetchone()[0]
        status_path = root / "STATUS.json"
        if status_path.is_file() and status_path.stat().st_size <= 65536:
            with suppress(OSError, ValueError, TypeError, AttributeError):
                previous_status = json.loads(status_path.read_bytes())
                stamp = previous_status.get("last_batch_committed_utc")
                if (
                    previous_status.get("identity_sha256") == digest(identity)
                    and previous_status.get("blocks_committed") == committed
                    and type(stamp) in (int, float)
                    and math.isfinite(stamp)
                ):
                    last_commit_utc = stamp
        if snapshot_bytes:
            from .afv_support import open_or_build

            dense, execution["support"] = open_or_build(
                maps,
                root,
                meta["tiles_sha256"],
                limit_bytes=snapshot_limit,
                reserve_bytes=config.get("reserve_bytes", 0),
            )
        else:
            execution["support"] = dict(snapshot="bounded-tiles")
        if chosen == "native" and dense is None:
            raise NativeUnavailable("native AFV clipping needs the dense support snapshot")
        if chosen in ("auto", "native") and dense is not None:
            from .afv_native import build, load

            try:
                library = build()
            except NativeUnavailable:
                if chosen == "native":
                    raise
                library = None
            execution["clipper"] = "native" if library is not None else "numpy"
        else:
            execution["clipper"] = "numpy"
        clip_support = dense if dense is not None else support
        progress("IMPORTING", force=True)

        def retained(fid, block):
            return db.execute(
                "SELECT sha FROM done WHERE fid=? AND block=?", (fid, block)
            ).fetchone()

        units = _batches(_source_blocks(src, source, meta, stop), retained)
        fast = resume == "fast"
        contract_pending = resume == "new"
        workers = execution["workers"]
        if workers > 1:
            from collections import deque

            submitted = deque()

            def submissions():
                for unit in units:
                    submitted.append(unit)
                    # Retained blocks of a fast resume need no clipping.
                    yield [None if (old and fast) else job[4] for job, old in unit]

            computed = ordered_map(
                _clip_worker,
                submissions(),
                workers=workers,
                max_inflight=execution["max_inflight"],
                initializer=_clip_initialize,
                initargs=(
                    str(run / "maps"),
                    64 << 20,
                    None if library is None else str(library),
                    None if dense is None else (str(dense.path), dense.meta),
                ),
                stop=stop,
            )

            def paired():
                # Closing this stream must reap the owned workers (ordered_map's finally).
                with closing(computed):
                    for results, stats in computed:
                        yield submitted.popleft(), results, stats

            stream = paired()
        else:
            if library is not None:
                dense.native = load(library)
                dense.native_counts = counters

            def serial():
                # One block at a time: clip, then publish, as the reference importer.
                for unit in units:
                    for job, old in unit:
                        if old and fast:
                            yield [(job, old)], [None], None
                        else:
                            # Lazy: the batch clock starts before the block is clipped.
                            yield [(job, old)], [_pieces(job[4], clip_support, stop)], None

            stream = serial()
        with closing(stream):
            for unit, results, stats in stream:
                if stats:
                    if stats["pid"] not in execution["worker_pids"]:
                        execution["worker_pids"].append(stats["pid"])
                    execution["worker_support_loads"] += stats["support_loads"]
                    counters["native_blocks"] += stats["native_blocks"]
                    counters["reference_blocks"] += stats["reference_blocks"]
                for (job, old), fragments in zip(unit, results):
                    fid, family, first, annotation, local = job
                    if stop and stop():
                        raise InterruptedError("AFV import paused")
                    if old and fast:
                        _verify_retained(db, fid, first, family, annotation, old[0])
                        execution["retained_verified"] += 1
                        processed += 1
                        progress("IMPORTING")
                        continue
                    if fragments is None:
                        execution["streaming_replays"] += int(stats is not None)
                        fragments = _clip([local], clip_support, None, stop, counters)[0]
                    if not old and not db.in_transaction:
                        db.execute("BEGIN")
                        batch_started = time.monotonic()
                    checksum = hashlib.sha256(digest([family, annotation]).encode())
                    rows, boxes, total = [], [], 0
                    for piece in fragments:
                        world = piece * meta["spacing_xyz"] + meta["origin_xyz"]
                        blob = _blob(world)
                        box = tuple(
                            float(v) for pair in zip(world.min(0), world.max(0)) for v in pair
                        )
                        checksum.update(hashlib.sha256(blob).digest())
                        rows.append(blob)
                        boxes.append(box)
                        total += len(blob)
                    if old:
                        _compare_retained(db, fid, first, family, annotation, rows, boxes)
                        if old != (checksum.hexdigest(),):
                            raise ValueError("imported AFV inventory changed")
                        execution["retained_reclipped"] += 1
                        processed += 1
                        progress("IMPORTING")
                        continue
                    if config.get("bounded_preparation") and rows:
                        size = (
                            db.execute("PRAGMA page_count").fetchone()[0]
                            * db.execute("PRAGMA page_size").fetchone()[0]
                        )
                        if 2 * (size + total + 65536) > config.get("bounded_bytes", 12 << 30) // 4:
                            raise OSError("bounded AFV quota exhausted; block not committed")
                    if rows and (
                        shutil.disk_usage(root).free
                        < total + config.get("reserve_bytes", 0) + 65536
                    ):
                        raise OSError("insufficient space for AFV import and reserve")
                    ids = [
                        db.execute(
                            "INSERT INTO curves(tile,points,family,source_id,source_block,annotation) VALUES (?,?,?,?,?,?)",
                            ("imported", blob, family, fid, first, annotation),
                        ).lastrowid
                        for blob in rows
                    ]
                    db.executemany(
                        "INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)",
                        [(cid, *box) for cid, box in zip(ids, boxes)],
                    )
                    if contract_pending:
                        _record_contract(db)
                        contract_pending = False
                    db.execute(
                        "INSERT INTO done VALUES (?,?,?)", (fid, first, checksum.hexdigest())
                    )
                    pending += 1
                    processed += 1
                    if pending and (
                        pending >= _BATCH_BLOCKS
                        or time.monotonic() - batch_started >= _BATCH_SECONDS
                    ):
                        commit_batch()
                    progress("IMPORTING")
        if (
            processed != source["block_count"]
            or db.execute("SELECT count(*) FROM done").fetchone()[0] != processed
        ):
            raise ValueError("imported AFV inventory changed")
        commit_batch()
        if stop and stop():
            raise InterruptedError("AFV import paused")
        progress("VALIDATING", force=True)
        if source != _checked_source(config, meta["source"], None):
            raise ValueError("external AFV source changed during import")
        if not supplied and maps.identity() != meta["tiles_sha256"]:
            raise ValueError("AFV import maps changed during import")
        if dense is not None:
            dense.verify()
        count = db.execute("SELECT count(*) FROM curves").fetchone()[0]
        kept = db.execute("SELECT count(DISTINCT source_id) FROM curves").fetchone()[0]
    except InterruptedError:
        db.rollback()
        pending = 0
        progress("PAUSED", force=True)
        return dict(status="PAUSED")
    except BaseException:
        db.rollback()
        pending = 0
        with suppress(OSError):
            progress("FAILED", force=True)
        raise
    finally:
        support.close()
        if dense is not None:
            dense.close()
        if not supplied:
            maps.close()
        src.close()
        db.close()
    result = dict(
        status="FIBRES_READY",
        method="external-afv",
        spatial_index=True,
        count=count,
        input_count=source["input_count"],
        imported_sources=kept,
        excluded_sources=source["input_count"] - kept,
        exclusion_reason="outside ROI or unsupported surface",
        identity=identity,
        sha256=file_sha(root / "fibres.sqlite"),
        coordinates="XYZ micrometres",
        physical_accuracy="UNVALIDATED",
    )
    atomic_json(run / "FIBRES.json", result)
    from .afv_support import remove

    remove(root)  # the snapshot is a resume aid, not an output
    progress("FIBRES_READY", force=True)
    return result


def _compare_retained(db, fid, first, family, annotation, blobs, boxes):
    """Legacy resume: committed rows must equal the freshly clipped block."""
    rows = db.execute(
        "SELECT id,points,family,annotation FROM curves WHERE source_id=? AND source_block=? ORDER BY id",
        (fid, first),
    ).fetchall()
    if len(rows) != len(blobs):
        raise ValueError("imported AFV inventory changed")
    for (cid, saved, label, note), blob, box in zip(rows, blobs, boxes):
        bound = db.execute(
            "SELECT min_x,max_x,min_y,max_y,min_z,max_z FROM curve_bounds WHERE id=?", (cid,)
        ).fetchone()
        if (
            blob != saved
            or label != family
            or note != annotation
            or bound is None
            or not np.allclose(bound, box, rtol=2e-7, atol=1e-6)
        ):
            raise ValueError("imported AFV payload changed")


def _verify_retained(db, fid, first, family, annotation, expected):
    """Fast resume: committed rows must reproduce the digest recorded with them.

    The ``done`` digest was written in the same transaction as the rows; the rows'
    own payloads, labels, annotations and R-tree bounds are re-checked against it
    and against the source metadata, without re-clipping. Corruption or a missing,
    extra or reordered row fails, as on the legacy path.
    """
    checksum = hashlib.sha256(digest([family, annotation]).encode())
    rows = db.execute(
        "SELECT id,points,family,annotation FROM curves WHERE source_id=? AND source_block=? ORDER BY id",
        (fid, first),
    ).fetchall()
    for cid, saved, label, note in rows:
        if label != family or note != annotation or not isinstance(saved, bytes):
            raise ValueError("imported AFV payload changed")
        try:
            world = np.load(io.BytesIO(saved), allow_pickle=False)
        except ValueError as error:
            raise ValueError("imported AFV payload changed") from error
        if world.dtype != np.float64 or world.ndim != 2 or world.shape[1] != 3 or len(world) < 2:
            raise ValueError("imported AFV payload changed")
        box = tuple(float(v) for pair in zip(world.min(0), world.max(0)) for v in pair)
        bound = db.execute(
            "SELECT min_x,max_x,min_y,max_y,min_z,max_z FROM curve_bounds WHERE id=?", (cid,)
        ).fetchone()
        if bound is None or not np.allclose(bound, box, rtol=2e-7, atol=1e-6):
            raise ValueError("imported AFV payload changed")
        checksum.update(hashlib.sha256(saved).digest())
    if checksum.hexdigest() != expected:
        raise ValueError("imported AFV inventory changed")
