"""Consume complete imported stages in a fresh run, retaining their producer identity."""

from contextlib import closing
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import time

from .util import file_sha

STAGES = {"prepare": "PREPARED.json", "fibres": "FIBRES.json"}
FILES = ("PREPARED.json", "LASAGNA.json", "FIBRES.json", "source.sqlite", "products.sqlite")
BOUNDED_FILES = ("PREPARED.json", "FIBRES.json", "source.sqlite", "products.sqlite")


def _import_directory(root):
    for name in ("lasagna-import", "lasagna-source"):
        if (root / name).is_dir():
            return name
    raise ValueError("reuse product directory missing or unsafe: lasagna-import")


def _directories(root, bounded=False):
    return (("coarse",) if bounded else ("maps", "lasagna")) + (_import_directory(root), "fibres")


VARIABLE = {
    "output",
    "reuse_from",
    "reuse_from_sha256",
    "field_chunk",
    "field_halo",
    "field_coarse_factor",
    "field_memory",
    "field_pair_scope",
    "field_prior_calibration",
    "field_orientation",
    "field_orientation_max_disagreement",
    "field_orientation_min_probes",
    "sweeps",
    "maxiter",
    "rtol",
    "fibre_weight",
    "nonconvergence_policy",
}


def _regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("reuse requires regular files: " + str(path))
    return path


def prepared_inventory(root):
    files = [_regular(root / name) for name in BOUNDED_FILES]
    for name in _directories(root, True):
        folder = root / name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("unsafe prepared asset directory")
        for p in folder.rglob("*"):
            if p.is_symlink():
                raise ValueError("symlink in prepared asset")
            if not p.is_dir():
                files.append(_regular(p))
    if any(p.name.endswith(("-wal", "-journal", "-shm")) for p in files):
        raise ValueError("prepared asset has open SQLite journals")
    return {str(p.relative_to(root)): file_sha(p) for p in files}


def seal_prepared_asset(root):
    from .util import atomic_json

    atomic_json(
        root / "PREPARED_ASSET.json",
        dict(schema="winding-prepared-asset-v1", files=prepared_inventory(root)),
    )
    return file_sha(root / "PREPARED_ASSET.json")


def validate_reuse_request(source: Path, config: dict, expected_sha: str) -> dict:
    if not isinstance(expected_sha, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha):
        raise ValueError("reuse requires the source RUN.json SHA256")
    source = Path(source).expanduser().absolute()
    if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_dir():
        raise ValueError("unsafe reuse source directory")
    source = source.resolve()
    output = Path(config["output"]).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("reuse source and output overlap")
    path = _regular(source / "RUN.json")
    if path.stat().st_size > 1 << 20:
        raise ValueError("reuse RUN.json exceeds metadata limit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("reuse source RUN.json changed")
    state = json.loads(raw)
    if (
        state.get("schema") != "winding-field-run-v1"
        or state.get("status") not in ("PAUSED", "FAILED", "BLOCKED", "COMPLETE", "PARTIAL")
        or not state.get("implementation_sha256")
    ):
        raise ValueError("reuse requires a stopped supported donor")
    old = state["config"]
    if state.get("reuse") or any(
        v.get("mode") == "reused" for v in state.get("stages", {}).values()
    ):
        raise ValueError("chained reuse unsupported; select the original producer run")
    if any(not c.get("fibres") or not c.get("lasagna_input") for c in (old, config)):
        raise ValueError("reuse supports imported Lasagna and AFV stages only")
    if old.get("bounded_preparation"):
        seal = _regular(source / "PREPARED_ASSET.json")
        if file_sha(seal) != state.get("prepared_asset_sha256"):
            raise ValueError("prepared asset seal changed")
        saved = json.loads(seal.read_bytes())
        if saved.get("schema") != "winding-prepared-asset-v1" or saved.get(
            "files"
        ) != prepared_inventory(source):
            raise ValueError("prepared asset inventory changed")
    changed = sorted(
        k for k in old.keys() | config.keys() if k not in VARIABLE and old.get(k) != config.get(k)
    )
    if changed:
        raise ValueError("reuse preparation configuration changed: " + ", ".join(changed))
    for stage, receipt in STAGES.items():
        record = state.get("stages", {}).get(stage, {})
        if record.get("receipt") != receipt or not re.fullmatch(
            "[0-9a-f]{64}", record.get("sha256", "")
        ):
            raise ValueError("reuse requires complete stage: " + stage)
    return state


def reuse_stages(
    source: Path, root: Path, config: dict, volume, expected_sha: str, stop=None
) -> dict:
    from .workflow import _validate_products, plan

    source = Path(source).expanduser().absolute()
    started = time.monotonic()
    state = validate_reuse_request(source, config, expected_sha)
    with _regular(source / ".run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        state = validate_reuse_request(source, config, expected_sha)
        estimate = plan(
            {k: v for k, v in config.items() if k not in ("reuse_from", "reuse_from_sha256")},
            volume,
        )
        if state["source"] != volume.identity or any(
            state["plan"].get(k) != estimate[k] for k in ("observations", "field")
        ):
            raise ValueError("reuse CT or preparation geometry changed")
        bounded = bool(state["config"].get("bounded_preparation"))
        files = [_regular(source / name) for name in (BOUNDED_FILES if bounded else FILES)]
        for name in _directories(source, bounded):
            folder = source / name
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError("reuse product directory missing or unsafe: " + name)
            for p in folder.rglob("*"):
                if p.is_symlink():
                    raise ValueError("symlink in reuse products")
                if not p.is_dir():
                    files.append(_regular(p))
        for p in files:
            if p.name.endswith(("-wal", "-journal", "-shm")):
                raise ValueError("reuse requires closed SQLite journals")
            for suffix in ("-wal", "-journal", "-shm"):
                if Path(str(p) + suffix).exists():
                    raise ValueError("reuse requires closed SQLite journals")
        required = sum(p.stat().st_size for p in files) + config.get("reserve_bytes", 0)
        if shutil.disk_usage(root).free < required:
            raise OSError("insufficient space for reuse copy and reserve")
        relative = {str(p.relative_to(source)) for p in files}
        with closing(
            sqlite3.connect(
                (source / "products.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
            )
        ) as db:
            exports = list(
                db.execute("SELECT path,stage,sha FROM exports WHERE stage IN ('prepare','fibres')")
            )
        if not exports or any(
            name not in relative or Path(name).is_absolute() or ".." in Path(name).parts
            for name, _, _ in exports
        ):
            raise ValueError("unsafe or incomplete reuse export inventory")
        for p in files:
            if stop and stop():
                raise InterruptedError("reuse copy paused")
            relative_path = p.relative_to(source)
            if relative_path == Path("products.sqlite") and not bounded:
                continue
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("reuse destination product already exists")
            before = file_sha(p)
            shutil.copy2(p, target)
            if file_sha(target) != before or file_sha(p) != before:
                raise ValueError("reuse source changed during copy")
        if not bounded:
            with closing(sqlite3.connect(root / "products.sqlite")) as db:
                db.execute("CREATE TABLE exports (path TEXT PRIMARY KEY,stage TEXT,sha TEXT)")
                db.executemany("INSERT INTO exports VALUES (?,?,?)", exports)
                db.commit()
        if stop and stop():
            raise InterruptedError("reuse validation paused")
        stages = {k: state["stages"][k] for k in STAGES}
        for stage, receipt in STAGES.items():
            if file_sha(root / receipt) != stages[stage]["sha256"]:
                raise ValueError("reuse stage receipt changed: " + stage)
        prepared = json.loads((root / "PREPARED.json").read_bytes())
        fibres = json.loads((root / "FIBRES.json").read_bytes())
        if (
            prepared.get("status") != "PREPARED"
            or fibres.get("status") != "FIBRES_READY"
            or fibres.get("method") != "external-afv"
        ):
            raise ValueError("reuse stages are incomplete or unsupported")
        geometry = prepared.get("native_geometry") if bounded else prepared
        if prepared.get("source") != volume.identity or any(
            geometry.get(k) != estimate["observations"][k]
            for k in ("shape_zyx", "origin_xyz", "spacing_xyz", "frame")
        ):
            raise ValueError("reuse prepared geometry changed")
        if not (root / "fibres/IMPORT.json").is_file():
            raise ValueError("reuse import provenance missing")
        if bounded:
            if (
                prepared_inventory(root)
                != json.loads((source / "PREPARED_ASSET.json").read_bytes())["files"]
            ):
                raise ValueError("prepared asset copy changed")
        else:
            _validate_inventory(root, prepared, fibres, exports, volume)
        _validate_products(root, dict(state, config=config, stages=stages), stop, volume)
        with closing(
            sqlite3.connect((root / "fibres/fibres.sqlite").as_uri() + "?mode=ro", uri=True)
        ) as db:
            if (
                db.execute("SELECT count(*) FROM done").fetchone()[0]
                != fibres["identity"]["input"]["block_count"]
                or db.execute("SELECT count(*) FROM curves").fetchone()[0] != fibres["count"]
            ):
                raise ValueError("reuse fibre inventory incomplete")
        volume.bind_ledger(root / "source.sqlite")
        volume.validate_used()
        if stop and stop():
            raise InterruptedError("reuse validation paused")
        if file_sha(source / "RUN.json") != expected_sha:
            raise ValueError("reuse donor changed during admission")
    return dict(
        stages={k: dict(v, mode="reused", seconds=0) for k, v in stages.items()},
        reuse=dict(
            source=str(source),
            run_sha256=expected_sha,
            implementation_sha256=state["implementation_sha256"],
            stages=stages,
            seconds=time.monotonic() - started,
        ),
    )


def _validate_inventory(root, prepared, fibres, exports, volume):
    # Legacy receipts do not seal source.sqlite; derive required CT coverage from the imported sampling lattice.
    axes = [
        sorted({(lo + i * prepared["stride"]) // chunk for i in range(n)})
        for lo, n, chunk in zip(prepared["roi_lo_zyx"], prepared["shape_zyx"], volume.chunks)
    ]
    separator = volume.metadata.get("dimension_separator", ".")
    prefix = volume.array_path + "/" if volume.array_path else ""
    with closing(
        sqlite3.connect((root / "source.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("corrupt reuse CT ledger")
        rows = dict(db.execute("SELECT key,sha FROM chunks"))
    for key, sha in rows.items():
        parts = key.removeprefix(prefix).split(separator)
        if (
            len(parts) != 3
            or any(not p.isdecimal() for p in parts)
            or prefix + separator.join(str(int(p)) for p in parts) != key
            or any(int(p) * c >= n for p, c, n in zip(parts, volume.chunks, volume.shape))
            or not isinstance(sha, str)
            or not re.fullmatch("[0-9a-f]{64}", sha)
        ):
            raise ValueError("invalid reuse CT ledger entry")
    if any(prefix + separator.join(map(str, key)) not in rows for key in itertools.product(*axes)):
        raise ValueError("incomplete reuse CT ledger")
    maps = json.loads((root / "maps/TILES.json").read_bytes())
    keys = [
        ".".join(map(str, key))
        for key in itertools.product(
            *(range((n + maps["chunk"] - 1) // maps["chunk"]) for n in maps["shape"])
        )
    ]
    imported = _import_directory(root)
    expected = {
        "fibres/fibres.afv": "fibres",
        "LASAGNA.json": "prepare",
        "lasagna/surfaces.lasagna.json": "prepare",
        imported + "/IDENTITY.json": "prepare",
    }
    for channel in ("presence", "nx", "ny", "grad_mag", "cos"):
        folder = "lasagna/surface_" + channel + ".ome.zarr/"
        expected.update({folder + name: "prepare" for name in (".zgroup", ".zattrs", "0/.zarray")})
        expected.update({folder + "0/" + key: "prepare" for key in keys})
    expected.update(
        {str(p.relative_to(root)): "prepare" for p in (root / imported).glob("document-*.json")}
    )
    inventory = {name: stage for name, stage, _ in exports}
    if any(inventory.get(name) != stage for name, stage in expected.items()):
        raise ValueError("incomplete reuse export inventory")
    if file_sha(root / "fibres/fibres.afv") != fibres.get("afv_sha256"):
        raise ValueError("reuse AFV export differs from stage receipt")
