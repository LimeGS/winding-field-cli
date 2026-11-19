"""Small durable-file and identity helpers."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def sqlite_quota(db, quota_bytes):
    """Bound database pages plus a full rollback journal before any write."""
    page = db.execute("PRAGMA page_size").fetchone()[0]
    pages = (quota_bytes - 65536) // (2 * page + 512)
    if pages < 1 or db.execute("PRAGMA page_count").fetchone()[0] > pages:
        raise OSError("SQLite quota exhausted")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute(f"PRAGMA max_page_count={pages}")
