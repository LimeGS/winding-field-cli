"""Lossless native encoded transport with persistent source-cache receipts."""

import hashlib
import fcntl
from pathlib import Path
import shutil
import sqlite3

from .native_cache_index import cache_index
from .util import file_sha


def runtime_identity():
    from vc_reader import _reader
    extension=Path(_reader.__file__).resolve()
    libraries={p.resolve() for directory in (extension.parent,extension.parent.parent/'lib')
               for p in directory.glob('libvc_reader_native.*') if p.is_file()}
    return dict(extension_sha256=file_sha(extension),
                libraries_sha256={p.name:file_sha(p) for p in sorted(libraries)})


def cache_size(root):
    index = Path(root) / '.vc_reader_cache.sqlite3'
    if not index.exists():
        return 0
    with cache_index(index) as db:
        return db.execute('SELECT total FROM budget WHERE id=1').fetchone()[0]


def trim_cache(root, limit):
    """Reclaim the owned native namespace under Volume's parent cache lock."""
    root = Path(root).resolve()
    with cache_index(root / '.vc_reader_cache.sqlite3') as db:
        db.execute('BEGIN IMMEDIATE')
        total = db.execute('SELECT total FROM budget WHERE id=1').fetchone()[0]
        for name, size in db.execute('SELECT path,size FROM chunks ORDER BY age,path').fetchall():
            if total <= limit:
                break
            path = Path(name)
            if path.parent != root / 'encoded' or path.suffix != '.blob':
                raise ValueError('unexpected path in owned native CT cache')
            path.unlink(missing_ok=True)
            db.execute('DELETE FROM chunks WHERE path=?', (name,))
            _forget_file(db, name)
            total -= size
        db.execute('UPDATE budget SET max_bytes=? WHERE id=1', (limit,))


def _forget_file(db, path):
    # Existing caches may still have the older reader's validator table.
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('files','file_headers')").fetchall():
        db.execute(f'DELETE FROM {table} WHERE path=?', (path,))


class EncodedTransport:
    def __init__(self, root, limit, *, offline=False, reserve=0):
        from vc_reader.reader.disk_cache import disk_budget
        from vc_reader.reader import ObjectStore

        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit, self.reserve, self.offline = limit, reserve, offline
        index = self.root / '.vc_reader_cache.sqlite3'
        if index.exists():
            with cache_index(index) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_headers'").fetchone():
                    # The new reader drops file_headers before reconciling owned files.
                    db.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY)')
                    db.execute('INSERT OR IGNORE INTO files SELECT path FROM chunks WHERE path IN (SELECT path FROM file_headers)')
        self._budget = disk_budget(str(self.root), limit)
        self._budget.set_min_free_bytes(reserve)
        self._local = ObjectStore(self.root, auth=None, max_read_bytes=256 << 20)
        self._source = self._store = None
        self.bytes_read = self.cache_hits = 0
        # Each worker owns its transport; the coordinator closes them after joining workers.
        self.db = sqlite3.connect(self.root / 'receipts.sqlite', timeout=120, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS receipts (url TEXT PRIMARY KEY, sha TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS validators (url TEXT PRIMARY KEY, etag TEXT)')
        self.db.commit()
        self._checked = set()

    def path(self, url):
        return self.root / 'encoded' / (hashlib.sha256(url.encode()).hexdigest() + '.blob')

    @property
    def size(self):
        return cache_size(self.root)

    def _discard(self, path):
        with cache_index(self.root / '.vc_reader_cache.sqlite3') as db:
            db.execute('BEGIN IMMEDIATE')
            path.unlink(missing_ok=True)
            db.execute('DELETE FROM chunks WHERE path=?', (str(path),))
            _forget_file(db, str(path))

    def new_epoch(self):
        self._checked.clear()

    def read(self, url):
        path = self.path(url)
        path.parent.mkdir(exist_ok=True)
        # Serialize receipts with their payload across processes, but allow other chunks in parallel.
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._read(url)

    def _read(self, url):
        from vc_reader.reader import ObjectStore

        path = self.path(url)
        receipt = self.db.execute('SELECT sha FROM receipts WHERE url=?', (url,)).fetchone()
        if receipt and receipt[0] is None and (self.offline or url in self._checked):
            return None
        raw = self._local.read(str(path.relative_to(self.root)))
        if raw is not None:
            if receipt is None or receipt[0] is None:
                if self.offline:
                    raise FileNotFoundError('offline cache miss: unverified ' + url)
                self._discard(path);raw = None
            elif hashlib.sha256(raw).hexdigest() != receipt[0]:
                raise ValueError('corrupt native CT cache: ' + url)
        if self.offline:
            if raw is None:raise FileNotFoundError('offline cache miss: ' + url)
            self.cache_hits += 1
            return raw
        if raw is not None and url in self._checked:
            self.cache_hits += 1
            return raw
        available = shutil.disk_usage(self.root).free - self.reserve
        if available < 0:
            raise OSError('insufficient cache filesystem space for payload and reserve')
        cap = max(0,min(self.limit, self.size+available))
        if self._budget.max_bytes != cap:self._budget.set_max_bytes(cap)
        validator = self.db.execute('SELECT etag FROM validators WHERE url=?', (url,)).fetchone()
        etag = None
        if raw is not None:
            # Conditional HTTP is an authoritative-byte check, never a second decoding path.
            from .io import fetch
            etag = validator[0] if validator else None
            headers = {'If-None-Match':etag} if etag and not etag.startswith('W/') else {}
            try:
                checked, new_etag = fetch(url, headers=headers)
            except FileNotFoundError:
                raw = etag = None
            else:
                if checked is None:
                    if not headers:raise ValueError('unexpected remote 304 without a validator')
                    self.cache_hits += 1
                else:
                    self.bytes_read += len(checked);raw = checked;etag = new_etag
        else:
            source, name = url.rsplit('/', 1)
            if source != self._source:
                self._store = ObjectStore(source, auth=None, max_read_bytes=256 << 20)
                self._source = source
            raw = self._store.read(name)
            if raw is not None:self.bytes_read += len(raw)
        sha = None if raw is None else hashlib.sha256(raw).hexdigest()
        if raw is None or receipt != (sha,):
            self._discard(path)
        if raw is not None and len(raw) <= cap and not path.exists():
            import numpy as np
            path.parent.mkdir(exist_ok=True)
            self._budget.write(str(path), np.frombuffer(raw, np.uint8))
        if etag is None:
            self.db.execute('DELETE FROM validators WHERE url=?', (url,))
        elif validator != (etag,):
            self.db.execute('INSERT OR REPLACE INTO validators VALUES (?,?)', (url,etag))
        if receipt != (sha,):
            self.db.execute('INSERT OR REPLACE INTO receipts VALUES (?,?)', (url, sha))
        if self.db.in_transaction:self.db.commit()
        self._checked.add(url)
        return raw

    def close(self):
        self.db.close()
