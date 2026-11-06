"""Lossless native encoded transport with persistent source-cache receipts."""

import hashlib
import os
from pathlib import Path
import shutil
import sqlite3

from .util import file_sha
from .native_cache_index import cache_index


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
        # Native streaming reserves its declared size before writing any payload.
        with cache_index(self.root / '.vc_reader_cache.sqlite3') as db:
            for operation in ('INSERT', 'UPDATE'):
                db.execute(f'''CREATE TRIGGER IF NOT EXISTS ct_transfer_{operation.lower()}
                    BEFORE {operation} ON downloads WHEN NEW.size > 268435456
                    BEGIN SELECT RAISE(ABORT, 'encoded CT transfer reservation exceeds 256 MiB'); END''')
        self.bytes_read = self.cache_hits = 0
        self.db = sqlite3.connect(self.root / 'receipts.sqlite', timeout=120)
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
        receipt = self.db.execute('SELECT sha FROM receipts WHERE url=?', (url,)).fetchone()
        if receipt and receipt[0] is None and (self.offline or url in self._checked):
            return None
        if path.exists():
            if receipt is None or receipt[0] is None:
                if self.offline:
                    raise FileNotFoundError('offline cache miss: unverified ' + url)
                self._discard(path)
            elif file_sha(path) != receipt[0]:
                raise ValueError('corrupt native CT cache: ' + url)
        validator = self.db.execute('SELECT etag FROM validators WHERE url=?', (url,)).fetchone()
        if not self.offline and url not in self._checked and path.exists():
            # Native open_file now trusts disk bytes; freshness remains CLI-owned.
            from .io import fetch
            etag = validator[0] if validator else None
            headers = {'If-None-Match':etag} if etag and not etag.startswith('W/') else {}
            try:
                checked, etag = fetch(url, headers=headers)
            except FileNotFoundError:
                self._discard(path)
                self.db.execute('INSERT OR REPLACE INTO receipts VALUES (?,NULL)', (url,))
                self.db.execute('DELETE FROM validators WHERE url=?', (url,))
                self.db.commit();self._checked.add(url)
                return None
            if checked is None:
                if not headers:raise ValueError('unexpected remote 304 without a validator')
            else:
                self.bytes_read += len(checked)
                if hashlib.sha256(checked).hexdigest() != receipt[0]:
                    self._discard(path)
                    receipt = None
                    # A changed object is fetched through native bounded publication.
                    etag = None
                self.db.execute('INSERT OR REPLACE INTO validators VALUES (?,?)', (url,etag))
                self.db.commit()
        if not self.offline:
            available = shutil.disk_usage(self.root).free - self.reserve
            if available < 0:
                raise OSError('insufficient cache filesystem space for payload and reserve')
            cap = max(0,min(self.limit, self.size+available))
            if self._budget.max_bytes != cap:
                self._budget.set_max_bytes(cap)
        try:
            received = 0
            cached = path.exists()
            if self.offline:
                handle = path.open('rb');self.cache_hits += 1
            else:
                from vc_reader.reader.auth import resolve_access
                fd, received = self._budget.open_file(str(path), url, resolve_access(url))
                self.bytes_read += received
                if fd < 0:raise FileNotFoundError(url)
                self.cache_hits += int(cached and received == 0)
                handle = os.fdopen(fd, 'rb')
            with handle:
                if not self.offline and (received or not cached):
                    # Native downloads do not return validators; an earlier one belongs to other bytes.
                    self.db.execute('DELETE FROM validators WHERE url=?', (url,));self.db.commit()
                before = os.fstat(handle.fileno())
                raw = handle.read((256 << 20) + 1)
                if len(raw) > 256 << 20:
                    self._discard(path)
                    raise ValueError('encoded CT chunk exceeds 256 MiB read budget')
                sha = hashlib.sha256(raw).hexdigest()
                after = os.fstat(handle.fileno())
                if (before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                    raise ValueError('native CT cache changed during read: ' + url)
                if cached and not received:
                    if receipt is not None and receipt[0] != sha:raise ValueError('corrupt native CT cache: '+url)
        except FileNotFoundError as exc:
            if self.offline:
                raise FileNotFoundError('offline cache miss: ' + url) from exc
            raw, sha = None, None
            # Native 404 does not remove a previously cached inode.
            self._discard(path)
            self.db.execute('DELETE FROM validators WHERE url=?', (url,))
        if receipt != (sha,):
            self.db.execute('INSERT OR REPLACE INTO receipts VALUES (?,?)', (url, sha))
        if self.db.in_transaction:self.db.commit()
        self._checked.add(url)
        return raw

    def close(self):
        self.db.close()
