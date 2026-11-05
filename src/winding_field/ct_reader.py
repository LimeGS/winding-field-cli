"""Lossless native encoded transport with persistent source-cache receipts."""

import hashlib
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
            db.execute('DELETE FROM file_headers WHERE path=?', (name,))
            total -= size
        db.execute('UPDATE budget SET max_bytes=? WHERE id=1', (limit,))


class EncodedTransport:
    def __init__(self, root, limit, *, offline=False, reserve=0):
        from vc_reader.reader.encoded_files import EncodedFiles
        from vc_reader.reader.disk_cache import disk_budget
        from vc_reader.errors import CacheMissError

        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit, self.reserve, self.offline = limit, reserve, offline
        self._budget = disk_budget(str(self.root), limit)
        # Native streaming reserves its declared size before writing any payload.
        with cache_index(self.root / '.vc_reader_cache.sqlite3') as db:
            for operation in ('INSERT', 'UPDATE'):
                db.execute(f'''CREATE TRIGGER IF NOT EXISTS ct_transfer_{operation.lower()}
                    BEFORE {operation} ON downloads WHEN NEW.size > 268435456
                    BEGIN SELECT RAISE(ABORT, 'encoded CT transfer reservation exceeds 256 MiB'); END''')
        self._miss = CacheMissError
        self.files = EncodedFiles(cache_dir=self.root, cache_max_bytes=None, offline=offline)
        self.db = sqlite3.connect(self.root / 'receipts.sqlite', timeout=120)
        self.db.execute('CREATE TABLE IF NOT EXISTS receipts (url TEXT PRIMARY KEY, sha TEXT)')
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
            db.execute('DELETE FROM file_headers WHERE path=?', (str(path),))

    def new_epoch(self):
        self.files.close()
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
        if not self.offline and url not in self._checked:
            # Last-Modified and weak ETags cannot establish byte identity on resume.
            # This schema is the pinned EncodedFiles disk-cache contract.
            with cache_index(self.root / '.vc_reader_cache.sqlite3') as db:
                db.execute("UPDATE file_headers SET modified='', etag=CASE WHEN etag LIKE 'W/%' THEN '' ELSE etag END WHERE path=?", (str(path),))
        if not self.offline:
            available = shutil.disk_usage(self.root).free - self.reserve
            if available < 0:
                raise OSError('insufficient cache filesystem space for payload and reserve')
            # Revalidation pins the old inode while downloading its replacement.
            pinned = path.stat().st_size if path.exists() and url not in self._checked else 0
            cap = max(0,min(self.limit-pinned, self.size+available-pinned))
            if self._budget.max_bytes != cap:
                self._budget.set_max_bytes(cap)
            if pinned and not path.exists():
                # Eviction happened before open_file could pin the old inode.
                available = max(0,shutil.disk_usage(self.root).free-self.reserve)
                self._budget.set_max_bytes(min(self.limit,self.size+available))
        try:
            with self.files.open(url) as (handle, identity):
                raw = handle.read((256 << 20) + 1)
                if len(raw) > 256 << 20:
                    self._discard(path)
                    raise ValueError('encoded CT chunk exceeds 256 MiB read budget')
                sha = hashlib.sha256(raw).hexdigest()
                if sha != identity[1]:
                    raise ValueError('native CT cache changed during read: ' + url)
        except self._miss as exc:
            raise FileNotFoundError('offline cache miss: ' + url) from exc
        except FileNotFoundError:
            if self.offline:
                raise
            raw, sha = None, None
            # Native 404 does not remove a previously cached inode.
            self._discard(path)
        if receipt != (sha,):
            self.db.execute('INSERT OR REPLACE INTO receipts VALUES (?,?)', (url, sha))
            self.db.commit()
        self._checked.add(url)
        return raw

    def close(self):
        self.files.close()
        self.db.close()
