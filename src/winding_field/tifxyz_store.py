"""Immutable bounded tiles, a guarded shared ledger, and streaming TIFXYZ IO."""
from contextlib import contextmanager
from copy import deepcopy
import errno
import fcntl
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import resource
import signal
import sqlite3
import stat
import struct
import tempfile
import threading
import weakref
import zlib

import numpy as np

from .tifxyz_contract import Blocked, parse_json, validate_frame
from .util import digest, file_sha


_HEADER = struct.Struct('<8s10I')
_MAGIC = b'WFTILE1\0'
_PHASE_LOCK = threading.Lock()
_LIVE_STORES = weakref.WeakSet()


def _path(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('symlink in storage path')
    return path


def _regular(path):
    try:
        info = _path(path).stat()
    except FileNotFoundError as exc:
        raise ValueError('missing storage file: '+str(path)) from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('storage file is not a private regular file')
    return info


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _load_json(path):
    if _regular(path).st_size > 1 << 20:
        raise ValueError('storage identity header too large')
    return parse_json(path.read_text())


def _write_bytes(path, data):
    # Retain interrupted attempts; only the freshly fsynced file is renamed.
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    with os.fdopen(fd, 'wb') as stream:
        for offset in range(0, len(data), 8 << 20):
            stream.write(data[offset:offset+(8 << 20)])
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise ValueError('immutable destination already exists')
    os.rename(temporary, path)
    _fsync_dir(path.parent)


@contextmanager
def _file_limit(cap):
    if not _PHASE_LOCK.acquire(blocking=False):
        raise RuntimeError('concurrent file-size phases are not supported')
    previous = resource.getrlimit(resource.RLIMIT_FSIZE)
    handler = signal.getsignal(signal.SIGXFSZ)
    try:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        soft = cap if previous[0] == resource.RLIM_INFINITY else min(cap, previous[0])
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, previous[1]))
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, previous)
        signal.signal(signal.SIGXFSZ, handler)
        _PHASE_LOCK.release()


def _pair(value, name, minimum=0):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(v) is not int or not minimum <= v < 2**31 for v in value)):
        raise ValueError('invalid '+name)
    return tuple(value)


def _grid(grid):
    shape = _pair(grid['shape_yx'], 'grid shape', 1)
    if grid['tile_pixels'] != 256:
        raise ValueError('storage requires tile_pixels=256')
    if not math.isfinite(grid['pitch_um']) or grid['pitch_um'] <= 0:
        raise ValueError('invalid grid pitch')
    if len(grid['uv_min_um']) != 2 or not all(math.isfinite(v) for v in grid['uv_min_um']):
        raise ValueError('invalid UV origin')
    if not math.isfinite(grid['level']):
        raise ValueError('invalid grid level')
    return shape


class _Cursor(sqlite3.Cursor):
    def _plan(self, sql, parameters):
        if len(sql) > 65536:
            raise ValueError('SQL statement budget exceeded')
        writes = False
        if not sql.lstrip().upper().startswith('EXPLAIN'):
            with_plan = sqlite3.Connection.cursor(self.connection)
            writes = False
            text_factory = self.connection.text_factory
            # Virtual-table EXPLAIN P4 operands can contain opaque, non-UTF8 bytes.
            self.connection.text_factory = bytes
            try:
                plan = with_plan.execute('EXPLAIN '+sql, parameters)
                for row in plan:
                    if row[1] in (b'SorterOpen', b'OpenEphemeral', b'OpenAutoindex'):
                        raise ValueError('SQL global sort/materialization is forbidden')
                    writes |= row[1] == b'Transaction' and row[3] != 0
            finally:
                self.connection.text_factory = text_factory
                with_plan.close()
        return writes

    def execute(self, sql, parameters=()):
        if self._plan(sql,parameters):self.connection.owner.guard(sql=True)
        try:
            return super().execute(sql, parameters)
        except sqlite3.Error as exc:
            if (getattr(exc, 'sqlite_errorcode', 0) & 255) in (7,10,13):
                self.connection.owner._db_failure = exc
            raise

    def executemany(self, sql, parameters):
        batch = list(islice(iter(parameters), 129))
        if len(batch) > 128:
            raise ValueError('SQL insert batch exceeds 128 rows')
        self.connection.owner.guard(sql=True)
        if not batch:return self
        self._plan(sql,batch[0])
        try:
            return super().executemany(sql,batch)
        except sqlite3.Error as exc:
            if (getattr(exc,'sqlite_errorcode',0)&255) in (7,10,13):
                self.connection.owner._db_failure=exc
            raise

    def executescript(self, sql):
        raise sqlite3.DatabaseError('executescript bypasses guarded transactions')


class _Connection(sqlite3.Connection):
    def cursor(self, factory=_Cursor):
        return super().cursor(factory)

    def execute(self, sql, parameters=()):
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql, parameters):
        return self.cursor().executemany(sql, parameters)

    def executescript(self, sql):
        return self.cursor().executescript(sql)


class TileStore:
    """Single-writer store. SQL/IO failures poison this instance; reopen to recover."""

    def __init__(self, out, identity, budget, phase, resume=False):
        self.root = _path(out)
        self.identity = deepcopy(identity)
        self.grid = deepcopy(identity['grid'])
        self.shape = _grid(self.grid)
        self.budget, self.phase = budget, phase
        if phase not in ('projected', 'delivered'):
            raise ValueError('invalid store phase')
        self.scope = _path(budget.scope_root)
        if self.root != self.scope and self.scope not in self.root.parents:
            raise ValueError('store outside scope_root')
        for name in ('scratch_bytes', 'reserve_disk_bytes', 'database_bytes'):
            value = getattr(budget, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError('invalid '+name)
        self.database_cap = budget.database_bytes // 4096 * 4096
        if self.database_cap < 4*4096:
            raise Blocked('DATABASE_BUDGET')
        self.journal_cap = (101*self.database_cap+99)//100 + (2 << 20)
        self.db = self._lock = None
        self._db_failure = None
        self._active = self._control = self._poisoned = False
        self._binding = dict(format='winding-field-tiles-v1', identity=self.identity, phase=phase)
        encoded = _json_bytes(self._binding)
        if len(encoded) > 1 << 20:
            raise ValueError('storage identity header too large')
        self.scope.mkdir(parents=True, exist_ok=True)
        _fsync_dir(self.scope.parent)
        self.guard(pending_bytes=len(encoded)+16384, sql=True)
        if self.root.exists() and not resume and any(self.root.iterdir()):
            raise ValueError('foreign/nonempty storage destination')
        if resume and not self.root.is_dir():
            raise ValueError('missing owned storage destination')
        self.root.mkdir(parents=True, exist_ok=True)
        _fsync_dir(self.root.parent)
        try:
            lockpath = self.root/'.lock'
            fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            self._lock = os.fdopen(fd, 'r+b')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = self.root/'STORE.json'
            if resume:
                if _load_json(marker) != self._binding:
                    raise ValueError('store identity mismatch')
                _regular(self.root/'store.sqlite')
            else:
                _write_bytes(marker, encoded)
            self._connect()
            if not resume:
                with self.transaction() as db:
                    db.execute('CREATE TABLE binding (sha256 TEXT NOT NULL)')
                    db.execute('INSERT INTO binding VALUES (?)', (digest(self._binding),))
                    db.execute('CREATE TABLE tiles (ty INTEGER, tx INTEGER, bytes INTEGER NOT NULL, '
                               'sha256 TEXT NOT NULL, receipt_sha256 TEXT NOT NULL, '
                               'PRIMARY KEY(ty,tx)) WITHOUT ROWID')
            if self.db.execute('SELECT sha256 FROM binding').fetchone() != (digest(self._binding),):
                raise ValueError('ledger identity checksum mismatch')
            # R-tree first access prepares shadow-table writes even for a read-only check.
            with self.transaction():
                if self.db.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('corrupt ledger')
            for tid in self.iter_ids():
                self.read(tid)
        except BaseException:
            self.close()
            raise

    def _connect(self):
        if len(_LIVE_STORES) >= 4:
            raise Blocked('SQLITE_CACHE_BUDGET', 'at most four 16MiB stores per worker')
        with _file_limit(self.journal_cap):
            self.db = sqlite3.connect(self.root/'store.sqlite', isolation_level=None,
                                      cached_statements=0, timeout=0, factory=_Connection)
            self.db.owner = self
            for pragma in ('page_size=4096', 'journal_mode=DELETE', 'synchronous=FULL',
                           'cache_size=-16384', 'temp_store=MEMORY', 'automatic_index=OFF',
                           'mmap_size=0', f'max_page_count={self.database_cap//4096}'):
                self.db.execute('PRAGMA '+pragma)
            if self.db.execute('PRAGMA page_size').fetchone() != (4096,):
                raise ValueError('invalid ledger page size')
            if (self.root/'store.sqlite').stat().st_size > self.database_cap:
                raise Blocked('DATABASE_BUDGET')
        self.db.set_authorizer(self._authorize)
        _LIVE_STORES.add(self)

    def _authorize(self, action, arg1, arg2, database, trigger):
        deny = sqlite3.SQLITE_DENY
        if self._poisoned:
            return deny
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return deny
        if database == 'temp':
            return deny
        if action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
            return sqlite3.SQLITE_OK if self._control else deny
        if action == sqlite3.SQLITE_PRAGMA:
            if arg2 is not None:
                # A smaller cache is useful for bounded spill testing, never a larger one.
                if arg1 == 'cache_size' and self._active:
                    try:
                        n = int(arg2)
                        if -16384 <= n <= 4096:
                            return sqlite3.SQLITE_OK
                    except ValueError:
                        pass
                return deny
            return sqlite3.SQLITE_OK
        readonly = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE)
        if action not in readonly and not self._active:
            return deny
        return sqlite3.SQLITE_OK

    def guard(self, pending_bytes=0, sql=False):
        if self._poisoned:
            raise RuntimeError('poisoned store; close and resume')
        if type(pending_bytes) is not int or pending_bytes < 0:
            raise ValueError('invalid pending bytes')
        fs = os.statvfs(self.scope)
        # f_bsize is preferred I/O length on macOS, not allocated space per entry.
        quantum = max(fs.f_frsize, 4096)
        device = self.scope.stat().st_dev
        def charge(path):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError('symlink in storage scope')
            if info.st_dev != device:
                raise ValueError('storage scope spans filesystems')
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError('nonregular storage entry')
            total = max(info.st_size, info.st_blocks*512) + quantum
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    for entry in entries:
                        total += charge(Path(entry.path))
            return total
        used = charge(self.scope)
        reserve = pending_bytes + 2*quantum
        if sql:
            dbpath = self.root/'store.sqlite'
            current = dbpath.stat().st_size if dbpath.exists() else 0
            reserve += max(0, self.database_cap-current)+self.journal_cap
        if used+reserve > self.budget.scratch_bytes:
            raise Blocked('SCRATCH_BUDGET')
        if fs.f_bavail*fs.f_frsize < self.budget.reserve_disk_bytes+reserve:
            raise Blocked('DISK_HEADROOM')
        return dict(used_bytes=used, reserved_bytes=reserve)

    def _sql_control(self, statement):
        self._control = True
        try:
            self.db.execute(statement)
        finally:
            self._control = False

    @contextmanager
    def transaction(self):
        if self._active:
            raise RuntimeError('nested store transaction')
        self.guard(sql=True)
        with _file_limit(self.journal_cap):
            self._active = True
            try:
                self._sql_control('BEGIN IMMEDIATE')
                yield self.db
                if self._db_failure is not None:
                    raise self._db_failure
                self._sql_control('COMMIT')
                _fsync_dir(self.root)
            except BaseException as exc:
                failed = isinstance(exc, (sqlite3.Error, OSError))
                try:
                    if self.db.in_transaction:
                        self._sql_control('ROLLBACK')
                except sqlite3.Error:
                    failed = True
                if failed:
                    self._poisoned = True
                    self.db.close()
                    sqlcode = getattr(exc, 'sqlite_errorcode', 0) & 255
                    oscode = getattr(exc, 'errno', None)
                    if sqlcode in (7,10,13) or oscode in (errno.EFBIG, errno.ENOSPC):
                        code = 'FILE_SIZE_LIMIT' if sqlcode == 10 or oscode == errno.EFBIG else 'SQLITE_WRITE_LIMIT'
                        raise Blocked(code, str(exc)) from exc
                raise
            finally:
                self._active = False

    def _boxes(self, tile_id):
        tid = _pair(tile_id, 'tile ID')
        lo = [v*256 for v in tid]
        hi = [min(v+256, n) for v, n in zip(lo, self.shape)]
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError('tile outside grid')
        box = [[max(0, a-1) for a in lo], [min(n, b+1) for b, n in zip(hi, self.shape)]]
        return tid, [lo, hi], box

    def _validate_tile(self, tile):
        tid, owned, box = self._boxes(tile['tile_id'])
        if tile['owned_box_yx'] != owned or tile['box_yx'] != box:
            raise ValueError('incorrect tile ownership or halo box')
        shape = tuple(b-a for a, b in zip(*box))
        for key, dtype, dims in (('xyz', 'float32', (*shape, 3)), ('mask', 'bool', shape),
                                 ('root_error_um', 'float64', shape)):
            value = tile[key]
            if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != dims:
                raise ValueError('invalid tile '+key)
            if not np.isfinite(value).all():
                raise ValueError('nonfinite tile '+key)
        xyz, mask, error = (tile[k] for k in ('xyz', 'mask', 'root_error_um'))
        if (error < 0).any() or (xyz[~mask] != -1).any() or (error[~mask] != 0).any():
            raise ValueError('invalid tile mask/error encoding')
        if (xyz[mask] < 0).any():
            raise ValueError('negative valid XYZ')
        return tid, owned, box

    def _receipt(self, tid, size, sha):
        return dict(tile_id=list(tid), path=f'tile-{tid[0]}-{tid[1]}.bin', bytes=size, sha256=sha)

    def commit(self, tile):
        tid, owned, box = self._validate_tile(tile)
        size = _HEADER.size + tile['mask'].size*21
        self.guard()
        data = _HEADER.pack(_MAGIC, *tid, *owned[0], *owned[1], *box[0], *box[1])
        data += tile['xyz'].astype('<f4', copy=False).tobytes()
        data += tile['mask'].tobytes()+tile['root_error_um'].astype('<f8', copy=False).tobytes()
        receipt = self._receipt(tid, len(data), hashlib.sha256(data).hexdigest())
        row = self.db.execute('SELECT bytes,sha256,receipt_sha256 FROM tiles WHERE ty=? AND tx=?', tid).fetchone()
        if row:
            self.read(tid)
            if row != (size, receipt['sha256'], digest(receipt)):
                raise ValueError('immutable committed tile has different data')
            return receipt
        path = self.root/receipt['path']
        self.guard(pending_bytes=0 if path.exists() else size, sql=True)
        if path.exists() or path.is_symlink():
            if _regular(path).st_size != size or path.read_bytes() != data:
                raise ValueError('orphan payload differs from recomputed bytes')
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            _fsync_dir(self.root)
        else:
            _write_bytes(path, data)
        with self.transaction() as db:
            db.execute('INSERT INTO tiles VALUES(?,?,?,?,?)', (*tid, size, receipt['sha256'], digest(receipt)))
        return receipt

    def read(self, tile_id):
        tid, owned, box = self._boxes(tile_id)
        row = self.db.execute('SELECT bytes,sha256,receipt_sha256 FROM tiles WHERE ty=? AND tx=?', tid).fetchone()
        if row is None:
            raise ValueError('missing committed tile')
        size, sha, checksum = row
        receipt = self._receipt(tid, size, sha)
        shape = tuple(b-a for a, b in zip(*box))
        pixels = math.prod(shape)
        if size != _HEADER.size+21*pixels or digest(receipt) != checksum:
            raise ValueError('corrupt ledger receipt checksum')
        path = self.root/receipt['path']
        if _regular(path).st_size != size:
            raise ValueError('corrupt payload size')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('corrupt payload checksum')
        expected = (_MAGIC, *tid, *owned[0], *owned[1], *box[0], *box[1])
        if _HEADER.unpack_from(data) != expected:
            raise ValueError('corrupt block header')
        offset = _HEADER.size
        xyz = np.frombuffer(data, '<f4', pixels*3, offset).reshape(*shape, 3).copy()
        raw_mask = np.frombuffer(data, 'u1', pixels, offset+12*pixels)
        if (raw_mask > 1).any():
            raise ValueError('corrupt mask bytes')
        mask = raw_mask.astype(bool).reshape(shape)
        error = np.frombuffer(data, '<f8', pixels, offset+13*pixels).reshape(shape).copy()
        tile = dict(tile_id=list(tid), owned_box_yx=owned, box_yx=box, xyz=xyz,
                    mask=mask, root_error_um=error)
        self._validate_tile(tile)
        return tile

    def iter_ids(self):
        yield from self.db.execute('SELECT ty,tx FROM tiles ORDER BY ty,tx')

    def read_region(self, lo_yx, hi_yx):
        lo, hi = _pair(lo_yx, 'region low'), _pair(hi_yx, 'region high')
        shape = tuple(b-a for a, b in zip(lo, hi))
        if any(not 0 < s <= 260 or b > n for s, b, n in zip(shape, hi, self.shape)):
            raise ValueError('invalid region; maximum 260x260')
        result = dict(xyz=np.empty((*shape, 3), np.float32), mask=np.empty(shape, bool),
                      root_error_um=np.empty(shape, np.float64))
        for ty in range(lo[0]//256, (hi[0]-1)//256+1):
            for tx in range(lo[1]//256, (hi[1]-1)//256+1):
                tile = self.read((ty, tx))
                a = [max(v, o) for v, o in zip(lo, tile['owned_box_yx'][0])]
                b = [min(v, o) for v, o in zip(hi, tile['owned_box_yx'][1])]
                src = tuple(slice(x-o, y-o) for x, y, o in zip(a, b, tile['box_yx'][0]))
                dst = tuple(slice(x-o, y-o) for x, y, o in zip(a, b, lo))
                for key in result:
                    result[key][dst] = tile[key][src]
        return result

    def close(self):
        _LIVE_STORES.discard(self)
        if self.db is not None:
            self.db.close()
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _tiff_limits(shape):
    count = ((shape[0]+255)//256)*((shape[1]+255)//256)
    # tifffile's offset bookkeeping is admitted separately from pixel buffers.
    if 128*count+65536 > 16 << 20:
        raise ValueError('TIFF tile metadata budget exceeded')
    return count, 5*math.prod(shape)+(1 << 20)+4096*count


def _tiff_header(stream, shape, size):
    """Parse only our single-plane layout, without a general TIFF/series parser."""
    def read_at(offset, count):
        if offset < 0 or offset+count > size:
            raise ValueError('TIFF header/offset outside file')
        stream.seek(offset)
        data = stream.read(count)
        if len(data) != count:
            raise ValueError('truncated TIFF')
        return data
    header = read_at(0, 8)
    if header[:2] not in (b'II', b'MM'):
        raise ValueError('invalid TIFF header')
    endian = '<' if header[:2] == b'II' else '>'
    version = struct.unpack(endian+'H', header[2:4])[0]
    if version == 42:
        unit, entry, count_size, fmt = 4, 12, 2, 'I'
        ifd = struct.unpack(endian+'I', header[4:8])[0]
        header_size = 8
    elif version == 43 and header[4:8] == struct.pack(endian+'HH', 8, 0):
        unit, entry, count_size, fmt = 8, 20, 8, 'Q'
        ifd = struct.unpack(endian+'Q', read_at(8, 8))[0]
        header_size = 16
    else:
        raise ValueError('unsupported TIFF header')
    count = struct.unpack(endian+('H' if count_size == 2 else 'Q'), read_at(ifd, count_size))[0]
    if ifd < header_size or not 1 <= count <= 32:
        raise ValueError('TIFF tag count/header budget exceeded')
    end = ifd+count_size+entry*count+unit
    if struct.unpack(endian+fmt, read_at(end-unit, unit))[0] != 0:
        raise ValueError('multiple TIFF pages/series are forbidden')
    tile_count, _ = _tiff_limits(shape)
    tags = {}
    ranges = [(0, header_size), (ifd, end)]
    allowed = {254,256,257,258,259,262,266,274,277,282,283,284,296,305,317,322,323,324,325,339}
    widths = {1:1, 2:1, 3:2, 4:4, 5:8, 16:8}
    for index in range(count):
        pos = ifd+count_size+entry*index
        raw = read_at(pos, entry)
        tag, dtype = struct.unpack(endian+'HH', raw[:4])
        number = struct.unpack(endian+fmt, raw[4:4+unit])[0]
        if tag not in allowed or tag in tags or dtype not in widths:
            raise ValueError('unsupported/duplicate TIFF layout tag')
        expected_count = tile_count if tag in (324,325) else 1
        if tag == 305:
            if dtype != 2 or not 1 <= number <= 256:
                raise ValueError('TIFF static metadata budget exceeded')
        elif number != expected_count:
            raise ValueError('TIFF tag payload count mismatch')
        if tag in (282,283):
            if dtype != 5:
                raise ValueError('invalid TIFF resolution type')
        elif tag != 305 and dtype not in (3,4,16):
            raise ValueError('invalid TIFF scalar/offset type')
        length = widths[dtype]*number
        offset = pos+4+unit
        if length > unit:
            offset = struct.unpack(endian+fmt, raw[4+unit:])[0]
            if offset < header_size or offset+length > size:
                raise ValueError('TIFF tag offset outside file')
            if any(offset < b and offset+length > a for a, b in ranges):
                raise ValueError('overlapping TIFF metadata')
            ranges.append((offset, offset+length))
        tags[tag] = (dtype, offset)

    def scalar(tag, index=0, default=None):
        if tag not in tags:
            if default is not None:
                return default
            raise ValueError('missing TIFF layout tag')
        dtype, offset = tags[tag]
        return struct.unpack(endian+{3:'H',4:'I',16:'Q'}[dtype],
                             read_at(offset+index*widths[dtype], widths[dtype]))[0]

    checks = {256:shape[1],257:shape[0],258:32,262:1,277:1,322:256,323:256,339:3}
    if any(scalar(tag) != value for tag, value in checks.items()):
        raise ValueError('TIFF shape/dtype/tile layout mismatch')
    if scalar(259) not in (8,32946):
        raise ValueError('TIFF must use deflate')
    for tag, default in ((254,0),(266,1),(274,1),(284,1),(317,1)):
        if scalar(tag, default=default) != default:
            raise ValueError('unsupported TIFF layout')

    def segment(index):
        return scalar(324, index), scalar(325, index)

    previous_end = 0
    for index in range(tile_count):
        offset, length = segment(index)
        if (not 0 < length <= 5*256*256 or offset < previous_end or offset+length > size
                or any(offset < b and offset+length > a for a, b in ranges)):
            raise ValueError('invalid TIFF tile offset/size')
        previous_end = offset+length
    return endian, tile_count, segment, read_at


def iter_tiff_tiles(path, expected_shape, max_pixels):
    """Yield cropped 256² tiles; header checks precede any bounded deflate decode."""
    shape = _pair(expected_shape, 'TIFF expected shape', 1)
    if type(max_pixels) is not int or max_pixels < math.prod(shape):
        raise ValueError('TIFF pixel budget exceeded')
    path = _path(path)
    info = _regular(path)
    _, cap = _tiff_limits(shape)
    if info.st_size > cap or info.st_blocks*512 > cap+4096:
        raise ValueError('TIFF file budget exceeded')
    with path.open('rb') as stream:
        endian, count, segment, read_at = _tiff_header(stream, shape, info.st_size)
        columns = (shape[1]+255)//256
        for index in range(count):
            offset, length = segment(index)
            decoder = zlib.decompressobj()
            try:
                raw = decoder.decompress(read_at(offset, length), 256*256*4+1)
            except zlib.error as exc:
                raise ValueError('corrupt TIFF tile deflate') from exc
            if (len(raw) != 256*256*4 or not decoder.eof or decoder.unused_data
                    or decoder.unconsumed_tail):
                raise ValueError('TIFF decoded tile size mismatch')
            array = np.frombuffer(raw, endian+'f4').reshape(256,256)
            y, x = index//columns*256, index%columns*256
            h, w = min(256, shape[0]-y), min(256, shape[1]-x)
            if not np.isfinite(array).all():
                raise ValueError('nonfinite TIFF tile')
            if (array[h:] != -1).any() or (array[:,w:] != -1).any():
                raise ValueError('TIFF tail padding must be -1')
            yield (y,x), array[:h,:w].astype(np.float32, copy=True)


def write_segment(store, grid, frame, pending_dir):
    """Assemble and verify immutable artifacts. Publication/READY belongs to T6.

    Manifest: {files: {relative_path: {bytes, sha256}}, shape_yx, dtype, scale}.
    Interrupted temporary files remain charged; retries verify completed channels.
    """
    import tifffile

    frame = validate_frame(frame)
    if grid != store.grid:
        raise ValueError('grid identity mismatch')
    bound_frame = store.identity.get('frame', store.identity.get('output_frame'))
    if bound_frame is not None and frame != bound_frame:
        raise ValueError('frame identity mismatch')
    shape = _grid(grid)
    _, cap = _tiff_limits(shape)
    pending = _path(pending_dir)
    if pending == store.root or store.scope not in pending.parents:
        raise ValueError('pending destination outside owned scope')
    binding = dict(store_sha256=digest(store._binding), grid=grid, output_frame=frame)
    marker = pending/'.segment.json'
    if pending.exists() and any(pending.iterdir()):
        if not marker.exists():
            raise ValueError('foreign pending destination')
        if _load_json(marker) != binding:
            raise ValueError('pending identity mismatch')
    store.guard(pending_bytes=3*cap+(1 << 20))
    pending.mkdir(parents=True, exist_ok=True)
    _fsync_dir(pending.parent)
    if not marker.exists():
        _write_bytes(marker, _json_bytes(binding))

    def channels(axis):
        for y in range(0, shape[0], 256):
            for x in range(0, shape[1], 256):
                store.guard(pending_bytes=5*256*256+4096)
                hi = min(y+256, shape[0]), min(x+256, shape[1])
                region = store.read_region((y,x), hi)
                if (region['xyz'][region['mask']] > np.asarray(frame['shape_zyx'][::-1])-1).any():
                    raise ValueError('XYZ outside output frame bounds')
                output = np.full((256,256), -1, np.float32)
                output[:hi[0]-y,:hi[1]-x] = region['xyz'][...,axis]
                yield output

    files = {}
    for axis, name in enumerate(('x.tif','y.tif','z.tif')):
        path = pending/name
        if not path.exists():
            store.guard(pending_bytes=cap)
            try:
                with tempfile.NamedTemporaryFile(prefix='.'+name+'-', dir=pending,
                                                  mode='w+b', delete=False) as stream, _file_limit(cap):
                    temporary = stream.name
                    with tifffile.TiffWriter(stream, bigtiff=cap >= 2**32-(32 << 20), byteorder='<') as writer:
                        writer.write(channels(axis), shape=shape, dtype='float32', tile=(256,256),
                                     compression='deflate', photometric='minisblack', metadata=None,
                                     software=False, maxworkers=1, buffersize=1 << 20)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _regular(temporary).st_size > cap:
                    raise Blocked('FILE_SIZE_LIMIT', 'TIFF encoder exceeded channel cap')
                os.rename(temporary, path)
                _fsync_dir(pending)
            except OSError as exc:
                if exc.errno in (errno.EFBIG, errno.ENOSPC):
                    raise Blocked('FILE_SIZE_LIMIT', str(exc)) from exc
                raise
        for (y,x), array in iter_tiff_tiles(path, shape, math.prod(shape)):
            expected = store.read_region((y,x), (y+array.shape[0], x+array.shape[1]))['xyz'][...,axis]
            if not np.array_equal(array, expected):
                raise ValueError('TIFF differs from committed tile bytes')
        files[name] = dict(bytes=_regular(path).st_size, sha256=file_sha(path))
    scale = [frame['voxel_um']/grid['pitch_um']]*2
    meta = dict(format='tifxyz', type='seg', uuid=digest(binding),
                shape_yx=list(shape), dtype='float32', scale=scale, level=grid['level'],
                pitch_um=grid['pitch_um'], uv_min_um=grid['uv_min_um'], output_frame=frame)
    path = pending/'meta.json'
    if path.exists():
        if _load_json(path) != meta:
            raise ValueError('immutable metadata mismatch')
    else:
        encoded = _json_bytes(meta)
        store.guard(pending_bytes=len(encoded))
        _write_bytes(path, encoded)
    files['meta.json'] = dict(bytes=_regular(path).st_size, sha256=file_sha(path))
    _fsync_dir(pending)
    store.guard()
    return dict(files=files, shape_yx=list(shape), dtype='float32', scale=scale)
