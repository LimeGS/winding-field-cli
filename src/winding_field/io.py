"""Strict ZYX CT access; XYZ coordinates are micrometres, voxel centres.

Portable Zarr-v2 decoding never treats absent local chunks as known air.
The compressed disk payload budget is shared across sources and processes.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen

import numpy as np
from numcodecs import get_codec

from .util import atomic_bytes, atomic_json, digest, file_sha

CATALOG = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json'


def http_url(source):
    source = str(source).rstrip('/')
    if source.startswith('s3://'):
        p = urlparse(source)
        return f'https://{p.netloc}.s3.amazonaws.com{quote(p.path, safe="/%")}'
    return source


def fetch(url, *, headers=None, limit=256 << 20):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers or {}), timeout=60) as response:
                if int(response.headers.get('Content-Length', 0)) > limit:
                    raise ValueError(f'remote object exceeds {limit} byte read budget: {url}')
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError('remote response exceeds read budget')
                return data, response.headers.get('ETag')
        except HTTPError as exc:
            if exc.code == 304:
                return None, None
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)


def source_choices(scroll, catalog):
    sample = catalog.get('samples', {}).get(scroll)
    if sample is None:
        raise ValueError(f'unknown scroll {scroll}; use its catalog ID, e.g. PHerc0175A')
    choices = []
    for key, volume in sample.get('volumes', {}).items():
        for data in volume.get('data', []):
            if data['type'] not in ('ome-zarr', 'zarr'):
                continue
            for origin in data.get('origins', []):
                for root in origin.get('access_roots', []):
                    if root.get('usage') == 'public-read':
                        choices.append(dict(scan=key, source=root['url'].rstrip('/')+'/'+origin['path'].strip('/'),
                                            voxel_um=volume['properties'].get('pixel_size_um')))
    return choices


def resolve_source(scroll=None, source=None, scan=None):
    if source:
        return dict(source=str(source), voxel_um=None)
    if not scroll:
        raise ValueError('provide --source or --scroll')
    catalog = json.loads(fetch(CATALOG, limit=64 << 20)[0])
    choices = source_choices(scroll, catalog)
    if scan:
        choices = [c for c in choices if c['scan'] == scan or scan in c['source']]
    # Multiple mirrors of exactly the same artifact do not create scan ambiguity.
    choices = list({c['source']: c for c in choices}.values())
    if len(choices) != 1:
        raise ValueError('choose an exact --scan or --source; candidates: '+json.dumps(choices))
    return choices[0]


def bounds(lo, hi, shape):
    lo, hi = np.asarray(lo), np.asarray(hi)
    if (lo.shape != (3,) or hi.shape != (3,) or lo.dtype.kind not in 'iu'
            or hi.dtype.kind not in 'iu' or np.any(lo < 0) or np.any(hi <= lo)
            or np.any(hi > shape)):
        raise ValueError('ROI must be integer half-open ZYX bounds inside source')
    return lo.astype(np.int64), hi.astype(np.int64)


class Volume:
    def __init__(self, source, *, local=None, cache=None, cache_bytes=2 << 30,
                 offline=False, voxel_um=None, level=0):
        if type(cache_bytes) is not int or cache_bytes < 0 or type(level) is not int or level < 0:
            raise ValueError('invalid cache budget or pyramid level')
        self.source = http_url(source)
        self.remote = self.source.startswith(('https://', 'http://'))
        if not self.remote:
            self.source = str(Path(self.source).expanduser().resolve())
        self.local = Path(local).expanduser().resolve() if local else None
        self.cache = Path(cache or '~/.cache/winding-field/ct').expanduser()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.limit, self.offline = cache_bytes, offline
        self._stats = dict(local_hits=0, cache_hits=0, downloads=0, downloaded_bytes=0)
        self._validated = set()
        self._ledger = None
        self.db = sqlite3.connect(self.cache/'index.sqlite', timeout=120)
        self.db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, size INTEGER, at REAL, etag TEXT, sha TEXT)')
        self.db.commit()
        with self._lock():
            self._evict(0)
        self.array_path = str(level)
        attrs = self._metadata('.zattrs', optional=True)
        self.metadata = self._metadata(self.array_path+'/.zarray', optional=True)
        if self.metadata is None:
            self.array_path = ''
            self.metadata = self._metadata('.zarray')
            if level:
                raise ValueError('a direct Zarr array has no pyramid levels')
        m = self.metadata
        if m.get('zarr_format') != 2:
            raise ValueError('portable reader requires Zarr v2; unsupported layout is not substituted')
        self.shape, self.chunks = tuple(m['shape']), tuple(m['chunks'])
        if (len(self.shape) != 3 or len(self.chunks) != 3
                or any(type(n) is not int or n < 1 for n in self.shape+self.chunks)):
            raise ValueError('only 3D ZYX CT arrays are supported')
        self.dtype = np.dtype(m['dtype'])
        if self.dtype.kind not in 'uif' or self.dtype.hasobject:
            raise ValueError('CT must have a real numeric dtype')
        if math.prod(self.chunks)*self.dtype.itemsize > 256 << 20:
            raise ValueError('source chunk exceeds 256 MiB decode budget')
        if m.get('dimension_separator', '.') not in ('.', '/') or m['order'] not in ('C','F'):
            raise ValueError('invalid chunk layout')
        self.spacing_xyz, self.origin_xyz = self._geometry(attrs or {}, voxel_um)
        self.identity = dict(source=self.source, level=level, shape=list(self.shape),
                             spacing_xyz=self.spacing_xyz, origin_xyz=self.origin_xyz,
                             metadata_sha256=digest([attrs,m]))
        self._namespace = digest(self.identity)

    @contextmanager
    def _lock(self):
        with (self.cache/'cache.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _evict(self, incoming):
        size = self.db.execute('SELECT coalesce(sum(size),0) FROM cache').fetchone()[0]
        for key, n in self.db.execute('SELECT key,size FROM cache ORDER BY at').fetchall():
            if size + incoming <= self.limit:
                break
            (self.cache/key).unlink(missing_ok=True)
            self.db.execute('DELETE FROM cache WHERE key=?', (key,))
            size -= n
        self.db.commit()

    def _metadata(self, key, optional=False):
        cached = self.cache/('meta-'+digest([self.source,key])+'.json')
        try:
            if not self.remote:
                raw = (Path(self.source)/key).read_bytes()
            elif self.offline:
                raw = cached.read_bytes()
            else:
                raw, _ = fetch(self.source+'/'+key, limit=4 << 20)
                atomic_bytes(cached, raw)
        except FileNotFoundError:
            if optional:
                return None
            raise FileNotFoundError(f'metadata missing (offline={self.offline}): {self.source}/{key}')
        value = json.loads(raw)
        if self.local and (self.local/key).exists():
            if json.loads((self.local/key).read_bytes()) != value:
                raise ValueError('local metadata differs from original source: '+key)
        return value

    def _geometry(self, attrs, voxel_um):
        spacing, origin = np.ones(3), np.zeros(3)
        multiscales = attrs.get('multiscales', [])
        if multiscales:
            ms = multiscales[0]
            axes = ms.get('axes', [{'name': a} for a in 'zyx'])
            if [a if isinstance(a,str) else a['name'] for a in axes] != list('zyx'):
                raise ValueError('CT axes must be ZYX')
            selected = next((d for d in ms['datasets'] if d['path'] == self.array_path), None)
            if selected is None:
                raise ValueError('selected level missing from multiscales geometry')
            for t in selected.get('coordinateTransformations', [])+ms.get('coordinateTransformations', []):
                if t['type'] == 'scale':
                    spacing *= np.asarray(t['scale'], float)
                    origin *= np.asarray(t['scale'], float)
                elif t['type'] == 'translation':
                    origin += np.asarray(t['translation'], float)
                else:
                    raise ValueError('unsupported geometry transform')
            units = [a.get('unit') if isinstance(a,dict) else None for a in axes]
        else:
            units = [None]*3
        factors = {'micrometer':1., 'micrometre':1., 'um':1., 'nanometer':.001, 'millimeter':1000., 'meter':1e6}
        if all(u in factors for u in units):
            factors_zyx = np.array([factors[u] for u in units])
        elif units == [None]*3 and voxel_um is not None:
            factors_zyx = np.full(3, float(voxel_um))
        else:
            raise ValueError('unknown physical units; supply --voxel-um for unitless voxel metadata')
        spacing, origin = spacing*factors_zyx, origin*factors_zyx
        if not np.isfinite([spacing,origin]).all() or (spacing <= 0).any():
            raise ValueError('invalid physical geometry')
        return spacing[::-1].tolist(), origin[::-1].tolist()

    def bind_ledger(self, path):
        """Pin every consumed encoded chunk; changed bytes cannot enter a resumed run."""
        self._ledger = sqlite3.connect(path)
        self._ledger.execute('CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, sha TEXT NOT NULL)')
        self._ledger.commit()

    def _record(self, key, data):
        if self._ledger is not None:
            sha = hashlib.sha256(data).hexdigest()
            old = self._ledger.execute('SELECT sha FROM chunks WHERE key=?', (key,)).fetchone()
            if old and old[0] != sha:
                raise ValueError('source chunk changed: '+key)
            self._ledger.execute('INSERT OR IGNORE INTO chunks VALUES (?,?)', (key,sha))
            self._ledger.commit()
        return data

    def validate_used(self):
        if self._ledger is not None:
            for (key,) in self._ledger.execute('SELECT key FROM chunks'):
                self._chunk_bytes(key)

    def _chunk_bytes(self, key):
        local = (self.local/key) if self.local else (Path(self.source)/key if not self.remote else None)
        if local and local.is_file():
            self._stats['local_hits'] += 1
            return self._record(key, local.read_bytes())
        if not self.remote:
            raise FileNotFoundError(f'missing local CT chunk (not known air): {key}')
        token = digest([self._namespace,key])
        path = self.cache/token
        with self._lock():
            record = self.db.execute('SELECT etag,sha FROM cache WHERE key=?', (token,)).fetchone()
            raw = path.read_bytes() if record and path.exists() else None
            if raw is not None and hashlib.sha256(raw).hexdigest() != record[1]:
                raw = None
            if raw is not None and (self.offline or token in self._validated):
                self._stats['cache_hits'] += 1
                self.db.execute('UPDATE cache SET at=? WHERE key=?', (time.time(),token))
                self.db.commit()
                return self._record(key,raw)
            if self.offline:
                raise FileNotFoundError(f'offline cache miss: {key}')
            headers = {'If-None-Match':record[0]} if raw is not None and record[0] and not record[0].startswith('W/') else {}
            data, etag = fetch(self.source+'/'+key, headers=headers)
            if data is None:
                self._stats['cache_hits'] += 1
                data = raw
            else:
                self._stats['downloads'] += 1
                self._stats['downloaded_bytes'] += len(data)
                if len(data) <= self.limit:
                    self.db.execute('DELETE FROM cache WHERE key=?', (token,))
                    path.unlink(missing_ok=True)
                    self._evict(len(data))
                    atomic_bytes(path,data)
                    self.db.execute('INSERT INTO cache VALUES (?,?,?,?,?)',
                                    (token,len(data),time.time(),etag,hashlib.sha256(data).hexdigest()))
                    self.db.commit()
            self._validated.add(token)
            return self._record(key,data)

    def read(self, lo_zyx, hi_zyx):
        lo, hi = bounds(lo_zyx,hi_zyx,self.shape)
        if math.prod((hi-lo).tolist())*self.dtype.itemsize > 512 << 20:
            raise ValueError('read exceeds 512 MiB; use tiled reads')
        result = np.empty(tuple(hi-lo),dtype=self.dtype)
        chunks = np.array(self.chunks)
        for index in itertools.product(*(range(a//c,(b-1)//c+1) for a,b,c in zip(lo,hi,chunks))):
            name = self.metadata.get('dimension_separator','.').join(map(str,index))
            key = self.array_path+'/'+name if self.array_path else name
            raw = self._chunk_bytes(key)
            if self.metadata.get('compressor'):
                raw = get_codec(self.metadata['compressor']).decode(raw)
            for spec in reversed(self.metadata.get('filters') or []):
                raw = get_codec(spec).decode(raw)
            array = np.frombuffer(raw,dtype=self.dtype)
            if array.size != math.prod(self.chunks):
                raise ValueError('corrupt or truncated CT chunk: '+key)
            array = array.reshape(self.chunks,order=self.metadata['order'])
            start = np.array(index)*chunks
            a,b = np.maximum(start,lo),np.minimum(start+chunks,hi)
            result[tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo))] = array[
                tuple(slice(int(x),int(y)) for x,y in zip(a-start,b-start))]
        return result

    def stats(self):
        return dict(self._stats, cache_bytes=self.db.execute('SELECT coalesce(sum(size),0) FROM cache').fetchone()[0],
                    backend='portable-zarr-v2')

    def close(self):
        self.db.close()
        if self._ledger is not None:
            self._ledger.close()
