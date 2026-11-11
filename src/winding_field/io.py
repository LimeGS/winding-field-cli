"""Strict ZYX CT access; XYZ coordinates are micrometres, voxel centres.

Checked native Zarr-v2 decoding never treats absent local chunks as known air.
The compressed disk payload budget is shared across sources and processes.
"""

from contextlib import contextmanager, nullcontext
import fcntl
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen

import numpy as np

from .util import atomic_bytes, digest
from .ct_reader import cache_size,trim_cache,runtime_identity

CATALOG = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json'


def http_url(source):
    source = str(source).rstrip('/')
    if source.startswith('s3://'):
        p = urlparse(source)
        return f'https://{p.netloc}.s3.amazonaws.com{quote(p.path, safe="/%")}'
    return source


def fetch(url, *, headers=None, limit=256 << 20, stream_policy=False):
    for attempt in range(3):
        try:
            if stream_policy:
                from .stream_policy import open_stream
                response=open_stream(url,headers or {})
            else:response=urlopen(Request(url,headers=headers or {}),timeout=60)
            with response:
                if int(response.headers.get('Content-Length', 0)) > limit:
                    raise ValueError(f'remote object exceeds {limit} byte read budget: {url}')
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError('remote response exceeds read budget')
                encoding=response.headers.get('Content-Encoding','identity').lower().strip()
                if stream_policy and encoding not in ('','identity'):
                    raise ValueError('unsupported streaming Content-Encoding')
                if encoding=='gzip':
                    with gzip.GzipFile(fileobj=io.BytesIO(data)) as decoded:
                        data=decoded.read(limit+1)
                    if len(data)>limit:
                        raise ValueError('decoded remote response exceeds read budget')
                elif encoding not in ('','identity'):
                    raise ValueError('unsupported HTTP Content-Encoding: '+encoding)
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


def resolve_source(scroll=None, source=None, scan=None, *, cache=None, offline=False):
    if source:
        return dict(source=str(source), voxel_um=None)
    if not scroll:
        raise ValueError('provide --source or --scroll')
    path=Path(cache or '~/.cache/winding-field/ct').expanduser()/'catalog.json'
    if offline:
        if not path.exists():
            raise FileNotFoundError('offline catalog missing; use --source or resolve --scroll online once with this cache')
        catalog=json.loads(path.read_bytes())
    else:
        raw=fetch(CATALOG,limit=64<<20)[0]
        catalog=json.loads(raw)
        atomic_bytes(path,raw)
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
                 offline=False, voxel_um=None, level=0, reserve_bytes=0, decoded_cache_bytes=0,
                 stream_policy=False, metadata_pins=None):
        self.stream_policy=stream_policy;self.metadata_pins=metadata_pins or {}
        try:
            from vc_reader.reader import ObjectStore, ZarrReader, ArrayMeta, decode_chunk
        except ImportError as exc:
            raise RuntimeError('vc_reader with bounded object reads and decode_chunk is required; install the pinned runtime') from exc
        if type(cache_bytes) is not int or cache_bytes < 0 or type(level) is not int or level < 0:
            raise ValueError('invalid cache budget or pyramid level')
        if type(reserve_bytes) is not int or reserve_bytes<0:
            raise ValueError('invalid cache reserve')
        if type(decoded_cache_bytes) is not int or decoded_cache_bytes<0:raise ValueError('invalid decoded cache budget')
        from collections import OrderedDict
        self._decoded=OrderedDict();self._decoded_bytes=0;self._decoded_limit=decoded_cache_bytes
        self._decoded_lock=threading.RLock()
        self.reserve=reserve_bytes
        self.source = http_url(source)
        self.remote = self.source.startswith(('https://', 'http://'))
        if not self.remote:
            self.source = str(Path(self.source).expanduser().resolve())
        self.local = Path(local).expanduser().resolve() if local else None
        self.cache = Path(cache or '~/.cache/winding-field/ct').expanduser()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.limit, self.offline = cache_bytes, offline
        self._stats = dict(local_hits=0, cache_hits=0, downloads=0, downloaded_bytes=0)
        self._stats.update(decoded_hits=0,decodes=0,decoded_evictions=0)
        self._verified_local = {}
        self._ledger = None
        self.db = None
        self._native = None
        self._reader = None
        self._transports = {}
        self._transport_lock = threading.RLock()
        self._transport = 'vc_reader.DiskCache' if self.remote else 'vc_reader.ObjectStore'
        self._objects = ObjectStore(self.local or self.source, auth=None, max_read_bytes=256 << 20) if self.local or not self.remote else None
        self.array_path = str(level)
        attrs = self._metadata('.zattrs', optional=True)
        self.metadata = self._metadata(self.array_path+'/.zarray', optional=True)
        if self.metadata is None:
            self.array_path = ''
            self.metadata = self._metadata('.zarray')
            if level:
                raise ValueError('a direct Zarr array has no pyramid levels')
        m = self.metadata
        if self.stream_policy:
            from .stream_policy import validate_array
            validate_array(m)
        if m.get('zarr_format') != 2:
            raise ValueError('checked reader requires Zarr v2; unsupported layout is not substituted')
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
        try:
            ArrayMeta.from_zarray(m).codec_id
        except (ValueError,RuntimeError) as exc:
            raise ValueError(f'unsupported native CT layout/codec: {exc}; use the original v1.5.0 runtime for this input') from exc
        self.spacing_xyz, self.origin_xyz = self._geometry(attrs or {}, voxel_um)
        self.identity = dict(source=self.source, level=level, shape=list(self.shape),
                             spacing_xyz=self.spacing_xyz, origin_xyz=self.origin_xyz,
                             metadata_sha256=digest([attrs,m]),backend='vc_reader-zarr-v2-checked-v1',
                             native_runtime=runtime_identity())
        from .ct_reader import EncodedTransport
        try:
            self.db = sqlite3.connect(self.cache/'index.sqlite', timeout=120, check_same_thread=False)
            self.db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, size INTEGER, at REAL, etag TEXT, sha TEXT)')
            self.db.commit()
            with self._lock():
                self._evict(0)
                legacy = self.db.execute('SELECT coalesce(sum(size),0) FROM cache').fetchone()[0]
                self._native_limit = max(0,self.limit-legacy)
                if self.remote:
                    self._native = EncodedTransport(self.cache/'vc-reader', self._native_limit,
                                                    offline=self.offline,reserve=self.reserve,stream_policy=self.stream_policy)
                    self._transports[threading.get_ident()] = self._native
            self._reader = ZarrReader.computed(self._chunk, shape=self.shape, chunks=self.chunks,
                dtype=self.dtype.str, fill_value=m.get('fill_value',0), memory_cache=0, threads=1)
        except BaseException:
            self.close();raise

    def _chunk(self,index):
        name=self.metadata.get('dimension_separator','.').join(map(str,index))
        return self.decoded_chunk(self.array_path+'/'+name if self.array_path else name)

    @contextmanager
    def _lock(self):
        with (self.cache/'cache.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _evict(self, incoming):
        size = self.db.execute('SELECT coalesce(sum(size),0) FROM cache').fetchone()[0] + cache_size(self.cache/'vc-reader')
        for key, n in self.db.execute('SELECT key,size FROM cache ORDER BY at').fetchall():
            if size + incoming <= self.limit:
                break
            (self.cache/key).unlink(missing_ok=True)
            self.db.execute('DELETE FROM cache WHERE key=?', (key,))
            size -= n
        self.db.commit()
        if size+incoming>self.limit:
            trim_cache(self.cache/'vc-reader',max(0,self.limit-incoming))

    def _metadata(self, key, optional=False):
        cached = self.cache/('meta-'+digest([self.source,key])+'.json')
        try:
            if not self.remote:
                raw = (Path(self.source)/key).read_bytes()
            elif self.offline:
                raw = (cached if cached.exists() else self.local/key if self.local else cached).read_bytes()
            else:
                raw, _ = fetch(self.source+'/'+key, limit=4 << 20,stream_policy=self.stream_policy)
        except FileNotFoundError:
            if self.source+'/'+key in self.metadata_pins:
                raise ValueError('pinned streaming metadata disappeared: '+key)
            if optional:
                return None
            raise FileNotFoundError(f'metadata missing (offline={self.offline}): {self.source}/{key}')
        expected=self.metadata_pins.get(self.source+'/'+key)
        if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('streaming metadata differs from imported snapshot: '+key)
        value = json.loads(raw)
        if self.local and (self.local/key).exists():
            if json.loads((self.local/key).read_bytes()) != value:
                raise ValueError('local metadata differs from original source: '+key)
        if self.remote and not self.offline:atomic_bytes(cached,raw)
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
        self._decoded.clear();self._decoded_bytes=0
        self._ledger = sqlite3.connect(path, check_same_thread=False)
        self._ledger.execute('CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, sha TEXT NOT NULL)')
        self._ledger.commit()

    def _record(self, key, data):
        if self._ledger is not None:
            sha = digest('authoritative-zarr-fill') if data is None else hashlib.sha256(data).hexdigest()
            with self._transport_lock:
                old = self._ledger.execute('SELECT sha FROM chunks WHERE key=?', (key,)).fetchone()
                if old and old[0] != sha:
                    raise ValueError('source chunk changed: '+key)
                if old is None:
                    self._ledger.execute('INSERT INTO chunks VALUES (?,?)', (key,sha))
                    self._ledger.commit()
        return data

    def validate_used(self,*,fresh=True):
        """Check ledger-bound encoded bytes, bypassing decoded RAM.

        Fresh checks start a new remote verification epoch (including resume).
        Within an invocation, fresh=False reuses per-chunk remote verification;
        local/cache bytes are still hashed against the ledger on every check.
        This is a per-chunk read snapshot, not an atomic remote-volume snapshot.
        """
        if fresh:
            self._decoded.clear();self._decoded_bytes=0
            self._verified_local.clear()
            for transport in self._transports.values():transport.new_epoch()
        if self._ledger is not None:
            for (key,) in self._ledger.execute('SELECT key FROM chunks'):
                self._chunk_bytes(key)

    def _chunk_bytes(self, key):
        local = (self.local/key) if self.local else (Path(self.source)/key if not self.remote else None)
        if local and local.is_file():
            raw = self._objects.read(key)
            if raw is None:raise FileNotFoundError('local CT chunk disappeared: '+key)
            sha = hashlib.sha256(raw).hexdigest()
            if self.remote and not self.offline and self._verified_local.get(key) != sha:
                if self._native_bytes(key) != raw:
                    raise ValueError('local CT chunk differs from original source: '+key)
                self._verified_local[key] = sha
            self._stats['local_hits'] += 1
            return self._record(key, raw)
        if not self.remote:
            raise FileNotFoundError(f'missing local CT chunk (not known air): {key}')
        return self._record(key,self._native_bytes(key))

    def _native_bytes(self,key):
        with self._transport_lock:
            with self._lock():
                if self.db.execute('SELECT COUNT(*) FROM cache').fetchone()[0]:
                    self._evict(min(self.limit,256<<20))
                legacy = self.db.execute('SELECT coalesce(sum(size),0) FROM cache').fetchone()[0]
                self._native_limit = max(0,self.limit-legacy)
            worker = threading.get_ident()
            if worker not in self._transports:
                from .ct_reader import EncodedTransport
                transport = EncodedTransport(self.cache/'vc-reader',self._native_limit,
                                             offline=self.offline,reserve=self.reserve,stream_policy=self.stream_policy)
                transport._checked = self._native._checked
                self._transports[worker] = transport
            transport = self._transports[worker]
            transport.limit = self._native_limit
        before,hits = transport.bytes_read,transport.cache_hits
        try:
            return transport.read(self.source+'/'+key)
        finally:
            with self._transport_lock:
                received = transport.bytes_read-before
                self._stats['downloaded_bytes'] += received
                self._stats['downloads'] += int(received>0)
                self._stats['cache_hits'] += transport.cache_hits-hits

    def _encoded_signature(self,key):
        local=self.local/key if self.local else Path(self.source)/key if not self.remote else None
        path=local if local and local.is_file() else self._native.path(self.source+'/'+key) if self.remote else local
        try:info=path.stat()
        except FileNotFoundError:return path,None
        return path,(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)

    def decoded_chunk(self,key):
        """Bounded immutable chunks; ledger-checked snapshots or stat-checked hits.

        validate_used discards this cache and revalidates authoritative sources.
        """
        with self._decoded_lock if self._decoded_limit else nullcontext():
            return self._decode_chunk(key)

    def _decode_chunk(self,key):
        ledger_bound=self._ledger is not None
        before=self._encoded_signature(key) if self._decoded_limit and not ledger_bound else None
        cached=self._decoded.get(key)
        if cached is not None:
            array,signature=cached
            if ledger_bound or before==signature:
                # bind_ledger clears RAM; its SHA pins the bytes used to decode every retained array.
                if ledger_bound:self._chunk_bytes(key)
                if ledger_bound or self._encoded_signature(key)==signature:
                    self._decoded.move_to_end(key);self._stats['decoded_hits']+=1
                    return array
            self._decoded_bytes-=array.nbytes+512;del self._decoded[key]
        raw=self._chunk_bytes(key)
        if raw is None:return None
        retained_signature=None
        if self._decoded_limit and not ledger_bound:
            after=self._encoded_signature(key)
            if after[1] is not None:
                if before!=after:
                    if self._chunk_bytes(key)!=raw or self._encoded_signature(key)!=after:
                        raise ValueError('CT source/cache changed during decode')
                retained_signature=after
        from vc_reader.reader import decode_chunk
        metadata=self.metadata
        if self.stream_policy:
            from .stream_policy import decode_chunk as bounded_decode
            raw=bounded_decode(raw,metadata)
            metadata={**metadata,'compressor':None}
        raw=decode_chunk(raw,metadata,max_decode_bytes=256<<20)
        array=np.frombuffer(raw,dtype=self.dtype)
        if array.size!=math.prod(self.chunks):raise ValueError('corrupt CT chunk: '+key)
        array=array.reshape(self.chunks);array.flags.writeable=False
        self._stats['decodes']+=1;charge=array.nbytes+512
        if (ledger_bound or retained_signature is not None) and charge<=self._decoded_limit:
            while self._decoded_bytes+charge>self._decoded_limit:
                _,(retired,_)=self._decoded.popitem(last=False)
                self._decoded_bytes-=retired.nbytes+512;self._stats['decoded_evictions']+=1
            self._decoded[key]=(array,retained_signature);self._decoded_bytes+=charge
        return array

    def read(self, lo_zyx, hi_zyx):
        lo, hi = bounds(lo_zyx,hi_zyx,self.shape)
        if math.prod((hi-lo).tolist())*self.dtype.itemsize > 512 << 20:
            raise ValueError('read exceeds 512 MiB; use tiled reads')
        return self._reader[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]

    def sample(self, indices_zyx):
        """Sample a Cartesian grid without materializing its dense native envelope."""
        indices = [np.asarray(axis) for axis in indices_zyx]
        if len(indices) != 3 or any(a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu'
                or np.any(a < 0) or np.any(a >= n) for a,n in zip(indices,self.shape)):
            raise ValueError('sample indices must be nonempty integer ZYX axes inside source')
        shape = tuple(len(a) for a in indices)
        if math.prod(shape)*self.dtype.itemsize > 512 << 20:
            raise ValueError('sample exceeds 512 MiB; use tiled sampling')
        groups = [[np.flatnonzero(a//c == k) for k in np.unique(a//c)]
                  for a,c in zip(indices,self.chunks)]
        result = np.empty(shape,dtype=self.dtype)
        for positions in itertools.product(*groups):
            values = [a[p] for a,p in zip(indices,positions)]
            lo = np.array([v.min() for v in values]); hi = np.array([v.max()+1 for v in values])
            tile = self.read(lo,hi)
            result[np.ix_(*positions)] = tile[np.ix_(*(v-a for v,a in zip(values,lo)))]
        return result

    def stats(self):
        return dict(self._stats, transport=self._transport, cache_bytes=self.db.execute('SELECT coalesce(sum(size),0) FROM cache').fetchone()[0]+cache_size(self.cache/'vc-reader'),
                    decoded_cache_bytes=self._decoded_bytes,
                    backend='vc_reader-zarr-v2-checked-v1')

    def close(self):
        if self._reader is not None:self._reader.close();self._reader=None
        self._decoded.clear();self._decoded_bytes=0
        if self.db is not None:self.db.close();self.db=None
        if self._ledger is not None:self._ledger.close();self._ledger=None
        for transport in self._transports.values():transport.close()
        self._transports.clear()
