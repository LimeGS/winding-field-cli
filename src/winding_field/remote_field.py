"""Read-only, hash-pinned remote winding-field Zarr v2 exports."""
from collections import OrderedDict
from copy import deepcopy
from http.client import HTTPException, HTTPSConnection
import hashlib
import ipaddress
import itertools
import json
import math
import os
import re
import socket
from urllib.parse import unquote, urljoin, urlsplit

import numpy as np

MAX_OBJECT = 256 << 20
MAX_METADATA = 1 << 20
MAX_MANIFEST = 32 << 20


def _url(url, *, redirect=False):
    try:
        p = urlsplit(url)
        path = p.path
        if redirect:
            for _ in range(4):
                path = unquote(path)
        if (p.scheme != 'https' or not p.hostname or p.username or p.password
                or p.port not in (None, 443) or p.fragment or (p.query and not redirect)
                or '%' in path or '\\' in url or '\\' in path
                or any(ord(c) <= 32 or ord(c) == 127 for c in url + path)
                or any(s in ('.', '..') for s in path.split('/'))):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError('unsafe remote field URL') from None
    return p


def fetch(url, *, limit):
    """Bounded HTTPS with DNS-bound connections and origin-scoped HF credentials."""
    original = _url(url)
    for hop in range(6):
        p = _url(url, redirect=hop > 0)
        try:
            addresses = socket.getaddrinfo(p.hostname, 443, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                raise ValueError('unsafe remote field address')
            connection = HTTPSConnection(p.hostname, timeout=60)

            def connect(address, timeout, source_address=None):
                for family, kind, protocol, _, destination in addresses:
                    sock = socket.socket(family, kind, protocol)
                    try:
                        sock.settimeout(timeout)
                        if source_address:
                            sock.bind(source_address)
                        sock.connect(destination)
                        return sock
                    except OSError:
                        sock.close()
                raise OSError('remote field connection failed')

            # TLS still checks the original hostname; no second DNS resolution.
            connection._create_connection = connect
            headers = {'Accept-Encoding': 'identity'}
            token = os.environ.get('HF_TOKEN')
            if token and p.hostname == original.hostname and p.hostname == 'huggingface.co':
                if not re.fullmatch('[A-Za-z0-9_.-]+', token):
                    raise ValueError('invalid HF_TOKEN format')
                headers['Authorization'] = 'Bearer ' + token
            response = None
            try:
                target = p.path or '/'
                if p.query:
                    target += '?' + p.query
                connection.request('GET', target, headers=headers)
                response = connection.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location:
                        raise ValueError('remote field redirect has no destination')
                    url = urljoin(url, location)
                    continue
                if response.status == 404:
                    raise FileNotFoundError('expected remote field object missing')
                if response.status != 200:
                    raise ValueError(f'remote field HTTP status {response.status}')
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise ValueError('unexpected remote field content encoding')
                size = int(response.headers.get('Content-Length', 0))
                if size < 0 or size > limit:
                    raise ValueError('remote field object exceeds read budget')
                raw = response.read(limit + 1)
                if len(raw) > limit or (size and len(raw) != size):
                    raise ValueError('remote field object size differs or exceeds read budget')
                return raw
            finally:
                if response is not None:
                    response.close()
                connection.close()
        except FileNotFoundError:
            raise
        except (OSError, HTTPException, UnicodeError):
            # Upstream exception strings can contain signed redirect URLs.
            raise OSError('remote field HTTPS request failed') from None
    raise ValueError('remote field redirect limit exceeded')


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('remote field requires lowercase SHA256')
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate remote field JSON key')
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError('nonfinite remote field JSON value')
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('remote field JSON object required')
    return value


def _triple(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(n) is not int or n < 1 for n in value)):
        raise ValueError('positive integer remote ZYX dimensions required')
    return tuple(value)


def _record(value, limit):
    if (not isinstance(value, dict) or type(value.get('bytes')) is not int
            or not 0 < value['bytes'] <= limit):
        raise ValueError('invalid remote field object byte budget')
    _sha(value.get('sha256'))
    return dict(bytes=value['bytes'], sha256=value['sha256'])


class RemoteField:
    def __init__(self, field, geometry, *, cache_bytes=0):
        if (field.keys() - {'kind', 'url', 'sha256', 'complete_sha256', 'support_policy'}
                or field.get('kind') != 'zarr-v2'
                or field.get('support_policy') not in ('stored', 'observed')
                or type(cache_bytes) is not int or cache_bytes < 0):
            raise ValueError('invalid remote field descriptor or cache budget')
        p = _url(field.get('url'))
        if not p.path.endswith('/field.json'):
            raise ValueError('remote field URL must name field.json')
        self.base = field['url'].rsplit('/', 1)[0] + '/'
        self._pins = {'field.json': _sha(field.get('sha256')),
                      'COMPLETE.json': _sha(field.get('complete_sha256'))}
        self._cache = OrderedDict()
        self._cache_bytes, self._budget = 0, cache_bytes
        self._used = set()
        self._closed = False
        self._metadata = {}
        metadata = self._metadata_json('field.json')
        self.shape = _triple(geometry['shape_zyx'])
        keys = ('frame', 'voxel_um', 'origin_xyz', 'spacing_xyz', 'shape_zyx', 'sheet_step')
        if (metadata.get('schema') != 'winding-field-zarr-v1'
                or any(metadata.get('geometry', {}).get(k) != geometry[k] for k in keys)
                or metadata.get('calibration', {}).get('sheet_step') != geometry['sheet_step']
                or metadata.get('completion') != 'COMPLETE.json'
                or metadata.get('receipts') != 'receipts/{z}/{y}/{x}.json'):
            raise ValueError('remote field schema, geometry or calibration differs')
        self.chunks = _triple(metadata.get('chunk_shape_zyx'))
        if math.prod(self.chunks) * 4 > MAX_OBJECT:
            raise ValueError('remote field chunk exceeds decode budget')
        self._arrays = metadata.get('arrays')
        if self._arrays != {n: 'field.zarr/' + n for n in ('phi', 'valid', 'observed')}:
            raise ValueError('unsupported remote field array paths')
        self._names = ['phi', 'valid']
        if field['support_policy'] == 'observed':
            if metadata.get('observation_support', {}).get('available') is not True:
                raise ValueError('remote observed support unavailable')
            self._names.append('observed')
        complete = self._metadata_json('COMPLETE.json')
        binding = field['sha256']
        if (complete.get('schema') != 'winding-field-zarr-complete-v1'
                or complete.get('status') != 'DATA_UPLOAD_COMPLETE'
                or complete.get('binding') != binding):
            raise ValueError('invalid remote field completion')
        manifest_pin = _record(complete.get('manifest'), MAX_MANIFEST)
        self._pins['manifest.json'] = manifest_pin['sha256']
        manifest = self._metadata_json('manifest.json', limit=manifest_pin['bytes'], size=manifest_pin['bytes'])
        if (manifest.get('schema') != 'winding-field-zarr-manifest-v1'
                or manifest.get('binding') != binding):
            raise ValueError('remote manifest binding differs')
        counts = tuple((n + c - 1) // c for n, c in zip(self.shape, self.chunks))
        count = math.prod(counts)
        entries = manifest.get('chunks')
        if (not isinstance(entries, list) or len(entries) != count
                or complete.get('summary', {}).get('chunks') != count):
            raise ValueError('incomplete remote field inventory')
        self._files = {}
        seen = set()
        for entry in entries:
            key = entry.get('key')
            if (not isinstance(key, list) or len(key) != 3
                    or any(type(k) is not int or not 0 <= k < n for k, n in zip(key, counts))
                    or tuple(key) in seen):
                raise ValueError('invalid or duplicate remote chunk key')
            seen.add(tuple(key))
            suffix = '/'.join(map(str, key))
            expected = {f'field.zarr/{name}/{suffix}' for name in self._arrays}
            expected.add('receipts/' + suffix + '.json')
            files = entry.get('files')
            if not isinstance(files, dict) or set(files) != expected:
                raise ValueError('remote chunk inventory paths differ')
            for path, record in files.items():
                self._files[path] = _record(record, MAX_METADATA if path.startswith('receipts/') else
                    min(MAX_OBJECT, math.prod(self.chunks) * (4 if '/phi/' in path else 1) * 2 + 65536))
        self._zarray = {}
        from vc_reader.reader import ArrayMeta
        for name in self._arrays:
            path = self._arrays[name] + '/.zarray'
            meta = self._metadata_json(path)
            if (meta.get('zarr_format') != 2 or _triple(meta.get('shape')) != self.shape
                    or _triple(meta.get('chunks')) != self.chunks
                    or meta.get('dtype') != ('<f4' if name == 'phi' else '|u1')
                    or meta.get('order') not in ('C', 'F')
                    or meta.get('dimension_separator') != '/'
                    or meta.get('filters') not in (None, [])):
                raise ValueError('unsupported remote Zarr array layout')
            ArrayMeta.from_zarray(meta).codec_id
            self._zarray[name] = meta
        from .ct_reader import runtime_identity
        from .util import file_sha
        self._identity = dict(field=deepcopy(field), geometry=deepcopy(geometry),
            source_identity=dict(kind='remote-field-zarr-v2-v1', metadata_sha256=dict(self._pins),
                                 native_runtime=runtime_identity(), implementation_sha256=file_sha(__file__)))

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _metadata_json(self, path, *, limit=MAX_METADATA, size=None):
        raw = fetch(self.base + path, limit=limit)
        if len(raw) > limit or (size is not None and len(raw) != size):
            raise ValueError('remote field metadata size differs')
        digest = hashlib.sha256(raw).hexdigest()
        if path in self._pins and digest != self._pins[path]:
            raise ValueError('remote field metadata hash differs: ' + path)
        self._pins[path] = digest
        self._metadata[path] = (limit, size)
        return _json(raw)

    def _raw(self, path):
        pin = self._files[path]
        raw = fetch(self.base + path, limit=pin['bytes'])
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('remote field payload hash/size differs: ' + path)
        self._used.add(path)
        return raw

    def _keys(self, lo, hi):
        if self._closed:
            raise RuntimeError('remote field reader closed')
        if (len(lo) != 3 or len(hi) != 3
                or any(type(a) is not int or type(b) is not int or not 0 <= a < b <= n
                       for a, b, n in zip(lo, hi, self.shape))):
            raise ValueError('remote field read outside grid')
        return itertools.product(*(range(a // c, (b - 1) // c + 1)
                                   for a, b, c in zip(lo, hi, self.chunks)))

    def _receipt(self, key):
        suffix = '/'.join(map(str, key))
        receipt = _json(self._raw('receipts/' + suffix + '.json'))
        shape = [min(c, n - k*c) for k, c, n in zip(key, self.chunks, self.shape)]
        arrays = receipt.get('arrays')
        if (receipt.get('schema') != 'winding-field-zarr-chunk-v1'
                or receipt.get('binding') != self._pins['field.json']
                or receipt.get('key') != list(key) or receipt.get('shape_zyx') != shape
                or not isinstance(arrays, dict) or set(arrays) != set(self._arrays)
                or any(_record(arrays[name], MAX_OBJECT) != self._files[f'field.zarr/{name}/{suffix}']
                       for name in self._arrays)):
            raise ValueError('remote chunk receipt differs')

    def _load(self, name, key):
        from vc_reader.reader import decode_chunk
        suffix = '/'.join(map(str, key))
        path = self._arrays[name] + '/' + suffix
        raw = self._raw(path)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        meta = self._zarray[name]
        decoded = decode_chunk(raw, meta, max_decode_bytes=MAX_OBJECT)
        if len(decoded) != math.prod(self.chunks) * np.dtype(meta['dtype']).itemsize:
            raise ValueError('remote decoded chunk size differs')
        array = np.frombuffer(decoded, dtype=meta['dtype']).reshape(self.chunks)
        if name != 'phi' and np.any(array > 1):
            raise ValueError('remote support mask is not binary')
        array.flags.writeable = False
        charge = array.nbytes + 512
        if charge <= self._budget:
            while self._cache_bytes + charge > self._budget:
                _, retired = self._cache.popitem(last=False)
                self._cache_bytes -= retired.nbytes + 512
            self._cache[path] = array
            self._cache_bytes += charge
        return array

    def read(self, lo, hi):
        keys = self._keys(lo, hi)
        shape = tuple(b-a for a, b in zip(lo, hi))
        if math.prod(shape) * 9 > MAX_OBJECT:
            raise ValueError('remote field crop exceeds 256 MiB; use tiled reads')
        values, valid = np.empty(shape, np.float64), np.ones(shape, bool)
        for key in keys:
            self._receipt(key)
            origin = tuple(k*c for k, c in zip(key, self.chunks))
            start = tuple(max(a, o) for a, o in zip(lo, origin))
            end = tuple(min(b, o+c) for b, o, c in zip(hi, origin, self.chunks))
            source = tuple(slice(a-o, b-o) for a, b, o in zip(start, end, origin))
            target = tuple(slice(a-o, b-o) for a, b, o in zip(start, end, lo))
            for name in self._names:
                array = self._load(name, key)
                if name == 'phi':
                    values[target] = array[source]
                else:
                    valid[target] &= array[source] != 0
        valid &= np.isfinite(values)
        return values, valid

    def validate_region(self, lo, hi):
        for key in self._keys(lo, hi):
            self._receipt(key)
            suffix = '/'.join(map(str, key))
            for name in self._names:
                self._raw(f'field.zarr/{name}/{suffix}')

    def validate(self, *, hashes=False):
        if self._closed:
            raise RuntimeError('remote field reader closed')
        for path, (limit, size) in list(self._metadata.items()):
            self._metadata_json(path, limit=limit, size=size)
        for path in sorted(self._files if hashes else self._used):
            self._raw(path)

    def close(self):
        self._cache.clear()
        self._cache_bytes = 0
        self._closed = True
