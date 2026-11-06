"""Native transport contracts; run with the isolated gpu-1 vc_reader runtime."""

import hashlib
import http.server
import threading
from functools import partial

import pytest

from test_io import served


@pytest.fixture(autouse=True)
def native_runtime():
    pytest.importorskip('vc_reader._reader')


@pytest.fixture
def endpoint(tmp_path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path=='/etag':
                raw=(tmp_path/'etag').read_bytes();etag='"'+hashlib.sha256(raw).hexdigest()+'"'
                if self.headers.get('If-None-Match')==etag:
                    self.send_response(304);self.end_headers();return
                self.send_response(200);self.send_header('Content-Length',str(len(raw)))
                self.send_header('ETag',etag);self.end_headers();self.wfile.write(raw);return
            if self.path in ('/oversized', '/unsized'):
                self.send_response(200)
                if self.path == '/oversized':
                    self.send_header('Content-Length', str((256 << 20) + 1))
                self.end_headers()
                try:
                    self.wfile.write(b'12345678')
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            super().do_GET()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield tmp_path, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_epoch_rejects_same_second_same_length_mutation(endpoint):
    """Last-Modified alone cannot certify the encoded-byte resume contract."""
    import os
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    source = root / 'chunk'
    source.write_bytes(b'original')
    stamp = source.stat().st_mtime_ns
    transport = EncodedTransport(root / 'cache', 1 << 20, offline=False)
    try:
        assert transport.read(url + '/chunk') == b'original'
        source.write_bytes(b'mutated!')
        os.utime(source, ns=(stamp, stamp))
        transport.new_epoch()
        assert transport.read(url + '/chunk') == b'mutated!'
    finally:
        transport.close()


def test_native_missing_cannot_revive_previous_payload_offline(endpoint):
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    source = root / 'chunk'
    source.write_bytes(b'original')
    transport = EncodedTransport(root / 'cache', 1 << 20, offline=False)
    try:
        assert transport.read(url + '/chunk') == b'original'
        source.unlink()
        transport.new_epoch()
        assert transport.read(url + '/chunk') is None
        assert transport.size == 0
    finally:
        transport.close()
    offline = EncodedTransport(root / 'cache', 1 << 20, offline=True)
    try:
        assert offline.read(url + '/chunk') is None
        with pytest.raises(FileNotFoundError, match='cache miss'):
            offline.read(url + '/never-seen')
    finally:
        offline.close()


def test_native_cache_corruption_is_rejected(endpoint):
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    (root / 'chunk').write_bytes(b'original')
    transport = EncodedTransport(root / 'cache', 1 << 20, offline=False)
    try:
        assert transport.read(url + '/chunk') == b'original'
        path = root / 'cache/encoded' / (hashlib.sha256((url + '/chunk').encode()).hexdigest() + '.blob')
        path.write_bytes(b'corrupt!')
        with pytest.raises(ValueError, match='corrupt'):
            transport.read(url + '/chunk')
    finally:
        transport.close()


def test_native_warm_hits_do_not_commit_receipts_again(endpoint):
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    (root / 'chunk').write_bytes(b'original')
    transport = EncodedTransport(root / 'cache', 1 << 20)
    try:
        assert transport.read(url + '/chunk') == b'original'
        statements = []
        transport.db.set_trace_callback(statements.append)
        for _ in range(3):
            assert transport.read(url + '/chunk') == b'original'
        assert 'COMMIT' not in statements
        assert transport.bytes_read == 8
    finally:
        transport.close()


def test_shared_volume_uses_native_and_preserves_coordinates(served):
    import numpy as np
    from winding_field.io import Volume

    root, url, expected = served
    volume = Volume(url, cache=root / 'cache', decoded_cache_bytes=4096)
    try:
        assert volume.stats()['transport'] == 'vc_reader.DiskCache'
        volume.bind_ledger(root / 'source.sqlite')
        indices = [np.array([0, 0, 3]), np.array([1, 3]), np.array([0, 2, 5, 5])]
        np.testing.assert_array_equal(volume.sample(indices), expected[np.ix_(*indices)])
        assert volume.stats()['cache_bytes'] == expected.nbytes
        assert volume.spacing_xyz == [2., 3., 4.]
        assert volume.origin_xyz == [10., 20., 30.]
        downloaded = volume.stats()['downloaded_bytes']
        np.testing.assert_array_equal(volume.sample(indices), expected[np.ix_(*indices)])
        assert volume.stats()['downloaded_bytes'] == downloaded
        (root / 'ct.zarr/0/0.0.0').write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match='source chunk changed'):
            volume.validate_used()
    finally:
        volume.close()


def test_decoded_cache_snapshot_recheck_uses_bounded_transport(served,monkeypatch):
    from contextlib import closing
    from pathlib import Path
    import numpy as np
    from winding_field.io import Volume
    root,url,expected=served
    read=Path.read_bytes
    def metadata_only(path):
        assert path.suffix!='.blob','payload recheck bypassed bounded native transport'
        return read(path)
    with closing(Volume(url,cache=root/'cache',decoded_cache_bytes=4096)) as volume:
        monkeypatch.setattr(Path,'read_bytes',metadata_only)
        np.testing.assert_array_equal(volume.decoded_chunk('0/0.0.0'),expected[:2,:2,:3])


def test_verified_small_native_cache_is_reused_offline(served):
    import numpy as np
    from winding_field.io import Volume

    root, url, expected = served
    # A small native cache remains reusable by a subsequent offline session.
    old = Volume(url, cache=root / 'cache', cache_bytes=128)
    old.read([0, 0, 0], [2, 2, 3])
    old.close()
    current = Volume(url, cache=root / 'cache', offline=True)
    try:
        assert current.stats()['transport'] == 'vc_reader.DiskCache'
        np.testing.assert_array_equal(current.read([0, 0, 0], [2, 2, 3]), expected[:2, :2, :3])
        assert current.stats()['downloaded_bytes'] == 0
        assert current.stats()['cache_bytes'] <= current.limit
    finally:
        current.close()


def test_native_reserve_refuses_download_without_recording_air(endpoint, monkeypatch):
    from collections import namedtuple
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    (root / 'chunk').write_bytes(b'original')
    transport = EncodedTransport(root / 'cache', 1 << 20, reserve=100)
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr('winding_field.ct_reader.shutil.disk_usage', lambda _: usage(1000, 950, 50))
    try:
        with pytest.raises(OSError, match='reserve'):
            transport.read(url + '/chunk')
        assert transport.bytes_read == 0
        assert transport.db.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0
    finally:
        transport.close()


def test_native_budget_eviction_is_an_offline_miss(endpoint):
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    (root / 'a').write_bytes(b'12345678')
    (root / 'b').write_bytes(b'abcdefgh')
    transport = EncodedTransport(root / 'cache', 8)
    try:
        assert transport.read(url + '/a') == b'12345678'
        assert transport.read(url + '/b') == b'abcdefgh'
        assert transport.size <= 8
    finally:
        transport.close()
    offline = EncodedTransport(root / 'cache', 8, offline=True)
    try:
        with pytest.raises(FileNotFoundError, match='cache miss'):
            offline.read(url + '/a')
        assert offline.read(url + '/b') == b'abcdefgh'
    finally:
        offline.close()


def test_decoded_hit_checks_ledger_even_when_stat_signature_collides(served, monkeypatch):
    from winding_field.io import Volume

    root, _, _ = served
    volume = Volume(root / 'ct.zarr', cache=root / 'cache', decoded_cache_bytes=1024)
    try:
        volume.bind_ledger(root / 'source.sqlite')
        volume.decoded_chunk('0/0.0.0')
        signature = volume._encoded_signature('0/0.0.0')
        monkeypatch.setattr(volume, '_encoded_signature', lambda key: signature)
        (root / 'ct.zarr/0/0.0.0').write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match='source chunk changed'):
            volume.decoded_chunk('0/0.0.0')
    finally:
        volume.close()


def test_full_legacy_cache_admits_native_miss(served):
    from winding_field.io import Volume

    root, url, _ = served
    volume = Volume(url, cache=root / 'cache', cache_bytes=256 << 20)
    try:
        key = 'f' * 64
        with (root / 'cache' / key).open('wb') as handle:
            handle.truncate(volume.limit)
        volume.db.execute('INSERT INTO cache VALUES (?,?,?,?,?)', (key,volume.limit,0,None,'unconsumed'))
        volume.db.commit()
        assert volume._chunk_bytes('0/0.0.0') is not None
        assert volume.stats()['cache_bytes'] <= volume.limit
    finally:
        volume.close()


def test_revalidation_of_object_larger_than_half_budget(endpoint):
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    (root / 'chunk').write_bytes(b'12345678')
    transport = EncodedTransport(root / 'cache', 12)
    try:
        assert transport.read(url + '/chunk') == b'12345678'
        transport.new_epoch()
        assert transport.read(url + '/chunk') == b'12345678'
        assert transport.size <= 12
    finally:
        transport.close()


def test_volume_requires_native_runtime_before_source_access(served, monkeypatch):
    import builtins
    from winding_field.io import Volume

    root, url, _ = served
    native = Volume(url, cache=root / 'cache')
    native._chunk_bytes('0/0.0.0')
    native.close()
    original = builtins.__import__

    def without_native(name, *args, **kwargs):
        if name.startswith('vc_reader'):
            raise ModuleNotFoundError('vc_reader is not installed', name='vc_reader')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_native)
    with pytest.raises(RuntimeError, match='vc_reader|required'):
        Volume(root / 'ct.zarr', cache=root / 'cache', cache_bytes=4)


@pytest.mark.parametrize('budget', [0, 4, 128])
def test_all_cache_budgets_use_native_volume_reads(served, budget):
    import numpy as np
    from winding_field.io import Volume

    root, url, expected = served
    volume = Volume(url, cache=root / 'cache', cache_bytes=budget)
    try:
        assert volume.stats()['backend'] == 'vc_reader-zarr-v2-checked-v1'
        np.testing.assert_array_equal(volume.read([1, 1, 2], [4, 4, 6]), expected[1:4, 1:4, 2:6])
        assert volume.stats()['cache_bytes'] <= budget
    finally:
        volume.close()


def test_volume_identity_binds_loaded_native_implementation(served):
    from vc_reader import _reader
    from winding_field.io import Volume
    from winding_field.util import file_sha
    root,url,_=served
    volume=Volume(url,cache=root/'cache')
    try:
        assert volume.identity['native_runtime']['extension_sha256']==file_sha(_reader.__file__)
        assert volume.identity['native_runtime']['libraries_sha256']
    finally:volume.close()


def test_native_refuses_oversized_reservation_before_payload(endpoint):
    from winding_field.ct_reader import EncodedTransport

    root, url = endpoint
    transport = EncodedTransport(root / 'cache', 1 << 30)
    try:
        with pytest.raises(RuntimeError, match='read byte limit'):
            transport.read(url + '/oversized')
        assert transport.size == 0
        assert transport.bytes_read == 0
        assert transport.db.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0
    finally:
        transport.close()


def test_native_unsized_response_respects_budget(endpoint):
    from winding_field.ct_reader import EncodedTransport
    root, url = endpoint
    transport = EncodedTransport(root / 'cache', 8)
    try:
        assert transport.read(url + '/unsized') == b'12345678'
        assert transport.size <= 8
    finally:
        transport.close()
    limited=EncodedTransport(root/'limited',4)
    try:
        assert limited.read(url+'/unsized')==b'12345678'
        assert limited.size==0
        assert limited.db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0]==1
    finally:limited.close()


def test_native_cached_source_uses_owned_strong_validator(endpoint):
    from winding_field.ct_reader import EncodedTransport
    root,url=endpoint;(root/'etag').write_bytes(b'original')
    transport=EncodedTransport(root/'cache',1<<20)
    try:
        assert transport.read(url+'/etag')==b'original'
        transport.new_epoch()
        assert transport.read(url+'/etag')==b'original'
        received=transport.bytes_read
        transport.new_epoch()
        assert transport.read(url+'/etag')==b'original'
        assert transport.bytes_read==received
        (root/'etag').write_bytes(b'mutated!');transport.new_epoch()
        assert transport.read(url+'/etag')==b'mutated!'
    finally:transport.close()


def test_native_eviction_cannot_reuse_validator_for_different_bytes(endpoint):
    from winding_field.ct_reader import EncodedTransport
    root,url=endpoint;(root/'etag').write_bytes(b'original');(root/'other').write_bytes(b'eviction')
    transport=EncodedTransport(root/'cache',8)
    try:
        assert transport.read(url+'/etag')==b'original'
        transport.new_epoch();assert transport.read(url+'/etag')==b'original'
        assert transport.read(url+'/other')==b'eviction'
        assert not transport.path(url+'/etag').exists()
        (root/'etag').write_bytes(b'mutated!')
        assert transport.read(url+'/etag')==b'mutated!'
        (root/'etag').write_bytes(b'original');transport.new_epoch()
        assert transport.read(url+'/etag')==b'original'
    finally:transport.close()


def test_repeated_missing_releases_receipt_write_lock(endpoint):
    import sqlite3
    from winding_field.ct_reader import EncodedTransport
    root,url=endpoint;transport=EncodedTransport(root/'cache',32)
    try:
        assert transport.read(url+'/absent') is None
        transport.new_epoch();assert transport.read(url+'/absent') is None
        with sqlite3.connect(root/'cache/receipts.sqlite',timeout=.1) as other:
            other.execute('BEGIN IMMEDIATE')
    finally:transport.close()


def test_old_encoded_cache_remains_accounted_and_evictable(endpoint):
    import sqlite3
    from winding_field.ct_reader import EncodedTransport
    root,url=endpoint;cache=root/'cache';encoded=cache/'encoded';encoded.mkdir(parents=True)
    old=encoded/(hashlib.sha256((url+'/old').encode()).hexdigest()+'.blob');old.write_bytes(b'original')
    with sqlite3.connect(cache/'.vc_reader_cache.sqlite3') as db:
        db.executescript('CREATE TABLE budget(id INTEGER PRIMARY KEY,max_bytes INTEGER,total INTEGER,clock INTEGER);'
            'INSERT INTO budget VALUES(1,8,8,1);CREATE TABLE chunks(path TEXT PRIMARY KEY,size INTEGER,age INTEGER);'
            'CREATE TABLE file_headers(path TEXT PRIMARY KEY,etag TEXT,modified TEXT);PRAGMA user_version=1;')
        db.execute('INSERT INTO chunks VALUES(?,8,1)',(str(old),))
        db.execute('INSERT INTO file_headers VALUES(?,?,?)',(str(old),'"old"',''))
    (root/'new').write_bytes(b'newbytes')
    transport=EncodedTransport(cache,8)
    try:
        assert transport.size==8
        assert transport.read(url+'/new')==b'newbytes'
        assert not old.exists()
        assert sum(p.stat().st_size for p in encoded.glob('*.blob'))<=8
    finally:transport.close()


def test_empty_download_after_eviction_invalidates_old_identity(endpoint):
    from winding_field.ct_reader import EncodedTransport
    root,url=endpoint;(root/'etag').write_bytes(b'original');(root/'other').write_bytes(b'eviction')
    transport=EncodedTransport(root/'cache',8)
    try:
        assert transport.read(url+'/etag')==b'original'
        transport.new_epoch();assert transport.read(url+'/etag')==b'original'
        transport.read(url+'/other');(root/'etag').write_bytes(b'')
        assert transport.read(url+'/etag')==b''
        assert transport.db.execute('SELECT etag FROM validators WHERE url=?',(url+'/etag',)).fetchone() is None
    finally:transport.close()


def test_native_index_binding_and_rollback(endpoint):
    from winding_field.ct_reader import EncodedTransport
    from winding_field.native_cache_index import cache_index

    root, _ = endpoint
    transport = EncodedTransport(root / 'cache', 1 << 20)
    index = root / 'cache/.vc_reader_cache.sqlite3'
    try:
        with cache_index(index) as db:
            db.execute('CREATE TABLE binding_test (value TEXT,n INTEGER,empty TEXT)')
            db.execute('INSERT INTO binding_test VALUES (?,?,?)', ("a'b?", 2**40, None))
        with pytest.raises(RuntimeError, match='interrupt'):
            with cache_index(index) as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('DELETE FROM binding_test')
                raise RuntimeError('interrupt')
        with cache_index(index) as db:
            assert db.execute('SELECT * FROM binding_test').fetchall() == [("a'b?", 2**40, None)]
    finally:
        transport.close()
