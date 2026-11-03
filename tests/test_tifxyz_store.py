"""Storage contract tests; coordinates encode absolute ownership independently."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import sqlite3
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from winding_field.tifxyz_contract import Blocked, Paused


def api():
    return importlib.import_module('winding_field.tifxyz_store')


def identity(shape=(259, 259)):
    return dict(source='source-sha', settings='settings-sha', code='code-sha',
                frame=frame(), grid=dict(shape_yx=list(shape), pitch_um=17.28,
                uv_min_um=[2., 3.], level=-4., tile_pixels=256))


def frame():
    return dict(id='selected-L1', origin_xyz_um=[11., 23., 37.], voxel_um=8.64,
                shape_zyx=[1000, 1000, 1000], source_identity_sha256='a'*64)


def budget(root, **changes):
    return SimpleNamespace(**(dict(scope_root=root, scratch_bytes=128 << 20,
        database_bytes=4 << 20, reserve_disk_bytes=0) | changes))


def tile(tid=(0, 0), shape=(259, 259), value=None):
    lo = [256*i for i in tid]
    hi = [min(a+256, n) for a, n in zip(lo, shape)]
    low = [max(0, a-1) for a in lo]
    high = [min(n, b+1) for n, b in zip(shape, hi)]
    y, x = np.mgrid[low[0]:high[0], low[1]:high[1]]
    xyz = np.stack([x, y, x+y], axis=-1).astype(np.float32)
    if value is not None:
        xyz[:] = value
    return dict(tile_id=list(tid), owned_box_yx=[lo, hi], box_yx=[low, high],
        xyz=xyz, mask=np.ones(y.shape, bool), root_error_um=np.full(y.shape, .125))


def open_store(root, ident=None, **kw):
    return api().TileStore(root/'blocks', ident or identity(), budget(root), 'projected', **kw)


def test_bounded_executemany_prepares_once_without_losing_guards(tmp_path):
    with open_store(tmp_path) as store:
        with store.transaction() as db:db.execute('CREATE TABLE batch_probe (x INTEGER)')
        preparations=[]
        def authorize(action,a,b,database,trigger):
            if action==sqlite3.SQLITE_INSERT and a=='batch_probe':preparations.append(a)
            return store._authorize(action,a,b,database,trigger)
        store.db.set_authorizer(authorize)
        with store.transaction() as db:db.executemany('INSERT INTO batch_probe VALUES (?)',[(i,) for i in range(128)])
        assert store.db.execute('SELECT count(*) FROM batch_probe').fetchone()==(128,)
        assert len(preparations)<=4,len(preparations)


def test_commit_resume_is_immutable_and_index_ordered(tmp_path):
    expected = tile((1, 1))
    with open_store(tmp_path) as store:
        receipt = store.commit(expected)
        assert store.commit(expected) == receipt
        store.commit(tile((0, 0)))
        assert list(store.iter_ids()) == [(0, 0), (1, 1)]
        plans = store.db.execute('EXPLAIN QUERY PLAN SELECT ty,tx FROM tiles ORDER BY ty,tx')
        assert all('TEMP B-TREE' not in row[-1] for row in plans)
        altered = tile((1, 1), value=99)
        with pytest.raises(ValueError, match='immutable|different'):
            store.commit(altered)
    with open_store(tmp_path, resume=True) as store:
        actual = store.read((1, 1))
        for key in ('xyz', 'mask', 'root_error_um'):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual['owned_box_yx'] == [[256, 256], [259, 259]]
        assert (store.root/receipt['path']).stat().st_size == receipt['bytes']


def test_region_uses_owners_and_never_fills_missing_pixels(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile(value=10))
        with pytest.raises(ValueError, match='missing'):
            store.read_region((254, 254), (259, 259))
        store.commit(tile((0, 1), value=20))
        store.commit(tile((1, 0), value=30))
        store.commit(tile((1, 1), value=40))
        region = store.read_region((254, 254), (259, 259))
        np.testing.assert_array_equal(region['xyz'][..., 0],
            [[10,10,20,20,20], [10,10,20,20,20], [30,30,40,40,40],
             [30,30,40,40,40], [30,30,40,40,40]])
        with pytest.raises(ValueError):
            store.read_region((-1, 0), (2, 2))
    with open_store(tmp_path/'large', identity((600, 600))) as store:
        with pytest.raises(ValueError, match='260|region'):
            store.read_region((0, 0), (261, 2))


@pytest.mark.parametrize('key', ['source', 'settings', 'code', 'frame', 'grid'])
def test_resume_rejects_changed_identity(tmp_path, key):
    original = identity()
    with open_store(tmp_path, original):
        pass
    changed = deepcopy(original)
    if key == 'frame': changed[key]['voxel_um'] = 1
    elif key == 'grid': changed[key]['pitch_um'] = 1
    else: changed[key] = 'different'
    with pytest.raises(ValueError, match='identity'):
        open_store(tmp_path, changed, resume=True)


def test_foreign_symlink_and_second_writer_are_rejected(tmp_path):
    foreign = tmp_path/'foreign'
    foreign.mkdir()
    (foreign/'keep').write_text('original')
    with pytest.raises(ValueError, match='foreign|empty|owned'):
        api().TileStore(foreign, identity(), budget(tmp_path), 'projected')
    assert (foreign/'keep').read_text() == 'original'
    alias = tmp_path/'alias'
    alias.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        api().TileStore(alias, identity(), budget(tmp_path), 'projected')
    with open_store(tmp_path/'lock-test'):
        with pytest.raises((ValueError, BlockingIOError), match='lock|writer|temporarily'):
            open_store(tmp_path/'lock-test', resume=True)


@pytest.mark.parametrize('damage', ['bytes', 'missing', 'symlink', 'ledger'])
def test_corrupt_committed_data_fails_resume(tmp_path, damage):
    with open_store(tmp_path) as store:
        receipt = store.commit(tile())
        path = store.root/receipt['path']
    if damage == 'bytes':
        with path.open('r+b') as f:
            f.seek(-1, 2); f.write(b'X')
    elif damage == 'missing': path.unlink()
    elif damage == 'symlink':
        other = tmp_path/'original'
        path.rename(other); path.symlink_to(other)
    else:
        with sqlite3.connect(tmp_path/'blocks/store.sqlite') as db:
            db.execute("UPDATE tiles SET sha256=?", ('0'*64,))
    with pytest.raises(ValueError, match='corrupt|missing|checksum|symlink'):
        open_store(tmp_path, resume=True)


@pytest.mark.parametrize('stage', ['before_fsync', 'after_fsync', 'before_dir_fsync',
                                 'after_dir_fsync', 'before_commit', 'after_commit'])
def test_pause_at_durability_boundaries_then_resume(tmp_path, monkeypatch, stage):
    module = api()
    with open_store(tmp_path) as store:
        if 'fsync' in stage:
            real = module.os.fsync
            fired = False
            def interrupted(fd):
                nonlocal fired
                is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
                if not fired and ('dir' not in stage or is_directory):
                    fired = True
                    if stage.startswith('after'): real(fd)
                    raise Paused()
                return real(fd)
            monkeypatch.setattr(module.os, 'fsync', interrupted)
        else:
            real = store.transaction
            @contextmanager
            def interrupted():
                with real() as db:
                    yield db
                    if stage == 'before_commit': raise Paused()
                if stage == 'after_commit': raise Paused()
            monkeypatch.setattr(store, 'transaction', interrupted)
        with pytest.raises(Paused):
            store.commit(tile())
    monkeypatch.undo()
    with open_store(tmp_path, resume=True) as store:
        store.commit(tile())
        np.testing.assert_array_equal(store.read((0, 0))['xyz'], tile()['xyz'])
        assert list(store.iter_ids()) == [(0, 0)]


def test_orphan_payload_needs_fresh_identical_recomputation(tmp_path):
    with open_store(tmp_path) as store:
        receipt = store.commit(tile())
        with store.transaction() as db:
            db.execute('DELETE FROM tiles')
    with open_store(tmp_path, resume=True) as store:
        with pytest.raises(ValueError, match='missing'):
            store.read((0, 0))
        with pytest.raises(ValueError, match='orphan|different|immutable'):
            store.commit(tile(value=2))
        assert store.commit(tile()) == receipt


@pytest.mark.parametrize('bad', ['dtype', 'nan', 'mask', 'error', 'box'])
def test_invalid_blocks_rejected_before_publication(tmp_path, bad):
    item = tile()
    if bad == 'dtype': item['xyz'] = item['xyz'].astype(np.float64)
    if bad == 'nan': item['xyz'][0, 0, 0] = np.nan
    if bad == 'mask': item['mask'][0, 0] = False
    if bad == 'error': item['root_error_um'][0, 0] = -1
    if bad == 'box': item['owned_box_yx'][1][0] -= 1
    with open_store(tmp_path) as store:
        with pytest.raises(ValueError): store.commit(item)
        assert list(store.iter_ids()) == []


def test_scope_accounting_includes_sparse_prior_attempts_and_disk_reserve(tmp_path, monkeypatch):
    with open_store(tmp_path) as store:
        prior = tmp_path/'previous-attempt'
        prior.mkdir()
        with (prior/'retained').open('wb') as f:
            f.truncate(store.budget.scratch_bytes)
        with pytest.raises(Blocked, match='SCRATCH'):
            store.guard()
        (prior/'retained').unlink()
        stat = os.statvfs(tmp_path)
        fake = SimpleNamespace(f_bavail=1, f_frsize=stat.f_frsize, f_bsize=stat.f_bsize)
        monkeypatch.setattr(api().os, 'statvfs', lambda path: fake)
        store.budget.reserve_disk_bytes = stat.f_frsize
        with pytest.raises(Blocked, match='DISK'):
            store.guard(pending_bytes=1)


def test_sql_guard_quotas_and_rtree_share_the_ledger(tmp_path):
    with open_store(tmp_path) as store:
        for name, expected in [('journal_mode','delete'), ('synchronous',2),
                ('page_size',4096), ('cache_size',-16384), ('temp_store',2), ('automatic_index',0)]:
            assert store.db.execute('PRAGMA '+name).fetchone()[0] == expected
        assert store.db.execute('PRAGMA max_page_count').fetchone()[0] == 1024
        with pytest.raises(sqlite3.DatabaseError):
            store.db.execute('CREATE TABLE outside(x)')
        previous = resource.getrlimit(resource.RLIMIT_FSIZE)
        with store.transaction() as db:
            assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] <= store.journal_cap
            db.execute('CREATE VIRTUAL TABLE points USING rtree(id,x0,x1,y0,y1,z0,z1)')
            db.execute('INSERT INTO points VALUES(1,0,1,0,1,0,1)')
        assert resource.getrlimit(resource.RLIMIT_FSIZE) == previous
        assert store.db.execute('SELECT count(*) FROM points').fetchone() == (1,)
        store.commit(tile())
        store.budget.scratch_bytes = 1 << 20
        with pytest.raises(Blocked, match='SCRATCH'):
            with store.transaction(): pass
        assert store.read((0, 0))['xyz'][1, 1, 0] == 1


def test_failed_transaction_preserves_commits_and_blocks_further_writes(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
        with pytest.raises(RuntimeError, match='interrupted'):
            with store.transaction() as db:
                db.execute('CREATE TABLE uncommitted(x)')
                raise RuntimeError('interrupted')
        assert not store.db.execute("SELECT name FROM sqlite_master WHERE name='uncommitted'").fetchone()
        # An ordinary caller cancellation rolls back cleanly and may continue.
        store.commit(tile((1, 1)))
    with open_store(tmp_path, resume=True) as store:
        assert list(store.iter_ids()) == [(0, 0), (1, 1)]
    assert not list(tmp_path.rglob('READY*'))


def fill(store):
    for y in range((store.grid['shape_yx'][0]+255)//256):
        for x in range((store.grid['shape_yx'][1]+255)//256):
            store.commit(tile((y, x), tuple(store.grid['shape_yx'])))


def test_streamed_tiffs_tail_padding_scale_and_manifest(tmp_path, monkeypatch):
    import tifffile
    from winding_field.util import file_sha
    def forbidden(*args, **kwargs):
        raise AssertionError('full image read or mmap')
    monkeypatch.setattr(tifffile, 'imread', forbidden)
    monkeypatch.setattr(tifffile, 'memmap', forbidden)
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', forbidden)
    monkeypatch.setattr(tifffile.TiffFile, 'asarray', forbidden)
    original_write = tifffile.TiffWriter.write
    def iterator_only(self, data, **kwargs):
        assert not isinstance(data, np.ndarray)
        assert iter(data) is data
        return original_write(self, data, **kwargs)
    monkeypatch.setattr(tifffile.TiffWriter, 'write', iterator_only)
    with open_store(tmp_path) as store:
        fill(store)
        pending = tmp_path/'level.pending'
        manifest = api().write_segment(store, store.grid, frame(), pending)
        assert manifest['shape_yx'] == [259, 259]
        assert manifest['scale'] == [.5, .5]
        assert set(manifest['files']) == {'x.tif','y.tif','z.tif','meta.json'}
        assert api().write_segment(store, store.grid, frame(), pending) == manifest
        for name, receipt in manifest['files'].items():
            assert receipt == dict(bytes=(pending/name).stat().st_size, sha256=file_sha(pending/name))
        meta = json.loads((pending/'meta.json').read_text())
        assert meta['scale'] == [.5, .5] and meta['level'] == -4
        assert meta['output_frame'] == frame()
        for axis, name in enumerate(('x','y','z')):
            chunks = list(api().iter_tiff_tiles(pending/(name+'.tif'), (259,259), 259*259))
            assert [lo for lo, a in chunks] == [(0,0),(0,256),(256,0),(256,256)]
            assert [a.shape for lo, a in chunks] == [(256,256),(256,3),(3,256),(3,3)]
            for (y,x), a in chunks:
                yy,xx = np.mgrid[y:y+a.shape[0], x:x+a.shape[1]]
                np.testing.assert_array_equal(a, (xx,yy,xx+yy)[axis])
                assert a.dtype == np.float32
            with tifffile.TiffFile(pending/(name+'.tif')) as tif:
                page = tif.pages[0]
                assert page.is_tiled and page.tilewidth == page.tilelength == 256
                assert int(page.compression) in (8,32946)
                # Decode one tail using the independent library to inspect physical padding.
                offset, size = page.dataoffsets[-1], page.databytecounts[-1]
                tif.filehandle.seek(offset)
                raw = tif.filehandle.read(size)
                decoded = page.decode(raw, 3)[0].reshape(256,256)
                assert (decoded[3:] == -1).all() and (decoded[:,3:] == -1).all()
        assert not list(tmp_path.rglob('READY*'))


def test_standard_metadata_uuid_is_stable_across_resume_and_paths(tmp_path):
    ident = identity((3,3))
    first = tmp_path/'first'
    pending = first/'original.pending'
    with open_store(first, ident) as store:
        fill(store)
        manifest = api().write_segment(store, store.grid, frame(), pending)
    path = pending/'meta.json'
    original = path.read_bytes()
    before = path.stat()
    meta = json.loads(original)
    assert meta['format'] == 'tifxyz' and meta['type'] == 'seg'
    assert isinstance(meta['uuid'], str) and meta['uuid']
    with open_store(first, ident, resume=True) as store:
        assert api().write_segment(store, store.grid, frame(), pending) == manifest
    assert path.read_bytes() == original
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    for name, source, same in (('relocated', ident['source'], True), ('changed', 'another-source', False)):
        root = tmp_path/name
        with open_store(root, dict(ident, source=source)) as store:
            fill(store)
            other = root/'another-pending-name'
            api().write_segment(store, store.grid, frame(), other)
        other_meta = json.loads((other/'meta.json').read_text())
        assert (other_meta['uuid'] == meta['uuid']) is same
        if same:
            assert (other/'meta.json').read_bytes() == original
    assert not list(tmp_path.rglob('READY*'))


@pytest.mark.parametrize('damage', ['legacy', 'uuid'])
def test_standard_metadata_mismatch_is_rejected_without_overwrite(tmp_path, damage):
    with open_store(tmp_path, identity((3,3))) as store:
        fill(store)
        pending = tmp_path/'segment.pending'
        api().write_segment(store, store.grid, frame(), pending)
        path = pending/'meta.json'
        meta = json.loads(path.read_text())
        if damage == 'legacy':
            for key in ('format', 'type', 'uuid'): meta.pop(key, None)
        else:
            meta['uuid'] = 'foreign-uuid'
        path.write_text(json.dumps(meta))
        original, before = path.read_bytes(), path.stat()
        with pytest.raises(ValueError, match='immutable metadata mismatch'):
            api().write_segment(store, store.grid, frame(), pending)
        assert path.read_bytes() == original
        assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not list(tmp_path.rglob('READY*'))


@pytest.mark.parametrize('failure', ['close', 'meta'])
def test_failed_assembly_retains_valid_blocks_and_resumes_without_ready(tmp_path, monkeypatch, failure):
    import tifffile
    with open_store(tmp_path, identity((3, 3))) as store:
        fill(store)
        pending = tmp_path/'level.pending'
        if failure == 'close':
            real = tifffile.TiffWriter.close
            def fail(self):
                real(self)
                raise OSError('injected close')
            monkeypatch.setattr(tifffile.TiffWriter, 'close', fail)
        else:
            real = api()._write_bytes
            def fail(path, data):
                if path.name == 'meta.json': raise OSError('injected meta')
                return real(path, data)
            monkeypatch.setattr(api(), '_write_bytes', fail)
        with pytest.raises(OSError, match='injected'):
            api().write_segment(store, store.grid, frame(), pending)
        assert store.read((0,0))['xyz'][2,2,2] == 4
        assert not list(tmp_path.rglob('READY*'))
        monkeypatch.undo()
        api().write_segment(store, store.grid, frame(), pending)


def test_assembly_rejects_foreign_frame_grid_and_incomplete_ownership(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
        with pytest.raises(ValueError, match='missing'):
            api().write_segment(store, store.grid, frame(), tmp_path/'incomplete.pending')
        other = frame(); other['voxel_um'] = 1
        with pytest.raises(ValueError, match='frame|identity'):
            api().write_segment(store, store.grid, other, tmp_path/'frame.pending')
        grid = deepcopy(store.grid); grid['level'] = 0
        with pytest.raises(ValueError, match='grid|identity'):
            api().write_segment(store, grid, frame(), tmp_path/'grid.pending')
        foreign = tmp_path/'foreign.pending'; foreign.mkdir()
        (foreign/'original').write_text('keep')
        with pytest.raises(ValueError, match='foreign|owned'):
            api().write_segment(store, store.grid, frame(), foreign)
        assert (foreign/'original').read_text() == 'keep'


@pytest.mark.parametrize('kind', ['shape', 'pixels', 'dtype', 'strips', 'series', 'tag_count', 'tag_payload', 'offset', 'tile_extent'])
def test_tiff_headers_rejected_before_decode(tmp_path, monkeypatch, kind):
    import tifffile
    path = tmp_path/'bad.tif'
    kwargs = dict(tile=(256,256), compression='deflate', metadata=None)
    if kind == 'strips': kwargs.pop('tile')
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.ones((3,3), np.float64 if kind == 'dtype' else np.float32), **kwargs)
        if kind == 'series': writer.write(np.ones((3,3), np.float32), **kwargs)
    if kind in ('tag_count','tag_payload','offset','tile_extent'):
        with path.open('r+b') as f:
            f.seek(4); ifd = struct.unpack('<I', f.read(4))[0]
            f.seek(ifd); count = struct.unpack('<H', f.read(2))[0]
            if kind == 'tag_count':
                f.seek(ifd); f.write(struct.pack('<H', 65535))
            else:
                for i in range(count):
                    pos = ifd+2+12*i
                    f.seek(pos); tag = struct.unpack('<H', f.read(2))[0]
                    if tag == (324 if kind != 'tile_extent' else 322):
                        f.seek(pos+(4 if kind == 'tag_payload' else 8))
                        f.write(struct.pack('<I', 0x7fffffff))
                        break
    def no_decode(*args, **kwargs):
        raise AssertionError('decoder reached before header rejection')
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', no_decode)
    monkeypatch.setattr(api().zlib, 'decompressobj', no_decode)
    with pytest.raises(ValueError):
        list(api().iter_tiff_tiles(path, (4,3) if kind == 'shape' else (3,3), 8 if kind == 'pixels' else 9))


def test_reader_rejects_zip_npy_and_deflate_bomb(tmp_path):
    import tifffile
    import zlib
    for header in (b'\x93NUMPY\x01\x00\xff\xff', b'PK\x03\x04'+b'\xff'*100):
        path = tmp_path/'foreign'
        path.write_bytes(header)
        with pytest.raises(ValueError): list(api().iter_tiff_tiles(path, (2,2), 4))
    path = tmp_path/'bomb.tif'
    # Precompressed iterator bypasses the fixture encoder's shape validation.
    with tifffile.TiffWriter(path) as writer:
        writer.write(iter([zlib.compress(b'0'*(256*256*4+1))]), shape=(2,2),
                     dtype='float32', tile=(256,256), compression='deflate', metadata=None)
    with pytest.raises(ValueError, match='decode|tile|size'):
        list(api().iter_tiff_tiles(path, (2,2), 4))


@pytest.mark.parametrize('big', [False, True])
def test_reader_accepts_classic_and_bigtiff(tmp_path, big):
    import tifffile
    path = tmp_path/'tail.tif'
    a = np.full((256,256), -1, np.float32); a[:2,:2] = [[1,2],[3,4]]
    with tifffile.TiffWriter(path, bigtiff=big) as writer:
        writer.write(iter([a]), shape=(2,2), dtype='float32', tile=(256,256),
                     compression='deflate', metadata=None)
    chunks = list(api().iter_tiff_tiles(path, (2,2), 4))
    assert len(chunks) == 1 and chunks[0][0] == (0,0)
    np.testing.assert_array_equal(chunks[0][1], [[1,2],[3,4]])


def test_low_cache_journal_cap_failure_poison_and_reopen(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
        with store.transaction() as db:
            db.execute('CREATE TABLE spill(id INTEGER PRIMARY KEY, payload BLOB)')
            db.executemany('INSERT INTO spill VALUES(?,?)', ((i,b'a'*3000) for i in range(100)))
        # A deliberately small phase cap makes journal spill deterministic on both OSes.
        store.journal_cap = 64 << 10
        with pytest.raises(Blocked):
            with store.transaction() as db:
                db.execute('PRAGMA cache_size=1')
                db.execute("UPDATE spill SET payload=zeroblob(3000)")
        with pytest.raises(RuntimeError, match='poison'):
            store.guard()
    with open_store(tmp_path, resume=True) as store:
        assert store.db.execute("SELECT count(*) FROM spill WHERE payload=?", (b'a'*3000,)).fetchone() == (100,)
        assert list(store.iter_ids()) == [(0,0)]
    assert not list(tmp_path.rglob('READY*'))


@pytest.mark.parametrize('stage', ['payload', 'transaction', 'committed'])
def test_process_death_releases_lock_and_recovers_prior_commits(tmp_path, stage):
    with open_store(tmp_path) as store:
        store.commit(tile((1,1)))
    script = '''
import os, sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_tifxyz_store import open_store, tile
from contextlib import contextmanager
with open_store(Path(sys.argv[1]), resume=True) as store:
    real = store.transaction
    @contextmanager
    def crash():
        if sys.argv[2] == 'payload': os._exit(17)
        with real() as db:
            yield db
            if sys.argv[2] == 'transaction': os._exit(17)
        os._exit(17)
    store.transaction = crash
    store.commit(tile())
'''
    proc = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path), stage],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 17, proc.stderr
    with open_store(tmp_path, resume=True) as store:
        assert (1,1) in store.iter_ids()
        assert ((0,0) in store.iter_ids()) == (stage == 'committed')
        store.commit(tile())
        assert store.read((0,0))['xyz'][1,1,0] == 1
    assert not list(tmp_path.rglob('READY*'))


def test_sql_rejects_global_materialization_and_unguarded_cached_writes(tmp_path):
    with open_store(tmp_path) as store:
        with store.transaction() as db:
            db.execute('CREATE TABLE samples(id INTEGER PRIMARY KEY, value INTEGER)')
            db.executemany('INSERT INTO samples VALUES(?,?)', [(1,3),(2,2),(3,1)])
        with pytest.raises(sqlite3.DatabaseError):
            store.db.execute('INSERT INTO samples VALUES(?,?)', (4,0))
        for sql in ('SELECT value FROM samples ORDER BY value',
                    'SELECT DISTINCT value FROM samples',
                    'SELECT value FROM samples UNION SELECT id FROM samples'):
            with pytest.raises((ValueError, sqlite3.DatabaseError), match='sort|material|temp'):
                store.db.cursor().execute(sql)
        assert list(store.db.execute('SELECT value FROM samples ORDER BY id')) == [(3,),(2,),(1,)]
        with store.transaction() as db:
            with pytest.raises(sqlite3.DatabaseError): db.commit()
            with pytest.raises(sqlite3.DatabaseError): db.execute('PRAGMA journal_mode=WAL')
            with pytest.raises(sqlite3.DatabaseError): db.execute('CREATE TEMP TABLE escape(x)')
        assert store.db.execute('SELECT count(*) FROM samples').fetchone() == (3,)


def test_sql_insert_batches_and_aggregate_cache_are_bounded(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        stores = [stack.enter_context(open_store(tmp_path/str(i))) for i in range(4)]
        with pytest.raises(Blocked, match='CACHE'):
            open_store(tmp_path/'fifth')
        with stores[0].transaction() as db:
            db.execute('CREATE TABLE rows(x)')
            with pytest.raises(ValueError, match='128|batch'):
                db.executemany('INSERT INTO rows VALUES(?)', ((n,) for n in range(129)))
            assert db.execute('SELECT count(*) FROM rows').fetchone() == (0,)
    with open_store(tmp_path/'after-close') as store:
        assert store.db.execute('PRAGMA cache_size').fetchone() == (-16384,)


def test_identical_commits_need_no_new_sql_space(tmp_path):
    with open_store(tmp_path) as store:
        original = store.commit(tile())
        charge = store.guard()
        store.budget.scratch_bytes = charge['used_bytes']+charge['reserved_bytes']+(64 << 10)
        assert store.commit(tile()) == original


def test_database_cap_failure_in_child_preserves_prior_commit(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_tifxyz_store import open_store
from winding_field.tifxyz_contract import Blocked
with open_store(Path(sys.argv[1]), resume=True) as store:
    try:
        with store.transaction() as db:
            db.execute('CREATE TABLE oversized(x)')
            db.execute('INSERT INTO oversized VALUES(zeroblob(8388608))')
    except Blocked:
        sys.exit(75)
    sys.exit(1)
'''
    proc = subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 75, proc.stderr
    with open_store(tmp_path, resume=True) as store:
        assert list(store.iter_ids()) == [(0,0)]
        assert store.db.execute('PRAGMA quick_check').fetchone() == ('ok',)
        assert not store.db.execute("SELECT name FROM sqlite_master WHERE name='oversized'").fetchone()
    assert not list(tmp_path.rglob('READY*'))


def test_journal_spill_failure_is_reaped_before_recovery(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
        with store.transaction() as db:
            db.execute('CREATE TABLE spill(id INTEGER PRIMARY KEY, payload BLOB)')
            db.executemany('INSERT INTO spill VALUES(?,?)', ((i,b'a'*3000) for i in range(100)))
    script = '''
import sys, resource
from pathlib import Path
sys.path.insert(0, 'tests')
from test_tifxyz_store import open_store
from winding_field.tifxyz_contract import Blocked
before = resource.getrlimit(resource.RLIMIT_FSIZE)
with open_store(Path(sys.argv[1]), resume=True) as store:
    store.journal_cap = 65536
    try:
        with store.transaction() as db:
            db.execute('PRAGMA cache_size=1')
            db.execute('UPDATE spill SET payload=zeroblob(3000)')
    except Blocked as error:
        assert error.code == 'FILE_SIZE_LIMIT', error
        assert resource.getrlimit(resource.RLIMIT_FSIZE) == before
        sys.exit(75)
    sys.exit(1)
'''
    proc = subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 75, proc.stderr
    with open_store(tmp_path, resume=True) as store:
        assert list(store.iter_ids()) == [(0,0)]
        assert store.db.execute('SELECT count(*) FROM spill WHERE payload=?', (b'a'*3000,)).fetchone() == (100,)
        assert store.db.execute('PRAGMA quick_check').fetchone() == ('ok',)
    assert not list(tmp_path.rglob('READY*'))


def test_caught_sqlite_full_still_poisoned_and_rolled_back(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
        with pytest.raises(Blocked):
            with store.transaction() as db:
                db.execute('CREATE TABLE overflow(x)')
                try:
                    db.execute('INSERT INTO overflow VALUES(zeroblob(8388608))')
                except sqlite3.DatabaseError:
                    pass
        with pytest.raises(RuntimeError, match='poison'):
            store.guard()
    with open_store(tmp_path, resume=True) as store:
        assert list(store.iter_ids()) == [(0,0)]
        assert not store.db.execute("SELECT name FROM sqlite_master WHERE name='overflow'").fetchone()


def test_sql_batches_recheck_external_disk_consumption(tmp_path):
    with open_store(tmp_path) as store:
        with pytest.raises(Blocked, match='SCRATCH'):
            with store.transaction() as db:
                db.execute('CREATE TABLE batches(x)')
                charge = store.guard(sql=True)
                store.budget.scratch_bytes = charge['used_bytes']+charge['reserved_bytes']+(512 << 10)
                with (tmp_path/'another-attempt').open('wb') as f:
                    f.truncate(1 << 20)
                db.executemany('INSERT INTO batches VALUES(?)', [(1,),(2,)])
        assert not store.db.execute("SELECT name FROM sqlite_master WHERE name='batches'").fetchone()


def test_assembly_checks_voxel_frame_bounds(tmp_path):
    with open_store(tmp_path, identity((3,3))) as store:
        store.commit(tile(shape=(3,3), value=1000))
        with pytest.raises(ValueError, match='frame|bounds'):
            api().write_segment(store, store.grid, frame(), tmp_path/'invalid.pending')
    assert not list(tmp_path.rglob('READY*'))


def test_nonresource_sql_errors_are_not_reported_as_budget_blocks(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile())
        with pytest.raises(sqlite3.OperationalError, match='no such table'):
            with store.transaction() as db:
                db.execute('INSERT INTO nonexistent VALUES(1)')
        with pytest.raises(RuntimeError, match='poison'):
            store.guard()
    with open_store(tmp_path, resume=True) as store:
        assert list(store.iter_ids()) == [(0,0)]


def test_resume_with_audit_rtrees_preserves_read_and_write_guards(tmp_path):
    with open_store(tmp_path) as store:
        store.commit(tile((1,1)))
        store.commit(tile())
        with store.transaction() as db:
            for name in ('audit_nodes', 'audit_triangles'):
                db.execute(f'CREATE VIRTUAL TABLE {name} USING rtree(id,x0,x1,y0,y1,z0,z1)')
                db.execute(f'INSERT INTO {name} VALUES(1,0,1,0,1,0,1)')
    with open_store(tmp_path, resume=True) as store:
        assert store.db.execute('PRAGMA quick_check').fetchone() == ('ok',)
        for name in ('audit_nodes', 'audit_triangles'):
            assert store.db.execute(f'SELECT id FROM {name} WHERE x0<=1 AND x1>=0').fetchone() == (1,)
            with pytest.raises(sqlite3.DatabaseError):
                store.db.execute(f'INSERT INTO {name} VALUES(2,0,1,0,1,0,1)')
        with pytest.raises(sqlite3.DatabaseError):
            store.db.execute("UPDATE sqlite_master SET sql='invalid' WHERE name='tiles'")
        with pytest.raises(sqlite3.DatabaseError):
            store.db.execute('PRAGMA writable_schema=ON')
        with pytest.raises(ValueError, match='sort|material'):
            store.db.execute('SELECT bytes FROM tiles ORDER BY bytes')
        assert list(store.iter_ids()) == [(0,0), (1,1)]
        query = 'SELECT ty,tx FROM tiles ORDER BY ty,tx'
        assert all('TEMP B-TREE' not in row[-1] for row in store.db.execute('EXPLAIN QUERY PLAN '+query))
        opcodes = {row[1] for row in store.db.execute('EXPLAIN '+query)}
        assert not opcodes & {'SorterOpen', 'OpenEphemeral', 'OpenAutoindex'}
        assert 'OpenRead' in opcodes
        assert store.db.execute("SELECT 'text decoding restored'").fetchone() == ('text decoding restored',)
        with store.transaction() as db:
            db.execute('INSERT INTO audit_nodes VALUES(2,2,3,2,3,2,3)')
        assert store.db.execute('SELECT count(*) FROM audit_nodes').fetchone() == (2,)
        store.budget.scratch_bytes = 1
        with pytest.raises(Blocked, match='SCRATCH'):
            with store.transaction(): pass
