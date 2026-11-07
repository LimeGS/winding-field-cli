import http.server
import json
import threading
from functools import partial

import numpy as np
import pytest


def make_volume(root, *, shape=(4, 4, 6), chunks=(2, 2, 3)):
    root.mkdir()
    (root / '.zattrs').write_text(json.dumps({'multiscales': [{
        'axes': [{'name': a, 'type': 'space', 'unit': 'micrometer'} for a in 'zyx'],
        'datasets': [{'path': '0', 'coordinateTransformations': [
            {'type': 'scale', 'scale': [4, 3, 2]},
            {'type': 'translation', 'translation': [30, 20, 10]}]}]}]}))
    p = root / '0'
    p.mkdir()
    (p / '.zarray').write_text(json.dumps(dict(zarr_format=2, shape=shape, chunks=chunks,
        dtype='|u1', compressor=None, fill_value=0, order='C', filters=None, dimension_separator='.')))
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    for z in range(0, shape[0], chunks[0]):
        for y in range(0, shape[1], chunks[1]):
            for x in range(0, shape[2], chunks[2]):
                key = f'{z//chunks[0]}.{y//chunks[1]}.{x//chunks[2]}'
                (p / key).write_bytes(data[z:z+chunks[0], y:y+chunks[1], x:x+chunks[2]].tobytes())
    return data


@pytest.fixture
def served(tmp_path):
    data = make_volume(tmp_path / 'ct.zarr')
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f'http://127.0.0.1:{server.server_port}/ct.zarr', data
    server.shutdown()
    server.server_close()
    thread.join()


def test_remote_roi_and_geometry(served):
    from winding_field.io import Volume
    root, url, data = served
    v = Volume(url, cache=root/'cache', cache_bytes=64)
    np.testing.assert_array_equal(v.read([1, 1, 2], [4, 4, 6]), data[1:4, 1:4, 2:6])
    assert v.spacing_xyz == [2., 3., 4.]
    assert v.origin_xyz == [10., 20., 30.]
    assert v.stats()['cache_bytes'] <= 64


def test_incomplete_local_mirror_fetches_missing(served):
    from winding_field.io import Volume
    root, url, data = served
    local = root/'partial'
    (local/'0').mkdir(parents=True)
    for name in ('.zattrs', '0/.zarray', '0/0.0.0'):
        (local/name).write_bytes((root/'ct.zarr'/name).read_bytes())
    v = Volume(url, local=local, cache=root/'cache', cache_bytes=128)
    np.testing.assert_array_equal(v.read([0, 0, 0], [4, 4, 6]), data)
    assert v.stats()['local_hits'] == 1
    assert v.stats()['downloads'] == 8  # Generic HTTP also verifies the one mirrored chunk.


def test_warm_offline_and_miss(served):
    from winding_field.io import Volume
    root, url, data = served
    v = Volume(url, cache=root/'cache', cache_bytes=128)
    v.read([0, 0, 0], [2, 2, 3])
    offline = Volume(url, cache=root/'cache', cache_bytes=128, offline=True)
    np.testing.assert_array_equal(offline.read([0, 0, 0], [2, 2, 3]), data[:2,:2,:3])
    with pytest.raises(FileNotFoundError, match='cache|offline'):
        offline.read([2, 2, 3], [4, 4, 6])


def test_incompatible_local_metadata_rejected(served):
    from winding_field.io import Volume
    root, url, _ = served
    local = root/'wrong'
    make_volume(local, shape=(4, 4, 3))
    with pytest.raises(ValueError, match='local|metadata'):
        Volume(url, local=local, cache=root/'cache')


def test_roi_outside_and_negative_cache_rejected(served):
    from winding_field.io import Volume
    root, url, _ = served
    with pytest.raises(ValueError):
        Volume(url, cache=root/'cache', cache_bytes=-1)
    v = Volume(str(root/'ct.zarr'), cache=root/'cache')
    with pytest.raises(ValueError):
        v.read([-1,0,0], [2,2,2])
    with pytest.raises(ValueError):
        v.read([0,0,0], [4,4,7])


def test_remote_mutation_does_not_reuse_cached_bytes(served):
    from winding_field.io import Volume
    root, url, _ = served
    first = Volume(url, cache=root/'cache')
    first.read([0,0,0], [2,2,3])
    (root/'ct.zarr/0/0.0.0').write_bytes(bytes([255])*12)
    second = Volume(url, cache=root/'cache')
    assert (second.read([0,0,0], [2,2,3]) == 255).all()


def test_local_missing_chunk_is_not_air(served):
    from winding_field.io import Volume
    root, _, _ = served
    (root/'ct.zarr/0/0.0.0').unlink()
    v = Volume(root/'ct.zarr', cache=root/'cache')
    with pytest.raises(FileNotFoundError):
        v.read([0,0,0], [2,2,3])


def test_metadata_change_changes_identity(served):
    from winding_field.io import Volume
    root, url, _ = served
    first = Volume(url, cache=root/'cache')
    p = root/'ct.zarr/.zattrs'
    attrs = json.loads(p.read_text())
    attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][0] = 5
    p.write_text(json.dumps(attrs))
    second = Volume(url, cache=root/'cache')
    assert first.identity != second.identity
    assert second.spacing_xyz == [2.,3.,5.]


def test_authoritative_remote_sparse_chunk_uses_fill_and_is_cached_offline(served):
    from winding_field.io import Volume
    root,url,_=served
    (root/'ct.zarr/0/0.0.0').unlink()
    online=Volume(url,cache=root/'cache')
    np.testing.assert_array_equal(online.read([0,0,0],[2,2,3]),np.zeros((2,2,3),np.uint8))
    offline=Volume(url,cache=root/'cache',offline=True)
    np.testing.assert_array_equal(offline.read([0,0,0],[2,2,3]),np.zeros((2,2,3),np.uint8))


def test_remote_update_larger_than_cache_cannot_revive_old_payload(served):
    from numcodecs import Zlib
    from winding_field.io import Volume
    root,url,_=served
    codec=Zlib()
    metadata=root/'ct.zarr/0/.zarray'
    value=json.loads(metadata.read_text()); value['compressor']=codec.get_config()
    metadata.write_text(json.dumps(value))
    chunk=root/'ct.zarr/0/0.0.0'
    chunk.write_bytes(codec.encode(bytes(12)))
    first=Volume(url,cache=root/'cache',cache_bytes=16)
    assert not first.read([0,0,0],[2,2,3]).any()
    first.close()
    expected=np.arange(41,53,dtype=np.uint8).reshape(2,2,3)
    encoded=codec.encode(expected)
    assert len(encoded)>16
    chunk.write_bytes(encoded)
    current=Volume(url,cache=root/'cache',cache_bytes=16)
    for _ in range(2):
        np.testing.assert_array_equal(current.read([0,0,0],[2,2,3]),expected)
    current.close()
    offline=Volume(url,cache=root/'cache',cache_bytes=16,offline=True)
    with pytest.raises(FileNotFoundError,match='cache miss'):
        offline.read([0,0,0],[2,2,3])
    offline.close()


def test_online_mirror_resume_detects_changed_authoritative_chunk(served):
    import shutil
    from winding_field.io import Volume
    root,url,data=served
    local=root/'mirror'; shutil.copytree(root/'ct.zarr',local)
    first=Volume(url,local=local,cache=root/'cache')
    first.bind_ledger(root/'source.sqlite')
    np.testing.assert_array_equal(first.read([0,0,0],[2,2,3]),data[:2,:2,:3])
    first.close()
    (root/'ct.zarr/0/0.0.0').write_bytes(bytes([255])*12)
    resumed=Volume(url,local=local,cache=root/'cache')
    resumed.bind_ledger(root/'source.sqlite')
    with pytest.raises(ValueError,match='changed|differs'):
        resumed.validate_used()
    resumed.close()


def test_online_mirror_rejects_wrong_content_with_matching_metadata(served):
    import shutil
    from winding_field.io import Volume
    root,url,_=served
    local=root/'mirror'; shutil.copytree(root/'ct.zarr',local)
    (local/'0/0.0.0').write_bytes(bytes([255])*12)
    volume=Volume(url,local=local,cache=root/'cache')
    with pytest.raises(ValueError,match='differs'):
        volume.read([0,0,0],[2,2,3])
    volume.close()


def test_sampled_reads_preserve_duplicates_and_cross_chunk_coordinates(served):
    from winding_field.io import Volume
    root,url,data=served
    volume=Volume(url,cache=root/'cache')
    indices=[np.array([0,0,3]),np.array([1,3]),np.array([0,2,5,5])]
    np.testing.assert_array_equal(volume.sample(indices),data[np.ix_(*indices)])
    with pytest.raises(ValueError):
        volume.sample([[0],[-1],[0]])
    volume.close()


def test_unsupported_codec_fails_at_open_with_compatibility_guidance(tmp_path):
    from winding_field.io import Volume
    make_volume(tmp_path/'ct')
    path=tmp_path/'ct/0/.zarray';metadata=json.loads(path.read_text())
    metadata['filters']=[{'id':'quantize','digits':1,'dtype':'<f4','astype':'<f4'}]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='v1.5.0'):
        Volume(tmp_path/'ct',cache=tmp_path/'cache')


@pytest.mark.parametrize('order',['C','F'])
def test_checked_native_snapshot_preserves_filtered_layout(tmp_path,order):
    from contextlib import closing
    from numcodecs import Delta,Shuffle,Zlib
    from winding_field.io import Volume
    source=tmp_path/'ct';source.mkdir()
    data=np.arange(24,dtype='>u2').reshape(2,3,4)
    filters=[Delta(dtype='>u2'),Shuffle(elementsize=2)];compressor=Zlib()
    metadata=dict(zarr_format=2,shape=data.shape,chunks=data.shape,dtype=data.dtype.str,
                  order=order,fill_value=0,compressor=compressor.get_config(),
                  filters=[f.get_config() for f in filters])
    (source/'.zarray').write_text(json.dumps(metadata))
    raw=data.tobytes(order=order)
    for codec in filters:raw=codec.encode(raw)
    (source/'0.0.0').write_bytes(compressor.encode(raw))
    with closing(Volume(source,voxel_um=1,cache=tmp_path/'cache')) as volume:
        np.testing.assert_array_equal(volume.read([0,1,1],[2,3,4]),data[:,1:,1:])


@pytest.mark.parametrize('decoded_cache_bytes',[0,1024])
def test_decoder_consumes_verified_snapshot_after_source_replacement(tmp_path,monkeypatch,decoded_cache_bytes):
    from contextlib import closing
    from winding_field.io import Volume
    data=make_volume(tmp_path/'ct')
    with closing(Volume(tmp_path/'ct',cache=tmp_path/'cache',decoded_cache_bytes=decoded_cache_bytes)) as volume:
        volume.bind_ledger(tmp_path/'source.sqlite');read=volume._chunk_bytes
        def replace_after_verification(key):
            raw=read(key)
            (tmp_path/'ct'/key).write_bytes(bytes([255])*len(raw))
            return raw
        monkeypatch.setattr(volume,'_chunk_bytes',replace_after_verification)
        np.testing.assert_array_equal(volume.read([0,0,0],[2,2,3]),data[:2,:2,:3])
        with pytest.raises(ValueError,match='changed'):volume.validate_used()


def test_concurrent_decoded_cache_charges_each_retained_chunk_once(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing
    import time
    from winding_field.io import Volume
    expected=make_volume(tmp_path/'ct')
    with closing(Volume(tmp_path/'ct',cache=tmp_path/'cache',decoded_cache_bytes=4096)) as volume:
        read=volume._chunk_bytes
        def delayed(key):
            raw=read(key);time.sleep(.05)
            return raw
        monkeypatch.setattr(volume,'_chunk_bytes',delayed)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(volume.decoded_chunk,[volume.array_path+'/0.0.0']*2))
        for result in results:np.testing.assert_array_equal(result,expected[:2,:2,:3])
        assert volume.stats()['decoded_cache_bytes']==results[0].nbytes+512


def test_volume_closes_indexes_when_native_reader_initialization_fails(tmp_path,monkeypatch):
    import sqlite3
    from vc_reader.reader import ZarrReader
    from winding_field.io import Volume
    make_volume(tmp_path/'ct');connections=[];connect=sqlite3.connect
    def tracked(*args,**kwargs):
        db=connect(*args,**kwargs);connections.append(db);return db
    def failed(*args,**kwargs):raise RuntimeError('computed reader initialization failed')
    monkeypatch.setattr(sqlite3,'connect',tracked)
    monkeypatch.setattr(ZarrReader,'computed',failed)
    with pytest.raises(RuntimeError,match='initialization failed'):
        Volume(tmp_path/'ct',cache=tmp_path/'cache')
    assert connections
    for db in connections:
        with pytest.raises(sqlite3.ProgrammingError,match='closed'):
            db.execute('SELECT 1')


def test_local_volumes_do_not_accumulate_native_disk_cache_descriptors(tmp_path):
    from contextlib import closing
    import os
    from winding_field.io import Volume
    make_volume(tmp_path/'ct')
    with closing(Volume(tmp_path/'ct',cache=tmp_path/'warmup')) as volume:
        volume.read([0,0,0],[2,2,3])
    before=len(os.listdir('/dev/fd'))
    for n in range(8):
        with closing(Volume(tmp_path/'ct',cache=tmp_path/f'cache-{n}')) as volume:
            volume.read([0,0,0],[2,2,3])
    assert len(os.listdir('/dev/fd'))<=before+2
