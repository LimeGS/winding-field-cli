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
    assert v.stats()['downloads'] == 7


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
