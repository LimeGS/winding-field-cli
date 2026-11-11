"""Remote field contracts: real encoded chunks, mocked network boundary only."""
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
from copy import deepcopy
from urllib.parse import urlsplit

import numpy as np
import pytest
from numcodecs import Blosc

BASE = 'https://example.org/pinned/'


def api():
    assert importlib.util.find_spec('winding_field.remote_field'), 'remote Zarr field reader not implemented'
    return importlib.import_module('winding_field.remote_field')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded_json(value):
    return json.dumps(value, allow_nan=False).encode()


def export_fixture(monkeypatch, *, order='C', observed=False):
    """Full tiny export, independently encoded using numcodecs, not the reader."""
    module = api()
    values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    values[2, 3, 4] = np.nan
    valid = np.ones(values.shape, np.uint8)
    valid[1, 2, 3] = 0
    masks = dict(valid=valid, observed=(valid & (values % 2 == 0)).astype('u1'))
    geometry = dict(frame='test/frame', voxel_um=9.6, origin_xyz=[1., -2., 3.],
                    spacing_xyz=[32., 32., 32.], shape_zyx=[3, 4, 5], sheet_step=1.192497988541921)
    metadata = dict(schema='winding-field-zarr-v1', geometry=geometry,
                    calibration=dict(sheet_step=geometry['sheet_step'], stored_quantity='raw phi'),
                    chunk_shape_zyx=[2, 3, 3],
                    arrays={n: 'field.zarr/'+n for n in ('phi', 'valid', 'observed')},
                    observation_support=dict(available=observed), completion='COMPLETE.json',
                    receipts='receipts/{z}/{y}/{x}.json')
    objects = {'field.json': encoded_json(metadata)}
    binding = sha(objects['field.json'])
    codec = Blosc(cname='zstd', clevel=5, shuffle=2)
    for name in ('phi', 'valid', 'observed'):
        dtype = '<f4' if name == 'phi' else '|u1'
        objects[f'field.zarr/{name}/.zarray'] = encoded_json(dict(zarr_format=2,
            shape=[3, 4, 5], chunks=[2, 3, 3], dtype=dtype, order=order,
            compressor=codec.get_config(), filters=None, dimension_separator='/',
            fill_value='NaN' if name == 'phi' else 0))
    manifest = dict(schema='winding-field-zarr-manifest-v1', binding=binding, chunks=[])
    for key in itertools.product(range(2), repeat=3):
        start = tuple(k*c for k, c in zip(key, (2, 3, 3)))
        shape = tuple(min(c, n-s) for c, n, s in zip((2, 3, 3), values.shape, start))
        slices = tuple(slice(s, s+n) for s, n in zip(start, shape))
        files, arrays = {}, {}
        suffix = '/'.join(map(str, key))
        for name in ('phi', 'valid', 'observed'):
            block = np.zeros((2, 3, 3), dtype='<f4' if name == 'phi' else 'u1', order=order)
            block[tuple(slice(0, n) for n in shape)] = values[slices] if name == 'phi' else masks[name][slices]
            raw = codec.encode(block.tobytes(order=order))
            path = f'field.zarr/{name}/{suffix}'
            objects[path] = raw
            files[path] = arrays[name] = dict(bytes=len(raw), sha256=sha(raw))
        receipt = dict(schema='winding-field-zarr-chunk-v1', key=list(key),
                       binding=binding, shape_zyx=list(shape), arrays=arrays)
        path = f'receipts/{suffix}.json'
        objects[path] = encoded_json(receipt)
        files[path] = dict(bytes=len(objects[path]), sha256=sha(objects[path]))
        manifest['chunks'].append(dict(key=list(key), files=files))
    complete = dict(schema='winding-field-zarr-complete-v1', binding=binding,
                    status='DATA_UPLOAD_COMPLETE', summary=dict(chunks=8))
    descriptor = dict(kind='zarr-v2', url=BASE+'field.json', sha256=binding,
                      support_policy='stored')
    requests = []

    def publish():
        objects['manifest.json'] = encoded_json(manifest)
        complete['manifest'] = dict(bytes=len(objects['manifest.json']), sha256=sha(objects['manifest.json']))
        objects['COMPLETE.json'] = encoded_json(complete)
        descriptor['complete_sha256'] = sha(objects['COMPLETE.json'])

    def fetch(url, *, limit):
        assert url.startswith(BASE)
        path = url[len(BASE):]
        requests.append(path)
        if path not in objects:
            raise FileNotFoundError('remote object missing')
        if len(objects[path]) > limit:
            raise ValueError('read budget exceeded')
        return objects[path]

    publish()
    monkeypatch.setattr(module, 'fetch', fetch)
    return module, descriptor, geometry, values, valid, objects, requests, manifest, complete, publish


@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('budget', [0, 1024*1024])
def test_crop_is_lazy_exact_and_keeps_support_and_units(monkeypatch, order, budget):
    m, d, g, values, valid, objects, requests, *_ = export_fixture(monkeypatch, order=order)
    source = m.RemoteField(d, g, cache_bytes=budget)
    assert not any(p.split('/')[-1].isdigit() for p in requests)
    requests.clear()
    result, support = source.read((0, 1, 2), (3, 4, 5))
    np.testing.assert_array_equal(result, values[:, 1:, 2:])
    np.testing.assert_array_equal(support, valid[:, 1:, 2:].astype(bool) & np.isfinite(values[:, 1:, 2:]))
    payloads = [p for p in requests if p.split('/')[-1].isdigit()]
    assert set(payloads) == {f'field.zarr/{n}/{z}/{y}/{x}' for n in ('phi', 'valid')
                            for z, y, x in itertools.product(range(2), repeat=3)}
    assert result.dtype == np.float64 and support.dtype == bool
    assert result.flags.owndata and result.flags.c_contiguous and support.flags.owndata
    assert source.identity['geometry'] == g
    result[:] = -99
    again, _ = source.read((0, 0, 0), (1, 1, 1))
    assert again.item() == 0
    source.close()
    source.close()
    with pytest.raises(RuntimeError):source.read((0, 0, 0), (1, 1, 1))


def test_small_crop_does_not_download_other_chunks(monkeypatch):
    m, d, g, _, _, _, requests, *_ = export_fixture(monkeypatch)
    source = m.RemoteField(d, g)
    requests.clear()
    values, valid = source.read((0, 0, 0), (1, 1, 1))
    assert values.item() == 0 and valid.item()
    assert [p for p in requests if p.split('/')[-1].isdigit()] == [
        'field.zarr/phi/0/0/0', 'field.zarr/valid/0/0/0']
    assert not any('/observed/' in p for p in requests)


def test_observed_policy_requires_available_support(monkeypatch):
    m, d, g, *_ = export_fixture(monkeypatch)
    d['support_policy'] = 'observed'
    with pytest.raises(ValueError, match='observ'):
        m.RemoteField(d, g)
    m, d, g, *_ = export_fixture(monkeypatch, observed=True)
    d['support_policy'] = 'observed'
    v, valid = m.RemoteField(d, g).read((0, 0, 0), (1, 1, 3))
    np.testing.assert_array_equal(v, [[[0, 1, 2]]])
    np.testing.assert_array_equal(valid, [[[True, False, True]]])


@pytest.mark.parametrize('change', ['field_pin', 'complete_pin', 'manifest_hash', 'binding',
    'missing_entry', 'duplicate_entry', 'path_traversal', 'extra_entry', 'bad_size',
    'geometry', 'dtype', 'shape', 'chunk_shape', 'order', 'codec', 'filters', 'unknown_option',
    'separator', 'manifest_size'])
def test_bad_metadata_rejected_before_payloads(monkeypatch, change):
    m, d, g, _, _, objects, requests, manifest, complete, publish = export_fixture(monkeypatch)
    if change == 'field_pin':d['sha256'] = '0'*64
    elif change == 'complete_pin':d['complete_sha256'] = '0'*64
    elif change == 'manifest_hash':objects['manifest.json'] += b' '
    elif change == 'binding':manifest['binding'] = '0'*64;publish()
    elif change == 'missing_entry':manifest['chunks'].pop();publish()
    elif change == 'duplicate_entry':manifest['chunks'][1] = deepcopy(manifest['chunks'][0]);publish()
    elif change == 'path_traversal':manifest['chunks'][0]['files']['../evil'] = dict(bytes=1,sha256='0'*64);publish()
    elif change == 'extra_entry':manifest['chunks'].append(dict(key=[2, 0, 0],files={}));publish()
    elif change == 'bad_size':manifest['chunks'][0]['files']['field.zarr/phi/0/0/0']['bytes'] = 2**40;publish()
    elif change == 'geometry':g = dict(g, sheet_step=2.)
    elif change == 'unknown_option':d['token'] = 'secret'
    elif change == 'manifest_size':
        complete['manifest']['bytes'] += 1
        objects['COMPLETE.json'] = encoded_json(complete)
        d['complete_sha256'] = sha(objects['COMPLETE.json'])
    else:
        key = 'field.zarr/phi/.zarray'
        meta = json.loads(objects[key])
        if change == 'dtype':meta['dtype'] = '|O'
        elif change == 'shape':meta['shape'] = [4, 4, 5]
        elif change == 'chunk_shape':meta['chunks'] = [2, 2, 2]
        elif change == 'order':meta['order'] = 'X'
        elif change == 'codec':meta['compressor'] = dict(id='pickle')
        elif change == 'filters':meta['filters'] = [dict(id='pickle')]
        elif change == 'separator':meta['dimension_separator'] = '.'
        objects[key] = encoded_json(meta)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.RemoteField(d, g)
    assert not any(p.split('/')[-1].isdigit() for p in requests)


@pytest.mark.parametrize('change', ['missing', 'corrupt', 'receipt', 'mask_value'])
def test_payload_failures_are_not_silent_fill(monkeypatch, change):
    m, d, g, _, _, objects, _, manifest, _, publish = export_fixture(monkeypatch)
    key = 'field.zarr/phi/0/0/0'
    if change == 'missing':objects.pop(key)
    elif change == 'corrupt':objects[key] = b'x'*len(objects[key])
    elif change == 'receipt':objects['receipts/0/0/0.json'] += b' '
    else:
        key = 'field.zarr/valid/0/0/0'
        objects[key] = Blosc(cname='zstd', clevel=5, shuffle=2).encode(bytes([2])*18)
        manifest['chunks'][0]['files'][key] = dict(bytes=len(objects[key]),sha256=sha(objects[key]))
        # Also bind the receipt to those invalid decoded mask values.
        path = 'receipts/0/0/0.json';r = json.loads(objects[path])
        r['arrays']['valid'] = manifest['chunks'][0]['files'][key]
        objects[path] = encoded_json(r)
        manifest['chunks'][0]['files'][path] = dict(bytes=len(objects[path]),sha256=sha(objects[path]))
        publish()
    source = m.RemoteField(d, g)
    with pytest.raises((ValueError, FileNotFoundError)):
        source.read((0, 0, 0), (1, 1, 1))


def test_cache_and_validation_cannot_hide_changed_source(monkeypatch):
    m, d, g, _, _, objects, _, *_ = export_fixture(monkeypatch)
    source = m.RemoteField(d, g, cache_bytes=1024*1024)
    source.read((0, 0, 0), (1, 1, 1))
    objects['field.zarr/phi/0/0/0'] += b'x'
    for action in [lambda: source.read((0, 0, 0), (1, 1, 1)),
                   lambda: source.validate_region((0, 0, 0), (1, 1, 1)),
                   source.validate]:
        with pytest.raises(ValueError):action()


@pytest.mark.parametrize('url', ['http://example.org/a', 'https://user:secret@example.org/a',
    'https://example.org/a?token=secret', 'https://example.org/a#x',
    'https://example.org/a/../b', 'https://example.org/%2e%2e/a', 'https://127.0.0.1/a'])
def test_unsafe_urls_never_open_connections(monkeypatch, url):
    m = api()
    with pytest.raises(ValueError) as caught:
        m.fetch(url, limit=1024)
    assert 'secret' not in str(caught.value)


def test_https_binds_validated_address_and_scopes_hf_token(monkeypatch):
    m = api()
    monkeypatch.setenv('HF_TOKEN', 'hf_DO_NOT_LEAK')
    calls, connected, resolutions = [], [], []
    import socket
    def resolve(host, port, **kwargs):
        resolutions.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
    class Sock:
        def settimeout(self, timeout):pass
        def connect(self, dest):connected.append(dest)
        def close(self):pass
    monkeypatch.setattr(m.socket, 'getaddrinfo', resolve)
    monkeypatch.setattr(m.socket, 'socket', lambda *args: Sock())
    class Response:
        def __init__(self, status, headers, raw=b''):self.status,self.headers,self.raw=status,headers,raw
        def read(self, n):return self.raw[:n]
        def close(self):pass
    responses = iter([Response(302, {'Location':'https://cdn.example.org/field.zarr%2Fphi%2F.zarray?signature=signed'}),
                      Response(200, {'Content-Length':'2'}, b'ok')])
    def request(self, method, path, headers):
        calls.append((self.host, path, headers))
        self._create_connection((self.host,443),60)
    monkeypatch.setattr(m.HTTPSConnection, 'request', request)
    monkeypatch.setattr(m.HTTPSConnection, 'getresponse', lambda self: next(responses))
    monkeypatch.setattr(m.HTTPSConnection, 'close', lambda self: None)
    assert m.fetch('https://huggingface.co/datasets/x/resolve/rev/a', limit=10) == b'ok'
    assert resolutions == ['huggingface.co', 'cdn.example.org']
    assert connected == [('93.184.216.34',443)]*2
    assert calls[0][2]['Authorization'] == 'Bearer hf_DO_NOT_LEAK'
    assert 'Authorization' not in calls[1][2]


@pytest.mark.parametrize('path', ['/a/%2e%2e/b', '/a/%252e%252e/b', '/a/%5cb', '/a/%00b'])
def test_encoded_redirect_traversal_is_not_accepted(path):
    with pytest.raises(ValueError):api()._url('https://example.org'+path,redirect=True)


def test_redirect_to_private_address_is_blocked_without_secret_in_error(monkeypatch):
    m = api()
    import socket
    monkeypatch.setattr(m.socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET,socket.SOCK_STREAM,6,'',('10.0.0.1',443))])
    with pytest.raises(ValueError, match='address') as caught:
        m.fetch('https://example.org/a', limit=10)
    assert 'token' not in str(caught.value)


def test_invalid_environment_token_is_not_echoed_by_http_headers(monkeypatch):
    m = api()
    monkeypatch.setenv('HF_TOKEN', 'hf_SECRET\nInjected: invalid')
    import socket
    monkeypatch.setattr(m.socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))])
    with pytest.raises(ValueError) as caught:
        m.fetch('https://huggingface.co/datasets/x/resolve/rev/a', limit=10)
    assert 'hf_SECRET' not in str(caught.value)


@pytest.mark.parametrize('lo,hi', [((-1,0,0),(1,1,1)), ((0,0,0),(4,4,5)),
    ((True,0,0),(2,2,2)), ((0.,0,0),(1,1,1)), ((0,0,0),(0,1,1))])
def test_invalid_crops_fail_without_payload_reads(monkeypatch, lo, hi):
    m, d, g, _, _, _, requests, *_ = export_fixture(monkeypatch)
    source = m.RemoteField(d, g)
    requests.clear()
    with pytest.raises(ValueError):source.read(lo, hi)
    assert not requests


def test_shared_field_source_and_cli_config_keep_remote_descriptor(monkeypatch, tmp_path):
    _, d, g, values, valid, *_ = export_fixture(monkeypatch)
    from winding_field.surface_sources import FieldSource
    from winding_field.surface_workflow import load_surface_config
    from winding_field.surface_parallel import _child_config
    config = dict(schema='winding-surfaces-v1', field=d, geometry=g, levels=[10.5, 30.5])
    path = tmp_path/'remote.json';path.write_text(json.dumps(config))
    loaded = load_surface_config(path)
    assert loaded['field'] == d
    assert _child_config(loaded, 30.5)['field'] == d
    with FieldSource(loaded['field'], g, cache_bytes=1024*1024) as source:
        actual, support = source.read(np.array([0,0,0]), np.array([3,4,5]))
        np.testing.assert_array_equal(actual, values)
        np.testing.assert_array_equal(support, valid.astype(bool) & np.isfinite(values))
        assert source.voxel_um == 9.6 and source.sheet_step == 1.192497988541921
        np.testing.assert_array_equal(source.origin, [1, -2, 3])
        assert source.identity['field'] == d


def test_remote_mesh_equals_local_and_resume_rejects_corruption_without_writes(monkeypatch, tmp_path):
    _, d, g, values, valid, objects, *_ = export_fixture(monkeypatch)
    from winding_field.surface_sources import FieldSource
    from winding_field.native_mesh import extract_mesh
    path = tmp_path/'values.npy';np.save(path, values)
    mask = tmp_path/'valid.npy';np.save(mask, valid.astype(bool))
    local = dict(kind='npy',path=str(path),sha256=sha(path.read_bytes()),
                 mask=dict(path=str(mask),sha256=sha(mask.read_bytes())))
    for descriptor, name in [(local,'local'),(d,'remote')]:
        with FieldSource(descriptor, g) as source:
            result = extract_mesh(source,tmp_path/name,[10.5,30.5],tile=2)
            assert result['status'] == 'MESH_READY'
    for name in ('level-0000.obj','level-0001.obj'):
        assert (tmp_path/'local'/name).read_bytes() == (tmp_path/'remote'/name).read_bytes()
    def snapshot():
        return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (tmp_path/'remote').iterdir() if p.is_file()}
    # Native extractor historical completion rewrites receipts; prepared supervisor
    # supplies the read-only completed-child path. Source validation itself is read-only.
    before = snapshot()
    with FieldSource(d, g) as source:source.validate(hashes=True)
    assert snapshot() == before
    objects['field.zarr/phi/0/0/0'] += b'x'
    with FieldSource(d, g) as source:
        with pytest.raises(ValueError):extract_mesh(source,tmp_path/'remote',[10.5,30.5],tile=2,resume=True)
    assert snapshot() == before


def test_prepared_pipeline_stop_resume_remote_source(monkeypatch, tmp_path):
    _, d, g, _, _, objects, *_ = export_fixture(monkeypatch)
    from winding_field.surface_workflow import run_surfaces
    config = dict(schema='winding-surfaces-v1', field=d, geometry=g, levels=[10.5],
        execution=dict(tile=2),render=dict(width=32,height=32))
    path=tmp_path/'run.json';path.write_text(json.dumps(config))
    out=tmp_path/'output'
    result=run_surfaces(path,out,stop=lambda:True)
    assert result['status']=='CHECKPOINTED_STOP'
    assert run_surfaces(path,out,resume=True)['status']=='SURFACES_READY'
    assert (out/'level-0000.obj').is_file()
    before={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    objects['field.json'] += b' '
    with pytest.raises(ValueError):run_surfaces(path,out,resume=True)
    assert {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}==before


def test_parallel_admission_and_early_stop_do_not_fetch_whole_remote_field(monkeypatch, tmp_path):
    _, d, g, _, _, _, requests, *_ = export_fixture(monkeypatch)
    from winding_field.surface_parallel import run_parallel_levels
    config=dict(schema='winding-surfaces-v1',field=d,geometry=g,levels=[10.5],execution=dict(tile=2))
    path=tmp_path/'run.json';path.write_text(json.dumps(config))
    result=run_parallel_levels(path,tmp_path/'collection',stop=lambda:True)
    assert result['status']=='CHECKPOINTED_STOP'
    assert not any(p.split('/')[-1].isdigit() for p in requests)


@pytest.mark.parametrize('tail', ['\x7f', '\u2603'])
def test_signed_redirect_errors_do_not_echo_credentials(monkeypatch, tail):
    m=api()
    import socket
    monkeypatch.setattr(m.socket,'getaddrinfo',lambda *a,**k:[
        (socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))])
    real_request=m.HTTPSConnection.request
    def request(self, method, path, headers):
        if self.host!='huggingface.co':real_request(self,method,path,headers=headers)
    class Response:
        status=302
        headers={'Location':'https://cdn.example.org/a?signature=SECRET_SIGNED_TOKEN'+tail}
        def close(self):pass
    monkeypatch.setattr(m.HTTPSConnection,'request',request)
    monkeypatch.setattr(m.HTTPSConnection,'getresponse',lambda self:Response())
    monkeypatch.setattr(m.HTTPSConnection,'close',lambda self:None)
    with pytest.raises((ValueError,OSError)) as caught:
        m.fetch('https://huggingface.co/datasets/a/resolve/rev/field.json',limit=10)
    assert 'SECRET_SIGNED_TOKEN' not in str(caught.value)


@pytest.mark.parametrize('paused', [False,True])
def test_remote_uvw_resume_verifies_payload_before_retained_output_writes(monkeypatch,tmp_path,paused):
    _,d,g,_,_,objects,*_=export_fixture(monkeypatch)
    from test_uvw_pipeline import inputs
    from winding_field.uvw_pipeline import export_existing
    args=inputs(tmp_path)
    config=__import__('pathlib').Path(args.field_config)
    config.write_text(json.dumps(dict(field=d,geometry=g)))
    result=export_existing(args,stop=(lambda:True) if paused else None)
    assert result['status'] in ('PAUSED','UVW_READY_DIAGNOSTIC','UVW_EMPTY_DIAGNOSTIC')
    out=__import__('pathlib').Path(args.output)
    before={str(p.relative_to(out)):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    objects['field.zarr/phi/0/0/0']+=b'x'
    args.resume=True
    with pytest.raises(ValueError):export_existing(args)
    assert {str(p.relative_to(out)):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('change',['order','payload'])
def test_remote_range_resume_binds_decode_metadata_and_verifies_payload_before_writes(monkeypatch,tmp_path,change):
    _,d,g,_,_,objects,*_=export_fixture(monkeypatch)
    from winding_field.range_surface import extract_range
    recipe=dict(field=d,geometry=g,umbilicus=[[1,-2,3],[1,-2,67]],h=1,samples_per_w=4)
    out=tmp_path/'range'
    assert extract_range(recipe,[-61,-60],out)['status']=='EMPTY'
    before={str(p.relative_to(out)):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    if change=='order':
        path='field.zarr/phi/.zarray';meta=json.loads(objects[path]);meta['order']='F';objects[path]=encoded_json(meta)
    else:objects['field.zarr/phi/0/0/0']+=b'x'
    with pytest.raises(ValueError):extract_range(recipe,[-61,-60],out,resume=True)
    assert {str(p.relative_to(out)):(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}==before


def paris4_config():
    root = ('https://huggingface.co/datasets/LimeGS/winding-field-PHerc.Paris.4-307.2um/'
            'resolve/6753c9812ec8c455401da69b4e758769e565e39f/')
    descriptor = dict(kind='zarr-v2', url=root+'field.json', support_policy='stored',
        sha256='8f148279de9fc0bc9a877c84e3209a7d0bda8d17cd9d7175820ce075512835be',
        complete_sha256='57df0ad0358b3f6963b165fd155f91b22da2d37f3d8b996e6a018afd7dc133e9')
    geometry = dict(frame='PHercParis4/20260411134726:L2',voxel_um=9.6,
        origin_xyz=[0.,0.,0.],spacing_xyz=[32.,32.,32.],shape_zyx=[593,256,256],sheet_step=1.192497988541921)
    return root, descriptor, geometry


@pytest.mark.skipif(os.environ.get('WF_REMOTE_ZARR_LIVE') != '1', reason='opt-in published Paris4 network acceptance')
def test_published_paris4_crop_matches_original_npy(monkeypatch):
    from pathlib import Path
    from winding_field.surface_sources import FieldSource
    m = api()
    root, descriptor, geometry = paris4_config()
    fetch = m.fetch
    objects = []
    def logged_fetch(url, *, limit):
        raw = fetch(url, limit=limit)
        objects.append((url.removeprefix(root),len(raw)))
        return raw
    monkeypatch.setattr(m,'fetch',logged_fetch)
    with FieldSource(descriptor,geometry,cache_bytes=32<<20) as source:
        objects.clear()
        values,valid=source.read((64,64,64),(66,66,66))
    assert sorted(name for name,_ in objects)==[
        'field.zarr/phi/0/0/0','field.zarr/valid/0/0/0','receipts/0/0/0.json']
    assert sum(size for _,size in objects)==5841880+320+873
    original=Path(os.environ['WF_PARIS4_ORIGINAL_NPY'])
    with original.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='ab3d3dc19d16eaac3ff7512aaa0ddfcc85ac3728c9cc594e03e07c8f7b2dd505'
    expected=np.load(original,mmap_mode='r')[64:66,64:66,64:66]
    np.testing.assert_array_equal(values,expected)
    np.testing.assert_array_equal(valid,np.isfinite(expected))


@pytest.mark.skipif(os.environ.get('WF_REMOTE_ZARR_MESH') != '1', reason='opt-in full Paris4 mesh acceptance')
def test_published_paris4_complete_mesh_and_completed_resume(tmp_path):
    from pathlib import Path
    from winding_field.surface_workflow import run_surfaces
    from winding_field.surface_parallel import run_parallel_levels
    _, descriptor, geometry=paris4_config()
    original=Path(os.environ['WF_PARIS4_ORIGINAL_NPY'])
    config=dict(schema='winding-surfaces-v1',field=descriptor,geometry=geometry,
        levels=[-60*1.192497988541921],execution=dict(tile=128,cache_bytes=64<<20))
    remote_config=tmp_path/'remote.json';remote_config.write_text(json.dumps(config))
    local=dict(config,field=dict(kind='npy',path=str(original),
        sha256='ab3d3dc19d16eaac3ff7512aaa0ddfcc85ac3728c9cc594e03e07c8f7b2dd505'))
    local_config=tmp_path/'local.json';local_config.write_text(json.dumps(local))
    assert run_surfaces(local_config,tmp_path/'local',render=False)['status']=='MESH_READY'
    collection=tmp_path/'remote'
    assert run_parallel_levels(remote_config,collection)['status']=='LEVELS_READY'
    local_obj=(tmp_path/'local/level-0000.obj').read_bytes()
    remote_obj=(collection/'level-0000/level-0000.obj').read_bytes()
    assert local_obj==remote_obj
    before={str(p.relative_to(collection)):(sha(p.read_bytes()),p.stat().st_mtime_ns)
            for p in collection.rglob('*') if p.is_file()}
    assert run_parallel_levels(remote_config,collection,resume=True)['status']=='LEVELS_READY'
    after={str(p.relative_to(collection)):(sha(p.read_bytes()),p.stat().st_mtime_ns)
           for p in collection.rglob('*') if p.is_file()}
    assert before==after
