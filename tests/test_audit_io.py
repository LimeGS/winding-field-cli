import gzip
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from test_io import make_volume


def test_http_gzip_decodes_and_limits_decoded_bytes(monkeypatch):
    import winding_field.io as reader
    raw=b'{"value":"'+b'x'*4096+b'"}'
    def response(*args,**kwargs):
        stream=io.BytesIO(gzip.compress(raw))
        stream.headers={'Content-Encoding':'gzip','Content-Length':str(len(stream.getvalue())),'ETag':'pinned'}
        return stream
    monkeypatch.setattr(reader,'urlopen',response)
    assert reader.fetch('https://test/catalog',limit=8192)==(raw,'pinned')
    with pytest.raises(ValueError,match='budget'):
        reader.fetch('https://test/catalog',limit=100)


def test_offline_mirror_uses_local_metadata_but_rejects_cached_conflict(tmp_path,monkeypatch):
    import winding_field.io as reader
    local=tmp_path/'local'; expected=make_volume(local)
    monkeypatch.setattr(reader,'fetch',lambda *a,**k:pytest.fail('offline network request'))
    url='https://test/ct.zarr'; cache=tmp_path/'cache'
    volume=reader.Volume(url,local=local,cache=cache,offline=True)
    np.testing.assert_array_equal(volume.read([0,0,0],[4,4,6]),expected)
    volume.close()
    from winding_field.util import digest
    conflict=json.loads((local/'0/.zarray').read_bytes()); conflict['shape']=[2,2,2]
    (cache/('meta-'+digest([url,'0/.zarray'])+'.json')).write_text(json.dumps(conflict))
    with pytest.raises(ValueError,match='metadata'):
        reader.Volume(url,local=local,cache=cache,offline=True)


def test_offline_catalog_is_cached_and_never_fetches(tmp_path,monkeypatch):
    import winding_field.io as reader
    catalog={'samples':{'scroll':{'volumes':{'scan':{'properties':{'pixel_size_um':8.64},'data':[
        {'type':'zarr','origins':[{'path':'ct.zarr','access_roots':[{'usage':'public-read','url':'https://test'}]}]}]}}}}}
    monkeypatch.setattr(reader,'fetch',lambda *a,**k:(json.dumps(catalog).encode(),None))
    selected=reader.resolve_source('scroll',cache=tmp_path)
    monkeypatch.setattr(reader,'fetch',lambda *a,**k:pytest.fail('offline network request'))
    assert reader.resolve_source('scroll',cache=tmp_path,offline=True)==selected
    with pytest.raises(FileNotFoundError,match='catalog|source'):
        reader.resolve_source('scroll',cache=tmp_path/'empty',offline=True)


@pytest.mark.parametrize('option,value',[
    ('stride',0),('stride',-1),('stride',9),('field_chunk',3),('chunk',129),
    ('rtol',float('nan')),('sweeps',0),('seed_stride',0),('fibre_weight',-1),
    ('chart_faces',8193),('threads',0),('levels',[float('inf')]),('levels',[1.,1.]),('field_spacing_um',0),
])
def test_plan_and_run_validate_before_source_io(tmp_path,monkeypatch,option,value):
    import winding_field.workflow as workflow
    monkeypatch.setattr(workflow,'open_volume',lambda *a:pytest.fail('source opened before validation'))
    config=dict(source='missing',output=str(tmp_path/'run'),**{option:value})
    for operation in (workflow.plan,workflow.run):
        with pytest.raises(ValueError):
            operation(config)
    assert not (tmp_path/'run').exists()


def test_plan_checks_cache_filesystem_separately(tmp_path,monkeypatch):
    import winding_field.workflow as workflow
    cache=tmp_path/'cache'; cache.mkdir()
    output=tmp_path/'output'; output.mkdir()
    monkeypatch.setattr(workflow.shutil,'disk_usage',lambda p:SimpleNamespace(free=512<<10 if Path(p)==cache else 1<<30))
    volume=SimpleNamespace(shape=(128,)*3,chunks=(128,)*3,dtype=np.dtype('u1'),
        spacing_xyz=[1.]*3,origin_xyz=[0.]*3,identity={},remote=True,limit=8<<20,cache=cache)
    config=dict(output=str(output),reserve_bytes=0)
    assert workflow.plan(config,volume)['ct_policy']=='stream'
    with pytest.raises(ValueError,match='download'):
        workflow.plan(dict(config,ct_mode='download'),volume)


def test_cache_write_admission_preserves_reserve(tmp_path,monkeypatch):
    import winding_field.io as reader
    local=tmp_path/'source'; make_volume(local)
    monkeypatch.setattr(reader,'fetch',lambda url,**k:((local/url.split('ct.zarr/')[1]).read_bytes(),None))
    v=reader.Volume('https://test/ct.zarr',cache=tmp_path/'cache',reserve_bytes=100)
    monkeypatch.setattr(reader.shutil,'disk_usage',lambda p:SimpleNamespace(free=105))
    with pytest.raises(OSError,match='cache|space'):
        v.read([0,0,0],[2,2,3])
    assert v.stats()['cache_bytes']==0
    assert not list((tmp_path/'cache').glob('[0-9a-f]'*64))
    v.close()


def test_concurrent_checkpoint_downloads_use_unique_verified_temporaries(tmp_path,monkeypatch):
    import winding_field.models as models
    payload=b'pinned model bytes'; barrier=threading.Barrier(2)
    monkeypatch.setattr(models,'MODEL_SHA',hashlib.sha256(payload).hexdigest())
    def fetch(*a,**k):
        barrier.wait(timeout=10)
        return io.BytesIO(payload)
    monkeypatch.setattr(models,'urlopen',fetch)
    verify=models.verify_checkpoint
    verified=[]
    def check(path):
        verify(path)
        if path.name!='sd2.pt':
            verified.append(path)
            barrier.wait(timeout=10)
    monkeypatch.setattr(models,'verify_checkpoint',check)
    path=tmp_path/'sd2.pt'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:models.checkpoint(path),range(2)))
    assert results==[path,path] and path.read_bytes()==payload
    assert len(set(verified))==2
    assert sorted(p.name for p in tmp_path.iterdir())==['sd2.pt']
