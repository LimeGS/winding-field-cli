"""Managed streaming enforces bounds before decoding untrusted channel bytes."""
from contextlib import closing
from email.message import Message
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
import urllib.request
from urllib.response import addinfourl

import numpy as np
from numcodecs import Blosc
import pytest

from ct_fixture import write_ct
from winding_field.io import Volume
from winding_field.util import atomic_json

HF='https://huggingface.co/buckets/SharpMeow/lasagna-fields/resolve/example/nx/2/0.0.0'


@pytest.fixture
def hf_network(monkeypatch):
    state=dict(redirect=True,requests=[],body=b'{}')
    monkeypatch.setattr('socket.getaddrinfo',lambda *a,**k:[(2,1,6,'',('1.1.1.1',443))])
    def request(handler,req):
        state['requests'].append(req.full_url)
        if len(state['requests'])==1 and state.get('delay'):time.sleep(state['delay'])
        headers=Message()
        if state['redirect']:
            headers['Location']='http://169.254.169.254/latest/meta-data/'
            response=addinfourl(io.BytesIO(b''),headers,req.full_url,302)
        else:
            headers['Content-Length']=str(len(state['body']))
            response=addinfourl(io.BytesIO(state['body']),headers,req.full_url,200)
        response.msg='response'
        return response
    monkeypatch.setattr(urllib.request.HTTPSHandler,'https_open',request)
    return state


def test_execution_fetch_rejects_private_redirect_before_request(hf_network):
    from winding_field.io import fetch
    with pytest.raises(ValueError,match='unsafe streaming URL'):
        fetch(HF,stream_policy=True)
    assert hf_network['requests']==[HF]


def test_managed_reader_metadata_uses_protected_fetch(tmp_path,hf_network):
    from test_lasagna_input import fixture
    from winding_field.lasagna_input import LasagnaInput
    ct,root,config=fixture(tmp_path)
    manifest=json.loads((root/'input.lasagna.json').read_bytes())
    manifest['required_transport']='vc_reader.DiskCache'
    for name,group in manifest['groups'].items():
        group['zarr']=HF.rsplit('/',1)[0]
    atomic_json(root/'input.lasagna.json',manifest)
    with closing(Volume(ct,cache=tmp_path/'ct-cache')) as volume:
        with pytest.raises(ValueError,match='unsafe streaming URL'):
            LasagnaInput(volume,tmp_path/'run',config)
    assert len(hf_network['requests'])==1


@pytest.mark.parametrize('warm',[False,True])
def test_native_stream_cold_and_revalidation_block_private_redirect(tmp_path,hf_network,warm):
    pytest.importorskip('vc_reader._reader')
    from winding_field.ct_reader import EncodedTransport
    with closing(EncodedTransport(tmp_path/'cache',1<<20,stream_policy=True)) as transport:
        if warm:
            hf_network.update(redirect=False,body=b'channel')
            assert transport.read(HF)==b'channel'
            transport.new_epoch()
            hf_network.update(redirect=True,requests=[])
        with pytest.raises(ValueError,match='unsafe streaming URL'):
            transport.read(HF)
        assert hf_network['requests']==[HF]
        if not warm:
            assert not transport.path(HF).exists()
            assert transport.db.execute('SELECT count(*) FROM receipts').fetchone()[0]==0


@pytest.mark.parametrize('uppercase_only',[False,True])
def test_native_loopback_cannot_be_redirected_through_environment_proxy(tmp_path,hf_network,monkeypatch,uppercase_only):
    pytest.importorskip('vc_reader._reader')
    from winding_field.ct_reader import EncodedTransport
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            raw=b'origin' if self.path=='/origin' else b'evil'
            self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    proxy=HTTPServer(('127.0.0.1',0),Proxy)
    thread=threading.Thread(target=proxy.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('http_proxy',f'http://127.0.0.1:{proxy.server_port}')
    monkeypatch.setenv('NO_PROXY','localhost' if uppercase_only else '')
    if uppercase_only:monkeypatch.delenv('no_proxy',raising=False)
    else:monkeypatch.setenv('no_proxy','')
    hf_network.update(redirect=False,body=b'channel')
    try:
        with closing(EncodedTransport(tmp_path/'cache',1<<20,stream_policy=True)) as transport:
            assert transport.read(HF)==b'channel'
            assert hf_network['requests']==[HF]
        if uppercase_only:
            with closing(EncodedTransport(tmp_path/'legacy-cache',1<<20)) as legacy:
                assert legacy.read(f'http://localhost:{proxy.server_port}/origin')==b'origin'
    finally:
        proxy.shutdown();proxy.server_close();thread.join()


def test_slow_upstream_download_is_not_limited_by_native_loopback_stall_timeout(tmp_path,hf_network):
    pytest.importorskip('vc_reader._reader')
    from winding_field.ct_reader import EncodedTransport
    hf_network.update(redirect=False,body=bytes(2<<20),delay=12)
    with closing(EncodedTransport(tmp_path/'cache',4<<20,stream_policy=True)) as transport:
        assert transport.read(HF)==bytes(2<<20)
        assert transport.path(HF).stat().st_size==2<<20
        assert hf_network['requests']==[HF]


@pytest.mark.parametrize('decoded',[False,True])
def test_oversized_blosc_header_is_rejected_before_decode(tmp_path,monkeypatch,decoded):
    root=write_ct(tmp_path/'ct',np.zeros((8,8,8),np.uint8),chunk=8)
    metadata=json.loads((root/'0/.zarray').read_bytes())
    metadata['compressor']=Blosc(cname='zstd').get_config()
    atomic_json(root/'0/.zarray',metadata)
    (root/'0/0.0.0').write_bytes(Blosc(cname='zstd').encode(bytes(16<<20)))
    def forbidden(*args,**kwargs):
        pytest.fail('Oversized header reached the decompressor')
    monkeypatch.setattr(Blosc,'decode',forbidden)
    with closing(Volume(root,cache=tmp_path/'cache',stream_policy=True)) as volume:
        with pytest.raises(ValueError,match='chunk|Blosc'):
            if decoded:volume.decoded_chunk('0/0.0.0')
            else:volume.read([0,0,0],[8,8,8])


@pytest.mark.parametrize('codec',[None,Blosc(cname='zstd')])
@pytest.mark.parametrize('decoded',[False,True])
def test_bounded_raw_and_blosc_reads_preserve_samples(tmp_path,codec,decoded):
    root=write_ct(tmp_path/'ct',np.full((8,8,8),37,np.uint8),chunk=8)
    metadata=json.loads((root/'0/.zarray').read_bytes())
    metadata['compressor']=None if codec is None else codec.get_config()
    atomic_json(root/'0/.zarray',metadata)
    (root/'0/0.0.0').write_bytes(bytes([37])*512 if codec is None else codec.encode(bytes([37])*512))
    with closing(Volume(root,cache=tmp_path/'cache',stream_policy=True)) as volume:
        array=volume.decoded_chunk('0/0.0.0') if decoded else volume.read([0,0,0],[8,8,8])
        assert array.shape==(8,8,8) and (array==37).all()
