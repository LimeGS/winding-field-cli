"""Managed streaming validates URLs and chunk bounds before native reads."""
from contextlib import closing
from email.message import Message
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
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
    monkeypatch.setattr('winding_field.stream_policy._PinnedHTTPSHandler.https_open',request)
    return state


def test_execution_fetch_rejects_private_redirect_before_request(hf_network):
    from winding_field.io import fetch
    with pytest.raises(ValueError,match='unsafe streaming URL'):
        fetch(HF,stream_policy=True)
    assert hf_network['requests']==[HF]


def test_managed_reader_metadata_uses_protected_fetch(tmp_path,hf_network):
    from test_lasagna_stream import fixture
    from winding_field.lasagna_input import LasagnaInput
    ct,root,config=fixture(tmp_path)
    manifest=json.loads((root/'input.lasagna.json').read_bytes())
    manifest['required_transport']='vc_reader.DiskCache'
    for group in manifest['groups'].values():group['zarr']=HF.rsplit('/',1)[0]
    atomic_json(root/'input.lasagna.json',manifest)
    with closing(Volume(ct,cache=tmp_path/'ct-cache')) as volume:
        with pytest.raises(ValueError,match='unsafe streaming URL'):
            LasagnaInput(volume,tmp_path/'run',config)
    assert len(hf_network['requests'])==1


@pytest.mark.parametrize('warm',[False,True])
def test_native_stream_cold_and_revalidation_block_private_redirect(tmp_path,hf_network,warm):
    from winding_field.ct_reader import EncodedTransport
    with closing(EncodedTransport(tmp_path/'cache',1<<20,stream_policy=True)) as transport:
        if warm:
            hf_network.update(redirect=False,body=b'channel')
            assert transport.read(HF)==b'channel'
            transport.new_epoch()
            hf_network.update(redirect=True,requests=[])
        with pytest.raises(ValueError,match='unsafe streaming URL'):transport.read(HF)
        assert hf_network['requests']==[HF]
        if not warm:
            assert not transport.path(HF).exists()
            assert transport.db.execute('SELECT count(*) FROM receipts').fetchone()[0]==0


@pytest.mark.parametrize('uppercase_only',[False,True])
def test_native_loopback_preserves_environment_proxy_exclusions(tmp_path,hf_network,monkeypatch,uppercase_only):
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
    finally:proxy.shutdown();proxy.server_close();thread.join()


def test_slow_upstream_finishes_before_native_loopback_timer(tmp_path,hf_network):
    from winding_field.ct_reader import EncodedTransport
    hf_network.update(redirect=False,body=bytes(2<<20),delay=12)
    with closing(EncodedTransport(tmp_path/'cache',4<<20,stream_policy=True)) as transport:
        assert transport.read(HF)==bytes(2<<20)
        assert transport.path(HF).stat().st_size==2<<20
        assert hf_network['requests']==[HF]


@pytest.mark.parametrize('decoded',[False,True])
def test_oversized_blosc_header_is_rejected_before_decode(tmp_path,monkeypatch,decoded):
    root=write_ct(tmp_path/'ct',np.zeros((8,8,8),np.uint8),chunk=8)
    metadata=json.loads((root/'0/.zarray').read_bytes());metadata['compressor']=Blosc(cname='zstd').get_config()
    atomic_json(root/'0/.zarray',metadata)
    (root/'0/0.0.0').write_bytes(Blosc(cname='zstd').encode(bytes(16<<20)))
    def forbidden(*args,**kwargs):pytest.fail('Oversized header reached decompressor')
    monkeypatch.setattr(Blosc,'decode',forbidden)
    with closing(Volume(root,cache=tmp_path/'cache',stream_policy=True)) as volume:
        with pytest.raises(ValueError,match='chunk|Blosc'):
            if decoded:volume.decoded_chunk('0/0.0.0')
            else:volume.read([0,0,0],[8,8,8])


@pytest.mark.parametrize('codec',[None,Blosc(cname='zstd')])
@pytest.mark.parametrize('decoded',[False,True])
@pytest.mark.parametrize('order',['C','F'])
def test_bounded_raw_blosc_and_layout_preserve_samples(tmp_path,codec,decoded,order):
    data=np.arange(512,dtype=np.uint8).reshape(8,8,8)
    root=write_ct(tmp_path/'ct',data,chunk=8)
    metadata=json.loads((root/'0/.zarray').read_bytes())
    metadata.update(compressor=None if codec is None else codec.get_config(),order=order)
    atomic_json(root/'0/.zarray',metadata)
    raw=data.tobytes(order=order)
    (root/'0/0.0.0').write_bytes(raw if codec is None else codec.encode(raw))
    with closing(Volume(root,cache=tmp_path/'cache',stream_policy=True)) as volume:
        array=volume.decoded_chunk('0/0.0.0') if decoded else volume.read([0,0,0],[8,8,8])
        np.testing.assert_array_equal(array,data)


def test_raw_metadata_pin_rejects_even_semantically_equal_changes(tmp_path):
    import hashlib
    root=write_ct(tmp_path/'ct',np.ones((8,8,8),np.uint8),chunk=8)
    path=root/'0/.zarray';original=path.read_bytes();pin=hashlib.sha256(original).hexdigest()
    path.write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='metadata'):
        Volume(root,cache=tmp_path/'cache',stream_policy=True,metadata_pins={str(path):pin})


def test_rejected_online_metadata_preserves_pinned_offline_cache(tmp_path,monkeypatch):
    import hashlib
    from winding_field import io as volume_io
    from winding_field.util import digest
    raw=b'{"verified":true}'
    volume=object.__new__(Volume)
    volume.source=HF.rsplit('/',1)[0];volume.cache=tmp_path
    volume.remote=True;volume.offline=False;volume.local=None;volume.stream_policy=True
    volume.metadata_pins={volume.source+'/.zattrs':hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(volume_io,'fetch',lambda *a,**k:(raw,None))
    assert volume._metadata('.zattrs')=={'verified':True}
    monkeypatch.setattr(volume_io,'fetch',lambda *a,**k:(b'{"verified":false}',None))
    with pytest.raises(ValueError,match='metadata'):volume._metadata('.zattrs')
    volume.offline=True
    assert volume._metadata('.zattrs')=={'verified':True}
    assert (tmp_path/('meta-'+digest([volume.source,'.zattrs'])+'.json')).read_bytes()==raw


def test_stream_connection_uses_validated_dns_address(monkeypatch):
    import socket
    from winding_field.stream_policy import open_stream
    resolutions=[];connections=[]
    def resolve(host,port,*args,**kwargs):
        resolutions.append(host)
        address='1.1.1.1' if len(resolutions)<=2 else '127.0.0.1'
        return [(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP,'',(address,443))]
    def connect(sock,address):
        connections.append(address)
        raise RuntimeError('stop before network I/O')
    monkeypatch.setattr(socket,'getaddrinfo',resolve)
    monkeypatch.setattr(socket.socket,'connect',connect)
    with pytest.raises(RuntimeError,match='stop before network'):open_stream(HF,{})
    assert connections==[('1.1.1.1',443)]


def test_pinned_redirect_preserves_tls_hostname_and_http_host(monkeypatch):
    import socket
    import ssl
    from winding_field.stream_policy import open_stream
    hosts=['huggingface.co','cdn.hf.co'];resolved=[];connected=[];tls=[];requests=[]
    responses=[b'HTTP/1.1 302 Found\r\nLocation: https://cdn.hf.co/chunk\r\nContent-Length: 0\r\n\r\n',
               b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK']
    def resolve(host,port,**kwargs):
        resolved.append(host)
        return [(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP,'',(f'1.1.1.{len(resolved)}',443))]
    class Socket:
        def __init__(self,*args):self.response=responses.pop(0)
        def settimeout(self,value):pass
        def connect(self,address):connected.append(address)
        def setsockopt(self,*args):pass
        def sendall(self,data):requests.append(data)
        def makefile(self,*args):return io.BytesIO(self.response)
        def close(self):pass
    def wrap(context,sock,server_hostname,**kwargs):
        assert context.verify_mode==ssl.CERT_REQUIRED and context.check_hostname
        tls.append(server_hostname);return sock
    monkeypatch.setattr(socket,'getaddrinfo',resolve)
    monkeypatch.setattr(socket,'socket',Socket)
    monkeypatch.setattr(ssl.SSLContext,'wrap_socket',wrap)
    with open_stream(HF,{}) as response:assert response.read()==b'OK'
    assert resolved==tls==hosts
    assert connected==[('1.1.1.1',443),('1.1.1.2',443)]
    for host,request in zip(hosts,requests):assert ('Host: '+host+'\r\n').encode() in request
