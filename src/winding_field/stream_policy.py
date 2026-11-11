"""Trust boundary for Studio-managed HF channels; legacy reads are unchanged."""
import ipaddress
import math
from functools import partial
from http.client import HTTPSConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import secrets
import socket
import struct
import threading
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

import numpy as np
from numcodecs import Blosc
from numcodecs.blosc import list_compressors

BUCKET='https://huggingface.co/buckets/SharpMeow/lasagna-fields/resolve/'


def validate_url(url,*,redirect=False):
    parsed=urlsplit(url);host=parsed.hostname or ''
    allowed=(host in ('huggingface.co','hf.co') or host.endswith(('.huggingface.co','.hf.co'))) if redirect else url.startswith(BUCKET)
    if (not allowed or parsed.scheme!='https' or parsed.username or parsed.password
            or parsed.port not in (None,443) or parsed.fragment or (parsed.query and not redirect)
            or '%' in parsed.path or '\\' in url or any(p in ('.','..') for p in parsed.path.split('/'))):
        raise ValueError('unsafe streaming URL')
    addresses=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('unsafe streaming URL address')
    return addresses


class _PinnedHTTPSConnection(HTTPSConnection):
    def __init__(self,*args,addresses,**kwargs):
        super().__init__(*args,**kwargs)
        # Keep the hostname for Host/SNI/certificate checks; never resolve it again.
        def connect(address,timeout,source_address=None):
            for family,kind,protocol,_,destination in addresses:
                sock=socket.socket(family,kind,protocol)
                try:
                    sock.settimeout(timeout)
                    if source_address:sock.bind(source_address)
                    sock.connect(destination)
                    return sock
                except OSError as error:
                    sock.close();last_error=error
                except BaseException:
                    sock.close();raise
            raise last_error
        self._create_connection=connect


class _PinnedHTTPSHandler(HTTPSHandler):
    def __init__(self,addresses):
        super().__init__();self.addresses=addresses

    def https_open(self,request):
        return self.do_open(partial(_PinnedHTTPSConnection,addresses=self.addresses),
                            request,context=self._context)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def open_stream(url,headers):
    for hop in range(6):
        addresses=validate_url(url,redirect=hop>0)
        opener=build_opener(ProxyHandler({}),_NoRedirect(),_PinnedHTTPSHandler(addresses))
        try:
            return opener.open(Request(url,headers={**headers,'Accept-Encoding':'identity'}),timeout=60)
        except HTTPError as error:
            if error.code not in (301,302,303,307,308):raise
            url=urljoin(url,error.headers.get('Location',''));error.close()
    raise ValueError('streaming redirect limit exceeded')


class ChunkRelay:
    """Only validated bytes reach the native downloader's loopback hop."""
    def __init__(self):
        owner=self
        self.token='/'+secrets.token_urlsafe(32);self.raw=None
        class Handler(BaseHTTPRequestHandler):
            timeout=60
            def log_message(self,*args):pass
            def do_GET(self):
                raw=owner.raw
                if self.path!=owner.token or raw is None:
                    self.send_error(403);return
                try:
                    self.send_response(200)
                    self.send_header('Content-Length',str(len(raw)))
                    self.end_headers();self.wfile.write(raw)
                except (BrokenPipeError,ConnectionResetError):pass
        self.server=HTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def location(self,url):
        from .io import fetch
        # Complete upstream I/O before the native downloader's shorter stall timer.
        self.raw,_=fetch(url,stream_policy=True)
        if self.raw is None:raise ValueError('unexpected streaming 304')
        effective=os.environ.get('no_proxy') or os.environ.get('NO_PROXY','')
        for name in ('no_proxy','NO_PROXY'):
            hosts=(os.environ.get(name) or effective).split(',')
            if '127.0.0.1' not in hosts:os.environ[name]=','.join([*filter(None,hosts),'127.0.0.1'])
        return f'http://127.0.0.1:{self.server.server_port}{self.token}'

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join()


def validate_array(metadata):
    chunks=metadata.get('chunks')
    if (metadata.get('dtype')!='|u1' or metadata.get('order') not in ('C','F')
            or metadata.get('dimension_separator','.') not in ('.','/')
            or metadata.get('filters') not in (None,[])
            or not isinstance(chunks,list) or len(chunks)!=3
            or any(type(n) is not int or n<1 for n in chunks) or math.prod(chunks)>256<<20):
        raise ValueError('invalid streaming channel layout or chunk budget')
    codec=metadata.get('compressor')
    if codec is not None and (
            not isinstance(codec,dict) or codec.get('id')!='blosc'
            or set(codec)-{'id','cname','clevel','shuffle','blocksize'}
            or codec.get('cname','lz4') not in list_compressors()
            or type(codec.get('clevel',5)) is not int or not 0<=codec.get('clevel',5)<=9
            or type(codec.get('shuffle',1)) is not int or codec.get('shuffle',1) not in (0,1,2)
            or type(codec.get('blocksize',0)) is not int or not 0<=codec.get('blocksize',0)<=256<<20):
        raise ValueError('streaming requires raw or validated Blosc channels')


def decode_chunk(raw,metadata):
    expected=math.prod(metadata['chunks']);codec=metadata.get('compressor')
    if codec is None:
        if len(raw)!=expected:raise ValueError('corrupt streaming raw chunk')
        return raw
    if len(raw)<16 or len(raw)>expected+16:raise ValueError('invalid Blosc chunk size')
    version,_,_,typesize,nbytes,blocksize,cbytes=struct.unpack('<BBBBIII',raw[:16])
    if (version not in (1,2) or typesize!=1 or nbytes!=expected or cbytes!=len(raw)
            or not 0<blocksize<=expected):
        raise ValueError('Blosc chunk header differs from declared chunk budget')
    output=np.empty(expected,dtype=np.uint8)
    Blosc(**{k:v for k,v in codec.items() if k!='id'}).decode(raw,out=output)
    return output
