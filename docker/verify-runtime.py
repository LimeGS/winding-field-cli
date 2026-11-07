"""Fail-fast installed-container checks; --gpu additionally requires a real GPU."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant',choices=('cpu','cu128'),required=True)
parser.add_argument('--gpu',action='store_true')
args=parser.parse_args()
assert platform.system()=='Linux' and platform.machine()=='x86_64', 'linux/amd64 required'
assert sys.version_info[:3]==(3,12,12), 'unexpected CLI Python'

import igl
import matplotlib
import torch
from vc_reader import _reader
from vc_reader.reader import ObjectStore, decode_chunk
from winding_field.lasagna import runtime_identity
from winding_field.patch_pipeline import check_flatboi

assert torch.__version__=='2.10.0+'+args.variant, 'unexpected CLI Torch build'
assert torch.version.cuda==('12.8' if args.variant=='cu128' else None)
reader_source=json.loads(metadata.distribution('vc-reader').read_text('direct_url.json'))
assert reader_source['url']=='https://github.com/LimeGS/villa/archive/84207424c3a988e6cafe6b50fe7d4d7e5f839929.tar.gz'
import numpy as np
from numcodecs import Delta, Shuffle, Zlib
from winding_field.io import Volume
from winding_field.original_pipeline import sample_ct
from vc_reader.reader.disk_cache import disk_budget
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);source=root/'ct';source.mkdir()
    data=np.arange(24,dtype='>u2').reshape(2,3,4)
    filters=[Delta(dtype='>u2'),Shuffle(elementsize=2)];compressor=Zlib()
    meta=dict(zarr_format=2,shape=data.shape,chunks=data.shape,dtype=data.dtype.str,order='F',
              fill_value=0,compressor=compressor.get_config(),filters=[f.get_config() for f in filters])
    raw=data.tobytes(order='F')
    for codec in filters:raw=codec.encode(raw)
    raw=compressor.encode(raw)
    (source/'.zarray').write_text(json.dumps(meta));(source/'0.0.0').write_bytes(raw)
    assert ObjectStore(source,auth=None,max_read_bytes=len(raw)).read('0.0.0')==raw
    assert decode_chunk(raw,meta)==data.tobytes()
    cache=root/'publication';cache.mkdir();payload=cache/'chunk'
    disk_budget(str(cache),1024).write(str(payload),np.frombuffer(raw,np.uint8))
    assert payload.read_bytes()==raw
    volume=Volume(source,voxel_um=1,cache=root/'cache',cache_bytes=0)
    try:
        np.testing.assert_array_equal(volume.read([0,1,1],[2,3,4]),data[:,1:,1:])
        values,valid=sample_ct(volume,np.array([[1.5,1.5,.5]]))
        assert valid.tolist()==[True] and values.tolist()==[13.5]
    finally:volume.close()
for dist in metadata.distributions():
    direct=json.loads(dist.read_text('direct_url.json') or '{}')
    assert not direct.get('dir_info',{}).get('editable'), dist.metadata['Name']+' is editable'

native=check_flatboi('/opt/vc3d/AppRun','historical')
lasagna=runtime_identity()
assert lasagna['python']=='3.14.0', 'unexpected Lasagna Python'
probe='''
import importlib.metadata as metadata,json,sys,torch
from lasagna.preprocess_cos_omezarr import LasagnaCosPredict3DAdapter
from lasagna.tiled_predict3d import _input_tile_to_device
for dist in metadata.distributions():
    direct=json.loads(dist.read_text('direct_url.json') or '{}')
    assert not direct.get('dir_info',{}).get('editable'), dist.metadata['Name']+' is editable'
if sys.argv[1]=='gpu':
    assert torch.cuda.is_available(), 'Lasagna CUDA device missing'
    assert (torch.ones(4,device='cuda')*2).sum().item()==8
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda)))
'''
result=subprocess.run([lasagna['executable'],'-I','-c',probe,'gpu' if args.gpu else 'imports'],
                      check=True,text=True,capture_output=True)
lasagna_torch=json.loads(result.stdout)
assert lasagna_torch==dict(torch='2.10.0+'+args.variant,cuda=torch.version.cuda)
if args.gpu:
    assert args.variant=='cu128' and torch.cuda.is_available(), 'CLI CUDA device missing'
    assert (torch.ones(4,device='cuda')*2).sum().item()==8

for backend in (('cpu','cuda') if args.variant=='cu128' else ('cpu',)):
    from winding_field.util import file_sha
    root=Path('/opt/accelerators')/backend
    manifest=json.loads((root/'accelerator.json').read_text())
    assert manifest['backend']==backend
    assert file_sha(root/manifest['library'])==manifest['library_sha256']
    if backend=='cpu' or args.gpu:
        env=dict(os.environ,WINDING_TIFXYZ_ACCELERATOR=str(root/'accelerator.json'))
        if backend=='cuda':
            uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0]
            env.update(CUDA_VISIBLE_DEVICES=uuid,WINDING_TIFXYZ_CUDA_DEVICE=uuid)
        subprocess.run([sys.executable,'-I','-c','''
import sys,numpy as np
from winding_field.tifxyz_predicates import Backend
rows=np.zeros((256,24),np.float64)
rows[:,6]=1;rows[:,13:15]=.25;rows[:,15]=1
with Backend(sys.argv[1]) as backend:
    assert (backend.classify(0,rows)==1).all()
''',backend],env=env,check=True)

print(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),
                     torch=torch.__version__,native=native,lasagna=lasagna,
                     gpu_checked=args.gpu),sort_keys=True))
