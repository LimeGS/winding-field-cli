"""Fail-fast installed-container checks; --gpu additionally requires a real GPU."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

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
from vc_reader.reader import open_zarr
from winding_field.lasagna import runtime_identity
from winding_field.patch_pipeline import check_flatboi

assert torch.__version__=='2.10.0+'+args.variant, 'unexpected CLI Torch build'
assert torch.version.cuda==('12.8' if args.variant=='cu128' else None)
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
