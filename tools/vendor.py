"""Reproduce namespaced imports, preserving upstream mathematics and provenance.

Run: python tools/vendor.py --core /path/to/winding-field-gh
Generated source files are checked in; installation never runs this tool.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIN = 'b29f310957c7b29e7d332f9876e0e487819cfaa3'


def write_package(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    if path.suffix == '.py':
        ast.parse(data)
        parent=path.parent
        while parent != ROOT/'src':
            init=parent/'__init__.py'
            if not init.exists():
                init.write_text('')
            parent=parent.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--core',type=Path,required=True)
    args=parser.parse_args()
    source=args.core/'src'
    pending=['field_observations','block_solver','surface_mesh','surface_sources']
    files={}
    while pending:
        name=pending.pop()
        if name in files:
            continue
        raw=(source/(name+'.py')).read_bytes()
        files[name]=raw
        for node in ast.walk(ast.parse(raw)):
            imports=([node.module] if isinstance(node,ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            pending.extend(n for n in imports if n and (source/(n+'.py')).exists() and n not in files)
    provenance={}
    for name,raw in files.items():
        text=raw.decode()
        for module in files:
            text=re.sub(r'(?m)^(\s*)from '+module+r' import ',r'\1from .'+module+' import ',text)
            text=re.sub(r'(?m)^(\s*)import '+module+r'$',r'\1from . import '+module,text)
        write_package(ROOT/'src/winding_field/_core'/f'{name}.py',text.encode())
        provenance[name+'.py']=hashlib.sha256(raw).hexdigest()
    write_package(ROOT/'licenses/winding-field-MIT.txt',(args.core/'LICENSE').read_bytes())
    write_package(ROOT/'provenance/core.json',json.dumps({'origin':'winding-field-gh working tree',
                  'changes':'namespace imports only','source_sha256':provenance},indent=2).encode())
    modules=['neural_tracing/fiber_trace_3d/model.py','neural_tracing/fiber_trace_3d/direction.py',
             'neural_tracing/fiber_trace/model.py','neural_tracing/nets/vesuvius_unet3d.py',
             'models/utilities/utils.py']
    provenance={}
    for module in modules:
        url=f'https://raw.githubusercontent.com/ScrollPrize/villa/{PIN}/vesuvius/src/vesuvius/{module}'
        with urllib.request.urlopen(url,timeout=60) as response:
            raw=response.read()
        text=raw.decode().replace('from vesuvius.', 'from winding_field._vendor.vesuvius.')
        write_package(ROOT/'src/winding_field/_vendor/vesuvius'/module,text.encode())
        provenance[module]=hashlib.sha256(raw).hexdigest()
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/ScrollPrize/villa/{PIN}/LICENSE',timeout=60) as r:
        write_package(ROOT/'licenses/villa-MIT.txt',r.read())
    write_package(ROOT/'provenance/villa.json',json.dumps({'revision':PIN,'url':'https://github.com/ScrollPrize/villa',
                  'changes':'namespace imports only; empty namespace initializers',
                  'source_sha256':provenance},indent=2).encode())
    assert len(provenance)==5


if __name__=='__main__':
    main()
