"""Reproduce the pinned Villa model imports and provenance.

Run: python tools/vendor.py
Generated source files are checked in; installation never runs this tool.
"""

import ast
import hashlib
import json
from pathlib import Path
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
