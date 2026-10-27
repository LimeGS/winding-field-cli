"""Installed command entry point."""

import argparse
import json

from . import __version__
from .io import Volume


def main(argv=None):
    parser = argparse.ArgumentParser(prog='winding-field')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    info = sub.add_parser('inspect', help='inspect exact original CT geometry')
    info.add_argument('source')
    info.add_argument('--voxel-um',type=float)
    info.add_argument('--level',type=int,default=0)
    args = parser.parse_args(argv)
    volume = Volume(args.source,voxel_um=args.voxel_um,level=args.level)
    try:
        print(json.dumps(volume.identity,indent=2))
    finally:
        volume.close()
