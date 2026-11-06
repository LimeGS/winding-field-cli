"""Installed stages for the source-preserving constant-W sheet recipe."""
import argparse
import json
from pathlib import Path

from . import flatboi_checkpoint
from .sheet_atlas import _json
from .sheet_finish import render, validate
from .sheet_prepare import load_prepared, prepare


def main(argv=None):
    parser=argparse.ArgumentParser(prog='winding-field constant-sheet',description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare',help='select constant W, cut exact mesh and initialize positive UV')
    prep.add_argument('--mesh',type=Path,required=True);prep.add_argument('--w',type=float,required=True)
    prep.add_argument('--centerline',type=Path,required=True,help='JSON array of XYZ points with increasing Z')
    prep.add_argument('--output',type=Path,required=True)
    for name in ('flatten','import-native'):
        cmd=commands.add_parser(name,help='checkpointed Flatboi' if name=='flatten' else 'copy a verified historical checkpoint into a fresh run')
        cmd.add_argument('--prepared',type=Path,required=True);cmd.add_argument('--output',type=Path,required=True)
        cmd.add_argument('--app-run',type=Path,required=True)
        if name=='flatten':
            cmd.add_argument('--iterations',type=int,default=200);cmd.add_argument('--batch-iterations',type=int,default=10)
            cmd.add_argument('--hours',type=float,default=12.)
        else:
            cmd.add_argument('--native',type=Path,required=True);cmd.add_argument('--runner-sha256',required=True)
    valid=commands.add_parser('validate',help='rebind native XYZ/source IDs; calibrate UV and audit overlaps')
    valid.add_argument('--prepared',type=Path,required=True);valid.add_argument('--native',type=Path,required=True)
    valid.add_argument('--output',type=Path,required=True);valid.add_argument('--reference-z',type=float,required=True)
    valid.add_argument('--centerline',type=Path,help='required for legacy prepared artifacts; must match new preparation')
    valid.add_argument('--mesh',type=Path,help='relocated original mesh; original manifest hash must match')
    ct=commands.add_parser('render',help='render19.2um CT and publish a verified atlas sheet')
    ct.add_argument('--validated',type=Path,required=True);ct.add_argument('--output',type=Path,required=True)
    ct.add_argument('--config',type=Path,required=True,help='explicit CT source, frame and cache settings JSON')
    status=commands.add_parser('status',help='read-only native progress (use watch for a live monitor)')
    status.add_argument('--native',type=Path,required=True);status.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':result=prepare(args.mesh,args.output,args.w,_json(args.centerline))
        elif args.command in ('flatten','import-native'):
            load_prepared(args.prepared)
            args.source_map=args.prepared/'initial-map.npz';args.input=args.prepared/'inputmesh.obj'
            args.native_recipe='installed-2f89bfc9'
            if args.command=='flatten':result=flatboi_checkpoint.run(args)
            else:result=flatboi_checkpoint.import_checkpoint(args.native,args.output,args.source_map,args.input,args.app_run,args.runner_sha256)
        elif args.command=='validate':
            result=validate(args.prepared,args.native,args.output,_json(args.centerline) if args.centerline else None,args.reference_z,mesh_source=args.mesh)
        elif args.command=='render':result=render(args.validated,args.output,_json(args.config))
        else:
            result=flatboi_checkpoint.observe(args.native)
            if not args.json:
                print(f'FLATBOI — {result["status"]}\nCalculadas: {result["calculated"]} / {result["target"]}\n'
                      f'Guardadas para retomar: {result["saved"]}\nEnergía: {result["energy"]}\nConvergencia: {result["converged"]}')
                return 0
        print(json.dumps(result,allow_nan=False),flush=True);return 0
    except (ValueError,KeyError,OSError,AssertionError,RuntimeError) as error:
        parser.exit(75 if isinstance(error,InterruptedError) else 2,f'constant-sheet: {error}\n')


if __name__=='__main__':raise SystemExit(main())
