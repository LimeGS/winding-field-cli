"""Installed command entry point."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .io import Volume,resolve_source


def main(argv=None):
    parser = argparse.ArgumentParser(prog='winding-field')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    info = sub.add_parser('inspect', help='inspect exact original CT geometry')
    info.add_argument('source')
    info.add_argument('--voxel-um',type=float)
    info.add_argument('--level',type=int,default=0)
    for command in ('run','plan'):
        p=sub.add_parser(command,help='run the whole pipeline' if command=='run' else 'inspect source and space estimates')
        p.add_argument('--source'); p.add_argument('--scroll'); p.add_argument('--scan')
        p.add_argument('--output',required=True)
        p.add_argument('--local'); p.add_argument('--cache')
        p.add_argument('--cache-gib',type=float,default=2.)
        p.add_argument('--reserve-gib',type=float,default=1.)
        p.add_argument('--ct-mode',choices=['auto','download','stream'],default='auto')
        p.add_argument('--offline',action='store_true')
        p.add_argument('--voxel-um',type=float); p.add_argument('--level',type=int,default=0)
        p.add_argument('--roi',nargs=6,type=int,metavar=('Z0','Z1','Y0','Y1','X0','X1'))
        p.add_argument('--stride',type=int,default=1,help='CT observation sampling step; keep 1 for fine inference')
        p.add_argument('--field-spacing-um',type=float,help='exact isotropic field pitch in micrometres; default: observation pitch')
        p.add_argument('--chunk',type=int,default=64)
        p.add_argument('--prediction',choices=['classical','sd2','qual-v2'],default='qual-v2')
        p.add_argument('--model-path'); p.add_argument('--device',default='cpu')
        p.add_argument('--threshold',type=float)
        p.add_argument('--invalid-value',type=float,help='acquisition fill value to exclude; zero remains valid unless declared')
        p.add_argument('--field-chunk',type=int,default=64); p.add_argument('--sweeps',type=int,default=4)
        p.add_argument('--maxiter',type=int,default=2000); p.add_argument('--rtol',type=float,default=1e-6)
        p.add_argument('--seed-stride',type=int,default=8); p.add_argument('--fibre-weight',type=float,default=1.)
        p.add_argument('--levels',nargs='+',type=float); p.add_argument('--mesh-tile',type=int,default=64)
        p.add_argument('--chart-faces',type=int,default=1024); p.add_argument('--threads',type=int,default=4)
        p.add_argument('--no-previews',dest='previews',action='store_false')
        p.add_argument('--until',choices=['prepare','fibres','field','mesh','flatten'],default='flatten')
    for command in ('resume','status','prepare','fibres','field','mesh','flatten'):
        p=sub.add_parser(command,help='continue or inspect an existing run')
        p.add_argument('run')
    p=sub.add_parser('demo',help='offline synthetic original-CT demonstration')
    p.add_argument('--output',required=True)
    p.add_argument('--prediction',choices=['classical','sd2'],default='classical')
    p.add_argument('--model-path'); p.add_argument('--device',default='cpu')
    sub.add_parser('doctor',help='report installed runtime and CUDA availability')
    p=sub.add_parser('models',help='download and verify a pinned fibre checkpoint')
    p.add_argument('--path')
    p.add_argument('--prediction',choices=['qual-v2','sd2'],default='qual-v2')
    args=parser.parse_args(argv)
    try:
        if args.command=='inspect':
            volume=Volume(args.source,voxel_um=args.voxel_um,level=args.level)
            try:
                result=volume.identity
            finally:
                volume.close()
        elif args.command=='doctor':
            import platform,torch
            result=dict(version=__version__,python=platform.python_version(),platform=platform.platform(),
                        torch=torch.__version__,cuda=torch.cuda.is_available(),reader='portable-zarr-v2')
        elif args.command=='models':
            from .models import checkpoint,model_sha
            result=dict(path=str(checkpoint(args.path,prediction=args.prediction)),
                        prediction=args.prediction,sha256=model_sha(args.prediction))
        elif args.command in ('run','plan'):
            from .workflow import run,plan
            from .config import validate_config
            config=vars(args).copy(); command=config.pop('command'); until=config.pop('until')
            config['output']=str(Path(config['output']).expanduser().resolve())
            for key in ('local','cache','model_path'):
                if config.get(key):
                    config[key]=str(Path(config[key]).expanduser().resolve())
            for key in ('cache','reserve'):
                value=config.pop(key+'_gib')
                if not 0<=value<=1e6:
                    raise ValueError('invalid storage budget')
                config[key+'_bytes']=int(value*2**30)
            roi=config.pop('roi')
            if roi:
                config.update(roi_lo=roi[::2],roi_hi=roi[1::2])
            validate_config(config)
            selected=resolve_source(config.pop('scroll'),config.get('source'),config.pop('scan'),
                                    cache=config.get('cache'),offline=config['offline'])
            config['source']=selected['source']
            if config.get('voxel_um') is None:
                config['voxel_um']=selected.get('voxel_um')
            if not config['source'].startswith(('http://','https://','s3://')):
                config['source']=str(Path(config['source']).expanduser().resolve())
            result=plan(config) if command=='plan' else run(config,until=until)
        elif args.command=='demo':
            from .demo import synthetic
            from .workflow import run
            output=Path(args.output).expanduser().resolve()
            if output.exists():
                raise FileExistsError('demo output already exists; use resume')
            source=synthetic(output.with_name(output.name+'-ct.zarr'))
            config=dict(source=source,output=str(output),prediction=args.prediction,model_path=args.model_path,
                        device=args.device,chunk=64,field_chunk=64,sweeps=1,seed_stride=4,chart_faces=1024,
                        cache=str(output.with_name(output.name+'-cache')),reserve_bytes=0)
            result=run(config)
        else:
            from .workflow import run
            root=Path(args.run).expanduser().resolve()
            state=json.loads((root/'RUN.json').read_bytes())
            result=state if args.command=='status' else run(state['config'],resume=True,
                    until='flatten' if args.command=='resume' else args.command)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,RuntimeError) as error:
        print(f'winding-field: {error}',file=sys.stderr)
        return 1
