"""Installed command entry point."""

import argparse
import importlib
import json
from pathlib import Path
import signal
import sys

from . import __version__
from .io import Volume,resolve_source

TOOLS = {
    'constant-sheet': ('constant_sheet', 'prepare, flatten, validate and render exact constant-W sheets'),
    'sheet-atlas': ('sheet_atlas', 'package and query side-by-side constant-W CT sheets'),
    'surface-export': ('surface_export', 'export a level set, flatten and sample CT'),
    'surface-atlas': ('surface_atlas', 'build bounded surface charts and previews'),
    'surface-probe': ('surface_probe', 'freeze a prior seed and inspect its surface'),
    'surface-geometry-check': ('surface_geometry_check', 'export TIFXYZ and invoke a pinned geometry gate'),
    'ct-visuals': ('ct_visuals', 'render frozen CT planes and field contours'),
    'patch-visuals': ('patch_visuals', 'evaluate and plot human-patch residuals'),
    'run-watch': ('run_watch', 'supervise a job with timeout and storage limits'),
}


def tifxyz_exit_code(result):
    status=result.get('status')
    if status in ('READY','TIFXYZ_VERIFIED','COMPLETE','PAUSED'):
        return 0
    if status in ('PARTIAL','TIFXYZ_PARTIAL_VERIFIED'):
        return 2
    if status=='BLOCKED':
        reasons=result.get('tifxyz_result',result).get('reasons',[])
        if not reasons:
            return 1
        resource={'RUNTIME_UNSUPPORTED','RSS_OBSERVATION','CPU_AFFINITY',
                  'RTREE_UNAVAILABLE','BUDGET_ENVELOPE','WORKER_RESOURCE_EXIT','ACCELERATOR_UNAVAILABLE'}
        return 75 if any(r.get('code','') in resource or r.get('code','').endswith(
            ('_LIMIT','_BUDGET','_HEADROOM')) for r in reasons) else 2
    return 1


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='prepared':
        from .field_workflow import main as prepared_main
        return prepared_main(argv[1:])
    if argv and argv[0] in TOOLS:
        module = importlib.import_module('.' + TOOLS[argv[0]][0], __package__)
        return module.main(argv[1:])
    parser = argparse.ArgumentParser(prog='winding-field')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepared',help='build/resume prepared fields, mesh and render (prepared --help)')
    for command, (_, help_text) in TOOLS.items():
        sub.add_parser(command, help=help_text)
    info = sub.add_parser('inspect', help='inspect exact original CT geometry')
    info.add_argument('source',help='CT Zarr path or URL')
    info.add_argument('--voxel-um',type=float,help='micrometre scale for CT geometry lacking units; not field/output spacing')
    info.add_argument('--level',type=int,default=0,help='CT Zarr pyramid level (default: 0); not a winding W value')
    for command in ('run','plan'):
        p=sub.add_parser(command,help='run the whole pipeline' if command=='run' else 'inspect source and space estimates')
        p.add_argument('--source',help='CT Zarr path or URL; alternatively select --scroll/--scan'); p.add_argument('--scroll',help='public catalog scroll identifier'); p.add_argument('--scan',help='public catalog scan identifier')
        p.add_argument('--output',required=True,help='run directory for configuration, checkpoints and products')
        p.add_argument('--local',help='local mirror of source CT chunks'); p.add_argument('--cache',help='CT chunk cache directory')
        p.add_argument('--cache-gib',type=float,default=2.,help='CT chunk cache budget in GiB (default: 2); not total RAM')
        p.add_argument('--reserve-gib',type=float,default=1.,help='free disk reserve in GiB (default: 1); not RAM')
        p.add_argument('--ct-mode',choices=['auto','download','stream'],default='auto',help='CT access: auto chooses by disk estimate, download prefetches, stream reads on demand')
        p.add_argument('--offline',action='store_true',help='disable CT network access; require local/cached CT data')
        p.add_argument('--voxel-um',type=float,help='micrometre scale for CT geometry lacking units; not field/output spacing'); p.add_argument('--level',type=int,default=0,help='CT Zarr pyramid level (default: 0); not a winding W value')
        p.add_argument('--roi',nargs=6,type=int,metavar=('Z0','Z1','Y0','Y1','X0','X1'),help='half-open CT voxel bounds at the selected pyramid level; default: whole volume')
        p.add_argument('--stride',type=int,default=1,help='CT observation sampling step; keep 1 for fine inference')
        p.add_argument('--field-spacing-um',type=float,help='exact isotropic field pitch in micrometres; default: observation pitch')
        p.add_argument('--bounded-preparation',action='store_true',help='recompute native Lasagna/Qual windows; retain endpoint-aligned coarse observations and global fibres')
        p.add_argument('--bounded-gib',type=float,default=12.,help='bounded producer retained/scratch allowance, excluding CT cache and disk reserve (default: 12 GiB)')
        p.add_argument('--chunk',type=int,default=64,help='observation tile edge in sampled voxels (default: 64)')
        p.add_argument('--prediction',choices=['classical','sd2','qual-v2'],default='qual-v2',help='fibre predictor (default: qual-v2); learned predictors use Lasagna surfaces, classical uses CT intensity')
        p.add_argument('--model-path',help='local checkpoint for the selected fibre predictor'); p.add_argument('--device',default='cpu',help='PyTorch prediction/field device, e.g. cpu or cuda (default: cpu)')
        p.add_argument('--fibres',help='external AFV; replaces fibre inference/tracing')
        p.add_argument('--fibres-geometry',help='explicit AFV XYZ-to-world geometry JSON')
        p.add_argument('--lasagna-python',help='isolated official Python 3.14 runtime')
        p.add_argument('--lasagna-model-path',help='local pinned Lasagna surface checkpoint')
        p.add_argument('--lasagna-input',help='existing local Lasagna v2 surface manifest; skips Lasagna inference')
        p.add_argument('--lasagna-geometry',help='explicit Lasagna base-grid XYZ-to-world geometry JSON')
        p.add_argument('--threshold',type=float,help='classical predictor CT intensity threshold; default: calibrated Otsu; unused by learned predictors')
        p.add_argument('--invalid-value',type=float,help='acquisition fill value to exclude; zero remains valid unless declared')
        p.add_argument('--field-chunk',type=int,default=64,help='field solver block edge in field-grid samples (default: 64)'); p.add_argument('--sweeps',type=int,default=4,help='field solver block sweeps (default: 4)')
        p.add_argument('--maxiter',type=int,default=2000,help='iteration limit per local field solve (default: 2000)'); p.add_argument('--rtol',type=float,default=1e-6,help='relative tolerance of local field solves (default: 1e-6)')
        p.add_argument('--seed-stride',type=int,default=8,help='fibre seed grid spacing in observation samples (default: 8)'); p.add_argument('--fibre-weight',type=float,default=1.,help='field solver fibre constraint weight (default: 1)')
        p.add_argument('--levels',nargs='+',type=float,help='winding W isovalues to extract, e.g. --levels -31; not CT pyramid levels'); p.add_argument('--mesh-tile',type=int,default=64,help='mesh extraction tile edge in field-grid cells (default: 64)')
        p.add_argument('--chart-faces',type=int,default=1024,help='native/atlas local chart face budget (default: 1024); not original global flattening'); p.add_argument('--threads',type=int,default=4,help='PyTorch CPU thread count (default: 4); not a global worker limit')
        p.add_argument('--no-previews',dest='previews',action='store_false',help='skip legacy atlas chart previews; retain final CT renders')
        p.add_argument('--surface-output',choices=['native','atlas','tifxyz'],default='native',help='pipeline branch: native surfaces (default), legacy atlas, or experimental direct-field tifxyz; not an output file format')
        p.add_argument('--flattening-method',choices=['default','original'],default='default',help='original global LSCM; independent of field resolution')
        p.add_argument('--flattening-config',help='flattening JSON: output_frame, optional pitch and memory_gib')
        p.add_argument('--coverage-target',type=float,default=1.,help='native mapped mesh area target, fraction in(0,1]')
        p.add_argument('--surface-time-limit',dest='surface_time_limit_s',type=float,default=1800.,help='cumulative active seconds per native W')
        p.add_argument('--export-tifxyz',action='store_true',help='add mesh-derived XYZ TIFFs to native/original CT renders; not direct-field export')
        p.add_argument('--tifxyz-config',help='experimental direct-export JSON; frozen by content')
        p.add_argument('--until',choices=['prepare','fibres','field','mesh','flatten','assemble','render','tifxyz','original'],
                       help='default: terminal stage of the selected branch')
    for command in ('resume','status','prepare','fibres','field','mesh','flatten','assemble','render'):
        p=sub.add_parser(command,help='continue or inspect an existing run')
        p.add_argument('run')
        if command=='resume':
            p.add_argument('--field-sweeps',type=int,help='absolute total sweep budget for an unpublished bounded field; retries do not add sweeps')
            p.add_argument('--extra-surface-seconds',type=float,default=0.,help='explicit extra native budget; original elapsed/target retained')
    p=sub.add_parser('tifxyz',help='experimental direct export of an existing field; Linux runtime')
    p.add_argument('field_run',help='existing run directory containing the field'); p.add_argument('--output',required=True,help='direct-field TIFXYZ export directory')
    p.add_argument('--config',help='direct-field export request JSON; required for a new export'); p.add_argument('--resume',action='store_true',help='continue the saved export request in --output')
    p=sub.add_parser('verify-tifxyz',help='read-only verification in an admitted Linux worker')
    p.add_argument('output')
    p=sub.add_parser('extract-range',help='extract a calibrated continuous W range')
    p.add_argument('--config',required=True,help='immutable field/geometry/umbilicus recipe JSON')
    p.add_argument('--w-range',nargs=2,required=True,metavar=('A','B'),help='exact decimal bounds on the global W lattice')
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('join-ranges',help='join compatible contiguous continuous ranges exactly')
    p.add_argument('inputs',nargs='+')
    p.add_argument('--output',required=True)
    p=sub.add_parser('export-sheets',help='LSCM/SLIM on patches or whole sheets; historical recipes also available')
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--field-config',help='input field JSON: data path/hash and grid geometry; requires --levels')
    inputs.add_argument('--mesh-bundle',help='immutable mesh bundle; patches also accepts a RANGE wrapper (join ranges first)')
    p.add_argument('--levels',nargs='+',type=float,help='winding W isovalues, e.g. -31; required with --field-config, forbidden with --mesh-bundle')
    p.add_argument('--source',required=True,help='original CT Zarr')
    p.add_argument('--level',type=int,default=0,help='CT Zarr pyramid level (default: 0); not a winding W value');p.add_argument('--voxel-um',type=float,help='micrometre scale for CT geometry lacking units; not field/output spacing')
    p.add_argument('--output',required=True,help='sheet export directory');p.add_argument('--flattening-config',required=True,help='JSON: output_frame and render settings; explicit slim requires app_run, see docs/FLATTENING.md')
    p.add_argument('--flattening-method',choices=['original','patches'],help='historical recipe; cannot combine with --flattening-solver/--scope')
    p.add_argument('--flattening-solver',choices=['lscm','slim'],help='explicit algorithm; requires --scope')
    p.add_argument('--scope',choices=['patches','sheet'],help='independent patches or whole connected sheets; requires --flattening-solver')
    p.add_argument('--export-tifxyz',action='store_true',help='add XYZ TIFFs and validity mask alongside CT renders');p.add_argument('--resume',action='store_true',help='continue the same export; repeat identical inputs and settings')
    p=sub.add_parser('export-uvw',help='aligned diagnostic W band from one original global UV map')
    p.add_argument('--field-config',required=True,help='SHA-bound field descriptor JSON')
    p.add_argument('--reference-sheet',required=True,help='original/level-NNNN with independent publication ledger')
    p.add_argument('--levels',nargs='+',type=float,required=True,help='finite W targets; band must contain reference W')
    p.add_argument('--source',required=True,help='same physical CT source as reference')
    p.add_argument('--voxel-um',type=float,help='CT units when metadata lacks them')
    p.add_argument('--level',type=int,default=0,help='CT pyramid level, not winding W')
    p.add_argument('--canvas-samples',type=int,default=512,help='shared square UV grid edge, 2..1024 (default: 512)')
    p.add_argument('--workers',type=int,default=1,help='read-only geometry processes, 1..8; CT/publication stay in coordinator')
    p.add_argument('--cache-mib',type=int,default=256,help='total decoded CT + field-region RAM cache budget, 0..8192 MiB (default: 256)')
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true',help='resume identical source/settings; remove output/STOP to continue')
    p=sub.add_parser('demo',help='offline synthetic original-CT demonstration')
    p.add_argument('--output',required=True,help='synthetic demo run directory')
    p.add_argument('--prediction',choices=['classical','sd2'],default='classical',help='demo observation predictor (default: classical)')
    p.add_argument('--model-path',help='local SD2 checkpoint path'); p.add_argument('--device',default='cpu',help='PyTorch device (default: cpu)')
    sub.add_parser('doctor',help='report installed runtime and CUDA availability')
    p=sub.add_parser('models',help='download and verify a pinned checkpoint')
    p.add_argument('--path',help='checkpoint destination; default: model cache')
    p.add_argument('--prediction',choices=['qual-v2','sd2','lasagna'],default='qual-v2',help='checkpoint to download (default: qual-v2)')
    args=parser.parse_args(argv)
    direct=False; native=False
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
            from .workflow import run,plan,stage_path
            from .config import validate_config,read_tifxyz_request
            from .util import digest
            config=vars(args).copy(); command=config.pop('command'); until=config.pop('until')
            original=config.pop('flattening_config')
            if original:
                config['original_request']=json.loads(Path(original).read_bytes())
            if config['flattening_method']=='original':
                from .original_pipeline import settings
                settings(config.get('original_request'))
            elif original:
                raise ValueError('--flattening-config requires --flattening-method original')
            request=config.pop('tifxyz_config')
            if request:
                config['tifxyz_request']=read_tifxyz_request(request)
                config['tifxyz_request_sha256']=digest(config['tifxyz_request'])
            direct=config['surface_output']=='tifxyz' and command=='run'
            native=config['surface_output']=='native' and config['flattening_method']!='original' and command=='run'
            validate_config(config)
            stages=stage_path(config)
            until=until or stages[-1][0]
            if until not in dict(stages) and not (config['surface_output']=='native' and until in ('mesh','flatten','assemble','render')):
                raise ValueError('--until '+until+' is incompatible with the selected surface output')
            config['output']=str(Path(config['output']).expanduser().resolve())
            for key in ('local','cache','model_path','fibres','fibres_geometry','lasagna_model_path','lasagna_input','lasagna_geometry'):
                if config.get(key):
                    config[key]=str(Path(config[key]).expanduser().resolve())
            if config.get('lasagna_python'):
                # Resolving the venv executable symlink would select the base interpreter.
                config['lasagna_python']=str(Path(config['lasagna_python']).expanduser().absolute())
            for key in ('cache','reserve','bounded'):
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
        elif args.command=='export-uvw':
            from .uvw_pipeline import export_existing
            interrupted=False
            def handle_uvw(signum,frame):
                nonlocal interrupted
                interrupted=True
            previous={sig:signal.signal(sig,handle_uvw) for sig in (signal.SIGINT,signal.SIGTERM)}
            try:result=export_existing(args,stop=lambda:interrupted)
            finally:
                for sig,handler in previous.items():signal.signal(sig,handler)
        elif args.command=='extract-range':
            from .range_surface import extract_range
            result=extract_range(json.loads(Path(args.config).read_bytes()),args.w_range,args.output,resume=args.resume)
        elif args.command=='join-ranges':
            from .range_surface import join_ranges
            result=join_ranges(args.inputs,args.output)
        elif args.command=='export-sheets':
            if args.flattening_solver is not None or args.scope is not None:
                if args.flattening_method or not (args.flattening_solver and args.scope):
                    raise ValueError('flattening selection requires --flattening-solver and --scope together, without --flattening-method')
                from .matrix_pipeline import export_existing
            elif args.flattening_method=='patches':
                from .patch_pipeline import export_existing
            elif args.flattening_method=='original':
                from .original_pipeline import export_existing
            else:
                raise ValueError('flattening selection required: --flattening-solver with --scope, or historical --flattening-method')
            result=export_existing(args)
        elif args.command in ('tifxyz','verify-tifxyz'):
            from .tifxyz import _path,export_field,verify_export
            from .config import read_tifxyz_request,tifxyz_settings
            direct=True
            out=_path(Path(args.output).expanduser())
            interrupted=False
            def handle(signum,frame):
                nonlocal interrupted
                interrupted=True
            previous={sig:signal.signal(sig,handle) for sig in (signal.SIGINT,signal.SIGTERM)}
            stop=lambda:interrupted or (out/'STOP').exists()
            try:
                if args.command=='verify-tifxyz':
                    result=verify_export(out,stop=stop)
                    if result.get('status') not in ('TIFXYZ_VERIFIED','TIFXYZ_PARTIAL_VERIFIED','PAUSED','BLOCKED','FAILED'):
                        raise ValueError('unexpected tifxyz verification result')
                else:
                    source=_path(Path(args.field_run).expanduser())
                    if out==source or source in out.parents:
                        raise ValueError('standalone output must be outside the read-only field run')
                    request=read_tifxyz_request(args.config) if args.config else None
                    settings=tifxyz_settings(out,request,resume=args.resume)
                    result=export_field(source,out,settings,resume=args.resume,stop=stop)
            finally:
                for sig,handler in previous.items():
                    signal.signal(sig,handler)
            result=dict(result,experimental=True,comparison_status='UNMEASURED',comparison_reason='reference OOM')
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
            direct=state['config'].get('surface_output','atlas')=='tifxyz' and args.command!='status'
            native=state['config'].get('surface_output','atlas')=='native' and state['config'].get('flattening_method')!='original' and args.command!='status'
            if args.command=='status':
                result=state
                progress=root/'surfaces/STATUS.json'
                if progress.is_file():
                    result=dict(state,native_progress=json.loads(progress.read_bytes()))
            else:
                result=run(state['config'],resume=True,until=None if args.command=='resume' else args.command,
                           extra_surface_seconds=getattr(args,'extra_surface_seconds',0.),
                           field_sweeps=getattr(args,'field_sweeps',None))
        print(json.dumps(result,indent=2,allow_nan=False))
        if native:
            native_result=result.get('native_result',result)
            if result.get('status')=='PAUSED' and native_result.get('reason') in ('TIME_LIMIT','RSS_LIMIT','RSS_OBSERVATION','DISK_HEADROOM','SCRATCH_LIMIT'):
                return 75
            return 2 if result.get('status')=='PARTIAL' else 1 if result.get('status')=='FAILED' else 0
        return tifxyz_exit_code(result) if direct else 0
    except (ValueError,OSError,RuntimeError) as error:
        print(f'winding-field: {error}',file=sys.stderr)
        return 1
