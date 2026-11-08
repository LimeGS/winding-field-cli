"""Shared preflight contracts; no source or model I/O."""

import math
from pathlib import Path

import numpy as np


def read_tifxyz_request(path):
    from .tifxyz_contract import validate_request
    from .tifxyz_field import read_metadata
    return validate_request(read_metadata(Path(path).expanduser())[0])


def tifxyz_settings(out,request=None,*,resume=False):
    """Reuse the worker's frozen effective settings; never resolve geometry here."""
    from .tifxyz_contract import Settings,validate_request
    from .tifxyz_field import read_metadata
    from .util import digest
    if not resume:
        if request is None:
            raise ValueError('--config is required for a new tifxyz export')
        return validate_request(request)
    out=Path(out)
    if any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('symlink in export path')
    saved=read_metadata(out/'EXPORT.json')[0]
    try:
        settings=Settings(**saved['effective_settings'])
    except (KeyError,TypeError) as error:
        raise ValueError('invalid saved tifxyz settings') from error
    if request is not None and digest(validate_request(request))!=settings.request_sha256:
        raise ValueError('tifxyz request changed; use a new output')
    return settings


def validate_observations(config,volume,shape):
    if (not config.get('lasagna_input') and (config.get('fibres') or config.get('prediction','classical')!='classical')
            and volume.dtype not in (np.dtype('uint8'),np.dtype('uint16'))):
        raise ValueError('Lasagna requires uint8 or uint16 CT')
    if config.get('prediction')=='qual-v2' and not config.get('fibres'):
        spacing=np.asarray(volume.spacing_xyz)*config.get('stride',1)
        if (volume.dtype!=np.uint8 or not np.allclose(spacing,spacing[0],rtol=1e-6,atol=0)
                or spacing.min()<8.64-1e-6 or spacing.max()>9.6+1e-6):
            raise ValueError('Qual V2 requires uint8 CT and isotropic 8.64–9.6 µm observations; '
                             'select --level/--stride to match, independently of --field-spacing-um')
        if min(shape)<8:
            raise ValueError('Qual V2 splines need at least 8 observation samples per axis')


def field_geometry(meta,spacing_um=None):
    geometry={k:meta[k] for k in ('shape_zyx','origin_xyz','spacing_xyz','frame')}
    if spacing_um is not None:
        if not math.isfinite(spacing_um) or spacing_um<=0:
            raise ValueError('field spacing must be positive finite micrometres')
        extent=(np.asarray(meta['shape_zyx'])[::-1]-1)*meta['spacing_xyz']
        shape=[]
        for length in extent:
            ratio=float(length)/spacing_um
            if not math.isfinite(ratio) or ratio>=np.iinfo(np.int64).max-2:
                raise ValueError('field grid is too large; increase --field-spacing-um')
            last=math.floor(ratio)
            # Division can round below an integer; coordinates decide ownership.
            last+=int((last+1)*spacing_um<=length)
            last-=int(last*spacing_um>length)
            shape.append(last+1)
        geometry.update(spacing_xyz=[float(spacing_um)]*3,shape_zyx=shape[::-1])
    if min(geometry['shape_zyx'])<2:
        raise ValueError('field grid needs at least two nodes per axis; enlarge ROI or reduce --field-spacing-um')
    return geometry


LASAGNA_DOWNSAMPLE={'cos':2,'grad_mag':4,'nx':4,'ny':4}


def lasagna_downsamples(config):
    factors={name:config.get('lasagna_'+name+'_downsample',default)
             for name,default in LASAGNA_DOWNSAMPLE.items()}
    if any(type(v) is not int or v not in (1,2,4,8) for v in factors.values()):
        raise ValueError('Lasagna downsample factors must be1,2,4 or8')
    if factors['nx']!=factors['ny']:
        raise ValueError('Lasagna nx and ny must share the same downsample factor')
    return factors


def validate_config(config):
    lasagna_downsamples(config)
    if config.get('lasagna_input') and any('lasagna_'+n+'_downsample' in config for n in LASAGNA_DOWNSAMPLE):
        raise ValueError('--lasagna-input conflicts with Lasagna generation downsample flags')
    factor=config.get('fibre_downsample',2)
    if type(factor) is not int or factor not in (2,4):raise ValueError('fibre_downsample must be2 or4')
    if factor!=2 and not config.get('bounded_preparation'):
        raise ValueError('4x fibre downsampling requires --bounded-preparation')
    if type(config.get('bounded_preparation',False)) is not bool:
        raise ValueError('bounded_preparation must be boolean')
    if config.get('bounded_preparation'):
        if config.get('lasagna_input'):
            raise ValueError('--lasagna-input conflicts with --bounded-preparation; omit bounded preparation to import surfaces')
        if config.get('prediction')!='qual-v2' or config.get('fibres') or config.get('stride',1)!=1:
            raise ValueError('bounded preparation requires native Qual V2 with stride1 and no external fibres')
        if config.get('field_spacing_um') is None:
            raise ValueError('bounded preparation requires --field-spacing-um')
        if any(v%(2*factor) for v in config.get('roi_lo',[0,0,0])):
            raise ValueError(f'bounded ROI origin must be a multiple of{2*factor} to preserve global seed parity')
        budget=config.get('bounded_bytes',12<<30)
        if type(budget) is not int or budget<16<<20:raise ValueError('bounded allowance must be at least16MiB')
    if config.get('lasagna_geometry') and not config.get('lasagna_input'):
        raise ValueError('--lasagna-geometry requires --lasagna-input')
    if config.get('lasagna_input') and (config.get('lasagna_model_path') or config.get('lasagna_python')):
        raise ValueError('--lasagna-input conflicts with --lasagna-model-path and --lasagna-python')
    if config.get('fibres_geometry') and not config.get('fibres'):
        raise ValueError('--fibres-geometry requires --fibres')
    if config.get('fibres') and config.get('model_path'):
        raise ValueError('external AFV --fibres conflicts with a fibre --model-path')
    integers={'stride':(1,8),'chunk':(4,128),'field_chunk':(4,256),'level':(0,None),
              'sweeps':(1,None),'maxiter':(1,None),'seed_stride':(1,None),
              'mesh_tile':(1,256),'chart_faces':(1,8192),'threads':(1,None),
              'cache_bytes':(0,None),'reserve_bytes':(0,None),'bounded_bytes':(16<<20,None),
              'prepared_cache_bytes':(0,None),'constraint_cache_bytes':(0,None)}
    for name,(low,high) in integers.items():
        if name in config:
            value=config[name]
            if type(value) is not int or value<low or (high is not None and value>high):
                raise ValueError(f'{name} must be an integer in {low}..{high or "unbounded"}')
    for name in ('voxel_um','field_spacing_um','rtol','fibre_weight','threshold','invalid_value'):
        value=config.get(name)
        if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)
                or (name in ('voxel_um','field_spacing_um','rtol') and value<=0)
                or (name=='fibre_weight' and value<0)):
            raise ValueError('invalid '+name)
    if config.get('levels') is not None:
        if not config['levels'] or not all(math.isfinite(v) for v in config['levels']):
            raise ValueError('levels must be nonempty finite numbers')
        if len(set(config['levels']))!=len(config['levels']):
            raise ValueError('duplicate levels')
    for name,choices in [('prediction',('classical','sd2','qual-v2')),('ct_mode',('auto','download','stream')),
                          ('surface_output',('atlas','tifxyz','native')),('flattening_method',('default','original'))]:
        if name in config and config[name] not in choices:
            raise ValueError('invalid '+name)
    if config.get('flattening_method')=='original' and config.get('surface_output')=='tifxyz':
        raise ValueError('original flattening uses --export-tifxyz, not the direct-field surface-output branch')
    if config.get('surface_output','atlas')=='native':
        coverage=config.get('coverage_target',1.)
        seconds=config.get('surface_time_limit_s',1800.)
        if type(coverage) not in (int,float) or not math.isfinite(coverage) or not 0<coverage<=1:
            raise ValueError('coverage_target must be finite in(0,1]')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('surface time limit must be positive finite seconds')
        if type(config.get('export_tifxyz',False)) is not bool:
            raise ValueError('export_tifxyz must be boolean')
    if config.get('surface_output','atlas')=='tifxyz':
        from .tifxyz_contract import validate_request
        from .util import digest
        request=validate_request(config.get('tifxyz_request'))
        if config.get('tifxyz_request_sha256')!=digest(request):
            raise ValueError('tifxyz request digest differs')
        if config.get('levels') is not None and config['levels']!=request['levels']:
            raise ValueError('--levels must exactly agree with tifxyz JSON levels')
    elif config.get('tifxyz_request') is not None:
        raise ValueError('--tifxyz-config requires --surface-output tifxyz')
