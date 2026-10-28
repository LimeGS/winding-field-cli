"""Shared preflight contracts; no source or model I/O."""

import math

import numpy as np


def validate_observations(config,volume,shape):
    if (config.get('fibres') or config.get('prediction','classical')!='classical') and volume.dtype not in (np.dtype('uint8'),np.dtype('uint16')):
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


def validate_config(config):
    if config.get('fibres_geometry') and not config.get('fibres'):
        raise ValueError('--fibres-geometry requires --fibres')
    if config.get('fibres') and config.get('model_path'):
        raise ValueError('external AFV --fibres conflicts with a fibre --model-path')
    integers={'stride':(1,8),'chunk':(4,128),'field_chunk':(4,256),'level':(0,None),
              'sweeps':(1,None),'maxiter':(1,None),'seed_stride':(1,None),
              'mesh_tile':(1,256),'chart_faces':(1,8192),'threads':(1,None),
              'cache_bytes':(0,None),'reserve_bytes':(0,None)}
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
    for name,choices in [('prediction',('classical','sd2','qual-v2')),('ct_mode',('auto','download','stream'))]:
        if name in config and config[name] not in choices:
            raise ValueError('invalid '+name)
