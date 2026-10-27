"""Shared preflight contracts; no source or model I/O."""

import math

import numpy as np


def field_geometry(meta,spacing_um=None):
    geometry={k:meta[k] for k in ('shape_zyx','origin_xyz','spacing_xyz','frame')}
    if spacing_um is not None:
        if not math.isfinite(spacing_um) or spacing_um<=0:
            raise ValueError('field spacing must be positive finite micrometres')
        extent=(np.asarray(meta['shape_zyx'])[::-1]-1)*meta['spacing_xyz']
        geometry.update(spacing_xyz=[float(spacing_um)]*3,
                        shape_zyx=(np.floor(extent/spacing_um).astype(np.int64)+1)[::-1].tolist())
    if min(geometry['shape_zyx'])<2:
        raise ValueError('field grid needs at least two nodes per axis; enlarge ROI or reduce --field-spacing-um')
    return geometry


def validate_config(config):
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
    for name,choices in [('prediction',('classical','sd2')),('ct_mode',('auto','download','stream'))]:
        if name in config and config[name] not in choices:
            raise ValueError('invalid '+name)
