"""Shared preflight contracts; no source or model I/O."""

import math


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
