"""All-component extraction from an immutable supported field."""

import itertools
import json
import math
from pathlib import Path

import numpy as np

from surface_mesh import extract_mesh
from .artifacts import Tiles
from .field import open_field,map_axes
from .util import atomic_json


def mesh(run,config,stop=None):
    run=Path(run)
    with open_field(run) as source:
        levels=config.get('levels')
        phase=complex(0); total=0.; low=math.inf; high=-math.inf
        if not levels:
            maps=Tiles(run/'maps')
            meta=json.loads((run/'PREPARED.json').read_bytes())
            geometry=json.loads((run/'FIELD.json').read_bytes())['geometry']
            tile=config.get('mesh_tile',64)
            for key in itertools.product(*(range((n-1)//tile+1) for n in source.shape)):
                lo=np.array(key)*tile; hi=np.minimum(lo+tile,source.shape)
                field,valid=source.read(lo,hi)
                if not valid.any():
                    continue
                low=min(low,float(field[valid].min())); high=max(high,float(field[valid].max()))
                presence=maps.sample(('presence',),map_axes(meta,geometry,lo,hi))['presence']; weight=presence[valid]
                phase+=np.sum(weight*np.exp(2j*np.pi*field[valid])); total+=float(weight.sum())
            maps.close()
            if not math.isfinite(low) or high-low<1e-6:
                raise ValueError('no nonconstant supported field to mesh')
            offset=float(np.angle(phase)/(2*np.pi)%1)
            levels=np.arange(math.ceil(low-offset+1e-6),math.floor(high-offset-1e-6)+1)+offset
            if len(levels)==0:
                levels=np.array([(low+high)/2])
            if len(levels)>4096:
                raise ValueError('more than 4096 levels; select an explicit level range')
            levels=levels.tolist()
        result=extract_mesh(source,run/'meshes',levels,tile=config.get('mesh_tile',64),
                    resume=(run/'meshes/RUN.json').exists(),stop=stop,
                    reserve_bytes=config.get('reserve_bytes',0))
    if result['status']!='MESH_READY':
        return dict(status='PAUSED',mesh=result)
    if not sum(item['faces'] for item in result['levels']):
        raise ValueError('requested levels contain no supported mesh cells')
    atomic_json(run/'MESHES.json',dict(result,phase_concentration=abs(phase)/max(total,1e-12),
                level_selection='explicit' if config.get('levels') else 'CT-presence circular phase'))
    return result
