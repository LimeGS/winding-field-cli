"""Explicit units and immutable request normalization for direct extraction."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re

from .util import digest

GIB=1<<30
ALGORITHM='direct-field-tifxyz-v1'


class Blocked(RuntimeError):
    def __init__(self,code,detail=''):
        self.code=code;self.detail=detail
        super().__init__(code+(': '+detail if detail else ''))


class Paused(RuntimeError):
    pass


def parse_json(text):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON: '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def positive(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be finite positive')
    return float(value)


def triple(value,name,*,integer=False,minimum=None):
    if not isinstance(value,(list,tuple)) or len(value)!=3:raise ValueError('invalid '+name)
    if any(type(x) not in ((int,) if integer else (int,float)) or not math.isfinite(x)
           or (minimum is not None and x<minimum) for x in value):raise ValueError('invalid '+name)
    return tuple(value)


def validate_frame(frame):
    keys={'id','origin_xyz_um','voxel_um','shape_zyx','source_identity_sha256'}
    if not isinstance(frame,dict) or set(frame)!=keys:raise ValueError('explicit output frame required')
    if not isinstance(frame['id'],str) or not frame['id'].strip():raise ValueError('invalid frame ID')
    triple(frame['origin_xyz_um'],'frame origin')
    triple(frame['shape_zyx'],'frame shape',integer=True,minimum=1)
    positive(frame['voxel_um'],'frame voxel_um')
    if not isinstance(frame['source_identity_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',frame['source_identity_sha256']):
        raise ValueError('invalid frame source identity')
    return deepcopy(frame)


def validate_request(mapping):
    if not isinstance(mapping,dict):raise ValueError('request must be an object')
    defaults=dict(schema='winding-field-tifxyz-v1',roi=None,pitch_um=None,tolerance_um=None,
                  memory_gib=3,scratch_gib=16,reserve_disk_gib=12.25,max_pixels=64_000_000,
                  output_frame='selected-source',audit_backend='numpy',coverage_policy='strict')
    if set(mapping)-set(defaults)-{'levels'}:raise ValueError('unknown tifxyz setting')
    result={**defaults,**deepcopy(mapping)}
    if result['schema']!=defaults['schema']:raise ValueError('unsupported tifxyz schema')
    if result['audit_backend'] not in ('numpy','cpu','cuda'):raise ValueError('invalid audit backend')
    if result['coverage_policy'] not in ('strict','allow_partial'):raise ValueError('invalid coverage policy')
    levels=result.get('levels')
    if (not isinstance(levels,list) or not 1<=len(levels)<=4096 or
        any(type(x) not in (int,float) or not math.isfinite(x) for x in levels) or len(set(levels))!=len(levels)):
        raise ValueError('explicit unique finite levels required')
    for name in ('pitch_um','tolerance_um'):
        if result[name] is not None:positive(result[name],name)
    for name in ('memory_gib','scratch_gib','reserve_disk_gib'):
        value=result[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or value*GIB>=2**63:
            raise ValueError('invalid '+name)
        if name!='reserve_disk_gib' and value==0:raise ValueError('invalid '+name)
    n=result['max_pixels']
    if type(n) is not int or not 4<=n<2**63:raise ValueError('invalid max_pixels')
    roi=result['roi']
    if roi is not None:
        if not isinstance(roi,dict) or set(roi)!={'lo_zyx','hi_zyx'}:raise ValueError('invalid ROI')
        lo=triple(roi['lo_zyx'],'ROI low',integer=True,minimum=0)
        hi=triple(roi['hi_zyx'],'ROI high',integer=True,minimum=0)
        if any(b-a<2 for a,b in zip(lo,hi)):raise ValueError('ROI requires two nodes per axis')
    if result['output_frame']!='selected-source':validate_frame(result['output_frame'])
    return result


@dataclass(frozen=True)
class Settings:
    levels: tuple
    roi_lo_zyx: tuple
    roi_hi_zyx: tuple
    pitch_um: float
    tolerance_um: float
    memory_bytes: int
    scratch_bytes: int
    reserve_disk_bytes: int
    reserve_ram_bytes: int
    max_pixels: int
    output_frame: dict
    request_sha256: str
    algorithm: str=ALGORITHM
    audit_backend: str='numpy'
    coverage_policy: str='strict'

    @property
    def scale(self):return [self.output_frame['voxel_um']/self.pitch_um]*2


def load_settings(request,field_geometry,output_frame):
    r=validate_request(request)
    shape=triple(field_geometry['shape_zyx'],'field shape',integer=True,minimum=2)
    spacing=triple(field_geometry['spacing_xyz'],'field spacing',minimum=0)
    unit=positive(field_geometry['voxel_um'],'field unit')
    h=[positive(x,'field spacing')*unit for x in spacing]
    lo=(0,0,0) if r['roi'] is None else tuple(r['roi']['lo_zyx'])
    hi=shape if r['roi'] is None else tuple(r['roi']['hi_zyx'])
    if any(b>n for b,n in zip(hi,shape)):raise ValueError('ROI outside field')
    frame=validate_frame(output_frame if r['output_frame']=='selected-source' else r['output_frame'])
    return Settings(tuple(float(x) for x in r['levels']),lo,hi,
        min(h) if r['pitch_um'] is None else r['pitch_um'],
        max(h) if r['tolerance_um'] is None else r['tolerance_um'],
        int(r['memory_gib']*GIB),int(r['scratch_gib']*GIB),int(r['reserve_disk_gib']*GIB),2*GIB,
        r['max_pixels'],frame,digest(r),audit_backend=r['audit_backend'],coverage_policy=r['coverage_policy'])


def to_output_xyz(xyz_um,frame):
    import numpy as np
    validate_frame(frame)
    points=np.asarray(xyz_um,float)
    if points.ndim<1 or points.shape[-1]!=3 or not np.isfinite(points).all():raise ValueError('invalid XYZ')
    output=(points-frame['origin_xyz_um'])/frame['voxel_um']
    if (output<0).any() or (output>np.array(frame['shape_zyx'][::-1])-1).any():raise ValueError('XYZ outside output frame')
    result=output.astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('XYZ float32 overflow')
    return result
