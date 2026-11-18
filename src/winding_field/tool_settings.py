"""Explicit settings for historical tools; frozen scientific inputs remain data."""
import argparse
from copy import deepcopy
from pathlib import Path

from .cli_settings import field_arguments,field_overrides,merge_explicit
from .tifxyz_field import read_metadata
from .util import atomic_json,digest,file_sha


def _document(args,path):
    value,record=read_metadata(Path(path))
    if not hasattr(args,'_request_sources'):args._request_sources={}
    name=str(Path(path).resolve())
    if args._request_sources.setdefault(name,record['sha256'])!=record['sha256']:
        raise ValueError('tool request source changed: '+name)
    return value


def check_sources(args):
    for path,pin in getattr(args,'_request_sources',{}).items():
        if file_sha(path)!=pin:raise ValueError('tool request source changed: '+path)


def metadata_arguments(parser,*,ct=False,support=True,ownership=True):
    field_arguments(parser,descriptor=False)
    group=parser.add_argument_group('field metadata and support',argument_default=argparse.SUPPRESS)
    if ownership:group.add_argument('--core-ownership',type=Path)
    if support:
        for name,value_type in (('support',Path),('support-sha256',str),('support-array-sha256',str)):
            if '--'+name not in parser._option_string_actions:group.add_argument('--'+name,type=value_type)
    if ct:ct_arguments(parser)


def ct_arguments(parser):
    group=parser.add_argument_group('CT WORLD geometry',argument_default=argparse.SUPPRESS)
    for name,value_type in (('frame',str),('voxel_um',float)):
        if '--ct-'+name.replace('_','-') not in parser._option_string_actions:
            group.add_argument('--ct-'+name.replace('_','-'),type=value_type)
    for name,value_type in (('origin_xyz',float),('spacing_xyz',float),('shape_zyx',int)):
        group.add_argument('--ct-'+name.replace('_','-'),type=value_type,nargs=3)
    group.add_argument('--ct-display-window',type=float,nargs=2)


def metadata_request(args,*,require_pin=False,world_units=True):
    import numpy as np
    from .surface_export import _metadata
    base=_document(args,args.metadata) if args.metadata else {}
    overrides=field_overrides(args)
    geometry=overrides.get('geometry',{})
    if 'sheet_step' in geometry:geometry['step']=geometry.pop('sheet_step')
    if 'field' in overrides:geometry['field_sha256']=overrides['field']['sha256']
    for flag,key in (('core_ownership','core_ownership'),('support','support'),
                     ('support_sha256','support_sha256'),('support_array_sha256','support_array_sha256')):
        value=getattr(args,flag,None)
        if value is not None:geometry[key]=str(value.resolve()) if isinstance(value,Path) else value
    meta=merge_explicit(base,geometry)
    if getattr(args,'voxel_um',None) is not None:meta['voxel_um']=args.voxel_um
    array=np.load(args.field,mmap_mode='r',allow_pickle=False)
    if world_units:_metadata(meta,array.shape)
    else:
        from dataset_config import frame,xyz
        frame(meta.get('frame'));xyz(meta.get('origin_xyz'),'field origin')
        xyz(meta.get('spacing_xyz'),'field spacing',positive=True)
    meta.setdefault('shape_zyx',list(array.shape))
    if require_pin or 'field_sha256' in meta:
        if file_sha(args.field)!=meta.get('field_sha256'):raise ValueError('field hash differs from declared metadata')
    args._inline_metadata=args.metadata is None or bool(geometry) or getattr(args,'voxel_um',None) is not None
    return meta


def metadata_identity(args,meta):
    return digest(meta) if args._inline_metadata else file_sha(args.metadata)


def ct_request(args,default=None):
    path=getattr(args,'ct_metadata',None)
    meta=_document(args,path) if path else deepcopy(default or {})
    overrides={name:vars(args)['ct_'+name] for name in ('frame','voxel_um','origin_xyz','spacing_xyz','shape_zyx')
               if 'ct_'+name in vars(args)}
    if hasattr(args,'ct_display_window'):overrides['display_window']=args.ct_display_window
    return merge_explicit(meta,overrides)


def materialize_metadata(args,output,meta,ct_meta=None):
    """Write only after the existing owner admits a fresh output directory."""
    if args._inline_metadata:
        args.metadata=output/'field-metadata.json';atomic_json(args.metadata,meta)
    if ct_meta is not None and (getattr(args,'ct_metadata',None) is None or
            any(name in vars(args) for name in ('ct_frame','ct_voxel_um','ct_origin_xyz','ct_spacing_xyz','ct_shape_zyx','ct_display_window'))):
        args.ct_metadata=output/'ct-metadata.json';atomic_json(args.ct_metadata,ct_meta)


_RENDER={'seed_margin':int,'ct_padding_samples':int,'render_pitch_world':float,
         'render_mesh_step':int}


def render_arguments(parser):
    group=parser.add_argument_group('prior/CT rendering settings',argument_default=argparse.SUPPRESS)
    for name,value_type in _RENDER.items():group.add_argument('--'+name.replace('_','-'),type=value_type)
    group.add_argument('--render-depth-offsets-world',type=float,nargs='+')
    group.add_argument('--ct-window',type=float,nargs=2)


def render_request(args):
    result={name.removeprefix('render_'):vars(args)[name] for name in _RENDER if name in vars(args)}
    if hasattr(args,'render_depth_offsets_world'):result['depth_offsets_world']=args.render_depth_offsets_world
    if hasattr(args,'ct_window'):result['ct_window']=args.ct_window
    if hasattr(args,'ct_display_window'):
        if 'ct_window' in result and result['ct_window']!=args.ct_display_window:
            raise ValueError('conflicting CT display windows')
        result['ct_window']=args.ct_display_window
    return result


def dataset_arguments(parser):
    ct_arguments(parser)
    render_arguments(parser)
    group=parser.add_argument_group('CT/prior dataset view',argument_default=argparse.SUPPRESS)
    group.add_argument('--dataset-frame')
    group.add_argument('--dataset-voxel-um',type=float)
    for prefix in ('ct','prior'):
        group.add_argument('--'+prefix+'-source')
        group.add_argument('--'+prefix+'-sha256')
        if prefix=='prior':
            group.add_argument('--prior-frame')
            group.add_argument('--prior-origin-xyz',type=float,nargs=3)
            group.add_argument('--prior-spacing-xyz',type=float,nargs=3)


def dataset_request(args,meta):
    from dataset_config import load_dataset,frame,positive,xyz,_path
    from .surface_probe import render_settings
    data=load_dataset(args.config) if args.config else {}
    if args.config:
        _document(args,args.config)
        if data['config_sha256']!=args._request_sources[str(args.config.resolve())]:
            raise ValueError('dataset recipe changed during validation')
    overrides={}
    for name in ('frame','voxel_um'):
        if 'dataset_'+name in vars(args):overrides[name]=vars(args)['dataset_'+name]
    for prefix,key in (('ct','ct'),('prior','surface')):
        entry={name:vars(args)[prefix+'_'+name] for name in ('frame','origin_xyz','spacing_xyz','sha256')
               if prefix+'_'+name in vars(args)}
        if prefix+'_source' in vars(args):entry['path']=vars(args)[prefix+'_source']
        if entry:overrides.setdefault('sources',{})[key]=entry
    render=render_request(args)
    if render:overrides.setdefault('context',{})['surface']=render
    data=merge_explicit(data,overrides)
    if frame(data.get('frame'))!=meta['frame'] or positive(data.get('voxel_um'),'dataset voxel_um')!=meta['voxel_um']:
        raise ValueError('field and dataset frame/units differ')
    if hasattr(args,'ct_voxel_um') and positive(args.ct_voxel_um,'CT voxel_um')!=data['voxel_um']:
        raise ValueError('CT and dataset units differ')
    for key in ('ct','surface'):
        entry=data.get('sources',{}).get(key,{})
        if frame(entry.get('frame'))!=data['frame']:raise ValueError('source/world frames differ')
        xyz(entry.get('origin_xyz'),'source origin')
        xyz(entry.get('spacing_xyz'),'source spacing',positive=True)
        _path(entry,Path.cwd())
    if hasattr(args,'ct_shape_zyx'):
        from dataset_config import open_source
        if list(open_source(data['sources']['ct']).shape)!=args.ct_shape_zyx:
            raise ValueError('CT shape differs from declared source')
    render_settings(data.get('context',{}).get('surface',{}))
    if overrides or args.config is None:
        data['config_sha256']=digest({k:v for k,v in data.items() if k not in ('config_path','config_sha256')})
    return data


def atlas_arguments(parser):
    metadata_arguments(parser,ct=True)
    render_arguments(parser)
    group=parser.add_argument_group('frozen scientific inputs',argument_default=argparse.SUPPRESS)
    for name in ('ct','ct_metadata','seed'):group.add_argument('--'+name.replace('_','-'),type=Path)
    for name in ('ct_sha256','ct_metadata_sha256','seed_sha256','inputs_field_sha256','inputs_metadata_sha256'):
        group.add_argument('--'+name.replace('_','-'))


def atlas_inputs(args,meta):
    inputs=_document(args,args.inputs) if args.inputs else {}
    for name in ('ct','ct_metadata','seed','ct_sha256','ct_metadata_sha256','seed_sha256'):
        if name in vars(args):inputs[name]=str(vars(args)[name]) if isinstance(vars(args)[name],Path) else vars(args)[name]
    inputs['render']=merge_explicit(inputs.get('render',{}),render_request(args))
    if not args.inputs:
        inputs.update(field_sha256=meta['field_sha256'],metadata_sha256=metadata_identity(args,meta))
    for flag,key in (('inputs_field_sha256','field_sha256'),('inputs_metadata_sha256','metadata_sha256')):
        if flag in vars(args):inputs[key]=vars(args)[flag]
    for name in ('ct',) if args.patch is not None else ('ct','seed'):
        if file_sha(inputs[name])!=inputs.get(name+'_sha256'):raise ValueError('frozen input hash mismatch: '+name)
    default=None
    if inputs.get('ct_metadata'):
        default=_document(args,inputs['ct_metadata'])
        if args._request_sources[str(Path(inputs['ct_metadata']).resolve())]!=inputs.get('ct_metadata_sha256'):
            raise ValueError('frozen CT metadata hash mismatch')
    if getattr(args,'ct_metadata',None) is not None or any('ct_'+name in vars(args) for name in ('frame','origin_xyz','spacing_xyz','voxel_um','shape_zyx')):
        return inputs,ct_request(args,default)
    if 'ct_metadata' not in inputs:raise ValueError('CT geometry or frozen metadata required')
    return inputs,None


def case_arguments(parser):
    group=parser.add_argument_group('frozen CT visualization request',argument_default=argparse.SUPPRESS)
    group.add_argument('--archive-manifest',type=Path)
    group.add_argument('--archive-manifest-sha256')
    group.add_argument('--visualization-schema',choices=('ct-winding-overlays-v1','ct-winding-overlays-v2'))
    group.add_argument('--reference-key')
    group.add_argument('--no-reference-key',dest='reference_key',action='store_const',const=None)
    group.add_argument('--reference-label')
    group.add_argument('--requires-support',action=argparse.BooleanOptionalAction)
    group.add_argument('--case-step',type=float)
    group.add_argument('--dataset-frame')
    group.add_argument('--dataset-voxel-um',type=float)


def case_request(args,meta):
    case=_document(args,args.case) if args.case else dict(schema='winding-benchmark-case-v1',
        frame=meta['frame'],voxel_um=meta['voxel_um'],step=meta['step'])
    spec={}
    for name in ('reference_key','reference_label'):
        if name in vars(args):spec[name]=vars(args)[name]
    if hasattr(args,'visualization_schema'):spec['schema']=args.visualization_schema
    if hasattr(args,'archive_manifest'):spec.setdefault('archive_manifest',{})['path']=str(args.archive_manifest.resolve())
    if hasattr(args,'archive_manifest_sha256'):spec.setdefault('archive_manifest',{})['sha256']=args.archive_manifest_sha256
    if not args.case:spec.setdefault('schema','ct-winding-overlays-v1')
    result=merge_explicit(case,dict(visualization=spec))
    if hasattr(args,'requires_support'):result['requires_support']=args.requires_support
    if hasattr(args,'case_step'):result['step']=args.case_step
    args._inline_case=args.case is None or bool(spec) or hasattr(args,'requires_support') or hasattr(args,'case_step')
    if args._inline_case:
        archive=result['visualization']['archive_manifest']
        base=args.case.parent if args.case else Path.cwd()
        archive['path']=str((base/Path(archive['path']).expanduser()).resolve())
    return result


def constant_render_request(args):
    from .export_settings import flattening_request
    config=_document(args,args.config) if args.config else {}
    if args.config:
        base=args.config.parent
        for name in ('prior_identity','prior_ledger'):
            if config.get(name):config[name]=str((base/config[name]).resolve())
        for name in ('app_run','cache_dir','reuse_export'):
            if config.get('settings',{}).get(name):config['settings'][name]=str((base/Path(config['settings'][name]).expanduser()).resolve())
    ct={name:vars(args)['ct_'+name] for name in ('source','level','voxel_um') if 'ct_'+name in vars(args)}
    config=merge_explicit(config,dict(ct=ct))
    config['settings']=flattening_request(args,config.get('settings',{}))
    for name in ('prior_identity','prior_ledger','prior_ledger_sha256'):
        if name in vars(args):
            value=vars(args)[name]
            config[name]=str(value.resolve()) if isinstance(value,Path) else value
    return config
