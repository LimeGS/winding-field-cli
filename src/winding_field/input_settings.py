"""Typed AFV/Lasagna declarations; artifact readers still validate the inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .cli_settings import merge_explicit


def declaration(value,*,limit=1<<20,remote=False):
    from .tifxyz_contract import parse_json
    from .io import http_url,fetch
    url=http_url(value) if remote and not isinstance(value,dict) else ''
    url=url if url.startswith(('http://','https://')) else ''
    path=None if isinstance(value,dict) else url or Path(value).expanduser().resolve()
    if path is None:
        raw=json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    elif url:
        raw=fetch(url,limit=limit)[0]
    else:
        with path.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('input declaration exceeds metadata budget')
    mapping=parse_json(raw)
    if not isinstance(mapping,dict):raise ValueError('input declaration must be an object')
    return mapping,hashlib.sha256(raw).hexdigest(),path


def declaration_location(value):
    if isinstance(value,dict):return deepcopy(value)
    return str(Path(value).expanduser().resolve()) if value else None


_GEOMETRY=('source','origin_xyz_um','spacing_xyz_um')
_MANIFEST=('base_shape_zyx','source_to_base','grad_mag_encode_scale','grad_mag_factor','provenance','required_transport')
IMPORT_OPTIONS=tuple(prefix+name for prefix in ('fibres_','lasagna_') for name in _GEOMETRY)+tuple(
    'lasagna_'+name for name in (*_MANIFEST,'channel','channel_metadata_sha256','crop','metadata_pin',
                                'ct_source','ct_level','ct_voxel_um','embedded_origin_xyz_um','embedded_spacing_xyz_um'))


def import_arguments(parser):
    group=parser.add_argument_group('inline AFV/Lasagna input declarations',argument_default=argparse.SUPPRESS)
    for prefix in ('fibres-','lasagna-'):
        group.add_argument('--'+prefix+'source',help='CT source bound by the explicit physical frame')
        for name in ('origin-xyz-um','spacing-xyz-um'):
            group.add_argument('--'+prefix+name,type=float,nargs=3,help='physical XYZ micrometres')
    group.add_argument('--lasagna-base-shape-zyx',type=int,nargs=3)
    for name in ('source_to_base','grad_mag_encode_scale','grad_mag_factor'):
        group.add_argument('--lasagna-'+name.replace('_','-'),type=float)
    group.add_argument('--lasagna-provenance',type=Path,help='optional actual producer receipt')
    group.add_argument('--lasagna-required-transport',choices=('vc_reader.DiskCache',))
    row=group.add_argument('--lasagna-channel',nargs=4,action='append',metavar=('GROUP','CHANNEL','ZARR','SCALEDOWN'))
    row.element_types=(str,str,str,int)
    group.add_argument('--lasagna-channel-metadata-sha256',nargs=2,action='append',metavar=('GROUP','SHA256'))
    group.add_argument('--lasagna-crop',type=int,nargs=6,action='append',metavar=('X','Y','Z','WIDTH','HEIGHT','DEPTH'))
    group.add_argument('--no-lasagna-crops',dest='lasagna_crop',action='store_const',const=[])
    group.add_argument('--lasagna-metadata-pin',nargs=2,action='append',metavar=('LOCATION','SHA256'))
    group.add_argument('--clear-lasagna-metadata-pins',dest='lasagna_metadata_pin',action='store_const',const=[])
    group.add_argument('--lasagna-ct-source',help='actual CT data source to bind; reads its identity, not a handwritten metadata recipe')
    group.add_argument('--lasagna-ct-level',type=int)
    group.add_argument('--lasagna-ct-voxel-um',type=float)
    group.add_argument('--lasagna-embedded-origin-xyz-um',type=float,nargs=3)
    group.add_argument('--lasagna-embedded-spacing-xyz-um',type=float,nargs=3)


def _source(value):
    from .io import http_url
    value=http_url(value)
    return value if value.startswith(('http://','https://')) else str(Path(value).expanduser().resolve())


def import_requests(args):
    from .lasagna_stream import location
    values=vars(args);result={};pins={}
    for prefix in ('fibres','lasagna'):
        explicit={name:values[prefix+'_'+name] for name in _GEOMETRY if prefix+'_'+name in values}
        if not explicit:continue
        native=values.get(prefix+'_geometry')
        base={}
        if native:
            base,pin,path=declaration(native,limit=65536 if prefix=='fibres' else 1<<20)
            if path is not None:pins[str(path)]=pin
        geometry=merge_explicit(base,explicit)
        if 'source' in explicit:geometry['source']=_source(geometry['source'])
        result[prefix+'_geometry']=geometry
    manifest_flags={name:values['lasagna_'+name] for name in _MANIFEST if 'lasagna_'+name in values}
    other=any('lasagna_'+name in values for name in ('channel','channel_metadata_sha256','crop','metadata_pin',
        'ct_source','ct_level','ct_voxel_um','embedded_origin_xyz_um','embedded_spacing_xyz_um'))
    if manifest_flags or other:
        native=values.get('lasagna_input');base={}
        if native:
            base,pin,path=declaration(native,remote=True)
            if path is not None:pins[str(path)]=pin
            for group in base.get('groups',{}).values():
                if 'zarr' in group:group['zarr']=location(group['zarr'],str(path) if path else None)
            if base.get('provenance'):base['provenance']=location(base['provenance'],str(path) if path else None)
        groups={}
        for key,channel,path,level in values.get('lasagna_channel',[]):
            if key in groups:raise ValueError('duplicate Lasagna group: '+key)
            groups[key]=dict(channels=[channel],zarr=location(path),scaledown=int(level))
        if groups:manifest_flags['groups']=groups
        manifest=merge_explicit({'version':2,'source_to_base':1.,'grad_mag_encode_scale':1000.,'grad_mag_factor':1.,**base},manifest_flags)
        if 'lasagna_provenance' in values:manifest['provenance']=str(args.lasagna_provenance.resolve())
        if 'lasagna_crop' in values:manifest['crops']=args.lasagna_crop
        seen=set()
        for key,pin in values.get('lasagna_channel_metadata_sha256',[]):
            if key in seen or key not in manifest.get('groups',{}):raise ValueError('unknown/duplicate Lasagna channel pin')
            seen.add(key);manifest['groups'][key]['metadata_sha256']=pin
        if 'lasagna_metadata_pin' in values:
            metadata={}
            for source,pin in args.lasagna_metadata_pin:
                source=location(source)
                if source in metadata:raise ValueError('duplicate Lasagna metadata pin')
                metadata[source]=pin
            manifest.setdefault('studio_stream',{})['metadata_sha256']=metadata
        binding={key:values['lasagna_'+key] for key in ('ct_source','ct_level','ct_voxel_um',
            'embedded_origin_xyz_um','embedded_spacing_xyz_um') if 'lasagna_'+key in values}
        if binding:
            if 'ct_source' not in binding:raise ValueError('Lasagna CT binding requires --lasagna-ct-source data')
            from contextlib import closing
            from .io import Volume
            with closing(Volume(binding['ct_source'],level=binding.get('ct_level',0),
                                voxel_um=binding.get('ct_voxel_um'))) as volume:
                identity=deepcopy(volume.identity)
            if 'embedded_origin_xyz_um' in binding or 'embedded_spacing_xyz_um' in binding:
                if not all(k in binding for k in ('embedded_origin_xyz_um','embedded_spacing_xyz_um')):
                    raise ValueError('embedded Lasagna physical frame requires both origin and spacing')
                manifest.update(original_ct=identity,world_origin_xyz_um=binding['embedded_origin_xyz_um'],
                                world_spacing_xyz_um=binding['embedded_spacing_xyz_um'])
            else:manifest['ct_source']=identity
        result['lasagna_input']=manifest
    if pins:result['input_recipe_sources']=pins
    return result


def check_recipe_sources(config):
    from .util import file_sha
    from .io import fetch
    for path,pin in config.get('input_recipe_sources',{}).items():
        # Offline continuation uses the frozen declaration; channel metadata/byte ledgers still validate.
        if config.get('offline') and path.startswith(('http://','https://')):continue
        actual=(hashlib.sha256(fetch(path,limit=1<<20)[0]).hexdigest()
                if path.startswith(('http://','https://')) else file_sha(path))
        if actual!=pin:raise ValueError('input recipe source changed: '+path)
