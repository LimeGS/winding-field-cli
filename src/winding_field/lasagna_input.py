"""Calibrated, read-only Lasagna v2 inputs normalized to resumable surface tiles."""

from contextlib import ExitStack, closing
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from common import decode_lasagna_normals
from .artifacts import Tiles
from .io import Volume, http_url
from .lasagna import CHUNK, physical_observations
from .splines import bind
from .util import atomic_json, file_sha


def streamed(config):
    """Explicit/embedded physical frames retain the calibrated interpolation route."""
    if not config.get('lasagna_input') or config.get('lasagna_geometry'):return False
    path=http_url(config['lasagna_input'])
    if path.startswith(('http://','https://')):return True
    manifest,_=_json(path)
    return not any(k in manifest for k in ('original_ct','world_origin_xyz_um','world_spacing_xyz_um'))


def _json(path):
    with Path(path).open('rb') as stream:
        raw=stream.read((1<<20)+1)
    if len(raw)>1<<20: raise ValueError('Lasagna header exceeds 1 MiB')
    value=json.loads(raw)
    if not isinstance(value,dict): raise ValueError('Lasagna header must be an object')
    return value,hashlib.sha256(raw).hexdigest()


def _local(path,base=Path('.')):
    if not isinstance(path,(str,Path)) or not str(path).strip() or '://' in str(path):
        raise ValueError('Lasagna inputs require local file paths')
    return (base/Path(path).expanduser()).resolve()


def _source(value):
    if not isinstance(value,str) or not value.strip(): raise ValueError('Lasagna geometry requires a CT source')
    value=http_url(value)
    return value if value.startswith(('http://','https://')) else str(Path(value).expanduser().resolve())


def source_identity(path,geometry_path,source,geometry,cache):
    path=_local(path); manifest,sha=_json(path)
    if type(manifest.get('version')) is not int or manifest['version']!=2:
        raise ValueError('Lasagna manifest v2 required')
    shape=manifest.get('base_shape_zyx')
    if (not isinstance(shape,list) or len(shape)!=3
            or any(type(n) is not int or not 1<=n<2**53 for n in shape)):
        raise ValueError('Lasagna requires a positive integer base_shape_zyx')
    for name,default in [('source_to_base',1.),('grad_mag_encode_scale',1000.),('grad_mag_factor',1.)]:
        value=manifest.setdefault(name,default)
        if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
            raise ValueError('Lasagna '+name+' must be positive and finite')
    embedded=None
    if any(name in manifest for name in ('original_ct','world_origin_xyz_um','world_spacing_xyz_um')):
        original=manifest.get('original_ct')
        if not isinstance(original,dict): raise ValueError('invalid Lasagna original CT frame')
        if original.get('level')==source['level'] and original!=source:
            raise ValueError('Lasagna original CT identity mismatch')
        embedded=dict(source=original.get('source'),origin_xyz_um=manifest.get('world_origin_xyz_um'),
                      spacing_xyz_um=manifest.get('world_spacing_xyz_um'))
    descriptor_sha=None; explicit=None
    if geometry_path:
        geometry_path=_local(geometry_path); explicit,descriptor_sha=_json(geometry_path)
        if set(explicit)!={'source','origin_xyz_um','spacing_xyz_um'}:
            raise ValueError('Lasagna geometry requires source, origin_xyz_um and spacing_xyz_um')
    for frame in (embedded,explicit):
        if frame is None: continue
        frame['source']=_source(frame['source'])
        if frame['source']!=_source(source['source']): raise ValueError('Lasagna CT source mismatch')
        for name in ('origin_xyz_um','spacing_xyz_um'):
            value=np.asarray(frame[name])
            if (value.shape!=(3,) or value.dtype.kind not in 'uif' or not np.isfinite(value).all()
                    or (name=='spacing_xyz_um' and (value<=0).any())):
                raise ValueError('invalid Lasagna geometry '+name)
            frame[name]=value.astype(float).tolist()
    if embedded and explicit and embedded!=explicit:
        raise ValueError('Lasagna geometry conflicts with embedded frame')
    frame=embedded or explicit
    if frame is None: raise ValueError('ambiguous Lasagna geometry; supply --lasagna-geometry')
    low=(np.asarray(geometry['origin_xyz'])-frame['origin_xyz_um'])/frame['spacing_xyz_um']
    high=low+(np.asarray(geometry['shape_zyx'])[::-1]-1)*geometry['spacing_xyz']/np.asarray(frame['spacing_xyz_um'])
    if not np.isfinite([low,high]).all() or (low<-1e-8).any() or (high>np.asarray(shape)[::-1]-1+1e-8).any():
        raise ValueError('observation ROI lies outside the Lasagna base grid')
    crops=manifest.get('crops',[])
    if not isinstance(crops,list): raise ValueError('invalid Lasagna crops')
    for crop in crops:
        if (not isinstance(crop,list) or len(crop)!=6 or any(type(n) is not int for n in crop)
                or min(crop[:3])<0 or min(crop[3:])<=0
                or (np.asarray(crop[:3])+crop[3:]>np.asarray(shape)[::-1]).any()):
            raise ValueError('invalid Lasagna crop XYZ/size')
    if crops and not any((low>=np.asarray(c[:3])-1e-8).all() and
                         (high<np.asarray(c[:3])+c[3:]).all() for c in crops):
        raise ValueError('observation ROI must lie inside one declared Lasagna crop')
    files={str(path):sha}
    if geometry_path: files[str(geometry_path)]=descriptor_sha
    if manifest.get('provenance'):
        provenance=_local(manifest['provenance'],path.parent); report,files[str(provenance)]=_json(provenance)
        if report.get('status','completed')!='completed': raise ValueError('Lasagna producer is not completed')
    groups=manifest.get('groups')
    if not isinstance(groups,dict) or not all(isinstance(g,dict) and isinstance(g.get('channels'),list)
            and all(isinstance(c,str) for c in g['channels']) for g in groups.values()):
        raise ValueError('invalid Lasagna channel groups')
    channels={}
    for name in ('grad_mag','nx','ny','cos','presence'):
        matches=[g for g in groups.values() if name in g.get('channels',[])]
        if not matches and name=='presence': continue
        if len(matches)!=1 or matches[0].get('channels')!=[name]:
            raise ValueError('Lasagna requires one scalar ZYX array for '+name)
        group=matches[0]; level=group.get('scaledown')
        if type(level) is not int or not 0<=level<=30: raise ValueError('invalid Lasagna scaledown')
        array=_local(group.get('zarr'),path.parent)
        if not (array/'.zarray').is_file(): raise ValueError('Lasagna group must name a Zarr v2 array: '+name)
        with closing(Volume(array,voxel_um=1.,cache=cache,offline=True)) as product:
            if list(product.shape)!=[(n+(1<<level)-1)//(1<<level) for n in shape]:
                raise ValueError('Lasagna channel shape does not match base grid/scaledown: '+name)
            channels[name]=dict(path=str(array),scaledown=level,identity=product.identity)
        for metadata in (array/'.zarray',array/'.zattrs',array.parent/'.zattrs'):
            if metadata.is_file(): files[str(metadata)]=file_sha(metadata)
    if channels['nx']['scaledown']!=channels['ny']['scaledown']:
        raise ValueError('Lasagna normal components nx/ny must share a grid')
    if any(file_sha(p)!=value for p,value in files.items()): raise ValueError('Lasagna inputs changed during preflight')
    return dict(path=str(path),geometry_path=str(geometry_path) if geometry_path else None,
                files=files,frame=frame,manifest=manifest,channels=channels)


def _open_products(stack,identity,root,*,required):
    products={}
    for name,channel in identity['input']['channels'].items():
        ledger=root/(name+'.sqlite')
        if required and not ledger.is_file(): raise ValueError('missing Lasagna input ledger: '+name)
        product=stack.enter_context(closing(Volume(channel['path'],voxel_um=1.,cache=root/'reader-cache',offline=True)))
        if product.identity!=channel['identity']: raise ValueError('Lasagna channel metadata changed: '+name)
        product.bind_ledger(ledger); product.validate_used()
        products[name]=product
    return products


def _inventory(root):
    paths=[root/'IDENTITY.json',*sorted(root.glob('*.sqlite')),root/'maps/TILES.json',root/'maps/tiles.sqlite']
    return {str(p.relative_to(root)):file_sha(p) for p in paths if p.is_file()}


def validate_retained(run,config):
    run=Path(run); root=run/'lasagna-source'
    if not root.exists():
        if (run/'LASAGNA.json').exists(): raise ValueError('missing Lasagna input directory')
        return
    if not (root/'IDENTITY.json').is_file(): raise ValueError('missing Lasagna input identity')
    identity=json.loads((root/'IDENTITY.json').read_bytes())
    seal=root/'WORK.json'; inventory=_inventory(root)
    if seal.exists():
        if json.loads(seal.read_bytes())!=inventory: raise ValueError('Lasagna input inventory changed or missing')
    elif set(inventory)!={'IDENTITY.json'} or (run/'LASAGNA.json').exists():
        raise ValueError('Lasagna input inventory is unsealed; use a new output')
    current=source_identity(config['lasagna_input'],config.get('lasagna_geometry'),identity['source'],
                            identity['geometry'],root/'reader-cache')
    if current!=identity['input']: raise ValueError('external Lasagna input identity changed')
    tiles=root/'maps/TILES.json'; receipt=run/'LASAGNA.json'
    if tiles.exists() or receipt.exists():
        if not tiles.is_file(): raise ValueError('missing imported Lasagna tiles')
        with ExitStack() as stack:
            _open_products(stack,identity,root,required=True)
            maps=stack.enter_context(closing(Tiles(root/'maps',shape=identity['geometry']['shape_zyx'],chunk=CHUNK,identity=identity)))
            if receipt.exists():
                saved=json.loads(receipt.read_bytes())
                ledgers={n:file_sha(root/(n+'.sqlite')) for n in current['channels']}
                if (saved['identity']!=identity or saved['tiles_sha256']!=maps.identity()
                        or saved['source_ledger_sha256']!=ledgers or saved['work_sha256']!=file_sha(seal)):
                    raise ValueError('imported Lasagna receipt or source ledger changed')
            else:
                for key in maps.keys(): maps.has(key)


def _sample(products,base_axes,level,names,crops):
    product=products[names[0]]; vector=names==('nx','ny')
    axes=[a/(1<<level) for a in base_axes]
    axes=[np.where(np.abs(a-np.rint(a))<1e-8,np.rint(a),a) for a in axes]
    inside=[(a>=0)&(a<=n-1) for a,n in zip(axes,product.shape)]
    clipped=[np.clip(a,0,n-1) for a,n in zip(axes,product.shape)]
    lower=[np.floor(a).astype(np.int64) for a in clipped]
    upper=[np.ceil(a).astype(np.int64) for a in clipped]
    nodes=[np.unique(np.concatenate([a,b])) for a,b in zip(lower,upper)]
    valid=np.ones(tuple(len(a) for a in nodes),bool); arrays=[]
    for name in names:
        raw=products[name].sample(nodes)
        if not np.isfinite(raw).all() or (raw<0).any() or (name!='grad_mag' and (raw>255).any()):
            raise ValueError('invalid Lasagna channel values: '+name)
        if name in ('grad_mag','presence'): valid &= raw>0
        arrays.append(raw)
    raw=decode_lasagna_normals(*arrays) if vector else arrays[0]
    if crops:
        covered=np.zeros(valid.shape,bool)
        xyz=[a*(1<<level) for a in nodes[::-1]]
        for crop in crops:
            masks=[(a>=low)&(a<low+size) for a,low,size in zip(xyz,crop[:3],crop[3:])]
            covered |= masks[2][:,None,None]&masks[1][None,:,None]&masks[0][None,None,:]
        valid &= covered
    result=np.zeros(tuple(len(a) for a in axes)+((3,) if vector else ()),np.float64)
    if vector: reference=raw[np.ix_(*(np.searchsorted(n,a) for n,a in zip(nodes,lower)))]
    supported=inside[0][:,None,None]&inside[1][None,:,None]&inside[2][None,None,:]
    fraction=[a-b for a,b in zip(clipped,lower)]
    for corner in itertools.product((0,1),repeat=3):
        indices=np.ix_(*(np.searchsorted(n,(upper if c else lower)[i]) for i,(n,c) in enumerate(zip(nodes,corner))))
        weights=[f if c else 1-f for c,f in zip(corner,fraction)]
        weight=weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:]
        supported &= (weight==0)|valid[indices]
        value=raw[indices]
        if vector:
            # Opposite directions encode the same unsigned surface; align before averaging.
            value=value*np.where((value*reference).sum(-1)<0,-1,1)[...,None]
            weight=weight[...,None]
        result+=value*weight
    if vector: result/=np.maximum(np.linalg.norm(result,axis=-1,keepdims=True),1e-12)
    return result,supported


def prepare(volume,run,config,geometry,stop=None):
    run=Path(run); root=run/'lasagna-source'
    source=source_identity(config['lasagna_input'],config.get('lasagna_geometry'),volume.identity,geometry,volume.cache)
    identity=dict(method='external-lasagna',source=volume.identity,geometry=geometry,input=source,
                  invalid_value=config.get('invalid_value'))
    validate_retained(run,config)
    root.mkdir(parents=True,exist_ok=True); bind(root/'IDENTITY.json',identity)
    with ExitStack() as stack:
        products=_open_products(stack,identity,root,required=(root/'maps/TILES.json').exists())
        maps=stack.enter_context(closing(Tiles(root/'maps',shape=geometry['shape_zyx'],chunk=CHUNK,identity=identity)))
        stack.callback(lambda:atomic_json(root/'WORK.json',_inventory(root)))
        for key in maps.keys():
            if stop and stop(): return dict(status='PAUSED')
            lo,hi=maps.box(key)
            ct=volume.sample([geometry['roi_lo_zyx'][i]+np.arange(a,b)*geometry['stride'] for i,(a,b) in enumerate(zip(lo,hi))])
            if not np.isfinite(ct).all(): raise ValueError('finite CT required for Lasagna inputs')
            ct_sha=hashlib.sha256(ct.tobytes()).hexdigest()
            if maps.has(key):
                if maps.info(key)['ct_sha256']!=ct_sha: raise ValueError('Lasagna source CT changed')
                continue
            frame=source['frame']; manifest=source['manifest']
            axes=[(geometry['origin_xyz'][2-i]+np.arange(a,b)*geometry['spacing_xyz'][2-i]-frame['origin_xyz_um'][2-i])
                  /frame['spacing_xyz_um'][2-i] for i,(a,b) in enumerate(zip(lo,hi))]
            valid=np.ones(ct.shape,bool) if identity['invalid_value'] is None else ct!=identity['invalid_value']
            values={}
            for names in (('grad_mag',),('nx','ny'),('cos',),('presence',)):
                if names[0] not in products: continue
                name='normals' if names==('nx','ny') else names[0]
                values[name],support=_sample(products,axes,source['channels'][names[0]]['scaledown'],names,manifest.get('crops',[]))
                valid &= support
                if stop and stop(): return dict(status='PAUSED')
            data=physical_observations(*(values[n] for n in ('grad_mag','normals','cos')),
                input_spacing_xyz=frame['spacing_xyz_um'],encode_scale=manifest['grad_mag_encode_scale'],
                grad_mag_factor=manifest['grad_mag_factor'],validity=valid)
            data['ct']=ct
            if shutil.disk_usage(root).free<sum(a.nbytes for a in data.values())+65536+config.get('reserve_bytes',0):
                raise OSError('insufficient disk space for imported Lasagna tiles and reserve')
            maps.write(key,data,dict(ct_sha256=ct_sha))
        if stop and stop(): return dict(status='PAUSED')
        for product in products.values(): product.validate_used()
        if source!=source_identity(config['lasagna_input'],config.get('lasagna_geometry'),volume.identity,geometry,volume.cache):
            raise ValueError('Lasagna input identity changed during import')
        result=dict(status='LASAGNA_READY',method='external-lasagna',identity=identity,tiles_sha256=maps.identity(),
                    source_ledger_sha256={n:file_sha(root/(n+'.sqlite')) for n in products},
                    density_units='cycles/micrometer',surface_phase='Lasagna cos encoded 0..255, scaled to 0..1')
    if (run/'LASAGNA.json').exists(): return json.loads((run/'LASAGNA.json').read_bytes())
    result['work_sha256']=file_sha(root/'WORK.json')
    atomic_json(run/'LASAGNA.json',result)
    return result


def read_observations(run,geometry,lo,hi):
    root=Path(run)/'lasagna-source'; saved=json.loads((Path(run)/'LASAGNA.json').read_bytes())
    identity=saved['identity']
    if identity['geometry']!=geometry: raise ValueError('Lasagna observation geometry changed')
    with closing(Tiles(root/'maps',shape=geometry['shape_zyx'],chunk=CHUNK,identity=identity)) as maps:
        return maps.read_many(('density','normals','support','surface_phase','ct'),lo,hi)
