"""Bounded, provenance-bound v2 Lasagna inputs on an explicitly selected CT grid."""

from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import shutil
from urllib.parse import urljoin

import numpy as np

from .io import Volume, bounds, fetch, http_url
from .lasagna import decode_observations
from .splines import bind
from .util import atomic_json, digest, file_sha

CHANNELS=('grad_mag','nx','ny','cos')


def retained_import(run):
    run=Path(run);root=run/'lasagna-import'
    if not (root/'IDENTITY.json').is_file():
        if (root/'CHUNKS.json').exists() or any((run/name).exists() for name in ('LASAGNA.json','PREPARED.json','maps','coarse')):
            raise ValueError('Lasagna import provenance missing for retained products; use a new output')
        return False
    if not (root/'CHUNKS.json').is_file():
        raise ValueError('Lasagna input ledger seal missing; use a new output')
    return True


def location(value,base=None):
    value=http_url(value)
    if value.startswith(('http://','https://')):return value
    if base and base.startswith(('http://','https://')):
        return urljoin(base,value)
    return str((Path(base).parent/value if base else Path(value)).expanduser().resolve())


class LasagnaInput:
    """Nearest pooled-cell sampling; channel downsampling never rescales density."""

    def __init__(self,volume,run,config):
        self.volume=volume; self.config=config; self.products={}; self.stack=ExitStack()
        self.root=Path(run)/'lasagna-import'; self.root.mkdir(parents=True,exist_ok=True)
        self.source=location(config['lasagna_input']); self.documents={}
        try:
            retained_import(run)
            seal=self.root/'CHUNKS.json'
            if seal.exists():
                for name,sha in json.loads(seal.read_bytes()).items():
                    if file_sha(self.root/name)!=sha:raise ValueError('Lasagna input ledger changed: '+name)
            manifest=self.document(self.source)
            if not isinstance(manifest,dict) or manifest.get('version')!=2:
                raise ValueError('external Lasagna requires a v2 manifest')
            self.manifest=manifest
            shape=manifest.get('base_shape_zyx')
            if not isinstance(shape,list) or len(shape)!=3 or any(type(n) is not int or n<1 for n in shape):
                raise ValueError('invalid Lasagna base_shape_zyx')
            self.base_shape=np.array(shape,dtype=np.int64)
            for name in ('source_to_base','grad_mag_encode_scale','grad_mag_factor'):
                value=manifest.get(name)
                if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
                    raise ValueError('Lasagna '+name+' must be positive finite')
            self.crops=manifest.get('crops',[[0,0,0,*shape[::-1]]])
            if not isinstance(self.crops,list):raise ValueError('Lasagna crops must be XYZ-width-height-depth boxes')
            for crop in self.crops:
                if (not isinstance(crop,list) or len(crop)!=6 or any(type(n) is not int for n in crop)
                        or min(crop[:3])<0 or min(crop[3:])<=0
                        or np.any(np.array(crop[:3])+crop[3:]>self.base_shape[::-1])):
                    raise ValueError('invalid Lasagna coverage crop')
            base=volume
            if volume.identity['level']:
                base=Volume(volume.source,local=volume.local,cache=volume.cache,cache_bytes=volume.limit,
                            offline=volume.offline,voxel_um=config.get('voxel_um'),reserve_bytes=volume.reserve)
                self.stack.callback(base.close)
            if list(base.shape)!=shape:raise ValueError('Lasagna base grid differs from selected CT base shape')
            self.base_spacing=np.asarray(base.spacing_xyz); self.base_origin=np.asarray(base.origin_xyz)
            ratio=np.asarray(volume.spacing_xyz)/self.base_spacing
            offset=(np.asarray(volume.origin_xyz)-self.base_origin)/self.base_spacing
            if (not np.allclose(ratio,np.rint(ratio),atol=1e-7,rtol=0) or np.any(ratio<1)
                    or not np.allclose(offset,0,atol=1e-7,rtol=0)
                    or not np.array_equal(np.ceil(self.base_shape/ratio[::-1]).astype(int),volume.shape)):
                raise ValueError('selected CT level does not share the Lasagna base grid')
            self.selected_to_base=np.rint(ratio[::-1]).astype(np.int64)
            explicit=manifest.get('ct_source')
            if explicit is not None and explicit!=base.identity:
                raise ValueError('Lasagna manifest CT source identity differs')
            provenance=self.document(location(manifest['provenance'],self.source)) if manifest.get('provenance') else None
            groups=manifest.get('groups',{})
            if not isinstance(groups,dict) or not all(name in groups for name in CHANNELS):
                raise ValueError('Lasagna requires grad_mag, nx, ny and cos groups')
            if 'mask' in groups and 'validity' in groups:raise ValueError('use one Lasagna validity mask')
            self.mask=next((n for n in ('mask','validity') if n in groups),None)
            names=(*CHANNELS,*([self.mask] if self.mask else []))
            if seal.exists() and set(json.loads(seal.read_bytes()))!={n+'.sqlite' for n in names}:
                raise ValueError('Lasagna input ledger inventory changed')
            for name in names:
                group=groups[name]; level=group.get('scaledown'); path=group.get('zarr')
                if (type(level) is not int or not 0<=level<=16 or not isinstance(path,str) or not path
                        or group.get('channels')!=[name]):
                    raise ValueError('invalid Lasagna group: '+name)
                source=location(path,self.source); scale=2**level
                parent=source.rsplit('/',1)[0]; selected=source.rsplit('/',1)[1]
                attrs=self.document(parent+'/.zattrs',optional=True)
                direct=self.document(source+'/.zattrs',optional=True)
                self.check_ome(attrs,selected,scale)
                if direct and direct.get('multiscales'):raise ValueError('Lasagna groups must name direct arrays')
                product=Volume(source,voxel_um=1.,cache=config.get('cache',volume.cache),
                    cache_bytes=config.get('cache_bytes',volume.limit),offline=config.get('offline',False),
                    reserve_bytes=config.get('reserve_bytes',volume.reserve))
                self.stack.callback(product.close)
                if (product.array_path or product.dtype!=np.uint8
                        or tuple(product.shape)!=tuple((self.base_shape+scale-1)//scale)
                        or product.metadata.get('fill_value')!=0):
                    raise ValueError('Lasagna channel geometry/uint8/zero-fill mismatch: '+name)
                product.bind_ledger(self.root/(name+'.sqlite'))
                self.products[name]=product
            self.identity=dict(kind='external-lasagna-v2',manifest=self.source,documents=self.documents,
                encoding=manifest,provenance=provenance,channels={n:p.identity for n,p in self.products.items()},
                binding=dict(kind='manifest-ct-source' if explicit is not None else 'user-selected-ct-base-grid',
                             ct_base=base.identity,ct_selected=volume.identity),
                input_spacing_xyz=(self.base_spacing*manifest['source_to_base']).tolist(),
                sampling='pooled-cell-floor-base-index-v1',coverage='manifest-crop-union-conservative-cell',
                invalid_value=config.get('invalid_value'))
            bind(self.root/'IDENTITY.json',self.identity)
        except BaseException:
            self.stack.close();raise

    def document(self,source,optional=False):
        path=self.root/('document-'+digest(source)+'.json')
        remote=source.startswith(('http://','https://'))
        if remote and self.config.get('offline',False):
            if not path.exists():raise FileNotFoundError('offline Lasagna metadata missing: '+source)
            record=json.loads(path.read_bytes())
        else:
            try:
                if remote:raw=fetch(source,limit=4<<20)[0]
                else:
                    with Path(source).open('rb') as stream:raw=stream.read((4<<20)+1)
                if len(raw)>4<<20:raise ValueError('Lasagna metadata exceeds 4MiB budget')
                record=dict(source=source,sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
            except FileNotFoundError:
                if not optional:raise
                record=dict(source=source,sha256=None,value=None)
            if shutil.disk_usage(self.root).free<self.config.get('reserve_bytes',0)+(4<<20):
                raise OSError('Lasagna metadata disk reserve exhausted')
            bind(path,record)
        self.documents[source]=dict(sha256=record['sha256'],value_sha256=digest(record['value']))
        return record['value']

    def check_ome(self,attrs,selected,scale):
        if not attrs or not attrs.get('multiscales'):return
        ms=attrs['multiscales']
        if len(ms)!=1:raise ValueError('ambiguous Lasagna OME geometry')
        ms=ms[0]; axes=ms.get('axes',[])
        if [a if isinstance(a,str) else a.get('name') for a in axes]!=list('zyx'):
            raise ValueError('Lasagna OME axes must be ZYX')
        datasets=[d for d in ms.get('datasets',[]) if d.get('path')==selected]
        if len(datasets)!=1:raise ValueError('Lasagna channel absent from OME geometry')
        pitch=np.ones(3);origin=np.zeros(3)
        for t in datasets[0].get('coordinateTransformations',[])+ms.get('coordinateTransformations',[]):
            value=np.asarray(t.get(t.get('type')),float)
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('invalid Lasagna OME transform')
            if t['type']=='scale':pitch*=value;origin*=value
            elif t['type']=='translation':origin+=value
            else:raise ValueError('unsupported Lasagna OME transform')
        units=[a.get('unit') if isinstance(a,dict) else None for a in axes]
        if units in ([None]*3,['pixel']*3):expected=np.full(3,scale);expected_origin=np.zeros(3)
        elif units in (['micrometer']*3,['micrometre']*3,['um']*3):
            expected=self.base_spacing[::-1]*scale;expected_origin=self.base_origin[::-1]
        else:raise ValueError('unsupported Lasagna OME units')
        if not np.allclose(pitch,expected,atol=1e-7,rtol=1e-7) or not np.allclose(origin,expected_origin,atol=1e-7,rtol=0):
            raise ValueError('Lasagna OME geometry conflicts with base grid/scaledown')

    def coverage(self,axes,scale=1):
        # Conservative at crop seams: a pooled cell must fit one declared crop.
        starts=[np.floor(a/scale).astype(np.int64)*scale for a in axes]
        ends=[np.minimum(a+scale,n) for a,n in zip(starts,self.base_shape)]
        mask=np.zeros(tuple(len(a) for a in axes),bool)
        for crop in self.crops:
            lo=np.array(crop[:3])[::-1];hi=lo+np.array(crop[3:])[::-1]
            parts=[(a>=x)&(b<=y) for a,b,x,y in zip(starts,ends,lo,hi)]
            mask|=parts[0][:,None,None]&parts[1][None,:,None]&parts[2][None,None,:]
        return mask

    def read(self,geometry,lo,hi):
        lo,hi=bounds(lo,hi,geometry['shape_zyx'])
        if np.prod(hi-lo,dtype=object)>128**3:raise ValueError('Lasagna read exceeds 128-cube sample budget')
        if geometry['frame']!=digest(self.volume.identity):raise ValueError('Lasagna CT frame changed')
        ct_axes=[geometry['roi_lo_zyx'][i]+np.arange(a,b)*geometry['stride'] for i,(a,b) in enumerate(zip(lo,hi))]
        # Validated integer CT grids avoid cancellation at decimal pitch/large origins.
        axes=[a*scale for a,scale in zip(ct_axes,self.selected_to_base)]
        if any(np.any((a<0)|(a>=n)) for a,n in zip(axes,self.base_shape)):
            raise ValueError('Lasagna requested samples outside base grid')
        valid=self.coverage(axes);values={}
        for name,product in self.products.items():
            scale=2**self.manifest['groups'][name]['scaledown']
            indices=[np.floor(a/scale).astype(np.int64) for a in axes]
            valid &= self.coverage(axes,scale)
            values[name]=np.zeros(valid.shape,np.uint8)
            # A missing normal/phase chunk is fill, not an observed direction/phase.
            chunk_axes=[a//c for a,c in zip(indices,product.chunks)]
            separator=product.metadata.get('dimension_separator','.')
            for key in itertools.product(*(np.unique(a) for a in chunk_axes)):
                selected=[np.flatnonzero(a==k) for a,k in zip(chunk_axes,key)]
                region=np.ix_(*selected)
                if not valid[region].any():continue
                values[name][region]=product.sample([a[s] for a,s in zip(indices,selected)])
                row=product._ledger.execute('SELECT sha FROM chunks WHERE key=?',
                    (separator.join(map(str,key)),)).fetchone()
                if row==(digest('authoritative-zarr-fill'),):
                    valid[region]=False
        ct=self.volume.sample(ct_axes)
        valid &= np.isfinite(ct)
        if self.config.get('invalid_value') is not None:valid &= ct!=self.config['invalid_value']
        if self.mask:valid &= values[self.mask]>0
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            result=decode_observations(*(values[n] for n in CHANNELS),
                input_spacing_xyz=self.identity['input_spacing_xyz'],encode_scale=self.manifest['grad_mag_encode_scale'],
                grad_mag_factor=self.manifest['grad_mag_factor'],validity=valid)
        if not all(np.isfinite(result[n]).all() for n in ('density','normals')):
            raise ValueError('Lasagna physical observations exceed finite float32 range')
        result['ct']=ct
        return result

    def validate_used(self):
        for product in self.products.values():product.validate_used()

    def close(self):
        self.stack.close()
        atomic_json(self.root/'CHUNKS.json',{name+'.sqlite':file_sha(self.root/(name+'.sqlite')) for name in self.products})

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()
