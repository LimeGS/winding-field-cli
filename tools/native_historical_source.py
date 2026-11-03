"""Historical-only range adapter; canonical field reconstruction stays unchanged.

No proof requested => tile_bounds returns None. A supplied proof must be complete
and SHA-bound; missing/changed/invalid proof fails closed, including on validate.
Bounds cover half-open NODE boxes relative to the PhysicalField view. Only strict
exclusion is safe: levels equal to either endpoint must still be extracted.
"""
from copy import deepcopy
import hashlib
import itertools
import math
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np

from field_store import _json_load,_read_file,_triple
from winding_field.surface_sources import _File
from winding_field.util import file_sha

HISTORICAL_PROOF=dict(
    config_sha256='4fdac93398fa748a51d4b43d0e27c89b041ef41ee880dff44df661b2d6594d81',
    extraction_sha256='7293f2d19c06ed604cdf6ab94e467afe884f21197840fb542dc99aeb23150f02')
COARSE_DETAIL_SHA='920f98cca270a7b66be08159cd261a68b2e87fd2d29a7aa44e7c5ef2ec481de9'


def _affine_read(source,low,high):
    """Return None unless both coordinate evaluation orders are exact binary64."""
    from scipy.ndimage import affine_transform
    delta=source.coarse_delta
    with np.errstate(all='ignore'):
        first=(source.origin+low[::-1]*source.spacing-delta.origin)/delta.spacing
        last=(source.origin+(high[::-1]-1)*source.spacing-delta.origin)/delta.spacing
        ratio=(source.spacing/delta.spacing)[::-1]
    if (not np.isfinite([first,last,ratio]).all() or (abs(first)>2**52).any() or (abs(last)>2**52).any()
        or (ratio<2**-20).any() or (ratio>1).any() or (np.frexp(ratio)[0]!=.5).any()):
        return None
    a=np.minimum(np.floor(first[::-1]).astype(int),np.array(delta.shape)-2)
    b=np.minimum(np.floor(last[::-1]).astype(int)+2,delta.shape)
    offset=first[::-1]-a; shift=offset/ratio
    # i*r+offset == (i+offset/r)*r exactly: integral numerator <=2**52, power-of-two r.
    if (shift<0).any() or (shift!=np.floor(shift)).any() or (shift+(high-low)-1>2**52).any():
        return None
    values,valid=source.base.read(low,high)
    coefficients,known=delta.read(a,b)
    if not known.all():raise ValueError('coarse correction coefficients are incomplete')
    correction=affine_transform(coefficients,ratio,offset=offset,output_shape=values.shape,
                                order=1,mode='constant',cval=np.nan,prefilter=False)
    if not np.isfinite(correction).all():raise ValueError('coarse correction does not cover native query')
    values*=source.base_gain; values+=correction
    return values,valid


class Historical:
    def __init__(self,physical,*,proof_root=None,proof_binding=None):
        self.physical=physical; self.shape=physical.shape
        self.origin=physical.origin_um; self.spacing=physical.spacing_um
        self._source_identity=deepcopy(physical.source.identity)
        self._files=[]; self._ranges={}; self._verified=set(); self._base=None; self._affine=None
        if self._source_identity.get('field',{}).get('code_sha256')==COARSE_DETAIL_SHA:
            code=Path(sys.modules[type(physical.source).__module__].__file__)
            self._affine=_File(code,COARSE_DETAIL_SHA); self._affine.read(); self._files.append(self._affine)
        self._identity={'kind':'pinned-historical-physical-field','source':physical.identity,
            'geometry':dict(shape_zyx=list(self.shape),origin_xyz=self.origin.tolist(),spacing_xyz=self.spacing.tolist(),
                voxel_um=1.,sheet_step=1.,frame=physical.source.frame),
            'adapter_sha256':file_sha(__file__),'tile_bounds_proof':None,
            'numerical':dict(coarse_interpolation='exact-dyadic-affine-or-canonical-v1' if self._affine else 'canonical',
                numpy=version('numpy'),scipy=version('scipy'),oracle_sha256=COARSE_DETAIL_SHA if self._affine else None)}
        if proof_root is not None:
            self._load_proof(Path(proof_root),HISTORICAL_PROOF if proof_binding is None else proof_binding)
            self.validate()
        elif proof_binding is not None:
            raise ValueError('proof binding requires a proof root')

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _document(self,path,expected=None):
        file=_File(path,expected or '0'*64)
        if file.size>16<<20:
            raise ValueError('historical proof metadata exceeds envelope')
        raw=_read_file(path,file.size); file.check()
        digest=hashlib.sha256(raw).hexdigest()
        if expected is not None and digest!=expected:
            raise ValueError('historical proof hash differs: '+str(path))
        file.digest=digest; self._files.append(file)
        return _json_load(raw),raw

    def _load_proof(self,root,binding):
        if root.is_symlink() or (root/'tiles').is_symlink():
            raise ValueError('historical proof directory cannot be a symlink')
        cfg,_=self._document(root/'CONFIG.json',binding['config_sha256'])
        extraction,_=self._document(root/'EXTRACTION.json',binding['extraction_sha256'])
        source=self.physical.source; field=source.identity['field']
        base=source.base; delta=source.coarse_delta
        if any(cfg.get('versions',{}).get(name)!=version(name) for name in ('numpy','scipy')):
            raise ValueError('historical proof numerical runtime versions differ')
        if (cfg['scope']!='FULL_Q1_NATIVE' or cfg['tile_indices'] is not None
            or type(cfg['tile']) is not int or not 1<=cfg['tile']<=256
            or field.get('kind')!='coarse-detail' or field['base']!=base.identity
            or field['coarse_delta']!=delta.identity or field['base_gain']!=cfg['alpha']
            or source.base_gain!=cfg['alpha'] or source.identity['geometry']!=cfg['geometry']
            or base.identity['field']!=cfg['native'] or base.identity['geometry']!=cfg['geometry']
            or delta.identity['field']!=cfg['delta'] or delta.identity['geometry']!=cfg['coarse_geometry']
            or cfg['native']['kind']!='chunks' or cfg['native']['support_policy']!='stored'
            or cfg['delta']['kind']!='npy' or base._file.digest!=cfg['native']['sha256']
            or delta._file.digest!=cfg['delta']['sha256']):
            raise ValueError('historical proof source recipe differs')
        code=Path(sys.modules[type(source).__module__].__file__)
        expected=cfg['pinned'][str(code)]
        file=_File(code,expected['sha256'],expected['bytes']); file.read(); self._files.append(file)
        if field['code_sha256']!=expected['sha256']:
            raise ValueError('historical correction code differs')
        reader_code=Path(sys.modules[type(base).__module__].__file__)
        decoder_code=Path(type(base)._load.__globals__['field_store'].__file__)
        for code in (reader_code,decoder_code):
            expected=cfg['pinned'][str(code)]
            file=_File(code,expected['sha256'],expected['bytes']); file.read(); self._files.append(file)
        for reader,descriptor in ((base,cfg['native']),(delta,cfg['delta'])):
            pinned=cfg['pinned'][descriptor['path']]
            if (pinned['sha256']!=descriptor['sha256'] or pinned['bytes']!=reader._file.size
                or str(reader._file.path)!=descriptor['path']):
                raise ValueError('historical proof reader/checkpoint binding differs')
        if len(base._payloads)!=math.prod((n-1)//base._chunk+1 for n in base.shape):
            raise ValueError('historical proof requires complete stored native nodes')
        self._base=base; self._tile=cfg['tile']
        shape=tuple(cfg['geometry']['shape_zyx'])
        grid=tuple((n-2)//self._tile+1 for n in shape)
        if (extraction['config_sha256']!=binding['config_sha256']
            or extraction['status']!='ALL_NATIVE_TILES_EXTRACTED' or extraction['scope']!=cfg['scope']
            or extraction['tiles']!=math.prod(grid) or extraction['total_cells']!=math.prod(n-1 for n in shape)):
            raise ValueError('historical proof extraction domain differs')
        chain=hashlib.sha256(); self._headers=self._files.copy()
        for index,key in enumerate(itertools.product(*(range(n) for n in grid))):
            row,raw=self._document(root/'tiles'/f'tile-{index:05d}.json')
            lo=[k*self._tile for k in key]; hi=[min(a+self._tile,n-1) for a,n in zip(lo,shape)]
            a,b=row['minimum'],row['maximum']
            if (row['config_sha256']!=binding['config_sha256'] or row['tile']!=index or row['key_zyx']!=list(key)
                or row['lo_zyx']!=lo or row['hi_zyx']!=hi or row['cells']!=math.prod(b-a for a,b in zip(lo,hi))
                or type(a) not in (int,float) or type(b) not in (int,float)
                or not math.isfinite(a) or not math.isfinite(b) or a>b):
                raise ValueError('historical proof tile domain/extrema differ')
            chain.update(raw); self._ranges[key]=(a,b,self._files[-1])
        if chain.hexdigest()!=extraction['tiles_sha256']:
            raise ValueError('historical proof receipt chain differs')
        self._identity['tile_bounds_proof']=dict(binding,tiles_sha256=chain.hexdigest(),tiles=math.prod(grid),
            tile_cells=self._tile,root=str(root),integrity='SHA once per covered native payload; immutable signature on every use')

    def validate(self,hashes=False):
        self.physical.validate(hashes=hashes)
        if self.physical.identity!=self._identity['source'] or self.physical.source.identity!=self._source_identity:
            raise ValueError('historical source identity changed')
        for file in self._files:file.check()
        if hashes and self._base is not None:
            self._verified.update(self._base._payloads)

    def _labels(self,lo,hi):
        return ( '-'.join(map(str,key)) for key in itertools.product(*(
            range(a//self._base._chunk,(b-1)//self._base._chunk+1) for a,b in zip(lo,hi))))

    def tile_bounds(self,low,high):
        lo=_triple(low); hi=_triple(high)
        if any(b<=a or b>n for a,b,n in zip(lo,hi,self.shape)):
            raise ValueError('bounds outside field view')
        if self._base is None or any(b-a<2 for a,b in zip(lo,hi)):
            return None
        if self.physical.source.identity!=self._source_identity or self.physical.identity!=self._identity['source']:
            raise ValueError('historical source identity changed')
        lo=tuple(a+b for a,b in zip(lo,self.physical.lo)); hi=tuple(a+b for a,b in zip(hi,self.physical.lo))
        for file in self._headers:file.check()
        selected=[self._ranges[key] for key in itertools.product(*(
            range(a//self._tile,(b-2)//self._tile+1) for a,b in zip(lo,hi)))]
        for _,_,file in selected:file.check()
        # Use the reader's pinned payload snapshots; never decode or trust index-only bytes.
        self._base._validate(payloads=False); self.physical.source.coarse_delta.validate()
        for label in self._labels(lo,hi):
            file=self._base._payloads[label]; file.check()
            if label not in self._verified:
                file.read(); self._verified.add(label)
        self._base._validate(payloads=False)
        for _,_,file in selected:file.check()
        return min(a for a,_,_ in selected),max(b for _,b,_ in selected)

    def read(self,lo,hi):
        lo=list(map(int,lo)); hi=list(map(int,hi)); result=None
        if self._affine is not None and self.physical._binding_check is None:
            lo=_triple(lo); hi=_triple(hi)
            if any(b<=a or b>n for a,b,n in zip(lo,hi,self.shape)):
                raise ValueError('read outside ROI')
            self._affine.check()
            result=_affine_read(self.physical.source,np.array(lo)+self.physical.lo,np.array(hi)+self.physical.lo)
            self._affine.check()
            if result is not None:
                self.physical.stats['read_calls']+=1
                self.physical.stats['crop_bytes']+=sum(a.nbytes for a in result)
        if result is None:
            result=self.physical.read(lo,hi)
        if self._base is not None:
            self._verified.update(self._labels([a+b for a,b in zip(lo,self.physical.lo)],
                                               [a+b for a,b in zip(hi,self.physical.lo)]))
        return result
