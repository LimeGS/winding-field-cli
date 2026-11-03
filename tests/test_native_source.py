import json
import hashlib
import itertools
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from tifxyz_fixture import field_run


@pytest.fixture(autouse=True)
def tools_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))


def test_historical_resume_hashes_without_reconstructing(tmp_path,monkeypatch):
    from bench_native_surfaces import Historical
    from winding_field.tifxyz_field import open_physical_field
    from winding_field._core import field_store
    root=field_run(tmp_path/'source')
    def no_decode(*args):
        pytest.fail('resume hash validation must not decode/reconstruct field crops')
    monkeypatch.setattr(field_store,'_decode',no_decode)
    with open_physical_field(root) as physical:
        Historical(physical).validate(hashes=True)
        assert physical.stats['read_calls']==0


def test_historical_resume_rejects_payload_corrupted_before_open(tmp_path):
    from bench_native_surfaces import Historical
    from winding_field.tifxyz_field import open_physical_field
    root=field_run(tmp_path/'source')
    checkpoint=json.loads((root/'field/CHECKPOINT.json').read_text())
    path=root/'field'/next(iter(checkpoint['chunks'].values()))['path']
    raw=path.read_bytes(); path.write_bytes(bytes([raw[0]^1])+raw[1:])
    with open_physical_field(root) as physical:
        with pytest.raises(ValueError,match='hash'):
            Historical(physical).validate(hashes=True)


class CoarseDetailFixture:
    """Canonical CoarseDetailSource.read arithmetic, over controlled readers."""
    def __init__(self,base,delta):
        self.base=base; self.coarse_delta=delta; self.base_gain=2.
        self.shape=base.shape; self.origin=base.origin; self.spacing=base.spacing
        self.frame=base.frame; self.voxel_um=base.voxel_um
        self.identity=dict(field=dict(kind='coarse-detail',base=base.identity,coarse_delta=delta.identity,
            base_gain=2.,formula='base_gain*native_old + trilinear(coarse_delta) at native nodes',
            code_sha256=sha(Path(__file__))),geometry=base.identity['geometry'])

    def validate(self,hashes=False):
        self.base.validate(hashes=hashes); self.coarse_delta.validate(hashes=hashes)

    def read(self,lo,hi):
        from scipy.ndimage import map_coordinates
        values,valid=self.base.read(lo,hi)
        low=np.asarray(lo); high=np.asarray(hi); delta=self.coarse_delta
        first=(self.origin+low[::-1]*self.spacing-delta.origin)/delta.spacing
        last=(self.origin+(high[::-1]-1)*self.spacing-delta.origin)/delta.spacing
        a=np.minimum(np.floor(first[::-1]).astype(int),np.array(delta.shape)-2)
        b=np.minimum(np.floor(last[::-1]).astype(int)+2,delta.shape)
        coefficients,known=delta.read(a,b)
        if not known.all():raise ValueError('coarse correction coefficients are incomplete')
        coords=np.indices(values.shape,dtype=float)*(self.spacing/delta.spacing)[::-1,None,None,None]
        coords+=(first[::-1]-a)[:,None,None,None]
        correction=map_coordinates(coefficients,coords,order=1,mode='constant',cval=np.nan,prefilter=False)
        if not np.isfinite(correction).all():raise ValueError('coarse correction does not cover native query')
        values*=self.base_gain; values+=correction
        return values,valid

    def close(self):
        self.base.close(); self.coarse_delta.close()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


@pytest.fixture
def historical(tmp_path):
    from winding_field._core.field_store import BlockStore
    from winding_field._core.surface_sources import FieldSource
    from winding_field.tifxyz_field import PhysicalField
    shape=(7,8,9); chunk=3; z,y,x=np.indices(shape)
    values=(x+2*y+3*z).astype(float)
    geometry=dict(shape_zyx=list(shape),origin_xyz=[0,0,0],spacing_xyz=[2,2,2],
                  voxel_um=8.64,sheet_step=1.,frame='PHerc0175A/20250521115057')
    with BlockStore(tmp_path/'field',shape,chunk,geometry) as store:
        for key in itertools.product(range(3),repeat=3):
            lo=np.array(key)*chunk; hi=np.minimum(lo+chunk,shape)
            store.commit(key,values[tuple(slice(a,b) for a,b in zip(lo,hi))],{},np.ones(hi-lo,bool))
    checkpoint=tmp_path/'field/CHECKPOINT.json'; delta=tmp_path/'delta.npy'
    np.save(delta,np.zeros((4,5,5)))
    native=dict(kind='chunks',path=str(checkpoint),sha256=sha(checkpoint),support_policy='stored')
    coarse=dict(kind='npy',path=str(delta),sha256=sha(delta))
    coarse_geometry=dict(geometry,shape_zyx=[4,5,5],spacing_xyz=[4,4,4])
    root=tmp_path/'proof'; root.mkdir(); (root/'tiles').mkdir()
    cfg=dict(scope='FULL_Q1_NATIVE',tile=chunk,tile_indices=None,alpha=2.,native=native,delta=coarse,
        geometry=geometry,coarse_geometry=coarse_geometry,
        versions={name:version(name) for name in ('numpy','scipy')},
        pinned={str(Path(__file__)):dict(bytes=Path(__file__).stat().st_size,sha256=sha(Path(__file__))),
                str(checkpoint):dict(bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)),
                str(delta):dict(bytes=delta.stat().st_size,sha256=sha(delta))})
    import winding_field._core.surface_sources as reader
    for path in (Path(reader.__file__),Path(reader.field_store.__file__)):
        cfg['pinned'][str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
    write_json(root/'CONFIG.json',cfg)
    chain=hashlib.sha256(); cells=0
    for index,key in enumerate(itertools.product(range(2),range(3),range(3))):
        lo=np.array(key)*chunk; hi=np.minimum(lo+chunk,np.array(shape)-1)
        data=2*values[tuple(slice(a,b+1) for a,b in zip(lo,hi))]
        row=dict(config_sha256=sha(root/'CONFIG.json'),tile=index,key_zyx=list(key),lo_zyx=lo.tolist(),
            hi_zyx=hi.tolist(),cells=int(np.prod(hi-lo)),minimum=float(data.min()),maximum=float(data.max()))
        path=root/'tiles'/f'tile-{index:05d}.json'; write_json(path,row)
        chain.update(path.read_bytes()); cells+=row['cells']
    extraction=dict(config_sha256=sha(root/'CONFIG.json'),status='ALL_NATIVE_TILES_EXTRACTED',
                    scope=cfg['scope'],tiles=18,total_cells=cells,tiles_sha256=chain.hexdigest())
    write_json(root/'EXTRACTION.json',extraction)
    binding=dict(config_sha256=sha(root/'CONFIG.json'),extraction_sha256=sha(root/'EXTRACTION.json'))
    opened=[]
    def physical(roi=None):
        p=PhysicalField(CoarseDetailFixture(FieldSource(native,geometry),FieldSource(coarse,coarse_geometry)),roi=roi)
        opened.append(p); return p
    yield root,binding,physical
    for p in opened:p.close()


def test_range_proof_preserves_field_and_qualifies_roi_and_shortened_tail(historical):
    from native_historical_source import Historical
    root,binding,open_field=historical
    p=open_field(); source=Historical(p,proof_root=root,proof_binding=binding)
    assert source.tile_bounds([0,0,0],[7,7,7])==(0.,72.)
    assert source.tile_bounds([6,7,8],[7,8,9]) is None  # no cells
    assert source.tile_bounds([3,6,6],[7,8,9])==(54.,80.)
    roi=dict(lo_zyx=[1,2,3],hi_zyx=[7,8,9])
    view=Historical(open_field(roi),proof_root=root,proof_binding=binding)
    assert view.tile_bounds([2,4,3],[6,6,6])==(54.,80.)
    original=p.read([3,6,6],[7,8,9]); actual=source.read([3,6,6],[7,8,9])
    for a,b in zip(original,actual):np.testing.assert_array_equal(a,b)
    assert actual[0].min()==54. and actual[0].max()==80.
    assert source.identity['tile_bounds_proof']['config_sha256']==binding['config_sha256']
    identity=source.identity; identity['tile_bounds_proof']['config_sha256']='wrong'
    assert source.identity['tile_bounds_proof']['config_sha256']==binding['config_sha256']
    assert view.identity!=source.identity


def test_missing_optional_proof_falls_back_but_supplied_bad_proof_rejects(historical):
    from native_historical_source import Historical
    root,binding,open_field=historical
    assert Historical(open_field()).tile_bounds([0,0,0],[4,4,4]) is None
    with pytest.raises((ValueError,FileNotFoundError)):
        Historical(open_field(),proof_root=root/'absent',proof_binding=binding)


def test_bounds_hash_only_relevant_payloads_once_and_recheck_cached_signatures(historical,monkeypatch):
    from native_historical_source import Historical
    from winding_field._core import field_store
    root,binding,open_field=historical; p=open_field()
    source=Historical(p,proof_root=root,proof_binding=binding)
    checked=[]
    file_type=type(p.source.base._file); read=file_type.read
    def counted(file):
        if file.path.suffix=='.npz':checked.append(file.path)
        return read(file)
    monkeypatch.setattr(file_type,'read',counted)
    monkeypatch.setattr(field_store,'_decode',lambda *a:pytest.fail('bounds must not decode'))
    assert source.tile_bounds([0,0,0],[4,4,4])==(0.,36.)
    assert len(checked)==8 and len(set(checked))==8
    assert source.tile_bounds([0,0,0],[4,4,4])==(0.,36.)
    assert len(checked)==8
    path=checked[-1]; raw=path.read_bytes(); path.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(ValueError,match='changed|hash'):
        source.tile_bounds([0,0,0],[4,4,4])


def test_bounds_reject_payload_corrupted_before_reader_open(historical):
    from native_historical_source import Historical
    root,binding,open_field=historical
    checkpoint=json.loads((root.parent/'field/CHECKPOINT.json').read_text())
    path=root.parent/'field'/checkpoint['chunks']['0-0-0']['path']
    raw=path.read_bytes(); path.write_bytes(bytes([raw[0]^1])+raw[1:])
    source=Historical(open_field(),proof_root=root,proof_binding=binding)
    with pytest.raises(ValueError,match='hash'):
        source.tile_bounds([0,0,0],[4,4,4])


@pytest.mark.parametrize('member',['CONFIG.json','EXTRACTION.json','tiles/tile-00017.json'])
def test_validate_rejects_changed_proof_even_outside_last_query(historical,member):
    from native_historical_source import Historical
    root,binding,open_field=historical
    source=Historical(open_field(),proof_root=root,proof_binding=binding)
    source.tile_bounds([0,0,0],[4,4,4])
    path=root/member; path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='changed|hash'):
        source.validate()


@pytest.mark.parametrize('change',['minimum','box','index','count','recipe','reader'])
def test_proof_rejects_invalid_domain_extrema_and_source_binding(historical,change):
    from native_historical_source import Historical
    root,binding,open_field=historical; p=open_field()
    if change=='recipe':
        p.source.identity['field']['base_gain']=3.
    elif change=='reader':
        p.source.base._file.digest='0'*64
    else:
        path=root/'tiles/tile-00000.json'; row=json.loads(path.read_text())
        if change=='minimum':row['minimum']=float('nan'); path.write_text(json.dumps(row))
        else:
            if change=='box':row['hi_zyx']=[2,3,3]
            if change=='index':row['tile']=1
            write_json(path,row)
        ex=json.loads((root/'EXTRACTION.json').read_text())
        ex['tiles_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((root/'tiles').iterdir()))).hexdigest()
        if change=='count':ex['tiles']=17
        write_json(root/'EXTRACTION.json',ex); binding['extraction_sha256']=sha(root/'EXTRACTION.json')
    with pytest.raises(ValueError):Historical(p,proof_root=root,proof_binding=binding)


def test_bound_equality_keeps_surface_and_matches_unfiltered_geometry(historical):
    from native_historical_source import Historical
    from winding_field._core.surface_mesh import _tile_mesh
    root,binding,open_field=historical
    p=open_field(); source=Historical(p,proof_root=root,proof_binding=binding)
    lo=[0,0,0]; hi=[4,4,4]; bounds=source.tile_bounds(lo,hi)
    for level in (0.,12.,36.,100.):
        values,valid=p.read(lo,hi)
        expected=_tile_mesh(values,valid,np.array(lo),level,source.origin,source.spacing)
        if not bounds[0]<=level<=bounds[1]:
            assert len(expected[2])==0
            continue
        v,k=source.read(lo,hi)
        actual=_tile_mesh(v,k,np.array(lo),level,source.origin,source.spacing)
        assert actual[0]==expected[0]
        np.testing.assert_array_equal(actual[1],expected[1]); np.testing.assert_array_equal(actual[2],expected[2])


def test_bounds_accept_native_driver_numpy_coordinates_and_reject_fractional(historical):
    from native_historical_source import Historical
    root,binding,open_field=historical
    source=Historical(open_field(),proof_root=root,proof_binding=binding)
    assert source.tile_bounds(np.array([0,0,0]),np.array([4,4,4]))==(0.,36.)
    for lo,hi in [([.1,0,0],[4,4,4]),([0,0,0],[8,8,9]),([-1,0,0],[4,4,4]),([False,0,0],[4,4,4])]:
        with pytest.raises(ValueError):source.tile_bounds(lo,hi)


def test_changed_reader_code_and_numerical_versions_do_not_qualify(historical):
    from native_historical_source import Historical
    root,binding,open_field=historical
    cfg=json.loads((root/'CONFIG.json').read_text())
    cfg['versions']={'numpy':'0.0','scipy':'0.0'}
    write_json(root/'CONFIG.json',cfg); binding['config_sha256']=sha(root/'CONFIG.json')
    # Rebind the metadata chain: rejection must concern the live numerical reader.
    chain=hashlib.sha256()
    for path in sorted((root/'tiles').iterdir()):
        row=json.loads(path.read_text()); row['config_sha256']=binding['config_sha256']
        write_json(path,row); chain.update(path.read_bytes())
    ex=json.loads((root/'EXTRACTION.json').read_text())
    ex.update(config_sha256=binding['config_sha256'],tiles_sha256=chain.hexdigest())
    write_json(root/'EXTRACTION.json',ex); binding['extraction_sha256']=sha(root/'EXTRACTION.json')
    with pytest.raises(ValueError,match='runtime|version|reader'):
        Historical(open_field(),proof_root=root,proof_binding=binding)


def test_hash_validated_resume_and_real_read_populate_bound_integrity_cache(historical,monkeypatch):
    from native_historical_source import Historical
    root,binding,open_field=historical
    p=open_field(); source=Historical(p,proof_root=root,proof_binding=binding)
    source.read([0,0,0],[4,4,4])
    original=type(p.source.base._file).read
    def no_payload(file):
        if file.path.suffix=='.npz':pytest.fail('unchanged payload already SHA-validated')
        return original(file)
    monkeypatch.setattr(type(p.source.base._file),'read',no_payload)
    assert source.tile_bounds([0,0,0],[4,4,4])==(0.,36.)
    monkeypatch.undo()
    source.validate(hashes=True)
    monkeypatch.setattr(type(p.source.base._file),'read',no_payload)
    assert source.tile_bounds([3,6,6],[7,8,9])==(54.,80.)


@pytest.mark.parametrize('which',['checkpoint','coarse','code'])
def test_bounds_reject_live_source_metadata_mutation(historical,which):
    from native_historical_source import Historical
    root,binding,open_field=historical
    p=open_field(); source=Historical(p,proof_root=root,proof_binding=binding)
    if which=='checkpoint':
        path=root.parent/'field/CHECKPOINT.json'; path.write_bytes(path.read_bytes()+b' ')
    elif which=='coarse':
        path=root.parent/'delta.npy'; raw=path.read_bytes(); path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    else:
        # Changing the in-memory reconstruction recipe must also invalidate the proof.
        p.source.identity['field']['code_sha256']='0'*64
    with pytest.raises(ValueError,match='changed|differs'):
        source.tile_bounds([0,0,0],[4,4,4])


class ArrayField:
    def __init__(self,values,origin,spacing):
        self.values=np.asarray(values,float); self.shape=self.values.shape
        self.valid=np.ones(self.shape,bool)
        self.origin=np.array(origin,float); self.spacing=np.array(spacing,float)
        self.frame='analytic'; self.voxel_um=8.64
        self.identity=dict(field={'fixture':'array'},geometry=dict(frame=self.frame,shape_zyx=list(self.shape),
            origin_xyz=list(origin),spacing_xyz=list(spacing),voxel_um=self.voxel_um,sheet_step=1.))

    def read(self,lo,hi):
        region=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        return self.values[region].copy(),self.valid[region].copy()

    def validate(self,hashes=False):pass
    def close(self):pass


def affine_fixture(monkeypatch,*,spacing=(2,2,2),coarse_spacing=(16,16,16),
                   origin=(0,0,0),coarse_origin=(0,0,0),roi=None):
    import native_historical_source as adapter
    from winding_field.tifxyz_field import PhysicalField
    rng=np.random.default_rng(2841)
    base=ArrayField(rng.standard_normal((17,19,21)),origin,spacing)
    base.valid[2,3,4]=False
    delta=ArrayField(rng.standard_normal((25,27,29)),coarse_origin,coarse_spacing)
    source=CoarseDetailFixture(base,delta)
    physical=PhysicalField(source,roi=roi)
    # The small independent reference above replaces the external frozen file in unit tests.
    monkeypatch.setattr(adapter,'COARSE_DETAIL_SHA',sha(Path(__file__)),raising=False)
    return physical,adapter.Historical(physical)


@pytest.mark.parametrize('settings',[
    {},
    dict(spacing=(2,4,1),coarse_spacing=(4,4,8)),
    dict(spacing=(2,4,1),coarse_spacing=(16,16,8),origin=(16,-16,8),coarse_origin=(0,-32,0)),
    dict(roi=dict(lo_zyx=[3,5,7],hi_zyx=[17,19,21])),
])
def test_affine_read_exact_bits_support_and_stats_without_coordinate_grid(monkeypatch,settings):
    physical,source=affine_fixture(monkeypatch,**settings)
    lo=[1,2,3]; hi=list(physical.shape)
    expected,known=physical.read(lo,hi)
    def no_grid(*args,**kwargs):pytest.fail('qualified read must not allocate dense coordinate grid')
    monkeypatch.setattr(np,'indices',no_grid)
    actual,valid=source.read(np.array(lo),np.array(hi))
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    np.testing.assert_array_equal(valid,known)
    assert physical.stats==dict(read_calls=2,crop_bytes=2*(actual.nbytes+valid.nbytes))
    assert source.identity['numerical']['coarse_interpolation']=='exact-dyadic-affine-or-canonical-v1'
    assert source.identity['numerical']['scipy']==version('scipy')


@pytest.mark.parametrize('settings',[
    dict(spacing=(2,3,5),coarse_spacing=(16,20,32)),
    dict(origin=(.25,0,0)),
    dict(spacing=(2**-24,2,2),coarse_spacing=(16,16,16)),
])
def test_affine_unqualified_coordinates_use_canonical_reader(monkeypatch,settings):
    import scipy.ndimage
    physical,source=affine_fixture(monkeypatch,**settings)
    expected,known=physical.read([0,0,0],[7,9,11])
    def no_affine(*args,**kwargs):pytest.fail('unqualified coordinates must retain canonical interpolation')
    monkeypatch.setattr(scipy.ndimage,'affine_transform',no_affine)
    actual,valid=source.read([0,0,0],[7,9,11])
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    np.testing.assert_array_equal(valid,known)


def test_affine_unknown_source_code_keeps_canonical_reader(monkeypatch):
    import native_historical_source as adapter
    import scipy.ndimage
    physical,_=affine_fixture(monkeypatch)
    monkeypatch.setattr(adapter,'COARSE_DETAIL_SHA','0'*64)
    source=adapter.Historical(physical)
    def no_affine(*args,**kwargs):pytest.fail('unrecognized source code must use canonical read')
    monkeypatch.setattr(scipy.ndimage,'affine_transform',no_affine)
    expected,known=physical.read([0,0,0],[7,9,11]); actual,valid=source.read([0,0,0],[7,9,11])
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    np.testing.assert_array_equal(valid,known)


@pytest.mark.parametrize('failure',['unknown-coefficient','outside-coarse'])
def test_affine_keeps_coefficient_support_and_nan_boundary_failures(monkeypatch,failure):
    physical,source=affine_fixture(monkeypatch)
    delta=physical.source.coarse_delta
    if failure=='unknown-coefficient':
        delta.valid[0,0,0]=False; message='coefficients are incomplete'
    else:
        delta.values=delta.values[:2,:2,:2]; delta.valid=delta.valid[:2,:2,:2]
        delta.shape=delta.values.shape; message='does not cover native query'
    with pytest.raises(ValueError,match=message):physical.read([0,0,0],[17,19,21])
    with pytest.raises(ValueError,match=message):source.read([0,0,0],[17,19,21])


def test_affine_does_not_bypass_physical_binding_checks(monkeypatch):
    physical,_=affine_fixture(monkeypatch)
    from native_historical_source import Historical
    def changed():raise ValueError('binding changed')
    physical._binding_check=changed
    source=Historical(physical)
    with pytest.raises(ValueError,match='binding changed'):source.read([0,0,0],[7,9,11])


def test_affine_fallback_runs_physical_binding_check_once(monkeypatch):
    physical,source=affine_fixture(monkeypatch,spacing=(2,3,5),coarse_spacing=(16,20,32))
    calls=0
    def check():
        nonlocal calls
        calls+=1
    physical._binding_check=check
    source.read([0,0,0],[7,9,11])
    assert calls==1


def test_affine_and_range_proof_share_live_payload_integrity(historical,monkeypatch):
    import native_historical_source as adapter
    root,binding,open_field=historical; physical=open_field()
    monkeypatch.setattr(adapter,'COARSE_DETAIL_SHA',sha(Path(__file__)))
    source=adapter.Historical(physical,proof_root=root,proof_binding=binding)
    expected,known=physical.read([0,0,0],[4,4,4])
    def no_grid(*args,**kwargs):pytest.fail('qualified proof-backed read must use affine interpolation')
    monkeypatch.setattr(np,'indices',no_grid)
    actual,valid=source.read([0,0,0],[4,4,4])
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    np.testing.assert_array_equal(valid,known)
    assert source.tile_bounds([0,0,0],[4,4,4])==(0.,36.)
    source.validate()
    payload=physical.source.base._payloads['0-0-0'].path
    raw=payload.read_bytes(); payload.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(ValueError,match='changed|hash'):source.read([0,0,0],[4,4,4])
    with pytest.raises(ValueError,match='changed|hash'):source.tile_bounds([0,0,0],[4,4,4])
