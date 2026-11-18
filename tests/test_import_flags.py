import argparse
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from winding_field.cli import main
from winding_field.io import Volume
from winding_field.util import file_sha


def parse_imports(tokens,*,paths=()):
    from winding_field.input_settings import import_arguments,import_requests
    from winding_field.cli_settings import parse_options
    parser=argparse.ArgumentParser()
    parser.add_argument('--fibres')
    parser.add_argument('--fibres-geometry')
    parser.add_argument('--lasagna-input')
    parser.add_argument('--lasagna-geometry')
    import_arguments(parser)
    return import_requests(parse_options(parser,tokens,paths=paths))


def manifest_flags(manifest,base):
    result=[]
    for name in ('source_to_base','grad_mag_encode_scale','grad_mag_factor'):
        result+=['--lasagna-'+name.replace('_','-'),str(manifest[name])]
    result+=['--lasagna-base-shape-zyx',*map(str,manifest['base_shape_zyx'])]
    for key,group in manifest['groups'].items():
        result+=['--lasagna-channel',key,group['channels'][0],str(base/group['zarr']),str(group['scaledown'])]
        if group.get('metadata_sha256'):
            result+=['--lasagna-channel-metadata-sha256',key,group['metadata_sha256']]
    for crop in manifest.get('crops',[]):result+=['--lasagna-crop',*map(str,crop)]
    if manifest.get('provenance'):result+=['--lasagna-provenance',str(base/manifest['provenance'])]
    if manifest.get('required_transport'):result+=['--lasagna-required-transport',manifest['required_transport']]
    for location,pin in manifest.get('studio_stream',{}).get('metadata_sha256',{}).items():
        result+=['--lasagna-metadata-pin',location,pin]
    return result


def physical_flags(geometry,prefix):
    return ['--'+prefix+'source',geometry['source'],'--'+prefix+'origin-xyz-um',*map(str,geometry['origin_xyz_um']),
            '--'+prefix+'spacing-xyz-um',*map(str,geometry['spacing_xyz_um'])]


@pytest.mark.parametrize('embedded',[False,True])
def test_inline_afv_geometry_matches_real_import_and_retains_source_pin(tmp_path,embedded):
    from test_afv_import import fixture,curves
    from winding_field import afv
    root=tmp_path/'native';root.mkdir()
    meta,config=fixture(root,embedded=embedded)
    descriptor=dict(source=meta['source']['source'],origin_xyz_um=meta['origin_xyz'],spacing_xyz_um=meta['spacing_xyz'])
    inline=deepcopy(config);inline['fibres_geometry']=descriptor
    before=file_sha(config['fibres'])
    expected=afv.source_identity(config['fibres'],config.get('fibres_geometry'),meta['source'])
    actual=afv.source_identity(inline['fibres'],inline['fibres_geometry'],meta['source'])
    assert actual['sha256']==expected['sha256']==before
    for key in ('source','origin_xyz_um','spacing_xyz_um'):assert actual[key]==expected[key]
    afv.import_fibres(root,inline)
    assert curves(root)[0][:2]==(1,'V')
    np.testing.assert_allclose(curves(root)[0][2][0],[109.,218.,327.])
    assert file_sha(config['fibres'])==before


def test_calibrated_lasagna_mapping_retains_interpolation_and_density(tmp_path):
    from test_lasagna_input import source_case
    from winding_field import lasagna_input
    volume,geometry,config=source_case(tmp_path)
    try:
        manifest=json.loads(Path(config['lasagna_input']).read_bytes())
        base=Path(config['lasagna_input']).parent
        for group in manifest['groups'].values():group['zarr']=str(base/group['zarr'])
        physical=json.loads(Path(config['lasagna_geometry']).read_bytes())
        inline=dict(config,lasagna_input=manifest,lasagna_geometry=physical)
        assert lasagna_input.streamed(inline) is False
        with lasagna_input.CalibratedLasagnaInput(volume,tmp_path/'native',config,geometry) as native:
            expected=native.read(geometry,[0,0,0],[4,4,4])
        with lasagna_input.CalibratedLasagnaInput(volume,tmp_path/'inline',inline,geometry) as reader:
            actual=reader.read(geometry,[0,0,0],[4,4,4])
        for key in ('support','normals','density','surface_phase','ct'):
            np.testing.assert_array_equal(expected[key],actual[key])
        assert actual['density'][actual['support']].mean()==pytest.approx(.05)
    finally:volume.close()


def test_channel_crop_and_zero_flags_reach_real_streamed_reader(tmp_path):
    from test_lasagna_stream import fixture,geometry
    from winding_field.lasagna_stream import LasagnaInput
    from winding_field.lasagna_input import streamed
    ct,root,config=fixture(tmp_path,crops=[[0,0,0,4,10,9],[8,0,0,3,10,9]])
    manifest=json.loads(Path(config['lasagna_input']).read_bytes())
    declared=parse_imports(manifest_flags(manifest,root))
    assert streamed(declared) is True
    assert declared['lasagna_input']['crops']==manifest['crops']
    with closing(Volume(ct,cache=tmp_path/'cache')) as volume:
        with LasagnaInput(volume,tmp_path/'file',config) as reader:
            expected=reader.read(geometry(volume),[0,0,0],[9,10,11])
        with LasagnaInput(volume,tmp_path/'flags',dict(config,**declared)) as reader:
            actual=reader.read(geometry(volume),[0,0,0],[9,10,11])
            assert reader.identity['binding']['kind']=='user-selected-ct-base-grid'
    for key in ('support','density','normals','surface_phase','ct'):
        np.testing.assert_array_equal(actual[key],expected[key])
    assert not actual['support'][...,4:8].any()


def test_cli_physical_flags_need_no_geometry_or_manifest_file(tmp_path,monkeypatch):
    from test_lasagna_input import source_case,forbid_models
    volume,_,config=source_case(tmp_path,shape=(8,8,8),whole=True,spacing=(9.,)*3);volume.close()
    manifest=json.loads(Path(config['lasagna_input']).read_bytes())
    physical=json.loads(Path(config['lasagna_geometry']).read_bytes())
    flags=manifest_flags(manifest,Path(config['lasagna_input']).parent)+physical_flags(physical,'lasagna-')
    Path(config['lasagna_input']).unlink();Path(config['lasagna_geometry']).unlink()
    forbid_models(monkeypatch)
    out=tmp_path/'run'
    assert main(['run','--source',config['source'],'--prediction','classical','--device','cpu',
        '--output',str(out),'--cache',str(tmp_path/'cache'),'--reserve-gib','0','--threads','1',
        '--chunk','8','--until','prepare',*flags])==0
    saved=json.loads((out/'RUN.json').read_bytes())['config']
    assert saved['lasagna_geometry']==physical and isinstance(saved['lasagna_input'],dict)
    assert json.loads((out/'PREPARED.json').read_bytes())['surface_origin']=='external'
    assert main(['prepare',str(out),'--offline'])==0


def test_mixed_manifest_flags_replace_crops_and_keep_metadata_pins(tmp_path):
    manifest=dict(version=2,source_to_base=1.,grad_mag_encode_scale=1000.,grad_mag_factor=1.,
        base_shape_zyx=[4,5,6],crops=[[0,0,0,6,5,4]],groups={
            'cos':dict(zarr='cos/0',scaledown=0,channels=['cos'],metadata_sha256='a'*64)},
        required_transport='vc_reader.DiskCache',studio_stream=dict(metadata_sha256={'https://example.org/.zattrs':'b'*64}))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    result=parse_imports(['--lasagna-input',str(path),'--lasagna-crop','1','0','0','4','5','4'])
    actual=result['lasagna_input']
    assert actual['crops']==[[1,0,0,4,5,4]]
    assert actual['groups']['cos']['zarr']==str(tmp_path/'cos/0')
    assert actual['groups']['cos']['metadata_sha256']=='a'*64
    assert actual['studio_stream']['metadata_sha256']==manifest['studio_stream']['metadata_sha256']


def test_changed_inline_afv_context_cannot_reuse_validation_proof(tmp_path):
    from test_afv_import import fixture
    from winding_field import afv
    meta,config=fixture(tmp_path)
    config['fibres_geometry']=json.loads(Path(config['fibres_geometry']).read_bytes())
    verified=afv.source_identity(config['fibres'],config['fibres_geometry'],meta['source'])
    config['fibres_geometry']['origin_xyz_um'][0]+=1
    with pytest.raises(ValueError,match='context'):
        afv._checked_source(config,meta['source'],verified)


def test_lasagna_python_config_base_preserves_venv_executable_symlink(tmp_path):
    from winding_field.cli_settings import parse_options
    base=tmp_path/'settings';base.mkdir()
    target=tmp_path/'python';target.write_text('placeholder')
    (base/'venv-python').symlink_to(target)
    path=base/'cli.json';path.write_text('{"lasagna_python": "venv-python"}')
    parser=argparse.ArgumentParser();parser.add_argument('--lasagna-python')
    args=parse_options(parser,['--config',str(path)],lexical_paths=('lasagna_python',))
    assert args.lasagna_python==str(base/'venv-python')


def test_cli_config_channel_rows_are_typed_and_paths_use_file_base(tmp_path,monkeypatch):
    options=dict(lasagna_base_shape_zyx=[4,5,6],lasagna_channel=[
        ['cos','cos','cos/0',0],['nx','nx','nx/0',2]],lasagna_crop=[[0,0,0,6,5,4]])
    path=tmp_path/'options.json';path.write_text(json.dumps(options))
    other=tmp_path/'other';other.mkdir();monkeypatch.chdir(other)
    result=parse_imports(['--config',str(path)],paths=(('lasagna_channel',2),))
    groups=result['lasagna_input']['groups']
    assert groups['cos']['zarr']==str(tmp_path/'cos/0') and groups['cos']['scaledown']==0
    assert groups['nx']['scaledown']==2


def test_flag_ct_binding_comes_from_real_source_artifact(tmp_path):
    from test_lasagna_stream import fixture,geometry
    from winding_field.lasagna_stream import LasagnaInput
    ct,root,config=fixture(tmp_path)
    manifest=json.loads(Path(config['lasagna_input']).read_bytes())
    declared=parse_imports([*manifest_flags(manifest,root),'--lasagna-ct-source',str(ct),'--lasagna-ct-level','0'])
    with closing(Volume(ct)) as volume:
        assert declared['lasagna_input']['ct_source']==volume.identity
        with LasagnaInput(volume,tmp_path/'flags',dict(config,**declared)) as reader:
            assert reader.identity['binding']['kind']=='manifest-ct-source'
            assert reader.read(geometry(volume),[0,0,0],[2,2,2])['support'].all()


def test_clear_crops_and_mismatched_explicit_geometry_remain_distinct(tmp_path):
    manifest=dict(version=2,base_shape_zyx=[4,5,6],crops=[[0,0,0,3,5,4]],groups={})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    result=parse_imports(['--lasagna-input',str(path),'--no-lasagna-crops'])
    assert result['lasagna_input']['crops']==[]
    from test_afv_import import fixture
    from winding_field import afv
    meta,config=fixture(tmp_path,embedded=True)
    config['fibres_geometry']=dict(source=meta['source']['source'],origin_xyz_um=[-1.,-2.,-3.],
                                  spacing_xyz_um=meta['spacing_xyz'])
    with pytest.raises(ValueError,match='conflicts'):
        afv.source_identity(config['fibres'],config['fibres_geometry'],meta['source'])


def test_mixed_geometry_raw_pin_is_rechecked_before_afv_publication(tmp_path):
    from test_afv_import import fixture
    from winding_field import afv
    _,config=fixture(tmp_path)
    path=Path(config['fibres_geometry'])
    declared=parse_imports(['--fibres-geometry',str(path),'--fibres-spacing-xyz-um','9','9','9'])
    config.update(declared)
    def change():
        path.write_bytes(path.read_bytes()+b' ')
        return False
    with pytest.raises(ValueError,match='recipe.*changed'):
        afv.import_fibres(tmp_path,config,stop=change)
    assert not (tmp_path/'FIBRES.json').exists()


def test_embedded_lasagna_physical_frame_can_be_declared_without_metadata_json(tmp_path):
    from test_lasagna_input import source_case
    from winding_field.lasagna_input import streamed,CalibratedLasagnaInput
    volume,geometry,config=source_case(tmp_path,embedded=True)
    try:
        path=Path(config['lasagna_input']);manifest=json.loads(path.read_bytes())
        flags=manifest_flags(manifest,path.parent)+[
            '--lasagna-ct-source',config['source'],'--lasagna-ct-level','0',
            '--lasagna-embedded-origin-xyz-um','100','200','300',
            '--lasagna-embedded-spacing-xyz-um','5','10','20']
        declared=parse_imports(flags)
        assert streamed(declared) is False
        assert declared['lasagna_input']['original_ct']==volume.identity
        with CalibratedLasagnaInput(volume,tmp_path/'inline',dict(config,**declared),geometry) as reader:
            actual=reader.read(geometry,[0,0,0],[4,4,4])
            assert actual['density'][actual['support']].mean()==pytest.approx(.05)
    finally:volume.close()


def test_oversized_input_declaration_is_read_with_a_bounded_buffer(tmp_path):
    import tracemalloc
    from winding_field.input_settings import declaration
    path=tmp_path/'huge.json';path.write_bytes(b' '*4*1024**2)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError,match='budget'):
            declaration(path,limit=65536)
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert peak<1<<20


def test_changed_original_import_recipe_rejects_plan_before_output(tmp_path):
    from test_lasagna_input import source_case
    from winding_field.workflow import plan
    volume,_,config=source_case(tmp_path);volume.close()
    path=Path(config['lasagna_input'])
    config.update(parse_imports(['--lasagna-input',str(path),'--lasagna-grad-mag-factor','.5']))
    config.update(output=str(tmp_path/'run'),prediction='classical',roi_lo=[2]*3,roi_hi=[10]*3,stride=2)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='recipe.*changed'):plan(config)
    assert not (tmp_path/'run').exists()


def test_remote_lasagna_manifest_overrides_keep_url_base_and_raw_pin(tmp_path,monkeypatch):
    from winding_field import io
    from winding_field.input_settings import check_recipe_sources
    import hashlib
    url='https://example.org/data/manifest.json'
    raw=json.dumps(dict(version=2,grad_mag_factor=1.,groups={
        'cos':dict(zarr='cos/0',scaledown=0,channels=['cos'])})).encode()
    current=raw
    def fetch(source,**kwargs):
        assert source==url
        return current,{}
    monkeypatch.setattr(io,'fetch',fetch)
    result=parse_imports(['--lasagna-input',url,'--lasagna-grad-mag-factor','.5'])
    assert result['lasagna_input']['groups']['cos']['zarr']=='https://example.org/data/cos/0'
    assert result['lasagna_input']['grad_mag_factor']==.5
    assert result['input_recipe_sources']=={url:hashlib.sha256(raw).hexdigest()}
    check_recipe_sources(result)
    current=raw+b' '
    with pytest.raises(ValueError,match='recipe.*changed'):check_recipe_sources(result)
    result['offline']=True
    def forbidden(*a,**kw):pytest.fail('offline continuation fetched a remote recipe')
    monkeypatch.setattr(io,'fetch',forbidden)
    check_recipe_sources(result)
