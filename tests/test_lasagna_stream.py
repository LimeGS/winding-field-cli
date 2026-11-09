import importlib.util
import json
from contextlib import closing

import numpy as np
import pytest

from ct_fixture import write_ct
from winding_field.io import Volume as _Volume
from winding_field.util import atomic_json, digest


def Volume(*args,**kwargs):
    return closing(_Volume(*args,**kwargs))


def reader_type():
    assert importlib.util.find_spec('winding_field.lasagna_stream'), 'external Lasagna reader missing'
    from winding_field.lasagna_stream import LasagnaInput
    return LasagnaInput


def fixture(tmp_path, *, crops=None, source_to_base=1):
    ct=write_ct(tmp_path/'ct',np.full((9,10,11),33,np.uint8),spacing=(5.,10.,20.),origin=(100.,200.,300.),chunk=4)
    root=tmp_path/'external'; root.mkdir()
    groups={}
    for name,value,level in [('grad_mag',100,2),('nx',255,2),('ny',128,2),('cos',255,1)]:
        scale=2**level; shape=tuple((n+scale-1)//scale for n in (9,10,11))
        folder=write_ct(root/(name+'.ome.zarr'),np.full(shape,value,np.uint8),chunk=4)
        (folder/'0').rename(folder/str(level))
        atomic_json(folder/'.zattrs',{'multiscales':[{'axes':[{'name':a,'unit':'pixel'} for a in 'zyx'],
            'datasets':[{'path':str(level),'coordinateTransformations':[{'type':'scale','scale':[scale]*3}]}]}]})
        groups[name]=dict(zarr=folder.name+'/'+str(level),scaledown=level,channels=[name])
    manifest=dict(version=2,source_to_base=source_to_base,grad_mag_encode_scale=1000.,grad_mag_factor=1.,
                  base_shape_zyx=[9,10,11],groups=groups,provenance='inference.json')
    if crops is not None: manifest['crops']=crops
    atomic_json(root/'input.lasagna.json',manifest)
    atomic_json(root/'inference.json',dict(status='completed',repository=dict(revision='custom',dirty=True)))
    config=dict(lasagna_input=str(root/'input.lasagna.json'),cache=str(tmp_path/'cache'),cache_bytes=1<<20,reserve_bytes=0,
                invalid_value=0,prediction='classical',chunk=4,threshold=10,threads=1)
    return ct,root,config


def geometry(volume):
    return dict(shape_zyx=list(volume.shape),spacing_xyz=volume.spacing_xyz,origin_xyz=volume.origin_xyz,
                frame=digest(volume.identity),roi_lo_zyx=[0,0,0],roi_hi_zyx=list(volume.shape),stride=1)


@pytest.mark.parametrize('factor,expected',[(1,.02),(2,.01)])
def test_mixed_levels_keep_input_pitch_and_ct_binding(tmp_path,factor,expected):
    cls=reader_type(); ct,root,config=fixture(tmp_path,source_to_base=factor)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with cls(volume,tmp_path/'run',config) as reader:
            result=reader.read(geometry(volume),[0,0,0],[9,10,11])
            assert result['support'].all()
            np.testing.assert_allclose(result['density'],expected,rtol=1e-6)
            assert reader.identity['binding']['kind']=='user-selected-ct-base-grid'
            assert reader.identity['provenance']['repository']['dirty'] is True
            assert reader.identity['kind']=='external-lasagna-v2'


def test_coverage_union_and_invalid_ct_never_become_support(tmp_path):
    cls=reader_type(); ct,root,config=fixture(tmp_path,crops=[[0,0,0,4,10,9],[8,0,0,3,10,9]])
    # The invalid chunk is genuine present data, not a missing chunk.
    (ct/'0/0.0.0').write_bytes(bytes(64))
    with Volume(ct,cache=tmp_path/'ct-cache') as volume, cls(volume,tmp_path/'run',config) as reader:
        result=reader.read(geometry(volume),[0,0,0],[9,10,11])
        assert not result['support'][:,:,:][...,4:8].any()
        assert not result['support'][:4,:4,:4].any()
        assert result['support'][4,4,0]
        assert not result['density'][~result['support']].any()


@pytest.mark.parametrize('changed',['chunk','manifest','provenance','geometry'])
def test_resume_rejects_mutated_external_inputs(tmp_path,changed):
    cls=reader_type(); ct,root,config=fixture(tmp_path); run=tmp_path/'run'
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with cls(volume,run,config) as reader:
            reader.read(geometry(volume),[0,0,0],[4,4,4])
        if changed=='chunk': (root/'grad_mag.ome.zarr/2/0.0.0').write_bytes(bytes(64))
        else:
            path={'manifest':root/'input.lasagna.json','provenance':root/'inference.json','geometry':root/'nx.ome.zarr/.zattrs'}[changed]
            data=json.loads(path.read_bytes()); data['changed']=True; atomic_json(path,data)
        with pytest.raises(ValueError,match='changed|identity'):
            with cls(volume,run,config) as reader: reader.validate_used()


def test_missing_local_prediction_chunk_fails_closed(tmp_path):
    cls=reader_type(); ct,root,config=fixture(tmp_path)
    (root/'nx.ome.zarr/2/0.0.0').unlink()
    with Volume(ct,cache=tmp_path/'ct-cache') as volume, cls(volume,tmp_path/'run',config) as reader:
        with pytest.raises(FileNotFoundError):reader.read(geometry(volume),[0,0,0],[4,4,4])


def test_classical_fibres_external_surface_bypasses_runtime(tmp_path,monkeypatch):
    reader_type(); ct,root,config=fixture(tmp_path)
    from winding_field import lasagna,predict
    from winding_field.artifacts import Tiles
    def forbidden(*a,**kw):pytest.fail('imported surfaces must not invoke Lasagna inference')
    monkeypatch.setattr(lasagna,'runtime_identity',forbidden)
    monkeypatch.setattr(lasagna,'checkpoint',forbidden)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        result=predict.prepare(volume,tmp_path/'run',config)
        assert result['surface_method']=='lasagna'
        assert result['surface_origin']=='external'
        assert result['fibre_method']=='classical'
        maps=Tiles(tmp_path/'run/maps')
        try:
            data=maps.load((0,0,0))
            assert np.isfinite(data['tangents']).all()
            np.testing.assert_allclose(data['density'],.02,rtol=1e-6)
        finally:maps.close()
        lasagna.validate_retained(tmp_path/'run',config)


def test_bounded_surface_reader_never_stages_native_volume(tmp_path,monkeypatch):
    reader_type(); ct,root,config=fixture(tmp_path)
    from winding_field import bounded_pipeline
    def forbidden(*a,**kw):pytest.fail('import must bypass model/runtime')
    monkeypatch.setattr(bounded_pipeline,'checkpoint',forbidden)
    monkeypatch.setattr(bounded_pipeline,'runtime_identity',forbidden)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with bounded_pipeline.surface_maps(volume,tmp_path/'run',config,np.zeros(3,dtype=int),np.array(volume.shape)) as (maps,identity):
            data=maps.load((0,0,0))
            np.testing.assert_allclose(data['density'],.02,rtol=1e-6)
            assert identity['kind']=='external-lasagna-v2'
    assert not (tmp_path/'run/lasagna-source/input.zarr').exists()


@pytest.fixture
def http_root(tmp_path):
    from functools import partial
    from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
    from threading import Thread
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(tmp_path)))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port)
    finally:server.shutdown();server.server_close();thread.join()


def test_remote_missing_normal_is_masked_and_offline_replay_is_identical(tmp_path,http_root):
    cls=reader_type(); ct,root,config=fixture(tmp_path)
    (root/'nx.ome.zarr/2/0.0.0').unlink()
    config['lasagna_input']=http_root+'/external/input.lasagna.json'
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with cls(volume,tmp_path/'run',config) as reader:
            data=reader.read(geometry(volume),[0,0,0],[4,4,4])
            assert not data['support'].any(), '404 normal fill is not a measured normal'
            assert sum(p.stats()['downloaded_bytes'] for p in reader.products.values())<=4*64
        with cls(volume,tmp_path/'run',dict(config,offline=True)) as reader:
            reader.validate_used()
            warm=reader.read(geometry(volume),[0,0,0],[4,4,4])
            for name in data:np.testing.assert_array_equal(warm[name],data[name])
            assert sum(p.stats()['downloads'] for p in reader.products.values())==0


def test_remote_cache_cap_and_offline_miss_fail_closed(tmp_path,http_root):
    cls=reader_type(); ct,root,config=fixture(tmp_path)
    config.update(lasagna_input=http_root+'/external/input.lasagna.json',cache_bytes=64)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with cls(volume,tmp_path/'run',config) as reader:
            reader.read(geometry(volume),[0,0,0],[4,4,4])
            assert all(p.stats()['cache_bytes']<=64 for p in reader.products.values())
        with cls(volume,tmp_path/'run',dict(config,offline=True)) as reader:
            with pytest.raises(FileNotFoundError):reader.validate_used()


@pytest.mark.parametrize('mutation',['shape','axes','translation','scale','channels','encoding','ct_source'])
def test_rejects_unbound_or_conflicting_geometry(tmp_path,mutation):
    cls=reader_type(); ct,root,config=fixture(tmp_path)
    path=root/'input.lasagna.json'; data=json.loads(path.read_bytes())
    if mutation=='shape':data['base_shape_zyx'][0]+=1
    if mutation=='channels':data['groups']['nx']['channels']=['ny']
    if mutation=='encoding':data['grad_mag_encode_scale']=0
    if mutation=='ct_source':data['ct_source']={'source':'another-scan'}
    if mutation in ('axes','translation','scale'):
        path=root/'nx.ome.zarr/.zattrs';data=json.loads(path.read_bytes());ms=data['multiscales'][0]
        if mutation=='axes':ms['axes'].reverse()
        if mutation=='translation':ms['datasets'][0]['coordinateTransformations'].append(dict(type='translation',translation=[1,0,0]))
        if mutation=='scale':ms['datasets'][0]['coordinateTransformations'][0]['scale']=[8]*3
    atomic_json(path,data)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with pytest.raises(ValueError):cls(volume,tmp_path/'run',config)


def test_external_afv_and_imported_surfaces_workflow_and_resume(tmp_path,monkeypatch):
    reader_type();ct,root,config=fixture(tmp_path)
    from test_splines import upstream_curves
    from winding_field import workflow,lasagna
    afv=tmp_path/'external.afv';upstream_curves(afv,(9,10,11),[(1,np.array([[1.,2.,3.],[5.,2.,3.]]))])
    descriptor=tmp_path/'geometry.json'
    atomic_json(descriptor,dict(source=str(ct),origin_xyz_um=[100.,200.,300.],spacing_xyz_um=[5.,10.,20.]))
    config.update(source=str(ct),output=str(tmp_path/'run'),fibres=str(afv),fibres_geometry=str(descriptor))
    def forbidden(*args,**kwargs):pytest.fail('imported surfaces/AFV must bypass all model preflight')
    monkeypatch.setattr(lasagna,'runtime_identity',forbidden)
    monkeypatch.setattr(lasagna,'checkpoint',forbidden)
    # Distribution metadata is outside this copied-source integration test.
    monkeypatch.setattr(workflow,'implementation_sha',lambda:'copied-source-fixture')
    planned=workflow.plan(config)
    assert planned['estimated_lasagna_staging_bytes']==planned['estimated_lasagna_workspace_bytes']==0
    result=workflow.run(config,until='fibres')
    assert set(result['stages'])=={'prepare','fibres'}
    exported=json.loads((tmp_path/'run/lasagna/surfaces.lasagna.json').read_bytes())
    assert 'cos' in exported['groups'], 'existing consumers must retain Lasagna phase selection'
    receipt=json.loads((tmp_path/'run/FIBRES.json').read_bytes())
    assert receipt['method']=='external-afv' and receipt['count']>0
    assert workflow.run(config,resume=True,until='fibres')['stages']==result['stages']
    (root/'grad_mag.ome.zarr/2/0.0.0').write_bytes(bytes(64))
    with pytest.raises(ValueError,match='changed'):workflow.run(config,resume=True,until='fibres')


def test_outside_manifest_coverage_does_not_fetch_missing_local_chunks(tmp_path):
    cls=reader_type();ct,root,config=fixture(tmp_path,crops=[[0,0,0,4,4,4]])
    for name in ('grad_mag','nx','ny','cos'):
        level=1 if name=='cos' else 2
        for path in (root/(name+'.ome.zarr')/str(level)).iterdir():
            if not path.name.startswith('.'):path.unlink()
    with Volume(ct,cache=tmp_path/'ct-cache') as volume, cls(volume,tmp_path/'run',config) as reader:
        result=reader.read(geometry(volume),[4,4,4],[8,8,8])
        assert not result['support'].any()
        assert not result['density'].any()


def test_explicit_mask_excludes_prediction_samples(tmp_path):
    cls=reader_type();ct,root,config=fixture(tmp_path)
    folder=write_ct(root/'mask.zarr',np.ones((9,10,11),np.uint8),chunk=4)
    (folder/'.zattrs').unlink()
    (folder/'0/0.0.0').write_bytes(bytes(64))
    path=root/'input.lasagna.json';manifest=json.loads(path.read_bytes())
    manifest['groups']['mask']=dict(zarr='mask.zarr/0',scaledown=0,channels=['mask']);atomic_json(path,manifest)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume, cls(volume,tmp_path/'run',config) as reader:
        result=reader.read(geometry(volume),[0,0,0],[8,8,8])
        assert not result['support'][:4,:4,:4].any()
        assert result['support'][4:,4:,4:].all()


def test_bounded_imported_prepare_and_completed_resume_require_no_lasagna_runtime(tmp_path,monkeypatch):
    reader_type();ct,root,config=fixture(tmp_path)
    path=ct/'.zattrs';attrs=json.loads(path.read_bytes())
    attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']=[8.64]*3
    atomic_json(path,attrs)
    from winding_field import workflow,lasagna,bounded_pipeline
    config.update(source=str(ct),output=str(tmp_path/'run'),prediction='qual-v2',
                  bounded_preparation=True,bounded_bytes=32<<20,field_spacing_um=17.,field_chunk=4)
    def forbidden(*args,**kwargs):pytest.fail('bounded imported preparation invoked inference')
    for module in (lasagna,bounded_pipeline):
        monkeypatch.setattr(module,'runtime_identity',forbidden)
        monkeypatch.setattr(module,'checkpoint',forbidden)
    monkeypatch.setattr(workflow,'implementation_sha',lambda:'copied-source-fixture')
    result=workflow.run(config,until='prepare')
    meta=json.loads((tmp_path/'run/PREPARED.json').read_bytes())
    assert meta['surface_origin']=='external'
    assert meta['calibration']['supported_voxels']==9*10*11
    np.testing.assert_allclose(meta['calibration']['mean_density_per_um'],.1/8.64,rtol=1e-6)
    assert not (tmp_path/'run/maps').exists()
    assert workflow.run(config,resume=True,until='prepare')['stages']==result['stages']


def test_cli_plan_accepts_imported_input_without_inference(tmp_path,capsys):
    ct,root,config=fixture(tmp_path)
    from winding_field.cli import main
    assert main(['plan','--source',str(ct),'--output',str(tmp_path/'run'),
        '--lasagna-input',config['lasagna_input'],'--prediction','classical','--cache',config['cache'],
        '--reserve-gib','0'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['estimated_lasagna_staging_bytes']==0


def test_selected_ct_level_and_stride_use_base_input_pitch(tmp_path):
    cls=reader_type();ct,root,config=fixture(tmp_path)
    # A second CT level sharing the exact same base frame.
    lower=write_ct(tmp_path/'lower',np.full((5,5,6),33,np.uint8),chunk=4)
    (lower/'0').rename(ct/'1')
    path=ct/'.zattrs';attrs=json.loads(path.read_bytes());dataset=dict(attrs['multiscales'][0]['datasets'][0])
    dataset['path']='1';dataset['coordinateTransformations']=[dict(type='scale',scale=[40.,20.,10.]),
        dict(type='translation',translation=[300.,200.,100.])]
    attrs['multiscales'][0]['datasets'].append(dataset);atomic_json(path,attrs)
    with Volume(ct,level=1,cache=tmp_path/'ct-cache') as volume,cls(volume,tmp_path/'run',config) as reader:
        g=geometry(volume);g.update(shape_zyx=[3,3,3],stride=2)
        result=reader.read(g,[0,0,0],[3,3,3])
        np.testing.assert_allclose(result['density'],.02,rtol=1e-6)


def test_decimal_ct_pitch_never_shifts_integer_channel_cells(tmp_path):
    cls=reader_type();ct,root,config=fixture(tmp_path)
    path=ct/'.zattrs';attrs=json.loads(path.read_bytes())
    attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']=[8.64]*3
    attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][1]['translation']=[1.e9]*3
    atomic_json(path,attrs)
    z,y,x=np.indices((4,4,4)); encoded=(10+z*20+y*4+x).astype(np.uint8)
    (root/'grad_mag.ome.zarr/2/0.0.0').write_bytes(encoded.tobytes())
    with Volume(ct,cache=tmp_path/'ct-cache') as volume,cls(volume,tmp_path/'run',config) as reader:
        result=reader.read(geometry(volume),[0,0,0],[9,10,11])
        z,y,x=np.indices((9,10,11));expected=(10+(z//4)*20+(y//4)*4+x//4)/1000/8.64
        np.testing.assert_allclose(result['density'],expected,rtol=1e-6)


@pytest.mark.parametrize('missing',['CHUNKS.json','nx.sqlite'])
def test_missing_resume_inventory_is_not_recreated(tmp_path,missing):
    cls=reader_type();ct,root,config=fixture(tmp_path);run=tmp_path/'run'
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with cls(volume,run,config) as reader:reader.read(geometry(volume),[0]*3,[4]*3)
        (run/'lasagna-import'/missing).unlink()
        with pytest.raises((ValueError,FileNotFoundError)):
            with cls(volume,run,config) as reader:reader.validate_used()


@pytest.mark.parametrize('encode_scale',[1.e-40,1.e300])
def test_external_density_encoding_cannot_publish_unrepresentable_values(tmp_path,encode_scale):
    cls=reader_type();ct,root,config=fixture(tmp_path)
    path=root/'input.lasagna.json';manifest=json.loads(path.read_bytes())
    manifest['grad_mag_encode_scale']=encode_scale;atomic_json(path,manifest)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume,cls(volume,tmp_path/'run',config) as reader:
        with pytest.raises(ValueError,match='finite|range'):
            reader.read(geometry(volume),[0]*3,[4]*3)


@pytest.mark.parametrize('producer',['standard','bounded'])
def test_partial_preparation_cannot_recreate_lost_import_provenance(tmp_path,producer):
    import shutil
    from winding_field import predict,bounded_pipeline
    ct,root,config=fixture(tmp_path);run=tmp_path/'run'
    config.update(field_spacing_um=40.,field_chunk=4)
    prepare=predict.prepare if producer=='standard' else bounded_pipeline.prepare
    marker=run/('maps/0_0_0.npz' if producer=='standard' else 'coarse/calibration-progress.json')
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        assert prepare(volume,run,config,stop=marker.exists)['status']=='PAUSED'
        shutil.rmtree(run/'lasagna-import')
        (root/'grad_mag.ome.zarr/2/0.0.0').write_bytes(bytes(64))
        with pytest.raises(ValueError,match='provenance|identity|inventory'):
            prepare(volume,run,config)


def test_retained_validation_rejects_missing_identity(tmp_path):
    from winding_field import lasagna
    cls=reader_type();ct,root,config=fixture(tmp_path);run=tmp_path/'run'
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with cls(volume,run,config) as reader:
            lasagna.prepare(volume,run,config,geometry(volume),reader=reader)
    (run/'lasagna-import/IDENTITY.json').unlink()
    with pytest.raises(ValueError,match='provenance|identity|inventory'):
        lasagna.validate_retained(run,config)


def test_offline_retained_validation_preserves_automatic_ct_mirror(tmp_path,monkeypatch):
    from winding_field import workflow,predict,lasagna
    ct,root,config=fixture(tmp_path)
    config.update(source='https://unavailable.invalid/ct',offline=True)
    monkeypatch.setenv('WINDING_CT_ROOTS',str(tmp_path))
    def forbidden(*args,**kwargs):pytest.fail('offline mirror must not use network')
    monkeypatch.setattr('winding_field.io.fetch',forbidden)
    with closing(workflow.open_volume(config)) as volume:
        predict.prepare(volume,tmp_path/'run',config)
    assert not list((tmp_path/'cache').glob('meta-*.json'))
    lasagna.validate_retained(tmp_path/'run',config)


def test_cancelling_large_density_scales_preserve_support(tmp_path):
    cls=reader_type();ct,root,config=fixture(tmp_path)
    path=root/'input.lasagna.json';manifest=json.loads(path.read_bytes())
    manifest.update(grad_mag_encode_scale=1.e40,grad_mag_factor=1.e40);atomic_json(path,manifest)
    with Volume(ct,cache=tmp_path/'ct-cache') as volume,cls(volume,tmp_path/'run',config) as reader:
        result=reader.read(geometry(volume),[0]*3,[4]*3)
        assert result['support'].all()
        np.testing.assert_allclose(result['density'],20.,rtol=1e-6)


def test_cli_offline_replay_preserves_saved_run_identity(tmp_path,http_root,monkeypatch,capsys):
    from winding_field import workflow
    from winding_field.cli import main
    ct,root,config=fixture(tmp_path);run=tmp_path/'run'
    monkeypatch.setattr(workflow,'implementation_sha',lambda:'copied-source-fixture')
    assert main(['run','--source',http_root+'/ct','--output',str(run),'--lasagna-input',
        http_root+'/external/input.lasagna.json','--prediction','classical','--surface-output','atlas',
        '--until','prepare','--chunk','4','--threads','1','--cache',config['cache'],
        '--cache-gib','.01','--reserve-gib','0','--threshold','10'])==0
    before=json.loads((run/'RUN.json').read_bytes())
    def forbidden(*args,**kwargs):pytest.fail('offline CLI replay attempted network')
    monkeypatch.setattr('winding_field.io.fetch',forbidden)
    monkeypatch.setattr('winding_field.lasagna_stream.fetch',forbidden)
    assert main(['prepare',str(run),'--offline'])==0
    after=json.loads((run/'RUN.json').read_bytes())
    assert before['config']==after['config']
    assert before['stages']==after['stages']
    assert before['source']==after['source']


@pytest.mark.parametrize('key',['original_ct','world_origin_xyz_um','world_spacing_xyz_um'])
def test_remote_physical_frame_is_not_silently_ignored(tmp_path,http_root,key):
    cls=reader_type();ct,root,config=fixture(tmp_path)
    path=root/'input.lasagna.json';manifest=json.loads(path.read_bytes())
    manifest[key]={'source':'different-scan'} if key=='original_ct' else [123.,456.,789.]
    atomic_json(path,manifest)
    config['lasagna_input']=http_root+'/external/input.lasagna.json'
    with Volume(ct,cache=tmp_path/'ct-cache') as volume:
        with pytest.raises(ValueError,match='physical|calibrated'):
            with cls(volume,tmp_path/'run',config):pass
