import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest

from ct_fixture import write_ct
from winding_field.io import Volume
from winding_field.util import digest, file_sha


def importer():
    assert importlib.util.find_spec('winding_field.lasagna_input'), 'external Lasagna importer missing'
    from winding_field import lasagna_input
    return lasagna_input


def source_case(root, *, shape=(16,16,40), level=1, embedded=False, whole=False,spacing=(5.,10.,20.)):
    ct=np.full(shape,100,np.uint8); ct[4,4,4]=0
    source=write_ct(root/'ct',ct,spacing=spacing,origin=(100.,200.,300.))
    volume=Volume(source,cache=root/'cache')
    folder=root/'predictions'; folder.mkdir()
    groups={}
    for name,value in [('cos',255),('grad_mag',500.),('nx',255),('ny',128),('presence',255)]:
        scaledown=0 if name=='cos' else level
        size=tuple((n+(1<<scaledown)-1)//(1<<scaledown) for n in shape)
        data=np.full(size,value,np.float32 if name=='grad_mag' else np.uint8)
        if name=='cos': data[:]=np.arange(size[2],dtype=np.uint8)[None,None,:]*4
        write_ct(folder/name,data,chunk=4)
        groups[name]=dict(zarr=name+'/0',scaledown=scaledown,channels=[name])
    manifest=dict(version=2,source_to_base=2.,base_shape_zyx=list(shape),
                  grad_mag_encode_scale=1000.,grad_mag_factor=.5,groups=groups)
    descriptor=dict(source=str(source),origin_xyz_um=[100.,200.,300.],spacing_xyz_um=list(spacing))
    if embedded:
        manifest.update(original_ct=volume.identity,world_origin_xyz_um=descriptor['origin_xyz_um'],
                        world_spacing_xyz_um=descriptor['spacing_xyz_um'])
    path=folder/'surface.lasagna.json'; path.write_text(json.dumps(manifest))
    geometry_path=root/'geometry.json'; geometry_path.write_text(json.dumps(descriptor))
    lo=[0,0,0] if whole else [2,2,2]; hi=list(shape) if whole else [10,10,10]; stride=1 if whole else 2
    geometry=dict(shape_zyx=((np.array(hi)-lo-1)//stride+1).tolist(),
                  origin_xyz=(np.array(volume.origin_xyz)+np.array(lo)[::-1]*volume.spacing_xyz).tolist(),
                  spacing_xyz=(np.array(volume.spacing_xyz)*stride).tolist(),frame=digest(volume.identity),
                  roi_lo_zyx=lo,roi_hi_zyx=hi,stride=stride)
    config=dict(source=str(source),lasagna_input=str(path),lasagna_geometry=None if embedded else str(geometry_path),
                invalid_value=0,reserve_bytes=0,cache=str(root/'cache'))
    return volume,geometry,config


def test_import_samples_physical_multiresolution_grid_and_preserves_density(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    before={str(p):file_sha(p) for p in (tmp_path/'predictions').rglob('*') if p.is_file()}
    try:
        result=module.prepare(volume,tmp_path/'run',config,geometry)
        assert result['status']=='LASAGNA_READY'
        data=module.read_observations(tmp_path/'run',geometry,[0,0,0],[4,4,4])
        valid=np.ones((4,4,4),bool); valid[1,1,1]=False
        np.testing.assert_array_equal(data['support'],valid)
        np.testing.assert_allclose(data['density'][valid],.05,rtol=1e-6)
        np.testing.assert_allclose(data['normals'][valid],np.tile([1.,0.,0.],(63,1)))
        np.testing.assert_allclose(data['surface_phase'][0,0],[8/255,16/255,24/255,32/255])
        assert data['ct'][1,1,1]==0 and data['density'][1,1,1]==0
        module.validate_retained(tmp_path/'run',config)
    finally: volume.close()
    assert before=={str(p):file_sha(p) for p in (tmp_path/'predictions').rglob('*') if p.is_file()}


def forbid_models(monkeypatch):
    from winding_field import lasagna,models,predict,splines
    def forbidden(*args,**kwargs): raise AssertionError('external inputs must not invoke models')
    for module,name in [(lasagna,'runtime_identity'),(lasagna,'checkpoint'),(lasagna,'run_child'),
                        (models,'checkpoint'),(predict,'QualV2'),(predict,'SD2'),(splines,'trace_qual')]:
        monkeypatch.setattr(module,name,forbidden)


def test_public_cli_plans_import_without_a_model_runtime(tmp_path,monkeypatch,capsys):
    from winding_field.cli import main
    volume,geometry,config=source_case(tmp_path); volume.close()
    forbid_models(monkeypatch)
    assert main(['plan','--source',config['source'],'--output',str(tmp_path/'run'),'--cache',config['cache'],
                 '--prediction','classical','--lasagna-input',config['lasagna_input'],
                 '--lasagna-geometry',config['lasagna_geometry'],'--roi','2','10','2','10','2','10',
                 '--stride','2','--field-spacing-um','17'])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['surface_method']=='external-lasagna'
    assert plan['estimated_lasagna_staging_bytes']==0 and plan['estimated_lasagna_workspace_bytes']==0
    assert plan['estimated_lasagna_import_bytes']>0
    assert plan['field']['spacing_xyz']==[17.]*3 and not (tmp_path/'run').exists()


def test_external_surfaces_and_fibres_feed_mesh_and_reimport_without_models(tmp_path,monkeypatch):
    from test_splines import upstream_curves
    from winding_field.artifacts import Tiles
    from winding_field.workflow import run,plan
    volume,geometry,config=source_case(tmp_path); volume.close()
    fibres=tmp_path/'fibres.afv'
    upstream_curves(fibres,(16,16,40),[(1,np.array([[4.,2.,6.],[4.,8.,6.]]))])
    config.update(output=str(tmp_path/'run'),prediction='qual-v2',fibres=str(fibres),
                  fibres_geometry=config['lasagna_geometry'],roi_lo=[2]*3,roi_hi=[10]*3,stride=2,
                  chunk=4,field_chunk=4,sweeps=1,maxiter=200,threads=1,previews=False)
    forbid_models(monkeypatch)
    estimate=plan(config)
    assert estimate['input_lo_zyx']==[2]*3 and estimate['input_hi_zyx']==[10]*3
    state=run(config,until='mesh'); root=tmp_path/'run'
    assert set(state['stages'])=={'prepare','fibres','field','mesh'}
    assert json.loads((root/'FIELD.json').read_bytes())['fibre_constraint_rows']>0
    assert list((root/'meshes').glob('level-*.obj'))
    assert json.loads((root/'PREPARED.json').read_bytes())['input_halo_voxels']==0
    with_files=['LASAGNA.json','PREPARED.json','FIBRES.json','FIELD.json','MESHES.json','field/CHECKPOINT.json']
    before={name:file_sha(root/name) for name in with_files}
    run(state['config'],resume=True,until='mesh')
    assert before=={name:file_sha(root/name) for name in with_files}
    second=dict(config,output=str(tmp_path/'reimport'),lasagna_input=str(root/'lasagna/surfaces.lasagna.json'),lasagna_geometry=None)
    run(second,until='prepare')
    original=Tiles(root/'maps'); reimported=Tiles(tmp_path/'reimport/maps')
    try:
        for key in original.keys():
            a,b=original.load(key),reimported.load(key)
            for name in ('density','normals','support','surface_phase','ct'):
                np.testing.assert_allclose(a[name],b[name],rtol=1e-6,atol=1e-7)
    finally: original.close(); reimported.close()
    (tmp_path/'predictions/nx/0/0.0.0').write_bytes(bytes([128])*4**3)
    with pytest.raises(ValueError,match='changed'):
        run(state['config'],resume=True,until='mesh')
    assert before=={name:file_sha(root/name) for name in with_files}


@pytest.mark.parametrize('method',['classical','qual-v2'])
def test_imported_surfaces_remain_independent_of_local_fibre_predictions(tmp_path,monkeypatch,method):
    from winding_field import predict
    from winding_field.artifacts import Tiles
    from winding_field.workflow import run
    volume,geometry,config=source_case(tmp_path,shape=(8,8,8),whole=True,spacing=(9.,)*3); volume.close()
    config.update(output=str(tmp_path/'run'),prediction=method,chunk=8,threads=1)
    forbid_models(monkeypatch)
    if method=='qual-v2':
        class EmptyQual:
            def __init__(self,*args): pass
            def predict(self,ct):
                return np.zeros(ct.shape,np.float32),np.zeros(ct.shape+(3,),np.float32),np.zeros(ct.shape+(2,),np.float32)
        monkeypatch.setattr(predict,'QualV2',EmptyQual)
    state=run(config,until='prepare')
    maps=Tiles(tmp_path/'run/maps')
    try:
        values=maps.load((0,0,0))
        assert values['support'].any()
        assert not values['fibre_support'].any()
        assert json.loads((tmp_path/'run/PREPARED.json').read_bytes())['fibre_method']==method
        assert state['lasagna_input']['path']==config['lasagna_input']
    finally: maps.close()


@pytest.mark.parametrize('options',[
    dict(lasagna_geometry='geometry.json'),
    dict(lasagna_input='surface.lasagna.json',lasagna_model_path='model.pt'),
    dict(lasagna_input='surface.lasagna.json',lasagna_python='python'),
])
def test_external_lasagna_option_conflicts_fail_before_io(options):
    from winding_field.config import validate_config
    with pytest.raises(ValueError,match='lasagna'):
        validate_config(options)


def test_float_density_decode_has_no_uint8_ceiling():
    from winding_field.lasagna import decode_observations
    shape=(2,2,2)
    result=decode_observations(np.full(shape,500.,np.float32),np.full(shape,255,np.uint8),
        np.full(shape,128,np.uint8),np.full(shape,255,np.uint8),input_spacing_xyz=[5.,10.,20.],
        encode_scale=1000.,grad_mag_factor=.5,validity=np.ones(shape,bool))
    np.testing.assert_allclose(result['density'],.05,rtol=1e-6)


def test_sd2_fibres_cross_imported_surface_support_gaps(tmp_path,monkeypatch):
    import io
    from winding_field import predict
    from winding_field.workflow import run
    volume,_,config=source_case(tmp_path,shape=(8,8,8),level=0,whole=True,spacing=(9.,)*3)
    volume.close()
    for path in (tmp_path/'predictions/presence/0').glob('*.*.1'):
        path.write_bytes(bytes(4**3))
    forbid_models(monkeypatch)
    class StraightSD2:
        def __init__(self,*args): pass
        def predict(self,ct):
            tangents=np.zeros(ct.shape+(3,),np.float32); tangents[...,0]=1
            return np.ones(ct.shape,np.float32),tangents
    monkeypatch.setattr(predict,'SD2',StraightSD2)
    config.update(output=str(tmp_path/'run'),prediction='sd2',chunk=8,threads=1,invalid_value=None)
    run(config,until='fibres')
    with sqlite3.connect(tmp_path/'run/fibres/fibres.sqlite') as db:
        curves=[np.load(io.BytesIO(row[0]),allow_pickle=False) for row in db.execute('SELECT points FROM curves')]
    assert curves
    assert any(points[:,0].min()<136 and points[:,0].max()>136 for points in curves)


def test_import_interpolation_does_not_bridge_a_presence_gap(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    # Observation X=3 lies halfway between prediction X=2 and X=4.
    geometry.update(origin_xyz=[115.,230.,360.],roi_lo_zyx=[3,3,3],roi_hi_zyx=[11,11,11])
    path=tmp_path/'predictions/presence/0/0.0.0'
    data=np.frombuffer(path.read_bytes(),np.uint8).copy().reshape((4,)*3); data[:,:,2]=0
    path.write_bytes(data.tobytes())
    try:
        module.prepare(volume,tmp_path/'run',config,geometry)
        values=module.read_observations(tmp_path/'run',geometry,[0,0,0],[4,4,4])
        assert not values['support'][0,0,:2].any()
        assert values['support'][0,0,2:].all()
    finally: volume.close()


@pytest.mark.parametrize('next_x,normal,density',[
    (1,[1.,0.,0.],.05),
    (128,[4/np.sqrt(17),0.,1/np.sqrt(17)],.25*np.sqrt(17/800)),
])
def test_import_interpolates_decoded_unsigned_normals(tmp_path,next_x,normal,density):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    geometry.update(origin_xyz=[115.,220.,340.],roi_lo_zyx=[2,2,3],roi_hi_zyx=[10,10,11])
    path=tmp_path/'predictions/nx/0/0.0.0'
    values=np.frombuffer(path.read_bytes(),np.uint8).copy().reshape((4,)*3)
    values[:,:,2]=next_x; path.write_bytes(values.tobytes())
    try:
        module.prepare(volume,tmp_path/'run',config,geometry)
        data=module.read_observations(tmp_path/'run',geometry,[0]*3,[1]*3)
        assert data['support'].all()
        np.testing.assert_allclose(np.abs(data['normals'][0,0,0]),normal,atol=1e-6)
        np.testing.assert_allclose(data['density'],density,rtol=1e-6)
    finally: volume.close()


def test_import_rejects_normal_components_on_different_grids(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    path=Path(config['lasagna_input']); manifest=json.loads(path.read_text())
    manifest['groups']['ny'].update(zarr='cos/0',scaledown=0)
    path.write_text(json.dumps(manifest))
    try:
        with pytest.raises(ValueError,match='normal.*grid'):
            module.source_identity(path,config['lasagna_geometry'],volume.identity,geometry,volume.cache)
    finally: volume.close()


def test_import_stops_and_resumes_without_rewriting_committed_tiles(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path,shape=(8,8,72),level=0,whole=True)
    run=tmp_path/'run'; first=run/'lasagna-source/maps/0_0_0.npz'
    try:
        result=module.prepare(volume,run,config,geometry,lambda:first.exists())
        assert result['status']=='PAUSED' and not (run/'LASAGNA.json').exists()
        old=first.read_bytes(); mtime=first.stat().st_mtime_ns
        assert module.prepare(volume,run,config,geometry)['status']=='LASAGNA_READY'
        assert first.read_bytes()==old and first.stat().st_mtime_ns==mtime
        saved=(run/'LASAGNA.json').read_bytes()
        module.prepare(volume,run,config,geometry)
        assert (run/'LASAGNA.json').read_bytes()==saved
    finally: volume.close()


@pytest.mark.parametrize('mutation',['manifest','descriptor','metadata','chunk','tiles','ledger','identity'])
def test_completed_import_rejects_changed_or_missing_retained_inputs(tmp_path,mutation):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    run=tmp_path/'run'
    try:
        module.prepare(volume,run,config,geometry)
        saved=(run/'LASAGNA.json').read_bytes()
        paths=dict(manifest=Path(config['lasagna_input']),descriptor=Path(config['lasagna_geometry']),
                   metadata=tmp_path/'predictions/nx/0/.zarray',chunk=tmp_path/'predictions/nx/0/0.0.0',
                   tiles=run/'lasagna-source/maps/0_0_0.npz',ledger=run/'lasagna-source/nx.sqlite',
                   identity=run/'lasagna-source/IDENTITY.json')
        path=paths[mutation]
        if mutation in ('ledger','identity'): path.unlink()
        elif mutation in ('manifest','descriptor','metadata'): path.write_bytes(path.read_bytes()+b'\n')
        else: path.write_bytes(b'changed')
        with pytest.raises((ValueError,FileNotFoundError),match='changed|corrupt|missing|identity'):
            module.validate_retained(run,config)
        assert (run/'LASAGNA.json').read_bytes()==saved
    finally: volume.close()


@pytest.mark.parametrize('bad',['uncalibrated','source','conflict','spacing','shape','scale','channel','channels-type','incomplete','roi','crop'])
def test_import_preflight_rejects_ambiguous_or_invalid_inputs(tmp_path,bad):
    module=importer(); volume,geometry,config=source_case(tmp_path,embedded=bad=='conflict')
    path=Path(config['lasagna_input']); manifest=json.loads(path.read_text())
    descriptor_path=tmp_path/'geometry.json'; descriptor=json.loads(descriptor_path.read_text())
    if bad=='uncalibrated': config['lasagna_geometry']=None
    elif bad=='source': descriptor['source']='different-ct'
    elif bad=='conflict':
        config['lasagna_geometry']=str(descriptor_path); descriptor['origin_xyz_um'][0]+=5
    elif bad=='spacing': descriptor['spacing_xyz_um'][0]=0
    elif bad=='shape': manifest['base_shape_zyx'][0]+=1
    elif bad=='scale': manifest['grad_mag_factor']=0
    elif bad=='channel': del manifest['groups']['cos']
    elif bad=='channels-type': manifest['groups']['cos']['channels']=None
    elif bad=='incomplete':
        manifest['provenance']='inference.json'
        (path.parent/'inference.json').write_text(json.dumps(dict(status='running')))
    elif bad=='roi': geometry['origin_xyz'][0]=10000
    elif bad=='crop': manifest['crops']=[[0,0,0,2,2,2]]
    path.write_text(json.dumps(manifest)); descriptor_path.write_text(json.dumps(descriptor))
    try:
        with pytest.raises(ValueError):
            module.source_identity(path,config['lasagna_geometry'],volume.identity,geometry,volume.cache)
        assert not (tmp_path/'run').exists()
    finally: volume.close()


def test_import_does_not_extrapolate_beyond_stored_channel_centres(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path,whole=True)
    try:
        module.prepare(volume,tmp_path/'run',config,geometry)
        data=module.read_observations(tmp_path/'run',geometry,[0,0,0],[16,16,40])
        assert data['support'][0,0,38]
        assert not data['support'][-1].any()
        assert not data['support'][:,-1].any()
        assert not data['support'][:,:,-1].any()
    finally: volume.close()


def test_partial_import_rejects_changed_consumed_chunks(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path,shape=(8,8,72),level=0,whole=True)
    run=tmp_path/'run'; tile=run/'lasagna-source/maps/0_0_0.npz'
    try:
        assert module.prepare(volume,run,config,geometry,lambda:tile.exists())['status']=='PAUSED'
        saved=tile.read_bytes()
        (tmp_path/'predictions/nx/0/0.0.0').write_bytes(bytes([128])*4**3)
        with pytest.raises(ValueError,match='changed'):
            module.prepare(volume,run,config,geometry)
        assert tile.read_bytes()==saved and not (run/'LASAGNA.json').exists()
    finally: volume.close()


@pytest.mark.parametrize('mutation',['source-row','tile-index','tile-row','seal','identity'])
def test_partial_import_rejects_missing_inventory_before_reusing_tiles(tmp_path,mutation):
    module=importer(); volume,geometry,config=source_case(tmp_path,shape=(8,8,72),level=0,whole=True)
    run=tmp_path/'run'; root=run/'lasagna-source'; tile=root/'maps/0_0_0.npz'
    try:
        assert module.prepare(volume,run,config,geometry,lambda:tile.exists())['status']=='PAUSED'
        saved=tile.read_bytes(); mtime=tile.stat().st_mtime_ns
        if mutation=='source-row':
            with sqlite3.connect(root/'nx.sqlite') as db:
                assert db.execute("DELETE FROM chunks WHERE key='0.0.0'").rowcount==1
            (tmp_path/'predictions/nx/0/0.0.0').write_bytes(bytes([128])*4**3)
        elif mutation=='tile-index': (root/'maps/tiles.sqlite').unlink()
        elif mutation=='tile-row':
            with sqlite3.connect(root/'maps/tiles.sqlite') as db:
                assert db.execute("DELETE FROM tiles WHERE key='0_0_0'").rowcount==1
        elif mutation=='identity': (root/'IDENTITY.json').unlink()
        else: (root/'WORK.json').unlink(missing_ok=True)
        with pytest.raises(ValueError,match='changed|missing|unsealed'):
            module.prepare(volume,run,config,geometry)
        assert tile.read_bytes()==saved and tile.stat().st_mtime_ns==mtime
        assert not (run/'LASAGNA.json').exists()
    finally: volume.close()


def test_completed_import_requires_retained_source_directory(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    try:
        module.prepare(volume,tmp_path/'run',config,geometry)
        shutil.rmtree(tmp_path/'run/lasagna-source')
        with pytest.raises(ValueError,match='missing'):
            module.validate_retained(tmp_path/'run',config)
    finally: volume.close()


def test_import_seals_committed_work_after_a_missing_chunk_error(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path,shape=(8,8,72),level=0,whole=True)
    path=tmp_path/'predictions/nx/0/0.0.8'; raw=path.read_bytes(); path.unlink()
    run=tmp_path/'run'; first=run/'lasagna-source/maps/0_0_0.npz'
    try:
        with pytest.raises(FileNotFoundError): module.prepare(volume,run,config,geometry)
        assert (run/'lasagna-source/WORK.json').is_file()
        saved=first.read_bytes(); mtime=first.stat().st_mtime_ns
        path.write_bytes(raw)
        assert module.prepare(volume,run,config,geometry)['status']=='LASAGNA_READY'
        assert first.read_bytes()==saved and first.stat().st_mtime_ns==mtime
    finally: volume.close()


def test_embedded_cli_frame_needs_no_descriptor(tmp_path):
    module=importer(); volume,geometry,config=source_case(tmp_path,embedded=True)
    try:
        assert module.prepare(volume,tmp_path/'run',config,geometry)['status']=='LASAGNA_READY'
        values=module.read_observations(tmp_path/'run',geometry,[0,0,0],[4,4,4])
        np.testing.assert_allclose(values['density'][values['support']],.05,rtol=1e-6)
    finally: volume.close()


def test_bounded_plan_rejects_calibrated_embedded_frame_before_output(tmp_path):
    from winding_field.workflow import plan
    volume,_,config=source_case(tmp_path,embedded=True,shape=(8,8,8),whole=True,spacing=(8.64,)*3)
    volume.close()
    config.update(output=str(tmp_path/'run'),prediction='qual-v2',bounded_preparation=True,
                  bounded_bytes=32<<20,field_spacing_um=17.)
    with pytest.raises(ValueError,match='bounded.*calibrated|calibrated.*bounded'):
        plan(config)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('bad',['missing','nan','negative','cos-range'])
def test_bad_channel_payloads_cannot_publish_a_ready_import(tmp_path,bad):
    module=importer(); volume,geometry,config=source_case(tmp_path)
    path=tmp_path/'predictions/grad_mag/0/0.0.0'
    if bad=='missing': path.unlink()
    elif bad=='cos-range':
        manifest=Path(config['lasagna_input']); data=json.loads(manifest.read_text())
        data['groups']['cos']['zarr']='grad_mag/0'; data['groups']['cos']['scaledown']=1
        manifest.write_text(json.dumps(data))
    else:
        raw=np.frombuffer(path.read_bytes(),np.float32).copy()
        raw[21]=np.nan if bad=='nan' else -1; path.write_bytes(raw.tobytes())
    try:
        with pytest.raises((ValueError,FileNotFoundError)):
            module.prepare(volume,tmp_path/'run',config,geometry)
        assert not (tmp_path/'run/LASAGNA.json').exists()
    finally: volume.close()
