import importlib.util
import json
import sys

import numpy as np
import pytest


def bridge():
    assert importlib.util.find_spec('winding_field.lasagna'), 'official Lasagna bridge missing'
    from winding_field import lasagna
    return lasagna


def test_pinned_checkpoint_rejects_corruption_and_missing_offline_weights(tmp_path):
    from winding_field import models
    path=tmp_path/'lasagna.pt'
    with pytest.raises(FileNotFoundError,match='lasagna'):
        models.checkpoint(path,offline=True,prediction='lasagna')
    path.write_bytes(b'not the public checkpoint')
    with pytest.raises(ValueError,match='SHA-256'):
        models.checkpoint(path,offline=True,prediction='lasagna')


def test_missing_runtime_preflight(tmp_path):
    lasagna=bridge()
    with pytest.raises((ValueError,FileNotFoundError),match='Lasagna|lasagna'):
        lasagna.runtime_identity(tmp_path/'absent/bin/python')
    assert not (tmp_path/'absent').exists()


def test_runtime_rejects_cli_python_version():
    lasagna=bridge()
    with pytest.raises(ValueError,match='3.14'):
        lasagna.runtime_identity(sys.executable)


@pytest.mark.parametrize('spacing,factor,expected',[
    ([10.,10.,10.],1.,.01),
    ([5.,10.,20.],1.,.02),
    ([10.,10.,10.],.5,.005),
])
def test_density_uses_input_pitch_once(spacing,factor,expected):
    lasagna=bridge()
    shape=(2,3,4)
    result=lasagna.decode_observations(np.full(shape,100,np.uint8),
        np.full(shape,255,np.uint8),np.full(shape,128,np.uint8),
        np.full(shape,255,np.uint8),input_spacing_xyz=spacing,
        encode_scale=1000.,grad_mag_factor=factor,validity=np.ones(shape,bool))
    assert result['support'].all()
    np.testing.assert_allclose(result['density'],expected,rtol=1e-6)
    np.testing.assert_allclose(result['normals'],np.broadcast_to([1.,0.,0.],shape+(3,)))
    np.testing.assert_array_equal(result['surface_phase'],np.ones(shape))


def test_decode_keeps_invalid_and_zero_density_unsupported():
    lasagna=bridge()
    shape=(2,2,2); valid=np.ones(shape,bool); valid[0,0,0]=False
    magnitude=np.full(shape,100,np.uint8); magnitude[1,1,1]=0
    result=lasagna.decode_observations(magnitude,np.full(shape,128,np.uint8),
        np.full(shape,128,np.uint8),np.full(shape,128,np.uint8),
        input_spacing_xyz=[10.]*3,encode_scale=1000.,grad_mag_factor=1.,validity=valid)
    assert result['support'].sum()==6
    assert not result['density'][~result['support']].any()
    assert not result['normals'][~result['support']].any()
    assert not result['surface_phase'][~result['support']].any()


def test_decode_rejects_nonfinite_products():
    lasagna=bridge()
    shape=(2,2,2); magnitude=np.ones(shape); magnitude[0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        lasagna.decode_observations(magnitude,np.full(shape,128),np.full(shape,128),
            np.full(shape,128),input_spacing_xyz=[10.]*3,encode_scale=1000.,
            grad_mag_factor=1.,validity=np.ones(shape,bool))


@pytest.mark.parametrize('settings',[
    {'input_spacing_xyz':[0,10,10]},
    {'encode_scale':0},
    {'grad_mag_factor':-1},
])
def test_decode_rejects_invalid_physical_scaling(settings):
    lasagna=bridge(); shape=(2,2,2)
    options=dict(input_spacing_xyz=[10.]*3,encode_scale=1000.,grad_mag_factor=1.)
    options.update(settings)
    with pytest.raises(ValueError,match='positive|spacing|scale'):
        lasagna.decode_observations(np.ones(shape),np.full(shape,128),
            np.full(shape,128),np.full(shape,128),validity=np.ones(shape,bool),**options)


def prepared_source(tmp_path,monkeypatch):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.util import digest
    lasagna=bridge()
    data=np.full((64,64,64),33,np.uint8); data[32,:,32]=0
    path=write_ct(tmp_path/'ct',data,spacing=(5.,10.,20.),origin=(100.,200.,300.))
    volume=Volume(path,cache=tmp_path/'cache')
    geometry=dict(shape_zyx=[8,8,8],spacing_xyz=[10.,20.,40.],origin_xyz=[180.,360.,620.],
                  frame=digest(volume.identity),roi_lo_zyx=[16,16,16],roi_hi_zyx=[32,32,32],stride=2)
    monkeypatch.setattr(lasagna,'runtime_identity',lambda python=None:dict(executable=sys.executable,revision='fixture'))
    monkeypatch.setattr(lasagna,'checkpoint',lambda *a,**kw:tmp_path/'model.pt',raising=False)
    monkeypatch.setattr(lasagna,'run_child',fake_official,raising=False)
    config=dict(stride=2,invalid_value=0,reserve_bytes=0,threads=1)
    return volume,geometry,config


def fake_official(command,log,stop=None,reserve_bytes=0,threads=4,**kwargs):
    """Replace only the model subprocess; retain staging, metadata and decoding."""
    from pathlib import Path
    from ct_fixture import write_ct
    from winding_field.util import atomic_json
    input_path=Path(command[command.index('--input')+1])
    destination=Path(command[command.index('--output')+1]); destination.parent.mkdir(exist_ok=True)
    shape=json.loads((input_path/'.zarray').read_bytes())['shape']
    groups={}
    for name,value in [('grad_mag',100),('nx',255),('ny',128),('cos',255)]:
        folder=destination.parent/f'surface_{name}.ome.zarr'
        if not folder.exists(): write_ct(folder,np.full(shape,value,np.uint8),chunk=32)
        groups[name]=dict(zarr=folder.name+'/0',scaledown=0,channels=[name])
    atomic_json(destination,dict(version=2,source_to_base=1.,grad_mag_encode_scale=1000.,
        grad_mag_factor=1.,groups=groups,base_shape_zyx=shape,provenance='inference.json'))
    return True


def test_prepare_samples_calibrated_crop_and_preserves_invalid_gap(tmp_path,monkeypatch):
    lasagna=bridge(); volume,geometry,config=prepared_source(tmp_path,monkeypatch)
    monkeypatch.setattr(lasagna,'run_child',fake_official)
    try:
        result=lasagna.prepare(volume,tmp_path/'run',config,geometry)
        assert result['status']=='LASAGNA_READY'
        data=lasagna.read_observations(tmp_path/'run',geometry,[0,0,0],[8,8,8])
        assert data['support'].all()
        np.testing.assert_allclose(data['density'],.01,rtol=1e-6)
        np.testing.assert_array_equal(data['surface_phase'],1.)
        lasagna.validate_retained(tmp_path/'run',config)
        # A separate ROI owns the acquisition gap, which predictions must not fill.
        geometry.update(origin_xyz=[260.,520.,940.],roi_lo_zyx=[32,32,32],roi_hi_zyx=[48,48,48])
        lasagna.prepare(volume,tmp_path/'gap',config,geometry)
        gap=lasagna.read_observations(tmp_path/'gap',geometry,[0,0,0],[8,8,8])
        assert not gap['support'][0,:,0].any()
        assert gap['support'][1:,:,1:].all()
    finally: volume.close()


@pytest.mark.parametrize('product',['input','output','input_metadata'])
def test_prepare_rejects_changed_retained_chunks(tmp_path,monkeypatch,product):
    lasagna=bridge(); volume,geometry,config=prepared_source(tmp_path,monkeypatch)
    monkeypatch.setattr(lasagna,'run_child',fake_official)
    run=tmp_path/'run'
    try:
        lasagna.prepare(volume,run,config,geometry)
        path=(run/'lasagna-source/input.zarr/0/0.0.0' if product=='input' else
              run/'lasagna-source/output/surface_grad_mag.ome.zarr/0/0.0.0')
        if product=='input_metadata': path=run/'lasagna-source/input.zarr/0/.zarray'
        path.write_bytes(b'changed')
        with pytest.raises(ValueError,match='changed|corrupt'):
            lasagna.validate_retained(run,config)
        with pytest.raises(ValueError,match='changed|corrupt'):
            lasagna.prepare(volume,run,config,geometry)
    finally: volume.close()


def test_prepare_stop_resume_staging_and_upstream(tmp_path,monkeypatch):
    lasagna=bridge(); volume,geometry,config=prepared_source(tmp_path,monkeypatch)
    run=tmp_path/'run'
    def stop_staging():
        return (run/'lasagna-source/input.zarr/0/0.0.0').exists()
    def pause_child(*args,**kwargs):
        fake_official(*args,**kwargs)
        return False
    try:
        monkeypatch.setattr(lasagna,'run_child',fake_official)
        assert lasagna.prepare(volume,run,config,geometry,stop_staging)['status']=='PAUSED'
        path=run/'lasagna-source/input.zarr/0/0.0.0'; before=path.stat().st_mtime_ns
        monkeypatch.setattr(lasagna,'run_child',pause_child)
        assert lasagna.prepare(volume,run,config,geometry)['status']=='PAUSED'
        monkeypatch.setattr(lasagna,'run_child',fake_official)
        assert lasagna.prepare(volume,run,config,geometry)['status']=='LASAGNA_READY'
        assert path.stat().st_mtime_ns==before
        receipt=(run/'LASAGNA.json').read_bytes()
        assert lasagna.prepare(volume,run,config,geometry)['status']=='LASAGNA_READY'
        assert (run/'LASAGNA.json').read_bytes()==receipt
    finally: volume.close()


def test_cpu_worker_preserves_float32_inference():
    import torch
    from winding_field.lasagna_worker import run_tile_inference
    torch.manual_seed(1)
    model=torch.nn.Conv3d(1,8,3,padding=1).eval()
    data=torch.randn(1,1,4,4,4)
    with torch.inference_mode():
        expected=model(data)
        actual=run_tile_inference(None,model,data,device=torch.device('cpu'))
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_lasagna_child_error_names_the_failed_stage(tmp_path):
    from winding_field.splines import run_child
    with pytest.raises(RuntimeError,match='Lasagna.*failed'):
        run_child([sys.executable,'-c','raise SystemExit(2)'],tmp_path/'child.log',label='Lasagna')
