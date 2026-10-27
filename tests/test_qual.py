import json

import numpy as np
import pytest
import torch


def test_qual_decodes_families_and_axis_without_assigning_intersections():
    import winding_field.models as models
    assert hasattr(models,'QualV2'), 'Qual V2 adapter missing'
    class Network(torch.nn.Module):
        def forward(self,x):
            shape=x.shape[2:]
            return dict(seg=torch.log(torch.tensor([1.,2.,3.,4.])).reshape(1,4,1,1,1).expand(1,4,*shape),
                        direction=torch.tensor([3.,4.,0.]).reshape(1,3,1,1,1).expand(1,3,*shape))
    model=models.QualV2.__new__(models.QualV2)
    model.device='cpu'; model.model=Network()
    ct=np.full((128,)*3,100,np.uint8)
    presence,tangent,probability=model.predict(ct)
    assert presence.shape==ct.shape and probability.shape==ct.shape+(2,)
    np.testing.assert_allclose(presence,.9,atol=1e-6)
    np.testing.assert_allclose(probability[...,0],.2,atol=1e-6)
    np.testing.assert_allclose(probability[...,1],.3,atol=1e-6)
    np.testing.assert_allclose(tangent[...,0],.6,atol=1e-6)
    np.testing.assert_allclose(tangent[...,1],.8,atol=1e-6)
    assert not tangent[...,2].any()
    with pytest.raises(ValueError,match='uint8'):
        model.predict(ct.astype(np.float32))


def test_qual_normalization_uses_sample_std_and_floor():
    import winding_field.models as models
    assert hasattr(models,'QualV2'), 'Qual V2 adapter missing'
    from winding_field._vendor.qual_model import normalize_ct
    np.testing.assert_allclose(normalize_ct(torch.tensor([100,102]).reshape(1,1,1,1,2)),
                               np.array([-.1,.1]).reshape(1,1,1,1,2),atol=1e-7)
    np.testing.assert_allclose(normalize_ct(torch.tensor([0,100]).reshape(1,1,1,1,2)),
                               np.array([-2**-.5,2**-.5]).reshape(1,1,1,1,2),atol=1e-7)


def test_qual_checkpoint_must_be_cached_and_verified(tmp_path):
    import winding_field.models as models
    assert hasattr(models,'QualV2'), 'Qual V2 adapter missing'
    path=tmp_path/'qual.pth'
    with pytest.raises(FileNotFoundError,match='cached'):
        models.checkpoint(path,offline=True,prediction='qual-v2')
    path.write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError,match='SHA'):
        models.QualV2(path,offline=True)


@pytest.mark.parametrize('spacing,dtype,stride',[
    ((4.5,)*3,np.uint8,1),((18.,)*3,np.uint8,1),((9.,9.,9.1),np.uint8,1),
    ((9.,)*3,np.uint16,1),((9.,)*3,np.uint8,2)])
def test_qual_rejects_incompatible_observations_before_model_io(tmp_path,spacing,dtype,stride):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.predict import prepare
    from winding_field.workflow import plan
    source=write_ct(tmp_path/'ct.zarr',np.zeros((16,)*3,dtype),spacing=spacing)
    config=dict(prediction='qual-v2',source=str(source),output=str(tmp_path/'run'),
                cache=str(tmp_path/'cache'),stride=stride,offline=True)
    with pytest.raises(ValueError,match='Qual V2'):
        plan(config)
    volume=Volume(source,cache=tmp_path/'cache')
    try:
        with pytest.raises(ValueError,match='Qual V2'):
            prepare(volume,tmp_path/'run',config)
    finally:
        volume.close()
    assert not (tmp_path/'run/maps').exists()


def test_cli_defaults_to_qual_and_keeps_field_pitch_independent(tmp_path,capsys):
    from ct_fixture import write_ct
    from winding_field.cli import main
    source=write_ct(tmp_path/'ct.zarr',np.zeros((16,)*3,np.uint8),spacing=(4.5,)*3)
    assert main(['plan','--source',str(source),'--output',str(tmp_path/'run'),
                 '--cache',str(tmp_path/'cache'),'--offline','--stride','2',
                 '--field-spacing-um','17'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['prediction']=='qual-v2'
    assert result['observations']['spacing_xyz']==[9.]*3
    assert result['field']['spacing_xyz']==[17.]*3


@pytest.mark.parametrize('stride,dilation,padding,groups',[
    ((1,1,1),(1,1,1),(1,1,1),1),((2,2,1),(2,1,1),(2,1,1),2)])
def test_bounded_cpu_convolution_matches_full_volume(monkeypatch,stride,dilation,padding,groups):
    import winding_field.models as models
    assert hasattr(models,'BoundedConv3d'), 'CPU convolution workspace is unbounded'
    torch.manual_seed(18)
    conv=torch.nn.Conv3d(4,8,3,stride=stride,dilation=dilation,padding=padding,groups=groups).eval()
    values=torch.randn(1,4,13,15,17)
    monkeypatch.setattr(models,'CONV_WORKSPACE_BYTES',32<<10)
    with torch.inference_mode():
        expected=conv(values)
        actual=models.BoundedConv3d(conv)(values)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
