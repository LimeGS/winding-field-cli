import copy
import numpy as np
import pytest


def test_frame_transform_and_nonunit_scale():
    from winding_field.tifxyz_contract import validate_request,load_settings,to_output_xyz
    frame=dict(id='target',origin_xyz_um=[86.4,172.8,259.2],voxel_um=8.64,shape_zyx=[100]*3,source_identity_sha256='a'*64)
    req=validate_request(dict(levels=[-4.],pitch_um=17.28,output_frame=frame))
    settings=load_settings(req,dict(shape_zyx=[10]*3,spacing_xyz=[2,3,4],voxel_um=8.64),frame)
    np.testing.assert_allclose(to_output_xyz(np.array([[112.32,250.56,423.36]]),settings.output_frame),[[3,9,19]])
    assert settings.scale==[.5,.5]
    assert settings.tolerance_um==34.56
    req['levels'].append(2)
    assert settings.levels==(-4.,)


@pytest.mark.parametrize('change',[{'pitch_um':0},{'pitch_um':True},{'memory_gib':float('inf')},
    {'levels':[float('nan')]},{'levels':[True]},{'levels':[1,1]},{'levels':[]},{'max_pixels':1e100},
    {'max_pixels':True},{'roi':{'lo_zyx':[0,0,False],'hi_zyx':[3,3,3]}},{'surprise':1}])
def test_invalid_request_rejected_before_io(change):
    from winding_field.tifxyz_contract import validate_request
    with pytest.raises(ValueError):validate_request(dict(levels=[0],**change) if 'levels' not in change else change)


def test_json_duplicates_and_nonfinite_rejected():
    from winding_field.tifxyz_contract import parse_json
    for text in ('{"levels":[1],"levels":[2]}','{"pitch_um":NaN}'):
        with pytest.raises(ValueError):parse_json(text)


def test_outside_volume_and_unknown_frame_rejected():
    from winding_field.tifxyz_contract import to_output_xyz,validate_frame
    frame=dict(id='target',origin_xyz_um=[0]*3,voxel_um=1.,shape_zyx=[3]*3,source_identity_sha256='a'*64)
    with pytest.raises(ValueError):to_output_xyz(np.array([[-1,0,0]]),frame)
    bad=copy.deepcopy(frame);bad['voxel_um']=[1,2,1]
    with pytest.raises(ValueError):validate_frame(bad)
