"""The new output branches retain main's observation and dependency contracts."""
import json

import numpy as np
import pytest

from ct_fixture import write_ct
from winding_field.cli import main


@pytest.mark.parametrize('branch', ['native', 'atlas', 'original', 'tifxyz'])
def test_qual_lasagna_plan_preserves_observation_costs_for_each_output(tmp_path, capsys, branch):
    source=write_ct(tmp_path/'ct',np.ones((8,8,8),np.uint8),spacing=(9.,9.,9.))
    args=['plan','--source',str(source),'--output',str(tmp_path/'run'),'--reserve-gib','0']
    if branch=='original':
        request=tmp_path/'flattening.json'
        request.write_text(json.dumps(dict(output_frame=dict(id='scan',origin_xyz_um=[0,0,0],voxel_um=9.))))
        args+=['--flattening-method','original','--flattening-config',str(request)]
    else:
        args+=['--surface-output',branch]
    if branch=='tifxyz':
        request=tmp_path/'direct.json'
        request.write_text(json.dumps(dict(levels=[1.],reserve_disk_gib=0,max_pixels=4096)))
        args+=['--tifxyz-config',str(request)]
    assert main(args)==0
    result=json.loads(capsys.readouterr().out)
    assert result['prediction']=='qual-v2'
    lasagna=result['estimated_lasagna_staging_bytes']+result['estimated_lasagna_workspace_bytes']
    splines=result['estimated_spline_dense_bytes']
    assert lasagna>0 and splines>0
    assert result['estimated_products_bytes']>=lasagna+splines+512*(60+40+8)
    if branch=='tifxyz':
        products=result['estimated_products_by_stage']
        assert products['prepare']==512*48+lasagna
        assert products['fibres']==512*(12+8)+splines
        assert sum(products.values())==result['estimated_products_bytes']
    assert not (tmp_path/'run').exists()


def test_direct_export_identity_binds_external_core(monkeypatch):
    from winding_field import tifxyz
    import field_store
    from pathlib import Path
    from winding_field.util import file_sha
    original=tifxyz._code_identity()
    assert original['files']['field_store.py']==file_sha(Path(field_store.__file__))
    assert 'surface_sources.py' in original['files']
    real=tifxyz.file_sha
    monkeypatch.setattr(tifxyz,'file_sha',lambda path: 'changed' if Path(path)==Path(field_store.__file__) else real(path))
    assert tifxyz._code_identity()!=original


@pytest.mark.parametrize('branch', ['native', 'original', 'tifxyz'])
def test_new_branches_reject_changed_retained_lasagna_before_resume(tmp_path, monkeypatch, branch):
    from test_ct_sources import source_config
    from winding_field.workflow import run
    from winding_field.tifxyz_contract import validate_request
    from winding_field.util import digest
    config=source_config(tmp_path,monkeypatch,external=True)
    config['surface_output']='native' if branch=='original' else branch
    if branch=='original':
        config.update(flattening_method='original',original_request=dict(
            output_frame=dict(id='scan',origin_xyz_um=[0,0,0],voxel_um=9.)))
    elif branch=='tifxyz':
        request=validate_request(dict(levels=[1.],reserve_disk_gib=0,max_pixels=4096))
        config.update(tifxyz_request=request,tifxyz_request_sha256=digest(request))
    state=run(config,until='prepare')
    assert state['status']=='PAUSED'
    product=tmp_path/'run/lasagna-source/output/surface_grad_mag.ome.zarr/0/0.0.0'
    product.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed|identity'):
        run(state['config'],resume=True,until='prepare')
