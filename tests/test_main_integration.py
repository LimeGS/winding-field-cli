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
        assert products['prepare']==512*48+lasagna+result['estimated_lasagna_output_bytes']
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
    product=tmp_path/'run/lasagna-source/output/surface_grad_mag.ome.zarr/2/0.0.0'
    product.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed|identity'):
        run(state['config'],resume=True,until='prepare')


@pytest.mark.parametrize('branch', ['native', 'atlas', 'original', 'tifxyz'])
def test_import_plan_accounts_for_retained_tiles_in_every_output_route(tmp_path, capsys, branch):
    from test_lasagna_input import source_case
    volume,_,config=source_case(tmp_path,shape=(8,8,8),whole=True,spacing=(9.,)*3)
    volume.close()
    args=['plan','--source',config['source'],'--cache',config['cache'],
          '--output',str(tmp_path/'run'),'--lasagna-input',config['lasagna_input'],
          '--lasagna-geometry',config['lasagna_geometry'],'--reserve-gib','0']
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
    assert result['surface_method']=='external-lasagna' and result['prediction']=='qual-v2'
    assert result['estimated_lasagna_import_bytes']==11264  # 512 samples, 21 surface bytes + uint8 CT.
    assert result['estimated_lasagna_staging_bytes']==result['estimated_lasagna_workspace_bytes']==0
    assert result['estimated_spline_dense_bytes']==5248
    assert result['estimated_products_bytes']>=71808
    if branch=='tifxyz':
        products=result['estimated_products_by_stage']
        assert products['prepare']==35840
        assert products['fibres']==15488
        assert sum(products.values())==result['estimated_products_bytes']
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('command', ['plan', 'run'])
def test_calibrated_import_rejects_bounded_preparation_before_source_io(tmp_path, capsys, command):
    assert main([command,'--source',str(tmp_path/'missing-ct'),'--output',str(tmp_path/'run'),
                 '--lasagna-input',str(tmp_path/'missing.lasagna.json'),'--bounded-preparation',
                 '--lasagna-geometry',str(tmp_path/'geometry.json'),
                 '--field-spacing-um','17','--until','field'])==1
    error=capsys.readouterr().err
    assert '--lasagna-geometry' in error and 'bounded' in error
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('branch', ['native', 'original', 'tifxyz'])
def test_imported_surfaces_reach_new_output_routes_and_bind_resume(tmp_path, monkeypatch, branch):
    from test_lasagna_input import source_case,forbid_models
    from test_splines import upstream_curves
    from winding_field.artifacts import Tiles
    from winding_field.workflow import run
    from winding_field.tifxyz_contract import validate_request
    from winding_field.util import digest,file_sha
    volume,_,config=source_case(tmp_path); volume.close()
    fibres=tmp_path/'fibres.afv'
    upstream_curves(fibres,(16,16,40),[(1,np.array([[4.,2.,6.],[4.,8.,6.]]))])
    root=tmp_path/'run'
    config.update(output=str(root),prediction='qual-v2',fibres=str(fibres),
                  fibres_geometry=config['lasagna_geometry'],roi_lo=[2]*3,roi_hi=[10]*3,stride=2,
                  chunk=4,field_chunk=4,sweeps=1,maxiter=200,threads=1,
                  surface_output='native' if branch=='original' else branch)
    if branch=='original':
        config.update(flattening_method='original',original_request=dict(
            output_frame=dict(id='scan',origin_xyz_um=[0,0,0],voxel_um=9.)))
    elif branch=='tifxyz':
        request=validate_request(dict(levels=[1.],reserve_disk_gib=0,max_pixels=4096))
        config.update(tifxyz_request=request,tifxyz_request_sha256=digest(request))
    forbid_models(monkeypatch)
    state=run(config,until='field')
    assert set(state['stages'])=={'prepare','fibres','field'}
    assert json.loads((root/'FIELD.json').read_bytes())['fibre_constraint_rows']>0
    maps=Tiles(root/'maps')
    try:
        values=maps.load((0,0,0))
        assert values['support'].any()
        np.testing.assert_allclose(values['density'][values['support']],.05,rtol=1e-6)
    finally: maps.close()
    before=file_sha(root/'field/CHECKPOINT.json')
    run(state['config'],resume=True,until='field')
    assert file_sha(root/'field/CHECKPOINT.json')==before
    (tmp_path/'predictions/nx/0/0.0.0').write_bytes(bytes([128])*4**3)
    with pytest.raises(ValueError,match='changed'):
        run(state['config'],resume=True,until='field')
    assert file_sha(root/'field/CHECKPOINT.json')==before
