import json
import sqlite3
import sys

import numpy as np
import pytest

from ct_fixture import write_ct
from test_lasagna import fake_official
from test_splines import upstream_curves


def source_config(root,monkeypatch,*,external=False):
    from winding_field import lasagna,predict
    ct=write_ct(root/'ct',np.full((16,16,16),100,np.uint8),spacing=(9.,)*3)
    config=dict(source=str(ct),output=str(root/'run'),prediction='qual-v2',cache=str(root/'cache'),
        reserve_bytes=0,chunk=8,field_chunk=8,sweeps=1,maxiter=100,threads=1,previews=False)
    monkeypatch.setattr(lasagna,'runtime_identity',lambda python=None:dict(executable=sys.executable,revision='fixture'))
    monkeypatch.setattr(lasagna,'checkpoint',lambda *a,**kw:root/'model.pt')
    monkeypatch.setattr(lasagna,'run_child',fake_official)
    class EmptyQual:
        def __init__(self,*args): pass
        def predict(self,ct):
            return np.zeros(ct.shape,np.float32),np.zeros(ct.shape+(3,),np.float32),np.zeros(ct.shape+(2,),np.float32)
    monkeypatch.setattr(predict,'QualV2',EmptyQual)
    if external:
        path=root/'input.afv'; upstream_curves(path,(16,)*3,[(1,np.array([[5.,1.,6.],[5.,14.,6.]]))])
        descriptor=root/'geometry.json'
        descriptor.write_text(json.dumps(dict(source=str(ct),origin_xyz_um=[0.]*3,spacing_xyz_um=[9.]*3)))
        config.update(fibres=str(path),fibres_geometry=str(descriptor))
    return config


def test_surface_support_independent_of_fibres(tmp_path,monkeypatch):
    from winding_field import predict
    from winding_field.artifacts import Tiles
    from winding_field.workflow import open_volume
    config=source_config(tmp_path,monkeypatch)
    class EmptyQual:
        def __init__(self,*args): pass
        def predict(self,ct):
            return np.zeros(ct.shape,np.float32),np.zeros(ct.shape+(3,),np.float32),np.zeros(ct.shape+(2,),np.float32)
    monkeypatch.setattr(predict,'QualV2',EmptyQual)
    volume=open_volume(config)
    try: meta=predict.prepare(volume,config['output'],config)
    finally: volume.close()
    maps=Tiles(tmp_path/'run/maps')
    try:
        values=maps.load((0,0,0))
        assert values['support'].all() and not values['presence'].any()
        np.testing.assert_allclose(values['density'],1/90,rtol=1e-6)
        assert meta['surface_method']=='lasagna' and meta['fibre_method']=='qual-v2'
    finally: maps.close()


def test_external_afv_bypasses_qual_and_feeds_field_mesh_exports(tmp_path,monkeypatch):
    from winding_field import predict,splines,models
    from winding_field.workflow import run
    from winding_field.artifacts import Tiles
    config=source_config(tmp_path,monkeypatch,external=True)
    def forbidden(*args,**kwargs): raise AssertionError('Qual must be omitted with external AFV')
    monkeypatch.setattr(predict,'QualV2',forbidden)
    monkeypatch.setattr(splines,'trace_qual',forbidden)
    monkeypatch.setattr(models,'checkpoint',forbidden)
    state=run(config,until='mesh')
    assert set(state['stages'])=={'prepare','fibres','field','mesh'}
    root=tmp_path/'run'; maps=Tiles(root/'maps')
    try:
        values=maps.load((0,0,0))
        assert 'surface_phase' in values and 'probabilities' not in values and 'presence' not in values
    finally: maps.close()
    field=json.loads((root/'FIELD.json').read_bytes())
    assert field['fibre_constraint_rows']>0
    assert 'Lasagna' in json.loads((root/'MESHES.json').read_bytes())['level_selection']
    with sqlite3.connect(root/'fibres/fibres.afv') as db:
        note=json.loads(db.execute('SELECT annotation FROM fibers LIMIT 1').fetchone()[0])
        assert note['automatically_traced'] is False and note['source_id']==1
    assert (root/'lasagna/surface_grad_mag.ome.zarr/0/.zarray').exists()
    assert not (root/'lasagna/fibres.lasagna.json').exists()
    saved=(root/'FIBRES.json').read_bytes()
    run(state['config'],resume=True,until='mesh')
    assert (root/'FIBRES.json').read_bytes()==saved


def test_external_plan_storage(tmp_path,monkeypatch):
    from winding_field.workflow import plan
    config=source_config(tmp_path,monkeypatch,external=True)
    result=plan(config)
    assert result['estimated_spline_dense_bytes']==0
    assert result['estimated_lasagna_staging_bytes']>0 and result['estimated_lasagna_workspace_bytes']>0


@pytest.mark.parametrize('mutation',['source','lasagna'])
def test_completed_prepare_rejects_changed_retained_inputs(tmp_path,monkeypatch,mutation):
    from winding_field.workflow import run
    config=source_config(tmp_path,monkeypatch,external=True)
    state=run(config,until='prepare')
    if mutation=='source':
        with sqlite3.connect(config['fibres']) as db: db.execute("UPDATE fibers SET family='H'")
    else:
        (tmp_path/'run/lasagna-source/output/surface_grad_mag.ome.zarr/0/0.0.0').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed|identity'):
        run(state['config'],resume=True,until='prepare')


def test_missing_lasagna_runtime_does_not_create_run(tmp_path):
    from winding_field.workflow import run
    ct=write_ct(tmp_path/'ct',np.ones((8,)*3,np.uint8),spacing=(9.,)*3)
    config=dict(source=str(ct),output=str(tmp_path/'run'),prediction='qual-v2',cache=str(tmp_path/'cache'),
                lasagna_python=str(tmp_path/'missing-python'),reserve_bytes=0)
    with pytest.raises(FileNotFoundError,match='Lasagna'):
        run(config,until='prepare')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('bad',[dict(fibres_geometry='geometry.json'),dict(fibres='input.afv',model_path='qual.pt')])
def test_conflicting_fibre_options_are_rejected(bad):
    from winding_field.config import validate_config
    with pytest.raises(ValueError,match='fibres|AFV'):
        validate_config(bad)


def test_public_cli_accepts_external_fibres_and_skips_qual_sampling_contract(tmp_path,monkeypatch,capsys):
    from winding_field.cli import main
    config=source_config(tmp_path,monkeypatch,external=True)
    assert main(['plan','--source',config['source'],'--output',config['output'],
                 '--cache',config['cache'],'--fibres',config['fibres'],'--fibres-geometry',config['fibres_geometry'],
                 '--lasagna-python',sys.executable,'--lasagna-model-path',str(tmp_path/'model.pt'),
                 '--stride','2','--field-spacing-um','17'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['prediction']=='external-afv' and result['field']['spacing_xyz']==[17.]*3


def test_direct_external_preparation_rechecks_staged_source_bytes(tmp_path,monkeypatch):
    from winding_field.workflow import open_volume
    from winding_field.predict import prepare
    config=source_config(tmp_path,monkeypatch,external=True)
    volume=open_volume(config)
    try: prepare(volume,config['output'],config)
    finally: volume.close()
    (tmp_path/'ct/0/0.0.0').write_bytes(bytes([42])*32**3)
    volume=open_volume(config)
    try:
        with pytest.raises(ValueError,match='source CT changed'):
            prepare(volume,config['output'],config)
    finally: volume.close()
