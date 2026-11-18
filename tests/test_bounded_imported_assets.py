import json
from pathlib import Path

import numpy as np
import sqlite3
import pytest

from test_lasagna_input import fixture
from test_splines import upstream_curves
from winding_field import workflow
from winding_field.util import atomic_json,file_sha


def test_chunk_ledger_quota_rolls_back_and_preserves_committed_chunks(tmp_path):
    from winding_field.io import Volume
    ct,_,_=fixture(tmp_path)
    volume=Volume(str(ct))
    ledger=tmp_path/'ledger.sqlite'
    try:
        volume.bind_ledger(ledger,quota_bytes=100352)
        volume._record('first',b'original')
        with pytest.raises(sqlite3.DatabaseError,match='full'):
            for i in range(1000):volume._record(str(i)*100,b'value')
        assert ledger.stat().st_size<=16384
        with sqlite3.connect(ledger) as db:
            assert db.execute("SELECT count(*) FROM chunks WHERE key='first'").fetchone()==(1,)
            assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
    finally:volume.close()


def test_bounded_metadata_rejected_before_retention_exceeds_allowance(tmp_path):
    from winding_field.io import Volume
    from winding_field.lasagna_input import LasagnaInput
    ct,_,config=fixture(tmp_path)
    manifest=Path(config['lasagna_input'])
    value=json.loads(manifest.read_bytes())
    value['extra']='x'*(3<<20)
    atomic_json(manifest,value)
    config.update(bounded_preparation=True,bounded_bytes=16<<20)
    root=tmp_path/'bounded'
    volume=Volume(str(ct))
    try:
        with pytest.raises(OSError,match='metadata quota'):
            with LasagnaInput(volume,root,config):pass
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=config['bounded_bytes']//8
    finally:volume.close()


@pytest.mark.parametrize('odd_roi', [False, True])
def test_imported_assets_prepare_without_retaining_native_maps(tmp_path,monkeypatch,odd_roi):
    ct,external,config=fixture(tmp_path)
    afv=tmp_path/'external.afv'
    upstream_curves(afv,(9,10,11),[(1,np.array([[1.,2.,3.],[5.,2.,3.]]))])
    geometry=tmp_path/'geometry.json'
    atomic_json(geometry,dict(source=str(ct),origin_xyz_um=[100.,200.,300.],spacing_xyz_um=[5.,10.,20.]))
    root=tmp_path/'prepared'
    config.update(source=str(ct),output=str(root),fibres=str(afv),fibres_geometry=str(geometry),
                  bounded_preparation=True,bounded_bytes=64<<20,field_spacing_um=40.,field_chunk=4)
    if odd_roi:config.update(roi_lo=[1,1,1],roi_hi=[8,9,10])
    result=workflow.run(config,until='fibres')
    assert set(result['stages'])=={'prepare','fibres'}
    assert not (root/'maps').exists()
    assert not (root/'lasagna').exists()
    meta=json.loads((root/'PREPARED.json').read_text())
    assert meta['fibre_method']=='external-afv'
    assert (root/'coarse/COARSE.json').is_file()
    assert json.loads((root/'FIBRES.json').read_text())['count']>0
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<config['bounded_bytes']
    assert Path(external).exists()
    legacy=tmp_path/'legacy'
    workflow.run(dict(config,output=str(legacy),bounded_preparation=False),until='fibres')
    def curves(path):
        with sqlite3.connect(path/'fibres/fibres.sqlite') as db:
            return db.execute('SELECT points,family,source_id,source_block,annotation FROM curves ORDER BY id').fetchall()
    assert curves(root)==curves(legacy)
    from winding_field.exports import export_afv
    # An export must fail before publication if its database cannot fit.
    exported=tmp_path/'export-quota'
    import shutil
    shutil.copytree(root,exported)
    (exported/'fibres/fibres.afv').unlink()
    (exported/'exports.sqlite').unlink()
    with pytest.raises((OSError,sqlite3.DatabaseError),match='quota|full'):
        export_afv(exported,quota_bytes=81920)
    assert not (exported/'fibres/fibres.afv').exists()
    assert workflow.run(config,resume=True,until='fibres')['stages']==result['stages']
    from winding_field.cli import main
    assert main(['resume',str(root)])==0
    assert set(json.loads((root/'RUN.json').read_bytes())['stages'])=={'prepare','fibres'}
    before={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
    from winding_field import bounded_pipeline
    def forbidden(*args,**kwargs):raise AssertionError('run must not prepare imported assets')
    monkeypatch.setattr(bounded_pipeline,'prepare',forbidden)
    monkeypatch.setattr(bounded_pipeline,'fibres',forbidden)
    config.update(output=str(tmp_path/'field'),reuse_from=str(root),reuse_from_sha256=file_sha(root/'RUN.json'),
                  nonconvergence_policy='continue',maxiter=1,sweeps=1)
    retried=workflow.run(config,until='field')
    assert 'field' in retried['stages']
    assert before=={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
