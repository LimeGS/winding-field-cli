import importlib.util
import io
import json
import sqlite3

import numpy as np
import pytest

from test_splines import prepared,upstream_curves


def adapter():
    assert importlib.util.find_spec('winding_field.afv'), 'external AFV importer missing'
    from winding_field import afv
    return afv


def fixture(root,points=None,*,gap=None,embedded=False,shape=(8,8,24),input_shape=None):
    maps,meta=prepared(root,shape=shape,gap=gap,chunk=8); maps.close()
    path=root/'external.afv'
    if points is None: points=np.array([[1.,2.,3.],[5.,2.,3.]])
    upstream_curves(path,input_shape or shape,[(1,points)])
    geometry=dict(source=meta['source']['source'],origin_xyz_um=meta['origin_xyz'],spacing_xyz_um=meta['spacing_xyz'])
    descriptor=root/'geometry.json'; descriptor.write_text(json.dumps(geometry))
    if embedded:
        frame=dict(source=meta['source'],coordinate_order='XYZ',units='selected-source voxels',level=0,
                   origin_xyz_um=meta['origin_xyz'],spacing_xyz_um=meta['spacing_xyz'])
        with sqlite3.connect(path) as db:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('frame',json.dumps(frame)))
    return meta,dict(fibres=str(path),fibres_geometry=None if embedded else str(descriptor),reserve_bytes=0)


def curves(root):
    with sqlite3.connect(root/'fibres/fibres.sqlite') as db:
        return [(fid,family,np.load(io.BytesIO(blob),allow_pickle=False)) for fid,family,blob in
                db.execute('SELECT source_id,family,points FROM curves ORDER BY id')]


@pytest.mark.parametrize('embedded',[True,False])
def test_import_preserves_physical_frame(tmp_path,embedded):
    afv=adapter(); meta,config=fixture(tmp_path,embedded=embedded)
    with sqlite3.connect(config['fibres']) as db:
        db.execute("UPDATE fibers SET family='unknown'")
    result=afv.import_fibres(tmp_path,config)
    assert result['status']=='FIBRES_READY'
    assert result['method']=='external-afv' and result['input_count']==1
    imported=curves(tmp_path)
    assert imported[0][:2]==(1,'unknown')
    np.testing.assert_allclose(imported[0][2][0],[109.,218.,327.])
    np.testing.assert_allclose(imported[-1][2][-1],[145.,218.,327.])


@pytest.mark.parametrize('mutation',[
    'missing_geometry','source_mismatch','conflict','incomplete','nonfinite','discontinuous','wal',
])
def test_invalid_afv_input(tmp_path,mutation):
    afv=adapter(); meta,config=fixture(tmp_path,embedded=mutation=='conflict')
    if mutation=='missing_geometry': config['fibres_geometry']=None
    if mutation=='source_mismatch':
        path=tmp_path/'geometry.json'; value=json.loads(path.read_bytes()); value['source']='other-ct'
        path.write_text(json.dumps(value))
    if mutation=='conflict':
        path=tmp_path/'geometry.json'; value=json.loads(path.read_bytes()); value['origin_xyz_um'][0]+=100
        path.write_text(json.dumps(value)); config['fibres_geometry']=str(path)
    if mutation=='wal': (tmp_path/'external.afv-wal').write_bytes(b'uncheckpointed')
    if mutation in ('incomplete','nonfinite','discontinuous'):
        with sqlite3.connect(config['fibres']) as db:
            if mutation=='incomplete': db.execute("UPDATE metadata SET value='false' WHERE key='complete'")
            if mutation=='nonfinite': db.execute('UPDATE blocks SET points=?',(np.full((2,3),np.nan,dtype='<f8').tobytes(),))
            if mutation=='discontinuous': db.execute('UPDATE blocks SET first_segment=1')
    with pytest.raises(ValueError,match='AFV|geometry|source|frame|block|finite|journal'):
        afv.import_fibres(tmp_path,config)
    assert not (tmp_path/'FIBRES.json').exists()


def test_sparse_crossing_edge_is_clipped_and_split_at_support_gap(tmp_path):
    afv=adapter()
    fixture_data=np.array([[0.,2.,3.],[50.,2.,3.]])
    meta,config=fixture(tmp_path,fixture_data,gap=12,input_shape=(8,8,64))
    meta['source'].update(shape=[8,8,64],origin_xyz=[10.,200.,300.])
    (tmp_path/'PREPARED.json').write_text(json.dumps(meta))
    descriptor=json.loads((tmp_path/'geometry.json').read_bytes()); descriptor['origin_xyz_um'][0]=10.
    (tmp_path/'geometry.json').write_text(json.dumps(descriptor))
    result=afv.import_fibres(tmp_path,config)
    imported=curves(tmp_path)
    assert result['count']==len(imported)==2
    np.testing.assert_allclose(imported[0][2][0],[100.,218.,327.])
    np.testing.assert_allclose(imported[0][2][-1],[199.,218.,327.])
    np.testing.assert_allclose(imported[1][2][0],[217.,218.,327.])
    np.testing.assert_allclose(imported[1][2][-1],[307.,218.,327.])


def test_import_resume_preserves_inventory(tmp_path):
    afv=adapter()
    points=np.column_stack([np.arange(400)/20+1,np.full(400,2.),np.full(400,3.)])
    _,config=fixture(tmp_path,points)
    def stop():
        path=tmp_path/'fibres/fibres.sqlite'
        if not path.exists(): return False
        with sqlite3.connect(path) as db:
            return db.execute('SELECT count(*) FROM done').fetchone()[0]>0
    assert afv.import_fibres(tmp_path,config,stop)['status']=='PAUSED'
    first=curves(tmp_path)
    assert first
    ready=afv.import_fibres(tmp_path,config)
    assert ready['status']=='FIBRES_READY'
    np.testing.assert_array_equal(curves(tmp_path)[0][2],first[0][2])
    before=(tmp_path/'fibres/fibres.sqlite').read_bytes()
    assert afv.import_fibres(tmp_path,config)==ready
    assert (tmp_path/'fibres/fibres.sqlite').read_bytes()==before


@pytest.mark.parametrize('mutation',['source','descriptor','payload','ledger'])
def test_import_rejects_changed_inputs(tmp_path,mutation):
    afv=adapter(); _,config=fixture(tmp_path)
    afv.import_fibres(tmp_path,config)
    if mutation=='source':
        with sqlite3.connect(config['fibres']) as db: db.execute("UPDATE fibers SET family='H'")
    elif mutation=='descriptor':
        (tmp_path/'geometry.json').write_text((tmp_path/'geometry.json').read_text()+' ')
    else:
        with sqlite3.connect(tmp_path/'fibres/fibres.sqlite') as db:
            db.execute('DELETE FROM done' if mutation=='ledger' else "UPDATE curves SET points=x'00'")
    with pytest.raises(ValueError,match='changed|corrupt|inventory|identity'):
        afv.validate_retained(tmp_path,config)
    with pytest.raises(ValueError,match='changed|corrupt|inventory|identity'):
        afv.import_fibres(tmp_path,config)


def test_import_reads_cli_selected_level_frame(tmp_path):
    from test_exports import qual_export_fixture
    from winding_field.exports import export_afv
    afv=adapter(); original=qual_export_fixture(tmp_path/'original'); export_afv(original)
    meta=json.loads((original/'PREPARED.json').read_bytes())
    identity=afv.source_identity(original/'fibres/fibres.afv',None,meta['source'])
    assert identity['origin_xyz_um']==[10.,20.,30.]
    assert identity['spacing_xyz_um']==[4.5]*3


def test_embedded_frame_rejects_conflicting_root_geometry(tmp_path):
    afv=adapter(); meta,config=fixture(tmp_path,embedded=True)
    with sqlite3.connect(config['fibres']) as db:
        root=json.loads(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
        root['coordinate_base_shape_zyx']=[8,8,99]
        db.execute("UPDATE metadata SET value=? WHERE key='root'",(json.dumps(root),))
    with pytest.raises(ValueError,match='frame|geometry'):
        afv.source_identity(config['fibres'],None,meta['source'])


def test_input_change_during_import_cannot_publish_receipt(tmp_path):
    afv=adapter(); _,config=fixture(tmp_path)
    def mutate():
        with sqlite3.connect(config['fibres']) as db:
            db.execute("UPDATE metadata SET value='\"mutated\"' WHERE key='uuid'")
        return False
    with pytest.raises(ValueError,match='changed'):
        afv.import_fibres(tmp_path,config,mutate)
    assert not (tmp_path/'FIBRES.json').exists()


def test_paused_import_rejects_unaccounted_curve(tmp_path):
    afv=adapter(); _,config=fixture(tmp_path)
    def stop():
        path=tmp_path/'fibres/fibres.sqlite'
        if not path.exists(): return False
        with sqlite3.connect(path) as db:
            return db.execute('SELECT count(*) FROM done').fetchone()[0]>0
    assert afv.import_fibres(tmp_path,config,stop)['status']=='PAUSED'
    with sqlite3.connect(tmp_path/'fibres/fibres.sqlite') as db: db.execute('DELETE FROM done')
    with pytest.raises(ValueError,match='inventory'):
        afv.import_fibres(tmp_path,config)
