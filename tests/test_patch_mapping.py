"""Bounded provenance without Unicode arrays or integer-ID rounding."""
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest


def source():
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE vertices(id INTEGER PRIMARY KEY,level INTEGER,x REAL,y REAL,z REAL,key TEXT); CREATE TABLE faces(id INTEGER PRIMARY KEY,level INTEGER,a INTEGER,b INTEGER,c INTEGER,key TEXT);')
    ids=[2**53+1,2**53+3,2**53+7]
    db.executemany('INSERT INTO vertices VALUES(?,0,?,?,1,?)',[(i,float(k==1),float(k==2),'λ'*4096+str(i)) for k,i in enumerate(ids)])
    db.execute('INSERT INTO faces VALUES(?,0,?,?,?,?)',(2**53+9,*ids,'face-key'))
    return SimpleNamespace(db=db,manifest_sha='mesh-manifest',manifest={'database_sha256':'source-db'}),ids


def test_read_level_keeps_large_keys_in_sqlite_and_exact_int64_ids():
    from winding_field.patch_pipeline import read_level
    reader,ids=source()
    v,f,vi,fi,vk,fk=read_level(reader,0,memory_gib=.1)
    assert not isinstance(vk,np.ndarray), 'source keys must remain in SQLite'
    assert not isinstance(fk,np.ndarray)
    assert vi.tolist()==ids and fi.tolist()==[2**53+9]
    assert f.tolist()==[[0,1,2]] and v.tolist()==[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]
    assert list(vk)==['λ'*4096+str(i) for i in ids]
    reader.db.close()


def test_numeric_reads_use_bounded_fetches_without_key_columns():
    from winding_field.patch_pipeline import read_level
    reader,ids=source()
    class Cursor:
        def __init__(self,cursor):self.cursor=cursor
        def fetchone(self):return self.cursor.fetchone()
        def fetchmany(self,n):
            assert 0<n<=4096
            return self.cursor.fetchmany(n)
        def __iter__(self):raise AssertionError('unbounded row-list read')
    class DB:
        def execute(self,sql,args=()):
            assert 'key' not in sql.lower(), 'numeric read must not load keys'
            return Cursor(reader.db.execute(sql,args))
    result=read_level(SimpleNamespace(db=DB()),0,memory_gib=.1)
    assert result[2].tolist()==ids
    reader.db.close()


def test_mapping_reader_preserves_projection_and_legacy_keys(tmp_path):
    from winding_field.patch_mapping import SourceKeys, write_key_mapping, PatchMapping
    reader,ids=source();root=tmp_path/'v2';root.mkdir()
    np.savez(root/'map.npz',source_vertex_ids=np.array(ids,np.int64),source_face_ids=np.array([2**53+9],np.int64))
    vk=SourceKeys(reader.db,'vertices',np.array(ids,np.int64))
    fk=SourceKeys(reader.db,'faces',np.array([2**53+9],np.int64))
    write_key_mapping(root,reader,vk,fk)
    reader.db.close()  # delivery must be usable without the original path online
    with PatchMapping(root) as mapping:
        assert list(mapping.iter_keys('vertices',batch_size=2))==[(i,'λ'*4096+str(i)) for i in ids]
        assert list(mapping.iter_keys('faces'))==[(2**53+9,'face-key')]
    manifest=json.loads((root/'MAPPING.json').read_text())
    assert manifest['source_database_sha256']=='source-db'
    old=tmp_path/'old';old.mkdir()
    np.savez(old/'map.npz',source_vertex_ids=np.array([9]),source_vertex_keys=np.array(['old-key']),source_face_ids=np.array([7]),source_face_keys=np.array(['old-face']))
    with PatchMapping(old) as mapping:assert list(mapping.iter_keys('vertices'))==[(9,'old-key')]
    with (root/'keys.sqlite').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='hash'):PatchMapping(root)


def test_large_key_export_never_materializes_unicode(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from winding_field import patch_pipeline as p
    from winding_field.patch_mapping import SourceKeys,PatchMapping
    reader,ids=source()
    reader.db.execute('UPDATE vertices SET key=key || ?',('long-key'*15000,))
    reader.db.executescript('CREATE TABLE levels(id INTEGER,value REAL); INSERT INTO levels VALUES(0,0);')
    reader.metadata={'voxel_um':1.};reader.validate=lambda:None
    @contextmanager
    def opened(path):yield reader,dict(kind='test_mesh')
    monkeypatch.setattr(p,'open_mesh',opened)
    monkeypatch.setattr(p,'check_flatboi',lambda *args:{})
    monkeypatch.setattr(p,'flatten_native',lambda v,f,*args,**kwargs:(v[:,:2].copy(),dict(status='test')))
    def no_unicode(*args):raise AssertionError('large keys materialized')
    monkeypatch.setattr(SourceKeys,'__getitem__',no_unicode)
    class CT:
        chunks=(2,2,2);dtype=np.dtype('uint8');identity={'synthetic':True}
        def sample(self,xyz,frame):return np.full(len(xyz),100.),np.ones(len(xyz),bool)
    cfg=dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/test',pitch_um=.25,tile_size=8,memory_gib=.5)
    out=tmp_path/'export';p.export_mesh(tmp_path/'source',out,CT(),cfg)
    with PatchMapping(out/'level-0000') as mapping:
        assert 'source_vertex_keys' not in mapping.arrays.files
        assert mapping.arrays['source_vertex_ids'].tolist()==ids
        pairs=list(mapping.iter_keys('vertices'))
        assert [i for i,key in pairs]==ids and all(key.endswith('long-key'*15000) for i,key in pairs)
    reader.db.close()


def test_mapping_reader_rejects_fractional_source_ids(tmp_path):
    from winding_field.patch_mapping import PatchMapping
    np.savez(tmp_path/'map.npz',source_vertex_ids=np.array([9.5]),source_vertex_keys=np.array(['key']),
             source_face_ids=np.array([7]),source_face_keys=np.array(['face']))
    with PatchMapping(tmp_path) as mapping:
        with pytest.raises(ValueError,match='integer'):
            list(mapping.iter_keys('vertices'))


def test_shared_vertex_export_falls_back_before_repeated_key_gather(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from winding_field import patch_pipeline as p
    from winding_field.patch_mapping import PatchMapping,SourceKeys
    reader,_=source();db=reader.db
    db.execute('DELETE FROM vertices');db.execute('DELETE FROM faces')
    vertices=np.array([[10,10,1],[11,10,1],[10,11,1],[8,10,1],[10,8,1],[11,11,1],[10,12,1]],float)
    faces=np.array([[0,1,2],[0,3,4],[0,5,6]])
    keys={i+1:str(i)+'k'*34999 for i in range(7)}
    db.executemany('INSERT INTO vertices VALUES(?,0,?,?,?,?)',[(i+1,*v,keys[i+1]) for i,v in enumerate(vertices)])
    db.executemany('INSERT INTO faces VALUES(?,0,?,?,?,?)',[(i+1,*map(int,f+1),str(i)) for i,f in enumerate(faces)])
    db.executescript('CREATE TABLE levels(id INTEGER,value REAL); INSERT INTO levels VALUES(0,0);')
    reader.metadata={'voxel_um':1.};reader.validate=lambda:None
    assert SourceKeys(db,'vertices',np.arange(1,8)).unicode_bytes()+SourceKeys(db,'faces',np.arange(1,4)).unicode_bytes()==980012
    @contextmanager
    def opened(path):yield reader,dict(kind='test_mesh')
    monkeypatch.setattr(p,'open_mesh',opened)
    monkeypatch.setattr(p,'check_flatboi',lambda *args:{})
    monkeypatch.setattr(p,'flatten_native',lambda v,f,*args,**kwargs:(v[:,:2].copy(),dict(status='test')))
    class CT:
        chunks=(2,2,2);dtype=np.dtype('uint8');identity={'synthetic':True}
        def sample(self,xyz,frame):return np.full(len(xyz),100.),np.ones(len(xyz),bool)
    cfg=dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/test',pitch_um=.25,tile_size=16,memory_gib=.5,max_patch_faces=1)
    out=tmp_path/'export';receipt=p.export_mesh(tmp_path/'source',out,CT(),cfg)
    assert receipt['status']=='PATCHES_READY_DIAGNOSTIC'
    with PatchMapping(out/'level-0000') as mapping:
        ids=mapping.arrays['source_vertex_ids'].tolist()
        assert ids==[1,2,3,1,4,5,1,6,7]
        assert len(ids)*35000*4==1260000>1<<20
        assert 'source_vertex_keys' not in mapping.arrays.files and 'source_face_keys' not in mapping.arrays.files
        assert list(mapping.iter_keys('vertices'))==[(i,keys[i]) for i in ids]
        assert list(mapping.iter_keys('faces'))==[(i+1,str(i)) for i in range(3)]
        assert np.array_equal(mapping.arrays['field_xyz'],vertices[np.array(ids)-1])
    assert (out/'level-0000/render/tile-000000000-000000000/image.png').exists()
    db.close()
