"""Numeric patch maps with bounded, hash-bound SQLite key provenance."""
import json
import sqlite3
from pathlib import Path

import numpy as np

from .util import atomic_json, file_sha


class SourceKeys:
    def __init__(self, db, table, ids):
        if table not in ('vertices', 'faces'):raise ValueError('unknown key table')
        self.db,self.table,self.ids=db,table,ids

    def __len__(self):return len(self.ids)

    def iter_pairs(self, batch_size=512):
        if not 1<=batch_size<=512:raise ValueError('key batch_size must be 1..512')
        for start in range(0,len(self.ids),batch_size):
            ids=[int(i) for i in self.ids[start:start+batch_size]]
            rows=dict(self.db.execute(f'SELECT id,key FROM {self.table} WHERE id IN ({",".join("?" for _ in ids)})',ids))
            for identity in ids:
                if identity not in rows:raise ValueError('mapping references missing source key')
                yield identity,rows[identity]

    def __iter__(self):return (key for _,key in self.iter_pairs())

    def unicode_bytes(self, *, count=None):
        width=self.db.execute(f'SELECT COALESCE(MAX(length(key)),0) FROM {self.table}').fetchone()[0]
        return (len(self.ids) if count is None else count)*max(1,width)*4

    def __getitem__(self, rows):
        selected=SourceKeys(self.db,self.table,np.atleast_1d(self.ids[rows]))
        if selected.unicode_bytes()>1<<20:raise ValueError('large keys require streaming SourceKeys.iter_pairs')
        keys=np.asarray(list(selected),dtype=str)
        return keys[0] if np.isscalar(rows) else keys


def write_key_mapping(root, reader, vertex_keys, face_keys, guard=lambda:None):
    root=Path(root);path=root/'keys.sqlite'
    if path.exists():raise ValueError('key mapping already exists')
    db=sqlite3.connect(path)
    try:
        db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=FILE')
        for keys in (vertex_keys,face_keys):
            db.execute(f'CREATE TABLE {keys.table}(id INTEGER PRIMARY KEY,key TEXT NOT NULL)')
            batch=[]
            for row in keys.iter_pairs():
                batch.append(row)
                if len(batch)==512:
                    guard();db.executemany(f'INSERT INTO {keys.table} VALUES(?,?)',batch);db.commit();batch.clear()
            guard();db.executemany(f'INSERT INTO {keys.table} VALUES(?,?)',batch);db.commit()
    finally:db.close()
    result=dict(schema='patch-key-mapping-v1',map_sha256=file_sha(root/'map.npz'),keys_sha256=file_sha(path),
                source_mesh_manifest_sha256=reader.manifest_sha,source_database_sha256=reader.manifest['database_sha256'],
                key_semantics='exact source database IDs and keys, including omitted source faces')
    atomic_json(root/'MAPPING.json',result)
    return result


class PatchMapping:
    """Read legacy embedded keys or a portable SQLite projection; never infer keys."""
    def __init__(self, root):
        root=Path(root);self.db=None;self.arrays=None
        manifest=root/'MAPPING.json'
        if manifest.exists():
            record=json.loads(manifest.read_bytes())
            if record.get('schema')!='patch-key-mapping-v1':raise ValueError('unknown mapping schema')
            for name,key in (('map.npz','map_sha256'),('keys.sqlite','keys_sha256')):
                if file_sha(root/name)!=record[key]:raise ValueError('mapping hash differs: '+name)
            self.db=sqlite3.connect((root/'keys.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
            self.db.execute('PRAGMA cache_size=-4096')
        try:
            self.arrays=np.load(root/'map.npz',allow_pickle=False)
            if self.db is None and not {'source_vertex_keys','source_face_keys'}<=set(self.arrays.files):
                raise ValueError('mapping lacks key provenance manifest')
        except Exception:
            self.close();raise

    def iter_keys(self, kind, *, batch_size=512):
        if kind not in ('vertices','faces'):raise ValueError('unknown key table')
        if not 1<=batch_size<=512:raise ValueError('key batch_size must be 1..512')
        prefix='source_vertex' if kind=='vertices' else 'source_face'
        ids=self.arrays[prefix+'_ids']
        if ids.ndim!=1 or ids.dtype!=np.dtype('int64'):raise ValueError('source IDs must be int64 integer vectors')
        if self.db is not None:
            yield from SourceKeys(self.db,kind,ids).iter_pairs(batch_size)
        else:
            keys=self.arrays[prefix+'_keys']
            if len(keys)!=len(ids):raise ValueError('legacy key count differs')
            yield from ((int(i),str(key)) for i,key in zip(ids,keys))

    def close(self):
        if self.arrays is not None:self.arrays.close()
        if self.db is not None:self.db.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
