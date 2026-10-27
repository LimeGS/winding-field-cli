"""Atomic, checksummed tiles for out-of-core intermediates."""

import io
import itertools
import json
from pathlib import Path
import sqlite3

import numpy as np

from .io import bounds
from .util import atomic_bytes, atomic_json, digest, file_sha


class Tiles:
    def __init__(self,root,shape=None,chunk=None,identity=None):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/'TILES.json'
        if path.exists():
            self.meta=json.loads(path.read_bytes())
            if shape is not None and self.meta != dict(shape=list(shape),chunk=chunk,identity=identity):
                raise ValueError('tile identity changed')
        else:
            if shape is None or len(shape)!=3 or any(int(n)!=n or n<1 for n in shape) or not isinstance(chunk,int) or chunk<1:
                raise ValueError('invalid tile geometry')
            self.meta=dict(shape=list(shape),chunk=chunk,identity=identity)
            atomic_json(path,self.meta)
        self.shape=tuple(self.meta['shape']); self.chunk=self.meta['chunk']
        self.db=sqlite3.connect(self.root/'tiles.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS tiles (key TEXT PRIMARY KEY, sha TEXT, info TEXT)')
        self.db.commit()

    def keys(self):
        return itertools.product(*(range((n-1)//self.chunk+1) for n in self.shape))

    def box(self,key):
        lo=np.asarray(key)*self.chunk
        return lo,np.minimum(lo+self.chunk,self.shape)

    def path(self,key):
        return self.root/('_'.join(map(str,key))+'.npz')

    def has(self,key):
        record=self.db.execute('SELECT sha FROM tiles WHERE key=?',(self.path(key).stem,)).fetchone()
        if record is None:
            return False
        if not self.path(key).exists() or file_sha(self.path(key))!=record[0]:
            raise ValueError('corrupt tile: '+str(key))
        return True

    def write(self,key,arrays,info=None):
        lo,hi=self.box(key)
        bounds(lo,hi,self.shape)
        if not arrays or any(np.asarray(a).shape[:3]!=tuple(hi-lo) or np.asarray(a).dtype.hasobject for a in arrays.values()):
            raise ValueError('tile arrays must match owned core geometry')
        buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays)
        path=self.path(key); atomic_bytes(path,buffer.getvalue())
        self.db.execute('INSERT OR REPLACE INTO tiles VALUES (?,?,?)',
                        (path.stem,file_sha(path),json.dumps(info or {},sort_keys=True)))
        self.db.commit()

    def info(self,key):
        row=self.db.execute('SELECT info FROM tiles WHERE key=?',(self.path(key).stem,)).fetchone()
        return json.loads(row[0]) if row else None

    def load(self,key):
        if not self.has(key):
            raise FileNotFoundError('uncomputed tile: '+str(key))
        with np.load(self.path(key),allow_pickle=False) as data:
            return {name:data[name] for name in data.files}

    def read(self,name,lo,hi):
        lo,hi=bounds(lo,hi,self.shape)
        result=None
        for key in itertools.product(*(range(a//self.chunk,(b-1)//self.chunk+1) for a,b in zip(lo,hi))):
            tile=self.load(key)[name]
            if result is None:
                result=np.empty((*tuple(hi-lo),*tile.shape[3:]),dtype=tile.dtype)
            start,end=self.box(key); a,b=np.maximum(start,lo),np.minimum(end,hi)
            result[tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo))]=tile[
                tuple(slice(int(x),int(y)) for x,y in zip(a-start,b-start))]
        return result

    def identity(self):
        import hashlib
        h=hashlib.sha256(digest(self.meta).encode())
        for key in self.keys():
            if not self.has(key):
                raise ValueError('incomplete tile set')
            record=self.db.execute('SELECT sha FROM tiles WHERE key=?',(self.path(key).stem,)).fetchone()
            h.update(record[0].encode())
        return h.hexdigest()

    def close(self):
        self.db.close()
