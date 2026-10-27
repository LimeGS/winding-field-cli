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

    def load(self,key,names=None):
        if not self.has(key):
            raise FileNotFoundError('uncomputed tile: '+str(key))
        with np.load(self.path(key),allow_pickle=False) as data:
            return {name:data[name] for name in (data.files if names is None else names)}

    def read(self,name,lo,hi):
        return self.read_many((name,),lo,hi)[name]

    def read_many(self,names,lo,hi):
        lo,hi=bounds(lo,hi,self.shape)
        result={}
        for key in itertools.product(*(range(a//self.chunk,(b-1)//self.chunk+1) for a,b in zip(lo,hi))):
            start,end=self.box(key); a,b=np.maximum(start,lo),np.minimum(end,hi)
            for name,tile in self.load(key,names).items():
                if name not in result:
                    result[name]=np.empty((*tuple(hi-lo),*tile.shape[3:]),dtype=tile.dtype)
                result[name][tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo))]=tile[
                    tuple(slice(int(x),int(y)) for x,y in zip(a-start,b-start))]
        return result

    def sample(self,names,indices_zyx,transform=None):
        """Trilinear Cartesian sampling; booleans require all contributing corners.

        Only touched tiles are loaded, once each, even for very sparse coarse grids.
        An optional tile transform can orient directors before interpolation.
        """
        axes=[np.asarray(a,dtype=float) for a in indices_zyx]
        if len(axes)!=3 or any(a.ndim!=1 or not len(a) or not np.isfinite(a).all()
                or np.any(a<0) or np.any(a>n-1) for a,n in zip(axes,self.shape)):
            raise ValueError('sample coordinates must be finite nonempty ZYX axes inside maps')
        lower=[np.floor(a).astype(np.int64) for a in axes]
        upper=[np.minimum(a+1,n-1) for a,n in zip(lower,self.shape)]
        fraction=[a-b for a,b in zip(axes,lower)]
        nodes=[lower,upper]; weights=[[1-f for f in fraction],fraction]
        keys=[np.unique(np.concatenate([a//self.chunk,b[f>0]//self.chunk]))
              for a,b,f in zip(lower,upper,fraction)]
        shape=tuple(len(a) for a in axes); result={}
        for key in itertools.product(*keys):
            data=self.load(key,names)
            if transform is not None:
                data=transform(data,self.box(key)[0])
            for name,tile in data.items():
                if name not in result:
                    result[name]=(np.ones if tile.dtype==np.bool_ else np.zeros)(
                        (*shape,*tile.shape[3:]),dtype=bool if tile.dtype==np.bool_ else np.result_type(tile.dtype,np.float32))
            for corner in itertools.product((0,1),repeat=3):
                positions=[np.flatnonzero((nodes[c][i]//self.chunk==key[i]) & (weights[c][i]>0))
                           for i,c in enumerate(corner)]
                if any(not len(p) for p in positions):
                    continue
                dst=np.ix_(*positions)
                src=np.ix_(*(nodes[c][i][p]%self.chunk for i,(c,p) in enumerate(zip(corner,positions))))
                w=[weights[c][i][p] for i,(c,p) in enumerate(zip(corner,positions))]
                weight=w[0][:,None,None]*w[1][None,:,None]*w[2][None,None,:]
                for name,tile in data.items():
                    if tile.dtype==np.bool_:
                        result[name][dst] &= tile[src]
                    else:
                        result[name][dst] += tile[src]*weight[(...,)+(None,)*(tile.ndim-3)]
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
