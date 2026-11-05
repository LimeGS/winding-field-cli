"""Spatially indexed global fibre pairs, restricted with live exterior values."""

import io
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from constraint_templates import StencilConstraints
from .field import spline_samples
from .splines import bind
from .util import atomic_json,file_sha


def build(root,source,source_sha,grid,*,weight,quota_bytes,reserve_bytes=0,stop=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if file_sha(source)!=source_sha:raise ValueError('fitted fibre source changed')
    if not np.isfinite(weight) or weight<0:raise ValueError('nonnegative fibre weight required')
    identity=dict(source_sha256=source_sha,grid=grid,weight=weight,algorithm='global-arclength-pairs-v1')
    bind(root/'IDENTITY.json',identity)
    marker=root/'PAIRS.json'
    if marker.exists():
        receipt=json.loads(marker.read_bytes())
        if receipt!=dict(identity=identity,sha256=file_sha(root/'pairs.sqlite')):
            raise ValueError('global pair catalog changed')
        return 'COMPLETE'
    src=sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(root/'pairs.sqlite')
    try:
        db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=-2048')
        db.executescript('CREATE TABLE IF NOT EXISTS done (id INTEGER PRIMARY KEY);'
            'CREATE TABLE IF NOT EXISTS pairs (id INTEGER PRIMARY KEY,curve_id INTEGER,points BLOB NOT NULL);'
            'CREATE VIRTUAL TABLE IF NOT EXISTS bounds USING rtree(id,x0,x1,y0,y1,z0,z1);')
        origin=np.asarray(grid['origin_xyz']);spacing=np.asarray(grid['spacing_xyz'])
        high=origin+(np.asarray(grid['shape_zyx'])[::-1]-1)*spacing
        for fid,blob in src.execute('SELECT id,points FROM curves ORDER BY id'):
            if stop and stop():return 'PAUSED'
            if db.execute('SELECT 1 FROM done WHERE id=?',(fid,)).fetchone():continue
            points=np.load(io.BytesIO(blob),allow_pickle=False)
            if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all():
                raise ValueError('finite XYZ fitted fibre required')
            with db:
                for sampled in spline_samples(points,origin,high,float(spacing.min())):
                    size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
                    allowance=len(sampled)*512+65536
                    if 2*(size+allowance)>quota_bytes or shutil.disk_usage(root).free<reserve_bytes+2*allowance:
                        raise OSError('global pair catalog quota/reserve exhausted; curve not committed')
                    for a,b in zip(sampled[:-1],sampled[1:]):
                        pair=np.asarray([a,b],dtype='<f8')
                        ij=(pair-origin)/spacing
                        low=np.floor(ij.min(0));upper=np.ceil(ij.max(0))
                        index=db.execute('INSERT INTO pairs(curve_id,points) VALUES (?,?)',(fid,pair.tobytes())).lastrowid
                        db.execute('INSERT INTO bounds VALUES (?,?,?,?,?,?,?)',
                                   (index,*[float(v) for p in zip(low,upper) for v in p]))
                db.execute('INSERT INTO done VALUES (?)',(fid,))
        if file_sha(source)!=source_sha:raise ValueError('fitted fibre source changed during catalog build')
    finally:src.close();db.close()
    atomic_json(root/'PAIRS.json',dict(identity=identity,sha256=file_sha(root/'pairs.sqlite')))
    return 'COMPLETE'


class PairCatalog:
    def __init__(self,root,grid,*,max_bytes=256<<20):
        root=Path(root);receipt=json.loads((root/'PAIRS.json').read_bytes())
        if receipt['identity']['grid']!=grid or file_sha(root/'pairs.sqlite')!=receipt['sha256']:
            raise ValueError('global pair catalog changed')
        self.grid=grid;self.weight=receipt['identity']['weight'];self.max_bytes=max_bytes
        self.db=sqlite3.connect((root/'pairs.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-2048')

    def __enter__(self):return self

    def __exit__(self,*args):self.db.close()

    def __call__(self,lo,hi,read_nodes):
        lo=np.asarray(lo);hi=np.asarray(hi)
        if lo.shape!=(3,) or hi.shape!=(3,) or np.any(lo<0) or np.any(hi>self.grid['shape_zyx']) or np.any(hi<=lo):
            raise ValueError('pair restriction outside global grid')
        query='''SELECT p.points FROM bounds b JOIN pairs p ON p.id=b.id
            WHERE b.x1>=? AND b.x0<? AND b.y1>=? AND b.y0<? AND b.z1>=? AND b.z0<? ORDER BY p.id'''
        values=[]
        for (blob,) in self.db.execute(query,tuple(float(v) for p in zip(lo[::-1],hi[::-1]) for v in p)):
            # Account for stencil, CSR, template and Python temporaries, not just endpoints.
            if (len(values)+1)*4096>self.max_bytes:raise OSError('constraint restriction exceeds memory budget')
            value=np.frombuffer(blob,dtype='<f8')
            if value.size!=6 or not np.isfinite(value).all():raise ValueError('invalid pair payload')
            values.append(value.reshape(2,3))
        if not values:return None
        pairs=np.stack(values)
        origin=np.asarray(self.grid['origin_xyz']);spacing=np.asarray(self.grid['spacing_xyz'])
        last=np.asarray(self.grid['shape_zyx'])[::-1]-1
        if np.any(pairs<origin) or np.any(pairs>origin+last*spacing):
            raise ValueError('pair outside physical grid support')
        # Validate physical support first; division can round an exact endpoint above the last node.
        grid_pairs=np.clip((pairs-origin)/spacing,0,last)
        rows=StencilConstraints.pairs(grid_pairs[:,0],grid_pairs[:,1],[0.]*3,
                                     [1.]*3,self.grid['shape_zyx'],weight=self.weight)
        return rows.restrict(lo,hi,read_nodes)
