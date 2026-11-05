"""Durable global raw curves and seed progress; cache coverage is disposable."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


class CurveJournal:
    def __init__(self,path,identity,*,quota_bytes):
        if type(quota_bytes) is not int or quota_bytes<65536:
            raise ValueError('journal quota must allow at least 65536 bytes')
        self.path=Path(path);self.quota_bytes=quota_bytes
        identity=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.db=sqlite3.connect(self.path)
        try:
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA cache_size=-2048')
            with self.db:
                self.db.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)')
                existing=self.db.execute('SELECT identity FROM metadata').fetchall()
                if existing and existing!=[(identity,)]:raise ValueError('journal identity mismatch')
                if not existing:self.db.execute('INSERT INTO metadata VALUES (?)',(identity,))
                self.db.execute('CREATE TABLE IF NOT EXISTS progress (family INTEGER PRIMARY KEY, next_seed INTEGER NOT NULL)')
                self.db.execute('''CREATE TABLE IF NOT EXISTS curves (
                    id INTEGER PRIMARY KEY, family INTEGER NOT NULL, seed INTEGER NOT NULL,
                    points BLOB NOT NULL, tangents BLOB NOT NULL, digest TEXT NOT NULL)''')
                self.db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS bounds USING rtree(id,x0,x1,y0,y1,z0,z1)')
                self.db.execute('CREATE INDEX IF NOT EXISTS curves_seed ON curves(family,seed,id)')
                self.db.execute('CREATE TABLE IF NOT EXISTS commits (family INTEGER,seed INTEGER,digest TEXT,PRIMARY KEY(family,seed))')
                self._validate_inventory()
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):return self

    def __exit__(self,*args):self.db.close()

    def _seed_digest(self,family,seed):
        digest=hashlib.sha256(json.dumps([family,seed]).encode())
        for row in self.db.execute('''SELECT c.id,c.points,c.tangents,c.digest,b.x0,b.x1,b.y0,b.y1,b.z0,b.z1
            FROM curves c LEFT JOIN bounds b ON b.id=c.id WHERE c.family=? AND c.seed=? ORDER BY c.id''',(family,seed)):
            cid,p,t,sha,*bounds=row
            if None in bounds or hashlib.sha256(p+t).hexdigest()!=sha:
                raise ValueError('raw curve inventory changed')
            digest.update(json.dumps([cid,sha,bounds],allow_nan=False).encode())
        return digest.hexdigest()

    def _validate_inventory(self):
        if (self.db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]
                or self.db.execute("SELECT rtreecheck('bounds')").fetchone()!=('ok',)
                or self.db.execute('SELECT 1 FROM bounds b LEFT JOIN curves c ON c.id=b.id WHERE c.id IS NULL LIMIT 1').fetchone()
                or self.db.execute('''SELECT 1 FROM curves c LEFT JOIN commits s ON s.family=c.family AND s.seed=c.seed
                                     WHERE s.seed IS NULL LIMIT 1''').fetchone()):
            raise ValueError('raw curve inventory changed')
        totals={}
        for family,seed,expected in self.db.execute('SELECT family,seed,digest FROM commits ORDER BY family,seed'):
            if family not in (1,2) or seed!=totals.get(family,0) or self._seed_digest(family,seed)!=expected:
                raise ValueError('raw curve commit inventory changed')
            totals[family]=seed+1
        if dict(self.db.execute('SELECT family,next_seed FROM progress'))!=totals:
            raise ValueError('raw curve progress inventory changed')

    def next_seed(self,family):
        if type(family) is not int or family not in (1,2):raise ValueError('invalid Qual family')
        row=self.db.execute('SELECT next_seed FROM progress WHERE family=?',(family,)).fetchone()
        return row[0] if row else 0

    def _admit(self,extra=0):
        pages=self.db.execute('PRAGMA page_count').fetchone()[0]
        page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
        # Reserve database growth plus rollback-journal pages before writing.
        if 2*(pages*page_size+extra+65536)>self.quota_bytes:
            raise OSError('raw curve journal quota exhausted; seed not committed')

    def commit(self,family,seed,curves):
        # The write lock makes progress and every raw span one crash-safe transaction.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if type(seed) is not int or seed!=self.next_seed(family):
                raise ValueError('seed sequence mismatch')
            self._admit()
            for points,tangents in curves:
                points=np.asarray(points,dtype='<f4');tangents=np.asarray(tangents,dtype='<f4')
                if (points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<2
                        or tangents.shape!=points.shape or not np.isfinite(points).all()
                        or not np.isfinite(tangents).all()):
                    raise ValueError('finite matching XYZ raw curves required')
                p=points.tobytes();t=tangents.tobytes()
                self._admit(len(p)+len(t))
                cursor=self.db.execute('INSERT INTO curves(family,seed,points,tangents,digest) VALUES (?,?,?,?,?)',
                    (family,seed,p,t,hashlib.sha256(p+t).hexdigest()))
                lo=points.min(axis=0);hi=points.max(axis=0)
                self.db.execute('INSERT INTO bounds VALUES (?,?,?,?,?,?,?)',
                    (cursor.lastrowid,*[float(v) for pair in zip(lo,hi) for v in pair]))
            self.db.execute('INSERT INTO commits VALUES (?,?,?)',(family,seed,self._seed_digest(family,seed)))
            self.db.execute('INSERT OR REPLACE INTO progress VALUES (?,?)',(family,seed+1))

    def curves(self,family,lo,hi):
        self.next_seed(family)
        lo=np.asarray(lo);hi=np.asarray(hi)
        if (lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite(lo).all()
                or not np.isfinite(hi).all() or np.any(hi<=lo)):
            raise ValueError('finite increasing ZYX coverage bounds required')
        # R-tree only selects candidates; coverage replay checks each raw point.
        rows=self.db.execute('''SELECT c.points,c.tangents,c.digest FROM bounds b
            JOIN curves c ON c.id=b.id WHERE c.family=?
            AND b.x1>=? AND b.x0<=? AND b.y1>=? AND b.y0<=? AND b.z1>=? AND b.z0<=?
            ORDER BY c.id''',(family,float(lo[2]),float(hi[2]),float(lo[1]),float(hi[1]),float(lo[0]),float(hi[0])))
        for p,t,digest in rows:
            if hashlib.sha256(p+t).hexdigest()!=digest:raise ValueError('raw curve checksum mismatch')
            yield np.frombuffer(p,dtype='<f4').reshape(-1,3),np.frombuffer(t,dtype='<f4').reshape(-1,3)
