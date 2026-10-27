"""Bounded streamline fragments and durable physical-coordinate fibre constraints."""

import io
import itertools
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.ndimage import map_coordinates

from .artifacts import Tiles
from .util import atomic_json, digest


def streamlines(presence,tangents,support,*,seeds,origin_xyz,spacing_xyz,threshold=.5,max_steps=128):
    spacing=np.asarray(spacing_xyz,float)
    curves=[]
    def sample(array,p,order=1):
        return float(map_coordinates(array,np.asarray(p)[:,None],order=order,mode='constant',cval=0)[0])
    for seed in seeds:
        seed=np.asarray(seed,float)
        if sample(presence,seed)<threshold or not sample(support,seed,0):
            continue
        halves=[]
        for sign in (-1,1):
            points=[]; p=seed.copy(); previous=None
            for _ in range(max_steps):
                if np.any(p<0) or np.any(p>np.array(support.shape)-1) or not sample(support,p,0) or sample(presence,p)<threshold:
                    break
                t=np.array([sample(tangents[...,i],p) for i in range(3)])[::-1]/spacing[::-1]
                norm=np.linalg.norm(t)
                if norm<1e-6:
                    break
                t=t/norm
                if previous is None:
                    t*=sign
                elif np.dot(t,previous)<0:
                    t=-t
                if previous is not None and np.dot(t,previous)<.5:
                    break
                points.append(p.copy()); previous=t; p=p+t*.75
            halves.append(points)
        points=halves[0][:0:-1]+halves[1]
        if len(points)>=4:
            curves.append(np.asarray(points)[:,::-1]*spacing+origin_xyz)
    return curves


def trace(run,config,stop=None):
    run=Path(run)
    prepared=json.loads((run/'PREPARED.json').read_bytes())
    if prepared['prediction']=='qual-v2':
        from .splines import trace_qual
        return trace_qual(run,config,stop)
    maps=Tiles(run/'maps')
    if maps.identity()!=prepared['tiles_sha256']:
        raise ValueError('prepared maps changed')
    root=run/'fibres'; root.mkdir(exist_ok=True)
    identity=dict(maps=prepared['tiles_sha256'],seed_stride=config.get('seed_stride',8),
                  threshold=config.get('fibre_threshold',.5),method='bounded-streamlines-v1')
    path=root/'IDENTITY.json'
    if path.exists() and json.loads(path.read_bytes())!=identity:
        raise ValueError('fibre trace identity changed')
    atomic_json(path,identity)
    db=sqlite3.connect(root/'fibres.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS done (tile TEXT PRIMARY KEY);'
                     'CREATE TABLE IF NOT EXISTS curves (id INTEGER PRIMARY KEY, tile TEXT, points BLOB);'
                     'CREATE INDEX IF NOT EXISTS curves_tile ON curves(tile);')
    seed_stride=identity['seed_stride']
    if type(seed_stride) is not int or seed_stride<1:
        raise ValueError('seed stride must be positive integer')
    for key in maps.keys():
        tag='_'.join(map(str,key))
        if db.execute('SELECT 1 FROM done WHERE tile=?',(tag,)).fetchone():
            continue
        data=maps.load(key); lo,hi=maps.box(key)
        seeds=itertools.product(*(range(seed_stride//2,int(n),seed_stride) for n in hi-lo))
        curves=streamlines(data['presence'],data['tangents'],data['support'],seeds=seeds,
                 origin_xyz=(np.array(prepared['origin_xyz'])+lo[::-1]*prepared['spacing_xyz']),
                 spacing_xyz=prepared['spacing_xyz'],threshold=identity['threshold'])
        with db:
            for points in curves:
                buffer=io.BytesIO(); np.save(buffer,points,allow_pickle=False)
                db.execute('INSERT INTO curves(tile,points) VALUES (?,?)',(tag,buffer.getvalue()))
            db.execute('INSERT INTO done VALUES (?)',(tag,))
        if stop and stop():
            db.close(); maps.close(); return dict(status='PAUSED')
    total=db.execute('SELECT count(*) FROM curves').fetchone()[0]
    db.close(); maps.close()
    from .util import file_sha
    receipt=dict(status='FIBRES_READY',count=total,identity=identity,sha256=file_sha(root/'fibres.sqlite'),
                 coordinates='XYZ micrometres',boundary_policy='fragments stop at tile bounds or unsupported gaps',
                 physical_accuracy='UNVALIDATED')
    atomic_json(run/'FIBRES.json',receipt)
    return receipt
