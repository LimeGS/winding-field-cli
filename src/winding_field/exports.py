"""Portable OME-Zarr/Lasagna and AFV interchange, with explicit coordinate frames."""

import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from numcodecs import Blosc

from .artifacts import Tiles
from .util import atomic_bytes,atomic_json,digest,file_sha


def export_lasagna(run):
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes())
    root=run/'lasagna'; root.mkdir(exist_ok=True)
    maps=Tiles(run/'maps'); codec=Blosc(cname='lz4',clevel=5,shuffle=Blosc.SHUFFLE)
    names=['presence','nx','ny','surface_presence','surface_nx','surface_ny']
    for name in names:
        folder=root/(name+'.ome.zarr')
        atomic_json(folder/'.zgroup',{'zarr_format':2})
        atomic_json(folder/'.zattrs',{'multiscales':[{'version':'0.4','axes':[
            {'name':a,'type':'space','unit':'micrometer'} for a in 'zyx'],
            'datasets':[{'path':'0','coordinateTransformations':[
                {'type':'scale','scale':meta['spacing_xyz'][::-1]},
                {'type':'translation','translation':meta['origin_xyz'][::-1]}]}]}]})
        atomic_json(folder/'0/.zarray',dict(zarr_format=2,shape=list(maps.shape),chunks=[maps.chunk]*3,
                    dtype='|u1',compressor=codec.get_config(),fill_value=0,filters=None,order='C',dimension_separator='.'))
    for key in maps.keys():
        data=maps.load(key)
        tangent=data['tangents']; normal=data['normals']
        tf=np.where(tangent[...,2]<0,-1.,1.); nf=np.where(normal[...,2]<0,-1.,1.)
        values=[data['presence']*255,tangent[...,0]*tf*127+128,tangent[...,1]*tf*127+128,
                data['support']*255,normal[...,0]*nf*127+128,normal[...,1]*nf*127+128]
        for name,value in zip(names,values):
            array=np.zeros((maps.chunk,)*3,np.uint8)
            array[tuple(slice(0,n) for n in value.shape)]=np.rint(np.clip(value,0,255)).astype(np.uint8)
            atomic_bytes(root/(name+'.ome.zarr')/'0'/'.'.join(map(str,key)),codec.encode(array))
    for kind,prefix in (('fibres',''),('surfaces','surface_')):
        groups={name:dict(zarr=prefix+name+'.ome.zarr/0',scaledown=0,channels=[name]) for name in ('presence','nx','ny')}
        atomic_json(root/(kind+'.lasagna.json'),dict(version=2,source_to_base=1.,grad_mag_encode_scale=1000.,
                     grad_mag_factor=1.,groups=groups,base_shape_zyx=list(maps.shape),
                     crops=[[0,0,0,*maps.shape[::-1]]],coordinate_frame='ROI-local output grid',
                     world_origin_xyz_um=meta['origin_xyz'],world_spacing_xyz_um=meta['spacing_xyz'],
                     original_ct=meta['source'],prediction=meta['prediction']))
    maps.close()


def export_afv(run):
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes())
    source=meta['source']; root=run/'fibres'; destination=root/'fibres.afv'; pending=root/'fibres.partial.afv'
    pending.unlink(missing_ok=True)
    src=sqlite3.connect((root/'fibres.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(pending)
    db.executescript('PRAGMA application_id=1447249475; PRAGMA user_version=1;'
        'CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);'
        'CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,point_count INTEGER NOT NULL,'
        'length REAL NOT NULL,min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT NOT NULL);'
        'CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER REFERENCES fibers(id),first_segment INTEGER,points BLOB,UNIQUE(fiber_id,first_segment));'
        'CREATE INDEX fibers_length ON fibers(length DESC,id);'
        'CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
    count=points_total=0
    try:
        for fid,tile,blob in src.execute('SELECT id,tile,points FROM curves ORDER BY id'):
            world=np.load(io.BytesIO(blob),allow_pickle=False)
            points=((world-source['origin_xyz'])/source['spacing_xyz']).astype('<f8')
            lo=points.min(0); hi=points.max(0); box=[float(v) for pair in zip(lo,hi) for v in pair]
            length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            annotation=json.dumps(dict(tile=tile,prediction=meta['prediction'],automatically_traced=True,fragment=True))
            db.execute('INSERT INTO fibers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (fid,f'fiber-{fid}','unknown',len(points),length,*box,annotation))
            db.execute('INSERT INTO blocks VALUES (?,?,0,?)',(fid,fid,points.tobytes()))
            db.execute('INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)',(fid,*box))
            count+=1; points_total+=len(points)
        metadata=dict(uuid=digest([source,meta['tiles_sha256']]),fiber_count=count,point_count=points_total,
                      frame=dict(source=source,coordinate_order='XYZ',units='selected-source voxels',
                                 level=source['level'],spacing_xyz_um=source['spacing_xyz'],origin_xyz_um=source['origin_xyz']),
                      root=dict(source=source['source']),complete=True)
        db.executemany('INSERT INTO metadata VALUES (?,?)',[(k,json.dumps(v)) for k,v in metadata.items()])
        db.commit()
    finally:
        db.close(); src.close()
    pending.replace(destination)
    return file_sha(destination)
